"""Batched, resumable tool-use supervised fine-tuning."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .checkpoint import load_checkpoint, load_model_weights, save_checkpoint
from .config import ModelConfig
from .model import GPT2ReasoningModel
from .train import optimizer_parameter_groups


def encode_sft_document(
    tokenizer: Tokenizer, text: str, max_length: int
) -> tuple[list[int], list[int]]:
    encoded = tokenizer.encode(text)
    ids = encoded.ids[:max_length]
    labels = ids.copy()
    answer_marker = text.find("<|tool_call|>")
    if answer_marker < 0:
        answer_marker = text.find("<|reasoning|>")
    spans = []
    if answer_marker > 0:
        spans.append((0, answer_marker))
    start = text.find("<|tool_result|>")
    while start >= 0:
        end = text.find("<|end_tool_result|>", start)
        end = len(text) if end < 0 else end + len("<|end_tool_result|>")
        spans.append((start, end))
        start = text.find("<|tool_result|>", end)
    for index, (offset_start, offset_end) in enumerate(encoded.offsets[:max_length]):
        if any(offset_start < end and offset_end > start for start, end in spans):
            labels[index] = -100
    return ids, labels


def collate_sft_batch(
    examples: Sequence[tuple[list[int], list[int]]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty SFT batch")
    length = max(len(ids) for ids, _ in examples)
    inputs = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    for row, (ids, targets) in enumerate(examples):
        inputs[row, : len(ids)] = torch.tensor(ids)
        labels[row, : len(targets)] = torch.tensor(targets)
    return inputs, labels


def _trajectory_texts(path: Path) -> Iterator[str]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"trajectory line {line_number} has no non-empty text")
            yield text


def _shuffle_buffer(values: Iterable[str], buffer_size: int, rng: random.Random) -> Iterator[str]:
    buffer: list[str] = []
    for value in values:
        if len(buffer) < buffer_size:
            buffer.append(value)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = value
    rng.shuffle(buffer)
    yield from buffer


def _learning_rate_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def fine_tune_tools(
    checkpoint_directory: Path,
    tokenizer_path: Path,
    trajectories_path: Path,
    output_directory: Path,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    *,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    shuffle_buffer_size: int = 10_000,
    warmup_fraction: float = 0.03,
    save_every_steps: int = 250,
    seed: int = 42,
    resume_from: Path | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tool fine-tuning")
    if epochs < 1 or micro_batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if not 0 <= warmup_fraction < 1 or shuffle_buffer_size < 1:
        raise ValueError("invalid warmup fraction or shuffle buffer size")

    example_count = sum(1 for _ in _trajectory_texts(trajectories_path))
    if example_count == 0:
        raise ValueError("tool trajectory dataset is empty")
    state = json.loads((checkpoint_directory / "state.json").read_text())
    config = ModelConfig(**state["config"]["model"])
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    random.seed(seed)
    model = GPT2ReasoningModel(config).to(device=device, dtype=torch.bfloat16)
    load_model_weights(checkpoint_directory, model, device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    if pad_token_id is None:
        raise ValueError("tokenizer is missing the required <|pad|> token")

    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, 0.1),
        lr=learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )
    batches_per_epoch = math.ceil(example_count / micro_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    total_steps = updates_per_epoch * epochs
    warmup_steps = round(total_steps * warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_factor(step, total_steps, warmup_steps),
    )
    start_epoch = start_cursor = step = 0
    if resume_from is not None:
        restored = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        step = int(restored["step"])
        start_epoch = int(restored["mixture_state"].get("epoch", 0))
        start_cursor = int(restored["mixture_state"].get("examples_in_epoch", 0))

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "metrics.jsonl"
    model.train()
    with metrics_path.open("a") as metrics:
        for epoch in range(start_epoch, epochs):
            texts = _shuffle_buffer(
                _trajectory_texts(trajectories_path),
                shuffle_buffer_size,
                random.Random(seed + epoch),
            )
            cursor = 0
            batch: list[tuple[list[int], list[int]]] = []
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            for text in texts:
                if epoch == start_epoch and cursor < start_cursor:
                    cursor += 1
                    continue
                encoded = encode_sft_document(tokenizer, text, config.max_seq_len)
                cursor += 1
                if len(encoded[0]) < 2 or all(label == -100 for label in encoded[1][1:]):
                    continue
                batch.append(encoded)
                if len(batch) < micro_batch_size:
                    continue
                inputs, labels = collate_sft_batch(batch, pad_token_id)
                batch.clear()
                output = model(inputs.to(device), labels.to(device))
                assert output.loss is not None
                (output.loss / gradient_accumulation_steps).backward()
                accumulation += 1
                if accumulation < gradient_accumulation_steps:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
                step += 1
                metrics.write(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "examples_in_epoch": cursor,
                            "loss": float(output.loss.detach()),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "gradient_norm": float(grad_norm),
                        }
                    )
                    + "\n"
                )
                metrics.flush()
                if step % save_every_steps == 0:
                    save_checkpoint(
                        output_directory / f"step-{step:08d}",
                        model,
                        optimizer,
                        scheduler,
                        step,
                        0,
                        {"epoch": epoch, "examples_in_epoch": cursor},
                        {**state["config"], "tool_sft": {"epochs": epochs, "seed": seed}},
                    )
            if batch:
                inputs, labels = collate_sft_batch(batch, pad_token_id)
                output = model(inputs.to(device), labels.to(device))
                assert output.loss is not None
                (output.loss / gradient_accumulation_steps).backward()
                accumulation += 1
            if accumulation:
                if accumulation < gradient_accumulation_steps:
                    correction = gradient_accumulation_steps / accumulation
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            start_cursor = 0

    save_checkpoint(
        output_directory,
        model,
        optimizer,
        scheduler,
        step,
        0,
        {"epoch": epochs, "examples_in_epoch": 0},
        {**state["config"], "tool_sft": {"epochs": epochs, "seed": seed}},
    )
    return output_directory
