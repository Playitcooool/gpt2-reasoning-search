"""Tool-use supervised fine-tuning with prompt and observation masking."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .checkpoint import load_model_weights
from .config import ModelConfig
from .model import GPT2ReasoningModel


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


def fine_tune_tools(
    checkpoint_directory: Path,
    tokenizer_path: Path,
    trajectories_path: Path,
    output_directory: Path,
    epochs: int = 1,
    learning_rate: float = 2e-5,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tool fine-tuning")
    state = json.loads((checkpoint_directory / "state.json").read_text())
    config = ModelConfig(**state["config"]["model"])
    device = torch.device("cuda")
    model = GPT2ReasoningModel(config).to(device=device, dtype=torch.bfloat16)
    load_model_weights(checkpoint_directory, model, device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    examples = []
    with trajectories_path.open() as handle:
        for line in handle:
            examples.append(
                encode_sft_document(tokenizer, json.loads(line)["text"], config.max_seq_len)
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    model.train()
    for _ in range(epochs):
        for ids, labels in examples:
            inputs = torch.tensor([ids], device=device)
            targets = torch.tensor([labels], device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs, targets)
            assert output.loss is not None
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    output_directory.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_model

    save_model(model, str(output_directory / "model.safetensors"), metadata={"format": "pt"})
    state["format_version"] = 2
    (output_directory / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    return output_directory
