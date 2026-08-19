"""Single-H100 pretraining loop with calibration-based budgeting."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig, TrainConfig
from .data import ExactTokenMixture, MixtureState
from .model import GPT2ReasoningModel


def cosine_schedule(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def projected_token_budget(tokens_per_second: float, hours: float, cap: int, quantum: int) -> int:
    projected = int(tokens_per_second * hours * 3600)
    return max(quantum, min(cap, projected) // quantum * quantum)


def train(config: TrainConfig, resume_from: Path | None = None) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production training command")
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")

    model_config = ModelConfig.preset(config.model_preset)
    model_config = ModelConfig(**{**model_config.to_dict(), "max_seq_len": config.sequence_length})
    raw_model = GPT2ReasoningModel(model_config).to(device=device, dtype=torch.bfloat16)
    model = torch.compile(raw_model) if config.compile_model else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    reasoning = np.load(config.reasoning_tokens, mmap_mode="r")
    general = np.load(config.general_tokens, mmap_mode="r")
    provisional_steps = config.max_tokens // config.tokens_per_optimizer_step
    scheduler = cosine_schedule(
        optimizer,
        max(1, int(provisional_steps * config.warmup_fraction)),
        provisional_steps,
        config.min_learning_rate / config.learning_rate,
    )
    restored_state = None
    if resume_from is not None:
        restored = load_checkpoint(resume_from, raw_model, optimizer, scheduler, device)
        restored_state = MixtureState(**restored["mixture_state"])
    mixture = ExactTokenMixture(
        reasoning, general, config.max_tokens, config.reasoning_ratio, state=restored_state
    )
    batches = mixture.batches(config.micro_batch_size, config.sequence_length, device)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    step = int(restored["step"]) if resume_from is not None else 0
    started = time.perf_counter()
    calibration_started = started
    calibrated_budget = config.max_tokens

    with metrics_path.open("a") as metrics:
        while mixture.state.total_tokens < calibrated_budget:
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(config.gradient_accumulation_steps):
                tokens = next(batches)
                output = model(tokens, tokens)
                assert output.loss is not None
                (output.loss / config.gradient_accumulation_steps).backward()
                step_loss += (
                    output.loss.detach().float().item() / config.gradient_accumulation_steps
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            elapsed = time.perf_counter() - started
            tokens_seen = mixture.state.total_tokens
            tokens_per_second = tokens_seen / max(elapsed, 1e-6)

            if step == config.calibration_steps:
                calibration_elapsed = time.perf_counter() - calibration_started
                calibration_tps = tokens_seen / max(calibration_elapsed, 1e-6)
                calibrated_budget = projected_token_budget(
                    calibration_tps,
                    config.time_budget_hours,
                    config.max_tokens,
                    config.tokens_per_optimizer_step,
                )

            if step % config.log_every_steps == 0:
                record = {
                    "step": step,
                    "loss": step_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": float(grad_norm),
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": tokens_per_second,
                    "reasoning_tokens": mixture.state.reasoning_tokens,
                    "general_tokens": mixture.state.general_tokens,
                    "calibrated_budget": calibrated_budget,
                }
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()

            if step % config.save_every_steps == 0:
                save_checkpoint(
                    output_dir / f"step-{step:08d}",
                    raw_model,
                    optimizer,
                    scheduler,
                    step,
                    tokens_seen,
                    mixture.state_dict(),
                    {**asdict(config), "model": model_config.to_dict()},
                )

    final = output_dir / "final"
    save_checkpoint(
        final,
        raw_model,
        optimizer,
        scheduler,
        step,
        mixture.state.total_tokens,
        mixture.state_dict(),
        {**asdict(config), "model": model_config.to_dict()},
    )
    return final
