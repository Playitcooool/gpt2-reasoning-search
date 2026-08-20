"""Single-H100 pretraining with steady-state calibration and token scheduling."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig, TrainConfig
from .data import ExactTokenMixture, MixtureState, load_token_array
from .model import GPT2ReasoningModel


class TokenCosineScheduler:
    """Warm up and decay by tokens, so gradient accumulation changes remain safe."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_fraction: float,
        total_tokens: int,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_fraction = warmup_fraction
        self.total_tokens = total_tokens
        self.tokens_seen = 0

    def learning_rate_at(self, tokens_seen: int) -> float:
        warmup_tokens = max(1, round(self.total_tokens * self.warmup_fraction))
        if tokens_seen < warmup_tokens:
            return self.peak_lr * tokens_seen / warmup_tokens
        progress = min(
            1.0,
            (tokens_seen - warmup_tokens) / max(1, self.total_tokens - warmup_tokens),
        )
        return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    def step(self, tokens_seen: int) -> float:
        self.tokens_seen = tokens_seen
        learning_rate = self.learning_rate_at(tokens_seen)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def set_total_tokens(self, total_tokens: int) -> None:
        if total_tokens < self.tokens_seen:
            raise ValueError("total token budget cannot be smaller than progress")
        self.total_tokens = total_tokens

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, int | float]:
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_fraction": self.warmup_fraction,
            "total_tokens": self.total_tokens,
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.peak_lr = float(state["peak_lr"])
        self.min_lr = float(state["min_lr"])
        self.warmup_fraction = float(state["warmup_fraction"])
        self.total_tokens = int(state["total_tokens"])
        self.tokens_seen = int(state["tokens_seen"])
        self.step(self.tokens_seen)


def projected_token_budget(
    tokens_per_second: float,
    hours: float,
    cap: int,
    quantum: int,
    *,
    tokens_seen: int = 0,
    elapsed_seconds: float = 0.0,
) -> int:
    remaining_seconds = max(0.0, hours * 3600 - elapsed_seconds)
    projected = tokens_seen + int(tokens_per_second * remaining_seconds)
    bounded = min(cap, projected) // quantum * quantum
    return max(min(cap, quantum), bounded)


def optimizer_parameter_groups(model: GPT2ReasoningModel, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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
    model = (
        torch.compile(raw_model, mode=config.compile_mode, dynamic=False)
        if config.compile_model
        else raw_model
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(raw_model, config.weight_decay),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=config.fused_optimizer,
    )

    reasoning = load_token_array(config.reasoning_tokens)
    general = load_token_array(config.general_tokens)
    quantum = config.tokens_per_optimizer_step
    maximum_budget = config.max_tokens // quantum * quantum
    if maximum_budget < quantum:
        raise ValueError("max_tokens must cover at least one optimizer step")
    scheduler = TokenCosineScheduler(
        optimizer,
        config.learning_rate,
        config.min_learning_rate,
        config.warmup_fraction,
        maximum_budget,
    )
    restored_state = None
    restored: dict[str, Any] | None = None
    if resume_from is not None:
        restored = load_checkpoint(resume_from, raw_model, optimizer, scheduler, device)
        restored_state = MixtureState(**restored["mixture_state"])
    mixture = ExactTokenMixture(
        reasoning, general, maximum_budget, config.reasoning_ratio, state=restored_state
    )
    batches = mixture.batches(config.micro_batch_size, config.sequence_length, device)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    step = int(restored["step"]) if restored is not None else 0
    started = time.perf_counter()
    deadline = started + config.time_budget_hours * 3600
    calibration_start_time: float | None = None
    calibration_start_tokens = 0
    calibrated_budget = scheduler.total_tokens
    parameter_count = raw_model.parameter_count()

    with metrics_path.open("a") as metrics:
        while mixture.state.total_tokens < calibrated_budget and time.perf_counter() < deadline:
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            next_tokens_seen = mixture.state.total_tokens + quantum
            learning_rate = scheduler.step(next_tokens_seen)
            for _ in range(config.gradient_accumulation_steps):
                tokens = next(batches)
                output = model(tokens, tokens)
                assert output.loss is not None
                (output.loss / config.gradient_accumulation_steps).backward()
                step_loss += (
                    output.loss.detach().float().item() / config.gradient_accumulation_steps
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
            optimizer.step()
            step += 1
            tokens_seen = mixture.state.total_tokens

            if step == config.calibration_warmup_steps:
                torch.cuda.synchronize()
                calibration_start_time = time.perf_counter()
                calibration_start_tokens = tokens_seen
            calibration_end = config.calibration_warmup_steps + config.calibration_steps
            if step == calibration_end and calibration_start_time is not None:
                torch.cuda.synchronize()
                steady_elapsed = time.perf_counter() - calibration_start_time
                steady_tokens = tokens_seen - calibration_start_tokens
                steady_tps = steady_tokens / max(steady_elapsed, 1e-6)
                calibrated_budget = projected_token_budget(
                    steady_tps,
                    config.time_budget_hours,
                    maximum_budget,
                    quantum,
                    tokens_seen=tokens_seen,
                    elapsed_seconds=time.perf_counter() - started,
                )
                scheduler.set_total_tokens(calibrated_budget)

            elapsed = time.perf_counter() - started
            tokens_per_second = tokens_seen / max(elapsed, 1e-6)
            model_flops_utilization = (
                6 * parameter_count * tokens_per_second / (config.peak_device_tflops * 1e12)
            )
            if step % config.log_every_steps == 0:
                record = {
                    "step": step,
                    "loss": step_loss,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(grad_norm),
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": tokens_per_second,
                    "model_flops_utilization": model_flops_utilization,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "reasoning_tokens": mixture.state.reasoning_tokens,
                    "general_tokens": mixture.state.general_tokens,
                    "calibrated_budget": calibrated_budget,
                    "elapsed_seconds": elapsed,
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
