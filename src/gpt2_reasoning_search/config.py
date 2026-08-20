"""Typed configuration for model and training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 50_304
    max_seq_len: int = 2_048
    n_layers: int = 24
    d_model: int = 1_024
    n_heads: int = 16
    n_kv_heads: int = 4
    intermediate_size: int = 3_072
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        _ = self.head_dim
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len < 2 or self.n_layers < 1 or self.intermediate_size < 1:
            raise ValueError("model dimensions must be positive")

    @classmethod
    def preset(cls, name: Literal["proxy-124m", "main-350m"]) -> ModelConfig:
        if name == "proxy-124m":
            return cls(
                n_layers=12,
                d_model=768,
                n_heads=12,
                n_kv_heads=4,
                intermediate_size=2_304,
            )
        if name == "main-350m":
            return cls()
        raise ValueError(f"unknown model preset: {name}")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(slots=True)
class TrainConfig:
    output_dir: Path
    reasoning_tokens: Path
    general_tokens: Path
    model_preset: Literal["proxy-124m", "main-350m"] = "main-350m"
    reasoning_ratio: float = 0.70
    max_tokens: int = 2_500_000_000
    sequence_length: int = 2_048
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_fraction: float = 0.01
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    save_every_steps: int = 1_000
    log_every_steps: int = 10
    time_budget_hours: float = 18.0
    calibration_steps: int = 20
    calibration_warmup_steps: int = 5
    compile_model: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    fused_optimizer: bool = True
    peak_device_tflops: float = 989.0

    def __post_init__(self) -> None:
        if not 0 <= self.reasoning_ratio <= 1:
            raise ValueError("reasoning_ratio must be between zero and one")
        if self.sequence_length <= 1 or self.max_tokens < self.sequence_length:
            raise ValueError("invalid token or sequence budget")
        if self.calibration_steps < 1 or self.calibration_warmup_steps < 0:
            raise ValueError("invalid calibration window")
        if self.time_budget_hours <= 0 or self.learning_rate <= 0:
            raise ValueError("time budget and learning rate must be positive")

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.sequence_length * self.micro_batch_size * self.gradient_accumulation_steps
