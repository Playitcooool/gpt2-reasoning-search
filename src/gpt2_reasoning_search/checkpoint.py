"""Complete resumable training checkpoints."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    tokens_seen: int,
    mixture_state: dict[str, int],
    config: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    torch.save(scheduler.state_dict(), directory / "scheduler.pt")
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        directory / "rng.pt",
    )
    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "mixture_state": mixture_state,
        "config": config,
    }
    (directory / "state.json").write_text(json.dumps(state, indent=2, default=str) + "\n")


def load_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location=device, weights_only=True)
    )
    optimizer.load_state_dict(
        torch.load(directory / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(directory / "scheduler.pt", map_location=device, weights_only=True)
    )
    rng = torch.load(directory / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((directory / "state.json").read_text())
