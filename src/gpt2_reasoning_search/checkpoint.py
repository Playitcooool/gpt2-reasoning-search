"""Atomic, resumable training checkpoints with SafeTensors model weights."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_model, save_model
from torch import nn


def _atomic_torch_save(value: Any, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(value: Any, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", mode="w", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_safetensors_save(model: nn.Module, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_model(model, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    _atomic_safetensors_save(model, directory / "model.safetensors")
    _atomic_torch_save(optimizer.state_dict(), directory / "optimizer.pt")
    _atomic_torch_save(scheduler.state_dict(), directory / "scheduler.pt")
    _atomic_torch_save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        directory / "rng.pt",
    )
    _atomic_json_save(
        {
            "format_version": 2,
            "step": step,
            "tokens_seen": tokens_seen,
            "mixture_state": mixture_state,
            "config": config,
        },
        directory / "state.json",
    )


def load_model_weights(directory: Path, model: nn.Module, device: torch.device) -> None:
    safe_path = directory / "model.safetensors"
    if safe_path.exists():
        missing, unexpected = load_model(model, safe_path, device=str(device))
        if missing or unexpected:
            raise RuntimeError(
                f"invalid checkpoint keys: missing={missing}, unexpected={unexpected}"
            )
        return
    legacy_path = directory / "model.pt"
    if not legacy_path.exists():
        raise FileNotFoundError(f"no model weights found in {directory}")
    model.load_state_dict(torch.load(legacy_path, map_location=device, weights_only=True))


def load_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> dict[str, Any]:
    load_model_weights(directory, model, device)
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
