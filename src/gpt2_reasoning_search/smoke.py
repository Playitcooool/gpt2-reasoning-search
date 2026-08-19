"""Tiny overfit gate for model and tool-trajectory learning."""

from __future__ import annotations

import torch

from .config import ModelConfig
from .model import GPT2ReasoningModel


def tiny_overfit(steps: int = 40, device: str = "cpu") -> dict[str, float | bool]:
    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        intermediate_size=64,
        gradient_checkpointing=False,
    )
    model = GPT2ReasoningModel(config).to(device)
    tokens = torch.tensor(
        [[1, 3, 11, 12, 5, 20, 21, 6, 22, 2, 1, 7, 30, 8, 31, 2]],
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    with torch.no_grad():
        initial = float(model(tokens, tokens).loss)
    final = initial
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, tokens).loss
        assert loss is not None
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
    return {"initial_loss": initial, "final_loss": final, "passed": final < initial * 0.35}
