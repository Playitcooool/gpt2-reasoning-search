from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from gpt2_reasoning_search.checkpoint import (
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
)
from gpt2_reasoning_search.config import ModelConfig
from gpt2_reasoning_search.model import GPT2ReasoningModel
from gpt2_reasoning_search.train import TokenCosineScheduler


def tiny_model() -> GPT2ReasoningModel:
    return GPT2ReasoningModel(
        ModelConfig(
            vocab_size=24,
            max_seq_len=8,
            n_layers=1,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=32,
        )
    )


def test_atomic_safetensors_round_trip_preserves_tied_weights(tmp_path: Path) -> None:
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = TokenCosineScheduler(optimizer, 0.01, 0.001, 0.1, 100)
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    save_checkpoint(tmp_path, model, optimizer, scheduler, 0, 0, {}, {"model": {}})
    assert (tmp_path / "model.safetensors").is_file()
    assert not list(tmp_path.glob("*.tmp"))

    restored = tiny_model()
    load_model_weights(tmp_path, restored, torch.device("cpu"))
    assert restored.lm_head.weight.data_ptr() == restored.token_embedding.weight.data_ptr()
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])


def test_load_model_weights_supports_legacy_model_pt(tmp_path: Path) -> None:
    model = tiny_model()
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    torch.save(model.state_dict(), tmp_path / "model.pt")

    restored = tiny_model()
    load_model_weights(tmp_path, restored, torch.device("cpu"))

    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])


def test_full_checkpoint_resume_restores_optimizer_scheduler_state_and_rng(
    tmp_path: Path,
) -> None:
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = TokenCosineScheduler(optimizer, 0.01, 0.001, 0.1, 1_000)
    tokens = torch.randint(0, model.config.vocab_size, (2, 5))
    loss = model(tokens, labels=tokens).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    scheduler.step(160)
    saved_optimizer = optimizer.state_dict()

    save_checkpoint(
        tmp_path,
        model,
        optimizer,
        scheduler,
        step=3,
        tokens_seen=160,
        mixture_state={
            "reasoning_tokens": 112,
            "general_tokens": 48,
            "reasoning_cursor": 12,
            "general_cursor": 5,
        },
        config={"model": model.config.to_dict()},
    )
    expected_random = (random.random(), np.random.random(), torch.rand(3))

    restored_model = tiny_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    restored_scheduler = TokenCosineScheduler(
        restored_optimizer, 0.5, 0.4, 0.5, 10
    )
    state = load_checkpoint(
        tmp_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        torch.device("cpu"),
    )

    assert state == json.loads((tmp_path / "state.json").read_text())
    assert state["step"] == 3
    assert state["mixture_state"]["reasoning_cursor"] == 12
    assert restored_scheduler.tokens_seen == 160
    assert restored_scheduler.total_tokens == 1_000
    assert restored_optimizer.state_dict()["param_groups"] == saved_optimizer["param_groups"]
    for old_state, new_state in zip(
        saved_optimizer["state"].values(),
        restored_optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        for key in old_state:
            torch.testing.assert_close(new_state[key], old_state[key])
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2])
