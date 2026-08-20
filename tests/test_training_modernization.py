from __future__ import annotations

import math

import pytest
import torch

from gpt2_reasoning_search.config import ModelConfig
from gpt2_reasoning_search.model import GPT2ReasoningModel
from gpt2_reasoning_search.train import (
    TokenCosineScheduler,
    optimizer_parameter_groups,
    projected_token_budget,
)


def test_optimizer_parameter_groups_are_complete_disjoint_and_dimension_based() -> None:
    model = GPT2ReasoningModel(
        ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            n_layers=1,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=32,
        )
    )
    groups = optimizer_parameter_groups(model, 0.15)
    decay = groups[0]["params"]
    no_decay = groups[1]["params"]

    assert groups[0]["weight_decay"] == 0.15
    assert groups[1]["weight_decay"] == 0.0
    assert {id(parameter) for parameter in decay}.isdisjoint(
        {id(parameter) for parameter in no_decay}
    )
    assert {id(parameter) for parameter in decay + no_decay} == {
        id(parameter) for parameter in model.parameters()
    }
    assert all(parameter.ndim >= 2 for parameter in decay)
    assert all(parameter.ndim < 2 for parameter in no_decay)
    assert sum(parameter is model.token_embedding.weight for parameter in decay) == 1


def test_token_cosine_scheduler_warmup_decay_and_dynamic_budget() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = TokenCosineScheduler(optimizer, 1.0, 0.1, 0.1, 1_000)

    assert scheduler.learning_rate_at(0) == 0.0
    assert scheduler.learning_rate_at(50) == pytest.approx(0.5)
    assert scheduler.learning_rate_at(100) == pytest.approx(1.0)
    assert scheduler.learning_rate_at(550) == pytest.approx(0.55)
    assert scheduler.learning_rate_at(1_000) == pytest.approx(0.1)
    assert scheduler.learning_rate_at(2_000) == pytest.approx(0.1)

    scheduler.step(200)
    scheduler.set_total_tokens(600)
    assert scheduler.total_tokens == 600
    assert scheduler.learning_rate_at(600) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="smaller than progress"):
        scheduler.set_total_tokens(199)

    state = scheduler.state_dict()
    restored = TokenCosineScheduler(optimizer, 9.0, 8.0, 0.5, 99)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(scheduler.learning_rate_at(200))


def test_calibration_projection_accounts_for_elapsed_time_and_progress() -> None:
    budget = projected_token_budget(
        tokens_per_second=100,
        hours=1,
        cap=1_000_000,
        quantum=1_000,
        tokens_seen=50_000,
        elapsed_seconds=600,
    )
    assert budget == 350_000
    assert math.isclose((budget - 50_000) / 100, 3_000)


def test_calibration_projection_never_exceeds_cap_after_elapsed_budget() -> None:
    assert (
        projected_token_budget(
            1_000,
            hours=1,
            cap=100_000,
            quantum=1_000,
            tokens_seen=80_000,
            elapsed_seconds=3_600,
        )
        == 80_000
    )
