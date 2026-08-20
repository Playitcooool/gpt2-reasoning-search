import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2_reasoning_search.checkpoint import load_checkpoint, save_checkpoint
from gpt2_reasoning_search.config import TrainConfig
from gpt2_reasoning_search.prepare import load_dataset_manifest
from gpt2_reasoning_search.train import projected_token_budget, train


def test_checkpoint_round_trip_restores_training_and_rng_state(tmp_path: Path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}

    directory = tmp_path / "checkpoint"
    save_checkpoint(
        directory,
        model,
        optimizer,
        scheduler,
        step=4,
        tokens_seen=128,
        mixture_state={
            "reasoning_tokens": 90,
            "general_tokens": 38,
            "reasoning_cursor": 9,
            "general_cursor": 3,
        },
        config={"name": "tiny"},
    )
    expected_rng = (random.random(), np.random.random(), torch.rand(1))
    with torch.no_grad():
        model.weight.zero_()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    state = load_checkpoint(directory, model, optimizer, scheduler, torch.device("cpu"))
    actual_rng = (random.random(), np.random.random(), torch.rand(1))

    assert state["step"] == 4
    assert state["tokens_seen"] == 128
    assert state["mixture_state"]["reasoning_tokens"] == 90
    assert scheduler.last_epoch == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name])
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    torch.testing.assert_close(actual_rng[2], expected_rng[2])


@pytest.mark.parametrize(
    ("rate", "hours", "cap", "quantum", "expected"),
    [
        (100.0, 1.0, 1_000_000, 1_000, 360_000),
        (1_000.0, 1.0, 100_000, 1_000, 100_000),
        (1.0, 0.01, 100_000, 1_000, 1_000),
    ],
)
def test_projected_token_budget_is_capped_and_quantized(
    rate: float, hours: float, cap: int, quantum: int, expected: int
) -> None:
    assert projected_token_budget(rate, hours, cap, quantum) == expected


def test_pretrain_deadline_saves_resumable_step_without_marking_final(
    tmp_path: Path, monkeypatch
) -> None:
    import gpt2_reasoning_search.train as train_module

    class FakeModel(torch.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, *args, **kwargs):
            return self

        def parameter_count(self) -> int:
            return 1

    original_sgd = torch.optim.SGD
    saves: list[tuple[Path, int, int, dict[str, int]]] = []
    restored_state = {
        "reasoning_tokens": 1,
        "general_tokens": 1,
        "reasoning_cursor": 1,
        "general_cursor": 1,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_module, "GPT2ReasoningModel", FakeModel)
    monkeypatch.setattr(train_module, "load_token_array", lambda _path: np.arange(16))
    monkeypatch.setattr(
        train_module.torch.optim,
        "AdamW",
        lambda params, lr, **_kwargs: original_sgd(params, lr=lr),
    )
    monkeypatch.setattr(
        train_module,
        "save_checkpoint",
        lambda directory, _model, _optimizer, _scheduler, step, tokens, mixture, _config: (
            saves.append((directory, step, tokens, mixture.copy()))
        ),
    )
    config = TrainConfig(
        output_dir=tmp_path / "output",
        reasoning_tokens=tmp_path / "reasoning.bin",
        general_tokens=tmp_path / "general.bin",
        sequence_length=2,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_tokens=4,
        time_budget_hours=1.0,
        compile_model=False,
        fused_optimizer=False,
    )

    clock_calls = 0

    def expired_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 3_601.0

    monkeypatch.setattr(train_module.time, "perf_counter", expired_clock)
    first = train(config)

    assert first == tmp_path / "output" / "step-00000000"
    assert saves[-1] == (
        first,
        0,
        0,
        {"reasoning_tokens": 0, "general_tokens": 0, "reasoning_cursor": 0, "general_cursor": 0},
    )
    assert not (tmp_path / "output" / "final").exists()

    monkeypatch.setattr(
        train_module,
        "load_checkpoint",
        lambda *_args: {"step": 3, "mixture_state": restored_state},
    )
    clock_calls = 0
    resumed = train(config, resume_from=tmp_path / "resume")

    assert resumed == tmp_path / "output" / "step-00000003"
    assert saves[-1] == (resumed, 3, 2, restored_state)

    completed_state = {
        "reasoning_tokens": 2,
        "general_tokens": 2,
        "reasoning_cursor": 2,
        "general_cursor": 2,
    }
    monkeypatch.setattr(
        train_module,
        "load_checkpoint",
        lambda *_args: {"step": 4, "mixture_state": completed_state},
    )
    monkeypatch.setattr(train_module.time, "perf_counter", lambda: 0.0)
    completed = train(config, resume_from=tmp_path / "completed-resume")

    assert completed == tmp_path / "output" / "final"
    assert saves[-1] == (completed, 4, 4, completed_state)


def test_dataset_manifest_has_immutable_revisions_and_license_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_dataset_manifest(root / "config" / "datasets.json")
    required = {
        "allenai/big-reasoning-traces",
        "open-r1/OpenR1-Math-220k",
        "HuggingFaceFW/fineweb-edu",
    }

    assert required <= manifest.keys()
    for dataset_id in required:
        source = manifest[dataset_id]
        assert len(source["revision"]) == 40
        int(source["revision"], 16)
        assert source["license"].strip()
        assert source["role"].strip()
    assert manifest["HuggingFaceFW/fineweb-edu"]["config"] == "sample-10BT"
    json.dumps(manifest)
