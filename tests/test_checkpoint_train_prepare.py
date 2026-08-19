import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2_reasoning_search.checkpoint import load_checkpoint, save_checkpoint
from gpt2_reasoning_search.prepare import load_dataset_manifest
from gpt2_reasoning_search.train import projected_token_budget


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
