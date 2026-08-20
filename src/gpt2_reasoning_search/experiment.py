"""Decision-complete one-H100 experiment schedule."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ScheduledRun:
    name: str
    preset: str
    reasoning_ratio: float
    token_cap: int
    time_budget_hours: float
    seed: int = 42


def one_h100_schedule() -> list[ScheduledRun]:
    """Schedule the 18.5 hours assigned to proxy and main pretraining."""
    proxy_cap = 150_000_000
    return [
        ScheduledRun("proxy-r0", "proxy-124m", 0.0, proxy_cap, 1.5),
        ScheduledRun("proxy-r30", "proxy-124m", 0.3, proxy_cap, 1.5),
        ScheduledRun("proxy-r70", "proxy-124m", 0.7, proxy_cap, 1.5),
        ScheduledRun("main-r70", "main-350m", 0.7, 2_500_000_000, 14.0),
    ]


def write_experiment_plan(output: Path) -> None:
    plan = {
        "hardware": "single NVIDIA H100",
        "total_scheduled_training_hours": sum(run.time_budget_hours for run in one_h100_schedule()),
        "reserved_calibration_sft_rl_evaluation_hours": 5.5,
        "runs": [asdict(run) for run in one_h100_schedule()],
        "notes": [
            "Each token cap is reduced after calibration when necessary to honor its time budget.",
            "Proxy runs use equal token and time caps; "
            "the 350M run is not used as a causal ablation.",
            "Do not claim a 70% mixture improvement unless held-out results support it.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n")
