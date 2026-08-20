import json
import math
from pathlib import Path

import pytest

from gpt2_reasoning_search.evaluation import (
    compare_proxy_runs,
    exact_match,
    normalize_answer,
    perplexity,
    score_grounded_records,
    score_reasoning_records,
    stream_jsonl,
    token_f1,
    write_report,
)
from gpt2_reasoning_search.experiment import one_h100_schedule, write_experiment_plan
from gpt2_reasoning_search.smoke import tiny_overfit


def test_answer_normalization_exact_match_and_duplicate_aware_f1() -> None:
    assert normalize_answer("  The, QUICK-brown fox! ") == "quick brown fox"
    assert exact_match("An Answer.", "answer") == 1.0
    assert exact_match("answer one", "answer two") == 0.0
    assert token_f1("red red blue", "red blue blue") == pytest.approx(2 / 3)
    assert token_f1("the", "an") == 1.0
    assert token_f1("the", "substance") == 0.0


def test_reasoning_metrics_are_weighted_and_grouped_by_task() -> None:
    report = score_reasoning_records(
        [
            {"task": "math", "prediction": "42", "answer": "42"},
            {"task": "math", "prediction": "41", "answer": "42"},
            {"task": "logic", "prediction": "Option C", "answer": "C"},
        ]
    )

    assert report["count"] == 3
    assert report["exact_match"] == pytest.approx(1 / 3)
    assert report["token_f1"] == pytest.approx(5 / 9)
    assert report["by_task"]["math"] == {
        "count": 2,
        "exact_match": 0.5,
        "token_f1": 0.5,
    }
    assert report["by_task"]["logic"]["count"] == 1
    assert score_reasoning_records([]) == {
        "count": 0,
        "exact_match": 0.0,
        "token_f1": 0.0,
        "by_task": {},
    }


def test_grounded_metrics_cover_retrieval_citations_tools_and_recovery() -> None:
    report = score_grounded_records(
        [
            {
                "prediction": "Paris",
                "answer": "Paris",
                "valid_tool_calls": True,
                "queries": ["France capital", "Paris France"],
                "search_required": True,
                "answer_found": True,
                "retrieved_ids": ["a", "noise"],
                "supporting_ids": ["a", "b"],
                "cited_ids": ["a", "invented"],
            },
            {
                "prediction": "4",
                "answer": "4",
                "valid_tool_calls": False,
                "queries": ["2 plus 2"],
                "search_required": False,
                "answer_found": True,
                "retrieved_ids": [],
                "supporting_ids": [],
                "cited_ids": [],
            },
        ]
    )

    assert report["count"] == 2
    assert report["answer_exact_match"] == 1.0
    assert report["answer_token_f1"] == 1.0
    assert report["retrieval_recall"] == 0.75
    assert report["citation_precision"] == 0.75
    assert report["citation_recall"] == 0.75
    assert report["citation_validity"] == 0.75
    assert report["valid_tool_call_rate"] == pytest.approx(2 / 3)
    assert report["search_rate"] == 1.0
    assert report["unnecessary_search_rate"] == 0.5
    assert report["query_recovery_rate"] == 1.0


def test_grounded_metrics_empty_input_has_finite_zero_rates() -> None:
    report = score_grounded_records([])

    assert report["count"] == 0
    assert report["valid_tool_call_rate"] == 1.0
    assert all(
        value == 0.0
        for key, value in report.items()
        if key not in {"count", "valid_tool_call_rate"}
    )


def test_perplexity_caps_extreme_loss() -> None:
    assert perplexity(0.0) == 1.0
    assert perplexity(math.log(10)) == pytest.approx(10.0)
    assert perplexity(100.0) == pytest.approx(math.exp(20.0))


def _write_metrics(directory: Path, ratio: float, loss: float, tokens: int = 100) -> None:
    directory.mkdir()
    reasoning = round(tokens * ratio)
    records = [
        {
            "step": 1,
            "loss": loss + 1,
            "tokens_seen": tokens // 2,
            "tokens_per_second": 10,
            "reasoning_tokens": reasoning // 2,
        },
        {
            "step": 2,
            "loss": loss,
            "tokens_seen": tokens,
            "tokens_per_second": 12,
            "reasoning_tokens": reasoning,
        },
    ]
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def test_proxy_comparison_uses_final_record_and_sorts_by_ratio(tmp_path: Path) -> None:
    run70 = tmp_path / "proxy-r70"
    run0 = tmp_path / "proxy-r0"
    run30 = tmp_path / "proxy-r30"
    _write_metrics(run70, 0.7, 2.2)
    _write_metrics(run0, 0.0, 2.8)
    _write_metrics(run30, 0.3, 2.5)

    report = compare_proxy_runs([run70, run0, run30])

    assert [run["run"] for run in report["runs"]] == ["proxy-r0", "proxy-r30", "proxy-r70"]
    assert [run["reasoning_ratio"] for run in report["runs"]] == [0.0, 0.3, 0.7]
    assert [run["final_loss"] for run in report["runs"]] == [2.8, 2.5, 2.2]
    assert all(run["tokens_seen"] == 100 for run in report["runs"])

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="no metrics"):
        compare_proxy_runs([empty])


def test_one_h100_schedule_has_equal_proxies_and_24_hour_total() -> None:
    schedule = one_h100_schedule()
    proxies = [run for run in schedule if run.preset == "proxy-124m"]
    main = [run for run in schedule if run.preset == "main-350m"]

    assert [run.reasoning_ratio for run in proxies] == [0.0, 0.3, 0.7]
    assert len({run.token_cap for run in proxies}) == 1
    assert len({run.time_budget_hours for run in proxies}) == 1
    assert len({run.seed for run in proxies}) == 1
    assert len(main) == 1 and main[0].reasoning_ratio == 0.7
    assert sum(run.time_budget_hours for run in schedule) == 18.5
    assert sum(run.time_budget_hours for run in schedule) + 5.5 == 24.0


def test_experiment_plan_and_report_writing_are_machine_readable(tmp_path: Path) -> None:
    plan_path = tmp_path / "nested" / "plan.json"
    report_path = tmp_path / "reports" / "report.json"

    write_experiment_plan(plan_path)
    write_report({"z": 2, "a": 1}, report_path)
    plan = json.loads(plan_path.read_text())

    assert plan["hardware"] == "single NVIDIA H100"
    assert plan["total_scheduled_training_hours"] == 18.5
    assert plan["reserved_calibration_sft_rl_evaluation_hours"] == 5.5
    assert len(plan["runs"]) == 4
    assert "Do not claim" in plan["notes"][-1]
    assert report_path.read_text() == '{\n  "a": 1,\n  "z": 2\n}\n'


def test_stream_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"prediction":"x","answer":"x"}\n\n  \n')
    assert list(stream_jsonl(path)) == [{"prediction": "x", "answer": "x"}]


def test_tiny_overfit_learning_gate_passes() -> None:
    result = tiny_overfit(steps=40, device="cpu")

    assert result["passed"] is True
    assert float(result["final_loss"]) < float(result["initial_loss"]) * 0.35


def test_readme_documents_complete_pipeline_and_generated_reasoning_warning() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    data_guide = (root / "docs" / "DATA.md").read_text()

    for command in (
        "experiment-plan",
        "smoke-overfit",
        "score-reasoning",
        "score-grounded",
        "compare-proxies",
    ):
        assert command in readme
    assert "not a faithful description" in " ".join(readme.split())
    assert "full commit hash" in data_guide
    assert "Common Crawl obligations" in data_guide
