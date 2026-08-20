import asyncio
import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpt2_reasoning_search.cli import app
from gpt2_reasoning_search.evaluation import (
    compare_proxy_runs,
    percentile,
    retrieval_metrics,
    run_grounded_benchmark,
    score_grounded_records,
    score_language_model_records,
)
from gpt2_reasoning_search.schemas import (
    AnswerResponse,
    Citation,
    SearchArguments,
    SearchResult,
    ToolCall,
    ToolTraceEntry,
)


def test_percentile_uses_linear_interpolation_and_validates_quantile() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([4], 0.95) == 4.0
    assert percentile([4, 1, 2, 3], 0.5) == 2.5
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)
    for invalid in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="quantile"):
            percentile([1], invalid)
        with pytest.raises(ValueError, match="quantile"):
            percentile([], invalid)


def test_retrieval_metrics_deduplicate_before_cutoff_and_rank_relevant_results() -> None:
    metrics = retrieval_metrics(
        ["noise", "noise", "relevant-a", "relevant-b"],
        ["relevant-a", "relevant-b", "relevant-b"],
        cutoff=2,
    )

    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5
    expected_dcg = 1 / math.log2(3)
    expected_idcg = 1 + 1 / math.log2(3)
    assert metrics["ndcg"] == pytest.approx(expected_dcg / expected_idcg)
    assert retrieval_metrics(["relevant-a"], [], cutoff=0) == {
        "recall": 1.0,
        "mrr": 1.0,
        "ndcg": 1.0,
    }
    assert retrieval_metrics(["relevant-a"], ["relevant-a"], cutoff=0) == {
        "recall": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }
    with pytest.raises(ValueError, match="cutoff"):
        retrieval_metrics([], [], cutoff=-1)


def test_language_model_metrics_are_token_weighted_and_validate_rows() -> None:
    report = score_language_model_records(
        [{"loss": 1.0, "tokens": 1}, {"loss": 3.0, "tokens": 3}]
    )

    assert report["count_tokens"] == 4
    assert report["mean_loss"] == 2.5
    assert report["perplexity"] == pytest.approx(math.exp(2.5))
    with pytest.raises(ValueError, match="empty"):
        score_language_model_records([])
    for tokens in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="token"):
            score_language_model_records([{"loss": 1.0, "tokens": tokens}])
    for loss in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="loss"):
            score_language_model_records([{"loss": loss, "tokens": 1}])


def test_grounded_metrics_separate_citation_support_from_returned_id_validity() -> None:
    report = score_grounded_records(
        [
            {
                "prediction": "Paris [support]",
                "answer": "Paris",
                "retrieved_ids": ["support", "returned-noise"],
                "supporting_ids": ["support", "not-returned-support"],
                "cited_ids": ["support", "returned-noise", "invented"],
                "queries": ["capital of France"],
                "tool_calls_total": 3,
                "valid_tool_calls": 2,
            }
        ]
    )

    assert report["answer_exact_match"] == 1.0
    assert report["answer_token_f1"] == 1.0
    assert report["citation_precision"] == pytest.approx(1 / 3)
    assert report["citation_recall"] == 0.5
    assert report["citation_validity"] == pytest.approx(2 / 3)
    assert report["valid_tool_call_rate"] == pytest.approx(2 / 3)


def test_grounded_valid_call_legacy_boolean_and_zero_call_conventions() -> None:
    valid = score_grounded_records(
        [
            {
                "prediction": "x",
                "answer": "x",
                "queries": ["one", "two"],
                "tool_calls_total": 2,
                "valid_tool_calls": True,
                "retrieved_ids": [],
                "supporting_ids": [],
                "cited_ids": [],
            }
        ]
    )
    invalid = score_grounded_records(
        [
            {
                "prediction": "x",
                "answer": "x",
                "queries": ["one", "two"],
                "tool_calls_total": 2,
                "valid_tool_calls": False,
                "retrieved_ids": [],
                "supporting_ids": [],
                "cited_ids": [],
            }
        ]
    )
    no_calls = score_grounded_records(
        [
            {
                "prediction": "x",
                "answer": "x",
                "queries": [],
                "retrieved_ids": [],
                "supporting_ids": [],
                "cited_ids": [],
            }
        ]
    )

    assert valid["valid_tool_call_rate"] == 1.0
    assert invalid["valid_tool_call_rate"] == 0.0
    assert no_calls["valid_tool_call_rate"] == 1.0
    assert no_calls["citation_precision"] == 1.0
    assert no_calls["citation_recall"] == 1.0
    assert no_calls["citation_validity"] == 1.0


def test_grounded_recovery_denominator_latency_percentiles_and_per_mode_reports() -> None:
    rows = [
        {
            "search_mode": "off",
            "prediction": "wrong",
            "answer": "right",
            "queries": [],
            "retrieved_ids": [],
            "supporting_ids": ["s"],
            "cited_ids": [],
            "elapsed_seconds": 1.0,
        },
        {
            "search_mode": "local",
            "prediction": "right",
            "answer": "right",
            "queries": ["first", "reformulated"],
            "answer_found": True,
            "retrieved_ids": ["s"],
            "supporting_ids": ["s"],
            "cited_ids": ["s"],
            "elapsed_seconds": 2.0,
        },
        {
            "search_mode": "local",
            "prediction": "wrong",
            "answer": "right",
            "queries": ["first", "second", "third"],
            "answer_found": False,
            "retrieved_ids": [],
            "supporting_ids": ["s"],
            "cited_ids": [],
            "elapsed_seconds": 10.0,
        },
    ]

    report = score_grounded_records(rows)

    assert report["query_recovery_rate"] == 0.5
    assert report["latency_p50_seconds"] == 2.0
    assert report["latency_p95_seconds"] == pytest.approx(9.2)
    assert set(report["by_search_mode"]) == {"off", "local"}
    assert report["by_search_mode"]["off"]["count"] == 1
    assert report["by_search_mode"]["local"]["count"] == 2
    assert "by_search_mode" not in report["by_search_mode"]["local"]


class RecordingBenchmarkAgent:
    def __init__(self) -> None:
        self.requests = []

    async def answer(self, request):
        self.requests.append(request)
        searched = request.search_mode != "off"
        result = SearchResult(
            id="source-1",
            title="Source",
            url="https://example.test/source",
            snippet="evidence",
            content="evidence",
        )
        trace = (
            [
                ToolTraceEntry(
                    call=ToolCall(
                        name="search",
                        arguments=SearchArguments(query="reformulated question", top_k=5),
                    ),
                    results=[result],
                    status="ok",
                    elapsed_seconds=0.1,
                ),
                ToolTraceEntry(status="invalid", error="bad JSON"),
            ]
            if searched
            else []
        )
        citations = (
            [
                Citation(
                    source_id="source-1",
                    title="Source",
                    url="https://example.test/source",
                )
            ]
            if searched
            else []
        )
        suffix = " [source-1]" if searched else ""
        return AnswerResponse(
            answer=f"Paris{suffix}",
            reasoning="scratch",
            citations=citations,
            tool_trace=trace,
            input_tokens=10,
            output_tokens=4,
            elapsed_seconds=0.25,
            searches_used=int(searched),
        )

    async def aclose(self) -> None:
        return None


def test_run_grounded_benchmark_writes_complete_matched_jsonl_records(tmp_path: Path) -> None:
    agent = RecordingBenchmarkAgent()
    output = tmp_path / "nested" / "grounded.jsonl"
    examples = [
        {
            "id": "q1",
            "question": "What is France's capital?",
            "answer": "Paris",
            "supporting_ids": ["source-1"],
            "search_required": True,
        }
    ]

    count = asyncio.run(
        run_grounded_benchmark(
            agent,
            examples,
            output,
            search_modes=("off", "local"),
            max_searches=2,
        )
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert count == 2
    assert [(request.search_mode, request.max_searches) for request in agent.requests] == [
        ("off", 2),
        ("local", 2),
    ]
    assert [record["search_mode"] for record in records] == ["off", "local"]
    assert records[0]["tool_calls_total"] == 0
    assert records[0]["valid_tool_calls"] == 0
    assert records[1]["queries"] == ["reformulated question"]
    assert records[1]["tool_calls_total"] == 2
    assert records[1]["valid_tool_calls"] == 1
    assert records[1]["retrieved_ids"] == ["source-1"]
    assert records[1]["cited_ids"] == ["source-1"]
    assert records[1]["answer_found"] is True
    assert records[1]["elapsed_seconds"] == 0.25


def _write_proxy_metrics(path: Path, ratio: float, tokens: int) -> None:
    path.mkdir()
    (path / "metrics.jsonl").write_text(
        json.dumps(
            {
                "loss": 2.0,
                "tokens_seen": tokens,
                "tokens_per_second": 100.0,
                "reasoning_tokens": round(tokens * ratio),
            }
        )
        + "\n"
    )


def test_proxy_comparison_reports_equal_and_unequal_budget_diagnostics(tmp_path: Path) -> None:
    equal = [tmp_path / f"equal-{ratio}" for ratio in (0, 30, 70)]
    for path, ratio in zip(equal, (0.0, 0.3, 0.7), strict=True):
        _write_proxy_metrics(path, ratio, 1_000)
    equal_report = compare_proxy_runs(equal)
    assert equal_report["equal_token_budget"] is True
    assert equal_report["token_budget_spread"] == 0

    unequal = [tmp_path / f"unequal-{ratio}" for ratio in (0, 30, 70)]
    for path, ratio, tokens in zip(unequal, (0.0, 0.3, 0.7), (900, 1_000, 1_100), strict=True):
        _write_proxy_metrics(path, ratio, tokens)
    unequal_report = compare_proxy_runs(unequal)
    assert unequal_report["equal_token_budget"] is False
    assert unequal_report["token_budget_spread"] == 200


def test_release_cli_exposes_new_commands_and_consistent_version(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["version"])
    lm_input = tmp_path / "losses.jsonl"
    lm_output = tmp_path / "lm-report.json"
    lm_input.write_text('{"loss":1.5,"tokens":2}\n')
    score_result = runner.invoke(
        app, ["score-lm", str(lm_input), "--output", str(lm_output)]
    )

    assert help_result.exit_code == 0
    assert "score-lm" in help_result.stdout
    assert "benchmark-grounded" in help_result.stdout
    assert version_result.exit_code == 0
    assert version_result.stdout.strip() == "0.2.0"
    assert score_result.exit_code == 0
    assert json.loads(lm_output.read_text())["count_tokens"] == 2


def test_benchmark_cli_auto_mode_enables_web_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text('{"question":"capital?","answer":"Paris"}\n')
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    index = tmp_path / "index"
    index.mkdir()
    output = tmp_path / "predictions.jsonl"
    enable_web_values: list[bool] = []

    def fake_load_agent(_checkpoint, _tokenizer, _index, enable_web=True):
        enable_web_values.append(enable_web)
        return RecordingBenchmarkAgent()

    monkeypatch.setattr("gpt2_reasoning_search.cli._load_agent", fake_load_agent)

    result = CliRunner().invoke(
        app,
        [
            "benchmark-grounded",
            str(examples),
            "--checkpoint",
            str(checkpoint),
            "--tokenizer-path",
            str(tokenizer),
            "--index",
            str(index),
            "--output",
            str(output),
            "--mode",
            "auto",
        ],
    )

    assert result.exit_code == 0
    assert enable_web_values == [True]
    assert json.loads(result.stdout)["records"] == 1


def test_documented_release_commands_and_formats_match_cli() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    data = (root / "docs" / "DATA.md").read_text()
    evaluation = (root / "docs" / "EVALUATION.md").read_text()
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text()

    assert ".npy" not in readme
    assert ".npy" not in data
    assert "reasoning.bin" in readme and "general.bin" in readme
    assert "BM25+dense HNSW" in readme
    assert "Default: Tantivy BM25 + SentenceTransformer" in readme
    assert "--output artifacts/wiki-index-lexical --lexical-only" in " ".join(readme.split())
    assert "score-lm" in readme
    assert "benchmark-grounded" in readme
    assert "Recall/MRR/nDCG" in evaluation
    assert "uv sync --dev --locked" in workflow
    assert "uv run ruff check ." in workflow
    assert "uv run pytest" in workflow
