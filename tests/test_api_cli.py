import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from gpt2_reasoning_search.agent import SearchAgent
from gpt2_reasoning_search.api import create_app
from gpt2_reasoning_search.cli import app
from gpt2_reasoning_search.schemas import SearchResult


class RecordingProvider:
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return []


def test_health_and_unconfigured_answer_return_503(monkeypatch) -> None:
    for name in ("GRS_CHECKPOINT", "GRS_TOKENIZER", "GRS_SEARCH_INDEX"):
        monkeypatch.delenv(name, raising=False)
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
    response = client.post("/v1/answer", json={"query": "question"})
    assert response.status_code == 503
    assert response.json()["detail"] == "model and search index are not configured"


def test_configured_api_health_and_answer() -> None:
    agent = SearchAgent(
        lambda _prompt, _limit: (
            "<|reasoning|>Simple arithmetic.<|answer|>Four.",
            8,
            5,
        ),
        RecordingProvider(),
    )
    client = TestClient(create_app(agent))

    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    response = client.post(
        "/v1/answer", json={"query": "What is 2 + 2?", "search_mode": "off"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Four."
    assert payload["reasoning"] == "Simple arithmetic."
    assert payload["citations"] == [] and payload["tool_trace"] == []
    assert payload["input_tokens"] == 8 and payload["output_tokens"] == 5
    assert "not a faithful account" in payload["reasoning_notice"]


def test_api_maps_missing_web_provider_to_503() -> None:
    agent = SearchAgent(
        lambda _prompt, _limit: (
            '<|tool_call|>{"name":"search","arguments":{"query":"today"}}'
            "<|end_tool_call|>",
            2,
            2,
        ),
        RecordingProvider(),
    )
    response = TestClient(create_app(agent)).post(
        "/v1/answer", json={"query": "current fact", "search_mode": "web"}
    )

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_cli_help_lists_training_search_and_serving_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "train-tokenizer",
        "prepare-data",
        "pretrain",
        "build-index",
        "make-trajectories",
        "sft-tools",
        "ask",
        "serve",
        "experiment-plan",
        "smoke-overfit",
        "score-reasoning",
        "score-grounded",
        "compare-proxies",
    ):
        assert command in result.stdout


def test_evaluation_and_plan_cli_commands_write_reports(tmp_path) -> None:
    runner = CliRunner()
    plan = tmp_path / "plan.json"
    reasoning_input = tmp_path / "reasoning.jsonl"
    reasoning_report = tmp_path / "reasoning-report.json"
    grounded_input = tmp_path / "grounded.jsonl"
    grounded_report = tmp_path / "grounded-report.json"
    reasoning_input.write_text(
        '{"task":"math","prediction":"The 4","answer":"4"}\n'
    )
    grounded_input.write_text(
        '{"prediction":"Paris","answer":"Paris","queries":[],"retrieved_ids":[],'
        '"supporting_ids":[],"cited_ids":[]}\n'
    )

    plan_result = runner.invoke(app, ["experiment-plan", "--output", str(plan)])
    reasoning_result = runner.invoke(
        app,
        [
            "score-reasoning",
            str(reasoning_input),
            "--output",
            str(reasoning_report),
        ],
    )
    grounded_result = runner.invoke(
        app,
        [
            "score-grounded",
            str(grounded_input),
            "--output",
            str(grounded_report),
        ],
    )

    assert plan_result.exit_code == 0 and plan.is_file()
    assert reasoning_result.exit_code == 0 and reasoning_report.is_file()
    assert grounded_result.exit_code == 0 and grounded_report.is_file()
    assert json.loads(reasoning_report.read_text())["exact_match"] == 1.0
    assert json.loads(grounded_report.read_text())["answer_exact_match"] == 1.0
