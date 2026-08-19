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
    ):
        assert command in result.stdout
