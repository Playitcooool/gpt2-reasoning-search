from __future__ import annotations

import asyncio

import httpx
from fastapi.testclient import TestClient

import gpt2_reasoning_search.api as api_module
from gpt2_reasoning_search.api import create_app
from gpt2_reasoning_search.schemas import AnswerRequest, AnswerResponse


def response_for(request: AnswerRequest) -> AnswerResponse:
    return AnswerResponse(
        answer=request.query,
        reasoning="test",
        citations=[],
        tool_trace=[],
        input_tokens=1,
        output_tokens=1,
        elapsed_seconds=0.0,
    )


class DelayedAgent:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    async def answer(self, request: AnswerRequest) -> AnswerResponse:
        await asyncio.sleep(self.delay)
        return response_for(request)


def test_liveness_readiness_request_id_and_metrics(monkeypatch) -> None:
    monkeypatch.setenv("GRS_MAX_CONCURRENT_REQUESTS", "1")
    agent = DelayedAgent()
    with TestClient(create_app(agent)) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "model_loaded": True,
        }
        answer = client.post(
            "/v1/answer",
            headers={"X-Request-ID": "caller-id"},
            json={"query": "hello", "search_mode": "off"},
        )
        assert answer.status_code == 200
        assert answer.headers["X-Request-ID"] == "caller-id"
        assert answer.json()["answer"] == "hello"
        assert client.get("/metrics").json() == {"requests": 1, "errors": 0, "busy": 0}


def test_unconfigured_readiness_is_503_but_liveness_is_healthy(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_agent_from_environment", lambda: None)
    with TestClient(create_app()) as client:
        assert client.get("/health/live").status_code == 200
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]


def test_concurrency_queue_timeout_returns_429_and_counts_busy(monkeypatch) -> None:
    monkeypatch.setenv("GRS_MAX_CONCURRENT_REQUESTS", "1")
    monkeypatch.setenv("GRS_QUEUE_TIMEOUT_SECONDS", "0.1")
    app = create_app(DelayedAgent(0.25))

    async def exercise() -> tuple[list[httpx.Response], dict[str, int]]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            responses = await asyncio.gather(
                client.post("/v1/answer", json={"query": "one"}),
                client.post("/v1/answer", json={"query": "two"}),
            )
            metrics = (await client.get("/metrics")).json()
            return responses, metrics

    responses, metrics = asyncio.run(exercise())

    assert sorted(response.status_code for response in responses) == [200, 429]
    assert metrics == {"requests": 2, "errors": 0, "busy": 1}


def test_request_timeout_returns_504_and_counts_error(monkeypatch) -> None:
    monkeypatch.setenv("GRS_REQUEST_TIMEOUT_SECONDS", "1")
    app = create_app(DelayedAgent(1.1))

    with TestClient(app) as client:
        response = client.post("/v1/answer", json={"query": "slow"})
        metrics = client.get("/metrics").json()

    assert response.status_code == 504
    assert response.json()["detail"] == "answer request timed out"
    assert metrics == {"requests": 1, "errors": 1, "busy": 0}


def test_lifespan_closes_environment_owned_agent(monkeypatch) -> None:
    class ClosableAgent(DelayedAgent):
        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        async def aclose(self) -> None:
            self.closed += 1

    agent = ClosableAgent()
    monkeypatch.setattr(api_module, "_agent_from_environment", lambda: agent)

    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200

    assert agent.closed == 1
