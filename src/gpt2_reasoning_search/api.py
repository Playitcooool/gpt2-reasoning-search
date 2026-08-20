"""FastAPI service for grounded answers."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response

from .agent import SearchAgent
from .retrieval import LocalWikipediaSearchProvider
from .runner import ModelRunner
from .schemas import AnswerRequest, AnswerResponse
from .web_search import BraveWebSearchProvider, SQLiteSearchCache

LOGGER = logging.getLogger(__name__)


def _agent_from_environment() -> SearchAgent | None:
    checkpoint = os.getenv("GRS_CHECKPOINT")
    tokenizer = os.getenv("GRS_TOKENIZER")
    index = os.getenv("GRS_SEARCH_INDEX")
    if not (checkpoint and tokenizer and index):
        return None
    runner = ModelRunner(Path(checkpoint), Path(tokenizer))
    local = LocalWikipediaSearchProvider(Path(index), enable_reranker=True)
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
    cache_path = Path(os.getenv("GRS_SEARCH_CACHE", "artifacts/search-cache.sqlite3"))
    web = (
        BraveWebSearchProvider(brave_key, cache=SQLiteSearchCache(cache_path))
        if brave_key
        else None
    )
    return SearchAgent(runner.generate, local, web)


def create_app(agent: SearchAgent | None = None) -> FastAPI:
    configured_agent = agent
    owns_agent = agent is None
    concurrency = max(1, int(os.getenv("GRS_MAX_CONCURRENT_REQUESTS", "1")))
    queue_timeout = max(0.1, float(os.getenv("GRS_QUEUE_TIMEOUT_SECONDS", "5")))
    request_timeout = max(1.0, float(os.getenv("GRS_REQUEST_TIMEOUT_SECONDS", "120")))
    semaphore = asyncio.Semaphore(concurrency)
    metrics = {"requests": 0, "errors": 0, "busy": 0}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal configured_agent
        if configured_agent is None:
            configured_agent = _agent_from_environment()
        yield
        if owns_agent and configured_agent is not None:
            await configured_agent.aclose()

    app = FastAPI(
        title="GPT-2 Reasoning Search",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str | bool]:
        is_ready = configured_agent is not None
        if not is_ready:
            raise HTTPException(status_code=503, detail="model and index are not configured")
        return {"status": "ready", "model_loaded": True}

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {"status": "ok", "model_loaded": configured_agent is not None}

    @app.get("/metrics")
    async def service_metrics() -> dict[str, int]:
        return dict(metrics)

    @app.post("/v1/answer", response_model=AnswerResponse)
    async def answer(
        request: AnswerRequest,
        response: Response,
        x_request_id: str | None = Header(default=None),
    ) -> AnswerResponse:
        if configured_agent is None:
            raise HTTPException(status_code=503, detail="model and search index are not configured")
        request_id = x_request_id or uuid.uuid4().hex
        response.headers["X-Request-ID"] = request_id
        metrics["requests"] += 1
        started = time.perf_counter()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=queue_timeout)
        except TimeoutError as exc:
            metrics["busy"] += 1
            raise HTTPException(status_code=429, detail="inference queue is full") from exc
        try:
            result = await asyncio.wait_for(
                configured_agent.answer(request), timeout=request_timeout
            )
            LOGGER.info(
                "answer_complete request_id=%s elapsed_seconds=%.3f searches=%d",
                request_id,
                time.perf_counter() - started,
                result.searches_used,
            )
            return result
        except TimeoutError as exc:
            metrics["errors"] += 1
            raise HTTPException(status_code=504, detail="answer request timed out") from exc
        except RuntimeError as exc:
            metrics["errors"] += 1
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            semaphore.release()

    return app


app = create_app()
