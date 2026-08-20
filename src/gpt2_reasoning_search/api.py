"""FastAPI service for grounded answers."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .agent import SearchAgent
from .retrieval import LocalWikipediaSearchProvider
from .runner import ModelRunner
from .schemas import AnswerRequest, AnswerResponse
from .web_search import BraveWebSearchProvider, SQLiteSearchCache


def create_app(agent: SearchAgent | None = None) -> FastAPI:
    app = FastAPI(title="GPT-2 Reasoning Search", version="0.1.0")
    configured_agent = agent

    if configured_agent is None:
        checkpoint = os.getenv("GRS_CHECKPOINT")
        tokenizer = os.getenv("GRS_TOKENIZER")
        index = os.getenv("GRS_SEARCH_INDEX")
        if checkpoint and tokenizer and index:
            runner = ModelRunner(Path(checkpoint), Path(tokenizer))
            local = LocalWikipediaSearchProvider(Path(index), enable_reranker=True)
            brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
            cache_path = Path(os.getenv("GRS_SEARCH_CACHE", "artifacts/search-cache.sqlite3"))
            web = (
                BraveWebSearchProvider(brave_key, cache=SQLiteSearchCache(cache_path))
                if brave_key
                else None
            )
            configured_agent = SearchAgent(runner.generate, local, web)

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {"status": "ok", "model_loaded": configured_agent is not None}

    @app.post("/v1/answer", response_model=AnswerResponse)
    async def answer(request: AnswerRequest) -> AnswerResponse:
        if configured_agent is None:
            raise HTTPException(status_code=503, detail="model and search index are not configured")
        try:
            return await configured_agent.answer(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
