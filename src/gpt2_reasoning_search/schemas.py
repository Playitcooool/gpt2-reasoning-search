"""Public request, response, search, and tool schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    snippet: str
    content: str
    score: float = 0.0
    provider: str = "unknown"
    published_at: str | None = None
    score_components: dict[str, float] = Field(default_factory=dict)


class SearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return " ".join(value.split())


class ToolCall(BaseModel):
    name: Literal["search"]
    arguments: SearchArguments


class Citation(BaseModel):
    source_id: str
    title: str
    url: str
    snippet: str = ""
    provider: str = "unknown"


class ToolTraceEntry(BaseModel):
    call: ToolCall | None = None
    results: list[SearchResult] = Field(default_factory=list)
    provider: str | None = None
    status: Literal["ok", "empty", "invalid", "rejected", "error"] = "ok"
    error: str | None = None
    elapsed_seconds: float = 0.0


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8_000)
    search_mode: Literal["auto", "local", "web", "off"] = "auto"
    max_searches: int = Field(default=3, ge=0, le=3)


class AnswerResponse(BaseModel):
    answer: str
    reasoning: str
    citations: list[Citation]
    tool_trace: list[ToolTraceEntry]
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    finish_reason: Literal["answer", "search_limit", "invalid_tool", "error"] = "answer"
    searches_used: int = 0
    reasoning_notice: str = (
        "Generated scratch work for research inspection; "
        "not a faithful account of internal computation."
    )
