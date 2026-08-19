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


class ToolTraceEntry(BaseModel):
    call: ToolCall
    results: list[SearchResult]


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
    reasoning_notice: str = (
        "Generated scratch work for research inspection; "
        "not a faithful account of internal computation."
    )
