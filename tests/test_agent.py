import asyncio

import pytest
from pydantic import ValidationError

from gpt2_reasoning_search.agent import (
    SearchAgent,
    format_tool_results,
    parse_final_response,
    parse_tool_call,
)
from gpt2_reasoning_search.schemas import AnswerRequest, SearchArguments, SearchResult


def _result(source_id: str = "doc:0") -> SearchResult:
    return SearchResult(
        id=source_id,
        title="A source",
        url=f"https://example.test/{source_id}",
        snippet="support",
        content="supporting evidence",
        score=2.0,
    )


class RecordingProvider:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or [_result()]
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results


def test_parse_tool_call_accepts_valid_and_rejects_invalid_calls() -> None:
    valid = parse_tool_call(
        '<|tool_call|>{"name":"search","arguments":{"query":"  red   planet ",'
        '"top_k":3}}<|end_tool_call|>'
    )

    assert valid is not None
    assert valid.arguments.query == "red planet"
    assert valid.arguments.top_k == 3
    assert parse_tool_call("ordinary answer") is None
    assert parse_tool_call("<|tool_call|>{bad json}<|end_tool_call|>") is None
    assert (
        parse_tool_call(
            '<|tool_call|>{"name":"delete","arguments":{"query":"x"}}'
            "<|end_tool_call|>"
        )
        is None
    )
    assert (
        parse_tool_call(
            '<|tool_call|>{"name":"search","arguments":{"query":"x","top_k":11}}'
            "<|end_tool_call|>"
        )
        is None
    )


def test_search_arguments_reject_whitespace_only_query() -> None:
    with pytest.raises(ValidationError):
        SearchArguments(query="   ")


def test_format_tool_results_marks_evidence_as_untrusted() -> None:
    formatted = format_tool_results([_result()])

    assert formatted.startswith("<|tool_result|>")
    assert "UNTRUSTED SEARCH EVIDENCE" in formatted
    assert '"id": "doc:0"' in formatted
    assert formatted.endswith("<|end_tool_result|>")


def test_final_response_filters_unknown_and_duplicate_citations() -> None:
    known = _result()
    reasoning, answer, citations = parse_final_response(
        "<|reasoning|>Check evidence."
        "<|answer|>Supported <|citation|>doc:0 duplicate <|citation|>doc:0 "
        "invented <|citation|>fake:9",
        {known.id: known},
    )

    assert reasoning == "Check evidence."
    assert answer == "Supported [doc:0] duplicate [doc:0] invented"
    assert [citation.source_id for citation in citations] == ["doc:0"]
    assert "fake:9" not in answer


def test_agent_enforces_three_search_bound_and_returns_only_known_citations() -> None:
    generations = iter(
        [
            '<|tool_call|>{"name":"search","arguments":{"query":"first"}}'
            "<|end_tool_call|>",
            '<|tool_call|>{"name":"search","arguments":{"query":"second"}}'
            "<|end_tool_call|>",
            '<|tool_call|>{"name":"search","arguments":{"query":"third"}}'
            "<|end_tool_call|>",
            "<|reasoning|>Combined evidence.<|answer|>Final [text] "
            "<|citation|>doc:0 <|citation|>invented",
        ]
    )

    def generate(prompt: str, max_new_tokens: int) -> tuple[str, int, int]:
        assert max_new_tokens == 512
        return next(generations), len(prompt), 4

    provider = RecordingProvider()
    agent = SearchAgent(generate, provider)
    response = asyncio.run(
        agent.answer(AnswerRequest(query="question", search_mode="local", max_searches=3))
    )

    assert provider.calls == [("first", 5), ("second", 5), ("third", 5)]
    assert len(response.tool_trace) == 3
    assert response.reasoning == "Combined evidence."
    assert response.answer == "Final [text] [doc:0]"
    assert [citation.source_id for citation in response.citations] == ["doc:0"]
    assert response.output_tokens == 16


def test_agent_off_mode_never_executes_generated_call() -> None:
    provider = RecordingProvider()
    agent = SearchAgent(
        lambda _prompt, _limit: (
            '<|tool_call|>{"name":"search","arguments":{"query":"do not run"}}'
            "<|end_tool_call|>",
            5,
            3,
        ),
        provider,
    )

    response = asyncio.run(
        agent.answer(AnswerRequest(query="known fact", search_mode="off", max_searches=3))
    )

    assert provider.calls == []
    assert response.tool_trace == []


def test_agent_web_mode_requires_configured_provider() -> None:
    agent = SearchAgent(
        lambda _prompt, _limit: (
            '<|tool_call|>{"name":"search","arguments":{"query":"current"}}'
            "<|end_tool_call|>",
            1,
            1,
        ),
        RecordingProvider(),
    )

    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(agent.answer(AnswerRequest(query="question", search_mode="web")))

