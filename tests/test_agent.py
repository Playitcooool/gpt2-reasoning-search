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
        self.results = results if results is not None else [_result()]
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results


def test_parse_tool_call_rejects_multiple_calls_in_one_generation() -> None:
    text = (
        '<|tool_call|>{"name":"search","arguments":{"query":"first"}}'
        "<|end_tool_call|>"
        '<|tool_call|>{"name":"search","arguments":{"query":"second"}}'
        "<|end_tool_call|>"
    )

    assert parse_tool_call(text) is None


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


def test_duplicate_query_is_rejected_and_search_limit_forces_final_answer() -> None:
    generations = iter(
        [
            '<|tool_call|>{"name":"search","arguments":{"query":"same"}}'
            "<|end_tool_call|>",
            '<|tool_call|>{"name":"search","arguments":{"query":" SAME "}}'
            "<|end_tool_call|>",
            '<|tool_call|>{"name":"search","arguments":{"query":"over budget"}}'
            "<|end_tool_call|>",
            "<|reasoning|>Use one result.<|answer|>Done <|citation|>doc:0",
        ]
    )
    provider = RecordingProvider()
    agent = SearchAgent(lambda _prompt, _limit: (next(generations), 1, 1), provider)

    response = asyncio.run(
        agent.answer(AnswerRequest(query="question", search_mode="local", max_searches=2))
    )

    assert provider.calls == [("same", 5)]
    assert [entry.status for entry in response.tool_trace] == ["ok", "rejected"]
    assert response.searches_used == 2
    assert response.finish_reason == "search_limit"
    assert response.answer == "Done [doc:0]"


def test_local_web_and_auto_provider_selection_and_fallback() -> None:
    async def exercise(mode: str, local_results: list[SearchResult]):
        generations = iter(
            [
                '<|tool_call|>{"name":"search","arguments":{"query":"fact"}}'
                "<|end_tool_call|>",
                "<|reasoning|>Evidence.<|answer|>Answer.",
            ]
        )
        local = RecordingProvider(local_results)
        web = RecordingProvider([_result("web:0")])
        agent = SearchAgent(lambda _prompt, _limit: (next(generations), 1, 1), local, web)
        response = await agent.answer(AnswerRequest(query="question", search_mode=mode))
        return local, web, response

    local, web, response = asyncio.run(exercise("local", [_result("local:0")]))
    assert local.calls == [("fact", 5)] and web.calls == []
    assert response.tool_trace[0].provider == "local"

    local, web, response = asyncio.run(exercise("web", [_result("local:0")]))
    assert local.calls == [] and web.calls == [("fact", 5)]
    assert response.tool_trace[0].provider == "web"

    local, web, response = asyncio.run(exercise("auto", []))
    assert local.calls == [("fact", 5)] and web.calls == [("fact", 5)]
    assert response.tool_trace[0].provider == "web"


def test_provider_timeout_becomes_trace_error_and_generation_continues() -> None:
    class SlowProvider(RecordingProvider):
        async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
            await asyncio.sleep(0.05)
            return await super().search(query, top_k)

    generations = iter(
        [
            '<|tool_call|>{"name":"search","arguments":{"query":"slow"}}'
            "<|end_tool_call|>",
            "<|reasoning|>Search failed.<|answer|>Unable to verify.",
        ]
    )
    agent = SearchAgent(
        lambda _prompt, _limit: (next(generations), 1, 1),
        SlowProvider(),
        search_timeout_seconds=0.001,
    )

    response = asyncio.run(agent.answer(AnswerRequest(query="question", search_mode="local")))

    assert response.answer == "Unable to verify."
    assert response.tool_trace[0].status == "error"
    assert "timed out" in (response.tool_trace[0].error or "").lower()


def test_generation_timeout_is_bounded() -> None:
    def slow_generate(_prompt: str, _limit: int) -> tuple[str, int, int]:
        import time

        time.sleep(0.05)
        return "<|answer|>late", 1, 1

    agent = SearchAgent(
        slow_generate,
        RecordingProvider(),
        generation_timeout_seconds=0.001,
    )

    with pytest.raises(TimeoutError):
        asyncio.run(agent.answer(AnswerRequest(query="question", search_mode="off")))


def test_citations_copy_only_returned_result_metadata() -> None:
    result = _result("doc:metadata").model_copy(
        update={"title": "Verified title", "snippet": "Verified snippet", "provider": "bm25"}
    )
    _, answer, citations = parse_final_response(
        "<|answer|>Claim <|citation|>doc:metadata <|citation|>not-returned",
        {result.id: result},
    )

    assert answer == "Claim [doc:metadata]"
    assert [citation.model_dump() for citation in citations] == [
        {
            "source_id": "doc:metadata",
            "title": "Verified title",
            "url": "https://example.test/doc:metadata",
            "snippet": "Verified snippet",
            "provider": "bm25",
        }
    ]


def test_agent_closes_sync_and_async_provider_resources_once() -> None:
    class AsyncClosable(RecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        async def aclose(self) -> None:
            self.closed += 1

    class SyncClosable(RecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    local = AsyncClosable()
    web = SyncClosable()
    agent = SearchAgent(lambda _prompt, _limit: ("", 0, 0), local, web)

    asyncio.run(agent.aclose())

    assert local.closed == 1 and web.closed == 1
