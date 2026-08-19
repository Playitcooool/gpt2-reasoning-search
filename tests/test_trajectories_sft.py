from pathlib import Path

from gpt2_reasoning_search.schemas import SearchResult
from gpt2_reasoning_search.sft import encode_sft_document
from gpt2_reasoning_search.tokenizer import train_tokenizer
from gpt2_reasoning_search.trajectories import (
    generate_trajectories,
    stream_jsonl,
    trajectory_document,
)


def _evidence() -> SearchResult:
    return SearchResult(
        id="earth:0",
        title="Earth",
        url="https://example.test/earth",
        snippet="Earth has one moon.",
        content="Earth has one natural satellite, the Moon.",
        score=1.0,
    )


def test_trajectory_document_with_search_contains_call_evidence_and_citation() -> None:
    document = trajectory_document(
        question="How many moons does Earth have?",
        answer="One.",
        query="Earth natural satellites",
        evidence=[_evidence()],
        reasoning="The evidence states the count.",
    )

    assert document.startswith("<|problem|>\nHow many moons")
    assert '<|tool_call|>{"name":"search"' in document
    assert '"top_k":1' in document
    assert "UNTRUSTED SEARCH EVIDENCE" in document
    assert "<|reasoning|>The evidence states the count." in document
    assert document.endswith("One. <|citation|>earth:0")


def test_no_search_trajectory_omits_all_tool_and_citation_markup() -> None:
    document = trajectory_document(
        question="What is 2 + 2?",
        answer="4",
        query=None,
        evidence=[],
        reasoning="Add two and two.",
    )

    assert document == (
        "<|problem|>\nWhat is 2 + 2?\n<|reasoning|>Add two and two.<|answer|>4"
    )
    assert "<|tool_" not in document and "<|citation|>" not in document


def test_generate_and_stream_trajectories_round_trip_jsonl(tmp_path: Path) -> None:
    rows = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "query": None,
            "reasoning": "Addition.",
        },
        {
            "question": "How many moons?",
            "answer": "One.",
            "query": "Earth moon count",
            "evidence": [_evidence().model_dump()],
        },
    ]
    output = tmp_path / "trajectories.jsonl"

    assert generate_trajectories(rows, output) == 2
    loaded = list(stream_jsonl(output))

    assert len(loaded) == 2
    assert all(set(row) == {"text"} for row in loaded)
    assert "<|tool_call|>" not in loaded[0]["text"]
    assert "<|tool_call|>" in loaded[1]["text"]


def test_sft_masks_prompt_and_tool_observation_but_trains_actions(tmp_path: Path) -> None:
    document = trajectory_document(
        question="How many moons does Earth have?",
        answer="One.",
        query="Earth moon count",
        evidence=[_evidence()],
        reasoning="Use the evidence.",
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(document)
    tokenizer = train_tokenizer(
        [corpus], tmp_path / "tokenizer.json", vocab_size=512, min_frequency=1
    )

    ids, labels = encode_sft_document(tokenizer, document, max_length=2_048)
    encoded = tokenizer.encode(document)
    prompt_end = document.index("<|tool_call|>")
    result_start = document.index("<|tool_result|>")
    result_end = document.index("<|end_tool_result|>") + len("<|end_tool_result|>")
    prompt_tokens = [
        index
        for index, (start, end) in enumerate(encoded.offsets)
        if start < prompt_end and end > 0
    ]
    result_tokens = [
        index
        for index, (start, end) in enumerate(encoded.offsets)
        if start < result_end and end > result_start
    ]
    action_tokens = [
        index
        for index, (start, end) in enumerate(encoded.offsets)
        if start >= prompt_end and end <= result_start and end > start
    ]
    final_tokens = [
        index
        for index, (start, end) in enumerate(encoded.offsets)
        if start >= result_end and end > start
    ]

    assert ids == encoded.ids
    assert prompt_tokens and all(labels[index] == -100 for index in prompt_tokens)
    assert result_tokens and all(labels[index] == -100 for index in result_tokens)
    assert action_tokens and all(labels[index] != -100 for index in action_tokens)
    assert final_tokens and all(labels[index] != -100 for index in final_tokens)


def test_sft_no_search_masks_problem_before_reasoning(tmp_path: Path) -> None:
    document = trajectory_document("Question?", "Answer.", None, [], "Reasoning.")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(document)
    tokenizer = train_tokenizer(
        [corpus], tmp_path / "tokenizer.json", vocab_size=384, min_frequency=1
    )
    _, labels = encode_sft_document(tokenizer, document, max_length=512)
    encoded = tokenizer.encode(document)
    reasoning_start = document.index("<|reasoning|>")

    for index, (start, end) in enumerate(encoded.offsets):
        if start < reasoning_start and end > 0:
            assert labels[index] == -100
        elif start >= reasoning_start and end > start:
            assert labels[index] != -100
