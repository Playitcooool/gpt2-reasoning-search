import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gpt2_reasoning_search.schemas import SearchResult
from gpt2_reasoning_search.sft import (
    _shuffle_buffer,
    _trajectory_texts,
    collate_sft_batch,
    encode_sft_document,
    fine_tune_tools,
)
from gpt2_reasoning_search.tokenizer import train_tokenizer
from gpt2_reasoning_search.train import warmup_cosine_factor
from gpt2_reasoning_search.trajectories import (
    generate_trajectories,
    multi_step_trajectory_document,
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

    assert document == ("<|problem|>\nWhat is 2 + 2?\n<|reasoning|>Add two and two.<|answer|>4")
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


def test_multi_step_reformulated_trajectory_and_three_search_cap() -> None:
    first = _evidence()
    second = first.model_copy(update={"id": "moon:0", "title": "Moon"})
    document = multi_step_trajectory_document(
        "Question?",
        "Answer.",
        [("broad query", [first]), ("reformulated query", [second])],
        "Combine both sources.",
    )

    assert document.count("<|tool_call|>") == 2
    assert "broad query" in document and "reformulated query" in document
    assert document.endswith("<|citation|>earth:0 <|citation|>moon:0")
    with pytest.raises(ValueError, match="at most three"):
        multi_step_trajectory_document(
            "Question?", "Answer.", [(str(i), []) for i in range(4)], "Reasoning."
        )


def test_sft_collation_pads_inputs_and_masks_padded_labels() -> None:
    inputs, labels = collate_sft_batch([([1, 2, 3], [-100, 2, 3]), ([4], [4])], pad_token_id=0)

    assert inputs.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert labels.tolist() == [[-100, 2, 3], [4, -100, -100]]
    with pytest.raises(ValueError, match="empty"):
        collate_sft_batch([], 0)


def test_sft_shuffle_is_deterministic_and_preserves_all_values() -> None:
    values = [str(index) for index in range(20)]
    first = list(_shuffle_buffer(values, 4, random.Random(17)))
    second = list(_shuffle_buffer(values, 4, random.Random(17)))

    assert first == second
    assert sorted(first, key=int) == values
    assert first != values


def test_sft_learning_rate_warmup_cosine_decay() -> None:
    assert warmup_cosine_factor(0, total_steps=100, warmup_steps=10) == pytest.approx(0.1)
    assert warmup_cosine_factor(9, total_steps=100, warmup_steps=10) == pytest.approx(1.0)
    assert warmup_cosine_factor(10, total_steps=100, warmup_steps=10) == pytest.approx(1.0)
    assert warmup_cosine_factor(55, total_steps=100, warmup_steps=10) == pytest.approx(0.55)
    assert warmup_cosine_factor(100, total_steps=100, warmup_steps=10) == pytest.approx(0.1)


def test_sft_trajectory_validation_reports_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"ok"}\n{}\n')

    iterator = _trajectory_texts(path)
    assert next(iterator) == "ok"
    with pytest.raises(ValueError, match="line 2"):
        next(iterator)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"micro_batch_size": 0},
        {"gradient_accumulation_steps": 0},
        {"shuffle_buffer_size": 0},
        {"warmup_fraction": 1.0},
        {"time_budget_hours": 0.0},
        {"time_budget_hours": float("nan")},
        {"time_budget_hours": float("inf")},
    ],
)
def test_sft_rejects_invalid_training_configuration(
    tmp_path: Path, monkeypatch, kwargs: dict[str, int | float]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError):
        fine_tune_tools(
            tmp_path,
            tmp_path / "tokenizer.json",
            tmp_path / "trajectories.jsonl",
            tmp_path / "output",
            **kwargs,
        )


def test_sft_resume_cursor_and_partial_accumulation_scaling_with_cpu_mocks(
    tmp_path: Path, monkeypatch
) -> None:
    import gpt2_reasoning_search.sft as sft_module

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model_config = {
        "vocab_size": 16,
        "max_seq_len": 8,
        "n_layers": 1,
        "d_model": 8,
        "n_heads": 2,
        "n_kv_heads": 1,
        "intermediate_size": 16,
    }
    (checkpoint / "state.json").write_text(
        __import__("json").dumps({"config": {"model": model_config}})
    )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text("".join(f'{{"text":"example-{index}"}}\n' for index in range(3)))

    class FakeModel(torch.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, *args, **kwargs):
            return self

        def forward(self, _inputs, _labels):
            return SimpleNamespace(loss=self.weight)

    class FakeTokenizer:
        def token_to_id(self, token: str) -> int | None:
            return 0 if token == "<|pad|>" else None

        def encode(self, text: str):
            return SimpleNamespace(ids=[1, 2, 3], offsets=[(0, 1), (1, 2), (2, 3)])

    class DeviceValue:
        def to(self, _device):
            return self

    saved: list[tuple[int, dict[str, int]]] = []
    clipped_gradients: list[float] = []
    original_sgd = torch.optim.SGD
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft_module, "GPT2ReasoningModel", FakeModel)
    monkeypatch.setattr(sft_module, "load_model_weights", lambda *_args: None)
    monkeypatch.setattr(
        sft_module,
        "load_checkpoint",
        lambda *_args: {
            "step": 5,
            "mixture_state": {"epoch": 1, "examples_in_epoch": 2},
        },
    )
    monkeypatch.setattr(sft_module.Tokenizer, "from_file", lambda _path: FakeTokenizer())
    monkeypatch.setattr(
        sft_module,
        "collate_sft_batch",
        lambda _batch, _pad: (DeviceValue(), DeviceValue()),
    )
    monkeypatch.setattr(
        sft_module.torch.optim,
        "AdamW",
        lambda params, lr, **_kwargs: original_sgd(params, lr=lr),
    )
    monkeypatch.setattr(
        sft_module.torch.nn.utils,
        "clip_grad_norm_",
        lambda parameters, _limit: (
            clipped_gradients.append(next(iter(parameters)).grad.item())
            or torch.tensor(abs(clipped_gradients[-1]))
        ),
    )
    monkeypatch.setattr(
        sft_module,
        "save_checkpoint",
        lambda _directory, _model, _optimizer, _scheduler, step, _tokens, mixture, _config: (
            saved.append((step, mixture.copy()))
        ),
    )

    result = fine_tune_tools(
        checkpoint,
        tmp_path / "tokenizer.json",
        trajectories,
        tmp_path / "output",
        epochs=3,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        resume_from=tmp_path / "resume",
    )

    assert result == tmp_path / "output"
    # Epoch 1 resumes after two examples; epoch 2 processes all three. Each
    # partial update is corrected back to the same effective unit gradient.
    assert clipped_gradients == pytest.approx([1.0, 1.0])
    assert saved[-1] == (7, {"epoch": 3, "examples_in_epoch": 0})

    saved.clear()
    clock_calls = 0

    def expired_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 3_601.0

    monkeypatch.setattr(sft_module.time, "perf_counter", expired_clock)
    timed_out = fine_tune_tools(
        checkpoint,
        tmp_path / "tokenizer.json",
        trajectories,
        tmp_path / "timed-out",
        epochs=3,
        time_budget_hours=1.0,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        resume_from=tmp_path / "resume",
    )

    assert timed_out == tmp_path / "timed-out" / "step-00000005"
    assert saved[-1][0] == 5
    assert saved[-1][1] == {"epoch": 1, "examples_in_epoch": 2}

    final_trajectory = tmp_path / "final-trajectory.jsonl"
    final_trajectory.write_text('{"text":"last example"}\n')
    saved.clear()
    clock = iter([0.0, 0.0, 3_601.0])
    monkeypatch.setattr(sft_module.time, "perf_counter", lambda: next(clock))
    completed_at_deadline = fine_tune_tools(
        checkpoint,
        tmp_path / "tokenizer.json",
        final_trajectory,
        tmp_path / "completed-at-deadline",
        epochs=1,
        time_budget_hours=1.0,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
    )

    assert completed_at_deadline == tmp_path / "completed-at-deadline"
    assert saved[-1] == (1, {"epoch": 1, "examples_in_epoch": 0})
