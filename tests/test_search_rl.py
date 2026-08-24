from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gpt2_reasoning_search.judge import JudgeScore
from gpt2_reasoning_search.rl import (
    PolicyGenerator,
    RewardWeights,
    RolloutSegment,
    SearchRLConfig,
    group_advantages,
    score_search_reward,
    segment_policy_loss,
    stream_rl_prompts,
    train_search_rl,
)
from gpt2_reasoning_search.schemas import (
    AnswerResponse,
    SearchArguments,
    SearchResult,
    ToolCall,
    ToolTraceEntry,
)


def config(tmp_path: Path, **overrides: object) -> SearchRLConfig:
    values: dict[str, object] = {
        "checkpoint_directory": tmp_path / "checkpoint",
        "tokenizer_path": tmp_path / "tokenizer.json",
        "prompts_path": tmp_path / "prompts.jsonl",
        "index_directory": tmp_path / "index",
        "output_directory": tmp_path / "output",
    }
    values.update(overrides)
    return SearchRLConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"group_size": 1},
        {"max_searches": -1},
        {"max_searches": 4},
        {"search_mode": "off"},
        {"learning_rate": 0},
        {"kl_coefficient": -0.1},
        {"warmup_fraction": 1.0},
        {"temperature": 0},
        {"top_k": -1},
        {"max_new_tokens": 0},
        {"grad_clip": 0},
        {"save_every_steps": 0},
        {"judge_device": "mps"},
        {"judge_max_input_tokens": 255},
        {"judge_max_new_tokens": 15},
        {"judge_model": "Qwen/Qwen3.5-2B", "judge_revision": None},
        {"judge_model": "Qwen/Qwen3.5-2B", "judge_revision": "main"},
        {"time_budget_hours": 0.0},
        {"time_budget_hours": float("nan")},
        {"time_budget_hours": float("inf")},
    ],
)
def test_search_rl_config_validation(tmp_path: Path, overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        config(tmp_path, **overrides)


def test_web_rl_requires_api_key_before_model_setup(tmp_path: Path, monkeypatch) -> None:
    import gpt2_reasoning_search.rl as rl_module

    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def should_not_load_model(*_args, **_kwargs):
        raise AssertionError("model setup should not run before web-key validation")

    monkeypatch.setattr(rl_module, "GPT2ReasoningModel", should_not_load_model)

    with pytest.raises(ValueError, match="BRAVE_SEARCH_API_KEY"):
        train_search_rl(config(tmp_path, search_mode="web"))


def test_stream_rl_prompts_accepts_schema_and_skips_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '\n{"question":"Where?","answer":"There","supporting_ids":["doc:1"],'
        '"supporting_sources":[{"id":"doc:1","url":"https://example.test/doc"}],'
        '"search_required":true}\n'
    )

    assert list(stream_rl_prompts(path)) == [
        {
            "question": "Where?",
            "answer": "There",
            "supporting_ids": ["doc:1"],
            "supporting_sources": [
                {"id": "doc:1", "url": "https://example.test/doc"}
            ],
            "search_required": True,
        }
    ]


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"answer":"x"}\n', "missing question"),
        ('{"question":"q"}\n', "missing answer"),
        ('{"question":"q","answer":"a","supporting_ids":"bad"}\n', "supporting_ids"),
        (
            '{"question":"q","answer":"a","supporting_sources":[{"url":4}]}\n',
            "supporting_sources",
        ),
        ('{"question":"q","answer":"a","search_required":"false"}\n', "search_required"),
    ],
)
def test_stream_rl_prompts_rejects_invalid_schema(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(line)

    with pytest.raises(ValueError, match=message):
        list(stream_rl_prompts(path))


def test_stream_rl_prompts_surfaces_malformed_json_and_empty_dataset(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{bad}\n")
    with pytest.raises(json.JSONDecodeError):
        list(stream_rl_prompts(malformed))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    assert list(stream_rl_prompts(empty)) == []


def result(source_id: str = "doc:1") -> SearchResult:
    return SearchResult(
        id=source_id,
        title="Source",
        url=f"https://example.test/{source_id}",
        snippet="support",
        content="support",
        provider="local",
    )


def result_with_url(source_id: str, url: str, provider: str = "web") -> SearchResult:
    return result(source_id).model_copy(update={"url": url, "provider": provider})


def trace(
    status: str = "ok",
    results: list[SearchResult] | None = None,
    query: str = "query",
    provider: str | None = None,
) -> ToolTraceEntry:
    return ToolTraceEntry(
        call=ToolCall(name="search", arguments=SearchArguments(query=query)),
        status=status,  # type: ignore[arg-type]
        results=results or [],
        provider=provider,
    )


def response(
    answer: str,
    traces: list[ToolTraceEntry],
    searches_used: int,
) -> AnswerResponse:
    return AnswerResponse(
        answer=answer,
        reasoning="reasoning",
        citations=[],
        tool_trace=traces,
        input_tokens=1,
        output_tokens=1,
        elapsed_seconds=0,
        searches_used=searches_used,
    )


def test_search_reward_exact_f1_supported_valid_citation_and_search_cost() -> None:
    scored = score_search_reward(
        {"answer": "Paris", "supporting_ids": ["doc:1"], "search_required": True},
        response("Paris [doc:1]", [trace(results=[result()])], 1),
        "<|answer|>Paris <|citation|>doc:1",
    )

    assert scored.components == {
        "answer_exact": 1.0,
        "answer_f1": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "citation_validity": 1.0,
        "valid_tool_calls": 1.0,
        "query_recovery": 0.0,
        "grounded_search": 1.0,
        "missing_required_grounding": 0.0,
        "unnecessary_search": 0.0,
        "invalid_tool_call": 0.0,
        "search_cost": 0.0,
        "judge_answer_correctness": 0.0,
        "judge_evidence_support": 0.0,
        "judge_search_quality": 0.0,
        "judge_valid": 0.0,
    }
    assert scored.total == pytest.approx(2.30)


def test_required_answer_and_judge_credit_requires_a_supported_returned_citation() -> None:
    row = {
        "answer": "Paris",
        "supporting_ids": ["wiki:1"],
        "search_required": True,
        "supporting_sources": [
            {"id": "wiki:1", "url": "https://example.test/paris"}
        ],
    }
    scored = score_search_reward(
        row,
        response("Paris", [trace(results=[result("unrelated")])], 1),
        "<|answer|>Paris",
        judge_score=JudgeScore(1.0, 1.0, 1.0, valid=True),
    )

    assert scored.components["answer_exact"] == 0.0
    assert scored.components["answer_f1"] == 0.0
    assert scored.components["judge_answer_correctness"] == 0.0
    assert scored.components["judge_evidence_support"] == 0.0
    assert scored.components["judge_search_quality"] == 0.0


def test_required_grounding_accepts_supported_brave_url_when_source_ids_differ() -> None:
    url = "https://en.wikipedia.org/wiki/Paris"
    row = {
        "answer": "Paris",
        "supporting_ids": ["wiki:1"],
        "search_required": True,
        "evidence": [{"id": "wiki:1", "url": url}],
    }
    brave_result = result_with_url("brave:abc", url)
    scored = score_search_reward(
        row,
        response("Paris [brave:abc]", [trace(results=[brave_result], provider="web")], 1),
        "<|answer|>Paris <|citation|>brave:abc",
        judge_score=JudgeScore(1.0, 0.75, 1.0, valid=True),
    )

    assert scored.components["answer_exact"] == 1.0
    assert scored.components["answer_f1"] == 1.0
    assert scored.components["citation_precision"] == 1.0
    assert scored.components["citation_recall"] == 1.0
    assert scored.components["citation_validity"] == 1.0
    assert scored.components["judge_answer_correctness"] == 1.0
    assert scored.components["judge_evidence_support"] == 0.75


def test_legacy_supporting_ids_and_evidence_match_brave_by_canonical_url() -> None:
    scored = score_search_reward(
        {
            "answer": "Paris",
            "supporting_ids": ["wiki:1"],
            "search_required": True,
            "evidence": [
                {
                    "id": "wiki:1",
                    "url": "https://en.wikipedia.org/wiki/Paris?utm_source=training",
                }
            ],
        },
        response(
            "Paris [brave:abc]",
            [
                trace(
                    results=[
                        result_with_url("brave:abc", "https://en.wikipedia.org/wiki/Paris")
                    ],
                    provider="web",
                )
            ],
            1,
        ),
        "<|answer|>Paris <|citation|>brave:abc",
    )

    assert scored.components["grounded_search"] == 1.0
    assert scored.components["missing_required_grounding"] == 0.0
    assert scored.components["citation_recall"] == 1.0
    assert scored.components["answer_exact"] == 1.0


def test_required_search_has_no_first_search_cost_but_penalizes_extra_searches() -> None:
    row = {
        "answer": "Paris",
        "supporting_ids": ["doc:1"],
        "search_required": True,
    }
    first = score_search_reward(
        row,
        response("Paris [doc:1]", [trace(results=[result()])], 1),
        "<|answer|>Paris <|citation|>doc:1",
    )
    extra = score_search_reward(
        row,
        response(
            "Paris [doc:1]",
            [trace(results=[result()]), trace(results=[result("doc:2")], query="extra")],
            2,
        ),
        "<|answer|>Paris <|citation|>doc:1",
    )

    assert first.components["search_cost"] == 0.0
    assert extra.components["search_cost"] == 1.0


def test_optional_question_keeps_no_search_answer_and_judge_path() -> None:
    scored = score_search_reward(
        {"answer": "Paris", "supporting_ids": [], "search_required": False},
        response("Paris", [], 0),
        "<|answer|>Paris",
        judge_score=JudgeScore(0.5, 0.25, 1.0, valid=True),
    )

    assert scored.components["answer_exact"] == 1.0
    assert scored.components["answer_f1"] == 1.0
    assert scored.components["judge_answer_correctness"] == 0.5
    assert scored.components["judge_evidence_support"] == 0.25
    assert scored.components["search_cost"] == 0.0
    assert scored.components["unnecessary_search"] == 0.0


def test_search_reward_adds_only_valid_auxiliary_judge_components() -> None:
    row = {"answer": "Paris", "supporting_ids": [], "search_required": False}
    base_response = response("Paris", [], 0)
    valid = score_search_reward(
        row,
        base_response,
        "<|answer|>Paris",
        judge_score=JudgeScore(0.5, 0.25, 1.0, valid=True, reason="ok"),
    )
    invalid = score_search_reward(
        row,
        base_response,
        "<|answer|>Paris",
        judge_score=JudgeScore(1.0, 1.0, 1.0, valid=False, reason="bad schema"),
    )

    assert valid.components["judge_answer_correctness"] == 0.5
    assert valid.components["judge_evidence_support"] == 0.25
    assert valid.components["judge_search_quality"] == 1.0
    assert valid.components["judge_valid"] == 1.0
    assert valid.total - invalid.total == pytest.approx(0.5 * 0.20 + 0.25 * 0.15 + 0.05)
    assert invalid.components["judge_answer_correctness"] == 0.0
    assert invalid.components["judge_evidence_support"] == 0.0
    assert invalid.components["judge_search_quality"] == 0.0
    assert invalid.components["judge_valid"] == 0.0


def test_search_reward_separates_support_from_validity_and_penalizes_behavior() -> None:
    weights = RewardWeights()
    scored = score_search_reward(
        {"answer": "Paris France", "supporting_ids": ["fabricated"], "search_required": False},
        response(
            "Paris [fabricated]",
            [trace("rejected"), trace("empty", query="reformulated")],
            2,
        ),
        "<|answer|>Paris <|citation|>fabricated",
        weights,
    )

    assert scored.components["answer_exact"] == 0.0
    assert scored.components["answer_f1"] == pytest.approx(2 / 3)
    assert scored.components["citation_precision"] == 0.0
    assert scored.components["citation_recall"] == 0.0
    assert scored.components["citation_validity"] == 0.0
    assert scored.components["valid_tool_calls"] == 0.5
    assert scored.components["unnecessary_search"] == 1.0
    assert scored.components["invalid_tool_call"] == 1.0
    assert scored.components["search_cost"] == 2.0


def test_query_recovery_requires_reformulation_and_correct_answer() -> None:
    scored = score_search_reward(
        {"answer": "yes", "supporting_ids": ["doc:1"], "search_required": True},
        response(
            "yes",
            [trace("empty", query="first"), trace(results=[result()], query="second")],
            2,
        ),
        "<|answer|>yes <|citation|>doc:1",
    )
    assert scored.components["query_recovery"] == 1.0

    no_failure = score_search_reward(
        {"answer": "yes", "supporting_ids": ["doc:1"], "search_required": True},
        response("yes", [trace(query="first"), trace(query="second")], 2),
        "<|answer|>yes <|citation|>doc:1",
    )
    assert no_failure.components["query_recovery"] == 0.0

    rejected_duplicate = score_search_reward(
        {"answer": "yes", "supporting_ids": []},
        response("yes", [trace("rejected", query="same"), trace(query="same")], 2),
        "<|answer|>yes",
    )
    assert rejected_duplicate.components["query_recovery"] == 0.0


def test_group_advantages_center_scale_constant_and_reject_nonfinite() -> None:
    advantages = group_advantages([1.0, 2.0, 3.0])
    assert sum(advantages) == pytest.approx(0.0, abs=1e-12)
    assert torch.tensor(advantages).std(correction=0).item() == pytest.approx(1.0, rel=2e-4)
    assert group_advantages([4.0, 4.0]) == [0.0, 0.0]
    with pytest.raises(ValueError, match="at least two"):
        group_advantages([1.0])
    for values in ([1.0, float("nan")], [1.0, float("inf")]):
        with pytest.raises(ValueError, match="finite"):
            group_advantages(values)


class FakeTokenizer:
    def encode(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[int(value) for value in prompt.split(",") if value])

    def token_to_id(self, token: str) -> int | None:
        return {"<|eos|>": 98, "<|end_tool_call|>": 99}.get(token)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert not skip_special_tokens
        return ",".join(map(str, ids))


class GeneratingModel:
    def __init__(self, max_seq_len: int = 8) -> None:
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self.calls: list[dict[str, object]] = []

    def generate(self, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.calls.append({"inputs": inputs.clone(), **kwargs})
        new_tokens = int(kwargs["max_new_tokens"])
        suffix = torch.arange(1, new_tokens + 1).unsqueeze(0)
        return torch.cat((inputs, suffix), dim=1)


def test_policy_generator_prompt_budget_sampling_stops_capture_and_reset() -> None:
    model = GeneratingModel(max_seq_len=8)
    generator = PolicyGenerator(
        model,  # type: ignore[arg-type]
        FakeTokenizer(),  # type: ignore[arg-type]
        torch.device("cpu"),
        temperature=0.7,
        top_k=9,
        maximum_new_tokens=3,
    )

    text, input_tokens, output_tokens = generator(",".join(str(value) for value in range(12)), 10)

    call = model.calls[0]
    assert call["inputs"].tolist() == [[0, 8, 9, 10, 11]]
    assert call["temperature"] == 0.7 and call["top_k"] == 9
    assert call["eos_token_id"] == 98 and call["stop_token_ids"] == {99}
    assert (text, input_tokens, output_tokens) == ("1,2,3", 5, 3)
    assert generator.segments == [RolloutSegment((0, 8, 9, 10, 11, 1, 2, 3), 5)]
    assert generator.outputs == ["1,2,3"]
    generator.reset()
    assert generator.segments == [] and generator.outputs == []


def test_policy_generator_never_records_segment_larger_than_training_context() -> None:
    model = GeneratingModel(max_seq_len=8)
    generator = PolicyGenerator(
        model,  # type: ignore[arg-type]
        FakeTokenizer(),  # type: ignore[arg-type]
        torch.device("cpu"),
        temperature=1.0,
        top_k=0,
        maximum_new_tokens=20,
    )

    generator("1,2,3", 20)

    assert len(generator.segments[0].token_ids) <= model.config.max_seq_len


class PositionPolicy(torch.nn.Module):
    def __init__(self, values: tuple[float, float, float]) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values))

    def forward(self, tokens: torch.Tensor) -> SimpleNamespace:
        batch, length = tokens.shape
        logits = torch.zeros(batch, length, 2)
        logits[:, :, 1] = self.values[:length]
        return SimpleNamespace(logits=logits)


def test_segment_policy_loss_masks_prompt_and_advantage_controls_direction() -> None:
    segment = RolloutSegment((0, 0, 0, 1), prompt_length=3)
    reference = PositionPolicy((0.0, 0.0, 0.0, 0.0))
    positive = PositionPolicy((0.0, 0.0, 0.0, 0.0))
    loss, kl = segment_policy_loss(positive, reference, segment, 1.0, 0.0, torch.device("cpu"))
    loss.backward()
    assert positive.values.grad is not None
    assert positive.values.grad[:2].tolist() == [0.0, 0.0]
    assert positive.values.grad[2] < 0
    assert kl == pytest.approx(0.0)

    negative = PositionPolicy((0.0, 0.0, 0.0, 0.0))
    negative_loss, _ = segment_policy_loss(
        negative, reference, segment, -1.0, 0.0, torch.device("cpu")
    )
    negative_loss.backward()
    assert negative.values.grad is not None and negative.values.grad[2] > 0


def test_segment_policy_kl_is_nonnegative_and_zero_for_equal_reference() -> None:
    segment = RolloutSegment((0, 0, 1), prompt_length=2)
    reference = PositionPolicy((0.0, -0.5, 0.0))
    equal = PositionPolicy((0.0, -0.5, 0.0))
    _, equal_kl = segment_policy_loss(equal, reference, segment, 0.0, 0.1, torch.device("cpu"))
    different = PositionPolicy((0.0, 1.5, 0.0))
    loss, different_kl = segment_policy_loss(
        different, reference, segment, 0.0, 0.1, torch.device("cpu")
    )

    assert equal_kl == pytest.approx(0.0, abs=1e-7)
    assert different_kl >= 0
    assert loss >= 0


def test_segment_policy_loss_rejects_segment_without_action() -> None:
    model = PositionPolicy((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="no model action"):
        segment_policy_loss(
            model,
            model,
            RolloutSegment((0, 1), prompt_length=2),
            1.0,
            0.0,
            torch.device("cpu"),
        )


def test_train_search_rl_resume_checkpoints_metrics_reference_and_provider_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    import gpt2_reasoning_search.rl as rl_module

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model_config = {
        "vocab_size": 8,
        "max_seq_len": 8,
        "n_layers": 1,
        "d_model": 8,
        "n_heads": 2,
        "n_kv_heads": 1,
        "intermediate_size": 16,
    }
    (checkpoint / "state.json").write_text(
        json.dumps({"config": {"model": model_config, "source": "base"}})
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"question":"q1","answer":"yes","search_required":false}\n'
        '{"question":"q2","answer":"yes","search_required":false}\n'
    )

    models: list[FakeRLModel] = []

    class FakeRLModel(torch.nn.Module):
        def __init__(self, model_configuration) -> None:
            super().__init__()
            self.config = model_configuration
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            models.append(self)

        def to(self, *args, **kwargs):
            return self

    class FakePolicyGenerator:
        def __init__(self, model, tokenizer, device, **kwargs) -> None:
            self.model = model
            self.segments: list[RolloutSegment] = []
            self.outputs: list[str] = []

        def reset(self) -> None:
            self.segments.clear()
            self.outputs.clear()

        def __call__(self, _prompt: str, _limit: int) -> tuple[str, int, int]:
            self.segments.append(RolloutSegment((0, 1), 1))
            self.outputs.append("<|answer|>yes")
            return "<|answer|>yes", 1, 1

    providers: list[FakeProvider] = []

    class FakeProvider:
        def __init__(self, index_directory, **kwargs) -> None:
            self.index_directory = index_directory
            self.kwargs = kwargs
            self.closed = 0
            providers.append(self)

    caches = []

    class FakeCache:
        def __init__(self, path, ttl_seconds) -> None:
            self.path = path
            self.ttl_seconds = ttl_seconds
            self.closed = 0
            caches.append(self)

        def close(self) -> None:
            self.closed += 1

    web_providers = []

    class FakeWebProvider:
        def __init__(self, api_key, *, cache) -> None:
            self.api_key = api_key
            self.cache = cache
            self.closed = 0
            web_providers.append(self)

        async def aclose(self) -> None:
            self.closed += 1
            self.cache.close()

    class FakeAgent:
        calls = 0
        requests = []
        trace_providers: tuple[str, ...] = ()

        def __init__(self, generate, provider, web_provider=None) -> None:
            self.generate = generate
            self.provider = provider
            self.web_provider = web_provider

        async def answer(self, request) -> AnswerResponse:
            self.generate("prompt", 512)
            FakeAgent.calls += 1
            FakeAgent.requests.append(request)
            answer = "yes" if FakeAgent.calls % 2 else "no"
            traces = []
            if FakeAgent.trace_providers:
                provider_index = (FakeAgent.calls - 1) % len(FakeAgent.trace_providers)
                traces = [trace(provider=FakeAgent.trace_providers[provider_index])]
            return response(answer, traces, 0)

        async def aclose(self) -> None:
            self.provider.closed += 1
            if self.web_provider is not None:
                await self.web_provider.aclose()

    judges = []

    class FakeJudge:
        raise_score = False
        raise_init = False

        def __init__(self, model_name, revision, **kwargs) -> None:
            if self.raise_init:
                raise RuntimeError("judge load failed")
            self.model_name = model_name
            self.revision = revision
            self.kwargs = kwargs
            self.calls = []
            self.closed = 0
            judges.append(self)

        def score(self, question, reference_answer, judged_response) -> JudgeScore:
            self.calls.append((question, reference_answer, judged_response.answer))
            if self.raise_score:
                raise RuntimeError("judge failed")
            return JudgeScore(0.5, 0.25, 1.0, valid=True, reason="mock")

        def close(self) -> None:
            self.closed += 1

    loaded_weights: list[tuple[Path, FakeRLModel]] = []
    loaded_resume: list[FakeRLModel] = []
    saves: list[tuple[Path, int, int, dict[str, int], dict[str, object]]] = []
    original_sgd = torch.optim.SGD
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rl_module, "GPT2ReasoningModel", FakeRLModel)
    monkeypatch.setattr(rl_module, "PolicyGenerator", FakePolicyGenerator)
    monkeypatch.setattr(rl_module, "LocalWikipediaSearchProvider", FakeProvider)
    monkeypatch.setattr(rl_module, "BraveWebSearchProvider", FakeWebProvider)
    monkeypatch.setattr(rl_module, "SQLiteSearchCache", FakeCache)
    monkeypatch.setattr(rl_module, "SearchAgent", FakeAgent)
    monkeypatch.setattr(rl_module, "QwenRewardJudge", FakeJudge)
    monkeypatch.setattr(rl_module.Tokenizer, "from_file", lambda _path: object())
    monkeypatch.setattr(
        rl_module,
        "load_model_weights",
        lambda path, model, _device: loaded_weights.append((path, model)),
    )

    def fake_load_checkpoint(_path, model, _optimizer, _scheduler, _device):
        loaded_resume.append(model)
        return {
            "step": 5,
            "mixture_state": {
                "epoch": 0,
                "prompt_cursor": 1,
                "rollouts": 10,
                "total_action_tokens": 20,
            },
        }

    monkeypatch.setattr(rl_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        rl_module.torch.optim,
        "AdamW",
        lambda params, lr, **_kwargs: original_sgd(params, lr=lr),
    )
    monkeypatch.setattr(
        rl_module,
        "segment_policy_loss",
        lambda policy, _reference, _segment, advantage, _kl, _device: (
            -policy.weight * advantage,
            torch.tensor(0.0),
        ),
    )
    monkeypatch.setattr(
        rl_module,
        "save_checkpoint",
        lambda directory, _model, _optimizer, _scheduler, step, tokens, progress, saved_config: (
            saves.append((directory, step, tokens, progress.copy(), saved_config.copy()))
        ),
    )

    settings = config(
        tmp_path,
        checkpoint_directory=checkpoint,
        prompts_path=prompts,
        epochs=2,
        group_size=2,
        save_every_steps=2,
        resume_from=tmp_path / "resume",
        judge_model="Qwen/Qwen3.5-2B",
        judge_revision="a" * 40,
        judge_device="cpu",
        judge_max_input_tokens=512,
        judge_max_new_tokens=32,
    )
    final = train_search_rl(settings)

    assert final == tmp_path / "output" / "final"
    assert len(models) == 2
    policy, reference = models
    assert loaded_weights == [(checkpoint, policy), (checkpoint, reference)]
    assert loaded_resume == [policy]
    assert reference.training is False
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert providers[0].kwargs == {}
    assert providers[0].closed == 1
    assert len(judges) == 1
    assert judges[0].model_name == "Qwen/Qwen3.5-2B"
    assert judges[0].revision == "a" * 40
    assert judges[0].kwargs == {
        "device": "cpu",
        "max_input_tokens": 512,
        "max_new_tokens": 32,
    }
    assert len(judges[0].calls) == 6
    assert judges[0].closed == 1
    assert FakeAgent.requests
    assert all(request.search_mode == "local" for request in FakeAgent.requests)
    assert all(request.max_searches == settings.max_searches for request in FakeAgent.requests)
    metrics_text = (tmp_path / "output" / "metrics.jsonl").read_text()
    records = [json.loads(line) for line in metrics_text.splitlines()]
    assert [record["step"] for record in records] == [6, 7, 8]
    assert records[-1]["rollouts"] == 16
    assert records[-1]["action_tokens"] == 26
    assert records[-1]["reward/judge_answer_correctness"] == pytest.approx(0.5)
    assert records[-1]["reward/judge_evidence_support"] == pytest.approx(0.25)
    assert records[-1]["reward/judge_search_quality"] == pytest.approx(1.0)
    assert records[-1]["reward/judge_valid"] == pytest.approx(1.0)
    assert records[-1]["judge_latency_seconds"] >= 0.0
    assert saves[-1][0] == final
    assert saves[-1][1:4] == (
        8,
        26,
        {"epoch": 2, "prompt_cursor": 0, "rollouts": 16, "total_action_tokens": 26},
    )
    assert saves[-1][4]["model"]["gradient_checkpointing"] is True
    search_rl_config = saves[-1][4]["search_rl"]
    assert search_rl_config["judge_model"] == "Qwen/Qwen3.5-2B"
    assert search_rl_config["judge_revision"] == "a" * 40
    assert search_rl_config["judge_device"] == "cpu"
    assert search_rl_config["reward_weights"]["judge_answer_correctness"] == 0.20

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret")
    FakeAgent.trace_providers = ("web", "local-fallback")
    web_settings = config(
        tmp_path,
        checkpoint_directory=checkpoint,
        prompts_path=prompts,
        output_directory=tmp_path / "web-output",
        epochs=1,
        group_size=2,
        search_mode="web",
    )
    web_final = train_search_rl(web_settings)

    assert web_final == tmp_path / "web-output" / "final"
    assert len(web_providers) == 1
    assert web_providers[0].api_key == "secret"
    assert web_providers[0].closed == 1
    assert len(caches) == 1
    assert caches[0].path == tmp_path / "web-output" / "brave-search-cache.sqlite3"
    assert caches[0].closed == 1
    web_records = [
        json.loads(line)
        for line in (tmp_path / "web-output" / "metrics.jsonl").read_text().splitlines()
    ]
    assert web_records
    assert sum(record["web_searches"] for record in web_records) > 0
    assert sum(record["local_fallbacks"] for record in web_records) > 0
    FakeAgent.trace_providers = ()
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    original_perf_counter = rl_module.time.perf_counter
    clock_calls = 0

    def expired_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 3_601.0

    monkeypatch.setattr(rl_module.time, "perf_counter", expired_clock)
    timed_out_settings = config(
        tmp_path,
        checkpoint_directory=checkpoint,
        prompts_path=prompts,
        output_directory=tmp_path / "timed-out-output",
        epochs=2,
        group_size=2,
        time_budget_hours=1.0,
        resume_from=tmp_path / "resume",
    )
    timed_out = train_search_rl(timed_out_settings)
    monkeypatch.setattr(rl_module.time, "perf_counter", original_perf_counter)

    assert timed_out == tmp_path / "timed-out-output" / "step-00000005"
    assert saves[-1][1:4] == (
        5,
        20,
        {"epoch": 0, "prompt_cursor": 1, "rollouts": 10, "total_action_tokens": 20},
    )
    assert saves[-1][4]["search_rl"]["time_budget_hours"] == 1.0

    # Judge inference errors abort the update instead of silently changing the
    # reward distribution, while still closing both judge and search resources.
    FakeJudge.raise_score = True
    failing = config(
        tmp_path,
        checkpoint_directory=checkpoint,
        prompts_path=prompts,
        output_directory=tmp_path / "failing-output",
        epochs=1,
        group_size=2,
        judge_model="Qwen/Qwen3.5-2B",
        judge_revision="a" * 40,
        judge_device="cpu",
    )
    with pytest.raises(RuntimeError, match="judge failed"):
        train_search_rl(failing)
    assert providers[-1].closed == 1
    assert judges[-1].closed == 1

    FakeJudge.raise_score = False
    FakeJudge.raise_init = True
    load_failing = config(
        tmp_path,
        checkpoint_directory=checkpoint,
        prompts_path=prompts,
        output_directory=tmp_path / "load-failing-output",
        epochs=1,
        group_size=2,
        judge_model="Qwen/Qwen3.5-2B",
        judge_revision="a" * 40,
        judge_device="cpu",
    )
    with pytest.raises(RuntimeError, match="judge load failed"):
        train_search_rl(load_failing)
    assert providers[-1].closed == 1
