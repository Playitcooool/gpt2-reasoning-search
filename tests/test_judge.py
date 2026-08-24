import json
import sys
from types import SimpleNamespace

import pytest
import torch

from gpt2_reasoning_search.judge import (
    DEFAULT_QWEN_JUDGE_MODEL,
    DEFAULT_QWEN_JUDGE_REVISION,
    QwenRewardJudge,
    parse_judge_output,
)
from gpt2_reasoning_search.schemas import AnswerResponse, SearchResult, ToolTraceEntry


def test_parse_judge_output_accepts_only_exact_bounded_finite_schema() -> None:
    valid = parse_judge_output(
        json.dumps(
            {
                "answer_correctness": 1,
                "evidence_support": 0.5,
                "search_quality": 0,
                "reason": "grounded",
            }
        )
    )
    assert valid.valid is True
    assert valid.answer_correctness == 1.0
    assert valid.evidence_support == 0.5
    assert valid.search_quality == 0.0
    assert valid.reason == "grounded"

    invalid_outputs = [
        "not json",
        '{"answer_correctness":1}',
        '{"answer_correctness":1,"evidence_support":1,"search_quality":1,"reason":"x"} tail',
        '{"answer_correctness":true,"evidence_support":1,"search_quality":1,"reason":"x"}',
        '{"answer_correctness":2,"evidence_support":1,"search_quality":1,"reason":"x"}',
        '{"answer_correctness":1,"evidence_support":1,"search_quality":1,"reason":7}',
    ]
    assert all(parse_judge_output(output).valid is False for output in invalid_outputs)


def test_qwen_judge_uses_pinned_deterministic_non_thinking_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    decoded = json.dumps(
        {
            "answer_correctness": 1.0,
            "evidence_support": 0.75,
            "search_quality": 0.5,
            "reason": "supported",
        }
    )

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name, revision):
            calls["processor_load"] = (model_name, revision)
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, tokens, skip_special_tokens):
            calls["decoded_tokens"] = tokens.tolist()
            calls["skip_special_tokens"] = skip_special_tokens
            return decoded

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, revision, **kwargs):
            calls["model_load"] = (model_name, revision, kwargs)
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 3, 4, 5]])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForMultimodalLM=FakeModel, AutoProcessor=FakeProcessor),
    )
    judge = QwenRewardJudge(device="cpu")
    result = SearchResult(
        id="doc:1",
        title="Source",
        url="https://example.test/source",
        snippet="evidence",
        content="supporting evidence",
    )
    response = AnswerResponse(
        answer="Paris [doc:1]",
        reasoning="scratch",
        citations=[],
        tool_trace=[ToolTraceEntry(results=[result])],
        input_tokens=1,
        output_tokens=1,
        elapsed_seconds=0,
        searches_used=1,
    )

    score = judge.score("Capital?", "Paris", response)

    assert calls["processor_load"] == (DEFAULT_QWEN_JUDGE_MODEL, DEFAULT_QWEN_JUDGE_REVISION)
    model_name, revision, model_kwargs = calls["model_load"]
    assert (model_name, revision) == (DEFAULT_QWEN_JUDGE_MODEL, DEFAULT_QWEN_JUDGE_REVISION)
    assert model_kwargs["dtype"] == torch.float32
    assert model_kwargs["attn_implementation"] == "sdpa"
    template_kwargs = calls["template_kwargs"]
    assert template_kwargs["enable_thinking"] is False
    assert template_kwargs["processor_kwargs"] == {
        "truncation": True,
        "max_length": 4096,
    }
    assert "truncation" not in template_kwargs
    assert "max_length" not in template_kwargs
    generate_kwargs = calls["generate_kwargs"]
    assert generate_kwargs["do_sample"] is False
    assert generate_kwargs["max_new_tokens"] == 128
    assert score.valid is True
    assert score.answer_correctness == 1.0
    assert score.evidence_support == 0.75
    assert score.search_quality == 0.5
    judge.close()
    assert judge.model is None and judge.processor is None
