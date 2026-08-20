from __future__ import annotations

from types import SimpleNamespace

import torch

from gpt2_reasoning_search.runner import ModelRunner


class FakeTokenizer:
    def encode(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[int(value) for value in prompt.split(",") if value])

    def token_to_id(self, token: str) -> int | None:
        return {"<|eos|>": 98, "<|end_tool_call|>": 99}.get(token)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return ",".join(map(str, ids))


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_seq_len=8)
        self.calls: list[dict[str, object]] = []

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.calls.append({"input_ids": input_ids.clone(), **kwargs})
        return torch.cat((input_ids, torch.tensor([[7, 99]])), dim=1)


def test_prompt_truncation_preserves_head_and_tail_and_generation_stop_tokens() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cpu")
    runner.tokenizer = FakeTokenizer()  # type: ignore[assignment]
    runner.model = FakeModel()  # type: ignore[assignment]

    text, input_count, output_count = runner.generate(
        ",".join(str(value) for value in range(12)), max_new_tokens=5
    )

    call = runner.model.calls[0]
    assert call["input_ids"].tolist() == [[0, 1, 6, 7, 8, 9, 10, 11]]
    assert call["eos_token_id"] == 98
    assert call["stop_token_ids"] == {99}
    assert call["max_new_tokens"] == 5
    assert (text, input_count, output_count) == ("7,99", 8, 2)
