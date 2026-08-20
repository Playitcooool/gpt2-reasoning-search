from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from gpt2_reasoning_search.config import ModelConfig
from gpt2_reasoning_search.model import GPT2ReasoningModel


def tiny_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 37,
        "max_seq_len": 8,
        "n_layers": 2,
        "d_model": 24,
        "n_heads": 6,
        "n_kv_heads": 2,
        "intermediate_size": 48,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]


def test_cached_logits_match_full_prefix_logits() -> None:
    torch.manual_seed(11)
    model = GPT2ReasoningModel(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    full = model(tokens).logits

    prefill = model(tokens[:, :4], use_cache=True)
    assert prefill.past_key_values is not None
    cached = model(tokens[:, 4:], past_key_values=prefill.past_key_values, use_cache=True)

    torch.testing.assert_close(cached.logits, full[:, 4:], rtol=1e-5, atol=1e-6)


def test_generate_uses_single_token_cached_decode_after_prefill() -> None:
    model = GPT2ReasoningModel(tiny_config(max_seq_len=12)).eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    sequence_lengths: list[int] = []
    original_forward = model.forward

    def recording_forward(input_ids: torch.Tensor, *args: object, **kwargs: object):
        sequence_lengths.append(input_ids.shape[1])
        return original_forward(input_ids, *args, **kwargs)

    with patch.object(model, "forward", side_effect=recording_forward):
        model.generate(prompt, max_new_tokens=4, temperature=0)

    assert sequence_lengths == [4, 1, 1, 1]


def test_generate_stops_on_configured_stop_token() -> None:
    model = GPT2ReasoningModel(tiny_config()).eval()
    prompt = torch.tensor([[1, 2]])
    with patch.object(model, "_sample", return_value=torch.tensor([[7]])) as sample:
        output = model.generate(prompt, max_new_tokens=5, stop_token_ids={7})

    assert output.tolist() == [[1, 2, 7]]
    sample.assert_called_once()


def test_generate_rebuilds_cache_when_context_window_is_full() -> None:
    model = GPT2ReasoningModel(tiny_config(max_seq_len=4)).eval()
    prompt = torch.tensor([[1, 2, 3]])
    calls: list[tuple[int, bool]] = []
    original_forward = model.forward

    def recording_forward(input_ids: torch.Tensor, *args: object, **kwargs: object):
        calls.append((input_ids.shape[1], kwargs.get("past_key_values") is not None))
        return original_forward(input_ids, *args, **kwargs)

    with (
        patch.object(model, "forward", side_effect=recording_forward),
        patch.object(model, "_sample", return_value=torch.tensor([[6]])),
    ):
        output = model.generate(prompt, max_new_tokens=4)

    assert output.shape == (1, 7)
    # Once the rolling window is full, rebuilding it is required on each step so
    # rotary positions remain aligned with the shifted context.
    assert calls == [(3, False), (1, True), (4, False), (4, False)]


def test_gradient_checkpointed_forward_backward() -> None:
    model = GPT2ReasoningModel(tiny_config(gradient_checkpointing=True)).train()
    tokens = torch.randint(0, model.config.vocab_size, (2, 6))

    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert model.blocks[-1].mlp.down.weight.grad is not None
    with pytest.raises(ValueError, match="inference"):
        model(tokens, use_cache=True)
