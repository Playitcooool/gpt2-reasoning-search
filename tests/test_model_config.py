import pytest
import torch

from gpt2_reasoning_search.config import ModelConfig, TrainConfig
from gpt2_reasoning_search.model import GPT2ReasoningModel


def _architectural_parameter_count(config: ModelConfig) -> int:
    embedding = config.vocab_size * config.d_model
    attention = (
        config.d_model * (config.d_model + 2 * config.n_kv_heads * config.head_dim)
        + config.d_model * config.d_model
    )
    per_block = attention + 3 * config.d_model * config.intermediate_size + 2 * config.d_model
    return embedding + config.n_layers * per_block + config.d_model


def test_model_forward_loss_weight_tying_and_greedy_generation() -> None:
    config = ModelConfig(
        vocab_size=64,
        max_seq_len=12,
        n_layers=2,
        d_model=32,
        n_heads=4,
        intermediate_size=64,
        gradient_checkpointing=False,
    )
    model = GPT2ReasoningModel(config)
    inputs = torch.randint(0, config.vocab_size, (2, 7))

    output = model(inputs, labels=inputs)

    assert output.logits.shape == (2, 7, config.vocab_size)
    assert output.loss is not None and output.loss.isfinite()
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    assert model.parameter_count() == _architectural_parameter_count(config)
    assert model.generate(inputs[:, :3], max_new_tokens=2, temperature=0).shape == (2, 5)


def test_model_rejects_invalid_input_shapes_and_context_length() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=4,
        n_layers=1,
        d_model=16,
        n_heads=4,
        intermediate_size=32,
        gradient_checkpointing=False,
    )
    model = GPT2ReasoningModel(config)

    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(4, dtype=torch.long))
    with pytest.raises(ValueError, match="maximum"):
        model(torch.ones((1, 5), dtype=torch.long))


def test_preset_sizes_and_head_dimensions() -> None:
    proxy = ModelConfig.preset("proxy-124m")
    main = ModelConfig.preset("main-350m")

    assert 120_000_000 <= _architectural_parameter_count(proxy) <= 130_000_000
    assert 340_000_000 <= _architectural_parameter_count(main) <= 360_000_000
    assert proxy.head_dim == 64
    assert main.head_dim == 64
    with pytest.raises(ValueError, match="unknown model preset"):
        ModelConfig.preset("missing")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divisible"):
        _ = ModelConfig(d_model=10, n_heads=3).head_dim


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_heads": 0},
        {"n_kv_heads": 0},
        {"n_heads": 4, "n_kv_heads": 3},
        {"d_model": 0},
    ],
)
def test_model_config_rejects_invalid_attention_dimensions(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**overrides)


def test_gqa_cache_shapes() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=2,
        d_model=24,
        n_heads=6,
        n_kv_heads=2,
        intermediate_size=48,
    )
    model = GPT2ReasoningModel(config).eval()

    output = model(torch.randint(0, config.vocab_size, (3, 5)), use_cache=True)

    assert output.past_key_values is not None
    assert len(output.past_key_values) == config.n_layers
    for key, value in output.past_key_values:
        assert key.shape == (3, config.n_kv_heads, 5, config.head_dim)
        assert value.shape == key.shape


def test_train_config_validates_ratio_and_computes_step_tokens(tmp_path) -> None:
    config = TrainConfig(
        output_dir=tmp_path,
        reasoning_tokens=tmp_path / "reasoning.npy",
        general_tokens=tmp_path / "general.npy",
        sequence_length=16,
        micro_batch_size=2,
        gradient_accumulation_steps=3,
        max_tokens=96,
    )
    assert config.tokens_per_optimizer_step == 96

    with pytest.raises(ValueError, match="reasoning_ratio"):
        TrainConfig(
            output_dir=tmp_path,
            reasoning_tokens=tmp_path / "r.npy",
            general_tokens=tmp_path / "g.npy",
            reasoning_ratio=1.1,
        )


@pytest.mark.parametrize("budget", [0.0, -1.0, float("nan"), float("inf")])
def test_train_config_rejects_non_finite_or_non_positive_time_budget(
    tmp_path, budget: float
) -> None:
    with pytest.raises(ValueError, match="time budget"):
        TrainConfig(
            output_dir=tmp_path,
            reasoning_tokens=tmp_path / "reasoning.bin",
            general_tokens=tmp_path / "general.bin",
            time_budget_hours=budget,
        )
