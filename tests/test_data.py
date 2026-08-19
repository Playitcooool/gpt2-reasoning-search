from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2_reasoning_search.data import ExactTokenMixture, MixtureState, write_token_file
from gpt2_reasoning_search.tokenizer import train_tokenizer


def test_exact_mixture_hits_integer_quotas_and_cycles_sources() -> None:
    mixture = ExactTokenMixture(
        reasoning=np.array([10, 11, 12], dtype=np.uint32),
        general=np.array([20, 21], dtype=np.uint32),
        total_tokens=10,
        reasoning_ratio=0.7,
    )

    output = np.concatenate([mixture.take(4), mixture.take(6)])

    assert output.shape == (10,)
    assert mixture.state.reasoning_tokens == 7
    assert mixture.state.general_tokens == 3
    assert mixture.state.total_tokens == 10
    assert sum(value < 20 for value in output) == 7
    assert sum(value >= 20 for value in output) == 3
    with pytest.raises(StopIteration):
        mixture.take(1)


def test_mixture_resume_state_reproduces_uninterrupted_suffix() -> None:
    reasoning = np.arange(100, 107, dtype=np.int64)
    general = np.arange(200, 205, dtype=np.int64)
    uninterrupted = ExactTokenMixture(reasoning, general, 24, 0.7)
    prefix = uninterrupted.take(9)
    saved = MixtureState(**uninterrupted.state_dict())
    expected_suffix = uninterrupted.take(15)

    resumed = ExactTokenMixture(reasoning, general, 24, 0.7, state=saved)
    actual_suffix = resumed.take(15)

    assert prefix.size + actual_suffix.size == 24
    np.testing.assert_array_equal(actual_suffix, expected_suffix)
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_batches_have_requested_packed_shape_and_drop_incomplete_tail() -> None:
    mixture = ExactTokenMixture(
        np.arange(10, dtype=np.int64),
        np.arange(10, 20, dtype=np.int64),
        total_tokens=18,
        reasoning_ratio=0.5,
    )

    batches = list(mixture.batches(batch_size=2, sequence_length=4, device=torch.device("cpu")))

    assert [batch.shape for batch in batches] == [torch.Size([2, 4]), torch.Size([2, 4])]
    assert all(batch.dtype == torch.int64 for batch in batches)
    assert mixture.state.total_tokens == 16


def test_mixture_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integers"):
        ExactTokenMixture(np.array([1.0]), np.array([2]), 2, 0.5)
    mixture = ExactTokenMixture(np.array([], dtype=np.int64), np.array([2]), 2, 0.5)
    with pytest.raises(ValueError, match="empty"):
        mixture.take(1)
    with pytest.raises(ValueError, match="positive"):
        ExactTokenMixture(np.array([1]), np.array([2]), 2, 0.5).take(0)


def test_write_token_file_deduplicates_and_records_hash(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma delta")
    tokenizer = train_tokenizer([corpus], tmp_path / "tokenizer.json", vocab_size=300)
    output = tmp_path / "tokens.npy"

    report = write_token_file(tokenizer, ["alpha beta", "alpha beta", "gamma"], output, 100)

    tokens = np.load(output)
    assert report["duplicates_removed"] == 1
    assert report["tokens"] == len(tokens)
    assert len(str(report["sha256"])) == 64
    assert output.with_suffix(".manifest.json").is_file()
