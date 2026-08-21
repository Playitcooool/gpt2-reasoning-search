from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2_reasoning_search.data import (
    ContaminationFilter,
    ExactTokenMixture,
    MixtureState,
    atomic_write_lines,
    atomic_write_text,
    load_token_array,
    write_token_file,
)
from gpt2_reasoning_search.tokenizer import train_tokenizer


def test_atomic_text_and_lines_replace_existing_files_without_partials(tmp_path: Path) -> None:
    text_path = tmp_path / "nested" / "manifest.json"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("old\n")
    atomic_write_text(text_path, "new\n")
    assert text_path.read_text() == "new\n"

    lines_path = tmp_path / "nested" / "trajectories.jsonl"
    atomic_write_lines(lines_path, (line for line in ("first", "second\n")))
    assert lines_path.read_text() == "first\nsecond\n"
    assert not list(tmp_path.rglob("*.partial"))


def test_atomic_writes_preserve_existing_target_when_generation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gpt2_reasoning_search.data as data_module

    text_path = tmp_path / "manifest.json"
    text_path.write_text("old\n")

    def fail_replace(_source, _target):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(data_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        atomic_write_text(text_path, "new\n")
    assert text_path.read_text() == "old\n"
    assert not list(tmp_path.glob("*.partial"))

    # A producer failure before replacement has the same recoverable behavior.
    def broken_lines():
        yield "partial"
        raise RuntimeError("simulated producer failure")

    monkeypatch.undo()
    lines_path = tmp_path / "trajectories.jsonl"
    lines_path.write_text("old\n")
    with pytest.raises(RuntimeError, match="simulated producer failure"):
        atomic_write_lines(lines_path, broken_lines())
    assert lines_path.read_text() == "old\n"
    assert not list(tmp_path.glob("*.partial"))


def test_write_token_file_rejects_empty_or_fully_filtered_documents(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("a small tokenizer training sample with enough words\n")
    tokenizer = train_tokenizer([sample], tmp_path / "tokenizer.json", vocab_size=64)

    with pytest.raises(ValueError, match="no usable tokens"):
        write_token_file(tokenizer, [], tmp_path / "empty.bin", max_tokens=32)

    with pytest.raises(ValueError, match="no usable tokens"):
        write_token_file(
            tokenizer,
            ["duplicate", "duplicate"],
            tmp_path / "filtered.bin",
            max_tokens=32,
            contamination_filter=ContaminationFilter(["duplicate"], ngram_size=1),
        )

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


def test_mixture_matches_configured_ratio_at_multiple_cumulative_prefixes() -> None:
    mixture = ExactTokenMixture(
        reasoning=np.arange(100, 120, dtype=np.int64),
        general=np.arange(200, 220, dtype=np.int64),
        total_tokens=100,
        reasoning_ratio=0.7,
    )
    emitted = 0

    for chunk_size in (1, 2, 4, 8, 13, 17, 23, 32):
        chunk = mixture.take(chunk_size)
        emitted += chunk_size

        assert len(chunk) == chunk_size
        assert mixture.state.total_tokens == emitted
        assert abs(mixture.state.reasoning_tokens - emitted * 0.7) <= 0.5 + 1e-12
        assert mixture.state.general_tokens == emitted - mixture.state.reasoning_tokens

    assert emitted == 100


@pytest.mark.parametrize(
    ("ratio", "reasoning", "general", "expected_reasoning", "expected_general"),
    [
        (0.0, np.array([], dtype=np.int64), np.array([20, 21]), 0, 9),
        (1.0, np.array([10, 11], dtype=np.int64), np.array([], dtype=np.int64), 9, 0),
    ],
)
def test_mixture_supports_zero_and_full_reasoning_boundaries(
    ratio: float,
    reasoning: np.ndarray,
    general: np.ndarray,
    expected_reasoning: int,
    expected_general: int,
) -> None:
    mixture = ExactTokenMixture(reasoning, general, total_tokens=9, reasoning_ratio=ratio)

    chunks = [mixture.take(2), mixture.take(3), mixture.take(4)]

    assert sum(len(chunk) for chunk in chunks) == 9
    assert mixture.state.reasoning_tokens == expected_reasoning
    assert mixture.state.general_tokens == expected_general
    assert mixture.state.total_tokens == 9


def test_mixture_wraparound_preserves_source_order_and_cursors() -> None:
    mixture = ExactTokenMixture(
        reasoning=np.array([10, 11], dtype=np.int64),
        general=np.array([20], dtype=np.int64),
        total_tokens=10,
        reasoning_ratio=0.7,
    )

    first = mixture.take(4)
    second = mixture.take(6)

    np.testing.assert_array_equal(first, np.array([10, 11, 10, 20]))
    np.testing.assert_array_equal(second, np.array([11, 10, 11, 10, 20, 20]))
    assert mixture.state.reasoning_cursor == 1
    assert mixture.state.general_cursor == 0


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


def test_resumed_mixture_keeps_exact_prefix_ratio_across_later_chunks() -> None:
    reasoning = np.arange(100, 104, dtype=np.int64)
    general = np.arange(200, 203, dtype=np.int64)
    original = ExactTokenMixture(reasoning, general, total_tokens=100, reasoning_ratio=0.7)
    original.take(9)
    restored_state = MixtureState(**original.state_dict())
    resumed = ExactTokenMixture(
        reasoning,
        general,
        total_tokens=100,
        reasoning_ratio=0.7,
        state=restored_state,
    )

    emitted = 9
    for chunk_size in (1, 5, 11, 21, 53):
        expected = original.take(chunk_size)
        actual = resumed.take(chunk_size)
        emitted += chunk_size

        np.testing.assert_array_equal(actual, expected)
        assert resumed.state.reasoning_tokens == round(emitted * 0.7)
        assert resumed.state.general_tokens == emitted - round(emitted * 0.7)
        assert resumed.state_dict() == original.state_dict()


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
        mixture.take(2)
    with pytest.raises(ValueError, match="positive"):
        ExactTokenMixture(np.array([1]), np.array([2]), 2, 0.5).take(0)


def test_write_token_file_deduplicates_and_records_hash(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma delta")
    tokenizer = train_tokenizer([corpus], tmp_path / "tokenizer.json", vocab_size=300)
    output = tmp_path / "tokens.bin"

    report = write_token_file(tokenizer, ["alpha beta", "alpha beta", "gamma"], output, 100)

    tokens = load_token_array(output)
    assert report["duplicates_removed"] == 1
    assert report["tokens"] == len(tokens)
    assert len(str(report["sha256"])) == 64
    assert output.with_suffix(".manifest.json").is_file()
