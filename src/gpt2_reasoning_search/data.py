"""Corpus normalization, exact mixture accounting, and packed token streams."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer


@dataclass(slots=True)
class MixtureState:
    reasoning_tokens: int = 0
    general_tokens: int = 0
    reasoning_cursor: int = 0
    general_cursor: int = 0

    @property
    def total_tokens(self) -> int:
        return self.reasoning_tokens + self.general_tokens


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    text: str
    source: str
    verification: str = "source-curated"


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write a text artifact atomically so interrupted preparation cannot publish a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    try:
        with temporary_handle:
            temporary_handle.write(text)
            temporary_handle.flush()
            os.fsync(temporary_handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_lines(
    path: Path,
    lines: Iterable[str],
    *,
    encoding: str = "utf-8",
    require_nonempty: bool = False,
) -> None:
    """Stream text lines to an atomic replacement without buffering the whole artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    try:
        with temporary_handle:
            wrote_line = False
            for line in lines:
                wrote_line = True
                temporary_handle.write(line)
                if not line.endswith("\n"):
                    temporary_handle.write("\n")
            if require_nonempty and not wrote_line:
                raise ValueError(f"no lines were written to {path}")
            temporary_handle.flush()
            os.fsync(temporary_handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class PreparationStats:
    rows_seen: int = 0
    rows_accepted: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    accepted_sources: Counter[str] = field(default_factory=Counter)

    def accept(self, source: str) -> None:
        self.rows_accepted += 1
        self.accepted_sources[source] += 1

    def reject(self, reason: str) -> None:
        self.rejection_reasons[reason] += 1

    def to_dict(self) -> dict:
        return {
            "rows_seen": self.rows_seen,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_seen - self.rows_accepted,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "accepted_sources": dict(sorted(self.accepted_sources.items())),
        }


class ExactTokenMixture:
    """Create a finite stream whose final source counts exactly match integer quotas."""

    def __init__(
        self,
        reasoning: np.ndarray,
        general: np.ndarray,
        total_tokens: int,
        reasoning_ratio: float,
        state: MixtureState | None = None,
    ) -> None:
        if reasoning.dtype.kind not in "iu" or general.dtype.kind not in "iu":
            raise TypeError("token arrays must contain integers")
        self.sources = {"reasoning": reasoning, "general": general}
        self.quotas = {
            "reasoning": round(total_tokens * reasoning_ratio),
            "general": total_tokens - round(total_tokens * reasoning_ratio),
        }
        self.state = state or MixtureState()

    def _remaining(self, source: str) -> int:
        return self.quotas[source] - getattr(self.state, f"{source}_tokens")

    def _take_source(self, source: str, count: int) -> list[np.ndarray]:
        pieces: list[np.ndarray] = []
        values = self.sources[source]
        if count and not len(values):
            raise ValueError(f"{source} token source is empty")
        cursor_name = f"{source}_cursor"
        while count:
            cursor = getattr(self.state, cursor_name)
            take_now = min(count, len(values) - cursor)
            pieces.append(values[cursor : cursor + take_now])
            setattr(self.state, cursor_name, (cursor + take_now) % len(values))
            setattr(
                self.state, f"{source}_tokens", getattr(self.state, f"{source}_tokens") + take_now
            )
            count -= take_now
        return pieces

    def take(self, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        total_quota = sum(self.quotas.values())
        emitted = self.state.total_tokens
        actual_count = min(count, total_quota - emitted)
        if actual_count <= 0:
            raise StopIteration
        target_reasoning = round((emitted + actual_count) * self.quotas["reasoning"] / total_quota)
        reasoning_count = target_reasoning - self.state.reasoning_tokens
        general_count = actual_count - reasoning_count
        if not 0 <= reasoning_count <= self._remaining("reasoning"):
            raise RuntimeError("reasoning mixture state is inconsistent with configured quota")
        if not 0 <= general_count <= self._remaining("general"):
            raise RuntimeError("general mixture state is inconsistent with configured quota")
        pieces = self._take_source("reasoning", reasoning_count)
        pieces.extend(self._take_source("general", general_count))
        return np.concatenate(pieces)

    def batches(
        self, batch_size: int, sequence_length: int, device: torch.device
    ) -> Iterator[torch.Tensor]:
        count = batch_size * sequence_length
        while sum(self.quotas.values()) - self.state.total_tokens >= count:
            yield (
                torch.from_numpy(self.take(count).astype(np.int64))
                .view(batch_size, sequence_length)
                .to(device, non_blocking=True)
            )

    def state_dict(self) -> dict[str, int]:
        return asdict(self.state)


class ContaminationFilter:
    def __init__(
        self, evaluation_prompts: Sequence[str], ngram_size: int = 8, threshold: float = 0.5
    ):
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.eval_sets = [self._ngrams(prompt) for prompt in evaluation_prompts]

    def _ngrams(self, text: str) -> set[str]:
        words = re.findall(r"\w+", text.lower())
        return {
            " ".join(words[i : i + self.ngram_size])
            for i in range(len(words) - self.ngram_size + 1)
        }

    def contaminated(self, text: str) -> bool:
        candidate = self._ngrams(text)
        if not candidate:
            return False
        return any(
            len(candidate & known) / min(len(candidate), len(known)) >= self.threshold
            for known in self.eval_sets
            if known
        )


def normalized_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


class SQLiteDeduplicator:
    """Bound exact-dedup memory by storing hashes in a temporary SQLite index."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS hashes (digest TEXT PRIMARY KEY) WITHOUT ROWID"
        )

    def add(self, digest: str) -> bool:
        cursor = self.connection.execute("INSERT OR IGNORE INTO hashes VALUES (?)", (digest,))
        return cursor.rowcount == 1

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def reasoning_document(problem: str, reasoning: str, answer: str) -> str:
    return (
        f"<|problem|>\n{problem.strip()}\n<|reasoning|>\n{reasoning.strip()}\n"
        f"<|answer|>\n{answer.strip()}"
    )


def stream_huggingface_texts(
    dataset_id: str,
    revision: str,
    split: str = "train",
    config_name: str | None = None,
    text_field: str = "text",
) -> Iterator[dict]:
    dataset = load_dataset(dataset_id, config_name, split=split, revision=revision, streaming=True)
    for row in dataset:
        if row.get(text_field):
            yield dict(row)


def write_token_file(
    tokenizer: Tokenizer,
    documents: Iterable[str | PreparedDocument],
    output_path: Path,
    max_tokens: int,
    *,
    batch_size: int = 256,
    contamination_filter: ContaminationFilter | None = None,
    preparation_stats: PreparationStats | None = None,
) -> dict:
    """Tokenize in batches and stream directly to disk without corpus-sized RAM use."""
    if max_tokens <= 0 or batch_size <= 0:
        raise ValueError("max_tokens and batch_size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    dtype = np.dtype("uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32")
    content_hash = hashlib.sha256()
    written = duplicates = contaminated = documents_written = truncated = 0
    source_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    pending: list[PreparedDocument] = []

    def normalized(document: str | PreparedDocument) -> PreparedDocument:
        if isinstance(document, PreparedDocument):
            return document
        return PreparedDocument(text=document, source="unspecified")

    def flush(handle) -> bool:
        nonlocal written, documents_written, truncated
        if not pending:
            return False
        encodings = tokenizer.encode_batch([document.text for document in pending])
        reached_cap = False
        for document, encoding in zip(pending, encodings, strict=True):
            remaining = max_tokens - written
            if remaining <= 0:
                reached_cap = True
                break
            ids = np.asarray(encoding.ids[:remaining], dtype=dtype)
            payload = ids.tobytes(order="C")
            handle.write(payload)
            content_hash.update(payload)
            written += len(ids)
            documents_written += 1
            source_counts[document.source] += 1
            verification_counts[document.verification] += 1
            if len(ids) < len(encoding.ids):
                truncated += 1
                reached_cap = True
                break
        pending.clear()
        return reached_cap

    temporary_output = output_path.with_suffix(output_path.suffix + ".partial")
    dedup_handle = tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix="dedup-", suffix=".sqlite3", delete=False
    )
    dedup_path = Path(dedup_handle.name)
    dedup_handle.close()
    deduplicator = SQLiteDeduplicator(dedup_path)
    try:
        with temporary_output.open("wb") as handle:
            for raw_document in documents:
                document = normalized(raw_document)
                if contamination_filter and contamination_filter.contaminated(document.text):
                    contaminated += 1
                    continue
                if not deduplicator.add(normalized_hash(document.text)):
                    duplicates += 1
                    continue
                pending.append(document)
                if len(pending) >= batch_size and flush(handle):
                    break
            else:
                flush(handle)
            if written == 0:
                raise ValueError(f"no usable tokens were written to {output_path}")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_output, output_path)
    finally:
        deduplicator.close()
        dedup_path.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)

    report = {
        "format_version": 2,
        "dtype": dtype.name,
        "vocab_size": vocab_size,
        "tokens": written,
        "bytes": output_path.stat().st_size,
        "documents": documents_written,
        "duplicates_removed": duplicates,
        "contaminated_removed": contaminated,
        "truncated_documents": truncated,
        "source_documents": dict(sorted(source_counts.items())),
        "verification": dict(sorted(verification_counts.items())),
        "tokenizer_sha256": hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
        "sha256": content_hash.hexdigest(),
    }
    if preparation_stats is not None:
        report["filtering"] = preparation_stats.to_dict()
    manifest_path = output_path.with_suffix(".manifest.json")
    atomic_write_text(manifest_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_token_array(path: Path) -> np.ndarray:
    """Load scalable raw token streams and legacy NumPy arrays read-only."""
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"token manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    dtype = np.dtype(manifest["dtype"])
    expected_bytes = int(manifest["tokens"]) * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError("token file size does not match its manifest")
    return np.memmap(path, dtype=dtype, mode="r", shape=(int(manifest["tokens"]),))
