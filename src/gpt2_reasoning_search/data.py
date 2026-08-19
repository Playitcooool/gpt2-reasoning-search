"""Corpus normalization, exact mixture accounting, and packed token streams."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
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
        target_reasoning = round(
            (emitted + actual_count) * self.quotas["reasoning"] / total_quota
        )
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
    documents: Iterable[str],
    output_path: Path,
    max_tokens: int,
) -> dict[str, int | str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = []
    seen: set[str] = set()
    written = duplicates = 0
    for document in documents:
        digest = normalized_hash(document)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        ids = np.asarray(tokenizer.encode(document).ids, dtype=np.uint32)
        ids = ids[: max_tokens - written]
        pieces.append(ids)
        written += len(ids)
        if written >= max_tokens:
            break
    tokens = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint32)
    np.save(output_path, tokens)
    report = {
        "tokens": int(len(tokens)),
        "duplicates_removed": duplicates,
        "sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
