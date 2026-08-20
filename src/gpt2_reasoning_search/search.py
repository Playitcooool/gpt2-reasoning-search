"""Shared search contracts, normalization, chunking, and rank fusion."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .schemas import SearchResult
from .tokenizer import SPECIAL_TOKENS


class SearchProvider(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[SearchResult], top_k: int
    ) -> list[SearchResult]: ...


def sanitize_retrieved_text(text: str, max_characters: int = 12_000) -> str:
    """Neutralize model control tokens and collapse unsafe control characters."""
    clean = "".join(
        character if character in "\n\t" or character.isprintable() else " " for character in text
    )
    for token in SPECIAL_TOKENS:
        clean = clean.replace(token, token.replace("<|", "< | ").replace("|>", " | >"))
    return clean[:max_characters]


def chunk_document(
    text: str, chunk_characters: int = 2_000, overlap_characters: int = 200
) -> list[str]:
    """Create sentence-aware chunks while retaining a small boundary overlap."""
    if chunk_characters <= overlap_characters or overlap_characters < 0:
        raise ValueError("chunk size must exceed a non-negative overlap")
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n{2,}", normalized)
        if unit.strip()
    ]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > chunk_characters:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(unit), chunk_characters - overlap_characters):
                piece = unit[start : start + chunk_characters]
                if piece:
                    chunks.append(piece)
                if start + chunk_characters >= len(unit):
                    break
            continue
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > chunk_characters:
            chunks.append(current)
            overlap = current[-overlap_characters:] if overlap_characters else ""
            current = f"{overlap} {unit}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    tracking = {"fbclid", "gclid", "mc_cid", "mc_eid"}
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in tracking
        )
    )
    return urlunsplit((scheme, netloc, path, query, ""))


def stable_source_id(provider: str, url: str) -> str:
    digest = hashlib.sha256(canonicalize_url(url).encode()).hexdigest()[:20]
    return f"{provider}:{digest}"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], constant: int = 60
) -> dict[str, float]:
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (constant + rank)
    return scores


def stream_jsonl_documents(path: Path) -> Iterable[dict[str, str]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            required = {"id", "title", "url", "text"}
            if missing := required - row.keys():
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            yield {key: str(row[key]) for key in required}


if TYPE_CHECKING:
    from .retrieval import LocalWikipediaSearchProvider, build_wikipedia_index
    from .web_search import BraveWebSearchProvider


def __getattr__(name: str) -> Any:
    """Lazily preserve the original imports without creating module cycles."""
    if name in {"LocalWikipediaSearchProvider", "build_wikipedia_index"}:
        from . import retrieval

        return getattr(retrieval, name)
    if name == "BraveWebSearchProvider":
        from .web_search import BraveWebSearchProvider

        return BraveWebSearchProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BraveWebSearchProvider",
    "LocalWikipediaSearchProvider",
    "Reranker",
    "SearchProvider",
    "build_wikipedia_index",
    "canonicalize_url",
    "chunk_document",
    "reciprocal_rank_fusion",
    "sanitize_retrieved_text",
    "stable_source_id",
    "stream_jsonl_documents",
]
