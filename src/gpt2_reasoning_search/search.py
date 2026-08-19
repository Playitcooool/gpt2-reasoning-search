"""Deterministic local BM25 and optional Brave Search providers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import bm25s
import httpx

from .schemas import SearchResult
from .tokenizer import SPECIAL_TOKENS


class SearchProvider(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


def sanitize_retrieved_text(text: str, max_characters: int = 6_000) -> str:
    """Neutralize model control tokens and collapse unsafe control characters."""
    clean = "".join(
        character if character in "\n\t" or character.isprintable() else " " for character in text
    )
    for token in SPECIAL_TOKENS:
        clean = clean.replace(token, token.replace("<|", "< | ").replace("|>", " | >"))
    return clean[:max_characters]


def chunk_document(text: str, chunk_characters: int = 2_000, overlap: int = 200) -> list[str]:
    if chunk_characters <= overlap or overlap < 0:
        raise ValueError("chunk size must exceed a non-negative overlap")
    normalized = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    for start in range(0, len(normalized), chunk_characters - overlap):
        chunk = normalized[start : start + chunk_characters]
        if chunk:
            chunks.append(chunk)
        if start + chunk_characters >= len(normalized):
            break
    return chunks


def build_wikipedia_index(documents: Iterable[dict[str, str]], output_directory: Path) -> int:
    """Build a persistent BM25 index from id/title/url/text dictionaries."""
    corpus: list[dict[str, str]] = []
    for document in documents:
        for position, content in enumerate(chunk_document(document["text"])):
            corpus.append(
                {
                    "id": f"{document['id']}:{position}",
                    "title": document["title"],
                    "url": document["url"],
                    "content": content,
                }
            )
    if not corpus:
        raise ValueError("cannot build an empty search index")
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize([item["title"] + " " + item["content"] for item in corpus]))
    output_directory.mkdir(parents=True, exist_ok=True)
    retriever.save(output_directory, corpus=corpus)
    metadata = {"documents": len(corpus), "format": 1}
    (output_directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return len(corpus)


def stream_jsonl_documents(path: Path) -> Iterable[dict[str, str]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            required = {"id", "title", "url", "text"}
            if missing := required - row.keys():
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            yield {key: str(row[key]) for key in required}


class LocalWikipediaSearchProvider:
    def __init__(self, index_directory: Path) -> None:
        self.retriever = bm25s.BM25.load(index_directory, load_corpus=True, mmap=True)

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        documents, scores = self.retriever.retrieve(
            bm25s.tokenize([query]), k=top_k, show_progress=False
        )
        results: list[SearchResult] = []
        for document, score in zip(documents[0], scores[0], strict=True):
            content = sanitize_retrieved_text(document["content"])
            results.append(
                SearchResult(
                    id=document["id"],
                    title=sanitize_retrieved_text(document["title"], 500),
                    url=document["url"],
                    snippet=content[:500],
                    content=content,
                    score=float(score),
                )
            )
        return results


class BraveWebSearchProvider:
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout_seconds: float = 15.0) -> None:
        if not api_key:
            raise ValueError("Brave API key is required")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                self.endpoint, params={"q": query, "count": top_k}, headers=headers
            )
            response.raise_for_status()
        rows = response.json().get("web", {}).get("results", [])[:top_k]
        return [
            SearchResult(
                id=f"brave:{position}",
                title=sanitize_retrieved_text(row.get("title", ""), 500),
                url=row.get("url", ""),
                snippet=sanitize_retrieved_text(row.get("description", ""), 500),
                content=sanitize_retrieved_text(row.get("description", "")),
                score=float(top_k - position),
            )
            for position, row in enumerate(rows)
        ]
