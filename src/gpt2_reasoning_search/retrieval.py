"""Disk-backed BM25 retrieval used as the local live-search fallback."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import tantivy

from .schemas import SearchResult


def _schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_unsigned_field("row_id", stored=True, fast=True)
    builder.add_text_field("chunk_id", stored=True)
    builder.add_text_field("title", stored=True)
    builder.add_text_field("url", stored=True)
    builder.add_text_field("content", stored=True)
    return builder.build()


def build_wikipedia_index(documents: Iterable[dict[str, str]], output_directory: Path) -> int:
    """Stream chunks into a compact, deterministic Tantivy BM25 index."""
    from .search import chunk_document

    if output_directory.exists():
        raise FileExistsError(f"index target already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}-", dir=output_directory.parent)
    )
    lexical_directory = temporary / "lexical"
    lexical_directory.mkdir()
    index = tantivy.Index(_schema(), path=str(lexical_directory))
    writer = index.writer(heap_size=512_000_000)
    metadata = sqlite3.connect(temporary / "metadata.sqlite3")
    metadata.execute(
        "CREATE TABLE chunks (row_id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, "
        "title TEXT, url TEXT, content TEXT)"
    )
    metadata.execute("PRAGMA journal_mode=OFF")
    metadata.execute("PRAGMA synchronous=OFF")
    count = 0
    metadata_closed = False

    try:
        for document in documents:
            for position, content in enumerate(chunk_document(document["text"])):
                row_id = count
                chunk_id = f"{document['id']}:{position}"
                indexed_document = tantivy.Document()
                indexed_document.add_unsigned("row_id", row_id)
                indexed_document.add_text("chunk_id", chunk_id)
                indexed_document.add_text("title", document["title"])
                indexed_document.add_text("url", document["url"])
                indexed_document.add_text("content", content)
                writer.add_document(indexed_document)
                metadata.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    (row_id, chunk_id, document["title"], document["url"], content),
                )
                count += 1
                if count % 50_000 == 0:
                    metadata.commit()
        if count == 0:
            raise ValueError("cannot build an empty search index")
        writer.commit()
        writer.wait_merging_threads()
        metadata.commit()
        metadata.close()
        metadata_closed = True
        manifest = {"format_version": 3, "chunks": count, "lexical": "tantivy-bm25"}
        (temporary / "retrieval-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output_directory)
        return count
    except Exception:
        try:
            writer.rollback()
            writer.wait_merging_threads()
        except (RuntimeError, ValueError):
            pass
        if not metadata_closed:
            metadata.close()
        del writer
        del index
        gc.collect()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class LocalWikipediaSearchProvider:
    """Small BM25-only fallback when live Brave search is unavailable."""

    def __init__(self, index_directory: Path) -> None:
        self.index = tantivy.Index.open(str(index_directory / "lexical"))
        self.index.reload()
        self.metadata = sqlite3.connect(
            index_directory / "metadata.sqlite3", check_same_thread=False
        )
        self.manifest = json.loads((index_directory / "retrieval-manifest.json").read_text())

    def _metadata_rows(self, row_ids: Sequence[int]) -> dict[int, tuple[str, str, str, str]]:
        if not row_ids:
            return {}
        placeholders = ",".join("?" for _ in row_ids)
        rows = self.metadata.execute(
            f"SELECT row_id, chunk_id, title, url, content FROM chunks "
            f"WHERE row_id IN ({placeholders})",
            tuple(row_ids),
        )
        return {int(row[0]): (row[1], row[2], row[3], row[4]) for row in rows}

    def _search_sync(self, query: str, top_k: int) -> list[SearchResult]:
        from .search import sanitize_retrieved_text

        parsed, _errors = self.index.parse_query_lenient(
            query,
            default_field_names=["title", "content"],
            field_boosts={"title": 2.0},
            conjunction_by_default=False,
        )
        hits = self.index.searcher().search(parsed, limit=top_k).hits
        ranked = []
        for score, address in hits:
            document = self.index.searcher().doc(address)
            ranked.append((int(document.get_first("row_id")), float(score)))
        metadata = self._metadata_rows([row_id for row_id, _score in ranked])
        return [
            SearchResult(
                id=metadata[row_id][0],
                title=sanitize_retrieved_text(metadata[row_id][1], 500),
                url=metadata[row_id][2],
                snippet=sanitize_retrieved_text(metadata[row_id][3], 500),
                content=sanitize_retrieved_text(metadata[row_id][3]),
                score=score,
                provider="local-wikipedia",
                score_components={"bm25": score},
            )
            for row_id, score in ranked
        ]

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        return await asyncio.to_thread(self._search_sync, query, top_k)

    def close(self) -> None:
        self.metadata.close()
