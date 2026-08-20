"""Disk-backed lexical+dense retrieval with RRF and cross-encoder reranking."""

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
from typing import Protocol

import numpy as np
import tantivy
from usearch.index import Index as VectorIndex

from .schemas import SearchResult


class DenseEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, revision: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode_document(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self.model.encode_query(
                [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )


class CrossEncoderReranker:
    def __init__(self, model_name: str, revision: str, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, revision=revision, device=device)

    def rerank(
        self, query: str, candidates: Sequence[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, f"{candidate.title}\n{candidate.content}") for candidate in candidates]
        scores = np.asarray(self.model.predict(pairs, show_progress_bar=False)).reshape(-1)
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        return [
            candidate.model_copy(
                update={
                    "score": float(score),
                    "score_components": {**candidate.score_components, "reranker": float(score)},
                }
            )
            for candidate, score in ranked[:top_k]
        ]


def _schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_unsigned_field("row_id", stored=True, fast=True)
    builder.add_text_field("chunk_id", stored=True)
    builder.add_text_field("title", stored=True)
    builder.add_text_field("url", stored=True)
    builder.add_text_field("content", stored=True)
    return builder.build()


def _load_retrieval_config(path: Path | None = None) -> dict:
    default = Path(__file__).parents[2] / "config" / "retrieval.json"
    return json.loads((path or default).read_text())


def build_wikipedia_index(
    documents: Iterable[dict[str, str]],
    output_directory: Path,
    *,
    dense_encoder: DenseEncoder | None = None,
    retrieval_config: Path | None = None,
    embedding_batch_size: int = 256,
) -> int:
    """Stream chunks into Tantivy and optional FAISS HNSW indexes."""
    from .search import chunk_document

    if output_directory.exists():
        raise FileExistsError(f"index target already exists: {output_directory}")
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")
    config = _load_retrieval_config(retrieval_config)
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
    pending_texts: list[str] = []
    pending_ids: list[int] = []
    dense_index = None
    count = 0
    metadata_closed = False

    def flush_dense() -> None:
        nonlocal dense_index
        if dense_encoder is None or not pending_texts:
            return
        embeddings = dense_encoder.encode_documents(pending_texts)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(pending_texts):
            raise ValueError("dense encoder returned an invalid document matrix")
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if dense_index is None:
            dense_index = VectorIndex(
                ndim=embeddings.shape[1],
                metric="cos",
                dtype="f32",
                connectivity=int(config["hnsw_connections"]),
                expansion_add=int(config["hnsw_construction_depth"]),
            )
        dense_index.add(np.asarray(pending_ids, dtype=np.uint64), embeddings)
        pending_texts.clear()
        pending_ids.clear()

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
                pending_ids.append(row_id)
                pending_texts.append(f"{document['title']}\n{content}")
                count += 1
                if len(pending_texts) >= embedding_batch_size:
                    flush_dense()
                if count % 50_000 == 0:
                    metadata.commit()
        flush_dense()
        if count == 0:
            raise ValueError("cannot build an empty search index")
        writer.commit()
        writer.wait_merging_threads()
        metadata.commit()
        metadata.close()
        metadata_closed = True
        if dense_index is not None:
            dense_index.save(temporary / "dense.usearch")
        manifest = {
            "format_version": 2,
            "chunks": count,
            "lexical": "tantivy-bm25",
            "dense": dense_index is not None,
            "dense_backend": "usearch-hnsw" if dense_index is not None else None,
            "embedding_model": config["embedding_model"] if dense_index is not None else None,
            "reranker_model": config["reranker_model"],
            "candidate_multiplier": config["candidate_multiplier"],
            "rrf_constant": config["rrf_constant"],
        }
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
    def __init__(
        self,
        index_directory: Path,
        *,
        dense_encoder: DenseEncoder | None = None,
        reranker=None,
        enable_reranker: bool = False,
        model_device: str | None = None,
    ) -> None:
        self.index = tantivy.Index.open(str(index_directory / "lexical"))
        self.index.reload()
        self.metadata = sqlite3.connect(
            index_directory / "metadata.sqlite3", check_same_thread=False
        )
        self.manifest = json.loads((index_directory / "retrieval-manifest.json").read_text())
        dense_path = index_directory / "dense.usearch"
        self.dense_index = VectorIndex.restore(dense_path) if dense_path.exists() else None
        if self.dense_index is not None and dense_encoder is None:
            model = self.manifest["embedding_model"]
            dense_encoder = SentenceTransformerEncoder(
                model["name"], model["revision"], model_device
            )
        self.dense_encoder = dense_encoder
        if enable_reranker and reranker is None:
            model = self.manifest["reranker_model"]
            reranker = CrossEncoderReranker(model["name"], model["revision"], model_device)
        self.reranker = reranker

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

    def _lexical(self, query: str, limit: int) -> list[tuple[int, float]]:
        parsed, _errors = self.index.parse_query_lenient(
            query,
            default_field_names=["title", "content"],
            field_boosts={"title": 2.0},
            conjunction_by_default=False,
        )
        searcher = self.index.searcher()
        hits = searcher.search(parsed, limit=limit).hits
        results = []
        for score, address in hits:
            document = searcher.doc(address)
            results.append((int(document.get_first("row_id")), float(score)))
        return results

    def _dense(self, query: str, limit: int) -> list[tuple[int, float]]:
        if self.dense_index is None or self.dense_encoder is None:
            return []
        embedding = np.ascontiguousarray(self.dense_encoder.encode_query(query), dtype=np.float32)
        if embedding.ndim == 2:
            if embedding.shape[0] != 1:
                raise ValueError("dense query encoder must return one embedding")
            embedding = embedding[0]
        matches = self.dense_index.search(embedding, count=limit)
        return [
            (int(identifier), float(1.0 - distance))
            for identifier, distance in zip(matches.keys, matches.distances, strict=True)
        ]

    def _search_sync(self, query: str, top_k: int) -> list[SearchResult]:
        from .search import reciprocal_rank_fusion, sanitize_retrieved_text

        candidate_limit = max(top_k, top_k * int(self.manifest["candidate_multiplier"]))
        lexical = self._lexical(query, candidate_limit)
        dense = self._dense(query, candidate_limit)
        rankings = [[str(row_id) for row_id, _ in lexical]]
        if dense:
            rankings.append([str(row_id) for row_id, _ in dense])
        fused = reciprocal_rank_fusion(rankings, int(self.manifest["rrf_constant"]))
        ordered_ids = [
            int(item[0]) for item in sorted(fused.items(), key=lambda item: item[1], reverse=True)
        ]
        metadata = self._metadata_rows(ordered_ids[:candidate_limit])
        lexical_scores = dict(lexical)
        dense_scores = dict(dense)
        candidates = []
        for row_id in ordered_ids[:candidate_limit]:
            chunk_id, title, url, content = metadata[row_id]
            candidates.append(
                SearchResult(
                    id=chunk_id,
                    title=sanitize_retrieved_text(title, 500),
                    url=url,
                    snippet=sanitize_retrieved_text(content, 500),
                    content=sanitize_retrieved_text(content),
                    score=fused[str(row_id)],
                    provider="local-wikipedia",
                    score_components={
                        "rrf": fused[str(row_id)],
                        "bm25": lexical_scores.get(row_id, 0.0),
                        "dense": dense_scores.get(row_id, 0.0),
                    },
                )
            )
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        return await asyncio.to_thread(self._search_sync, query, top_k)

    def close(self) -> None:
        self.metadata.close()
