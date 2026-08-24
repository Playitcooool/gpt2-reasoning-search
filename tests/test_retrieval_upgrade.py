import asyncio
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from typer.testing import CliRunner

from gpt2_reasoning_search.cli import app
from gpt2_reasoning_search.retrieval import LocalWikipediaSearchProvider, build_wikipedia_index
from gpt2_reasoning_search.schemas import SearchResult
from gpt2_reasoning_search.search import (
    canonicalize_url,
    chunk_document,
    reciprocal_rank_fusion,
    stable_source_id,
)
from gpt2_reasoning_search.web_search import (
    BraveWebSearchProvider,
    SQLiteSearchCache,
    WebPageFetcher,
    validate_public_url,
)

DOCUMENTS = [
    {
        "id": "mars",
        "title": "Mars Mission",
        "url": "https://example.org/mars",
        "text": "Mars is the red planet and hosts Olympus Mons.",
    },
    {
        "id": "ocean",
        "title": "Ocean",
        "url": "https://example.org/ocean",
        "text": "The Pacific is Earth's largest ocean. Mars appears only in this sentence.",
    },
]


def test_web_search_module_supports_direct_import() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from gpt2_reasoning_search.web_search import BraveWebSearchProvider",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


class FakeDenseEncoder:
    def encode_documents(self, texts):
        return np.asarray(
            [[1.0, 0.0] if "Olympus" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    def encode_query(self, query):
        return np.asarray([[1.0, 0.0] if "volcano" in query else [0.0, 1.0]], dtype=np.float32)


def _result(identifier: str, url: str = "https://example.org/a") -> SearchResult:
    return SearchResult(
        id=identifier,
        title=identifier,
        url=url,
        snippet="snippet",
        content="content",
        provider="test",
    )


def test_sentence_aware_chunking_preserves_sentence_boundaries_and_splits_long_units() -> None:
    chunks = chunk_document(
        "Alpha is first. Beta is second. Gamma is third.",
        chunk_characters=31,
        overlap_characters=0,
    )
    assert chunks == ["Alpha is first. Beta is second.", "Gamma is third."]
    long_chunks = chunk_document("x" * 25, chunk_characters=10, overlap_characters=2)
    assert long_chunks == ["x" * 10, "x" * 10, "x" * 9]


def test_url_canonicalization_and_stable_source_ids() -> None:
    variants = [
        "HTTPS://Example.COM:443//docs/?b=2&utm_source=x&a=1#fragment",
        "https://example.com/docs/?a=1&b=2",
    ]
    assert [canonicalize_url(url) for url in variants] == [
        "https://example.com/docs/?a=1&b=2",
        "https://example.com/docs/?a=1&b=2",
    ]
    assert stable_source_id("brave", variants[0]) == stable_source_id("brave", variants[1])
    assert stable_source_id("other", variants[0]) != stable_source_id("brave", variants[0])


def test_reciprocal_rank_fusion_combines_rankings_and_validates_constant() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], constant=10)
    assert scores["b"] > scores["a"] > scores["c"]
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], constant=0)


def test_index_build_streams_batches_writes_manifest_and_cleans_failure_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    class RecordingEncoder(FakeDenseEncoder):
        def __init__(self):
            self.batch_sizes = []

        def encode_documents(self, texts):
            self.batch_sizes.append(len(texts))
            return super().encode_documents(texts)

    encoder = RecordingEncoder()
    index = tmp_path / "index"
    consumed = []

    class FakeHNSW:
        def __init__(self, **settings):
            self.settings = settings
            self.rows = []

        def add(self, keys, rows):
            self.rows.extend(zip(keys.tolist(), rows.tolist(), strict=True))

        def save(self, path):
            Path(path).write_bytes(b"fake-usearch")

    monkeypatch.setattr("gpt2_reasoning_search.retrieval.VectorIndex", FakeHNSW)

    def stream():
        for document in DOCUMENTS:
            consumed.append(document["id"])
            yield document

    count = build_wikipedia_index(stream(), index, dense_encoder=encoder, embedding_batch_size=1)
    assert count == 2
    assert consumed == ["mars", "ocean"]
    assert encoder.batch_sizes == [1, 1]
    manifest = json.loads((index / "retrieval-manifest.json").read_text())
    assert manifest == {
        "candidate_multiplier": 10,
        "chunks": 2,
        "dense": True,
        "dense_backend": "usearch-hnsw",
        "embedding_model": {
            "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "revision": "b207367332321f8e44f96e224ef15bc607f4dbf0",
        },
        "format_version": 2,
        "lexical": "tantivy-bm25",
        "reranker_model": {
            "name": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        },
        "rrf_constant": 60,
    }
    assert (index / "dense.usearch").is_file()

    class BrokenEncoder(FakeDenseEncoder):
        def encode_documents(self, texts):
            raise RuntimeError("encoder failed")

    failed = tmp_path / "failed-index"
    with pytest.raises(RuntimeError, match="encoder failed"):
        build_wikipedia_index(DOCUMENTS, failed, dense_encoder=BrokenEncoder())
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed-index-*"))


def test_index_build_creates_missing_output_parents(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "artifacts" / "wiki-index"
    assert build_wikipedia_index([DOCUMENTS[0]], output) == 1
    assert (output / "retrieval-manifest.json").is_file()


def test_real_usearch_hnsw_and_hybrid_fusion_in_clean_process(tmp_path: Path) -> None:
    script = r"""
import asyncio, json, sys
from pathlib import Path
import numpy as np
from gpt2_reasoning_search.retrieval import LocalWikipediaSearchProvider, build_wikipedia_index
class Encoder:
    def encode_documents(self, texts):
        rows = [[1., 0.] if "Olympus" in x else [0., 1.] for x in texts]
        return np.asarray(rows, dtype=np.float32)
    def encode_query(self, query):
        return np.asarray([[1., 0.] if "volcano" in query else [0., 1.]], dtype=np.float32)
docs = [
 {"id":"mars","title":"Mars Mission","url":"https://example.org/mars",
  "text":"Mars is red and hosts Olympus Mons."},
 {"id":"ocean","title":"Ocean","url":"https://example.org/ocean",
  "text":"The Pacific is Earth's largest ocean. Mars appears here."},
]
target = Path(sys.argv[1])
build_wikipedia_index(docs, target, dense_encoder=Encoder(), embedding_batch_size=1)
provider = LocalWikipediaSearchProvider(target, dense_encoder=Encoder())
results = asyncio.run(provider.search("volcano", 2))
print(json.dumps({"ids":[r.id for r in results], "scores":[r.score_components for r in results]}))
provider.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "real-index")],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["ids"][0] == "mars:0"
    assert payload["scores"][0]["dense"] > 0
    assert payload["scores"][0]["rrf"] > 0


def test_lexical_title_boost_reranking_async_and_close(tmp_path: Path) -> None:
    index = tmp_path / "index"
    build_wikipedia_index(DOCUMENTS, index)

    lexical = LocalWikipediaSearchProvider(index)
    title_results = asyncio.run(lexical.search("Mars Mission", top_k=2))
    assert title_results[0].id == "mars:0"
    assert title_results[0].score_components["bm25"] > 0
    lexical.close()
    with pytest.raises(sqlite3.ProgrammingError):
        lexical._metadata_rows([0])

    class ReverseReranker:
        def rerank(self, query, candidates, top_k):
            return list(reversed(candidates))[:top_k]

    reranked = LocalWikipediaSearchProvider(index, reranker=ReverseReranker())
    baseline = reranked._search_sync("Mars", 2)
    assert baseline[0].id == "ocean:0"

    original = reranked._search_sync

    def slow_search(query, top_k):
        time.sleep(0.05)
        return original(query, top_k)

    reranked._search_sync = slow_search

    async def prove_nonblocking():
        search_task = asyncio.create_task(reranked.search("Mars", 1))
        await asyncio.sleep(0.005)
        assert not search_task.done()
        return await search_task

    assert asyncio.run(prove_nonblocking())
    reranked.close()


def test_provider_disabled_dense_retrieval_does_not_restore_or_load_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    index = tmp_path / "index"
    build_wikipedia_index(
        DOCUMENTS, index, dense_encoder=FakeDenseEncoder(), embedding_batch_size=1
    )
    assert (index / "dense.usearch").is_file()

    class FailingVectorIndex:
        @staticmethod
        def restore(_path):
            raise AssertionError("dense index should not be restored")

    class FailingEncoder:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dense encoder should not be loaded")

    monkeypatch.setattr("gpt2_reasoning_search.retrieval.VectorIndex", FailingVectorIndex)
    monkeypatch.setattr(
        "gpt2_reasoning_search.retrieval.SentenceTransformerEncoder", FailingEncoder
    )

    provider = LocalWikipediaSearchProvider(index, enable_dense=False)
    assert provider.dense_index is None
    assert provider.dense_encoder is None
    provider.close()


def test_sqlite_cache_hit_expiry_and_close(tmp_path: Path, monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("gpt2_reasoning_search.web_search.time.time", lambda: clock[0])
    cache = SQLiteSearchCache(tmp_path / "cache.sqlite3", ttl_seconds=10)
    cache.put("key", [_result("a")])
    assert cache.get("key") == [_result("a")]
    clock[0] = 110.0
    assert cache.get("key") is None
    cache.close()
    with pytest.raises(sqlite3.ProgrammingError):
        cache.get("key")


class FakePageFetcher:
    def __init__(self):
        self.calls = 0

    async def enrich(self, results):
        self.calls += 1
        return [result.model_copy(update={"content": f"full:{result.id}"}) for result in results]


def test_brave_deduplicates_stable_ids_caches_and_closes(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "One",
                            "url": "https://EXAMPLE.org/a?utm_source=x",
                            "description": "a",
                        },
                        {"title": "Duplicate", "url": "https://example.org/a", "description": "b"},
                        {"title": "Two", "url": "https://example.org/b", "description": "c"},
                    ]
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cache = SQLiteSearchCache(tmp_path / "cache.sqlite3")
    fetcher = FakePageFetcher()
    provider = BraveWebSearchProvider("secret", client=client, cache=cache, page_fetcher=fetcher)

    first = asyncio.run(provider.search("  Test   Query ", top_k=2))
    second = asyncio.run(provider.search("test query", top_k=2))
    assert len(calls) == 1 and fetcher.calls == 1
    assert first == second and len(first) == 2
    assert first[0].url == "https://example.org/a"
    assert first[0].id == stable_source_id("brave", first[0].url)
    assert first[0].content.startswith("full:")
    asyncio.run(provider.aclose())
    assert not client.is_closed
    with pytest.raises(sqlite3.ProgrammingError):
        cache.get("x")
    asyncio.run(client.aclose())


@pytest.mark.parametrize("statuses", [[429, 200], [503, 200]])
def test_brave_retries_rate_limits_and_server_errors(monkeypatch, statuses) -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses[len(attempts)]
        attempts.append(status)
        return httpx.Response(status, headers={"retry-after": "0"}, json={"web": {"results": []}})

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gpt2_reasoning_search.web_search.asyncio.sleep", no_sleep)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = BraveWebSearchProvider(
        "key", client=client, page_fetcher=FakePageFetcher(), retries=2
    )
    assert asyncio.run(provider.search("query")) == []
    assert attempts == statuses
    asyncio.run(client.aclose())


def test_brave_exhausted_retries_raises_runtime_error(monkeypatch) -> None:
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gpt2_reasoning_search.web_search.asyncio.sleep", no_sleep)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(503)))
    provider = BraveWebSearchProvider(
        "key", client=client, page_fetcher=FakePageFetcher(), retries=1
    )
    with pytest.raises(RuntimeError, match="after retries"):
        asyncio.run(provider.search("query"))
    asyncio.run(client.aclose())


async def public_resolver(_hostname: str):
    return ["93.184.216.34"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/admin",
        "http://10.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "https://user:password@example.org/secret",
    ],
)
def test_url_validation_rejects_ssrf_private_and_credentials(url: str) -> None:
    async def resolver(hostname: str):
        if hostname == "example.org":
            return ["93.184.216.34"]
        return [hostname]

    with pytest.raises(ValueError):
        asyncio.run(validate_public_url(url, resolver))


def test_url_validation_rejects_dns_rebinding_candidates() -> None:
    async def resolver(_hostname: str):
        return ["93.184.216.34", "192.168.1.1"]

    with pytest.raises(ValueError, match="private"):
        asyncio.run(validate_public_url("https://example.org", resolver))


def test_web_fetcher_html_extraction_plaintext_content_type_size_and_fallback(monkeypatch) -> None:
    routes = {
        "/page": (200, {"content-type": "text/html"}, b"<html><main>Article body</main></html>"),
        "/plain": (200, {"content-type": "text/plain"}, b"plain body"),
        "/image": (200, {"content-type": "image/png"}, b"image"),
        "/large": (200, {"content-type": "text/plain"}, b"0123456789"),
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                routes[request.url.path][0],
                headers=routes[request.url.path][1],
                content=routes[request.url.path][2],
            )
        )
    )
    fetcher = WebPageFetcher(client, resolver=public_resolver, respect_robots=False, max_bytes=5)
    monkeypatch.setattr(
        "gpt2_reasoning_search.web_search.trafilatura.extract",
        lambda *_a, **_k: "Article body",
    )
    assert asyncio.run(fetcher.fetch("https://example.org/page")) is None  # raw page exceeds cap
    fetcher.max_bytes = 1_000
    assert asyncio.run(fetcher.fetch("https://example.org/page")) == "Article body"
    assert asyncio.run(fetcher.fetch("https://example.org/plain")) == "plain body"
    assert asyncio.run(fetcher.fetch("https://example.org/image")) is None
    fetcher.max_bytes = 5
    assert asyncio.run(fetcher.fetch("https://example.org/large")) is None

    monkeypatch.setattr(
        "gpt2_reasoning_search.web_search.trafilatura.extract", lambda *_a, **_k: None
    )
    original = _result("original", "https://example.org/page")
    assert asyncio.run(fetcher.enrich([original]))[0].content == original.content
    asyncio.run(client.aclose())


def test_web_fetcher_robots_allow_and_deny() -> None:
    fetched_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text="User-agent: *\nDisallow: /private\nAllow: /public\n",
            )
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="body")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = WebPageFetcher(client, resolver=public_resolver)
    assert asyncio.run(fetcher.fetch("https://example.org/private")) is None
    assert asyncio.run(fetcher.fetch("https://example.org/public")) == "body"
    assert fetched_paths == ["/robots.txt", "/public"]
    asyncio.run(client.aclose())


def test_robots_redirect_to_private_address_is_not_fetched() -> None:
    fetched_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched_urls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/robots.txt"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="body")

    async def resolver(hostname: str):
        return ["127.0.0.1"] if hostname == "127.0.0.1" else ["93.184.216.34"]

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = WebPageFetcher(client, resolver=resolver)
    assert asyncio.run(fetcher.fetch("https://example.org/public")) == "body"
    assert all("127.0.0.1" not in fetched for fetched in fetched_urls)
    asyncio.run(client.aclose())


def test_web_fetcher_concurrency_bound() -> None:
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="body")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = WebPageFetcher(
        client, resolver=public_resolver, respect_robots=False, max_concurrency=2
    )
    results = [_result(str(index), f"https://example.org/{index}") for index in range(5)]
    asyncio.run(fetcher.enrich(results))
    assert peak == 2
    asyncio.run(client.aclose())


def test_build_index_cli_lexical_only_and_hybrid_do_not_download_when_mocked(
    tmp_path: Path, monkeypatch
) -> None:
    documents = tmp_path / "docs.jsonl"
    documents.write_text(json.dumps(DOCUMENTS[0]) + "\n")
    config = tmp_path / "retrieval.json"
    config.write_text(
        json.dumps(
            {
                "embedding_model": {"name": "pinned", "revision": "revision"},
                "reranker_model": {"name": "reranker", "revision": "revision"},
            }
        )
    )
    calls = []

    class FakeEncoder:
        def __init__(self, *args):
            calls.append(("encoder", args))

    def fake_build(documents, output, *, dense_encoder, retrieval_config):
        calls.append(("build", list(documents), output, dense_encoder, retrieval_config))
        return 1

    monkeypatch.setattr("gpt2_reasoning_search.cli.SentenceTransformerEncoder", FakeEncoder)
    monkeypatch.setattr("gpt2_reasoning_search.cli.build_wikipedia_index", fake_build)
    runner = CliRunner()

    lexical = runner.invoke(
        app,
        [
            "build-index",
            str(documents),
            "--output",
            str(tmp_path / "lexical"),
            "--lexical-only",
            "--retrieval-config",
            str(config),
        ],
    )
    assert lexical.exit_code == 0
    assert calls[-1][3] is None

    hybrid = runner.invoke(
        app,
        [
            "build-index",
            str(documents),
            "--output",
            str(tmp_path / "hybrid"),
            "--no-lexical-only",
            "--retrieval-config",
            str(config),
            "--embedding-device",
            "cpu",
        ],
    )
    assert hybrid.exit_code == 0
    assert calls[-2] == ("encoder", ("pinned", "revision", "cpu"))
    assert isinstance(calls[-1][3], FakeEncoder)
