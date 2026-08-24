import asyncio
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

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
    WebSearchUnavailable,
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


def test_bm25_index_has_a_small_manifest_and_no_dense_artifacts(tmp_path: Path) -> None:
    index = tmp_path / "index"
    assert build_wikipedia_index(DOCUMENTS, index) == 2

    manifest = json.loads((index / "retrieval-manifest.json").read_text())
    assert manifest == {"chunks": 2, "format_version": 3, "lexical": "tantivy-bm25"}
    assert (index / "lexical").is_dir()
    assert (index / "metadata.sqlite3").is_file()
    assert not (index / "dense.usearch").exists()
    assert not (index / "dense").exists()


def test_bm25_index_build_failure_is_atomic_and_streams_documents(tmp_path: Path) -> None:
    consumed: list[str] = []

    def stream():
        for document in DOCUMENTS:
            consumed.append(document["id"])
            yield document
            if document["id"] == "mars":
                raise RuntimeError("source failed")

    failed = tmp_path / "failed-index"
    with pytest.raises(RuntimeError, match="source failed"):
        build_wikipedia_index(stream(), failed)
    assert consumed == ["mars"]
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed-index-*"))


def test_retrieval_module_has_no_dense_retrieval_dependencies_or_symbols() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src/gpt2_reasoning_search/retrieval.py"
    ).read_text()
    for forbidden in (
        "usearch",
        "sentence_transformers",
        "SentenceTransformer",
        "VectorIndex",
        "dense_encoder",
        "reranker",
        "reciprocal_rank_fusion",
    ):
        assert forbidden not in source


def test_index_build_creates_missing_output_parents(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "artifacts" / "wiki-index"
    assert build_wikipedia_index([DOCUMENTS[0]], output) == 1
    assert (output / "retrieval-manifest.json").is_file()


def test_bm25_title_boost_async_and_close(tmp_path: Path) -> None:
    index = tmp_path / "index"
    build_wikipedia_index(DOCUMENTS, index)

    lexical = LocalWikipediaSearchProvider(index)
    title_results = asyncio.run(lexical.search("Mars Mission", top_k=2))
    assert title_results[0].id == "mars:0"
    assert title_results[0].score_components["bm25"] > 0

    original = lexical._search_sync

    def slow_search(query, top_k):
        time.sleep(0.05)
        return original(query, top_k)

    lexical._search_sync = slow_search

    async def prove_nonblocking():
        search_task = asyncio.create_task(lexical.search("Mars", 1))
        await asyncio.sleep(0.005)
        assert not search_task.done()
        return await search_task

    assert asyncio.run(prove_nonblocking())
    lexical.close()
    with pytest.raises(sqlite3.ProgrammingError):
        lexical._metadata_rows([0])


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


def test_brave_cache_is_checked_before_unavailable_latch(tmp_path: Path) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("a cached query must not call Brave")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cache = SQLiteSearchCache(tmp_path / "cache.sqlite3")
    cached = _result("cached")
    cache.put(json.dumps(["brave", "test query", 2]), [cached])
    provider = BraveWebSearchProvider(
        "secret", client=client, cache=cache, page_fetcher=FakePageFetcher()
    )
    provider.unavailable_reason = "Brave is unavailable"

    results = asyncio.run(provider.search(" Test   Query ", top_k=2))

    assert results == [cached]
    assert requests == []
    asyncio.run(provider.aclose())
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
    attempts = []

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gpt2_reasoning_search.web_search.asyncio.sleep", no_sleep)
    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = BraveWebSearchProvider(
        "key", client=client, page_fetcher=FakePageFetcher(), retries=1
    )
    with pytest.raises(WebSearchUnavailable, match="after retries"):
        asyncio.run(provider.search("query"))
    with pytest.raises(WebSearchUnavailable, match="disabled for this run"):
        asyncio.run(provider.search("another query"))
    assert len(attempts) == 1
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
