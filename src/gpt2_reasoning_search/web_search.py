"""Resilient live web search with extraction, caching, and network safety controls."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import sqlite3
import threading
import time
import urllib.robotparser
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura

from .schemas import SearchResult
from .search import canonicalize_url, sanitize_retrieved_text, stable_source_id

Resolver = Callable[[str], Awaitable[Sequence[str]]]


class SQLiteSearchCache:
    def __init__(self, path: Path, ttl_seconds: int = 3_600) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, expires REAL NOT NULL, payload TEXT NOT NULL)"
        )
        self.connection.commit()
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()

    def get(self, key: str) -> list[SearchResult] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT expires, payload FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if float(row[0]) <= time.time():
                self.connection.execute("DELETE FROM cache WHERE key = ?", (key,))
                self.connection.commit()
                return None
            return [SearchResult.model_validate(item) for item in json.loads(row[1])]

    def put(self, key: str, results: Sequence[SearchResult]) -> None:
        payload = json.dumps([result.model_dump() for result in results], ensure_ascii=False)
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                (key, time.time() + self.ttl_seconds, payload),
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()


async def _default_resolver(hostname: str) -> Sequence[str]:
    rows = await asyncio.to_thread(socket.getaddrinfo, hostname, None, type=socket.SOCK_STREAM)
    return list({str(row[4][0]) for row in rows})


async def validate_public_url(url: str, resolver: Resolver = _default_resolver) -> str:
    raw = urlsplit(url)
    if raw.username or raw.password:
        raise ValueError("credential-bearing URLs cannot be fetched")
    canonical = canonicalize_url(url)
    parsed = urlsplit(canonical)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only public HTTP(S) URLs can be fetched")
    addresses = await resolver(parsed.hostname)
    if not addresses:
        raise ValueError("hostname did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private, loopback, and reserved addresses cannot be fetched")
    return canonical


class WebPageFetcher:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        resolver: Resolver = _default_resolver,
        max_concurrency: int = 5,
        max_bytes: int = 2_000_000,
        user_agent: str = "GPT2ReasoningSearchBot/0.2",
        respect_robots: bool = True,
    ) -> None:
        self.client = client
        self.resolver = resolver
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    async def _robots_allows(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self.robots.get(origin)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser(urljoin(origin, "/robots.txt"))
            try:
                robots_url = parser.url
                for _ in range(6):
                    safe_url = await validate_public_url(robots_url, self.resolver)
                    response = await self.client.get(
                        safe_url,
                        headers={"User-Agent": self.user_agent},
                        follow_redirects=False,
                    )
                    if not response.is_redirect:
                        parser.parse(
                            response.text.splitlines() if response.status_code == 200 else []
                        )
                        break
                    location = response.headers.get("location")
                    if not location:
                        parser.parse([])
                        break
                    robots_url = urljoin(safe_url, location)
                else:
                    parser.parse([])
            except (httpx.HTTPError, ValueError):
                parser.parse([])
            self.robots[origin] = parser
        return parser.can_fetch(self.user_agent, url)

    async def fetch(self, url: str) -> str | None:
        async with self.semaphore:
            current_url = url
            for _ in range(6):
                try:
                    safe_url = await validate_public_url(current_url, self.resolver)
                except ValueError:
                    return None
                if not await self._robots_allows(safe_url):
                    return None
                try:
                    async with self.client.stream(
                        "GET",
                        safe_url,
                        headers={
                            "User-Agent": self.user_agent,
                            "Accept": "text/html,text/plain",
                        },
                        follow_redirects=False,
                    ) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                return None
                            current_url = urljoin(safe_url, location)
                            continue
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").lower()
                        if "text/html" not in content_type and "text/plain" not in content_type:
                            return None
                        payload = bytearray()
                        async for chunk in response.aiter_bytes():
                            payload.extend(chunk)
                            if len(payload) > self.max_bytes:
                                return None
                        encoding = response.encoding or "utf-8"
                    text = payload.decode(encoding, errors="replace")
                    break
                except (httpx.HTTPError, UnicodeError):
                    return None
            else:
                return None
        if "text/html" in content_type:
            extracted = trafilatura.extract(
                text, include_comments=False, include_tables=True, no_fallback=False
            )
            text = extracted or ""
        return sanitize_retrieved_text(text) if text.strip() else None

    async def enrich(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        contents = await asyncio.gather(*(self.fetch(result.url) for result in results))
        return [
            result.model_copy(update={"content": content or result.content})
            for result, content in zip(results, contents, strict=True)
        ]


class BraveWebSearchProvider:
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 15.0,
        *,
        client: httpx.AsyncClient | None = None,
        cache: SQLiteSearchCache | None = None,
        page_fetcher: WebPageFetcher | None = None,
        retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("Brave API key is required")
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self.owns_client = client is None
        self.cache = cache
        self.page_fetcher = page_fetcher or WebPageFetcher(self.client)
        self.retries = retries

    async def _request(self, query: str, count: int) -> dict:
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = await self.client.get(
                    self.endpoint,
                    params={"q": query, "count": count, "safesearch": "moderate"},
                    headers=headers,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    delay = min(4.0, float(response.headers.get("retry-after", 2**attempt)))
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    await asyncio.sleep(min(4.0, 2**attempt))
        raise RuntimeError("web search failed after retries") from last_error

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            return []
        cache_key = json.dumps(["brave", normalized_query.lower(), top_k])
        if self.cache and (cached := self.cache.get(cache_key)) is not None:
            return cached
        payload = await self._request(normalized_query, min(20, max(top_k * 2, top_k)))
        rows = payload.get("web", {}).get("results", [])
        results = []
        seen_urls = set()
        for position, row in enumerate(rows):
            try:
                url = canonicalize_url(str(row.get("url", "")))
                parsed = urlsplit(url)
            except ValueError:
                continue
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or url in seen_urls:
                continue
            seen_urls.add(url)
            description = sanitize_retrieved_text(str(row.get("description", "")), 1_000)
            results.append(
                SearchResult(
                    id=stable_source_id("brave", url),
                    title=sanitize_retrieved_text(str(row.get("title", "")), 500),
                    url=url,
                    snippet=description,
                    content=description,
                    score=float(len(rows) - position),
                    provider="brave",
                    published_at=row.get("age"),
                    score_components={"provider_rank": float(position + 1)},
                )
            )
            if len(results) >= top_k:
                break
        enriched = await self.page_fetcher.enrich(results)
        if self.cache:
            self.cache.put(cache_key, enriched)
        return enriched

    async def aclose(self) -> None:
        if self.owns_client:
            await self.client.aclose()
        if self.cache:
            self.cache.close()
