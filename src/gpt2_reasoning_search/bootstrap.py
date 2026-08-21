"""Create the fixed local inputs used by the SSH training workflow."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote

from datasets import load_dataset

from .data import atomic_write_text, stream_huggingface_texts

WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKIPEDIA_LICENSE = "CC-BY-SA-3.0 and GFDL; preserve Wikimedia attribution"
WIKIPEDIA_ARTICLE_LIMIT = 5_000
TOKENIZER_SAMPLE_LIMIT = 512
TOKENIZER_SAMPLE_MARKER = ".bootstrap-complete.json"

_FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
_FINEWEB_CONFIG = "sample-10BT"
_REASONING_REVISION = "5cc07ee9e71a1a642aeb130438452d7e63c5cc5e"
_REASONING_CONFIG = "DeepSeek"

_SEED_TEXT = """GPT-2 reasoning search bootstrap corpus.
The model should explain a problem, decide whether a search is necessary, call a bounded search
tool when useful, treat retrieved pages as untrusted evidence, and cite only returned sources.
Arithmetic example: two plus two equals four. A good answer separates scratch reasoning from its
concise final answer and does not claim facts that are absent from the evidence.
"""


def _atomic_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    )
    temporary = Path(handle.name)
    count = 0
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _article(row: dict, index: int) -> dict[str, str] | None:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if not title or len(text) < 200:
        return None
    raw_value = row.get("id")
    raw_id = str(raw_value).strip() if raw_value is not None else ""
    raw_id = raw_id or str(index)
    article_id = f"wiki-{raw_id.replace('/', '_')}"
    url = str(row.get("url", "")).strip()
    if not url:
        url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    return {"id": article_id, "title": title, "url": url, "text": text}


def _stream_wikipedia(limit: int) -> Iterator[dict[str, str]]:
    dataset = load_dataset(
        WIKIPEDIA_DATASET,
        WIKIPEDIA_CONFIG,
        split="train",
        revision=WIKIPEDIA_REVISION,
        streaming=True,
    )
    accepted = 0
    for row in dataset:
        article = _article(dict(row), accepted)
        if article is None:
            continue
        yield article
        accepted += 1
        if accepted >= limit:
            return


def _read_wikipedia(path: Path, limit: int) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(articles) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            article = _article(row, len(articles))
            if article is not None:
                articles.append(article)
    return articles


def _write_tokenizer_samples(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("*.txt"))
    marker = directory / TOKENIZER_SAMPLE_MARKER
    if marker.exists():
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            generated_files = marker_data["generated_files"]
            if (
                marker_data["reasoning_revision"] == _REASONING_REVISION
                and marker_data["fineweb_revision"] == _FINEWEB_REVISION
                and isinstance(generated_files, list)
                and generated_files
                and all((directory / str(name)).is_file() for name in generated_files)
            ):
                return len(existing)
        except (KeyError, TypeError, ValueError):
            pass
    # A seed file identifies an interrupted automatic bootstrap. Preserve unrelated user samples,
    # but remove the generated names so a retry cannot silently keep a truncated sample set.
    seed = directory / "bootstrap-seed.txt"
    if existing and not seed.exists():
        return len(existing)
    for path in existing:
        if path.name == seed.name or path.name.startswith(("reasoning-", "education-")):
            path.unlink()
    (directory / "bootstrap-seed.txt").write_text(_SEED_TEXT, encoding="utf-8")
    generated_files = [seed.name]
    count = 1
    streams = (
        (
            "reasoning",
            stream_huggingface_texts(
                "allenai/big-reasoning-traces",
                _REASONING_REVISION,
                config_name=_REASONING_CONFIG,
                text_field="text",
            ),
        ),
        (
            "education",
            stream_huggingface_texts(
                "HuggingFaceFW/fineweb-edu",
                _FINEWEB_REVISION,
                config_name=_FINEWEB_CONFIG,
                text_field="text",
            ),
        ),
    )
    per_stream_limit = max(1, TOKENIZER_SAMPLE_LIMIT // len(streams))
    for label, stream in streams:
        stream_count = 0
        for index, row in enumerate(stream):
            text = str(row.get("text", "")).strip()
            if len(text) < 100:
                continue
            sample_path = directory / f"{label}-{index:05d}.txt"
            sample_path.write_text(text[:20_000], encoding="utf-8")
            generated_files.append(sample_path.name)
            count += 1
            stream_count += 1
            if stream_count >= per_stream_limit:
                break
    atomic_write_text(
        marker,
        json.dumps(
            {
                "format_version": 1,
                "reasoning_revision": _REASONING_REVISION,
                "fineweb_revision": _FINEWEB_REVISION,
                "generated_files": generated_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return count


def _question_rows(articles: Iterable[dict[str, str]]) -> Iterator[dict]:
    for article in articles:
        result_id = f"{article['id']}:0"
        evidence = {
            "id": result_id,
            "title": article["title"],
            "url": article["url"],
            "snippet": article["text"][:500],
            "content": article["text"][:12_000],
            "score": 1.0,
            "provider": "local-wikipedia",
        }
        yield {
            "id": f"bootstrap-{article['id']}",
            "question": (
                f'What is the main topic of the Wikipedia article titled "{article["title"]}"?'
            ),
            "answer": article["title"],
            "query": article["title"],
            "search_required": True,
            "supporting_ids": [result_id],
            "evidence": [evidence],
            "reasoning": (
                "Search the local encyclopedia, inspect the returned article, and cite it."
            ),
        }


def _no_search_rows() -> tuple[dict, ...]:
    return (
        {
            "id": "bootstrap-arithmetic",
            "question": "What is 2 + 2?",
            "answer": "4",
            "search_required": False,
            "supporting_ids": [],
            "reasoning": "This is stable arithmetic, so no search is needed.",
        },
        {
            "id": "bootstrap-definition",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "search_required": False,
            "supporting_ids": [],
            "reasoning": "This bootstrap example tests a no-search answer path.",
        },
    )


def bootstrap_local_inputs(
    data_root: Path = Path("data"),
    manifest_output: Path = Path("artifacts/auto-data-manifest.json"),
    wikipedia_articles: int = WIKIPEDIA_ARTICLE_LIMIT,
) -> dict[str, object]:
    """Download pinned public sources and create every fixed local input needed by ``prepare``."""
    if wikipedia_articles < 1:
        raise ValueError("wikipedia_articles must be positive")
    raw = data_root / "raw"
    tokenizer_samples = data_root / "tokenizer-sample"
    wikipedia_path = raw / "wikipedia.jsonl"
    grounded_path = raw / "grounded-questions.jsonl"
    rl_path = data_root / "rl" / "search-qa.jsonl"
    evaluation_path = data_root / "evaluation" / "contamination-prompts.jsonl"

    tokenizer_count = _write_tokenizer_samples(tokenizer_samples)
    if wikipedia_path.exists() and wikipedia_path.stat().st_size > 0:
        articles = _read_wikipedia(wikipedia_path, wikipedia_articles)
    else:
        articles = list(_stream_wikipedia(wikipedia_articles))
        _atomic_jsonl(wikipedia_path, articles)
    if not articles:
        raise ValueError("the automatic Wikipedia snapshot produced no usable articles")

    if not grounded_path.exists() or grounded_path.stat().st_size == 0:
        _atomic_jsonl(grounded_path, (*_question_rows(articles), *_no_search_rows()))
    if not rl_path.exists() or rl_path.stat().st_size == 0:
        _atomic_jsonl(rl_path, (*_question_rows(articles), *_no_search_rows()))
    if not evaluation_path.exists() or evaluation_path.stat().st_size == 0:
        _atomic_jsonl(
            evaluation_path,
            ({"prompt": row["question"]} for row in _no_search_rows()),
        )

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "sources": {
            "allenai/big-reasoning-traces": {
                "config": _REASONING_CONFIG,
                "revision": _REASONING_REVISION,
                "license": "per-source permissive licenses; see upstream dataset card",
            },
            "HuggingFaceFW/fineweb-edu": {
                "config": _FINEWEB_CONFIG,
                "revision": _FINEWEB_REVISION,
                "license": "ODC-By-1.0; also subject to Common Crawl terms",
            },
        },
        "wikipedia": {
            "dataset": WIKIPEDIA_DATASET,
            "config": WIKIPEDIA_CONFIG,
            "revision": WIKIPEDIA_REVISION,
            "license": WIKIPEDIA_LICENSE,
            "articles": len(articles),
            "sha256": _sha256(wikipedia_path),
        },
        "tokenizer_samples": tokenizer_count,
        "inputs": {
            "wikipedia": str(wikipedia_path),
            "grounded_questions": str(grounded_path),
            "rl_prompts": str(rl_path),
            "evaluation_prompts": str(evaluation_path),
        },
        "input_sha256": {
            "wikipedia": _sha256(wikipedia_path),
            "grounded_questions": _sha256(grounded_path),
            "rl_prompts": _sha256(rl_path),
            "evaluation_prompts": _sha256(evaluation_path),
        },
    }
    atomic_write_text(manifest_output, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
