"""Reproducible, source-aware Hugging Face corpus preparation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .data import (
    ContaminationFilter,
    PreparationStats,
    PreparedDocument,
    reasoning_document,
    stream_huggingface_texts,
    write_token_file,
)


def load_dataset_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text())
    for dataset_id, entry in manifest.items():
        revision = str(entry.get("revision", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{dataset_id} must be pinned to a full commit hash")
        if not entry.get("license"):
            raise ValueError(f"{dataset_id} is missing license metadata")
        if not entry.get("role"):
            raise ValueError(f"{dataset_id} is missing its training role")
        if "config" in entry and not entry["config"]:
            raise ValueError(f"{dataset_id} has an empty dataset configuration")
        if entry["role"] == "reasoning" and not entry.get("config"):
            raise ValueError(f"{dataset_id} reasoning data requires an explicit configuration")
        if entry["role"] == "reasoning" and not entry.get("included_sources"):
            raise ValueError(f"{dataset_id} reasoning data requires an included-source allowlist")
    return manifest


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _reasoning_rejection(row: dict[str, Any], included_sources: set[str]) -> str | None:
    source = str(row.get("source", ""))
    if source not in included_sources:
        return "source_not_in_math_code_logic_mix"
    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    if not prompt:
        return "missing_prompt"
    if not response:
        return "missing_response"
    if len(prompt) + len(response) > 200_000:
        return "too_long"
    if _extract_tag(response, "think") is None:
        return "missing_reasoning"
    if _extract_tag(response, "answer") is None:
        return "missing_answer"
    finish_reason = row.get("finish_reason")
    if finish_reason and finish_reason not in {"stop", "eos_token"}:
        return "incomplete_generation"
    return None


def stream_reasoning_documents(
    manifest_path: Path, stats: PreparationStats | None = None
) -> Iterator[PreparedDocument]:
    manifest = load_dataset_manifest(manifest_path)
    source_config = manifest["allenai/big-reasoning-traces"]
    included_sources = set(source_config["included_sources"])
    stats = stats or PreparationStats()
    for row in stream_huggingface_texts(
        "allenai/big-reasoning-traces",
        source_config["revision"],
        config_name=source_config["config"],
        text_field="text",
    ):
        stats.rows_seen += 1
        rejection = _reasoning_rejection(row, included_sources)
        if rejection:
            stats.reject(rejection)
            continue
        source = str(row["source"])
        prompt = str(row["prompt"]).strip()
        response = str(row["response"]).strip()
        reasoning = _extract_tag(response, "think")
        answer = _extract_tag(response, "answer")
        assert reasoning is not None and answer is not None
        verification = (
            "upstream-math-verified" if source == "OpenR1-Math-220k" else "upstream-curated"
        )
        stats.accept(source)
        yield PreparedDocument(
            reasoning_document(prompt, reasoning, answer),
            source=source,
            verification=verification,
        )


def _general_rejection(text: str) -> str | None:
    if len(text) < 200:
        return "too_short"
    if len(text) > 200_000:
        return "too_long"
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    if printable / len(text) < 0.98:
        return "non_printable"
    alphanumeric = sum(character.isalnum() for character in text)
    if alphanumeric / len(text) < 0.25:
        return "low_alphanumeric_ratio"
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
        return "repetitive_lines"
    return None


def stream_general_documents(
    manifest_path: Path, stats: PreparationStats | None = None
) -> Iterator[PreparedDocument]:
    manifest = load_dataset_manifest(manifest_path)
    source = manifest["HuggingFaceFW/fineweb-edu"]
    stats = stats or PreparationStats()
    for row in stream_huggingface_texts(
        "HuggingFaceFW/fineweb-edu",
        source["revision"],
        config_name=source["config"],
        text_field="text",
    ):
        stats.rows_seen += 1
        text = str(row["text"]).strip()
        rejection = _general_rejection(text)
        if rejection:
            stats.reject(rejection)
            continue
        stats.accept("FineWeb-Edu")
        yield PreparedDocument(text, source="FineWeb-Edu", verification="upstream-edu-filtered")


def load_evaluation_prompts(path: Path | None) -> list[str]:
    if path is None:
        return []
    prompts = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                prompts.append(line.strip())
                continue
            value = row.get("prompt") or row.get("question") or row.get("text")
            if value:
                prompts.append(str(value))
    return prompts


def prepare_token_corpora(
    tokenizer: Tokenizer,
    manifest_path: Path,
    output_directory: Path,
    reasoning_token_cap: int,
    general_token_cap: int,
    evaluation_prompts: Sequence[str] = (),
) -> dict[str, dict]:
    output_directory.mkdir(parents=True, exist_ok=True)
    contamination_filter = (
        ContaminationFilter(evaluation_prompts) if evaluation_prompts else None
    )
    reasoning_stats = PreparationStats()
    general_stats = PreparationStats()
    reports = {
        "reasoning": write_token_file(
            tokenizer,
            stream_reasoning_documents(manifest_path, reasoning_stats),
            output_directory / "reasoning.bin",
            reasoning_token_cap,
            contamination_filter=contamination_filter,
            preparation_stats=reasoning_stats,
        ),
        "general": write_token_file(
            tokenizer,
            stream_general_documents(manifest_path, general_stats),
            output_directory / "general.bin",
            general_token_cap,
            contamination_filter=contamination_filter,
            preparation_stats=general_stats,
        ),
    }
    run_manifest = {
        "format_version": 2,
        "datasets": load_dataset_manifest(manifest_path),
        "evaluation_prompts": len(evaluation_prompts),
        "outputs": reports,
    }
    (output_directory / "preparation-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    return reports
