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
    atomic_write_text,
    reasoning_document,
    stream_huggingface_texts,
    write_token_file,
)
from .verifiers import normalize_math_answer, verify_logic_choice, verify_python_code


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


def _verify_reasoning_row(
    row: dict[str, Any], source: str, response: str, answer: str
) -> tuple[bool, str]:
    domain = " ".join(
        str(row.get(field, "")) for field in ("domain", "task", "category", "problem_type")
    ).lower()
    reference = next(
        (
            str(row[field])
            for field in ("expected_answer", "reference_answer", "ground_truth", "answer")
            if row.get(field) is not None and str(row[field]).strip()
        ),
        None,
    )
    tests = row.get("tests") or row.get("test_code")
    if tests is not None:
        code = str(row.get("code", "")).strip()
        if not code:
            fenced = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
            code = fenced[-1].strip() if fenced else answer
        test_text = "\n".join(map(str, tests)) if isinstance(tests, list) else str(tests)
        result = verify_python_code(code, test_text)
        return result.passed, "local-code-tests"
    if reference is not None and any(label in domain for label in ("logic", "choice", "puzzle")):
        return verify_logic_choice(answer, reference), "local-logic-answer"
    if reference is not None and (source == "OpenR1-Math-220k" or "math" in domain):
        return (
            normalize_math_answer(answer) == normalize_math_answer(reference),
            "local-math-final-answer",
        )
    if source == "OpenR1-Math-220k":
        return True, "upstream-math-verified"
    return True, "upstream-curated"


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
        verified, verification = _verify_reasoning_row(row, source, response, answer)
        if not verified:
            stats.reject(f"{verification}_failed")
            continue
        stats.accept(source)
        yield PreparedDocument(
            reasoning_document(prompt, reasoning, answer),
            source=source,
            verification=verification,
        )


def _general_rejection(text: str, metadata: dict[str, Any] | None = None) -> str | None:
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
    if metadata is not None:
        language = str(metadata.get("language", "en")).lower()
        if language not in {"en", "eng", "english"}:
            return "non_english"
        language_score = metadata.get("language_score")
        if language_score is not None and float(language_score) < 0.65:
            return "low_language_score"
        education_score = metadata.get("score", metadata.get("int_score"))
        if education_score is not None and float(education_score) < 3.0:
            return "low_education_score"
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
        rejection = _general_rejection(text, row)
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
    contamination_filter = ContaminationFilter(evaluation_prompts) if evaluation_prompts else None
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
    atomic_write_text(
        output_directory / "preparation-manifest.json",
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
    )
    return reports
