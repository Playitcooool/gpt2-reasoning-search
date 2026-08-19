"""Reproducible Hugging Face corpus preparation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .data import reasoning_document, stream_huggingface_texts, write_token_file


def load_dataset_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text())


def _answer_from_row(row: dict[str, Any]) -> str:
    answer = row.get("answer") or row.get("solution") or ""
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return str(answer)


def stream_reasoning_documents(manifest_path: Path) -> Iterator[str]:
    manifest = load_dataset_manifest(manifest_path)
    source = manifest["allenai/big-reasoning-traces"]
    for row in stream_huggingface_texts(
        "allenai/big-reasoning-traces", source["revision"], text_field="text"
    ):
        text = str(row.get("text", "")).strip()
        if not text or len(text) > 200_000:
            continue
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        if prompt and response:
            yield reasoning_document(prompt, response, _answer_from_row(row) or response[-256:])
        else:
            yield text


def stream_general_documents(manifest_path: Path) -> Iterator[str]:
    manifest = load_dataset_manifest(manifest_path)
    source = manifest["HuggingFaceFW/fineweb-edu"]
    for row in stream_huggingface_texts(
        "HuggingFaceFW/fineweb-edu",
        source["revision"],
        config_name=source["config"],
        text_field="text",
    ):
        text = str(row["text"]).strip()
        if 200 <= len(text) <= 200_000:
            yield text


def prepare_token_corpora(
    tokenizer: Tokenizer,
    manifest_path: Path,
    output_directory: Path,
    reasoning_token_cap: int,
    general_token_cap: int,
) -> dict[str, dict[str, int | str]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    return {
        "reasoning": write_token_file(
            tokenizer,
            stream_reasoning_documents(manifest_path),
            output_directory / "reasoning.npy",
            reasoning_token_cap,
        ),
        "general": write_token_file(
            tokenizer,
            stream_general_documents(manifest_path),
            output_directory / "general.npy",
            general_token_cap,
        ),
    }
