from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpt2_reasoning_search import bootstrap as bootstrap_module
from gpt2_reasoning_search.agent import TOOL_INSTRUCTION
from gpt2_reasoning_search.bootstrap import (
    WIKIPEDIA_CONFIG,
    WIKIPEDIA_DATASET,
    WIKIPEDIA_REVISION,
    bootstrap_local_inputs,
)
from gpt2_reasoning_search.cli import app

ROOT = Path(__file__).resolve().parents[1]


def _article_row(identifier: str, title: str) -> dict[str, str]:
    return {
        "id": identifier,
        "title": title,
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "text": (f"{title} is an encyclopedia topic. " + "Useful context. " * 40).strip(),
    }


def test_bootstrap_downloads_pinned_sources_writes_schemas_and_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_calls: list[tuple[object, ...]] = []
    text_calls: list[tuple[object, ...]] = []
    wikipedia_rows = [
        {"id": "bad", "title": "", "text": "too short"},
        {**_article_row("ignored", "None Identifier"), "id": None},
        _article_row("1", "Alpha Topic"),
        _article_row("2", "Beta Topic"),
        _article_row("3", "Ignored After Limit"),
    ]

    def fake_load_dataset(*args, **kwargs):
        dataset_calls.append((*args, *[kwargs[key] for key in sorted(kwargs)]))
        return wikipedia_rows

    def fake_text_stream(dataset_id, revision, **kwargs):
        text_calls.append((dataset_id, revision, kwargs))
        return iter(
            [
                {"text": "reasoning sample " + "explain each step. " * 20},
                {"text": "education sample " + "teach this topic. " * 20},
            ]
        )

    monkeypatch.setattr(bootstrap_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(bootstrap_module, "stream_huggingface_texts", fake_text_stream)
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "artifacts" / "manifest.json"

    first = bootstrap_local_inputs(data_root, manifest_path, wikipedia_articles=2)

    wiki_path = data_root / "raw" / "wikipedia.jsonl"
    grounded_path = data_root / "raw" / "grounded-questions.jsonl"
    rl_path = data_root / "rl" / "search-qa.jsonl"
    evaluation_path = data_root / "evaluation" / "contamination-prompts.jsonl"
    assert first["wikipedia"]["dataset"] == WIKIPEDIA_DATASET
    assert first["wikipedia"]["config"] == WIKIPEDIA_CONFIG
    assert first["wikipedia"]["revision"] == WIKIPEDIA_REVISION
    assert first["wikipedia"]["license"]
    assert first["wikipedia"]["articles"] == 2
    assert len(wiki_path.read_text().splitlines()) == 2
    article_ids = [json.loads(line)["id"] for line in wiki_path.read_text().splitlines()]
    assert len(article_ids) == len(set(article_ids)) == 2
    assert article_ids[1] == "wiki-1"

    grounded = [json.loads(line) for line in grounded_path.read_text().splitlines()]
    rl = [json.loads(line) for line in rl_path.read_text().splitlines()]
    evaluation = [json.loads(line) for line in evaluation_path.read_text().splitlines()]
    assert len(grounded) == len(rl) == 4
    for row in grounded + rl:
        assert {"id", "question", "answer", "search_required"} <= row.keys()
        if row["search_required"]:
            assert isinstance(row["query"], str) and row["query"]
            assert row["supporting_ids"] == [row["evidence"][0]["id"]]
            assert row["evidence"][0]["provider"] == "local-wikipedia"
        else:
            assert "query" not in row
            assert row["supporting_ids"] == []
    assert evaluation == [
        {"prompt": "What is 2 + 2?"},
        {"prompt": "What is the capital of France?"},
    ]
    assert first["input_sha256"] == {
        "wikipedia": hashlib.sha256(wiki_path.read_bytes()).hexdigest(),
        "grounded_questions": hashlib.sha256(grounded_path.read_bytes()).hexdigest(),
        "rl_prompts": hashlib.sha256(rl_path.read_bytes()).hexdigest(),
        "evaluation_prompts": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
    }
    assert len(text_calls) == 2
    assert text_calls[0][0] == "allenai/big-reasoning-traces"
    assert text_calls[0][1] == bootstrap_module._REASONING_REVISION
    assert text_calls[0][2]["config_name"] == "DeepSeek"
    assert text_calls[1][0] == "HuggingFaceFW/fineweb-edu"
    assert text_calls[1][1] == bootstrap_module._FINEWEB_REVISION
    assert text_calls[1][2]["config_name"] == "sample-10BT"
    assert dataset_calls == [
        (
            WIKIPEDIA_DATASET,
            WIKIPEDIA_CONFIG,
            WIKIPEDIA_REVISION,
            "train",
            True,
        )
    ]
    assert not list(tmp_path.rglob("*.partial"))

    before_calls = (len(dataset_calls), len(text_calls))
    second = bootstrap_local_inputs(data_root, manifest_path, wikipedia_articles=2)
    assert second == json.loads(manifest_path.read_text())
    assert (len(dataset_calls), len(text_calls)) == before_calls


def test_bootstrap_cli_forwards_fixed_paths_and_prints_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    captured: list[tuple[Path, Path, int]] = []

    def fake_bootstrap(data_root: Path, manifest_output: Path, wikipedia_articles: int):
        captured.append((data_root, manifest_output, wikipedia_articles))
        return {"format_version": 1, "inputs": {"wikipedia": "data/raw/wikipedia.jsonl"}}

    monkeypatch.setattr("gpt2_reasoning_search.cli.bootstrap_local_inputs", fake_bootstrap)
    result = CliRunner().invoke(
        app,
        [
            "bootstrap-data",
            "--data-root",
            str(tmp_path / "data"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
            "--wikipedia-articles",
            "7",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured == [(tmp_path / "data", tmp_path / "manifest.json", 7)]
    assert json.loads(result.stdout) == {
        "format_version": 1,
        "inputs": {"wikipedia": "data/raw/wikipedia.jsonl"},
    }


def test_bootstrap_manifest_replacement_failure_preserves_previous_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    (data_root / "tokenizer-sample").mkdir(parents=True)
    (data_root / "tokenizer-sample" / "sample.txt").write_text("sample\n")
    article = _article_row("1", "Alpha Topic")
    raw = data_root / "raw"
    raw.mkdir()
    (raw / "wikipedia.jsonl").write_text(json.dumps(article) + "\n")
    (raw / "grounded-questions.jsonl").write_text("{}\n")
    (data_root / "rl").mkdir()
    (data_root / "rl" / "search-qa.jsonl").write_text("{}\n")
    (data_root / "evaluation").mkdir()
    (data_root / "evaluation" / "contamination-prompts.jsonl").write_text("{}\n")
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("previous\n")

    def fail_atomic_write(_path, _text):
        raise OSError("simulated manifest replacement failure")

    monkeypatch.setattr(bootstrap_module, "atomic_write_text", fail_atomic_write)
    with pytest.raises(OSError, match="simulated manifest replacement failure"):
        bootstrap_local_inputs(data_root, manifest_path, wikipedia_articles=1)

    assert manifest_path.read_text() == "previous\n"
    assert not list(tmp_path.rglob("*.partial"))


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_worker_prepare_invokes_bootstrap_before_other_prepare_commands(tmp_path: Path) -> None:
    project = tmp_path / "school project"
    (project / "scripts" / "ssh").mkdir(parents=True)
    (project / "config").mkdir()
    shutil.copy2(ROOT / "scripts" / "ssh" / "worker.sh", project / "scripts" / "ssh" / "worker.sh")
    shutil.copy2(ROOT / "config" / "ssh.env.example", project / "config" / "ssh.env.example")
    data = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    for path in (
        data / "tokenizer.json",
        data / "reasoning.bin",
        data / "general.bin",
        data / "trajectories.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".bin":
            path.write_bytes(b"\0")
            path.with_suffix(".manifest.json").write_text("{}\n")
        elif path.name == "trajectories.jsonl":
            path.write_text(TOOL_INSTRUCTION + "\nready\n")
        else:
            path.write_text("ready\n")
    (data / "preparation-manifest.json").write_text("{}\n")
    (data / "wiki-index" / "lexical").mkdir(parents=True)
    (data / "wiki-index" / "metadata.sqlite3").write_bytes(b"sqlite")
    (data / "wiki-index" / "retrieval-manifest.json").write_text("{}\n")
    (data / "rl.jsonl").write_text("ready\n")
    config = project / "config" / "ssh.env"
    config.write_text(
        "\n".join(
            [
                f"TRAIN_CACHE={tmp_path / 'cache'}",
                f"TOKENIZER_INPUT_DIR={data / 'tokenizer'}",
                f"TOKENIZER_PATH={data / 'tokenizer.json'}",
                f"REASONING_TOKENS={data / 'reasoning.bin'}",
                f"GENERAL_TOKENS={data / 'general.bin'}",
                f"WIKIPEDIA_JSONL={data / 'wikipedia.jsonl'}",
                f"GROUNDED_QUESTIONS={data / 'grounded.jsonl'}",
                f"RL_PROMPTS={data / 'rl.jsonl'}",
                f"TRAJECTORIES={data / 'trajectories.jsonl'}",
                f"WIKI_INDEX={data / 'wiki-index'}",
                f"EVALUATION_PROMPTS={data / 'evaluation.jsonl'}",
                f"CHECKPOINT_ROOT={outputs}",
                f"MAIN_OUTPUT={outputs / 'main'}",
                f"SFT_OUTPUT={outputs / 'sft'}",
                f"RL_OUTPUT={outputs / 'rl'}",
                "TRAIN_PROFILE=custom",
                "ALLOW_CUSTOM_PATHS=1",
            ]
        )
        + "\n"
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for command in ("bash", "date", "dirname", "grep", "mkdir", "rm", "tee"):
        target = shutil.which(command)
        assert target is not None
        (fake_bin / command).symlink_to(target)
    calls = tmp_path / "calls.log"
    _write_executable(
        fake_bin / "uv",
        '{ printf "CALL "; printf "%s " "$@"; printf "\\n"; } >> "$FAKE_CALLS"\n',
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin),
            "SSH_TRAIN_CONFIG": str(config),
            "FAKE_CALLS": str(calls),
        }
    )
    result = subprocess.run(
        [str(project / "scripts" / "ssh" / "worker.sh"), "prepare"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = calls.read_text().splitlines()
    assert lines and lines[0].startswith("CALL run --locked gpt2-reasoning-search bootstrap-data")
    assert f"--data-root {project / 'data'}" in lines[0]
    assert f"--manifest-output {project / 'artifacts' / 'auto-data-manifest.json'}" in lines[0]
    assert len(lines) == 1
