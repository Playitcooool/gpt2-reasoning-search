import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gpt2_reasoning_search.cli import app
from gpt2_reasoning_search.data import (
    ContaminationFilter,
    PreparationStats,
    PreparedDocument,
    load_token_array,
    normalized_hash,
    write_token_file,
)
from gpt2_reasoning_search.prepare import (
    _general_rejection,
    _verify_reasoning_row,
    load_dataset_manifest,
    load_evaluation_prompts,
    prepare_token_corpora,
    stream_general_documents,
    stream_reasoning_documents,
)


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class RecordingTokenizer:
    def __init__(self, vocab_size: int = 1_024) -> None:
        self.vocab_size = vocab_size
        self.batch_sizes: list[int] = []

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        assert with_added_tokens
        return self.vocab_size

    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        self.batch_sizes.append(len(texts))
        return [
            _Encoding([ord(character) % self.vocab_size for character in text]) for text in texts
        ]

    def encode(self, _text: str) -> _Encoding:
        raise AssertionError("write_token_file must use batched tokenization")

    def to_str(self) -> str:
        return json.dumps({"vocab_size": self.vocab_size}, sort_keys=True)


def _manifest(tmp_path: Path, **big_overrides: object) -> Path:
    big = {
        "revision": "a" * 40,
        "config": "DeepSeek",
        "license": "per-source permissive",
        "role": "reasoning",
        "included_sources": ["OpenThoughts-114k", "OpenR1-Math-220k"],
        **big_overrides,
    }
    payload = {
        "allenai/big-reasoning-traces": big,
        "open-r1/OpenR1-Math-220k": {
            "revision": "b" * 40,
            "license": "Apache-2.0",
            "role": "verified-math",
        },
        "HuggingFaceFW/fineweb-edu": {
            "revision": "c" * 40,
            "config": "sample-10BT",
            "license": "ODC-By-1.0",
            "role": "general-text",
        },
    }
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps(payload))
    return path


def test_repository_dataset_manifest_is_complete_and_source_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_dataset_manifest(root / "config" / "datasets.json")

    assert set(manifest) == {
        "allenai/big-reasoning-traces",
        "open-r1/OpenR1-Math-220k",
        "HuggingFaceFW/fineweb-edu",
    }
    for entry in manifest.values():
        assert entry["role"].strip()
        assert entry["license"].strip()
        assert len(entry["revision"]) == 40
        int(entry["revision"], 16)
    assert manifest["allenai/big-reasoning-traces"]["config"] == "DeepSeek"
    assert manifest["allenai/big-reasoning-traces"]["included_sources"] == [
        "OpenThoughts-114k",
        "OpenR1-Math-220k",
    ]
    assert manifest["HuggingFaceFW/fineweb-edu"]["config"] == "sample-10BT"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"revision": "main"}, "full commit hash"),
        ({"license": ""}, "license"),
        ({"config": ""}, "config"),
        ({"role": ""}, "role"),
        ({"included_sources": []}, "allowlist"),
    ],
)
def test_dataset_manifest_rejects_incomplete_source_metadata(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_dataset_manifest(_manifest(tmp_path, **overrides))


def test_reasoning_stream_enforces_allowlist_structure_and_reports_every_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "source": "OpenR1-Math-220k",
            "prompt": "Prove it",
            "response": "<think>work</think><answer>42</answer>",
            "finish_reason": "stop",
        },
        {
            "source": "OpenThoughts-114k",
            "prompt": "Write code",
            "response": "<THINK>steps</THINK><ANSWER>code</ANSWER>",
            "finish_reason": "eos_token",
        },
        {"source": "unapproved", "prompt": "x", "response": "<think>x</think><answer>x</answer>"},
        {
            "source": "OpenR1-Math-220k",
            "prompt": "",
            "response": "<think>x</think><answer>x</answer>",
        },
        {"source": "OpenR1-Math-220k", "prompt": "x", "response": ""},
        {"source": "OpenR1-Math-220k", "prompt": "x", "response": "<answer>x</answer>"},
        {"source": "OpenR1-Math-220k", "prompt": "x", "response": "<think>x</think>"},
        {
            "source": "OpenR1-Math-220k",
            "prompt": "x",
            "response": "<think>x</think><answer>x</answer>",
            "finish_reason": "length",
        },
        {
            "source": "OpenR1-Math-220k",
            "prompt": "x" * 200_001,
            "response": "<think>x</think><answer>x</answer>",
        },
    ]
    calls: list[tuple] = []

    def fake_stream(*args, **kwargs):
        calls.append((args, kwargs))
        yield from rows

    monkeypatch.setattr("gpt2_reasoning_search.prepare.stream_huggingface_texts", fake_stream)
    stats = PreparationStats()
    documents = list(stream_reasoning_documents(_manifest(tmp_path), stats))

    assert [document.source for document in documents] == [
        "OpenR1-Math-220k",
        "OpenThoughts-114k",
    ]
    assert documents[0].verification == "upstream-math-verified"
    assert documents[1].verification == "upstream-curated"
    assert documents[0].text == "<|problem|>\nProve it\n<|reasoning|>\nwork\n<|answer|>\n42"
    assert calls == [
        (
            ("allenai/big-reasoning-traces", "a" * 40),
            {"config_name": "DeepSeek", "text_field": "text"},
        )
    ]
    assert stats.to_dict() == {
        "rows_seen": 9,
        "rows_accepted": 2,
        "rows_rejected": 7,
        "rejection_reasons": {
            "incomplete_generation": 1,
            "missing_answer": 1,
            "missing_prompt": 1,
            "missing_reasoning": 1,
            "missing_response": 1,
            "source_not_in_math_code_logic_mix": 1,
            "too_long": 1,
        },
        "accepted_sources": {"OpenR1-Math-220k": 1, "OpenThoughts-114k": 1},
    }


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("a" * 199, "too_short"),
        ("a" * 200_001, "too_long"),
        ("a" * 195 + "\x00" * 5, "non_printable"),
        ("!" * 200, "low_alphanumeric_ratio"),
        (("same repeated line\n" * 10) + "a" * 200, "repetitive_lines"),
        ("A useful educational paragraph with facts. " * 10, None),
    ],
)
def test_general_quality_filter_categories(text: str, reason: str | None) -> None:
    assert _general_rejection(text) == reason


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({"language": "fr", "language_score": 1.0, "score": 5}, "non_english"),
        ({"language": "en", "language_score": 0.64, "score": 5}, "low_language_score"),
        ({"language": "eng", "language_score": 0.65, "score": 2.99}, "low_education_score"),
        ({"language": "english", "language_score": 0.65, "int_score": 3}, None),
        ({}, None),
    ],
)
def test_general_quality_filter_uses_available_language_and_education_metadata(
    metadata: dict[str, object], reason: str | None
) -> None:
    text = "A useful educational paragraph with facts. " * 10

    assert _general_rejection(text, metadata) == reason


@pytest.mark.parametrize(
    ("row", "source", "answer", "expected"),
    [
        (
            {"domain": "math", "reference_answer": "3/4"},
            "OpenThoughts-114k",
            r"$\boxed{3/4}$.",
            (True, "local-math-final-answer"),
        ),
        (
            {"domain": "math", "reference_answer": "3/4"},
            "OpenThoughts-114k",
            "1/4",
            (False, "local-math-final-answer"),
        ),
        (
            {"category": "logic puzzle", "ground_truth": "C"},
            "OpenThoughts-114k",
            "Option C",
            (True, "local-logic-answer"),
        ),
        (
            {"category": "logic puzzle", "ground_truth": "C"},
            "OpenThoughts-114k",
            "Option B",
            (False, "local-logic-answer"),
        ),
        ({}, "OpenR1-Math-220k", "42", (True, "upstream-math-verified")),
        ({}, "OpenThoughts-114k", "42", (True, "upstream-curated")),
    ],
)
def test_reasoning_verification_selects_local_checks_or_explicit_upstream_provenance(
    row: dict[str, object], source: str, answer: str, expected: tuple[bool, str]
) -> None:
    assert _verify_reasoning_row(row, source, "response", answer) == expected


def test_reasoning_code_verification_uses_tests_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class Result:
        passed = False

    def fake_verify(code: str, tests: str) -> Result:
        calls.append((code, tests))
        return Result()

    monkeypatch.setattr("gpt2_reasoning_search.prepare.verify_python_code", fake_verify)
    row = {
        "code": "def add(a, b): return a - b",
        "tests": ["assert add(2, 3) == 5", "assert add(0, 0) == 0"],
    }

    result = _verify_reasoning_row(row, "OpenThoughts-114k", "response", "answer")

    assert result == (False, "local-code-tests")
    assert calls == [
        (
            "def add(a, b): return a - b",
            "assert add(2, 3) == 5\nassert add(0, 0) == 0",
        )
    ]


def test_reasoning_stream_rejects_failed_local_verification_with_accurate_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "source": "OpenR1-Math-220k",
            "prompt": "Compute",
            "response": "<think>work</think><answer>42</answer>",
            "expected_answer": "42",
        },
        {
            "source": "OpenR1-Math-220k",
            "prompt": "Compute",
            "response": "<think>bad work</think><answer>41</answer>",
            "expected_answer": "42",
        },
    ]
    monkeypatch.setattr(
        "gpt2_reasoning_search.prepare.stream_huggingface_texts",
        lambda *args, **kwargs: iter(rows),
    )
    stats = PreparationStats()

    documents = list(stream_reasoning_documents(_manifest(tmp_path), stats))

    assert len(documents) == 1
    assert documents[0].verification == "local-math-final-answer"
    assert stats.to_dict() == {
        "rows_seen": 2,
        "rows_accepted": 1,
        "rows_rejected": 1,
        "rejection_reasons": {"local-math-final-answer_failed": 1},
        "accepted_sources": {"OpenR1-Math-220k": 1},
    }


def test_general_stream_records_filtering_and_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"text": "short"},
        {"text": "A useful educational paragraph with facts. " * 10},
    ]
    monkeypatch.setattr(
        "gpt2_reasoning_search.prepare.stream_huggingface_texts",
        lambda *args, **kwargs: iter(rows),
    )
    stats = PreparationStats()

    documents = list(stream_general_documents(_manifest(tmp_path), stats))

    assert len(documents) == 1
    assert documents[0].source == "FineWeb-Edu"
    assert documents[0].verification == "upstream-edu-filtered"
    assert stats.to_dict()["rejection_reasons"] == {"too_short": 1}
    assert stats.to_dict()["accepted_sources"] == {"FineWeb-Edu": 1}


def test_evaluation_prompt_loader_accepts_jsonl_and_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"prompt":"alpha"}\n'
        '{"question":"beta"}\n'
        '{"text":"gamma"}\n'
        '{"ignored":"value"}\n'
        "plain text prompt\n\n"
    )

    assert load_evaluation_prompts(path) == ["alpha", "beta", "gamma", "plain text prompt"]
    assert load_evaluation_prompts(None) == []


def test_write_token_file_batches_streams_and_does_not_use_numpy_concatenate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = RecordingTokenizer()
    yielded = 0

    def documents() -> Iterable[str]:
        nonlocal yielded
        for index in range(10):
            yielded += 1
            yield f"document-{index}"

    monkeypatch.setattr(
        np,
        "concatenate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("corpus accumulation")),
    )
    output = tmp_path / "tokens.bin"

    report = write_token_file(tokenizer, documents(), output, 1_000, batch_size=3)

    assert yielded == 10
    assert tokenizer.batch_sizes == [3, 3, 3, 1]
    assert max(tokenizer.batch_sizes) == 3
    assert report["documents"] == 10
    assert output.stat().st_size == report["tokens"] * 2


@pytest.mark.parametrize(
    ("vocab_size", "expected_dtype", "itemsize"),
    [(65_535, "uint16", 2), (65_536, "uint32", 4)],
)
def test_token_dtype_is_selected_without_overflow(
    tmp_path: Path, vocab_size: int, expected_dtype: str, itemsize: int
) -> None:
    tokenizer = RecordingTokenizer(vocab_size)
    output = tmp_path / f"tokens-{vocab_size}.bin"

    report = write_token_file(tokenizer, ["abc"], output, 10)

    assert report["dtype"] == expected_dtype
    assert output.stat().st_size == report["tokens"] * itemsize
    assert load_token_array(output).dtype == np.dtype(expected_dtype)


def test_disk_backed_dedup_is_exact_normalized_and_leaves_no_database(tmp_path: Path) -> None:
    tokenizer = RecordingTokenizer()
    output = tmp_path / "tokens.bin"

    report = write_token_file(
        tokenizer,
        ["Alpha   Beta", " alpha beta ", "Alpha Beta!"],
        output,
        1_000,
        batch_size=2,
    )

    assert report["duplicates_removed"] == 1
    assert report["documents"] == 2
    assert normalized_hash("Alpha   Beta") == normalized_hash(" alpha beta ")
    assert normalized_hash("Alpha Beta!") != normalized_hash("Alpha Beta")
    assert not list(tmp_path.glob("dedup-*.sqlite3"))


def test_partial_final_document_is_written_and_reported_as_truncated(tmp_path: Path) -> None:
    tokenizer = RecordingTokenizer()
    output = tmp_path / "tokens.bin"

    report = write_token_file(
        tokenizer,
        [PreparedDocument("abc", "one", "verified"), PreparedDocument("defgh", "two", "curated")],
        output,
        max_tokens=5,
        batch_size=2,
    )
    tokens = load_token_array(output)

    assert tokens.tolist() == [ord(c) for c in "abcde"]
    assert report["tokens"] == 5
    assert report["documents"] == 2
    assert report["truncated_documents"] == 1
    assert report["source_documents"] == {"one": 1, "two": 1}
    assert report["verification"] == {"curated": 1, "verified": 1}


def test_token_report_hashes_bytes_tokenizer_sources_verification_and_filter_stats(
    tmp_path: Path,
) -> None:
    tokenizer = RecordingTokenizer()
    output = tmp_path / "tokens.bin"
    stats = PreparationStats(rows_seen=4)
    stats.accept("source-a")
    stats.accept("source-a")
    stats.reject("invalid")
    stats.reject("duplicate-upstream")
    documents = [
        PreparedDocument("abc", "source-a", "math-verified"),
        PreparedDocument("def", "source-a", "math-verified"),
    ]

    report = write_token_file(tokenizer, documents, output, 100, preparation_stats=stats)
    persisted = json.loads(output.with_suffix(".manifest.json").read_text())

    assert report == persisted
    assert report["bytes"] == output.stat().st_size
    assert report["bytes"] == report["tokens"] * np.dtype(report["dtype"]).itemsize
    assert report["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["tokenizer_sha256"] == hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
    assert report["source_documents"] == {"source-a": 2}
    assert report["verification"] == {"math-verified": 2}
    assert report["filtering"] == {
        "rows_seen": 4,
        "rows_accepted": 2,
        "rows_rejected": 2,
        "rejection_reasons": {"duplicate-upstream": 1, "invalid": 1},
        "accepted_sources": {"source-a": 2},
    }


def test_contamination_filter_counts_removals_before_dedup(tmp_path: Path) -> None:
    tokenizer = RecordingTokenizer()
    output = tmp_path / "tokens.bin"
    prompt = "one two three four five six seven eight nine"
    filter_ = ContaminationFilter([prompt], ngram_size=3, threshold=0.5)

    report = write_token_file(
        tokenizer,
        [prompt, prompt, "different educational source"],
        output,
        1_000,
        contamination_filter=filter_,
    )

    assert report["contaminated_removed"] == 2
    assert report["duplicates_removed"] == 0
    assert report["documents"] == 1


def test_load_token_array_validates_raw_file_size_and_loads_legacy_npy(tmp_path: Path) -> None:
    tokenizer = RecordingTokenizer()
    raw = tmp_path / "tokens.bin"
    write_token_file(tokenizer, ["abcd"], raw, 100)
    loaded = load_token_array(raw)
    assert isinstance(loaded, np.memmap)
    assert loaded.tolist() == [ord(c) for c in "abcd"]

    raw.write_bytes(raw.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size does not match"):
        load_token_array(raw)

    legacy_path = tmp_path / "legacy.npy"
    expected = np.array([1, 2, 3], dtype=np.uint32)
    np.save(legacy_path, expected)
    legacy = load_token_array(legacy_path)
    assert isinstance(legacy, np.memmap)
    np.testing.assert_array_equal(legacy, expected)


def test_prepare_token_corpora_writes_auditable_run_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _manifest(tmp_path)
    output = tmp_path / "prepared"
    tokenizer = RecordingTokenizer()
    reasoning = PreparedDocument("reasoning document", "OpenR1-Math-220k", "upstream-math-verified")
    general = PreparedDocument("general document", "FineWeb-Edu", "upstream-edu-filtered")

    def fake_reasoning(_path, stats):
        stats.rows_seen += 1
        stats.accept(reasoning.source)
        yield reasoning

    def fake_general(_path, stats):
        stats.rows_seen += 1
        stats.accept(general.source)
        yield general

    monkeypatch.setattr("gpt2_reasoning_search.prepare.stream_reasoning_documents", fake_reasoning)
    monkeypatch.setattr("gpt2_reasoning_search.prepare.stream_general_documents", fake_general)

    reports = prepare_token_corpora(
        tokenizer,
        manifest_path,
        output,
        reasoning_token_cap=100,
        general_token_cap=100,
        evaluation_prompts=["held out prompt"],
    )
    run_manifest = json.loads((output / "preparation-manifest.json").read_text())

    assert (output / "reasoning.bin").is_file()
    assert (output / "general.bin").is_file()
    assert run_manifest["format_version"] == 2
    assert run_manifest["datasets"] == load_dataset_manifest(manifest_path)
    assert run_manifest["evaluation_prompts"] == 1
    assert run_manifest["outputs"] == reports
    assert reports["reasoning"]["filtering"]["rows_accepted"] == 1
    assert reports["general"]["filtering"]["rows_accepted"] == 1


def test_prepare_data_cli_exposes_evaluation_prompt_option() -> None:
    result = CliRunner().invoke(app, ["prepare-data", "--help"])

    assert result.exit_code == 0
    assert "--evaluation-prompts" in result.stdout
    assert "prompts to exclude" in result.stdout
