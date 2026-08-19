"""Metrics and reports for reasoning, retrieval, tools, and mixture controls."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = 0
    remaining = expected.copy()
    for token in predicted:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def perplexity(mean_loss: float) -> float:
    return math.exp(min(mean_loss, 20.0))


def score_reasoning_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in records:
        task = str(row.get("task", "unknown"))
        grouped[task].append(
            {
                "exact_match": exact_match(str(row["prediction"]), str(row["answer"])),
                "token_f1": token_f1(str(row["prediction"]), str(row["answer"])),
            }
        )
    by_task = {
        task: {
            "count": len(values),
            "exact_match": sum(item["exact_match"] for item in values) / len(values),
            "token_f1": sum(item["token_f1"] for item in values) / len(values),
        }
        for task, values in grouped.items()
    }
    count = sum(item["count"] for item in by_task.values())
    return {
        "count": count,
        "exact_match": sum(item["exact_match"] * item["count"] for item in by_task.values())
        / max(1, count),
        "token_f1": sum(item["token_f1"] * item["count"] for item in by_task.values())
        / max(1, count),
        "by_task": by_task,
    }


def score_grounded_records(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    count = valid_calls = searches = unnecessary = recoveries = 0
    answer_score = retrieval_recall = citation_precision = 0.0
    for row in records:
        count += 1
        answer_score += exact_match(str(row["prediction"]), str(row["answer"]))
        valid_calls += int(bool(row.get("valid_tool_calls", True)))
        calls = row.get("queries", [])
        searches += int(bool(calls))
        unnecessary += int(bool(calls) and not row.get("search_required", True))
        recoveries += int(len(calls) > 1 and row.get("answer_found", False))
        retrieved = set(row.get("retrieved_ids", []))
        supporting = set(row.get("supporting_ids", []))
        cited = set(row.get("cited_ids", []))
        retrieval_recall += len(retrieved & supporting) / max(1, len(supporting))
        citation_precision += len(cited & retrieved) / max(1, len(cited)) if cited else 1.0
    return {
        "count": count,
        "answer_exact_match": answer_score / max(1, count),
        "retrieval_recall": retrieval_recall / max(1, count),
        "citation_precision": citation_precision / max(1, count),
        "valid_tool_call_rate": valid_calls / max(1, count),
        "search_rate": searches / max(1, count),
        "unnecessary_search_rate": unnecessary / max(1, count),
        "query_recovery_rate": recoveries / max(1, count),
    }


def compare_proxy_runs(run_directories: Iterable[Path]) -> dict[str, Any]:
    runs = []
    for directory in run_directories:
        metrics_path = directory / "metrics.jsonl"
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        if not records:
            raise ValueError(f"no metrics found in {metrics_path}")
        final = records[-1]
        runs.append(
            {
                "run": directory.name,
                "reasoning_ratio": final["reasoning_tokens"] / max(1, final["tokens_seen"]),
                "final_loss": final["loss"],
                "tokens_seen": final["tokens_seen"],
                "tokens_per_second": final["tokens_per_second"],
            }
        )
    return {"runs": sorted(runs, key=lambda item: item["reasoning_ratio"])}


def stream_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
