"""Metrics and reports for reasoning, retrieval, tools, and mixture controls."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schemas import AnswerRequest

if TYPE_CHECKING:
    from .agent import SearchAgent


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


def strip_citation_markers(text: str, cited_ids: Sequence[str]) -> str:
    for source_id in cited_ids:
        text = text.replace(f"[{source_id}]", " ")
    return " ".join(text.split())


def perplexity(mean_loss: float) -> float:
    return math.exp(min(mean_loss, 20.0))


def percentile(values: Sequence[float], quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def retrieval_metrics(
    retrieved_ids: Sequence[str], supporting_ids: Sequence[str], cutoff: int | None = None
) -> dict[str, float]:
    if cutoff is not None and cutoff < 0:
        raise ValueError("retrieval cutoff must be non-negative")
    retrieved = list(dict.fromkeys(retrieved_ids))
    if cutoff is not None:
        retrieved = retrieved[:cutoff]
    supporting = set(supporting_ids)
    if not supporting:
        return {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0}
    relevant = [int(identifier in supporting) for identifier in retrieved]
    recall = sum(relevant) / len(supporting)
    first = next((index for index, value in enumerate(relevant, 1) if value), None)
    mrr = 1.0 / first if first is not None else 0.0
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(relevant, 1))
    ideal_count = min(len(supporting), len(retrieved))
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    return {"recall": recall, "mrr": mrr, "ndcg": dcg / ideal if ideal else 0.0}


def score_language_model_records(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    total_tokens = 0
    weighted_loss = 0.0
    for row in records:
        tokens = row.get("tokens", 1)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError(
                "language-model evaluation rows require a positive integer token count"
            )
        loss = float(row["loss"])
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("language-model loss must be finite and non-negative")
        total_tokens += tokens
        weighted_loss += loss * tokens
    if total_tokens == 0:
        raise ValueError("language-model evaluation is empty")
    mean_loss = weighted_loss / total_tokens
    return {
        "count_tokens": total_tokens,
        "mean_loss": mean_loss,
        "perplexity": perplexity(mean_loss),
    }


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


def score_grounded_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    count = searches = unnecessary = recoveries = recovery_opportunities = 0
    valid_calls = total_calls = 0
    answer_score = answer_f1 = 0.0
    retrieval_recall = retrieval_mrr = retrieval_ndcg = 0.0
    citation_precision = citation_recall = citation_validity = 0.0
    latencies: list[float] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        count += 1
        cited_ids = [str(value) for value in row.get("cited_ids", [])]
        prediction = strip_citation_markers(str(row["prediction"]), cited_ids)
        answer_score += exact_match(prediction, str(row["answer"]))
        answer_f1 += token_f1(prediction, str(row["answer"]))
        calls_total = int(row.get("tool_calls_total", len(row.get("queries", []))))
        valid_value = row.get("valid_tool_calls", calls_total)
        valid_calls += (
            int(valid_value)
            if not isinstance(valid_value, bool)
            else calls_total * int(valid_value)
        )
        total_calls += calls_total
        calls = row.get("queries", [])
        searches += int(bool(calls))
        unnecessary += int(bool(calls) and not row.get("search_required", True))
        if len(calls) > 1:
            recovery_opportunities += 1
            recoveries += int(row.get("answer_found", False))
        retrieved_list = row.get("retrieved_ids", [])
        supporting_list = row.get("supporting_ids", [])
        ranked = retrieval_metrics(retrieved_list, supporting_list)
        retrieval_recall += ranked["recall"]
        retrieval_mrr += ranked["mrr"]
        retrieval_ndcg += ranked["ndcg"]
        retrieved = set(retrieved_list)
        supporting = set(supporting_list)
        cited = set(cited_ids)
        citation_precision += (
            len(cited & supporting) / max(1, len(cited)) if cited else float(not supporting)
        )
        citation_recall += len(cited & supporting) / len(supporting) if supporting else 1.0
        citation_validity += len(cited & retrieved) / max(1, len(cited)) if cited else 1.0
        if "elapsed_seconds" in row:
            latencies.append(float(row["elapsed_seconds"]))
        grouped[str(row.get("search_mode", "unknown"))].append(row)
    report: dict[str, Any] = {
        "count": count,
        "answer_exact_match": answer_score / max(1, count),
        "answer_token_f1": answer_f1 / max(1, count),
        "retrieval_recall": retrieval_recall / max(1, count),
        "retrieval_mrr": retrieval_mrr / max(1, count),
        "retrieval_ndcg": retrieval_ndcg / max(1, count),
        "citation_precision": citation_precision / max(1, count),
        "citation_recall": citation_recall / max(1, count),
        "citation_validity": citation_validity / max(1, count),
        "valid_tool_call_rate": valid_calls / total_calls if total_calls else 1.0,
        "search_rate": searches / max(1, count),
        "unnecessary_search_rate": unnecessary / max(1, count),
        "query_recovery_rate": recoveries / max(1, recovery_opportunities),
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
    }
    if len(grouped) > 1:
        report["by_search_mode"] = {
            mode: score_grounded_records(values) for mode, values in grouped.items()
        }
    return report


async def run_grounded_benchmark(
    agent: SearchAgent,
    records: Iterable[dict[str, Any]],
    output: Path,
    search_modes: Sequence[str] = ("off", "local"),
    max_searches: int = 3,
) -> int:
    rows = list(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w") as handle:
        for mode in search_modes:
            for row in rows:
                response = await agent.answer(
                    AnswerRequest(
                        query=str(row["question"]),
                        search_mode=mode,  # type: ignore[arg-type]
                        max_searches=max_searches,
                    )
                )
                trace = response.tool_trace
                queries = [entry.call.arguments.query for entry in trace if entry.call is not None]
                retrieved_ids = [result.id for entry in trace for result in entry.results]
                valid_calls = sum(entry.status not in {"invalid", "rejected"} for entry in trace)
                record = {
                    "id": row.get("id"),
                    "search_mode": mode,
                    "prediction": response.answer,
                    "answer": row["answer"],
                    "queries": queries,
                    "tool_calls_total": len(trace),
                    "valid_tool_calls": valid_calls,
                    "retrieved_ids": retrieved_ids,
                    "supporting_ids": row.get("supporting_ids", []),
                    "cited_ids": [citation.source_id for citation in response.citations],
                    "search_required": row.get("search_required", True),
                    "answer_found": exact_match(
                        strip_citation_markers(
                            response.answer,
                            [citation.source_id for citation in response.citations],
                        ),
                        str(row["answer"]),
                    )
                    == 1.0,
                    "elapsed_seconds": response.elapsed_seconds,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
    return written


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
    ordered = sorted(runs, key=lambda item: item["reasoning_ratio"])
    token_budgets = {int(item["tokens_seen"]) for item in ordered}
    return {
        "runs": ordered,
        "equal_token_budget": len(token_budgets) == 1,
        "token_budget_spread": max(token_budgets) - min(token_budgets) if token_budgets else 0,
    }


def stream_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
