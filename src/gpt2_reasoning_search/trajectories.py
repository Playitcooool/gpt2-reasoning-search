"""Build evidence-grounded search trajectories from JSONL question data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .agent import format_tool_results
from .data import atomic_write_lines
from .schemas import SearchArguments, SearchResult, ToolCall


def trajectory_document(
    question: str,
    answer: str,
    query: str | None,
    evidence: list[SearchResult],
    reasoning: str,
) -> str:
    prefix = f"<|problem|>\n{question.strip()}\n"
    if query is None:
        return f"{prefix}<|reasoning|>{reasoning.strip()}<|answer|>{answer.strip()}"
    call = ToolCall(
        name="search", arguments=SearchArguments(query=query, top_k=min(10, len(evidence) or 5))
    )
    citations = " ".join(f"<|citation|>{item.id}" for item in evidence)
    return (
        prefix
        + "<|tool_call|>"
        + call.model_dump_json()
        + "<|end_tool_call|>\n"
        + format_tool_results(evidence)
        + f"\n<|reasoning|>{reasoning.strip()}<|answer|>{answer.strip()} {citations}".rstrip()
    )


def multi_step_trajectory_document(
    question: str,
    answer: str,
    searches: list[tuple[str, list[SearchResult]]],
    reasoning: str,
) -> str:
    if not searches:
        return trajectory_document(question, answer, None, [], reasoning)
    if len(searches) > 3:
        raise ValueError("a tool trajectory may contain at most three searches")
    parts = [f"<|problem|>\n{question.strip()}\n"]
    cited: dict[str, SearchResult] = {}
    for query, evidence in searches:
        call = ToolCall(
            name="search",
            arguments=SearchArguments(query=query, top_k=min(10, len(evidence) or 5)),
        )
        parts.extend(
            [
                "<|tool_call|>",
                call.model_dump_json(),
                "<|end_tool_call|>\n",
                format_tool_results(evidence),
                "\n",
            ]
        )
        for item in evidence:
            cited[item.id] = item
    citations = " ".join(f"<|citation|>{source_id}" for source_id in cited)
    parts.append(f"<|reasoning|>{reasoning.strip()}<|answer|>{answer.strip()} {citations}".rstrip())
    return "".join(parts)


def generate_trajectories(rows: Iterable[dict], output: Path) -> int:
    """Normalize evidence-grounded rows; expected fields are documented in README."""
    count = 0

    def serialized_rows() -> Iterable[str]:
        nonlocal count
        for row in rows:
            reasoning = row.get("reasoning", "Use the supplied evidence to answer the question.")
            if "searches" in row:
                searches = [
                    (
                        step["query"],
                        [
                            SearchResult.model_validate(item)
                            for item in step.get("evidence", [])
                        ],
                    )
                    for step in row["searches"]
                ]
                document = multi_step_trajectory_document(
                    row["question"], row["answer"], searches, reasoning
                )
            else:
                evidence = [
                    SearchResult.model_validate(item) for item in row.get("evidence", [])
                ]
                document = trajectory_document(
                    question=row["question"],
                    answer=row["answer"],
                    query=row.get("query"),
                    evidence=evidence,
                    reasoning=reasoning,
                )
            count += 1
            yield json.dumps({"text": document}, ensure_ascii=False)

    atomic_write_lines(output, serialized_rows(), require_nonempty=True)
    return count


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
