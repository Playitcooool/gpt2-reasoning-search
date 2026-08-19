"""Build evidence-grounded search trajectories from JSONL question data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .agent import format_tool_results
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


def generate_trajectories(rows: Iterable[dict], output: Path) -> int:
    """Normalize evidence-grounded rows; expected fields are documented in README."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as handle:
        for row in rows:
            evidence = [SearchResult.model_validate(item) for item in row.get("evidence", [])]
            document = trajectory_document(
                question=row["question"],
                answer=row["answer"],
                query=row.get("query"),
                evidence=evidence,
                reasoning=row.get("reasoning", "Use the supplied evidence to answer the question."),
            )
            handle.write(json.dumps({"text": document}, ensure_ascii=False) + "\n")
            count += 1
    return count


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
