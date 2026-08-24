"""Build evidence-grounded search trajectories from JSONL question data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .agent import TOOL_INSTRUCTION, format_tool_results
from .data import atomic_write_lines
from .schemas import SearchArguments, SearchResult, ToolCall


def trajectory_document(
    question: str,
    answer: str,
    query: str | None,
    evidence: list[SearchResult],
    reasoning: str,
) -> str:
    # Keep the fixed context byte-for-byte aligned with SearchAgent.answer().  SFT masks this
    # instruction and the retrieved observations, then learns only the tool call and grounded
    # response that follow them.
    prefix = f"{TOOL_INSTRUCTION}\n<|problem|>\n{question.strip()}\n"
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
    parts = [f"{TOOL_INSTRUCTION}\n<|problem|>\n{question.strip()}\n"]
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

    def validated_searches(row: dict) -> list[tuple[str, list[SearchResult]]]:
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("trajectory row is missing a non-empty question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("trajectory row is missing a non-empty answer")
        search_required = row.get("search_required")
        if search_required is not None and not isinstance(search_required, bool):
            raise ValueError("trajectory row has invalid search_required")

        if "searches" in row:
            raw_searches = row["searches"]
            if not isinstance(raw_searches, list) or len(raw_searches) > 3:
                raise ValueError("trajectory row has invalid searches")
            searches = []
            for step in raw_searches:
                if not isinstance(step, dict) or not isinstance(step.get("query"), str):
                    raise ValueError("trajectory search step is missing query")
                if not step["query"].strip() or not isinstance(step.get("evidence", []), list):
                    raise ValueError("trajectory search step has invalid evidence")
                searches.append(
                    (
                        step["query"],
                        [SearchResult.model_validate(item) for item in step.get("evidence", [])],
                    )
                )
        else:
            query = row.get("query")
            if query is not None and (not isinstance(query, str) or not query.strip()):
                raise ValueError("trajectory row has invalid query")
            raw_evidence = row.get("evidence", [])
            if not isinstance(raw_evidence, list):
                raise ValueError("trajectory row has invalid evidence")
            searches = (
                [(query, [SearchResult.model_validate(item) for item in raw_evidence])]
                if query is not None
                else []
            )

        has_search = bool(searches)
        has_evidence = any(evidence for _query, evidence in searches)
        requires_search = bool(search_required) if search_required is not None else has_search
        if requires_search and (not has_search or not has_evidence):
            raise ValueError("search-required trajectory needs a successful search with evidence")
        if not requires_search and has_search:
            raise ValueError("no-search trajectory must not contain a search call")
        return searches

    def serialized_rows() -> Iterable[str]:
        nonlocal count
        for row in rows:
            reasoning = row.get("reasoning", "Use the supplied evidence to answer the question.")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("trajectory row has invalid reasoning")
            searches = validated_searches(row)
            if "searches" in row:
                document = multi_step_trajectory_document(
                    row["question"], row["answer"], searches, reasoning
                )
            else:
                query, evidence = searches[0] if searches else (None, [])
                document = trajectory_document(
                    question=row["question"],
                    answer=row["answer"],
                    query=query,
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
