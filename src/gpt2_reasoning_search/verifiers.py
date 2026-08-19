"""Deterministic filters for reasoning examples."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def normalize_math_answer(value: str) -> str:
    value = value.strip().rstrip(".").replace("$", "").replace("\\,", "").replace(" ", "")
    value = re.sub(r"^\\boxed\{(.*)\}$", r"\1", value)
    return value.lower()


def extract_boxed_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1]
    final = re.search(r"(?:final answer|answer is)\s*[:=]?\s*([^\n]+)", text, re.IGNORECASE)
    return final.group(1).strip() if final else None


def verify_math_trace(trace: str, expected: str) -> bool:
    answer = extract_boxed_answer(trace)
    return answer is not None and normalize_math_answer(answer) == normalize_math_answer(expected)


@dataclass(slots=True)
class CodeVerificationResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool = False


def verify_python_code(
    code: str, tests: str, timeout_seconds: float = 3.0
) -> CodeVerificationResult:
    harness = f"{code}\n\n{tests}\n"
    with tempfile.TemporaryDirectory(prefix="reasoning-code-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(harness)
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={"PYTHONHASHSEED": "0"},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CodeVerificationResult(False, exc.stdout or "", exc.stderr or "", True)
    return CodeVerificationResult(result.returncode == 0, result.stdout, result.stderr)


def verify_logic_choice(prediction: str, expected: str) -> bool:
    def choice(value: str) -> str | None:
        match = re.search(r"\b([A-E])\b", value.upper())
        return match.group(1) if match else None

    return choice(prediction) is not None and choice(prediction) == choice(expected)


def write_verification_report(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
