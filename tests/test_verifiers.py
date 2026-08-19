import time

from gpt2_reasoning_search.verifiers import (
    extract_boxed_answer,
    normalize_math_answer,
    verify_logic_choice,
    verify_math_trace,
    verify_python_code,
)


def test_math_answer_extraction_and_normalization() -> None:
    assert extract_boxed_answer(r"work \boxed{42}") == "42"
    assert extract_boxed_answer("Therefore, final answer: 17\n") == "17"
    assert normalize_math_answer(r" $\boxed{1\, 024}$. ") == "1024"
    assert verify_math_trace(r"steps... \boxed{3/4}", " 3/4. ")
    assert not verify_math_trace("No final result", "3/4")


def test_code_verifier_pass_failure_and_timeout() -> None:
    passed = verify_python_code("def add(a, b): return a + b", "assert add(2, 3) == 5")
    failed = verify_python_code("def add(a, b): return a - b", "assert add(2, 3) == 5")
    started = time.monotonic()
    timed_out = verify_python_code("while True: pass", "", timeout_seconds=0.1)

    assert passed.passed and not passed.timed_out
    assert not failed.passed and "AssertionError" in failed.stderr
    assert not timed_out.passed and timed_out.timed_out
    assert time.monotonic() - started < 2.0


def test_logic_choice_uses_standalone_option_letters() -> None:
    assert verify_logic_choice("After checking, option C is correct.", "C")
    assert not verify_logic_choice("option B", "C")
    assert not verify_logic_choice("CAB", "C")
