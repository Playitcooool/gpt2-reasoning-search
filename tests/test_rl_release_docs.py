import re
import tomllib
from pathlib import Path

from click.utils import strip_ansi
from typer.main import get_command
from typer.testing import CliRunner

from gpt2_reasoning_search.api import create_app
from gpt2_reasoning_search.cli import app
from gpt2_reasoning_search.experiment import one_h100_schedule
from gpt2_reasoning_search.judge import (
    DEFAULT_QWEN_JUDGE_MODEL,
    DEFAULT_QWEN_JUDGE_REVISION,
)

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent_across_package_cli_and_openapi() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    command = get_command(app)
    version = command.commands["version"]
    api = create_app()

    assert project["project"]["version"] == "0.3.0"
    assert version.name == "version"
    assert api.version == "0.3.0"
    assert api.openapi()["info"]["version"] == "0.3.0"
    assert 'version = "0.3.0"' in (ROOT / "uv.lock").read_text()


def test_rl_cli_registers_documented_paths_options_and_safe_retrieval_defaults() -> None:
    command = get_command(app).commands["rl-search"]
    parameters = {parameter.name: parameter for parameter in command.params}

    expected_options = {
        "checkpoint": "--checkpoint",
        "tokenizer_path": "--tokenizer-path",
        "prompts": "--prompts",
        "index": "--index",
        "output": "--output",
        "epochs": "--epochs",
        "group_size": "--group-size",
        "max_searches": "--max-searches",
        "learning_rate": "--learning-rate",
        "kl_coefficient": "--kl-coefficient",
        "resume_from": "--resume-from",
        "enable_reranker": "--enable-reranker",
        "retrieval_device": "--retrieval-device",
        "llm_judge": "--llm-judge",
        "judge_model": "--judge-model",
        "judge_revision": "--judge-revision",
        "judge_device": "--judge-device",
    }
    for name, option in expected_options.items():
        assert option in parameters[name].opts

    assert parameters["checkpoint"].required is True
    assert parameters["tokenizer_path"].required is True
    assert parameters["prompts"].required is True
    assert parameters["index"].required is True
    assert parameters["output"].default == Path("checkpoints/search-rl")
    assert parameters["group_size"].default == 4
    assert parameters["max_searches"].default == 3
    assert parameters["enable_reranker"].default is False
    assert parameters["retrieval_device"].default == "cpu"
    assert parameters["llm_judge"].default is True
    assert "--no-llm-judge" in parameters["llm_judge"].secondary_opts
    assert parameters["judge_model"].default == DEFAULT_QWEN_JUDGE_MODEL
    assert parameters["judge_revision"].default is None
    assert re.fullmatch(r"[0-9a-f]{40}", DEFAULT_QWEN_JUDGE_REVISION)
    assert parameters["judge_device"].default == "cuda"


def test_rl_cli_enables_pinned_judge_by_default_and_supports_ablation(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"question":"q","answer":"a"}\n')
    index = tmp_path / "index"
    index.mkdir()
    captured = []

    def fake_train(config):
        captured.append(config)
        return config.output_directory / "final"

    monkeypatch.setattr("gpt2_reasoning_search.cli.train_search_rl", fake_train)
    base_args = [
        "rl-search",
        "--checkpoint",
        str(checkpoint),
        "--tokenizer-path",
        str(tokenizer),
        "--prompts",
        str(prompts),
        "--index",
        str(index),
        "--output",
        str(tmp_path / "output"),
    ]

    enabled = CliRunner().invoke(app, base_args)
    disabled = CliRunner().invoke(app, [*base_args, "--no-llm-judge"])
    invalid_custom = CliRunner().invoke(
        app, [*base_args, "--judge-model", "organization/custom-judge"]
    )

    assert enabled.exit_code == 0
    assert captured[0].judge_model == DEFAULT_QWEN_JUDGE_MODEL
    assert captured[0].judge_revision == DEFAULT_QWEN_JUDGE_REVISION
    assert captured[0].judge_device == "cuda"
    assert disabled.exit_code == 0
    assert captured[1].judge_model is None
    assert captured[1].judge_revision is None
    assert invalid_custom.exit_code != 0
    normalized_error = " ".join(strip_ansi(invalid_custom.output).split())
    assert "custom --judge-model requires --judge-revision" in normalized_error


def test_search_rl_documentation_matches_local_training_and_checkpoint_behavior() -> None:
    readme = (ROOT / "README.md").read_text()
    rl_guide = (ROOT / "docs" / "SEARCH_RL.md").read_text()
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    model_card = " ".join((ROOT / "MODEL_CARD.md").read_text().split())
    security = " ".join((ROOT / "SECURITY.md").read_text().split())

    documented_command = " ".join(readme.split())
    for option in (
        "--checkpoint checkpoints/tool-sft",
        "--tokenizer-path artifacts/tokenizer.json",
        "--prompts data/rl/search-qa.jsonl",
        "--index artifacts/wiki-index",
        "--output checkpoints/search-rl",
    ):
        assert option in documented_command
    assert "--checkpoint checkpoints/search-rl/final" in documented_command
    assert "RL uses local search only" in readme

    assert "Only model-generated tokens receive policy gradients" in rl_guide
    assert "frozen copy of the tool-SFT checkpoint" in rl_guide
    assert "The reference always remains the original `--checkpoint`" in rl_guide
    assert "epoch, prompt cursor, rollout count, and action-token count resume" in rl_guide
    assert "embedding model defaults to CPU" in rl_guide
    assert "reranker is disabled by default" in rl_guide
    assert "Use only the frozen local index for RL" in rl_guide
    assert "grouped online trajectories against the frozen local index" in architecture
    assert "grouped online RL against a frozen local" in model_card
    assert "Search RL must use the frozen local index" in security


def test_search_rl_docs_cover_input_schema_reward_components_and_metrics() -> None:
    guide = (ROOT / "docs" / "SEARCH_RL.md").read_text()

    for field in ("question", "answer", "supporting_ids", "search_required"):
        assert f'"{field}"' in guide
    assert "`id` is optional" in guide
    for term in (
        "answer exact match and token F1",
        "citation precision/recall",
        "citation validity",
        "valid tool-call rate",
        "query recovery",
        "unnecessary searches",
        "invalid/duplicate calls",
        "each search attempt",
    ):
        assert term in guide
    for metric in (
        "total reward",
        "reward variance",
        "KL",
        "gradient norm",
        "action-token count",
        "learning rate",
        "metrics.jsonl",
    ):
        assert metric in guide


def test_auxiliary_judge_defaults_and_ablation_are_documented() -> None:
    readme = " ".join((ROOT / "README.md").read_text().split())
    guide = " ".join((ROOT / "docs" / "SEARCH_RL.md").read_text().split())
    runbook = " ".join((ROOT / "docs" / "H100_RUNBOOK.md").read_text().split())
    security = " ".join((ROOT / "SECURITY.md").read_text().split())

    assert "Qwen3.5-2B" in readme
    for document in (guide, runbook):
        assert "Qwen/Qwen3.5-2B" in document
        assert "revision-pinned" in document
    assert "--llm-judge --judge-device cuda" in readme
    assert "--no-llm-judge" in readme
    assert "--no-llm-judge" in guide
    assert "scores for answer correctness, evidence support, and search quality" in guide
    assert "prompt injection" in security
    assert "judge" in security


def test_h100_runbook_allocation_totals_exactly_one_day_and_includes_rl() -> None:
    runbook = (ROOT / "docs" / "H100_RUNBOOK.md").read_text()
    allocation = runbook.split("## Original 24-hour allocation", 1)[1].split(
        "The trainer measures", 1
    )[0]
    hours = [float(value) for value in re.findall(r"(?:up to )?(\d+(?:\.\d+)?) hours?", allocation)]

    assert hours == [0.5, 4.5, 14.0, 1.5, 2.0, 1.5]
    assert sum(hours) == 24.0
    assert "Search RL: 2 hours" in allocation
    assert (
        "--output checkpoints/search-rl --epochs 8 --group-size 4 --time-budget-hours 7.5"
        in " ".join(runbook.split())
    )

    scheduled = one_h100_schedule()
    proxies = [run for run in scheduled if run.preset == "proxy-124m"]
    main = [run for run in scheduled if run.preset == "main-350m"]
    assert [run.time_budget_hours for run in proxies] == [1.5, 1.5, 1.5]
    assert len(main) == 1 and main[0].time_budget_hours == 14.0


def test_rl_evaluation_and_safety_docs_require_held_out_regression_gates() -> None:
    evaluation = (ROOT / "docs" / "EVALUATION.md").read_text()
    model_card = " ".join((ROOT / "MODEL_CARD.md").read_text().split())
    security = " ".join((ROOT / "SECURITY.md").read_text().split())

    assert "tool-SFT versus RL checkpoints" in evaluation
    assert "prompts excluded from RL training" in evaluation
    assert "Training reward alone is not an evaluation metric" in evaluation
    assert "reward-hacking" in model_card
    assert "high training reward is not evidence of a better model" in model_card
    assert "Do not place secrets, private" in security
    assert "held-out answer, citation, tool-validity, and search-restraint gates" in security
