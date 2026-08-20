from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "train-ssh"
SCRIPT_NAMES = ("setup.sh", "worker.sh", "status.sh")
STAGES = ("prepare", "smoke", "proxies", "pretrain", "sft", "rl", "all")


def _copy_workflow(tmp_path: Path) -> Path:
    project = tmp_path / "school project"
    (project / "scripts" / "ssh").mkdir(parents=True)
    (project / "config").mkdir()
    shutil.copy2(LAUNCHER, project / "train-ssh")
    for name in SCRIPT_NAMES:
        shutil.copy2(ROOT / "scripts" / "ssh" / name, project / "scripts" / "ssh" / name)
    shutil.copy2(ROOT / "config" / "ssh.env.example", project / "config" / "ssh.env.example")
    return project


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_bin(tmp_path: Path, *system_commands: str) -> Path:
    directory = tmp_path / "fake-bin"
    directory.mkdir(exist_ok=True)
    for command in ("bash", "dirname", *system_commands):
        target = shutil.which(command)
        assert target is not None, command
        (directory / command).symlink_to(target)
    return directory


def _write_config(project: Path, tmp_path: Path, **overrides: object) -> Path:
    data = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    values: dict[str, object] = {
        "TRAIN_CACHE": tmp_path / "cache",
        "SESSION_PREFIX": "school-grs",
        "AUTO_RESUME": 1,
        "TOKENIZER_INPUT_DIR": data / "tokenizer",
        "TOKENIZER_PATH": data / "tokenizer.json",
        "EVALUATION_PROMPTS": data / "evaluation.jsonl",
        "REASONING_TOKENS": data / "reasoning.bin",
        "GENERAL_TOKENS": data / "general.bin",
        "WIKIPEDIA_JSONL": data / "wikipedia.jsonl",
        "GROUNDED_QUESTIONS": data / "grounded.jsonl",
        "RL_PROMPTS": data / "rl.jsonl",
        "TRAJECTORIES": data / "trajectories.jsonl",
        "WIKI_INDEX": data / "wiki-index",
        "REASONING_TOKEN_CAP": 100,
        "GENERAL_TOKEN_CAP": 100,
        "LEXICAL_ONLY": 1,
        "CHECKPOINT_ROOT": outputs,
        "PROXY_TOKEN_CAP": 60,
        "PROXY_HOURS": 1.5,
        "MAIN_TOKEN_CAP": 100,
        "MAIN_HOURS": 14,
        "MAIN_OUTPUT": outputs / "main",
        "SFT_OUTPUT": outputs / "sft",
        "SFT_EPOCHS": 1,
        "SFT_MICRO_BATCH": 2,
        "SFT_GRAD_ACCUM": 2,
        "RL_OUTPUT": outputs / "rl",
        "RL_EPOCHS": 1,
        "RL_GROUP_SIZE": 2,
        "USE_LLM_JUDGE": 0,
        "SLURM_PARTITION": "gpu-school",
        "SLURM_ACCOUNT": "class-account",
        "SLURM_GRES": "gpu:h100:1",
        "SLURM_CPUS": 8,
        "SLURM_MEMORY": "64G",
        "SLURM_TIME": "12:00:00",
    }
    values.update(overrides)
    path = project / "config" / "ssh.env"
    config_lines = (f"{name}={shlex.quote(str(value))}" for name, value in values.items())
    path.write_text("\n".join(config_lines))
    return path


def _run(
    command: list[str | Path],
    *,
    project: Path,
    config: Path | None = None,
    path: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if config is not None:
        environment["SSH_TRAIN_CONFIG"] = str(config)
    if path is not None:
        environment["PATH"] = path
    environment.update(extra_env or {})
    return subprocess.run(
        [str(value) for value in command],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _fake_recorder(path: Path, name: str, *, exit_code: int = 0, stdout: str = "") -> None:
    _write_executable(
        path,
        f'{{ echo "CALL {name}"; printf "ARG <%s>\\n" "$@"; }} >> "$FAKE_CALLS"\n'
        + (f"printf '%s\\n' {shlex.quote(stdout)}\n" if stdout else "")
        + f"exit {exit_code}\n",
    )


def _complete_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng.pt", "state.json"):
        (path / name).touch()


def _wait_for_file(path: Path) -> str:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text()
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path}")


def test_cluster_entrypoints_are_executable_and_bash_syntax_is_valid() -> None:
    paths = [LAUNCHER, *(ROOT / "scripts" / "ssh" / name for name in SCRIPT_NAMES)]
    for path in paths:
        assert path.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        ["bash", "-n", *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_help_and_stage_validation_do_not_require_config(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    help_result = _run([project / "train-ssh", "--help"], project=project)
    invalid = _run([project / "train-ssh", "foreground", "download"], project=project)

    assert help_result.returncode == 0
    for stage in STAGES:
        assert f"./train-ssh {stage}" in help_result.stdout
    assert invalid.returncode == 2
    assert "Choose: prepare, smoke, proxies, pretrain, sft, rl, or all" in invalid.stderr


def test_setup_creates_config_and_uses_locked_uv_without_network(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "cp", "mkdir")
    _fake_recorder(fake_bin / "uv", "uv", stdout="0.3.0")
    environment = {
        "FAKE_CALLS": str(calls),
        "HOME": str(tmp_path / "home"),
        "SCRATCH": str(tmp_path / "scratch"),
    }

    result = _run(
        [project / "train-ssh", "setup"],
        project=project,
        path=str(fake_bin),
        extra_env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert (project / "config" / "ssh.env").read_text() == (
        project / "config" / "ssh.env.example"
    ).read_text()
    log = calls.read_text()
    assert "ARG <sync>" in log and "ARG <--dev>" in log and "ARG <--locked>" in log
    assert "ARG <gpt2-reasoning-search>" in log and "ARG <version>" in log


def test_doctor_uses_fake_gpu_and_uv_and_reports_pending_inputs(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir", "df", "awk")
    _fake_recorder(fake_bin / "uv", "uv", stdout="CUDA: True bf16: True")
    _fake_recorder(fake_bin / "nvidia-smi", "nvidia-smi", stdout="NVIDIA H100, 81920 MiB, 555")

    result = _run(
        [project / "train-ssh", "doctor"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 0, result.stderr
    assert "NVIDIA H100" in result.stdout
    assert "CUDA: True bf16: True" in result.stdout
    assert result.stdout.count("PENDING:") == 6
    assert "ARG <--query-gpu=name,memory.total,driver_version>" in calls.read_text()


def test_doctor_fails_cleanly_when_gpu_and_uv_are_unavailable(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    fake_bin = _fake_bin(tmp_path, "mkdir", "df", "awk")

    result = _run(
        [project / "train-ssh", "doctor"],
        project=project,
        config=config,
        path=str(fake_bin),
    )

    assert result.returncode == 1
    assert "MISSING: uv" in result.stdout
    assert "MISSING: nvidia-smi" in result.stdout


def test_doctor_fails_when_torch_cannot_use_cuda_or_bf16(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir", "df", "awk")
    _fake_recorder(fake_bin / "uv", "uv", stdout="CUDA: False bf16: False")
    _fake_recorder(fake_bin / "nvidia-smi", "nvidia-smi", stdout="NVIDIA H100")

    result = _run(
        [project / "train-ssh", "doctor"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 1
    assert "CUDA: False bf16: False" in result.stdout


def test_tmux_launcher_preserves_paths_and_stage_as_single_arguments(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir")
    _write_executable(
        fake_bin / "tmux",
        'if [[ "$1" == "has-session" ]]; then exit 1; fi\n'
        '{ echo "CALL tmux"; printf "ARG <%s>\\n" "$@"; } >> "$FAKE_CALLS"\n',
    )

    result = _run(
        [project / "train-ssh", "pretrain"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 0, result.stderr
    log = _wait_for_file(calls)
    assert "ARG <new-session>" in log
    assert "ARG <-s>" in log and "ARG <school-grs-pretrain>" in log
    assert f"SSH_TRAIN_CONFIG='{config}'" in log
    assert f"'{project / 'scripts' / 'ssh' / 'worker.sh'}' 'pretrain'" in log


def test_nohup_fallback_uses_argument_safe_env_invocation(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir")
    _fake_recorder(fake_bin / "nohup", "nohup")

    result = _run(
        [project / "train-ssh", "rl"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 0, result.stderr
    log = _wait_for_file(calls)
    assert "ARG <env>" in log
    assert f"ARG <SSH_TRAIN_CONFIG={config}>" in log
    assert f"ARG <{project / 'scripts' / 'ssh' / 'worker.sh'}>" in log
    assert "ARG <rl>" in log
    assert (project / "logs" / "rl.pid").is_file()


def test_slurm_submission_uses_configured_resources_and_validated_stage(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir")
    _fake_recorder(fake_bin / "sbatch", "sbatch", stdout="Submitted batch job 42")

    result = _run(
        [project / "train-ssh", "slurm", "all"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )
    invalid = _run(
        [project / "train-ssh", "slurm", "unknown"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 0
    log = calls.read_text()
    for argument in (
        "--job-name=school-grs-all",
        "--gres=gpu:h100:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=12:00:00",
        "--partition=gpu-school",
        "--account=class-account",
    ):
        assert f"ARG <{argument}>" in log
    assert f"ARG <--export=ALL,SSH_TRAIN_CONFIG={config}>" in log
    assert "ARG <all>" in log
    assert invalid.returncode == 2
    assert "Choose:" in invalid.stderr


@pytest.mark.parametrize("command", ["logs", "attach"])
def test_log_commands_reject_unknown_or_path_traversal_stages(tmp_path: Path, command: str) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir", "touch")
    _fake_recorder(fake_bin / "tail", "tail")

    result = _run(
        [project / "train-ssh", command, "../../outside"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 2
    assert "Choose:" in result.stderr
    assert not (tmp_path / "outside.log").exists()
    assert not calls.exists()


def test_pretrain_auto_resume_selects_latest_complete_checkpoint_only(tmp_path: Path) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "reasoning.bin").touch()
    (inputs / "general.bin").touch()
    output = tmp_path / "outputs" / "main"
    _complete_checkpoint(output / "step-00000010")
    _complete_checkpoint(output / "step-00000020")
    (output / "step-00000030").mkdir()
    (output / "step-00000030" / "state.json").touch()
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir", "tee", "date", "sort", "rm")
    _fake_recorder(fake_bin / "uv", "uv")

    result = _run(
        [project / "scripts" / "ssh" / "worker.sh", "pretrain"],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 0, result.stderr
    log = calls.read_text()
    assert "ARG <--resume-from>" in log
    assert f"ARG <{output / 'step-00000020'}>" in log
    assert str(output / "step-00000030") not in log


def test_completed_pretrain_stage_skips_uv_and_missing_inputs_fail_before_uv(
    tmp_path: Path,
) -> None:
    project = _copy_workflow(tmp_path)
    output = tmp_path / "outputs" / "main"
    (output / "final").mkdir(parents=True)
    completed_config = _write_config(project, tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir", "tee", "date", "sort", "rm")
    _fake_recorder(fake_bin / "uv", "uv")

    completed = _run(
        [project / "scripts" / "ssh" / "worker.sh", "pretrain"],
        project=project,
        config=completed_config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )
    shutil.rmtree(output / "final")
    missing = _run(
        [project / "scripts" / "ssh" / "worker.sh", "pretrain"],
        project=project,
        config=completed_config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert completed.returncode == 0
    assert "already complete" in completed.stdout
    assert missing.returncode == 2
    assert "Missing required file" in missing.stdout + missing.stderr
    assert not calls.exists() or calls.read_text() == ""


@pytest.mark.parametrize("stage", ["sft", "rl"])
def test_completed_downstream_stage_skips_missing_inputs_and_uv(tmp_path: Path, stage: str) -> None:
    project = _copy_workflow(tmp_path)
    config = _write_config(project, tmp_path)
    outputs = tmp_path / "outputs"
    if stage == "sft":
        marker = outputs / "sft" / "model.safetensors"
        marker.parent.mkdir(parents=True)
        marker.touch()
    else:
        marker = outputs / "rl" / "final"
        marker.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_bin(tmp_path, "mkdir", "tee", "date", "sort", "rm")
    _fake_recorder(fake_bin / "uv", "uv")

    result = _run(
        [project / "scripts" / "ssh" / "worker.sh", stage],
        project=project,
        config=config,
        path=str(fake_bin),
        extra_env={"FAKE_CALLS": str(calls)},
    )

    assert result.returncode == 0
    assert "already complete" in result.stdout
    assert not calls.exists()


def test_config_is_ignored_and_documented_commands_match_launcher() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    example = (ROOT / "config" / "ssh.env.example").read_text()
    guide = (ROOT / "docs" / "SSH_TRAINING.md").read_text()
    readme = (ROOT / "README.md").read_text()
    launcher = LAUNCHER.read_text()

    assert "config/ssh.env" in ignored
    assert (
        "config/ssh.env"
        not in subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    for variable in (
        "TRAIN_CACHE",
        "AUTO_RESUME",
        "REASONING_TOKENS",
        "GENERAL_TOKENS",
        "MAIN_HOURS",
        "RL_OUTPUT",
        "SLURM_GRES",
        "SLURM_TIME",
    ):
        assert f"{variable}=" in example
    for command in (
        "setup",
        "doctor",
        "smoke",
        "proxies",
        "pretrain",
        "sft",
        "rl",
        "all",
        "status",
        "logs",
        "attach",
        "foreground",
        "slurm",
    ):
        assert f"./train-ssh {command}" in guide + readme
        assert command in launcher
    assert "./train-ssh logs prepare" in guide
    assert "./train-ssh all" in guide
