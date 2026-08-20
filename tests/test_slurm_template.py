from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "slurm" / "train_h100.sbatch"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_path(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for command in ("bash", "date", "dirname", "env", "hostname", "mkdir"):
        target = shutil.which(command)
        assert target is not None, command
        (fake_bin / command).symlink_to(target)
    _write_executable(fake_bin / "nvidia-smi", "printf 'fake H100\\n'\n")
    return fake_bin


def _copy_template_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "school project"
    (project / "scripts" / "slurm").mkdir(parents=True)
    (project / "scripts" / "ssh").mkdir()
    script = project / "scripts" / "slurm" / TEMPLATE.name
    shutil.copy2(TEMPLATE, script)
    return project, script


def test_slurm_template_is_executable_and_valid_bash() -> None:
    assert TEMPLATE.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        ["bash", "-n", str(TEMPLATE)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_slurm_template_has_portable_resources_and_log_paths() -> None:
    lines = TEMPLATE.read_text().splitlines()
    directives = [line for line in lines if line.startswith("#SBATCH ")]

    assert directives == [
        "#SBATCH --job-name=grs-train",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=128G",
        "#SBATCH --time=08:00:00",
        "#SBATCH --output=logs/slurm-%x-%j.out",
        "#SBATCH --error=logs/slurm-%x-%j.err",
    ]
    assert "##SBATCH --partition=REPLACE_ME" in lines
    assert "##SBATCH --account=REPLACE_ME" in lines
    assert not any("--partition=" in line for line in directives)
    assert not any("--account=" in line for line in directives)
    assert not any("--constraint=" in line for line in directives)
    assert max(lines.index(line) for line in directives) < lines.index("set -euo pipefail")


def test_template_delegates_doctor_and_stage_safely_without_nested_sbatch(
    tmp_path: Path,
) -> None:
    project, script = _copy_template_project(tmp_path)
    calls = tmp_path / "worker calls.log"
    config = tmp_path / "cluster config.env"
    config.touch()
    escaped = tmp_path / "should-not-exist"
    _write_executable(
        project / "scripts" / "ssh" / "worker.sh",
        'printf "CONFIG <%s>\\n" "${SSH_TRAIN_CONFIG:-}" >> "$FAKE_CALLS"\n'
        'printf "ARG <%s>\\n" "$@" >> "$FAKE_CALLS"\n',
    )
    fake_bin = _fake_path(tmp_path)
    stage = f"pretrain; touch {escaped}"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin),
            "SSH_TRAIN_CONFIG": str(config),
            "FAKE_CALLS": str(calls),
            "SLURM_JOB_ID": "1234",
        }
    )

    result = subprocess.run(
        [str(script), stage],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Job 1234;")
    assert calls.read_text().splitlines() == [
        f"CONFIG <{config}>",
        "ARG <doctor>",
        f"CONFIG <{config}>",
        f"ARG <{stage}>",
    ]
    assert not escaped.exists()
    executable_lines = "\n".join(
        line for line in TEMPLATE.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    assert re.search(r"\bsbatch\b", executable_lines) is None
    assert 'exec env SSH_TRAIN_CONFIG="$CONFIG_FILE" scripts/ssh/worker.sh "$STAGE"' in (
        TEMPLATE.read_text()
    )


def test_template_rejects_missing_config_before_worker_execution(tmp_path: Path) -> None:
    project, script = _copy_template_project(tmp_path)
    calls = tmp_path / "worker-calls.log"
    _write_executable(
        project / "scripts" / "ssh" / "worker.sh",
        'printf "called\\n" >> "$FAKE_CALLS"\n',
    )
    fake_bin = _fake_path(tmp_path)
    missing_config = tmp_path / "missing.env"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin),
            "SSH_TRAIN_CONFIG": str(missing_config),
            "FAKE_CALLS": str(calls),
        }
    )

    result = subprocess.run(
        [str(script), "all"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert f"Missing {missing_config}" in result.stderr
    assert not calls.exists()


def test_slurm_template_documentation_commands_and_outputs_are_accurate() -> None:
    readme = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs" / "SSH_TRAINING.md").read_text()
    combined = readme + guide

    assert "`scripts/slurm/submit_8h_pipeline.sh`" in readme
    for command in (
        "chmod +x scripts/slurm/train_h100.sbatch",
        "scripts/slurm/submit_8h_pipeline.sh",
        "sbatch scripts/slurm/train_h100.sbatch pretrain",
    ):
        assert command in guide
    assert "logs/slurm-<job-name>-<job-id>.*" in guide
    assert "#SBATCH --output=logs/slurm-%x-%j.out" in TEMPLATE.read_text()
    assert "#SBATCH --error=logs/slurm-%x-%j.err" in TEMPLATE.read_text()
    assert "scripts/slurm/train_h100.sbatch" in combined
