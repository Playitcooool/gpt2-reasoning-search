from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts" / "slurm" / "submit_8h_pipeline.sh"
BATCH = ROOT / "scripts" / "slurm" / "train_h100.sbatch"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _copy_project(tmp_path: Path, *, worker_exit: int = 0) -> tuple[Path, Path]:
    project = tmp_path / "school project"
    (project / "scripts" / "slurm").mkdir(parents=True)
    (project / "scripts" / "ssh").mkdir()
    (project / "config").mkdir()
    shutil.copy2(SUBMITTER, project / "scripts" / "slurm" / SUBMITTER.name)
    shutil.copy2(BATCH, project / "scripts" / "slurm" / BATCH.name)
    _write_executable(
        project / "scripts" / "ssh" / "worker.sh",
        f'{{ echo WORKER; printf "ARG <%s>\\n" "$@"; }} >> "$FAKE_CALLS"\nexit {worker_exit}\n',
    )
    config = project / "config" / "ssh.env"
    config.touch()
    return project, config


def _fake_path(tmp_path: Path, sbatch_body: str) -> Path:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for command in ("bash", "cat", "dirname", "mkdir"):
        target = shutil.which(command)
        assert target is not None, command
        (fake_bin / command).symlink_to(target)
    _write_executable(fake_bin / "sbatch", sbatch_body)
    return fake_bin


def _run_submitter(
    submitter: Path,
    *,
    project: Path,
    config: Path,
    fake_bin: Path,
    calls: Path,
    counter: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin),
            "SSH_TRAIN_CONFIG": str(config),
            "FAKE_CALLS": str(calls),
            "FAKE_COUNTER": str(counter),
        }
    )
    return subprocess.run(
        [str(submitter)],
        cwd=project.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_pipeline_submitter_is_executable_and_bash_3_syntax_compatible() -> None:
    assert SUBMITTER.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        ["bash", "-n", str(SUBMITTER), str(BATCH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "mapfile" not in SUBMITTER.read_text()
    assert "readarray" not in SUBMITTER.read_text()


def test_pipeline_submits_three_independent_eight_hour_jobs_with_afterok_dependencies(
    tmp_path: Path,
) -> None:
    project, config = _copy_project(tmp_path)
    config.write_text(
        "\n".join(
            (
                "SESSION_PREFIX=school-grs",
                "SLURM_ACCOUNT=class-account",
                "SLURM_PARTITION=gpu-school",
                "SLURM_GRES=gpu:h100:1",
                "SLURM_CPUS=8",
                "SLURM_MEMORY=64G",
                "SLURM_TIME=08:00:00",
            )
        )
        + "\n"
    )
    calls = tmp_path / "calls.log"
    counter = tmp_path / "counter"
    fake_bin = _fake_path(
        tmp_path,
        "count=0\n"
        'if [[ -f "$FAKE_COUNTER" ]]; then read -r count < "$FAKE_COUNTER"; fi\n'
        "count=$((count + 1))\n"
        'printf "%s\\n" "$count" > "$FAKE_COUNTER"\n'
        '{ echo CALL; printf "ARG <%s>\\n" "$@"; } >> "$FAKE_CALLS"\n'
        'case "$count" in 1) echo "101;school" ;; 2) echo 202 ;; 3) echo 303 ;; esac\n',
    )

    result = _run_submitter(
        project / "scripts" / "slurm" / SUBMITTER.name,
        project=project,
        config=config,
        fake_bin=fake_bin,
        calls=calls,
        counter=counter,
    )

    assert result.returncode == 0, result.stderr
    groups = calls.read_text().split("CALL\n")[1:]
    assert len(groups) == 3
    assert calls.read_text().split("CALL\n")[0].splitlines() == [
        "WORKER",
        "ARG <prepare>",
    ]
    batch = project / "scripts" / "slurm" / BATCH.name
    assert all("ARG <--parsable>" in group for group in groups)
    assert all(f"ARG <--export=ALL,SSH_TRAIN_CONFIG={config}>" in group for group in groups)
    assert all(f"ARG <{batch}>" in group for group in groups)
    for stage, group in zip(("pretrain", "sft", "rl"), groups, strict=True):
        assert f"ARG <--job-name=school-grs-{stage}>" in group
        assert "ARG <--account=class-account>" in group
        assert "ARG <--partition=gpu-school>" in group
        assert "ARG <--gres=gpu:h100:1>" in group
        assert "ARG <--cpus-per-task=8>" in group
        assert "ARG <--mem=64G>" in group
        assert "ARG <--time=08:00:00>" in group
    assert "ARG <pretrain>" in groups[0]
    assert "--dependency" not in groups[0]
    assert "ARG <--dependency=afterok:101>" in groups[1]
    assert "ARG <sft>" in groups[1]
    assert "ARG <--dependency=afterok:202>" in groups[2]
    assert "ARG <rl>" in groups[2]
    assert "ARG <all>" not in calls.read_text()
    assert "squeue -j 101,202,303" in result.stdout


def test_pipeline_stops_if_scheduler_job_id_is_not_valid(tmp_path: Path) -> None:
    project, config = _copy_project(tmp_path)
    calls = tmp_path / "calls.log"
    counter = tmp_path / "counter"
    fake_bin = _fake_path(
        tmp_path,
        '{ echo CALL; printf "ARG <%s>\\n" "$@"; } >> "$FAKE_CALLS"\necho "not-a-job-id"\n',
    )

    result = _run_submitter(
        project / "scripts" / "slurm" / SUBMITTER.name,
        project=project,
        config=config,
        fake_bin=fake_bin,
        calls=calls,
        counter=counter,
    )

    assert result.returncode == 3
    assert "Could not parse Slurm job id" in result.stderr
    assert calls.read_text().count("CALL") == 1
    assert "ARG <sft>" not in calls.read_text()
    assert "ARG <rl>" not in calls.read_text()


def test_pipeline_checks_config_and_scheduler_before_submission(tmp_path: Path) -> None:
    project, config = _copy_project(tmp_path)
    submitter = project / "scripts" / "slurm" / SUBMITTER.name
    calls = tmp_path / "calls.log"
    counter = tmp_path / "counter"
    fake_bin = _fake_path(tmp_path, 'echo called >> "$FAKE_CALLS"\n')
    config.unlink()

    missing_config = _run_submitter(
        submitter,
        project=project,
        config=config,
        fake_bin=fake_bin,
        calls=calls,
        counter=counter,
    )
    (fake_bin / "sbatch").unlink()
    config.touch()
    missing_scheduler = _run_submitter(
        submitter,
        project=project,
        config=config,
        fake_bin=fake_bin,
        calls=calls,
        counter=counter,
    )

    assert missing_config.returncode == 2
    assert "Missing" in missing_config.stderr
    assert missing_scheduler.returncode == 2
    assert "sbatch is not available" in missing_scheduler.stderr
    assert not calls.exists()


def test_pipeline_prepare_failure_prevents_any_slurm_submission(tmp_path: Path) -> None:
    project, config = _copy_project(tmp_path, worker_exit=75)
    calls = tmp_path / "calls.log"
    counter = tmp_path / "counter"
    fake_bin = _fake_path(
        tmp_path,
        'echo called >> "$FAKE_CALLS"\necho 999\n',
    )

    result = _run_submitter(
        project / "scripts" / "slurm" / SUBMITTER.name,
        project=project,
        config=config,
        fake_bin=fake_bin,
        calls=calls,
        counter=counter,
    )

    assert result.returncode == 75
    assert calls.read_text().splitlines() == ["WORKER", "ARG <prepare>"]
    assert "called" not in calls.read_text()


def test_pipeline_docs_use_direct_submitter_and_never_submit_combined_all() -> None:
    readme = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs" / "SSH_TRAINING.md").read_text()
    setup = (ROOT / "scripts" / "ssh" / "setup.sh").read_text()
    batch = BATCH.read_text()

    for document in (readme, guide, setup, batch):
        assert "scripts/slurm/submit_8h_pipeline.sh" in document
        assert "sbatch scripts/slurm/submit_8h_pipeline.sh" not in document
    assert "sbatch scripts/slurm/train_h100.sbatch all" not in readme + guide + batch
    assert 'STAGE="${1:-pretrain}"' in batch
    assert "dependent jobs" in guide
    assert "separate eight-hour" in guide
    normalized_guide = " ".join(guide.split())
    assert "cancel the still-pending dependent jobs" in normalized_guide
    assert "rerun `scripts/slurm/submit_8h_pipeline.sh`" in normalized_guide
    assert "Completed stages are skipped" in normalized_guide
    assert "incomplete stage resumes" in normalized_guide
