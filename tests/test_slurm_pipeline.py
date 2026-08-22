from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts" / "slurm" / "submit_8h_pipeline.sh"
STAGE_SUBMITTER = ROOT / "scripts" / "slurm" / "submit_stage.sh"
RUNNER = ROOT / "scripts" / "slurm" / "run_stage.sh"
BATCH = ROOT / "scripts" / "slurm" / "train_h100.sbatch"
STAGES = ("prepare", "smoke", "proxies", "pretrain", "sft", "rl", "all")


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _copy_project(tmp_path: Path, *, worker_exit: int = 0) -> tuple[Path, Path]:
    project = tmp_path / "school project"
    (project / "scripts" / "slurm").mkdir(parents=True)
    (project / "scripts" / "ssh").mkdir()
    (project / "config").mkdir()
    shutil.copy2(SUBMITTER, project / "scripts" / "slurm" / SUBMITTER.name)
    shutil.copy2(STAGE_SUBMITTER, project / "scripts" / "slurm" / STAGE_SUBMITTER.name)
    shutil.copy2(RUNNER, project / "scripts" / "slurm" / RUNNER.name)
    for stage in STAGES:
        shutil.copy2(
            ROOT / "scripts" / "slurm" / f"{stage}.sbatch",
            project / "scripts" / "slurm" / f"{stage}.sbatch",
        )
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


def test_slurm_wrappers_are_executable_and_bash_3_syntax_compatible() -> None:
    scripts = [SUBMITTER, STAGE_SUBMITTER, RUNNER, BATCH]
    scripts.extend(ROOT / "scripts" / "slurm" / f"{stage}.sbatch" for stage in STAGES)
    assert all(path.stat().st_mode & stat.S_IXUSR for path in scripts)
    result = subprocess.run(
        ["bash", "-n", *(str(path) for path in scripts)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "mapfile" not in SUBMITTER.read_text()
    assert "readarray" not in SUBMITTER.read_text()


def test_explicit_stage_scripts_delegate_to_common_runner_and_have_resources(
    tmp_path: Path,
) -> None:
    project, _ = _copy_project(tmp_path)
    runner = project / "scripts" / "slurm" / RUNNER.name
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'RUN <%s>\\n' \"$1\" >> \"$FAKE_CALLS\"\n"
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    environment = os.environ.copy()
    environment["FAKE_CALLS"] = str(tmp_path / "stage-calls.log")

    for stage in STAGES:
        script = project / "scripts" / "slurm" / f"{stage}.sbatch"
        result = subprocess.run(
            [str(script)],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, f"{stage}: {result.stderr}"

    calls = (tmp_path / "stage-calls.log").read_text().splitlines()
    assert calls == [f"RUN <{stage}>" for stage in STAGES]
    for stage in STAGES:
        text = (ROOT / "scripts" / "slurm" / f"{stage}.sbatch").read_text()
        assert text.count('run_stage.sh"') == 1
        assert "--gres" not in text
        assert f"--job-name=grs-{stage}" in text
        assert "#SBATCH --cpus-per-task=" in text
        assert "#SBATCH --mem=" in text
        assert "#SBATCH --time=08:00:00" in text
        assert "#SBATCH --output=logs/slurm-%x-%j.out" in text
        assert "#SBATCH --error=logs/slurm-%x-%j.err" in text
        if stage == "prepare":
            assert "#SBATCH --gpus=" not in text
        else:
            assert "#SBATCH --gpus=h100" in text


def test_common_runner_delegates_config_and_doctor_for_gpu_stages(tmp_path: Path) -> None:
    project, config = _copy_project(tmp_path)
    worker = project / "scripts" / "ssh" / "worker.sh"
    _write_executable(
        worker,
        'printf "CONFIG <%s>\\n" "${SSH_TRAIN_CONFIG:-}" >> "$FAKE_CALLS"\n'
        'printf "ARG <%s>\\n" "$@" >> "$FAKE_CALLS"\n',
    )
    calls = tmp_path / "runner-calls.log"
    environment = os.environ.copy()
    environment.update({"SSH_TRAIN_CONFIG": str(config), "FAKE_CALLS": str(calls)})

    prepare = subprocess.run(
        [str(project / "scripts" / "slurm" / RUNNER.name), "prepare"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stderr
    assert calls.read_text().splitlines()[-2:] == [f"CONFIG <{config}>", "ARG <prepare>"]

    pretrain = subprocess.run(
        [str(project / "scripts" / "slurm" / RUNNER.name), "pretrain"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert pretrain.returncode == 0, pretrain.stderr
    assert calls.read_text().splitlines()[-4:] == [
        f"CONFIG <{config}>",
        "ARG <doctor>",
        f"CONFIG <{config}>",
        "ARG <pretrain>",
    ]


def test_common_runner_rejects_unknown_stage_without_worker_execution(tmp_path: Path) -> None:
    project, config = _copy_project(tmp_path)
    calls = tmp_path / "runner-calls.log"
    environment = os.environ.copy()
    environment.update({"SSH_TRAIN_CONFIG": str(config), "FAKE_CALLS": str(calls)})
    result = subprocess.run(
        [str(project / "scripts" / "slurm" / RUNNER.name), "not-a-stage"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "Unknown Slurm stage" in result.stderr
    assert not calls.exists()


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
                "SLURM_GPUS=h100",
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
    stage_scripts = {
        stage: project / "scripts" / "slurm" / f"{stage}.sbatch"
        for stage in ("pretrain", "sft", "rl")
    }
    assert all("ARG <--parsable>" in group for group in groups)
    assert all(f"ARG <--export=ALL,SSH_TRAIN_CONFIG={config}>" in group for group in groups)
    for stage, group in zip(("pretrain", "sft", "rl"), groups, strict=True):
        assert f"ARG <{stage_scripts[stage]}>" in group
        assert f"ARG <--job-name=school-grs-{stage}>" in group
        assert "ARG <--account=class-account>" in group
        assert "ARG <--partition=gpu-school>" in group
        assert "ARG <--gpus=h100>" in group
        assert "ARG <--cpus-per-task=8>" in group
        assert "ARG <--mem=64G>" in group
        assert "ARG <--time=08:00:00>" in group
    assert "--dependency" not in groups[0]
    assert "ARG <--dependency=afterok:101>" in groups[1]
    assert "ARG <--dependency=afterok:202>" in groups[2]
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


def test_submit_stage_applies_config_resources_and_dependency_safely(tmp_path: Path) -> None:
    project, config = _copy_project(tmp_path)
    config.write_text(
        "\n".join(
            (
                "SESSION_PREFIX=school-grs",
                "SLURM_ACCOUNT=class-account",
                "SLURM_PARTITION=gpu-school",
                "SLURM_GPUS=h100",
                "SLURM_CPUS=12",
                "SLURM_MEMORY=96G",
                "SLURM_TIME=07:30:00",
            )
        )
        + "\n"
    )
    calls = tmp_path / "calls.log"
    counter = tmp_path / "counter"
    fake_bin = _fake_path(
        tmp_path,
        '{ echo CALL; printf "ARG <%s>\\n" "$@"; } >> "$FAKE_CALLS"\n'
        'echo "918;cluster"\n',
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin),
            "SSH_TRAIN_CONFIG": str(config),
            "FAKE_CALLS": str(calls),
            "FAKE_COUNTER": str(counter),
        }
    )

    prepare = subprocess.run(
        [str(project / "scripts" / "slurm" / STAGE_SUBMITTER.name), "prepare"],
        cwd=project.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stderr
    prepare_call = calls.read_text()
    assert "ARG <--job-name=school-grs-prepare>" in prepare_call
    assert "ARG <--cpus-per-task=12>" in prepare_call
    assert "ARG <--mem=96G>" in prepare_call
    assert "ARG <--time=07:30:00>" in prepare_call
    assert "ARG <--account=class-account>" in prepare_call
    assert "ARG <--partition=gpu-school>" in prepare_call
    assert "ARG <--gpus=" not in prepare_call
    assert f"ARG <--export=ALL,SSH_TRAIN_CONFIG={config}>" in prepare_call
    assert f"ARG <{project / 'scripts' / 'slurm' / 'prepare.sbatch'}>" in prepare_call

    pretrain = subprocess.run(
        [str(project / "scripts" / "slurm" / STAGE_SUBMITTER.name), "pretrain", "918"],
        cwd=project.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert pretrain.returncode == 0, pretrain.stderr
    pretrain_call = calls.read_text().split("CALL\n")[2]
    assert "ARG <--gpus=h100>" in pretrain_call
    assert "ARG <--dependency=afterok:918>" in pretrain_call
    assert f"ARG <{project / 'scripts' / 'slurm' / 'pretrain.sbatch'}>" in pretrain_call
    assert pretrain.stdout.strip() == "918"


def test_submit_stage_rejects_unknown_stage_before_scheduler_call(tmp_path: Path) -> None:
    project, config = _copy_project(tmp_path)
    calls = tmp_path / "calls.log"
    fake_bin = _fake_path(tmp_path, 'echo "called" >> "$FAKE_CALLS"\necho 123\n')
    environment = os.environ.copy()
    environment.update(
        {"PATH": str(fake_bin), "SSH_TRAIN_CONFIG": str(config), "FAKE_CALLS": str(calls)}
    )
    result = subprocess.run(
        [str(project / "scripts" / "slurm" / STAGE_SUBMITTER.name), "unknown"],
        cwd=project.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "Choose a stage" in result.stderr
    assert not calls.exists()


def test_submit_stage_ignores_legacy_slurm_gres_and_defaults_to_h100_gpus(
    tmp_path: Path,
) -> None:
    project, config = _copy_project(tmp_path)
    config.write_text("SESSION_PREFIX=legacy\nSLURM_GRES=gpu:h100:1\n")
    calls = tmp_path / "calls.log"
    fake_bin = _fake_path(
        tmp_path,
        '{ echo CALL; printf "ARG <%s>\\n" "$@"; } >> "$FAKE_CALLS"\necho 456\n',
    )
    environment = os.environ.copy()
    environment.update(
        {"PATH": str(fake_bin), "SSH_TRAIN_CONFIG": str(config), "FAKE_CALLS": str(calls)}
    )

    result = subprocess.run(
        [str(project / "scripts" / "slurm" / STAGE_SUBMITTER.name), "pretrain"],
        cwd=project.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = calls.read_text()
    assert "ARG <--gpus=h100>" in log
    assert "--gres" not in log
    assert "gpu:h100:1" not in log
    assert result.stdout.strip() == "456"


def test_pipeline_docs_use_direct_submitter_and_never_submit_combined_all() -> None:
    readme = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs" / "SSH_TRAINING.md").read_text()
    setup = (ROOT / "scripts" / "ssh" / "setup.sh").read_text()
    batch = BATCH.read_text()

    for document in (readme, guide, setup):
        assert "scripts/slurm/submit_8h_pipeline.sh" in document
        assert "sbatch scripts/slurm/submit_8h_pipeline.sh" not in document
    assert "sbatch scripts/slurm/train_h100.sbatch all" not in readme + guide + batch
    assert all(f"{stage}.sbatch" in guide for stage in STAGES)
    assert "scripts/slurm/submit_stage.sh <stage>" in guide
    assert "SLURM_GPUS" in readme + guide + setup
    assert "--gpus" in guide
    assert "SLURM_GRES" not in readme + guide + setup
    assert "sbatch --gres" not in readme + guide + setup
    assert "not converted into a `--gres` request" in guide
    assert 'STAGE="${1:-pretrain}"' in batch
    assert "dependent jobs" in guide
    assert "Each stage receives an eight-hour reservation" in guide
    normalized_guide = " ".join(guide.split())
    assert "cancel the still-pending dependent jobs" in normalized_guide
    assert "rerun `scripts/slurm/submit_8h_pipeline.sh`" in normalized_guide
    assert "Completed stages are skipped" in normalized_guide
    assert "incomplete stage resumes" in normalized_guide
