# Easy training over SSH

This workflow is for a single H100 reached through SSH, including school clusters where jobs must
survive disconnects. It needs no administrator access. Long direct-server runs use `tmux` when it is
available and fall back to `nohup`; Slurm clusters can submit the same stages with `sbatch`.

## Slurm start (recommended)

```bash
git clone git@github.com:Playitcooool/gpt2-reasoning-search.git
cd gpt2-reasoning-search
./train-ssh setup
```

Edit `config/ssh.env` only for school Slurm resources (`SLURM_ACCOUNT`, `SLURM_TIME`, and, if needed,
`SLURM_PARTITION`/`SLURM_GPUS`). Cache and data paths are selected automatically. Then:

```bash
scripts/slurm/submit_8h_pipeline.sh
squeue -u "$USER"
./train-ssh status
```

`SLURM_GPUS="h100"` becomes `sbatch --gpus=h100`, matching clusters that use the modern Slurm GPU
syntax. It is intentionally not converted into a `--gres` request.

The wrapper prepares the data before submitting the GPU jobs, then submits pretrain -> SFT -> RL with
`afterok` dependencies and passes the configured account, GPU, memory, CPU, and wall-time settings
to `sbatch`. Each stage receives an eight-hour reservation and trains for 7.5 hours,
leaving margin for startup and checkpoint finalization. The pretrain job runs the small CUDA smoke
gate once before training, and it skips proxy ablations by default.

## Direct SSH start (without Slurm)

Use this mode only when the H100 is already allocated to your shell:

```bash
./train-ssh setup       # once per checkout; creates config/ssh.env
./train-ssh doctor
./train-ssh prepare
./train-ssh pretrain
./train-ssh sft
./train-ssh rl
```

Each stage survives disconnects through `tmux` or `nohup`. Follow progress from a new login:

```bash
./train-ssh status
./train-ssh logs pretrain
./train-ssh attach pretrain
```

Detach from tmux with `Ctrl-b`, then `d`. Re-running an individual stage skips completed outputs and
resumes the newest complete `step-*` checkpoint.

## Automatic data preparation

`./train-ssh prepare` creates the fixed tokenizer samples, downloads the pinned reasoning and
FineWeb-Edu streams, downloads a pinned English Wikipedia snapshot sample, generates grounded
questions and search-RL prompts, builds the local index, and creates tool trajectories. It records
the Wikipedia revision and generated-input SHA-256 hashes in `artifacts/auto-data-manifest.json`.

Run it on a networked login/preprocessing node before reserving the GPU. If compute nodes cannot
access the internet, copy the resulting `artifacts/`, `data/raw/`, `data/processed/`, and
`artifacts/wiki-index/` directories with hash-preserving tooling; no path edits are required.

In direct SSH mode, `prepare` launches in the background; wait for `./train-ssh logs prepare` to
report completion before launching another stage. The Slurm wrapper waits for preparation itself. A
process lock prevents two preparation/training stages from corrupting the same outputs.

## Run one stage at a time

This is safer when the school gives several shorter reservations:

```bash
./train-ssh smoke
./train-ssh pretrain
./train-ssh sft
./train-ssh rl
```

The eight-hour profile gives the 350M main run a 7.5-hour training budget and a 2.5B-token cap.
SFT and RL are separate stages with 7.5-hour training budgets and 30 minutes reserved for startup
and checkpoint finalization. This restores the full 2.5B maximum while keeping each stage below its
eight-hour allocation. The proxy comparison remains available with `./train-ssh proxies` but
requires a separate reservation.

The profile is selected by `TRAIN_PROFILE="8h"` in `config/ssh.env`. Existing config files are
automatically treated as `8h` after upgrading; set `TRAIN_PROFILE="custom"` if you deliberately
want to use manually chosen budgets.

Every command is idempotent around completed outputs. `AUTO_RESUME=1` is the default. Set it to `0`
only when deliberately starting a clean output directory.

For a quick pipeline rehearsal, lower `REASONING_TOKEN_CAP`, `GENERAL_TOKEN_CAP`, `PROXY_TOKEN_CAP`,
`MAIN_TOKEN_CAP`, and the time budgets in `config/ssh.env`. Do not compare that rehearsal with the
full experiment.

## Slurm stage scripts, retries, and logs

There is one checked-in batch script for every worker stage. The config-aware submitter is the
easiest way to use them because it applies `SLURM_ACCOUNT`, `SLURM_TIME`, partition, GPU, CPU, and
memory settings from `config/ssh.env`:

```bash
scripts/slurm/submit_stage.sh prepare   # optional CPU preparation job
scripts/slurm/submit_stage.sh smoke
scripts/slurm/submit_stage.sh proxies   # optional proxy ablations
scripts/slurm/submit_stage.sh pretrain
scripts/slurm/submit_stage.sh sft
scripts/slurm/submit_stage.sh rl
```

If the fixed training data does not exist, `submit_stage.sh pretrain` first queues `prepare` as a
CPU-only job, then submits pretraining with an `afterok` dependency. The command prints the
pretraining job ID and does not spend H100 time on downloads or preprocessing. Supplying an explicit
dependency, or setting `AUTO_PREPARE=0`, leaves dependency management in your control.

If you want to call Slurm directly, the corresponding scripts are
`scripts/slurm/prepare.sbatch`, `smoke.sbatch`, `proxies.sbatch`, `pretrain.sbatch`, `sft.sbatch`,
`rl.sbatch`, and `all.sbatch`:

```bash
sbatch scripts/slurm/pretrain.sbatch
sbatch scripts/slurm/sft.sbatch
sbatch scripts/slurm/rl.sbatch
```

Direct `sbatch` uses the safe eight-hour defaults written in each file. Use `submit_stage.sh` when
you want the values from `config/ssh.env` without editing scripts. Every stage writes Slurm
stdout/stderr to `logs/slurm-<job-name>-<job-id>.*` and runs the same safe, auto-resuming worker as
`train-ssh`. The combined `all` stage is blocked by the default 8-hour profile so it cannot
accidentally exceed the reservation. `train_h100.sbatch` remains as a compatibility shim for old
commands; new runs should use the explicit stage names above.

If a stage reaches its eight-hour limit, cancel the still-pending dependent jobs from that chain and
rerun `scripts/slurm/submit_8h_pipeline.sh`. Completed stages are skipped, and the incomplete stage
resumes its newest complete checkpoint before new `afterok` dependencies are created. For a
single-stage retry, use `scripts/slurm/submit_stage.sh <stage>` (or the matching explicit `.sbatch`
file) and launch the next stage only after its final checkpoint exists.

## Useful commands

```bash
./train-ssh help
./train-ssh foreground smoke   # debugging without tmux/nohup
./train-ssh logs pretrain
./train-ssh attach rl
```

Logs stay under `logs/`; model state stays under `checkpoints/`; large dependency and model caches
stay under `TRAIN_CACHE`. None of these are committed to Git.
