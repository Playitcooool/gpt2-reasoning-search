# Easy training over SSH

This workflow is for a single H100 reached through SSH, including school clusters where jobs must
survive disconnects. It needs no administrator access. Long direct-server runs use `tmux` when it is
available and fall back to `nohup`; Slurm clusters can submit the same stages with `sbatch`.

## Three-command start

```bash
git clone git@github.com:Playitcooool/gpt2-reasoning-search.git
cd gpt2-reasoning-search
./train-ssh setup
```

Edit `config/ssh.env`. At minimum, put `TRAIN_CACHE` on fast scratch storage and confirm the input
paths. Then:

```bash
./train-ssh doctor
./train-ssh prepare       # do this before reserving the GPU when possible
./train-ssh pretrain      # wait for completion, then run SFT and RL
```

The default profile is sized for an eight-hour GPU reservation. Prepare artifacts before reserving
the GPU when possible; run `pretrain`, then `sft`, then `rl` as each stage completes. It skips proxy
ablations by default because they cannot fit in eight hours. Each stage survives an SSH disconnect.
Follow progress from a new login:

```bash
cd gpt2-reasoning-search
./train-ssh status
./train-ssh logs pretrain
# Or interact with the tmux job:
./train-ssh attach pretrain
```

Detach from tmux with `Ctrl-b`, then `d`. Re-running an individual stage skips completed outputs and
resumes the newest complete `step-*` checkpoint.

## Required input files

Before `prepare`, provide:

- representative tokenizer text files under `data/tokenizer-sample/`;
- `data/raw/wikipedia.jsonl` with `id`, `title`, `url`, and `text` fields;
- `data/raw/grounded-questions.jsonl` for tool-SFT trajectories;
- `data/rl/search-qa.jsonl` for search RL;
- preferably `data/evaluation/contamination-prompts.jsonl` before corpus filtering.

The pinned reasoning and FineWeb-Edu datasets stream from Hugging Face during `prepare` (or an
explicit `PREPARE_IN_JOB=1` custom profile). If compute
nodes cannot access the internet, run `prepare` on a networked login/preprocessing node or copy the
resulting tokenizer, `.bin` corpora, and Wikipedia index to the paths in `config/ssh.env`.

Because `prepare` launches in the background, wait for `./train-ssh logs prepare` to report
completion before launching another stage. A process lock prevents two preparation/training stages
from corrupting the same outputs.

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

## Slurm school cluster

Ask the administrator for the correct partition, account, memory, and GPU resource syntax, then set
the `SLURM_*` values in `config/ssh.env`:

```bash
scripts/slurm/submit_8h_pipeline.sh
squeue -u "$USER"
./train-ssh status
```

If you prefer a normal editable `sbatch` file, use the included template. Edit its `#SBATCH` lines
for your partition/account before submitting; these scheduler directives must be present when Slurm
accepts the job and cannot be read later from `config/ssh.env`.

```bash
chmod +x scripts/slurm/train_h100.sbatch
# Submit all GPU stages as separate dependent jobs:
scripts/slurm/submit_8h_pipeline.sh
# Or submit/retry one stage:
sbatch scripts/slurm/train_h100.sbatch pretrain
```

The script writes Slurm stdout/stderr to `logs/slurm-<job-name>-<job-id>.*` and runs the same safe,
auto-resuming worker as `train-ssh`. Each stage gets a separate eight-hour allocation. The combined
`all` stage is blocked by the default 8-hour profile so it cannot accidentally exceed the reservation.

If a stage reaches its eight-hour limit, cancel the still-pending dependent jobs from that chain and
rerun `scripts/slurm/submit_8h_pipeline.sh`. Completed stages are skipped, and the incomplete stage
resumes its newest complete checkpoint before new `afterok` dependencies are created. For a
single-stage retry, submit `sbatch scripts/slurm/train_h100.sbatch <stage>` and launch the next
stage only after its final checkpoint exists.

## Useful commands

```bash
./train-ssh help
./train-ssh foreground smoke   # debugging without tmux/nohup
./train-ssh logs pretrain
./train-ssh attach rl
```

Logs stay under `logs/`; model state stays under `checkpoints/`; large dependency and model caches
stay under `TRAIN_CACHE`. None of these are committed to Git.
