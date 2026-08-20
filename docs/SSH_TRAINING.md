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
./train-ssh all
```

`all` prepares missing artifacts and then runs every training stage in one background session. It
survives an SSH disconnect. Follow progress from a new login:

```bash
cd gpt2-reasoning-search
./train-ssh status
./train-ssh logs all
# Or interact with the tmux job:
./train-ssh attach all
```

Detach from tmux with `Ctrl-b`, then `d`. Re-running `./train-ssh all` or an individual stage skips
completed outputs and resumes the newest complete `step-*` checkpoint.

## Required input files

Before `prepare`, provide:

- representative tokenizer text files under `data/tokenizer-sample/`;
- `data/raw/wikipedia.jsonl` with `id`, `title`, `url`, and `text` fields;
- `data/raw/grounded-questions.jsonl` for tool-SFT trajectories;
- `data/rl/search-qa.jsonl` for search RL;
- preferably `data/evaluation/contamination-prompts.jsonl` before corpus filtering.

The pinned reasoning and FineWeb-Edu datasets stream from Hugging Face during `prepare` (or the
preparation phase at the start of `all`). If compute
nodes cannot access the internet, run `prepare` on a networked login/preprocessing node or copy the
resulting tokenizer, `.bin` corpora, and Wikipedia index to the paths in `config/ssh.env`.

Because `prepare` launches in the background, wait for `./train-ssh logs prepare` to report
completion before launching another stage. A process lock prevents two preparation/training stages
from corrupting the same outputs.

## Run one stage at a time

This is safer when the school gives several shorter reservations:

```bash
./train-ssh smoke
./train-ssh proxies
./train-ssh pretrain
./train-ssh sft
./train-ssh rl
```

Every command is idempotent around completed outputs. `AUTO_RESUME=1` is the default. Set it to `0`
only when deliberately starting a clean output directory.

For a quick pipeline rehearsal, lower `REASONING_TOKEN_CAP`, `GENERAL_TOKEN_CAP`, `PROXY_TOKEN_CAP`,
`MAIN_TOKEN_CAP`, and the time budgets in `config/ssh.env`. Do not compare that rehearsal with the
full experiment.

## Slurm school cluster

Ask the administrator for the correct partition, account, memory, and GPU resource syntax, then set
the `SLURM_*` values in `config/ssh.env`:

```bash
./train-ssh slurm all
squeue -u "$USER"
./train-ssh status
```

If the 24-hour job limit is shorter than the full pipeline, submit stages separately. A timed-out
stage can be submitted again and will resume its newest complete checkpoint.

## Useful commands

```bash
./train-ssh help
./train-ssh foreground smoke   # debugging without tmux/nohup
./train-ssh logs pretrain
./train-ssh attach rl
```

Logs stay under `logs/`; model state stays under `checkpoints/`; large dependency and model caches
stay under `TRAIN_CACHE`. None of these are committed to Git.
