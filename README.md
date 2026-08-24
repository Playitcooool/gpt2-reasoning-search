# GPT-2 Reasoning Search

An English research prototype for training a modernized GPT-2-style decoder from random
initialization, with a deliberately unusual **70% verified-reasoning / 30% educational-text**
pretraining mixture, followed by supervised and reinforcement-learned search-tool training.

The repository contains reproducible data preparation, tokenizer, model, pretraining, hybrid
Wikipedia retrieval, optional Brave web search, tool SFT, online search RL, evaluation, CLI, and
FastAPI serving.
Downloaded data and trained weights are not included. A 350M model trained for one H100-day should
be treated as a narrow experiment, not a production-grade general assistant.

## What is mainstream, and what is different?

The model and systems stack follows common current practice: RoPE, RMSNorm, SwiGLU, grouped-query
attention, PyTorch SDPA/Flash Attention, bf16, fused AdamW, gradient checkpointing, KV-cached
generation, SafeTensors checkpoints, BM25+dense HNSW retrieval, reciprocal-rank fusion, cross-encoder
reranking, validated JSON tools, bounded retries, caching, and service backpressure.

The intentional research variable is the **data mixture**. Pretraining consumes exactly 70%
reasoning and 30% general text by non-padding tokens. Equal-token 124M proxies at 0%, 30%, and 70%
must be reported even when the 70% run does not win.

## Install

```bash
git clone https://github.com/Playitcooool/gpt2-reasoning-search.git
cd gpt2-reasoning-search
uv sync --dev --locked
uv run gpt2-reasoning-search --help
```

Python 3.11 is pinned. See [the H100 runbook](docs/H100_RUNBOOK.md) before starting a GPU run.

### Easy SSH / school-cluster workflow

For a remote H100, the shortest path is:

```bash
./train-ssh setup
# Edit only SLURM_ACCOUNT, SLURM_TIME, and (if required) SLURM_PARTITION/SLURM_GPUS.
# Cache and data paths are automatic.
# This prepares data, then submits pretrain -> SFT -> RL as dependent eight-hour jobs.
scripts/slurm/submit_8h_pipeline.sh
./train-ssh status
```

The default profile is sized for an eight-hour GPU reservation. Preparation runs before GPU jobs,
then pretraining, tool SFT, and search RL each get a separate 7.5-hour training budget. Checkpoints
are resumable; the pretrain job runs the small CUDA smoke gate once before training, and proxy
ablations are disabled by default. See the [SSH training guide](docs/SSH_TRAINING.md).

Each worker stage also has its own Slurm script (`scripts/slurm/pretrain.sbatch`, `sft.sbatch`,
`rl.sbatch`, and the supporting `prepare.sbatch`, `smoke.sbatch`, `proxies.sbatch`, and `all.sbatch`).
Use `scripts/slurm/submit_stage.sh <stage>` to apply the account and time from `config/ssh.env`, or
call an `.sbatch` file directly with its checked-in eight-hour defaults.

When data has not been prepared yet, `scripts/slurm/submit_stage.sh pretrain` automatically queues a
CPU-only preparation job and makes the H100 pretraining job wait for it. It prints the pretraining
job ID; use the full pipeline command above when you also want SFT and RL queued automatically.

For a direct server without Slurm, use the same worker one stage at a time:

```bash
./train-ssh setup
./train-ssh doctor
./train-ssh prepare
# Wait for `./train-ssh logs prepare` to report completion.
./train-ssh pretrain
./train-ssh sft
./train-ssh rl
```

## Data and training pipeline

`prepare` is the only data command you need. It idempotently runs the pinned bootstrap, tokenizer
training, 70/30 corpus preparation, local Wikipedia index build, and tool-trajectory generation.
It writes `data/processed/reasoning.bin`, `data/processed/general.bin`, `artifacts/tokenizer.json`,
`artifacts/wiki-index`, and `data/processed/tool-trajectories.jsonl`.

```bash
./train-ssh prepare
```

Each `.bin` has a manifest containing dtype, token count, hashes, source counts, verification counts,
and rejection statistics. Training checkpoints include model, optimizer, scheduler, RNG, exact data
cursors, and mixture counters.

The worker invokes tokenizer training, pretraining, `sft-tools`, and `rl-search` internally. Search
RL uses the local index by default, the Qwen3.5-2B judge by default, and at most three searches per
rollout. Export `BRAVE_SEARCH_API_KEY` immediately before submitting the RL job to enable live Brave
search with a persistent cache and automatic local fallback when the API is unavailable. Set
`USE_LLM_JUDGE=0` for the deterministic ablation. Detailed JSONL schemas and advanced CLI
equivalents are in [the search-RL guide](docs/SEARCH_RL.md).

Search RL is a distinct third stage. It samples groups of online tool trajectories, scores
verifiable QA outcomes and grounded tool behavior, optimizes only model-generated action tokens,
and regularizes against a frozen copy of the tool-SFT checkpoint. RL uses local search by default;
when a Brave key is exported it can use live search with a cached local fallback. Its tokens do not
alter the audited 70/30 pretraining ratio. See [search RL](docs/SEARCH_RL.md).
The CLI also uses a revision-pinned Qwen3.5-2B auxiliary judge by default. Deterministic answer,
citation, and tool checks remain the primary reward; use `USE_LLM_JUDGE=0` for the ablation.

The optional 0%/30%/70% proxy comparison is not part of the eight-hour pipeline. Run it only with
a separate reservation; `experiment-plan` and `compare-proxies` are diagnostics for that study.

## Serving

```bash
uv run gpt2-reasoning-search serve \
  --checkpoint checkpoints/search-rl/final \
  --tokenizer-path artifacts/tokenizer.json \
  --index artifacts/wiki-index
```

`POST /v1/answer` accepts `query`, `search_mode` (`auto`, `local`, `web`, or `off`), and
`max_searches` from 0 to 3. `auto` uses the deterministic local index first and only falls back to
web when configured and local retrieval fails or returns nothing. Set `BRAVE_SEARCH_API_KEY` to
enable live search. Liveness, readiness, and service counters are exposed at `/health/live`,
`/health/ready`, and `/metrics`.

The response includes the answer, generated scratch work, citations, complete tool trace, finish
reason, token counts, and timings. Scratch work is useful for research inspection but is not a
faithful description of internal computation.

## Evaluation

```bash
uv run gpt2-reasoning-search score-lm artifacts/fineweb-losses.jsonl
uv run gpt2-reasoning-search score-reasoning artifacts/reasoning-predictions.jsonl
uv run gpt2-reasoning-search benchmark-grounded data/evaluation/grounded.jsonl \
  --checkpoint checkpoints/tool-sft \
  --tokenizer-path artifacts/tokenizer.json \
  --index artifacts/wiki-index --mode off --mode local
uv run gpt2-reasoning-search score-grounded artifacts/grounded-predictions.jsonl
```

Reports include answer EM/F1, retrieval Recall/MRR/nDCG, citation precision/recall/validity, valid
tool-call rate, unnecessary-search and query-recovery rates, latency percentiles, and per-mode
search-off/search-on comparisons. See [evaluation protocol](docs/EVALUATION.md),
[data provenance](docs/DATA.md), [architecture](docs/ARCHITECTURE.md),
[search RL](docs/SEARCH_RL.md), and [security](SECURITY.md).
