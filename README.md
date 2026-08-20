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
# Edit config/ssh.env, especially TRAIN_CACHE and input paths.
./train-ssh doctor
./train-ssh prepare
./train-ssh all
./train-ssh status
```

The default profile is sized for an eight-hour GPU reservation: prepare large artifacts before the
GPU job when possible, then run smoke, the shortened main run, SFT, and RL. Proxy ablations are
disabled by default. Long stages survive disconnects through tmux or nohup, automatically resume
complete checkpoints, and can also be submitted with
`./train-ssh slurm all` or the editable
`sbatch scripts/slurm/train_h100.sbatch all`. See the
[SSH training guide](docs/SSH_TRAINING.md).

## Training pipeline

Preprocessing is intentionally separate from the timed H100 window.

```bash
# Train the 50K BPE tokenizer from representative reasoning and education samples.
uv run gpt2-reasoning-search train-tokenizer data/tokenizer-sample/*.txt \
  --output artifacts/tokenizer.json --vocab-size 50304

# Stream pinned Hugging Face revisions, filter/deduplicate, and write memory-mapped arrays.
uv run gpt2-reasoning-search prepare-data \
  --tokenizer-path artifacts/tokenizer.json \
  --manifest config/datasets.json \
  --output data/processed \
  --evaluation-prompts data/evaluation/contamination-prompts.jsonl \
  --reasoning-token-cap 2000000000 \
  --general-token-cap 1000000000

# Verify the small-model learning gate and write the one-H100 experiment schedule.
uv run gpt2-reasoning-search smoke-overfit --device cpu
uv run gpt2-reasoning-search experiment-plan

# Main calibrated run. Use step-* or final as --resume-from after interruption.
uv run gpt2-reasoning-search pretrain \
  --reasoning-tokens data/processed/reasoning.bin \
  --general-tokens data/processed/general.bin \
  --output checkpoints/main-350m \
  --preset main-350m --reasoning-ratio 0.70 --max-tokens 2500000000 \
  --time-budget-hours 14
```

Each `.bin` has a manifest containing dtype, token count, hashes, source counts, verification counts,
and rejection statistics. Training checkpoints include model, optimizer, scheduler, RNG, exact data
cursors, and mixture counters.

Run the matched proxies with `--preset proxy-124m` and reasoning ratios `0`, `0.3`, and `0.7`, then:

```bash
uv run gpt2-reasoning-search compare-proxies \
  checkpoints/proxy-r0 checkpoints/proxy-r30 checkpoints/proxy-r70
```

## Search and tool fine-tuning

Wikipedia JSONL rows require `id`, `title`, `url`, and `text`.

```bash
# Default: Tantivy BM25 + SentenceTransformer embeddings + USearch HNSW.
uv run gpt2-reasoning-search build-index data/raw/wikipedia.jsonl \
  --output artifacts/wiki-index

# Or build a deterministic BM25-only index without model downloads.
uv run gpt2-reasoning-search build-index data/raw/wikipedia.jsonl \
  --output artifacts/wiki-index-lexical --lexical-only

uv run gpt2-reasoning-search make-trajectories \
  data/raw/grounded-questions.jsonl \
  --output data/processed/tool-trajectories.jsonl

uv run gpt2-reasoning-search sft-tools \
  --checkpoint checkpoints/main-350m/final \
  --tokenizer-path artifacts/tokenizer.json \
  --trajectories data/processed/tool-trajectories.jsonl \
  --output checkpoints/tool-sft

# Optimize answer, citation, and search behavior against the frozen local index.
uv run gpt2-reasoning-search rl-search \
  --checkpoint checkpoints/tool-sft \
  --tokenizer-path artifacts/tokenizer.json \
  --prompts data/rl/search-qa.jsonl \
  --index artifacts/wiki-index \
  --output checkpoints/search-rl \
  --llm-judge --judge-device cuda
```

Trajectory rows may use the simple `query` + `evidence` form, omit `query` for no-search examples,
or provide `searches`, an array of up to three `{query, evidence}` objects for multi-hop query
reformulation. Retrieved observations and prompts are masked from loss; tool calls, generated
reasoning, answers, and citations are trained.

Search RL is a distinct third stage. It samples groups of online tool trajectories, scores
verifiable QA outcomes and grounded tool behavior, optimizes only model-generated action tokens,
and regularizes against a frozen copy of the tool-SFT checkpoint. RL uses local search only; its
tokens do not alter the audited 70/30 pretraining ratio. See [search RL](docs/SEARCH_RL.md).
The CLI also uses a revision-pinned Qwen3.5-2B auxiliary judge by default. Deterministic answer,
citation, and tool checks remain the primary reward; use `--no-llm-judge` for the ablation.

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
