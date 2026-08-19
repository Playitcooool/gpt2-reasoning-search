# GPT-2 Reasoning Search

A research prototype for training a modern GPT-2-style decoder from random initialization on a
reasoning-heavy corpus, then teaching it to call deterministic local search or optional live web
search. The project defaults to a 70% verified-reasoning / 30% educational-text token mixture.

This repository contains the full data, training, indexing, evaluation, API, and CLI pipelines. It
does not include downloaded corpora or trained weights.

## Quick start

```bash
uv sync --dev
uv run gpt2-reasoning-search --help
```

The generated reasoning field is model-produced scratch work for research inspection. It is not a
faithful explanation of the model's internal computation.

## Pipeline

The commands below intentionally separate preprocessing from the timed H100 run.

```bash
# 1. Train the project tokenizer from representative plain-text samples.
uv run gpt2-reasoning-search train-tokenizer data/tokenizer-sample/*.txt

# 2. Stream the pinned corpora and create token arrays.
uv run gpt2-reasoning-search prepare-data \
  --tokenizer-path artifacts/tokenizer.json \
  --reasoning-token-cap 2000000000 --general-token-cap 1000000000

# 3. Run the calibrated 70/30 main pretraining job on an H100.
uv run gpt2-reasoning-search pretrain \
  --reasoning-tokens data/processed/reasoning.npy \
  --general-tokens data/processed/general.npy

# Proxy mixture controls use --preset proxy-124m with --reasoning-ratio 0, 0.3, and 0.7.

# 4. Build deterministic retrieval and tool trajectories.
uv run gpt2-reasoning-search build-index data/raw/wikipedia.jsonl
uv run gpt2-reasoning-search make-trajectories data/raw/grounded-questions.jsonl

# 5. Teach the pretrained model to call search.
uv run gpt2-reasoning-search sft-tools \
  --checkpoint checkpoints/main-350m/final \
  --tokenizer-path artifacts/tokenizer.json \
  --trajectories data/processed/tool-trajectories.jsonl
```

`wikipedia.jsonl` rows require `id`, `title`, `url`, and `text`. Grounded-question rows require
`question` and `answer`; optional fields are `query`, `reasoning`, and an `evidence` array containing
the public search-result schema (`id`, `title`, `url`, `snippet`, `content`, and optional `score`). A
missing `query` deliberately produces a no-search training example.

## Serving

```bash
uv run gpt2-reasoning-search serve \
  --checkpoint checkpoints/tool-sft \
  --tokenizer-path artifacts/tokenizer.json \
  --index artifacts/wiki-index
```

`POST /v1/answer` accepts `query`, `search_mode` (`auto`, `local`, `web`, or `off`), and up to three
searches. Live web search is enabled only when `BRAVE_SEARCH_API_KEY` is present. Retrieved text is
treated as untrusted evidence, model control tokens in results are neutralized, and returned
citations are restricted to source identifiers actually observed during the request.
