# One-H100 runbook

## Before the timed run

Clone the repository onto fast local storage and install the locked environment:

```bash
git clone git@github.com:Playitcooool/gpt2-reasoning-search.git
cd gpt2-reasoning-search
uv sync --dev --locked
uv run python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
```

Confirm an H100-class CUDA device, bf16 support, enough local capacity for datasets/checkpoints, and
the expected revisions in `config/datasets.json`. Prepare tokenizer, token arrays, contamination
prompts, and Wikipedia index before the 24-hour GPU window when possible. Copy these artifacts with
hash-preserving tooling and compare their manifests after transfer.

Run `smoke-overfit` before spending the main budget. Keep the 0/30/70 proxy token budgets equal.

## Suggested allocation

- Throughput calibration and smoke gates: 0.5 hour.
- Three equal-budget 124M proxies: 4.5 hours total.
- 350M 70% main pretraining: up to 14 hours.
- Tool SFT: 1.5 hours.
- Search RL: 2 hours.
- Held-out evaluation and buffer: 1.5 hours.

The trainer measures steady-state throughput after compile warmup and reduces the main token cap to
fit the configured wall-clock window. The 2.5B cap is a maximum, not a promise that one H100 can
consume it in a day.

## Resume and monitoring

Metrics are appended to `metrics.jsonl`. Watch loss, gradient norm, learning rate, tokens/second,
MFU estimate, peak memory, reasoning/general token counters, and the calibrated final token budget.
Resume only from a complete checkpoint directory:

```bash
uv run gpt2-reasoning-search pretrain \
  --reasoning-tokens data/processed/reasoning.bin \
  --general-tokens data/processed/general.bin \
  --output checkpoints/main-350m \
  --resume-from checkpoints/main-350m/step-00001000
```

Do not combine metrics from runs with different tokenizer hashes, data hashes, model configuration,
or token budgets. Preserve failed or neutral results; do not relabel a shorter run as the planned
2.5B-token experiment.

## Serving after training

Build the index, run tool SFT, then run grouped local-search RL:

```bash
uv run gpt2-reasoning-search rl-search \
  --checkpoint checkpoints/tool-sft \
  --tokenizer-path artifacts/tokenizer.json \
  --prompts data/rl/search-qa.jsonl \
  --index artifacts/wiki-index \
  --output checkpoints/search-rl --group-size 4 \
  --llm-judge --judge-device cuda
```

The default auxiliary judge is revision-pinned `Qwen/Qwen3.5-2B`, run greedily with a 4,096-token
input cap. Its roughly 2B parameters are practical on an 80 GB H100 alongside the 350M policy and
frozen reference when generation is sequential. Do not use the model's 262K maximum context here.

Compare tool-SFT and RL checkpoints on the frozen held-out set before deploying the RL checkpoint.
Then start the service. Keep the API bound to localhost unless it
is behind authenticated TLS termination. Configure secrets through the host secret manager, not a
repository file. The default single-request concurrency avoids unsafe simultaneous generation on
one model instance; raise it only after memory and latency measurement.
