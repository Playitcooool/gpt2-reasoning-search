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
