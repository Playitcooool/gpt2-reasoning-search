# Model card: GPT-2 Reasoning Search 350M

This repository defines a model and training recipe; it does not currently publish trained weights
or measured benchmark claims.

The intended model is an English, approximately 350M-parameter causal decoder trained from random
initialization on a 70% verified-reasoning / 30% educational-text token mixture, fine-tuned on
bounded search trajectories, and optionally optimized with grouped online RL against a frozen local
index. It is intended for research into small-model reasoning-data mixtures, retrieval grounding,
and tool-call reliability.

It is not intended for high-stakes medical, legal, financial, security, or autonomous decisions.
Expected limitations include shallow reasoning, hallucination, brittle multi-hop behavior, prompt
injection susceptibility, source bias, stale local knowledge, and generated scratch work that may
not faithfully describe internal computation. Search grounding reduces but does not eliminate these
risks. RL adds reward-hacking, mode-collapse, verbosity, and capability-regression risks; a high
training reward is not evidence of a better model.

The default RL command uses a pinned Qwen3.5-2B model as a low-weight semantic judge. Its ratings
can be biased, inconsistent, or vulnerable to injected text, so deterministic answer and source
provenance rewards remain primary. Report a matched `--no-llm-judge` ablation and a human-rated
judge-calibration sample.

Any weight release should add exact training tokens, source/filter counts, licenses, compute and
emissions estimates, checkpoint/tokenizer/data hashes, complete proxy comparisons, benchmark
contamination notes, safety evaluation, and known failure examples.
