# Evaluation protocol

Freeze prompts and supporting document identifiers before data preparation so contamination filters
can exclude them. Record benchmark version, split, sample count, prompt template, decoding settings,
checkpoint hash, tokenizer hash, index manifest, seed, and whether search is local, web, or disabled.

## Required suites

- Language modeling: token-weighted held-out FineWeb-Edu loss and perplexity.
- Math: contamination-checked arithmetic, GSM8K, and MATH-500.
- Code: HumanEval+ and MBPP with code execution in a network-disabled hardened sandbox.
- Logic: exact-solver subsets where deterministic validation exists.
- Retrieval: frozen HotpotQA-style multi-hop questions and a small reasoning-intensive retrieval set.
- Tools: held-out no-search, one-search, reformulation, empty-result, and malformed-call cases.
- RL: matched tool-SFT versus RL checkpoints, including reward-component and KL curves, evaluated on
  prompts excluded from RL training.

Run grounded records with identical examples and decoding settings for `off` and `local`. Live web
results are diagnostic only because they change over time. Report answer EM/F1, retrieval
Recall/MRR/nDCG, citation precision/recall/validity, valid-call rate, search rate, unnecessary-search
rate, query-recovery rate, and p50/p95 latency.

## JSONL schemas

- `score-lm`: each row contains `loss` (finite, non-negative) and `tokens` (positive integer).
- `score-reasoning`: each row contains `task`, `prediction`, and `answer` strings.
- `benchmark-grounded` input: each row contains `question` and `answer`; `id`, `supporting_ids`, and
  `search_required` are optional.
- `score-grounded`: consumes benchmark output or equivalent rows containing `prediction`, `answer`,
  `search_mode`, `queries`, `tool_calls_total`, `valid_tool_calls`, `retrieved_ids`,
  `supporting_ids`, `cited_ids`, `search_required`, `answer_found`, and `elapsed_seconds`.

`citation_validity` asks whether a cited ID was returned. `citation_precision` asks whether it is in
the annotated supporting set. These are different: a citation can be syntactically valid but not
support the answer.

## Decision rule

The proxy comparison is causal only when the 0%, 30%, and 70% runs use equal token budgets and the
same tokenizer, architecture, optimizer schedule, and evaluation data. Publish all three. If 70%
does not improve the predeclared reasoning metrics, report the result and diagnostics without a
reasoning-improvement claim.

Tool acceptance requires at least 95% valid calls on held-out tool examples and citation validity of
100%. Stable training, exact checkpoint resume, and auditable mixture counters are separate gates;
good downstream scores do not excuse a failed gate.

For RL acceptance, require non-zero within-group reward variance on a useful fraction of training
prompts. Compare answer accuracy, citation validity, valid-call rate, and unnecessary-search rate
against the tool-SFT checkpoint. Training reward alone is not an evaluation metric.

When the auxiliary LLM judge is enabled, hand-rate a fixed sample for answer correctness, evidence
support, and search quality. Report judge validity plus agreement/error against those ratings, and
run the same seed and budget with `--no-llm-judge`. Never use the judge's own score as the only
checkpoint-selection metric.
