# Search reinforcement learning

Search RL is a separate post-training stage after tool SFT. It optimizes generated reasoning, JSON
tool calls, query reformulation, citations, and final answers while treating prompts and retrieved
observations as fixed environment state.

The implementation is GRPO-style rather than a complete reproduction of PPO/GRPO. For each QA
prompt it samples a group of fresh on-policy trajectories, executes calls against Brave when live
web search is configured (with local BM25 only as outage fallback), scores each complete trajectory,
normalizes rewards within the group, and performs one policy update. It does not reuse a rollout for
multiple clipped PPO epochs.

For group rewards `r_i`, advantages are:

```text
A_i = (r_i - mean(r)) / (std(r) + 1e-4)
```

Only model-generated tokens receive policy gradients. User prompts, controller instructions, and
tool observations are context but not actions. A frozen copy of the tool-SFT checkpoint supplies a
sampled reverse-ratio estimator of `KL(policy || reference)` to limit drift.

## Input

RL JSONL rows require:

```json
{
  "id": "question-001",
  "question": "Which city ...?",
  "answer": "Paris",
  "supporting_sources": [
    {"id": "wiki-page:3", "url": "https://en.wikipedia.org/wiki/Paris"}
  ],
  "search_required": true
}
```

`id` is optional. Each `supporting_sources` object identifies one accepted source. It may contain an
index `id`, a canonical public `url`, or both. When both identify the same source, either a local
Wikipedia result or a Brave result for that URL can satisfy the target. Legacy `supporting_ids` and
`supporting_urls` lists are also accepted, but paired `supporting_sources` avoid double-counting.
Questions without a required lookup set `search_required` to `false`; keep them in the mix to teach
search restraint. Do not use evaluation questions for RL.

## Reward

The default scalar reward combines:

- normalized answer exact match and token F1;
- citation precision/recall against supporting IDs;
- citation validity against results actually returned in the trajectory;
- valid tool-call rate;
- genuine query recovery after a failed/non-supporting first retrieval;
- a bonus for a successful, correctly cited search on lookup-required questions;
- penalties for missing required grounding, unnecessary searches, invalid/duplicate calls, and
  searches beyond the first necessary lookup;
- bounded Qwen3.5-2B scores for answer correctness, evidence support, and search quality.

Fabricated IDs cannot earn citation-support credit, even if they coincidentally match a gold ID,
unless that ID was returned. Query recovery requires distinct valid queries, a failed first lookup,
a later supporting lookup, and a correct answer. Every component, total reward, reward variance, KL,
gradient norm, action-token count, and learning rate is written to `metrics.jsonl`.
Judge latency and valid-output rate are recorded so its H100 cost and formatting reliability remain
visible.

For `search_required: true`, exact-match, F1, and all Qwen judge components are **gated** on a
successful search plus a citation to a returned result that matches an accepted supporting source.
An answer from model memory alone therefore receives a missing-grounding penalty even if its text is
correct. The first required search is free; only further searches incur the per-search cost. This is
the central incentive: retrieve evidence, cite it, then answer from it.

The LLM score is auxiliary: its combined maximum weight is 0.40, while exact-answer, citation,
and tool provenance checks remain authoritative. An invalid judge response earns zero judge reward
and sets the `judge_valid` diagnostic to zero rather than stopping training. This avoids turning a
small, fallible judge into the only definition of success.

The default judge is [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), an Apache-2.0
post-trained model, at pinned revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. It runs in its default non-thinking mode with greedy
decoding, a 4,096-token input cap, a 128-token output cap, and an exact JSON schema. Returned
evidence is serialized as untrusted data and capped before judging. Calibrate its scores against a
small human-rated validation set; a 2B judge is suitable for a training signal, not a final arbiter.

## Run

```bash
uv run gpt2-reasoning-search rl-search \
  --checkpoint checkpoints/tool-sft \
  --tokenizer-path artifacts/tokenizer.json \
  --prompts data/rl/search-qa.jsonl \
  --index artifacts/wiki-index \
  --output checkpoints/search-rl \
  --epochs 8 --group-size 4 --time-budget-hours 7.5 --max-searches 3 \
  --llm-judge --judge-device cuda
```

Export `BRAVE_SEARCH_API_KEY` and use `--search-mode web` to train against live Brave results.
Successful results are cached for 30 days inside the RL output directory. If Brave reaches a
quota/rate limit or becomes unavailable, it is disabled for the rest of that run and each attempted
web search falls back to compact local BM25 retrieval rather than aborting the training job. This
makes the run resilient, but changes its retrieval environment; record `web_searches` and
`local_fallbacks` from `metrics.jsonl` and do not treat it as a fully reproducible benchmark.

There is no dense-vector index or reranker. The compact local BM25 index is retained solely as an
offline fallback for API outage and quota handling.
The revision-pinned Qwen judge is enabled by default in the CLI. Use `--no-llm-judge` for a matched
deterministic-only ablation; changing `--judge-model` also requires an explicit pinned
`--judge-revision`.

## SFT alignment

Tool SFT and RL use the same fixed `TOOL_INSTRUCTION`, JSON `search` call grammar, untrusted-result
wrapper, `<|reasoning|>`, `<|answer|>`, and returned-source citation format. SFT masks that fixed
context and all retrieved evidence, and trains only model actions: tool calls, query reformulations,
reasoning, answers, and citations. Its prepared trajectories use pinned local evidence so they can be
audited and regenerated without consuming a web API. RL is what adapts that learned interface to
live Brave results.

After pulling an update on an SSH server, run `./train-ssh setup` once before RL. It synchronizes the
locked Qwen runtime, including the Pillow and Torchvision components required by this multimodal
model; do not install them separately with system `pip`.

Resume from a complete step checkpoint:

```bash
uv run gpt2-reasoning-search rl-search \
  --checkpoint checkpoints/tool-sft \
  --tokenizer-path artifacts/tokenizer.json \
  --prompts data/rl/search-qa.jsonl \
  --index artifacts/wiki-index \
  --output checkpoints/search-rl \
  --resume-from checkpoints/search-rl/step-00000100
```

The reference always remains the original `--checkpoint`; only the trainable policy, optimizer,
scheduler, RNG, epoch, prompt cursor, rollout count, and action-token count resume.

## Safety and evaluation

Live web content makes rewards non-stationary and introduces unreviewed text into optimization; keep
the cached evidence and provider-usage metrics. Run matched held-out evaluations for tool SFT versus
search RL, including search-off cases. Reject a checkpoint that improves reward while degrading
held-out answer accuracy, citation validity, language quality, or unnecessary-search rate; this is
likely reward hacking rather than improved search reasoning.

Treat the question, candidate answer, and retrieved passages as possible prompt-injection inputs to
the judge. The judge prompt tells it to treat them as quoted data, but that boundary is not a
security proof. Audit judge validity, compare with deterministic metrics, and test adversarial
passages before accepting a checkpoint.

Background: [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO for mathematical
reasoning; [DeepSeek-R1](https://arxiv.org/abs/2501.12948) describes cold-start and multi-stage RL;
[WebGPT](https://arxiv.org/abs/2112.09332) demonstrates browser-assisted QA with imitation and human
feedback. This repository combines deterministic local rewards with a low-weight local LLM judge.
