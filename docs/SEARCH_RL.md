# Search reinforcement learning

Search RL is a separate post-training stage after tool SFT. It optimizes generated reasoning, JSON
tool calls, query reformulation, citations, and final answers while treating prompts and retrieved
observations as fixed environment state.

The implementation is GRPO-style rather than a complete reproduction of PPO/GRPO. For each QA
prompt it samples a group of fresh on-policy trajectories, executes calls against the frozen local
Wikipedia index, scores each complete trajectory, normalizes rewards within the group, and performs
one policy update. It does not reuse a rollout for multiple clipped PPO epochs.

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
  "supporting_ids": ["wiki-page:3"],
  "search_required": true
}
```

`id` is optional. `supporting_ids` should contain stable chunk identifiers from the exact frozen
index. Questions without a required lookup set `search_required` to `false`; keep them in the mix to
teach search restraint. Do not use evaluation questions for RL.

## Reward

The default scalar reward combines:

- normalized answer exact match and token F1;
- citation precision/recall against supporting IDs;
- citation validity against results actually returned in the trajectory;
- valid tool-call rate;
- genuine query recovery after a failed/non-supporting first retrieval;
- penalties for unnecessary searches, invalid/duplicate calls, and each search attempt.
- bounded Qwen3.5-2B scores for answer correctness, evidence support, and search quality.

Fabricated IDs cannot earn citation-support credit, even if they coincidentally match a gold ID,
unless that ID was returned. Query recovery requires distinct valid queries, a failed first lookup,
a later supporting lookup, and a correct answer. Every component, total reward, reward variance, KL,
gradient norm, action-token count, and learning rate is written to `metrics.jsonl`.
Judge latency and valid-output rate are recorded so its H100 cost and formatting reliability remain
visible.

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
  --group-size 2 --max-searches 3 \
  --llm-judge --judge-device cuda
```

The embedding model defaults to CPU during RL so the policy, frozen reference, optimizer, and
activations retain H100 memory. Use `--retrieval-device cuda` only after measuring memory headroom.
The reranker is disabled by default for throughput and can be enabled with `--enable-reranker`.
The revision-pinned Qwen judge is enabled by default in the CLI. Use `--no-llm-judge` for a matched
deterministic-only ablation; changing `--judge-model` also requires an explicit pinned
`--judge-revision`.

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

Use only the frozen local index for RL. Live web content makes rewards non-stationary and introduces
unreviewed text into optimization. Run matched held-out evaluations for tool SFT versus search RL,
including search-off cases. Reject a checkpoint that improves reward while degrading held-out answer
accuracy, citation validity, language quality, or unnecessary-search rate; this is likely reward
hacking rather than improved search reasoning.

Treat the question, candidate answer, and retrieved passages as possible prompt-injection inputs to
the judge. The judge prompt tells it to treat them as quoted data, but that boundary is not a
security proof. Audit judge validity, compare with deterministic metrics, and test adversarial
passages before accepting a checkpoint.

Background: [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO for mathematical
reasoning; [DeepSeek-R1](https://arxiv.org/abs/2501.12948) describes cold-start and multi-stage RL;
[WebGPT](https://arxiv.org/abs/2112.09332) demonstrates browser-assisted QA with imitation and human
feedback. This repository combines deterministic local rewards with a low-weight local LLM judge.
