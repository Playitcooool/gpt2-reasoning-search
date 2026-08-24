# Architecture

## Training path

1. The SSH `prepare` stage runs the pinned data bootstrap, tokenizer training, corpus preparation,
   Wikipedia index build, and trajectory generation at fixed repository paths. The tokenizer
   reserves explicit problem, reasoning, answer, tool-call, tool-result, citation,
   padding, and end-of-sequence tokens.
2. Pinned Hugging Face sources are streamed through source allowlists, structural quality checks,
   upstream-verification gates, optional deterministic local checks when references/tests are
   present, contamination filters, and disk-backed exact deduplication.
3. Tokenization writes raw memory-mapped arrays and audit manifests without materializing the full
   corpus in memory.
4. `ExactTokenMixture` schedules non-padding tokens with cumulative 70/30 accounting, reproducible
   wraparound cursors, and exact resume state.
5. The decoder uses pre-norm RMSNorm blocks, RoPE, SwiGLU, grouped-query attention, SDPA, bf16,
   fused AdamW, token-based cosine decay, gradient accumulation, and optional `torch.compile`.
6. Tool SFT streams JSONL through a deterministic shuffle buffer, dynamically pads batches, masks
   user prompts and retrieved observations, and saves full resumable checkpoints.
7. Search RL samples grouped online trajectories against Brave when a key is configured, caching
   results locally and downgrading to BM25 only when Brave is unavailable. A required-lookup answer
   must cite returned evidence matching an accepted source before it earns correctness or judge
   reward. Group-normalized rewards train only generated tokens, while a frozen tool-SFT reference
   supplies KL regularization. A pinned local Qwen3.5-2B model adds a bounded semantic score;
   deterministic outcomes and source provenance remain the primary rewards.

## Retrieval path

Live Brave search is the external knowledge provider. It is protected by a shared asynchronous
client, canonical URLs and stable hashed identifiers, a TTL SQLite cache, bounded retry/backoff,
concurrent page enrichment, robots policy, content-type/size limits, main-text extraction, and
public-address checks on every redirect hop.

`query -> Brave Search -> fetched, sanitized evidence -> model answer with returned-source citations`

The local index stores text metadata in SQLite and writes a compact Tantivy BM25 artifact atomically.
It is used only when Brave is absent, rate-limited, or otherwise unavailable; there is no dense-vector
index, embedding model, reciprocal-rank fusion, or reranker.

## Tool loop

The decoder emits exactly one JSON search call or a final response. The controller rejects malformed,
unknown, multiple, duplicate, or over-budget calls; executes at most three attempts; applies provider
and generation deadlines; and forces a final response when the budget is exhausted. Local search is
the deterministic default. `auto` may fall back to configured web search after an empty or failed
local lookup.

Search text is untrusted data. Control tokens are neutralized before prompting, and citations are
materialized only when their stable identifier appeared in this request's returned results.

## Serving path

FastAPI exposes liveness, readiness, counters, and `/v1/answer`. A semaphore limits concurrent model
generation, queue and request deadlines provide backpressure, request IDs are returned for tracing,
and owned providers/HTTP/cache resources close during application shutdown. Model generation uses a
KV cache and stops at tool-call or EOS boundaries.
