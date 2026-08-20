# Security notes

Search results are untrusted. The controller neutralizes model control tokens in retrieved text,
wraps observations in an explicit untrusted-evidence boundary, rejects multiple, malformed,
duplicate, unknown, and over-budget calls, applies deadlines, and accepts citations only for results
observed in the current request. These controls reduce prompt-injection risk but do not eliminate it;
do not give this prototype write-capable tools or sensitive credentials.

Live page enrichment accepts only HTTP(S), rejects credentials and non-public resolved addresses,
revalidates every redirect, respects robots policy, limits concurrency and response bytes, checks
content types, and extracts main text. DNS rebinding and parser/library vulnerabilities remain
possible. Run the service with outbound-network policy, a non-root user, read-only credentials, and
an allowlist proxy when stronger isolation is required.

The Python verifier runs candidates in a temporary directory, in isolated interpreter mode, with a
timeout. This is a correctness filter, not a hardened security sandbox. Run untrusted code inside a
network-disabled container or microVM with strict CPU, memory, process, and filesystem limits.

Keep `BRAVE_SEARCH_API_KEY` in the environment or a secret manager. Never commit it. Bind the API to
localhost unless an authenticated TLS reverse proxy protects it. Live-search responses can change
and must not be used for reproducible benchmark numbers; use the frozen local Wikipedia index for
reported experiments.

Search RL must use the frozen local index and reviewed QA/reward data. Do not place secrets, private
documents, or live-web results in the RL environment. Deterministic rewards can be exploited; inspect
high-reward trajectories and require held-out answer, citation, tool-validity, and search-restraint
gates before deployment.

The optional Qwen reward judge adds another attack surface: retrieved text or candidate answers may
try to instruct the judge, and a small judge can be biased, inconsistent, or reward superficial
phrasing. Its inputs are quoted and capped, outputs are schema-validated, and its reward weight is
bounded, but these controls do not eliminate prompt injection or reward hacking. Keep deterministic
provenance checks authoritative and audit against human-rated and adversarial examples.
