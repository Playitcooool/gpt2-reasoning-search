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
