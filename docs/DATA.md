# Data provenance and preparation

`config/datasets.json` pins every upstream dataset to a full commit hash. The manifest records its
training role and advertised license. Before any public or commercial release, review the upstream
cards and source-level restrictions again; a dataset-level label is not a substitute for legal
review.

The reasoning stream uses the pinned `allenai/big-reasoning-traces` configuration and accepts only
the manifest's OpenR1/OpenThoughts source allowlist. Rows must pass structural checks and retain
source and verifier metadata. OpenR1 rows retain their upstream Math Verify/judge provenance. When
a streamed row also exposes a reference answer, code tests, or an exact logic target, the pipeline
reruns the corresponding deterministic local check; otherwise it records the upstream verification
category explicitly. These checks improve data quality but do not prove every accepted trace correct.

The general stream comes from the pinned FineWeb-Edu sample and remains subject to ODC-By 1.0
attribution and Common Crawl obligations. Minimum education score, language, length, and printable
character filters are recorded in rejection counters.

Evaluation prompts must be collected before training-corpus filtering. Exact hashes and eight-word
overlap filtering remove known benchmark copies. Disk-backed exact hash deduplication prevents RAM
growth with corpus size. Store the rejection report beside the token arrays. The raw `.bin` files
use `uint16` when the vocabulary permits and `uint32` otherwise; adjacent manifests contain token
counts, byte size, content and tokenizer hashes, source counts, verifier counts, and rejection
statistics. `preparation-manifest.json` ties both streams to the pinned dataset manifest.

The trainer, not preprocessing, enforces 70/30 by non-padding tokens. Its cumulative integer
schedule preserves the ratio at every practical prefix within rounding tolerance and saves both
source cursors and mixture counters in checkpoints.

Downloaded source data, processed arrays, and model weights are intentionally excluded from Git.
Do not place credentials, private documents, personal data, or unreviewed proprietary corpora in the
pipeline.
