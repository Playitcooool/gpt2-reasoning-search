# Data provenance and preparation

`config/datasets.json` pins every upstream dataset to a full commit hash. The manifest records its
training role and advertised license. Before any public or commercial release, review the upstream
cards and source-level restrictions again; a dataset-level label is not a substitute for legal
review.

The reasoning stream uses the permissively compiled `allenai/big-reasoning-traces` corpus. It
contains OpenR1 and OpenThoughts material. Prefer rows with deterministic correctness metadata,
retain the original source identifier, and report acceptance/rejection counts. The general stream
comes from the pinned FineWeb-Edu 10B sample and remains subject to ODC-By 1.0 attribution and
Common Crawl terms.

Evaluation prompts must be collected before training-corpus filtering. Exact hashes and eight-word
overlap filtering remove known benchmark copies. Store the resulting rejection report beside the
token arrays. The `.npy` manifests contain final token counts and content hashes; these establish the
auditable 70/30 mixture used by the trainer.

Downloaded source data, processed arrays, and model weights are intentionally excluded from Git.
Do not place credentials, private documents, personal data, or unreviewed proprietary corpora in the
pipeline.
