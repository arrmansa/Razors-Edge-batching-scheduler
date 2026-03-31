# Publication Scope

This file freezes what is and is not covered by the publication claims.

## Frozen Claim Scope Language

All claim language is frozen to: **"ahead-of-time variable-size batching for encoder-style inference."**

## Included Scope

### Models
- Encoder-style inference models only (e.g., embedding and encoder-only/classification transformer workloads).

### Hardware
- CPU and GPU environments that are explicitly benchmarked and reported in the paper artifacts.

### Traces / Workloads
- Variable-size request traces used for ahead-of-time batching evaluation.
- Synthetic and real traces where batch duration can be modeled from batch size and maximum input size.

## Excluded Scope

The following are explicitly out of scope and must not be used to broaden claims:

- Continuous decoding / continuous batching systems.
- KV-cache-based serving systems and iteration-level token schedulers.
- Causal generation pipelines where mid-execution request insertion is a core scheduling primitive.

## Change-Control Rule

Any future scope expansion (models, hardware, traces, or serving regimes) requires:
1. An explicit diff in this file.
2. Updated acceptance evidence in `docs/publication/ACCEPTANCE_GATES_V1.md` (preprint) and/or `docs/publication/ACCEPTANCE_GATES_V2.md` (post-preprint) as applicable.
3. A new sign-off entry with date and commit hash.

## Sign-off

- Sign-off date (UTC): 2026-03-26
- Sign-off commit hash: bc5855c
