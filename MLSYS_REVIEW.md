# MLSys-Style Critical Review (Self-Assessment)

If this paper were submitted to MLSys, a critical review would likely raise the following concerns:

### 1.1 Novelty and Positioning Risk
- The core ingredients (dynamic batching, queue heuristics, benchmark-based cost models) are individually well known in systems and inference serving practice.
- The paper positions the contribution as a synthesis, which may be judged as engineering integration rather than clear algorithmic novelty unless a stronger theoretical or systems-level novelty claim is demonstrated.
- The relationship to production systems such as Triton/vLLM/TGI is discussed, but the current comparisons are indirect and may not satisfy MLSys expectations for state-of-practice baselines.

### 1.2 Evaluation Breadth
- Results are concentrated on a small number of models and one primary hardware platform. MLSys reviewers typically expect broader coverage across GPUs, CPUs, and model classes.
- Throughput gains are reported, but confidence intervals and statistical tests are limited. Five-run means are useful but may not fully characterize variance under realistic production jitter.
- The workload mix is largely synthetic or controlled. Real trace-driven experiments (bursty arrivals, heavy-tail sequence lengths, multi-tenant interference) are missing.

### 1.3 Baseline Strength
- Baselines are mostly FIFO/fixed-cap variants. A strong MLSys evaluation would compare against tuned industrial schedulers and modern serving stacks with their best-practice settings.
- The paper does not yet include ablations that cleanly isolate how much gain comes from sorting, DP partitioning, estimator quality, and first-batch policy choices respectively.

### 1.4 Theoretical Guarantees and Objective Alignment
- The paper is explicit that global optimality is not claimed for online multi-group ordering, but MLSys reviewers may still ask for tighter approximation guarantees or regret bounds.
- The objective used for decision-making (batch-duration minimization plus policy heuristics) is not always aligned with user-level SLAs across heterogeneous traffic classes.

### 1.5 Reproducibility and External Validity
- Reproducibility notes are helpful, but notebook-driven evaluation can be seen as weaker than scripted end-to-end experiment harnesses with pinned environments and automated artifact generation.
- Claims about broad applicability are scoped in text, yet the current evidence may still be interpreted as narrow to the specific runtime/hardware configuration tested.

### 1.6 What Would Likely Strengthen an MLSys Submission
1. Add direct head-to-head comparisons with tuned Triton-like dynamic batching and at least one continuous-batching serving stack where applicable.
2. Expand hardware/model matrix (multiple GPU generations, at least one datacenter CPU, and additional encoder workloads).
3. Include full ablation study (sorting only, DP only, policy only, estimator variants) plus sensitivity to estimator error.
4. Report richer latency-quality metrics under SLA constraints (tail at fixed throughput, throughput at fixed p99, fairness under mixed tenants).
5. Provide an automated reproducibility package (single-command scripts, fixed seeds where possible, artifact checksums, raw logs).

This critique does not invalidate the observed gains in the tested setup; rather, it describes the likely bar for acceptance at a top-tier systems venue.
