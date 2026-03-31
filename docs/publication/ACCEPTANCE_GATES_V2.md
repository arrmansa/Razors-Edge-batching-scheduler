# Publication Acceptance Gates V2 (Post-Preprint)

This document defines hard pass/fail publication criteria for Razor's Edge claims versus a Triton dynamic batching baseline. This V2 gate is intended to run **after preprint**.

## Gate 1: Throughput at Fixed p99 (vs Triton baseline)

**Metric:** Relative throughput delta at matched p99 latency target.

- Definition:  
  `delta_throughput_pct = ((throughput_razors_edge - throughput_triton) / throughput_triton) * 100`
- Match condition: Compare at the same p99 latency target (or within an agreed interpolation tolerance documented in the run log).

**PASS:** `delta_throughput_pct > 0`  
**FAIL:** `delta_throughput_pct <= 0`

---

## Gate 2: p99 at Fixed Throughput (vs Triton baseline)

**Metric:** Relative p99 latency delta at matched throughput target.

- Definition:  
  `delta_p99_pct = ((p99_razors_edge - p99_triton) / p99_triton) * 100`
- Match condition: Compare at the same throughput target (or within an agreed interpolation tolerance documented in the run log).

**PASS:** `delta_p99_pct < 0`  
**FAIL:** `delta_p99_pct >= 0`

---

## Gate 3: Reproducibility Rerun Tolerance

For each accepted benchmark point (Razor's Edge and Triton), rerun at least once under the same config.

- Definition (per metric):  
  `rerun_drift_pct = abs(rerun_value - original_value) / original_value * 100`
- Required metrics: throughput and p99.

**PASS:** both throughput and p99 rerun drift are `<= 5%`  
**FAIL:** any required rerun drift is `> 5%`

---

## Publication Rule

Publication-ready status requires **all three gates to pass** for each in-scope model/hardware/trace tuple in `docs/publication/SCOPE.md`.

## Sign-off

- Sign-off date (UTC): 2026-03-26
- Sign-off commit hash: bc5855c
