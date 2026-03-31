# Publication Acceptance Gates V1 (Preprint)

This document defines lightweight, notebook-driven publication checks to run **before preprint**.  
Instead of requiring direct head-to-head inference metric comparisons, we validate scheduler behavior using the same style of workflow as `demos/scheduler_tests/strategy_tests_saturated.ipynb`.

## Gate 1: Saturated Strategy Behavior Check

**Method:** Run a saturated synthetic strategy notebook (or a close copy) with the same structure as `demos/scheduler_tests/simulation_strategy_tests_saturated.ipynb`.

**PASS:**
- Notebook executes end-to-end without errors.
- Razor's Edge strategy outputs remain internally consistent across tested load levels.
- The qualitative strategy ordering/behavior matches the notebook narrative (no obvious regressions in decision quality).

**FAIL:**
- Notebook fails to run, or strategy behavior is inconsistent/regresses relative to the notebook expectations.

---

## Gate 2: Unsaturated/Contrastive Sanity Check

**Method:** Run an unsaturated or contrastive notebook check (for example, mirroring `demos/scheduler_tests/simulation_strategy_tests_unsaturated.ipynb`).

**PASS:**
- Notebook executes end-to-end without errors.
- Results are directionally consistent with expected low-pressure scheduling behavior.

**FAIL:**
- Notebook fails to run, or behavior is directionally inconsistent with expected low-pressure behavior.

---

## Gate 3: Reproducibility via Notebook Re-run

For each accepted notebook scenario, rerun once under the same configuration.

**PASS:**
- The second run reaches the same qualitative conclusion as the first run.
- No material strategy flip/regression appears between runs.

**FAIL:**
- Re-run changes the qualitative conclusion or shows unstable strategy behavior.

---

## Publication Rule

Preprint-ready status requires **all three gates to pass** for each in-scope model/hardware/workload class in `docs/publication/SCOPE.md`.

## Sign-off

- Sign-off date (UTC): 2026-03-28
- Sign-off commit hash: TBD
