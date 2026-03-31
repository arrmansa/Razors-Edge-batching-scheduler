#!/usr/bin/env python3
"""Verify PAPER.md replay/ablation tables against simulation_results.json.

Usage:
  python demos/scheduler_tests/verify_paper_sim_tables.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "PAPER.md"
SIM_JSON = Path(__file__).resolve().parent / "simulation_results.json"

REPLAY_ROW_RE = re.compile(
    r"\| (Synthetic|`BAAI/bge-m3` \(GPU estimator replay\)|`jinaai/jina-embeddings-v2-base-en` \(CPU estimator replay\)) "
    r"\| ([0-9.]+) \| `?([A-Z_]+)`? \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|"
)

ABLATION_ROW_RE = re.compile(
    r"\| (Synthetic|`BAAI/bge-m3` replay|`jinaai/jina-embeddings-v2-base-en` replay) "
    r"\| ([0-9]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|"
)

WORKLOAD_FROM_PAPER = {
    "Synthetic": "synthetic",
    "`BAAI/bge-m3` (GPU estimator replay)": "gpu_bge_m3",
    "`jinaai/jina-embeddings-v2-base-en` (CPU estimator replay)": "cpu_jina",
    "`BAAI/bge-m3` replay": "gpu_bge_m3",
    "`jinaai/jina-embeddings-v2-base-en` replay": "cpu_jina",
}

ABLATION_VARIANTS = [
    "A_baseline_fifo_unsorted",
    "B_sort_only_fixed_cap",
    "C_sort_plus_dp_fifo",
    "D_sort_plus_dp_minmax",
    "E_sort_plus_dp_batch_size",
    "F_sort_plus_greedy_minmax",
]


def fmt(v: float, ndigits: int) -> str:
    return f"{round(v, ndigits):.{ndigits}f}"


def main() -> int:
    paper_text = PAPER.read_text(encoding="utf-8")
    data = json.loads(SIM_JSON.read_text(encoding="utf-8"))

    replay_rows = REPLAY_ROW_RE.findall(paper_text)
    ablation_rows = ABLATION_ROW_RE.findall(paper_text)

    failures: list[str] = []

    print("=== Replay table checks (PAPER §7.2.1) ===")
    for name, best_base, strategy, best_re, uplift, claim in replay_rows:
        workload = WORKLOAD_FROM_PAPER[name]
        w = data["workloads"][workload]

        expected = {
            "best_baseline_rps": fmt(w["best_baseline"]["throughput_rps"], 3),
            "best_strategy": w["best_razors_edge"]["strategy"],
            "best_re_rps": fmt(w["best_razors_edge"]["throughput_rps"], 3),
            "replay_uplift": fmt(w["replay_uplift_pct"], 2),
            "paper_claim": fmt(w["paper_claim_uplift_pct"], 2),
        }
        actual = {
            "best_baseline_rps": best_base,
            "best_strategy": strategy,
            "best_re_rps": best_re,
            "replay_uplift": uplift,
            "paper_claim": claim,
        }

        row_ok = expected == actual
        status = "PASS" if row_ok else "FAIL"
        print(f"[{status}] {workload}: {actual}")
        if not row_ok:
            failures.append(f"replay::{workload} expected={expected} actual={actual}")

    print("\n=== Ablation table checks (PAPER §7.2.2) ===")
    for name, fixed_n, a, b, c, d, e, f in ablation_rows:
        workload = WORKLOAD_FROM_PAPER[name]
        w = data["workloads"][workload]

        expected_fixed_n = str(w["mechanism_isolation_ablation"]["fixed_n_for_isolation"])
        expected_vals = [
            fmt(
                w["mechanism_isolation_ablation"]["variants"][variant]["throughput_rps"]["mean"],
                3,
            )
            for variant in ABLATION_VARIANTS
        ]
        actual_vals = [a, b, c, d, e, f]

        row_ok = (expected_fixed_n == fixed_n) and (expected_vals == actual_vals)
        status = "PASS" if row_ok else "FAIL"
        print(
            f"[{status}] {workload}: fixed_n={fixed_n}, "
            f"A..F={actual_vals}"
        )
        if not row_ok:
            failures.append(
                "ablation::"
                f"{workload} expected_fixed_n={expected_fixed_n} actual_fixed_n={fixed_n} "
                f"expected={expected_vals} actual={actual_vals}"
            )

    print("\n=== Summary ===")
    if failures:
        print(f"FAIL ({len(failures)} mismatches)")
        for item in failures:
            print(f" - {item}")
        return 1

    print("PASS (all checked cells match simulation_results.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
