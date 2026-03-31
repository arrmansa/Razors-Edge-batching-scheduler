import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from jinja2 import Environment, FileSystemLoader

from greedy_batching import get_batch_start_end_idx_and_duration as get_greedy_batch_start_end_idx_and_duration
from optimal_batching import get_batch_start_end_idx_and_duration

ROOT = Path(__file__).resolve().parents[2]
EST_DIR = Path(__file__).resolve().parent / "estimator_arrays"
OUT_MD = ROOT / "simulation results.md"
OUT_JSON = Path(__file__).resolve().parent / "simulation_results.json"
MD_TEMPLATE = Path(__file__).resolve().parent / "simulation_results_template.md.j2"

SIM_SEEDS = [42, 43, 44, 45, 46]
N_USERS = 16
TOTAL_REQUESTS = 20_000


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    est_file: str
    token_min: int
    token_max: int


WORKLOADS = [
    WorkloadConfig("synthetic", "est_store_synthetic.txt", 1, 1000),
    WorkloadConfig("gpu_bge_m3", "est_store_bge_m3.txt", 1, 1000),
    WorkloadConfig("cpu_jina", "est_store_jina.txt", 1, 500),
]


PAPER_CLAIMS = {
    "synthetic": 17.0,
    "gpu_bge_m3": 26.0,
    "cpu_jina": 47.0,
}


def load_estimator(file_name: str) -> np.ndarray:
    with open(EST_DIR / file_name, "r", encoding="utf-8") as f:
        return np.array(json.load(f), dtype=np.int64)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _metrics(latencies_ns: list[int], batch_sizes: list[int], current_time: int, total_requests: int) -> dict:
    lat = np.array(latencies_ns, dtype=np.float64) / 1e9
    bs = np.array(batch_sizes, dtype=np.float64)
    throughput = total_requests / (current_time / 1e9)
    return {
        "throughput_rps": float(throughput),
        "mean_latency_s": float(lat.mean()),
        "p95_latency_s": float(np.percentile(lat, 95)),
        "p99_latency_s": float(np.percentile(lat, 99)),
        "max_latency_s": float(lat.max()),
        "mean_batch_size": float(bs.mean()),
        "n_batches": int(len(bs)),
        "final_time_ns": int(current_time),
    }


def simulate_razors_edge(
    *,
    estimator: np.ndarray,
    strategy: str,
    seed: int,
    n_users: int,
    total_requests: int,
    min_tokens: int,
    max_tokens: int,
    sort_queue: bool = True,
    use_greedy_batching: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    queue: list[tuple[int, int]] = []

    generated_requests = 0
    current_time = 0
    latencies_ns: list[int] = []
    batch_sizes: list[int] = []

    while len(latencies_ns) < total_requests:
        requests_to_add = min(n_users - len(queue), total_requests - generated_requests)
        for _ in range(requests_to_add):
            token_count = int(rng.integers(min_tokens, max_tokens + 1))
            queue.append((current_time, token_count))
            generated_requests += 1

        if not queue:
            break

        if sort_queue:
            queue.sort(key=lambda x: x[1])

        arrival_times = tuple(x[0] for x in queue)
        token_sizes = tuple(x[1] for x in queue)
        if use_greedy_batching:
            greedy_strategy = "GUARDED_BATCH_SIZE" if strategy == "BATCH_SIZE" else strategy
            start_idx, end_idx, process_duration = get_greedy_batch_start_end_idx_and_duration(
                token_sizes,
                estimator,
                arrival_times,
                current_time,
                greedy_strategy,
            )
        else:
            start_idx, end_idx, process_duration = get_batch_start_end_idx_and_duration(
                token_sizes,
                estimator,
                arrival_times,
                current_time,
                strategy,
            )

        current_time += int(process_duration)
        batch = queue[start_idx:end_idx]
        for arrival, _ in batch:
            latencies_ns.append(current_time - arrival)
        batch_sizes.append(end_idx - start_idx)
        del queue[start_idx:end_idx]

    return _metrics(latencies_ns, batch_sizes, current_time, total_requests)


def simulate_fixed_batch(
    *,
    estimator: np.ndarray,
    batch_size_cap: int,
    seed: int,
    n_users: int,
    total_requests: int,
    min_tokens: int,
    max_tokens: int,
    sort_queue: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    queue: list[tuple[int, int]] = []

    generated_requests = 0
    current_time = 0
    latencies_ns: list[int] = []
    batch_sizes: list[int] = []

    cap = min(batch_size_cap, estimator.shape[0])

    while len(latencies_ns) < total_requests:
        requests_to_add = min(n_users - len(queue), total_requests - generated_requests)
        for _ in range(requests_to_add):
            token_count = int(rng.integers(min_tokens, max_tokens + 1))
            queue.append((current_time, token_count))
            generated_requests += 1

        if not queue:
            break

        if sort_queue:
            queue.sort(key=lambda x: x[1])

        take = min(cap, len(queue))
        batch = queue[:take]
        max_token = max(tok for _, tok in batch)
        process_duration = int(estimator[take - 1, max_token])

        current_time += process_duration
        for arrival, _ in batch:
            latencies_ns.append(current_time - arrival)
        batch_sizes.append(take)
        del queue[:take]

    return _metrics(latencies_ns, batch_sizes, current_time, total_requests)


def _collect_trials_for_strategy(
    estimator: np.ndarray,
    strategy: str,
    min_tokens: int,
    max_tokens: int,
    use_greedy_batching: bool = False,
) -> list[dict]:
    return [
        simulate_razors_edge(
            estimator=estimator,
            strategy=strategy,
            seed=seed,
            n_users=N_USERS,
            total_requests=TOTAL_REQUESTS,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            sort_queue=True,
            use_greedy_batching=use_greedy_batching,
        )
        for seed in SIM_SEEDS
    ]


def _collect_trials_fixed(
    estimator: np.ndarray,
    batch_size_cap: int,
    min_tokens: int,
    max_tokens: int,
    sort_queue: bool,
) -> list[dict]:
    return [
        simulate_fixed_batch(
            estimator=estimator,
            batch_size_cap=batch_size_cap,
            seed=seed,
            n_users=N_USERS,
            total_requests=TOTAL_REQUESTS,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            sort_queue=sort_queue,
        )
        for seed in SIM_SEEDS
    ]


def run() -> None:
    all_results: dict = {
        "config": {
            "sim_seeds": SIM_SEEDS,
            "n_users": N_USERS,
            "total_requests": TOTAL_REQUESTS,
            "estimator_source_notebook": "demos/scheduler_tests/save_estimators_for_simulation.ipynb",
        },
        "workloads": {},
    }

    for w in WORKLOADS:
        estimator = load_estimator(w.est_file)

        # Final scheduler strategies on sorted+DP chain
        strategy_trials = {
            s: _collect_trials_for_strategy(estimator, s, w.token_min, w.token_max)
            for s in ["FIFO", "MINMAX", "BATCH_SIZE"]
        }

        # Baseline sweep for best fixed-cap (arrival-order FIFO)
        baseline_trials = {
            f"n={n}": _collect_trials_fixed(estimator, n, w.token_min, w.token_max, sort_queue=False)
            for n in range(1, estimator.shape[0] + 1)
        }
        baseline_mean_rps = {
            k: mean(t["throughput_rps"] for t in v)
            for k, v in baseline_trials.items()
        }
        best_baseline_key = max(baseline_mean_rps, key=baseline_mean_rps.get)
        best_n = int(best_baseline_key.split("=")[1])

        # Mechanism-isolation ablation at fixed n = best baseline n
        mechanism_variants = {
            "A_baseline_fifo_unsorted": _collect_trials_fixed(estimator, best_n, w.token_min, w.token_max, sort_queue=False),
            "B_sort_only_fixed_cap": _collect_trials_fixed(estimator, best_n, w.token_min, w.token_max, sort_queue=True),
            "C_sort_plus_dp_fifo": strategy_trials["FIFO"],
            "D_sort_plus_dp_minmax": strategy_trials["MINMAX"],
            "E_sort_plus_dp_batch_size": strategy_trials["BATCH_SIZE"],
            "F_sort_plus_greedy_minmax": _collect_trials_for_strategy(
                estimator,
                "MINMAX",
                w.token_min,
                w.token_max,
                use_greedy_batching=True,
            ),
        }

        summarized_strategies = {
            s: {
                "throughput_rps": summarize([x["throughput_rps"] for x in trials]),
                "mean_latency_s": summarize([x["mean_latency_s"] for x in trials]),
                "p95_latency_s": summarize([x["p95_latency_s"] for x in trials]),
                "p99_latency_s": summarize([x["p99_latency_s"] for x in trials]),
                "max_latency_s": summarize([x["max_latency_s"] for x in trials]),
                "mean_batch_size": summarize([x["mean_batch_size"] for x in trials]),
                "raw_trials": trials,
            }
            for s, trials in strategy_trials.items()
        }

        baseline_summary = {
            k: {
                "throughput_rps": summarize([x["throughput_rps"] for x in v]),
                "raw_trials": v,
            }
            for k, v in baseline_trials.items()
        }

        mechanism_summary = {
            k: {
                "throughput_rps": summarize([x["throughput_rps"] for x in v]),
                "p95_latency_s": summarize([x["p95_latency_s"] for x in v]),
                "raw_trials": v,
            }
            for k, v in mechanism_variants.items()
        }

        best_strategy = max(summarized_strategies, key=lambda s: summarized_strategies[s]["throughput_rps"]["mean"])
        best_strategy_rps = summarized_strategies[best_strategy]["throughput_rps"]["mean"]
        best_baseline_rps = baseline_summary[best_baseline_key]["throughput_rps"]["mean"]

        all_results["workloads"][w.name] = {
            "estimator_file": w.est_file,
            "estimator_shape": estimator.shape,
            "token_range": [w.token_min, w.token_max],
            "paper_claim_uplift_pct": PAPER_CLAIMS[w.name],
            "strategies": summarized_strategies,
            "fixed_batch_baseline": baseline_summary,
            "best_baseline": {
                "batch_cap": best_baseline_key,
                "throughput_rps": best_baseline_rps,
            },
            "best_razors_edge": {
                "strategy": best_strategy,
                "throughput_rps": best_strategy_rps,
            },
            "replay_uplift_pct": ((best_strategy_rps - best_baseline_rps) / best_baseline_rps) * 100.0,
            "mechanism_isolation_ablation": {
                "fixed_n_for_isolation": best_n,
                "variants": mechanism_summary,
                "incremental_rps_gain": {
                    "sort_only_minus_baseline": mechanism_summary["B_sort_only_fixed_cap"]["throughput_rps"]["mean"]
                    - mechanism_summary["A_baseline_fifo_unsorted"]["throughput_rps"]["mean"],
                    "dp_minmax_minus_sort_only": mechanism_summary["D_sort_plus_dp_minmax"]["throughput_rps"]["mean"]
                    - mechanism_summary["B_sort_only_fixed_cap"]["throughput_rps"]["mean"],
                    "dp_fifo_minus_dp_minmax": mechanism_summary["C_sort_plus_dp_fifo"]["throughput_rps"]["mean"]
                    - mechanism_summary["D_sort_plus_dp_minmax"]["throughput_rps"]["mean"],
                    "batch_size_minus_dp_minmax": mechanism_summary["E_sort_plus_dp_batch_size"]["throughput_rps"]["mean"]
                    - mechanism_summary["D_sort_plus_dp_minmax"]["throughput_rps"]["mean"],
                    "greedy_minmax_minus_dp_minmax": mechanism_summary["F_sort_plus_greedy_minmax"]["throughput_rps"]["mean"]
                    - mechanism_summary["D_sort_plus_dp_minmax"]["throughput_rps"]["mean"],
                },
            },
        }

    OUT_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    write_markdown(all_results)


def write_markdown(results: dict) -> None:
    cfg = results["config"]
    workload_rows = []
    strategy_sections = []
    ablation_sections = []

    for wname, wres in results["workloads"].items():
        workload_rows.append(
            {
                "name": wname,
                "best_baseline_rps": f"{wres['best_baseline']['throughput_rps']:.3f}",
                "best_strategy": wres["best_razors_edge"]["strategy"],
                "best_strategy_rps": f"{wres['best_razors_edge']['throughput_rps']:.3f}",
                "replay_uplift_pct": f"{wres['replay_uplift_pct']:.2f}",
                "paper_claim_uplift_pct": f"{wres['paper_claim_uplift_pct']:.2f}",
            }
        )

        strategy_rows = []
        for strategy, sres in wres["strategies"].items():
            strategy_rows.append(
                {
                    "name": strategy,
                    "throughput": ", ".join(f"{x['throughput_rps']:.3f}" for x in sres["raw_trials"]),
                    "mean_latency": ", ".join(f"{x['mean_latency_s']:.3f}" for x in sres["raw_trials"]),
                    "p95_latency": ", ".join(f"{x['p95_latency_s']:.3f}" for x in sres["raw_trials"]),
                }
            )
        strategy_sections.append({"workload": wname, "rows": strategy_rows})

        ablation_rows = []
        for variant, vres in wres["mechanism_isolation_ablation"]["variants"].items():
            ablation_rows.append(
                {
                    "name": variant,
                    "throughput": ", ".join(f"{x['throughput_rps']:.3f}" for x in vres["raw_trials"]),
                    "p95_latency": ", ".join(f"{x['p95_latency_s']:.3f}" for x in vres["raw_trials"]),
                }
            )
        gains = wres["mechanism_isolation_ablation"]["incremental_rps_gain"]
        ablation_sections.append(
            {
                "workload": wname,
                "fixed_n": wres["mechanism_isolation_ablation"]["fixed_n_for_isolation"],
                "rows": ablation_rows,
                "gains": {
                    "sort_only_minus_baseline": f"{gains['sort_only_minus_baseline']:.3f}",
                    "dp_minmax_minus_sort_only": f"{gains['dp_minmax_minus_sort_only']:.3f}",
                    "dp_fifo_minus_dp_minmax": f"{gains['dp_fifo_minus_dp_minmax']:.3f}",
                    "batch_size_minus_dp_minmax": f"{gains['batch_size_minus_dp_minmax']:.3f}",
                    "greedy_minmax_minus_dp_minmax": f"{gains['greedy_minmax_minus_dp_minmax']:.3f}",
                },
            }
        )

    env = Environment(
        loader=FileSystemLoader(MD_TEMPLATE.parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(MD_TEMPLATE.name)
    rendered = template.render(
        seeds=cfg["sim_seeds"],
        n_users=cfg["n_users"],
        total_requests=cfg["total_requests"],
        workload_rows=workload_rows,
        strategy_sections=strategy_sections,
        ablation_sections=ablation_sections,
    )
    OUT_MD.write_text(rendered.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
