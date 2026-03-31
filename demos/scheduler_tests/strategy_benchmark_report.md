# Saturated Scheduler Strategy Benchmark Report

Date run: 2026-03-26 (UTC)

## What was benchmarked

Based on `demos/scheduler_tests/strategy_tests_saturated.ipynb`:
- Queue model: saturated refill loop (`N_USERS=16`, `TOTAL_REQUESTS=10_000`, seed=42)
- Input sizes: uniform random tokens from 1..1000
- Estimator: synthetic monotonic cost
- Compared strategies:
  - `RMS`, `FIFO`, `MINMAX`, `MEANMAX`, `MINMAX_GUARDED_RMSE`, `EFFICIENCY`, `BATCH_SIZE`, `SRPT_AGING`, `HRRN`, `THROUGHPUT_AGING`

## Results (single run)

| Strategy | Throughput (req/s) | Mean latency (s) | P95 latency (s) | Max latency (s) | Mean batch size |
|---|---:|---:|---:|---:|---:|
| SRPT_AGING | 11.925 | 1.340 | 4.126 | 158.174 | 1.327 |
| RMS | 12.712 | 1.257 | 5.280 | 22.518 | 2.372 |
| HRRN | 12.959 | 1.233 | 3.353 | 8.864 | 2.498 |
| MEANMAX | 13.168 | 1.214 | 2.250 | 3.925 | 2.997 |
| THROUGHPUT_AGING | 13.174 | 1.214 | 2.336 | 3.956 | 3.026 |
| FIFO | 13.179 | 1.213 | 2.236 | 3.671 | 3.085 |
| MINMAX_GUARDED_RMSE | 13.143 | 1.216 | 3.469 | 15.157 | 3.157 |
| EFFICIENCY | 13.181 | 1.213 | 3.537 | 10.406 | 3.159 |
| MINMAX | 13.196 | 1.211 | 2.295 | 3.274 | 3.167 |
| BATCH_SIZE | 13.231 | 1.208 | 3.220 | 8.577 | 3.547 |

## Practicality assessment

### Clearly practical
- **FIFO**: nearly top-tier throughput and latency, simple, deterministic, and fairness-oriented (oldest waits are prioritized).
- **MINMAX**: best tail behavior (`max` and low `p95`) with strong throughput; practical when SLO/latency guardrails matter.
- **MEANMAX**: close to FIFO/MINMAX, practical as a smooth compromise heuristic.
- **HRRN**: practical as a starvation-resistant latency-biased policy; less fair than FIFO but far more robust than SRPT_AGING.
- **THROUGHPUT_AGING**: practical if throughput is slightly prioritized while still adding age pressure.

### Conditionally practical
- **BATCH_SIZE**: best throughput and mean latency in this run, but visibly worse tail than FIFO/MINMAX. Good for offline/bulk workloads where tail latency is less critical.
- **EFFICIENCY**: competitive central metrics, but weaker tail behavior than FIFO/MINMAX. Reasonable as an opt-in throughput mode.

### Not practical (for production default)
- **SRPT_AGING**: much lower throughput, very small batches, and catastrophic max latency outlier (158s). Not production-safe as implemented.
- **RMS**: weaker throughput and significantly worse p95/max compared to simpler options under saturation.
- **MINMAX_GUARDED_RMSE**: tail latency outliers remain large; dominates neither throughput nor tails.

## Recommended production strategy set

Keep these selectable strategies:
1. **FIFO** (default fairness policy)
   - Reason: best fairness intuition and stable tails with excellent overall performance.
2. **MINMAX** (latency-SLO / tail-safe policy)
   - Reason: best worst-case behavior with near-best throughput.
3. **BATCH_SIZE** (throughput-first policy)
   - Reason: highest throughput, useful when SLA tolerates larger tails.
4. **HRRN** (anti-starvation balanced policy)
   - Reason: classical response-ratio scheduling tradeoff that improves priority for long-waiting work.

Optional keep:
- **THROUGHPUT_AGING** or **MEANMAX** (choose one to reduce maintenance surface; both are close in behavior).

Recommended deprecations:
- **SRPT_AGING**, **RMS**, **MINMAX_GUARDED_RMSE** (poor saturated behavior relative to alternatives).

## Notes
- This is a **single-seed synthetic saturated** benchmark. Before final pruning, re-run across multiple seeds and unsaturated traces to check robustness.

## Ablation addendum: greedy batch-construction variant

Date run: 2026-03-29 (UTC)

`demos/scheduler_tests/simulation_strategy_tests_saturated_ablation.ipynb` now includes an additional mechanism-isolation variant:

- **F**: sort + greedy batch-construction + MINMAX objective (`demos/scheduler_tests/greedy_batching.py`)

Mean throughput (RPS) in the estimator-array replay ablation:

| Workload | D: sort+DP+MINMAX | F: sort+Greedy+MINMAX | F - D (RPS) |
|---|---:|---:|---:|
| Synthetic | 13.060 | 12.952 | -0.108 |
| `BAAI/bge-m3` replay | 37.173 | 31.368 | -5.805 |
| `jinaai/jina-embeddings-v2-base-en` replay | 4.370 | 4.121 | -0.250 |
