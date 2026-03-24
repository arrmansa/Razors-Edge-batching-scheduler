# Razor's Edge: Throughput Optimized Dynamic Batching with Latency Objectives

**Author:** Arrman Anicket Saha  
**Affiliation:** Independent researcher (unaffiliated)  
**Contact (GitHub):** <https://github.com/arrmansa>  
**Email:** arrmansa99@gmail.com  
**ORCID:** 0009-0004-6884-0644  
**Date:** March 24, 2026

## Abstract
Serving systems for embedding, LLM, and other matrix-multiplication-dominated inference workloads rely on batching for efficient hardware utilization. We observe that batching efficiency exhibits a sharp input-size-dependent structure driven by the transition between memory-bound and compute-bound regimes: small inputs can be batched flexibly across heterogeneous sizes, while large inputs require near-uniformity, leading to a rapid collapse in batching efficiency. This produces a characteristic blade-like ("razor's edge") shape in the batch performance landscape.

We present the Razor's Edge batching scheduler, a practical framework that combines (i) dynamic-programming-based throughput optimization over sorted requests, (ii) multiple latency objectives for next-batch selection, and (iii) startup-time-efficient model benchmarking that builds batch timing estimators for real hardware. The approach is designed for real-time online serving with queueing. Our claims are scoped to the variable-size batched inference regimes evaluated in this paper, not to universal superiority across all serving stacks.

## 1. Introduction
Requests cannot be processed instantaneously at arrival in any real serving system. Under load, queueing is inevitable. A first-come, first-served single-request policy is simple, but it fails to exploit throughput gains available through batching. Simply batching the first `n` requests in a queue is common, but it is not throughput-optimal. Sorting requests by size and then batching can create high latency for large requests that may not be selected quickly under sustained load.

This work addresses a common inference regime where workloads are dominated by batched matrix multiplication with variable input sizes (for example, variable token lengths in embedding or classification models). The scheduler pursues two goals simultaneously:

1. **Maximum throughput**: Sort and choose batch partitions that minimize total completion time for queued work.
2. **Latency objective**: choose near-term execution order with three possible objectives:
- (RMS) minimize the sum of squared latencies of queued work
- (FIFO) choose the batch with the oldest input
- (MINMAX) choose the batch with the highest prospective latency (waiting time + processing time)

In addition to runtime scheduling, we include startup benchmarking and estimator construction techniques that reduce calibration overhead while preserving scheduling quality.

## 2. Related Work

This paper builds on two strands of prior work: (1) classical single-machine scheduling theory for completion-time style objectives, and (2) practical production batching systems used in modern ML inference.

### 2.1 Classical Scheduling Background
- Smith's foundational sequencing rule for weighted completion-time minimization gives the interchange-argument template used throughout modern scheduling analysis [1].
- For quadratic completion/lateness penalties, prior work studies single-machine objectives where squared terms increase the penalty for tail latency and variability [2], [3].

Our RMS-latency ordering pass is in this family: it uses pairwise interchange logic under a squared-latency objective, adapted to grouped/batched online inference decisions.

### 2.2 Production ML Batching Systems
- **NVIDIA Triton dynamic batching** provides queue-delay windows and preferred batch sizes for online serving [4].
- **Infinity / batched integrations** expose practical dynamic batching controls for embedding and inference workloads [5], [6].
- **Hugging Face pipeline batching** documents practical throughput-oriented batch inference usage [7].

Unlike fixed-threshold accumulation approaches, Razor's Edge combines throughput-optimal partitioning (DP on sorted requests) with a second RMS-latency ordering pass.

### 2.3 Positioning Relative to Existing Batching Practice
Most production dynamic batchers expose preferred batch sizes, and simple first-ready policies [4]–[7]. These policies are robust and easy to deploy, but they typically do not solve an explicit global partitioning objective on the current queue state.

The key distinction of Razor's Edge is its two-stage decision process:
1. **Throughput-optimal contiguous partitioning** on size-sorted requests (DP over candidate cuts).
2. **RMS-latency-aware first-batch selection** among DP-consistent candidates.

This makes Razor's Edge closer in spirit to objective-driven scheduling than to size-only batching heuristics, while remaining practical for online serving.

### 2.4 Scope of Novelty Claim
The contribution of this work is not a new worst-case approximation ratio in scheduling theory. Instead, it is a systems-oriented synthesis with bounded claims:

1. **Algorithmic synthesis claim (implementation-level):** for variable-size batched inference queues where Section 3.2 timing assumptions are a reasonable approximation, we provide a practical sorted-queue batching model with efficient DP partitioning plus an RMS-based first-batch ordering pass.
2. **Engineering claim (deployment-level):** for deployments that can run startup calibration, we provide startup-efficient benchmarking/estimator construction and numerically safe scheduling implementation suitable for the tested runtime stack.
3. **Empirical claim (evaluation-level):** on the specific synthetic and `BAAI/bge-m3` workloads in Section 7 (with calibrated estimator tables and stated hardware/runtime), we observe throughput/latency improvements versus the baselines defined there.

What is **not** claimed:
- no proof of global optimality for the multi-group online or request ordering ordering pass;
- no universal dominance over all dynamic batching policies, model families, or hardware environments.

What is **validated in this paper**:
- DP-based sorted contiguous partitioning and the RMS/FIFO/MINMAX-guided first-batch heuristic can be integrated in an online executor with bounded per-decision overhead on the tested workload class;
- estimator calibration plus the scheduler improves measured throughput relative to the tested FIFO/fixed-cap baselines in Section 7 under the reported setup.

The empirical sections evaluate whether this synthesis yields operational gains in realistic serving settings under these scope conditions.


## 3. Problem Setting and Model Assumptions
Let 
```math
F([x_1, ..., x_n]) = [y_1, ..., y_n]
```

be a batch inference function executed on system S, returning one output per input.

### 3.1 Consistent Practical Utility
For an input \(x_i\), denote singleton output as \(a\), and output in different batching contexts as \(y_i\) or \(d_i\). Exact equality is not required; practical equivalence (e.g., for retrieval, ranking, or classification quality) is sufficient.

### 3.2 Consistent Timing

1. The execution time of a batch is independent of the internal ordering of requests within that batch.
2. Batch execution time is monotonically non-decreasing with both batch size and the largest input size in the batch.
3. If two batches have the same size and the same largest input size, they have the same execution time.
4. The largest element in a batch is the element with the highest singleton compute cost.

These 4 axioms imply that batching on sorted `x_i` is at least as fast as batching on unsorted partitions.

#### Proof (exchange argument sketch)

Consider two internally ascending batches drawn from an optimal partition on unsorted inputs. We can always find two batches with at least two elements each:
```math
B1 = [x_1, x_2, ... x_m]
```

```math
B2 = [y_1, y_2, ... y_n]
```
with cross-overlap conditions `x_m > y_1` and `y_n > x_1`.

There are two cases:

1. If `x_m <= y_n`, construct:
```math
B1' = [x_1, x_2, ... x_m-1, y_1]
```

```math
B2' = [x_m, y_2, ... y_n]
```
Then `time(B1') < time(B1)` while `time(B2') = time(B2)`. Repeating this local exchange step moves toward a partition that is consistent with sorted-input batching.

2. If `x_m > y_n`, construct:
```math
B1" = [y_n, x_2, ... x_m]
```

```math
B2" = [y_1, y_2, ... y_n-1, x_1]
```
Then `time(B2") < time(B2)` while `time(B1") = time(B1)`. Repeating this exchange yields a partition equivalent to one obtainable from sorted inputs.

### 3.3 Applicability

The method is intended for **variable-size batched inference** workloads where a calibration phase can build a timing estimator over `(batch_size, max_input_size)` and where measured batch durations are stable enough for Section 3.2 assumptions to be useful. In this paper, that condition is tested on (i) a synthetic controlled workload and (ii) a `BAAI/bge-m3` inference pipeline under the reported hardware/runtime setup. Applicability to other PyTorch/TensorFlow/ONNX/OpenVINO deployments depends on whether their measured timing surfaces exhibit similar stability after calibration.

### 3.4 Notation and Timing Terms (used consistently below)

To avoid ambiguity, we use the following timing notation throughout:
- \(a_i\): request arrival timestamp.
- \(s_i\): service start timestamp for request \(i\).
- \(c_i\): completion timestamp for request \(i\).
- \(q_i = s_i - a_i\): queueing delay (waiting before service starts).
- \(\ell_i = c_i - a_i\): end-to-end latency.
- \(D_k\): processing duration of batch/group \(k\).

The RMS objective in this paper refers to minimizing \(\sum_i \ell_i^2\) for the currently queued work considered by the scheduler.

## 4. Core Scheduling Method
Given a queue of requests, the scheduler runs four main steps:
1. We first order the requests based on their size.
2. We then partition them into batches to minimize total time using dynamic programming.
3. We do a single ordering pass usinig a latency objective.
4. We pick the first batch and update the expected start time of the next batch.

### 4.1 Batch creation by Dynamic Programming
The scheduler’s primary objective is to partition a queue of $N$ sorted requests into a set of batches $\mathcal{B} = \{b_1, b_2, \dots, b_k\}$ such that the total processing time is minimized. Given that the hardware exhibits a non-linear relationship between batch size and input length, we formalize this as a shortest-path problem on a Directed Acyclic Graph (DAG).

#### Formal Definition
Let $Q = \{r_1, r_2, \dots, r_n\}$ be the set of requests in the queue, sorted by their input size $s_i$ such that $s_1 \le s_2 \le \dots \le s_n$. 

We define a cost function $C(b)$ that represents the expected execution time of a batch $b$. Based on our consistency axioms (Section 3.2), the execution time depends only on the number of elements in the batch $|b|$ and the size of the largest element $\max(s \in b)$. Since the queue is sorted, for any contiguous sub-sequence from index $i$ to $j$, the cost is:
$$C(i, j) = \text{Estimator}[j - i + 1, s_j]$$

#### Recurrence Relation
Let $T[j]$ be the minimum time required to process the first $j$ requests. The optimal time for the prefix of length $j$ is given by the recurrence:
$$T[j] = \min_{1 \le k \le M} \{ T[j - k] + C(j - k + 1, j) \}$$
where $M$ is the maximum allowable batch size.

We initialize $T[0] = 0$ and $T[j] = \infty$ for all $j > 0$. The complexity of this computation is $O(N \cdot M)$. Since $M$ is typically small ( < $256$ ), this optimization is computationally efficient for real-time online serving.

#### Practical Implementation
To ensure high-performance execution within the scheduling loop, the dynamic program is implemented using JIT-compiled kernels. We pre-compute a 2D lookup table for $C(i, j)$, where the dimensions represent (batch size, input size). This allows for $O(1)$ cost lookups during the DP pass, minimizing the overhead added to the request lifecycle.

Implementation entry point: `get_batch_start_end_idx_and_duration` in `src/razors_edge/optimal_batching.py`.

Core internals:
- `_compiled_dynamic_batcher`
- `_get_slice_indexes_and_duration`

### 4.2 Overflow-Safe RMS Optimizing Pass

This is the recommended strategy for most cases where delays have quadratic cost. We employ a weighted squared latency objective with a pairwise interchange argument based on classical single-machine sequencing proofs [1]. The use of squared penalties follows quadratic completion/lateness formulations that penalize tail outcomes than linear objectives [2], [3]. In our setting, the resulting priority score jointly depends on batch processing duration and accumulated waiting time, which directly targets RMS latency reduction.

Given request arrivals (time-ordered) `R1, R2, ...` at times `x0, x1, ...` such that `x_i < x_(i+1)`, with processing start times `y0, y1, ...` and output times `z0, z1, ...`, a system with minimum mean squared latency minimizes:

`sum((z_i - x_i)^2)` over all `i`.

For a traditional one-by-one processor, `y_i = z_(i-1)`.

#### Two-request case

Let `R1` and `R2` arrive while current request `R0` is processing.

Let:

- `R1` processing time be `D1`
- `R2` processing time be `D2`
- `R1` queueing delay be `W1`
- `R2` queueing delay be `W2`


```math
RMS_LATENCY_1 = (W1 + D1) ** 2 + (W2 + D1 + D2) ** 2
```

```math
RMS_LATENCY_2 = (W2 + D2) ** 2 + (W1 + D1 + D2) ** 2
```

If `RMS_LATENCY_1 < RMS_LATENCY_2`, pick `R1` before `R2`.

This simplifies to:
```math
D1 * (W2 + D2 + D1/2) < D2 * (W1 + D1 + D2/2)
```

#### Grouped-requests case

Consider groups `G1, G2, G3, ...`, processed one-by-one.

- Group durations: `D1, D2, D3, ...`
- Group end times: `T1 = D1`, `T2 = T1 + D2`, ... (positive, relative to when `G1` started)

Within group `i`, the arrival time of request `j` (negative relative to when `G1` started) is `Gi_Wj`.

To minimize RMS, order groups to minimize:

```math
(G1_W1 - T1)^2 + (G1_W2 - T1)^2 + ... + \
(G2_W1 - T2)^2 + (G2_W2 - T2)^2 + ... +
...
```

Which expands to 

```math
sum(sum(Gi_Wj**2)) - 2 * sum(Ti * sum(Gi_Wj)) + sum(len(Gi) * Ti**2)
```

Ignoring the `sum(sum(Gi_Wj**2))` since it's constant, this simplifies to

```math
sum(Ti * len(Gi) * (Ti - 2 * mean(Gi_Wj)))
```


This shows that exact per-request arrival terms are constant regardless of group ordering. The ordering-relevant term is the average arrival time within each group.

##### For 2 groups:

- Group `R1` has duration `D1`, size `N1`, average queueing delay `W1`
- Group `R2` has duration `D2`, size `N2`, average queueing delay `W2`

Define:

```math
RMS_LATENCY_1 = N1 * (W1 + D1) ** 2 + N2 * (W2 + D1 + D2) ** 2
```

```math
RMS_LATENCY_2 = N2 * (W2 + D2) ** 2 + N1 * (W1 + D1 + D2) ** 2
```

If `RMS_LATENCY_1 < RMS_LATENCY_2`, pick `R1` before `R2`.

This simplifies to:

```math
N_2 D_1 \left(W_2 + D_2 + \frac{D_1}{2}\right)
\;<\;
N_1 D_2 \left(W_1 + D_1 + \frac{D_2}{2}\right)
```
Pick `R1` before `R2` when the inequality above holds.

In practice we use 64-bit integer nanosecond timing and limit batch size to 256 (`uint8`). For numerical safety, `log2((2**63 / (256 * 5)) ** 0.5) = 26.8247...`, so we conservatively use 26-bit scaled operands in the overflow-safe comparison path.

##### For more than 2 groups

For more than two groups, this pairwise ordering property is not transitive, so a single greedy pass is not globally optimal in all cases. However, it is close to optimal under brute-force checks (`N`, `D`, and `T` from 1 to 10).

For three requests (`i`, `j`, `k`), we observe the optimal ordering starts with `i` in 99.995% of cases (47,066 / 997,002,000 suboptimal orderings) when:
1. i should be processed first given only i and j
2. j should be processed first given only j and k

For three requests (`i`, `j`, `k`), we observe the optimal ordering starts with `i` in 99.995% of cases (50,853 / 997,002,000 suboptimal orderings) when:
1. i should be processed first given only i and j
2. i should be processed first given only i and k

These edge cases are sufficiently rare for the intended real-time setting. Additional pre-sorting by required processing time before RMS comparison may recover small gains, but likely does not justify the extra scheduling overhead for online serving.

#### Intuitive example

Prioritize `R2` for lower RMS latency when:

- `R1`: 1 request, 10 seconds processing, arrived 2 seconds ago
- `R2`: 2 requests, 1 second processing, arrived 1 second ago

`2 * 10 * (2 + 2 + 10) < 1 * 1 * (4 + 20 + 1)`

`280 < 25` (false), so `R2` should be picked before `R1`.

If `W1` is treated as unknown, the crossover shows that if `R1` was already delayed by ~129.5s, processing `R1` first becomes optimal for RMS latency.


Implementation entry point: `get_batch_start_end_idx_and_duration` in `src/razors_edge/optimal_batching.py`.

Core internals:
- `_compiled_bit_length`
- `_prospective_rms_latency_improvement`
- `_get_slice_indexes_and_duration`

Mathematical mapping of these functions:

- `_compiled_bit_length(x)` computes
```math
\mathrm{bitlen}(x)=\lfloor \log_2(x)\rfloor + 1,\quad x>0
```
which is used to determine safe right-shift scales for 64-bit arithmetic.

- `_prospective_rms_latency_improvement(...)` compares two orderings of two candidate groups (`chosen` then `prospective` vs the reverse) with the same symbols and ordering semantics used in code:

  Let `c = chosen`, `p = prospective`, with `N` (group size), `D` (group duration), `W` (mean queueing time). The two-ordering squared-latency totals are:
```math
L_{c\to p}=N_c(W_c+D_c)^2+N_p(W_p+D_c+D_p)^2,\quad
L_{p\to c}=N_p(W_p+D_p)^2+N_c(W_c+D_p+D_c)^2
```
  The decision in code is "prospective improves RMS latency" iff \(L_{c\to p}>L_{p\to c}\). Expanding and cancelling common terms gives:
```math
N_pD_c\!\left(2W_p+2D_p+D_c\right)\;>\;N_cD_p\!\left(2W_c+2D_c+D_p\right)
```
  Divide both sides by \(2\) (positive, so direction is unchanged):
```math
N_pD_c\!\left(W_p+D_p+\frac{D_c}{2}\right)\;>\;N_cD_p\!\left(W_c+D_c+\frac{D_p}{2}\right)
```
  Reordering to match the exact comparator form in `optimal_batching.py`:
```math
N_cD_p\!\left(W_c+D_c+\frac{D_p}{2}\right)\;<\;N_pD_c\!\left(W_p+D_p+\frac{D_c}{2}\right)
```
  which is the same chosen-vs-prospective inequality shape implemented by `_prospective_rms_latency_improvement(...)`.

  For numeric safety, the implementation right-shifts (`>>`) the time terms before multiplying, and uses integer half terms (`// 2`). These are monotone, positive scaling/rounding steps in the compared operands, so they preserve comparator direction for the scheduling decision while reducing overflow risk in 64-bit arithmetic.

- `_get_slice_indexes_and_duration(...)` reconstructs candidate batches along the optimal-throughput DP backpointer chain and selects
```math
b^\*=\arg\min_{b \in \mathcal{C}} \sum_{i\in b}(w_i + D_b)^2
```
in one pass via pairwise comparisons (`\mathcal{C}` is the set of DP-derived contiguous candidates ending on the current suffix).


### 4.3 FIFO Optimizing Pass
This is an alternative to RMS and should be used when fairness is required and older requests should be processed first.

This strategy involves choosing the batch containing the oldest element.

There is no possible overflow.

Based on experiments this has slightly higher throughput than RMS.

### 4.4 MINMAX Optimizing Pass

This is an alternative to RMS and should be used when high maximum latency is extremely undesirable, even at the cost of possible higher average latency.

This strategy involves choosing the batch containing an element which would have the highest latency (time waited in queue + processing time) to be processed first.

Overflow is avoided during addition by rightshifting one bit.

## 5. Benchmarking and Estimator Construction

A dynamic program requires a 2D array containing the expected duration to process a batch given its size and the size of the largest element, indexed as `arr[batch_size, max_input_size]`. Creating this array using exhaustive benchmarking would be prohibitively slow. Instead, for each batch size we measure selected points and extrapolate with a spline. Additionally, we may observe slow-changing or redundant data for very high `batch_size` and `max_input_size`, where timings are approximately scaled versions of the previous batch size:
`arr[batch_size-1, max_input_size] * batch_size / (batch_size-1)`.
This typically indicates hardware saturation. To reduce startup cost, we detect this saturation point during benchmarking.

Function: `RazorsEdgeComputeTask.get_batch_timing_data` in `src/razors_edge/razors_edge_compute_task.py`.

The routine can be summarized as:
1. Generate initial input-size points
```math
\mathcal{X}_0=\texttt{generate\_benchmark\_points}(s_{\min}, s_{\max}, \rho)
```
2. Measure timings for first two batch sizes \(b_1, b_2\):
```math
\mathcal{T}_{b_k}(x)\approx \text{bench}(b_k,x),\quad k\in\{1,2\}
```
3. For each next batch size \(b_m\), adapt the next point set by a saturation heuristic:
```math
\mathcal{X}_{m}=\texttt{calculate\_next\_benchmark\_points}(\mathcal{X}_{m-2},\mathcal{T}_{m-2},b_{m-2},\mathcal{X}_{m-1},\mathcal{T}_{m-1},b_{m-1},\rho)
```
4. Pad late-stage points using scaled previous spline values:
```math
\tilde{\mathcal{T}}_{m}(x)=\alpha_m \cdot \hat{\mathcal{T}}_{m-1}(x),\quad \alpha_m=\frac{b_m}{b_{m-1}}
```
for padding-only \(x\) values beyond the newly measured range.

### 5.1 CPU Benchmarking Strategy
CPU timing is possibly sensitive to garbage collection and inter-run state. The CPU protocol runs controlled sequences with GC and pause variations, then uses the **minimum observed time** as the hardware-limited best-case estimate.

Function: `model_test_pattern_cpu` in `src/razors_edge/optimal_benchmarking.py`.

Mathematically, if the run set is \(\{t_1,\dots,t_K\}\) under different GC/sleep states, the estimator is:
```math
\hat{t}_{\mathrm{cpu}}=\min_{1\le k\le K} t_k
```
which targets the lower-envelope compute cost under transient software noise.

### 5.2 GPU Benchmarking Strategy
GPU timings are sensitive to transient boost and warmup. The GPU protocol performs bounded repeated warm runs (time- and iteration-constrained) and uses the **median** measured value for stability.

Function: `model_test_pattern_gpu` in `src/razors_edge/optimal_benchmarking.py`.

If post-warmup measurements are \(\{g_1,\dots,g_K\}\), the estimator is:
```math
\hat{t}_{\mathrm{gpu}}=\mathrm{median}(g_1,\dots,g_K)
```
which is robust to occasional jitter spikes.

### 5.3 Adaptive Benchmark Point Reduction
To reduce startup cost, benchmark points are generated nonlinearly over token/input size and adapted using inter-batch timing ratio behavior (empirically sigmoid-like). If saturation is detected, the next benchmark range is reduced accordingly.

Functions:
- `get_points_ratio`
- `generate_benchmark_points`
- `calculate_next_benchmark_points`
- `get_benchmark_data_paddings`

(all in `src/razors_edge/optimal_benchmarking.py`).

Mathematical definitions:

- `get_points_ratio(min_tokens, max_tokens, max_points)`:
```math
\rho = \frac{P_{\max}-1}{\sqrt{s_{\max}-s_{\min}}}
```

- `generate_benchmark_points(...)` uses uniform spacing in \(\sqrt{s}\)-space:
```math
s_i = \left(\sqrt{s_{\min}} + \frac{i}{n-1}(\sqrt{s_{\max}}-\sqrt{s_{\min}})\right)^2,\quad i=0,\dots,n-1
```
where \(n=\max(\lfloor \sqrt{s_{\max}-s_{\min}}\rho +1\rfloor,\text{MINIMUM\_SPLINE\_POINTS})\).

- `calculate_next_benchmark_points(...)` forms smoothed curves \(\hat{t}_1(s),\hat{t}_2(s)\) and ratio
```math
r(s)=\frac{\hat{t}_2(s)/b_2}{\hat{t}_1(s)/b_1}
```
then enforces a monotone saturation envelope by reverse cumulative minimum and picks a new endpoint from sigmoid midpoint and saturation-threshold rules.

- `get_benchmark_data_paddings(...)` extends a shorter new curve by
```math
t_{\mathrm{pad}}(s)=\alpha\cdot \hat{t}_{\mathrm{prev}}(s)
```
at padding points \(s\), where \(\alpha\) is the adjacent batch-size scaling ratio.

- `create_batch_timing_estimators(...)` builds
```math
\mathrm{Estimator}[b,s]\approx \hat{t}_b(s)
```
using spline interpolation for measured batch sizes and linear interpolation across neighboring benchmarked batch sizes:
```math
\hat{t}_{b}(s)=(1-\lambda)\hat{t}_{b_{\mathrm{lo}}}(s)+\lambda\hat{t}_{b_{\mathrm{hi}}}(s),\quad
\lambda=\frac{b-b_{\mathrm{lo}}}{b_{\mathrm{hi}}-b_{\mathrm{lo}}}.
```

## 6. Systems Architecture

### 6.1 Process and Threading Design
Execution is isolated in a spawned process with:
- bounded async/sync admission,
- dedicated send/receive channels,
- result-setting loop,
- periodic inter-process time synchronization,
- fair queue selection by earliest operation id.

Primary class: `ComputeExecutor` in `src/executor/process_manager.py`.

Key methods:
- `async_compute_fn`, `sync_compute_fn`
- `set_result_loop`, `set_time_loop`
- internal `_InternalProcess.accumulate_data`, `choose_fair_queue`, `post_process_and_send_result`

Mathematical interpretation:

- `choose_fair_queue` performs earliest-key selection:
```math
q^\*=\arg\min_q (o_q,\,\tau_q)
```
where \((o_q,\tau_q)\) is the first pending `(operation_id, queue_time)` in queue \(q\).

- `set_time_loop` / `_InternalProcess.get_time_loop` maintain clock alignment by repeatedly updating
```math
\Delta = t_{\mathrm{internal}} - t_{\mathrm{external}}
```
and using \(\tau_{\mathrm{aligned}}=\tau_{\mathrm{recv}}+\Delta\) for queue timestamps.

- `post_process_and_send_result` is an order-preserving map over batch ids:
```math
\{(id_i, y_i)\}_{i=1}^m,\quad y_i=\mathrm{postprocess}(\mathrm{model}(x_{1:m}))_i
```
with one emitted result per original request id.

### 6.2 Compute Task Contract
The base plugin interface defines preprocessing, batching acceptance, model call, and postprocessing hooks.

Base class: `BaseBatchedComputeTask` in `src/executor/base_batched_compute_task.py`.

For the default base implementation, batching is a maximal prefix under acceptance predicate \(A\):
```math
b^\* = (x_1,\dots,x_k),\quad
k=\max\{j: A((x_1,\dots,x_{j-1}),x_j)=\text{True}\}
```
with default \(A=\text{False}\), giving \(k=1\) (single-item batches).

### 6.3 Razor's Edge Compute Task Contract
The plugin interface builds on the base plugin and defines model loading, preprocessing, and postprocessing hooks while defining the creation of benchmarking and batch creation.

Base class: `RazorsEdgeComputeTask` in `src/razors_edge/razors_edge_compute_task.py`.

At runtime, batch selection is a composition:
```math
\text{inputs} \xrightarrow{\text{preprocess}} (u_i,s_i)
\xrightarrow{\text{sort by }s_i} \xrightarrow{\text{DP+Latency}} [i:j)
\xrightarrow{\text{create\_batch}} B
\xrightarrow{\text{model}} \hat{y}
\xrightarrow{\text{postprocess}} y
```
and expected start-time tracking updates as
```math
t_{\mathrm{expected}} \leftarrow \max(t_{\mathrm{expected}}, t_{\mathrm{now}}) + D_{[i:j)}.
```

## 7. Results

We evaluate Razor's Edge on both synthetic and real inference workloads to study batching behavior under controlled and realistic conditions.

Unless otherwise noted, each reported throughput statistic is based on 5 independent notebook runs per setting. We report means, sample standard deviations. Plotted curves/contours use representative (lower-noise) runs for visual clarity.

For the synthetic task, Razor's Edge achieved a throughput of 13.3 RPS (17% higher than the baseline), with a maximum allowed batch size of 8.
For the synthetic task, the baseline strategy achieved a peak throughput of 11.4 RPS, with best performance when the batch size was limited to 1. In this setting, simple batching did not improve throughput.

**Figure 1:** Synthetic-load benchmark comparison.  
**Notebook source:** `demos/synthetic/dummy performance comparison basic.ipynb`.

![Synthetic Load Basic Benchmarks](images/Synthetic%20Load%20Basic%20Benchmarks.png)

For the real model workload, Razor's Edge achieved a throughput of 35.3 RPS (25% higher than the baseline), with a maximum allowed batch size of 16.
For the real model workload, the baseline strategy achieved a throughput of 28.2 RPS, with best performance when the batch size was limited to 1. In this setting, simple batching did not improve throughput.

**Figure 2:** Real-load benchmark comparison.  
**Notebook source:** `demos/real/gpu benchmark performance comparison basic.ipynb`.

![Real Load Basic Benchmarks](images/Real%20Load%20Basic%20Benchmarks.png)

For the real model workload with limited token input, Razor's Edge achieved a throughput of 84.6 RPS (26% higher than the baseline with batch size = 10), with a maximum allowed batch size of 16.
For this limited-token workload, the baseline strategy achieved a throughput of 67.4 RPS, with best performance at batch size = 10. Baseline batching improved throughput relative to the non-batched baseline of 41.4 RPS (39% lower than the baseline with batch size = 10).

**Figure 3:** Real-load benchmark comparison with limited tokens.  
**Notebook source:** `demos/real/gpu benchmark performance comparison basic limited.ipynb`.

![Real Limited Load Basic Benchmarks](images/Real%20Limited%20Load%20Basic%20Benchmarks.png)

**Figure 4:** Real-load benchmark comparison with limited tokens.  


For the synthetic workload, we compare the three outlined strategies.
**Notebook source:** `demos\synthetic\dummy latency comparison.ipynb`.

![Synthetic Latency Comparison with Different Strategies](images/Synthetic%20Latency%20Comparison%20with%20Different%20Strategies.png)


For the real workload with limited tokens, we compare the three outlined strategies.
**Notebook source:** `demos\real\gpu benchmark limited latency comparison.ipynb`.
![Real Latency Comparison with Different Strategies](images/Real%20Latency%20Comparison%20with%20Different%20Strategies.png)

### 7.1 Baseline Policy Definition

The baseline in Section 7 is the `BaseBatchedComputeTask` policy run under `ComputeExecutor`, with behavior matching the implementation:

- **FIFO oldest-first selection (no reordering/sorting):** each per-task queue is a Python `dict` keyed by `(operation_id, queue_time)` and consumed from iteration order; `choose_fair_queue` picks the queue whose first key is globally smallest, and `get_batch_ids_list_and_batch` consumes from the first enqueued item onward. This preserves arrival order and does not apply size-based sorting.  
  Implementation points: `ComputeExecutor._InternalProcess.accumulate_data`, `ComputeExecutor._InternalProcess.choose_fair_queue`, and `BaseBatchedComputeTask.get_batch_ids_list_and_batch`.
- **Immediate dispatch on idle:** when all queues are empty, `accumulate_data` blocks until at least one request arrives; the loop then immediately selects a fair queue, forms a batch, removes ids, and submits execution without any delay.  
  Implementation points: `ComputeExecutor._InternalProcess.__init__` main loop and `ComputeExecutor._InternalProcess.accumulate_data`.
- **Append until max batch size `n`:** baseline task variants implement `_accept_in_batch` as `len(current_batch) < self.max_batch_size`, so batching is a maximal prefix up to fixed cap `n`.  
  Implementation points: `demos/synthetic/base_batched_dummy_task.py` and `demos/real/base_batched_gpu_benchmark_task.py` (both define `max_batch_size` and this predicate), plus variant classes that set `n`.

**Exact baseline hyperparameter grids (`n`) used in Section 7 comparisons**

- **Synthetic basic benchmark grid:** `n ∈ {1, 2, 3, 4}` (via `BaseBatchedDummyTask`, `BaseBatchedDummyTaskB2`, `BaseBatchedDummyTaskB3`, `BaseBatchedDummyTaskB4`).
- **Real basic benchmark grid:** `n ∈ {1, 2, 3, 4}` (via `BaseBatchedGPUBenchmarkTask`, `BaseBatchedGPUBenchmarkTaskB2`, `BaseBatchedGPUBenchmarkTaskB3`, `BaseBatchedGPUBenchmarkTaskB4`).
- **Real limited-token benchmark grid:** `n ∈ {1, 2, 8, 10, 13, 16}`.

**Method used to choose reported baseline setting**

For each workload, we benchmark each candidate `n` over 5 runs and report the mean throughput (RPS) of 5 runs. This is the setting described in Section 7 text as the performance for that workload.

### 7.2 Data

Throughput across 5 runs (mean ± standard deviation)

#### Synthetic Load

With Razor's Edge (RMS):

[13.3, 13.3, 13.2, 13.3, 13.2], 13.3 +- 0.05

With no batching:

[11.4, 11.3, 11.4, 11.4, 11.4], 11.4 +- 0.04

#### Real load Basic 

from `demos/real/gpu benchmark performance comparison.ipynb`

With Razor's Edge (RMS):

[35.8, 35.5, 36.3, 34.4, 34.4], 35.3 +- 0.76

With no batching:

[27.9, 28.1, 27.6, 29.0, 28.4], 28.2 +- 0.48

#### Real load limited tokens

from `demos/real/gpu benchmark performance comparison basic limited.ipynb`

With Razor's Edge (RMS):

[85.1, 82.8, 83.3, 86.0, 85.9], 84.6 +- 1.33

With baseline batching (batch size = 10; best among batch sizes 1, 2, 8, 10, 13, 16):

[68.1, 69.0, 68.3, 66.0, 65.5], 67.4 +- 1.37

With no batching:

[42.0, 37.9, 45.5, 41.1, 40.6], 41.4 +- 2.46

#### Synthetic Latency Comparison with Different Strategies

`demos\synthetic\dummy latency comparison.ipynb`

RMS shows lower rms, mean and p95 latency as expected.

FIFO shows higher throughput. FIFO shows lower P99 and Max latency.

MINMAX shows the worst latencies in avg, rms and mean. MINMAX has almost equal p95, p99 and max latency.  MINMAX is slightly better than RMS at p99 and max latency. 

| Strategy | RPS | RMS Latency | Mean Latency | P95 Latency | P99 Latency | Max Latency |
| --- | --- | --- | --- | --- | --- | --- |
| RMS | 13.113518 | 6.479189 | 4.902618 | 13.512671 | 15.257354 | 15.261522 |
| FIFO | 13.435045 | 8.555332 | 7.346844 | 14.084836 | 14.884194 | 14.890185 |
| MINMAX | 13.199825 | 11.006600 | 10.000695 | 15.145113 | 15.158157 | 15.158657 |

#### Real Latency Comparison with Different Strategies, limited token input

`demos\synthetic\dummy latency comparison.ipynb`

Not many trends can be seen due to noise.

For rms and mean latency the increasing order of latency is - rms > fifo > minmax (same as synthetic).

| Strategy | RPS | RMS Latency | Mean Latency | P95 Latency | P99 Latency | Max Latency |
| --- | --- | --- | --- | --- | --- | --- |
| RMS | 95.433185 | 1.151876 | 0.974708 | 2.043573 | 2.090140 | 2.096659 |
| FIFO | 95.962956 | 1.240765 | 1.091206 | 1.983491 | 2.080144 | 2.080418 |
| MINMAX | 95.411471 | 1.419826 | 1.284810 | 2.061702 | 2.096055 | 2.099771 |

### 7.2.1 Practical Strategy-Selection Rule (Deployment Guidance)

Until per-strategy latency tables are complete for a target deployment, use the following operational policy:
- **RMS** as default for balanced throughput + average latency behavior.
- **FIFO** when fairness/oldest-first service is a requirement and throughtput is more important.
- **MINMAX** when controlling worst-case latency is the top priority, with acceptance of potentially higher average latency and even worse throughput.

Once Section 7.2 per-strategy numbers are filled in, this rule should be validated or adjusted for each production workload and SLO profile.

### 7.3 Hardware and software

#### Hardware and OS

Experiments were conducted on an ASUS ROG Zephyrus Duo 16 laptop.
- CPU: AMD Ryzen 9 6900HX (16 threads) (power limits: SPL 20W, sPPT 20W, fPPT 25W)
- GPU: NVIDIA RTX 3080 Ti Laptop GPU (835 MHz fixed max clock, 185 MHz core OC, 210 MHz memory OC, 5 W dynamic boost, 80°C temperature target)
- OS: Windows 24H2 IoT with AtlasOS modifications
- Cooling: maximum fan speeds; no thermal throttling observed

The most stable practical environment was used, with no background application load and Python run at high process priority (using `psutil`) to improve timing stability.
Graphs/figures shown are least-noisy representative runs for visual readability, while all throughput values reported in text are means across 5 runs per setting.

#### Software

Python 3.14.3

coverage                  7.13.5
huggingface_hub           1.7.2
ipython                   9.11.0
jupyter                   1.1.1
numba                     0.64.0
numpy                     2.4.3
onnx                      1.20.1
onnxruntime               1.24.4
optimum                   2.1.0
optimum-onnx              0.1.0
pip                       26.0.1
safetensors               0.7.0
scikit-learn              1.8.0
scipy                     1.17.1
tokenizers                0.22.2
torch                     2.10.0+cu126

### 7.4 Workloads

Synthetic workload: We construct a dummy model where processing time is a function of input size and batch size. This isolates batching effects and enables clear visualization of performance structure. The synthetic workload uses an accurate delay based on batch input size. RMS latency objective is used unless specified otherwise.
Real GPU workload: We evaluate a batched inference pipeline using the `BAAI/bge-m3` model on gpu.
CPU workload: We evaluate the shape of the batching performance contours of `jinaai/jina-embeddings-v2-base-en` on cpu.

Requests were created using uniform-random characters with varying lengths:
- 1 to 1000 characters for the synthetic task
- 1 to 1300 chars (corresponding to 1 to 1000 tokens) for the real task `BAAI/bge-m3`
- 1 to 500 chars (corresponding to 1 to 400 tokens) for the limited real task `BAAI/bge-m3`

### 7.5 Throughput under Increasing Load

We next evaluate system throughput as a function of load by varying the number of concurrent requests.

The comparison figures in this section show Razor's Edge against the baseline dynamic batching strategy. The baseline exhibits approximately constant throughput as load increases, reflecting its reliance on fixed-size aggregation without improved batching quality. In contrast, Razor's Edge shows increasing throughput with load up to saturation.


**Figure 5:** Baseline scheduler throughput vs. parallelism (synthetic workload).  
**Notebook source:** `demos/synthetic/dummy performance comparison.ipynb`.

![BaseBatchedDummyTask Throughput vs Parallelism](images/BaseBatchedDummyTask%20Throughput%20vs%20Parallelism.png)

**Figure 6:** Razor's Edge scheduler throughput vs. parallelism (synthetic workload).  
**Notebook source:** `demos/synthetic/dummy performance comparison.ipynb`.

![RazorsEdgeDummyTask Throughput vs Parallelism](images/RazorsEdgeDummyTask%20Throughput%20vs%20Parallelism.png)

**Figure 7:** Direct throughput comparison of baseline and Razor's Edge (synthetic workload).  
**Notebook source:** `demos/synthetic/dummy performance comparison.ipynb`.

![BaseBatchedDummyTask and RazorsEdgeDummyTask Throughput vs Parallelism](images/BaseBatchedDummyTask%20and%20RazorsEdgeDummyTask%20Throughput%20vs%20Parallelism.png)

**Figure 8:** Baseline scheduler throughput vs. parallelism (real workload).  
**Notebook source:** `demos/real/gpu benchmark performance comparison.ipynb`.

![BaseBatchedGPUBenchmarkTask Throughput vs Parallelism](images/BaseBatchedGPUBenchmarkTask%20Throughput%20vs%20Parallelism.png)

**Figure 9:** Razor's Edge scheduler throughput vs. parallelism (real workload).  
**Notebook source:** `demos/real/gpu benchmark performance comparison.ipynb`.

![RazorsEdgeGPUBenchmarkTask Throughput vs Parallelism](images/RazorsEdgeGPUBenchmarkTask%20Throughput%20vs%20Parallelism.png)

**Figure 10:** Direct throughput comparison of baseline and Razor's Edge (real workload).  
**Notebook source:** `demos/real/gpu benchmark performance comparison.ipynb`.

![BaseBatchedGPUBenchmarkTask and RazorsEdgeGPUBenchmarkTask Throughput vs Parallelism](images/BaseBatchedGPUBenchmarkTask%20and%20RazorsEdgeGPUBenchmarkTask%20Throughput%20vs%20Parallelism.png)

This effect arises because larger queue sizes enable better sorting of input sizes, allowing the scheduler to construct more efficient batches. As concurrency increases, the dynamic program has greater flexibility to select near-optimal partitions, improving overall hardware utilization.

We observe this trend clearly in both synthetic and real workloads. These results demonstrate that Razor's Edge can exploit queue depth to improve batching efficiency, partially offsetting increased-load effects.

### 7.6 Razor's Edge Structure in Batching Efficiency

We first examine how batching efficiency varies with input size.

On the synthetic workload, we plot throughput as a function of minimum and maximum input sizes within a batch. In this tested setup, the contour exhibits a transition where smaller inputs permit efficient batching across wider size ranges, while larger inputs only benefit from tighter within-batch size uniformity and suffer otherwise. This measured transition forms the "razor's edge" structure in that workload's performance landscape.

For these experiments, this pattern is consistent with a transition between memory-dominated to compute-dominated behavior.

In the real `BAAI/bge-m3` workload on gpu, we observe a qualitatively similar contour trend with higher measurement noise.

In the real `jinaai/jina-embeddings-v2-base-en` workload on cpu we observe a qualitatively similar contour shapes.

This supports applicability to the evaluated workload class (variable-size batched inference with calibrated estimators).

We do not guarantee the same structure on untested models and hardware. 

**Figure 11:** Improvement from allowing variable batch sizes (synthetic task performance contours).  
**Notebook source:** `demos/synthetic/razors edge dummy graphs.ipynb`.

![Improvement by Allowing Different Batch Sizes for RazorsEdgeDummyTask](images/Improvement%20by%20Allowing%20Different%20Batch%20Sizes%20for%20RazorsEdgeDummyTask.png)

**Figure 12:** Improvement from allowing variable batch sizes (real-model gpu performance contours).  
**Notebook source:** `demos/real/razors edge gpu benchmark graphs.ipynb`.

![Improvement by Allowing Different Batch Sizes for RazorsEdgeGPUBenchmarkTask](images/Improvement%20by%20Allowing%20Different%20Batch%20Sizes%20for%20RazorsEdgeGPUBenchmarkTask.png)


**Figure 12:** Improvement from allowing variable batch sizes (real-model cpu performance contours).  
**Notebook source:** `demos/cpu/azors edge cpu benchmark graphs.ipynb`.
![Improvement by Allowing Different Batch Sizes for RazorsEdgeCPUBenchmarkTask](images/Improvement%20by%20Allowing%20Different%20Batch%20Sizes%20for%20RazorsEdgeCPUBenchmarkTask.png)

### 7.7 Summary of Findings

Across both workloads, we observe:
- Razor's Edge outperforms simple batching strategies
- Razor's Edge enables performance gains even when simple batching strategies degrade performance relative to no batching
- Input size dependent batching structure: batching efficiency exhibits a transition driven by memory-bound vs. compute-bound regimes.
- Throughput improvement under load: Razor's Edge increases effective throughput as concurrency grows by enabling better batch construction.

Together, these results demonstrate that, on the evaluated workloads and setup, Razor's Edge improves batching efficiency and exhibits load-adaptive behavior relative to the baselines considered in this paper.


## 8. Limitations and Future Work
- Partitioning and RMS calculations are expensive and cause CPU overhead. This may cause slowdowns in the case of small models, low-CPU environments, or large batch sizes.
- Timing estimators could be created directly from model architecture and benchmarking basic operations.
- It is unknown why FIFO seems to consistently have slightly higher throughput. This is theorized to be because FIFO ends up choosing more "efficient" small, fast batches at random while RMS and MINMAX actively discriminiate against "efficient" small, fast batches.

Future work includes:
1. Better estimators by analysis of model components
2. Estimator uncertainty modeling and online recalibration.
3. Throughput may be further improved by selecting batches that are most "efficient".

### 8.1 Threats to Validity

To clarify interpretation boundaries, we separate validity risks into three categories:

- **Internal validity (measurement and implementation effects):** timing on commodity systems is sensitive to transient scheduler state, thermal effects, and runtime noise. We mitigate this by repeated runs and reported variation, but residual noise remains.
- **Construct validity (objective-to-quality mapping):** we optimize throughput and \(\sum_i \ell_i^2\) (RMS-related squared latency objective). These are useful operational proxies, but production SLOs may weight p95/p99 latency, fairness, or tenant isolation differently.
- **External validity (generalization):** experiments in this manuscript focus on synthetic workloads and `BAAI/bge-m3` under the reported setup. Gains may differ for other model architectures, kernels, serving stacks, accelerator classes, and request-size distributions.


## 9. Conclusion
Razor's Edge provides a practical batching framework that unifies throughput optimization and RMS latency for variable-size inference workloads.

Our core contribution is a systems synthesis: DP-based contiguous partitioning on sorted requests, an RMS-guided online ordering pass, and practical estimator construction for deployment. In the evaluated synthetic and real (`BAAI/bge-m3`) settings with calibrated timing estimators, this combination improves throughput and maintains favorable latency behavior relative to the tested FIFO/fixed-cap baselines.

Validated in this paper:
- On the reported experiments, the proposed scheduler outperforms the defined baselines on throughput for the tested variable-size batched inference tasks.
- The observed contour structure and load-scaling behavior are reproducible within the tested synthetic and `BAAI/bge-m3` setups.

Not claimed:
- A universal guarantee across all serving stacks, models, and hardware.
- Global optimality of the multi-group online RMS ordering pass.

Accordingly, the method should be interpreted as an objective-driven, empirically supported scheduler design for the evaluated workload class, rather than a general theorem about all inference systems.

## 10. Funding and Conflict of Interest
- **Funding:** None.
- **Conflict of interest:** The author declares no competing financial or non-financial interests related to this work.

## 11. Ethical Considerations / Potential Misuse
This work improves serving efficiency and resource allocation for batched inference systems. In deployment, operators should monitor service stability if these mechanisms are integrated. Efficiency gains can also vary based on random variance at startup or request size distribution; operators should apply appropriate usage controls, audit logging, and governance aligned with their application domain. Latency objective should be selected carefully based on application.

## 12. Reproducibility Notes
Repo <https://github.com/arrmansa/Razors-Edge-batching-scheduler>
To reproduce results (throughput and RMS transitivity) and figures, install dependencies from `requirements.txt`, open each notebook in `/demos`, and execute **Run All** from the first cell in a clean kernel. Generated plots will be saved in `/images`.

Seeds are set in notebooks where appropriate for deterministic or near-deterministic sections of the experiments.

We keep benchmark-critical dependencies in `requirements.txt`; non-benchmark package version changes (for example notebook tooling/plotting helpers) should not materially change measured throughput trends.

Absolute throughput values can vary across hardware and runtime conditions, but the contour-shape phenomenon ("razor's edge" transition between efficient and inefficient batching regions) is expected to be present on most modern hardware and models.

Notebook-to-figure mapping for paper images:
- Figure 1 (`images/Synthetic Load Basic Benchmarks.png`) <- `demos/synthetic/dummy performance comparison basic.ipynb`
- Figure 2 (`images/Real Load Basic Benchmarks.png`) <- `demos/real/gpu benchmark performance comparison basic.ipynb`
- Figure 3 (`images/Real Limited Load Basic Benchmarks.png`) <- `demos/real/gpu benchmark performance comparison basic limited.ipynb`
- Figure 4 (`images/Synthetic Load OldestPicker RMS Benchmarks.png`) <- `demos/synthetic/dummy performance comparison rms.ipynb`
- Figure 5 (`images/BaseBatchedDummyTask Throughput vs Parallelism.png`) <- `demos/synthetic/dummy performance comparison.ipynb`
- Figure 6 (`images/RazorsEdgeDummyTask Throughput vs Parallelism.png`) <- `demos/synthetic/dummy performance comparison.ipynb`
- Figure 7 (`images/BaseBatchedDummyTask and RazorsEdgeDummyTask Throughput vs Parallelism.png`) <- `demos/synthetic/dummy performance comparison.ipynb`
- Figure 8 (`images/BaseBatchedGPUBenchmarkTask Throughput vs Parallelism.png`) <- `demos/real/gpu benchmark performance comparison.ipynb`
- Figure 9 (`images/RazorsEdgeGPUBenchmarkTask Throughput vs Parallelism.png`) <- `demos/real/gpu benchmark performance comparison.ipynb`
- Figure 10 (`images/BaseBatchedGPUBenchmarkTask and RazorsEdgeGPUBenchmarkTask Throughput vs Parallelism.png`) <- `demos/real/gpu benchmark performance comparison.ipynb`
- Figure 11 (`images/Improvement by Allowing Different Batch Sizes for RazorsEdgeDummyTask.png`) <- `demos/synthetic/razors edge dummy graphs.ipynb`
- Figure 12 (`images/Improvement by Allowing Different Batch Sizes for RazorsEdgeGPUBenchmarkTask.png`) <- `demos/real/razors edge gpu benchmark graphs.ipynb`

The paper text describes implemented behavior and references concrete function names in:
- `src/razors_edge/optimal_batching.py`
- `src/razors_edge/optimal_benchmarking.py`
- `src/razors_edge/razors_edge_compute_task.py`
- `src/executor/base_batched_compute_task.py`
- `src/executor/process_manager.py`


## 13. References

[1] W. E. Smith, "Various optimizers for single-stage production," *Naval Research Logistics Quarterly*, vol. 3, no. 1-2, pp. 59-66, 1956.

[2] U. Bagchi, Y. L. Chang, and R. S. Sullivan, "Minimizing squared deviation of completion times about a common due date," Manag. Sci., vol. 33, no. 7, pp. 894–906, Jul. 1987.

[3] S. K. Gupta and T. Sen, "Minimizing a quadratic function of job lateness on a single machine," Eng. Costs Prod. Econ., vol. 7, no. 3, pp. 187–194, Sep. 1983.

[4] NVIDIA, "Dynamic Batcher," Triton Inference Server Documentation. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html (accessed 2026-03-24).

[5] mixedbread-ai, "batched" (GitHub repository). https://github.com/mixedbread-ai/batched (accessed 2026-03-24).

[6] michaelfeil, "infinity" integrations (GitHub repository). https://github.com/michaelfeil/infinity?tab=readme-ov-file#integrations (accessed 2026-03-24).

[7] Hugging Face, "Pipeline batch inference" documentation. https://huggingface.co/docs/transformers/pipeline_tutorial#batch-inference (accessed 2026-03-24).
