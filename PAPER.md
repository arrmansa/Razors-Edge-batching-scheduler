# Razor's Edge: Throughput-Optimal Dynamic Batching with Mean Square Latency Optimization

## Abstract
Serving systems for embedding, LLM, and other matrix-multiplication-dominated inference workloads rely on batching for efficient hardware utilization. We observe that batching efficiency exhibits a sharp input size dependent structure driven by the transition between memory-bound and compute-bound regimes: small inputs can be batched flexibly across heterogeneous sizes, while large inputs require near-uniformity, leading to a rapid collapse in batching efficiency. This produces a characteristic blade like (razor's edge) shape in the batch performance landscape.

We present the Razor's Edge batching scheduler, a practical framework that combines (i) dynamic-programming-based throughput optimization over sorted requests, (ii) a root-mean-square latency objective for next-batch selection, and (iii) startup-time-efficient model benchmarking that builds batch timing estimators for real hardware. The approach is designed for real-time online serving with queueing. The method is broadly applicable to batched inference systems across CPU and GPU deployments.

## 1. Introduction
Requests cannot be processed instantaneously at arrival in any real serving system. Under load, queueing is inevitable. A first-come-first-serve single-request policy is simple, but it fails to exploit the throughput gains available through batching. Simply batching the first n requests in a queue is common but is also not the most optimal option in terms of throughput. Sorting requests by size and batching after creates a problem of high latency for large requests that may not be picked up under load.

This work addresses a common inference regime where workloads are dominated by batched matrix multiplication with variable input sizes (for example, variable token lengths in embedding or classification models). The scheduler pursues two goals simultaneously:

1. **Maximum throughput**: Sort and choose batch partitions that minimize total completion time for queued work.
2. **Minimum mean squared latency**: choose near-term execution order that minimizes sum of square of latencies of the queued work.

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


## 3. Problem Setting and Model Assumptions
Let 
```math
F([x_1, ..., x_n]) = [y_1, ..., y_n]
```

be a batch inference function executed on system S, returning one output per input.

### 3.1 Consistent Practical Utility
For an input \(x_i\), denote singleton output as \(a\), and output in different batching contexts as \(y_i\) or \(d_i\). Exact equality is not required; practical equivalence (e.g., for retrieval, ranking, or classification quality) is sufficient.

### 3.2 Consistent Timing

1. The execution time of a single batch should not be dependent on the internal ordering of that batch.
2. The execution time of a batch should be monotonically increasing with the number of elements in the batch and the size of the largest element in the batch.
3. If 2 batches have the same number of elements and the largest element is of the same size, then the time taken by both batches is the same
4. The largest element in the batch is one that would take the maximum time to compute on it's own.

These 4 a axioms mply batching on sorted `x_i` is faster than batching on non-sorted partitions.

#### Proof

Consider 2 internally ascending ordered batches from a set of batches which were created by optimally partitioning batches to minimize time on unsorted inputs.
We can always find 2 batches with atleast 2 elements each such that given 2 internally ascending ordered batches
```math
B1 = [x_1, x_2, ... x_m]
```

```math
B2 = [y_1, y_2, ... y_n]
```
and x_m > y_1 and y_n > x_1

Now there are 2 possibilities

1. the size of x_n <= size of y_m
We can now construct a second set of batches B1' and B2' after sorting such that
```math
B1' = [x_1, x_2, ... x_m-1, y_1]
```

```math
B2' = [x_m, y_2, ... y_n]
```
In which case time of B1' is less than time of B1 while time of B2 and B2' remain the same. We can repeat this sorting step until the batches are equivalent to ones that could be formed by partitioning on sorted inputs.

2. the size of x_n > size of b_m

We can now construct a second set of batches B1" and B2" after sorting such that
```math
B1" = [y_n, x_2, ... x_m]
```

```math
B2" = [y_1, y_2, ... y_n-1, x_1]
```

In which case time of B2" is less than time of B2 while time of B1 and B1" remain the same. We can repeat this sorting step until the batches are equivalent to ones that could be formed by partitioning on sorted inputs.

### 3.3 Applicability

The method targets systems where memory and compute bottlenecks produce stable compute time vs batch size vs input size structure under batching. This includes many PyTorch, TensorFlow, ONNX, and OpenVINO pipelines for embeddings, classification, and related batched inference tasks.

## 4. Core Scheduling Method
Given a queue of requests there are 4 main steps.
1. We first order the requests based on their size.
2. We then parition them into batches to minimize total time using dynamic programming.
3. We do a single ordering pass to minimize mean squared latency.
4. We pick the first batch, and update the expected start time of the next batch

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

We employ a weighted squared latency objective with a pairwise interchange argument in the spirit of classical single-machine sequencing proofs [1]. The use of squared penalties follows quadratic completion/lateness formulations that more strongly penalize tail outcomes than linear objectives [2], [3]. In our setting, the resulting priority score jointly depends on batch processing duration and accumulated waiting time, which directly targets latency variance reduction.

Given request arrivals (time-ordered) `R1, R2, ...` at times `x0, x1, ...` such that `x_i < x_(i+1)`, with processing start times `y0, y1, ...` and output times `z0, z1, ...`, a system with minimum mean squared latency minimizes:

`sum((z_i - x_i)^2)` over all `i`.

For a traditional one-by-one processor, `y_i = z_(i-1)`.

#### Two-request case

Let `R1` and `R2` arrive while current request `R0` is processing.

Let:

- `R1` processing time be `T1`
- `R2` processing time be `T1 * r_p`, where `r_p` is processing-time ratio (`0..1`)
- `R1` waiting time be `T1 * w1`
- `R2` waiting time be `T1 * w1 * r_w`, where `r_w` is waiting-time ratio (`0..1`)

Pick `R1` before `R2` if:

```math
(T1*w1+T1)^2 + (T1*w1*r_w+T1+T1*r_p)^2 < (T1*w1*r_w + T1*r_p)^2 + (T1*w1+T1*r_p+ T1)^2
```
This simplifies to:
```math
r_p > sqrt(w1^2 + 2*w1*r_w + 1) - w1
```
Note: (Not used, but has some interesting plots/ improvements)  A linear approximation between `r_p` and `r_w`:
```math
r_p > sqrt(w1^2 + 2*w1*r_w + 1) - w1 + r_w * (1 + w1 - sqrt(w1^2 + 2*w1*r_w + 1))
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
sum(Ti * len(Gi) * (Ti - 2 * mean(Gi_Wj))
```


This shows exact per-request arrival terms are constant regardless of group ordering. The ordering-relevant term is the average arrival time within each group.

##### For 2 groups:

- Group `R1` has duration `D1`, size `N1`, avg waiting `W1`
- Group `R2` has duration `D2`, size `N2`, avg waiting `W2`

Define:

```math
RMS_LATENCY_1 = N1 * (W1 + D1) ** 2 + N2 * (W2 + D1 + D2) ** 2 \
RMS_LATENCY_2 = N2 * (W2 + D2) ** 2 + N1 * (W1 + D1 + D2) ** 2
```

If `RMS_LATENCY_1 < RMS_LATENCY_2`, pick `R1` before `R2`.

This simplifies to:

```math
If `N2 * D1 * (W2 + D2 + D1/2) < N1 * D2 * (W1 + D1 + D2/2)`, pick `R1` before `R2`.
```

Practically since we will be using 64 bit math with nanoseconds, and a batch will be limited to 256 (uint8). For numerical safety we use `log2((2**63 / (256 * 5)) ** 0.5) = 26.8247...` to see that max precision of x to be 26 bits.

##### For more than 2 Groups

For more than 2 batches, this property is not transitive, and we don't expect perfect outputs in 1 pass of selecting the best request. However it is close to optimal if we check by brute force (N, D and T from 1 to 10)
For 3 requests (i, j and k) we expect the optimal ordering to start with i first 99% of the time if 
1. i should be processed first given only i and j
2. j should be processed first given only j and k
For 3 requests (i, j and k) we expect the optimal ordering to start with i first 99.995% of the time if 
1. i should be processed first given only i and j
2. i should be processed first given only i and k
This should be sufficient for real time scheduling.
There maybe small performance gains by sorting batches by time required before doing RMS comparison, however given the rarity and performance cost it is likely not worth doing for real time serving systems.

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

- `_prospective_rms_latency_improvement(...)` compares two orderings of two candidate groups (`chosen` then `prospective` vs the reverse) using the simplified inequality:
```math
N_c D_p \left(W_c + D_c + \frac{W_p}{2}\right) \;<\; N_p D_c \left(W_p + D_p + \frac{W_c}{2}\right)
```
where `N` is group size, `D` is duration, and `W` is mean queueing time. This is algebraically equivalent to comparing total squared latencies, after rearrangement and constant-term cancellation.

- `_get_slice_indexes_and_duration(...)` reconstructs candidate batches along the optimal-throughput DP backpointer chain and selects
```math
b^\*=\arg\min_{b \in \mathcal{C}} \sum_{i\in b}(w_i + D_b)^2
```
in one pass via pairwise comparisons (`\mathcal{C}` is the set of DP-derived contiguous candidates ending on the current suffix).

## 5. Benchmarking and Estimator Construction

A dynamic program requires a 2d array that should contain the expected duration to process a batch of given it's size and the size of the largest element in it by indexing `arr[batch_size, max_input_size]`. Creating this using actual data would be prohibitively slow. Instead for each batch size we select some points and extrapolate with a spline. Additionally, we might see slow / redundant sata for very high `batch_size` and `max_input_size`. Where the numbers simply correspond to the previous batch size timing `arr[batch_size-1, max_input_size]*batch_size/(batch_size-1)` just scaled due to the hardware reaching saturation. To speed this up, we detect this saturation point while benchmarking.

Function: `RazorsEdgeComputeTask.get_batch_timing_data` in `src\razors_edge\razors_edge_compute_task.py`.

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

Base class: `RazorsEdgeComputeTask` in `tests\src\razors_edge\test_razors_edge_compute_task.py`.

At runtime, batch selection is a composition:
```math
\text{inputs} \xrightarrow{\text{preprocess}} (u_i,s_i)
\xrightarrow{\text{sort by }s_i} \xrightarrow{\text{DP+RMS}} [i:j)
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

For the synthetic task, razors edge was able to obtain a peak throughput of 13.4 RPS (17% higher) with max allowed batch size of 8
For the synthetic task, the base strategy was able to obtain a peak throughput of 11.4 RPS with best performance when batch size was limited to 1. Naive batching reduced performance for the sythetic task.

![Synthetic Load Basic Benchmarks](images/Synthetic%20Load%20Basic%20Benchmarks.png)

For the actual model, razors edge was able to obtain a peak throughput of 35.9 RPS (28% higher) with max allowed batch size of 16
For the actual model, the base strategy was able to obtain a peak throughput of 27.9 RPS with best performance when batch size was limited to 1. Naive batching reduced performance for the normal model.

![Real Load Basic Benchmarks](images/Real%20Load%20Basic%20Benchmarks.png)

For the actual model with limited token input, razors edge was able to obtain a peak throughput of 84.2 RPS (33% higher) with max allowed batch size of 16
For the actual model, the base strategy was able to obtain a peak throughput of 63.3 RPS with best performance when batch size was 10. Naive batching improved performance compared to the non batched 43.2 RPS (32% lower).

![Real Limited Load Basic Benchmarks](images/Real%20Limited%20Load%20Basic%20Benchmarks.png)

### 7.1 Hardware
Experiments are conducted on Asus Duo 16 Laptop. 
- CPU: AMD Ryzen 9 6900HX 16 Core cpu (Power limits: SPL 20W, sPPT 20W, fPPT 25W)
- GPU: NVIDIA RTX 3080 Ti laptop GPU (835MHz fixed max clock, 185mhz core OC, 210Mhz Memory OC, 5W dynamic boost, 80C temperature target)
- OS: Windows 24H2 Iot with AtlasOS Mod
- Maximum Fan speeds, No thermal throttling was observed

The most stable possible settings were used with no background processes and python on high priority to ensure timing stability.
Results shown are handpicked least noisy runs.

### 7.3 Workloads.

Synthetic workload. We construct a dummy model where processing time is a function of input size and batch size. This allows isolation of batching effects and clear visualization of performance structure. The synthetic workload uses and accurate delay based on input size of the batch.
Real workload. We evaluate on a batched inference pipeline using the `BAAI/bge-m3` model, where execution time reflects actual system behavior, including noise and hardware effects.
Requests were created with uniform random characters of varying lengths
- 1 to 1000 characters for the synthetic task
- and 1 to 1300 chars (corresponding to 1 to 1000 tokens) for the real task.
- and 1 to 500 chars (corresponding to 1 to 400 tokens) for the limited real task.

### 7.4 Razor's Edge Structure in Batching Efficiency

We first examine how batching efficiency varies with input size.

On the synthetic workload, we plot throughput as a function of minimum and maximum input sizes within a batch. The resulting contour exhibits a sharp transition: small inputs allow efficient batching across heterogeneous sizes, while large inputs require near-uniformity to maintain efficiency. This produces a characteristic "razor's edge" shape in the performance landscape.

This structure arises from the transition between memory-bound and compute-bound regimes. When inputs are small, memory dominates and batching remains efficient across a wide range of sizes. As input size increases, compute effects dominate, and batching efficiency becomes highly sensitive to size variation.

We observe qualitatively similar behavior in the real workload, though with increased noise due to hardware variability. The persistence of this structure supports the applicability of the model to many practical systems.

Base scheduler throughput vs parallelism.

![BaseBatchedDummyTask Throughput vs Parallelism](images/BaseBatchedDummyTask%20Throughput%20vs%20Parallelism.png)

Razor's Edge scheduler throughput vs parallelism.

![RazorsEdgeDummyTask Throughput vs Parallelism](images/RazorsEdgeDummyTask%20Throughput%20vs%20Parallelism.png)

Direct throughput comparison of baseline and Razor's Edge.

![BaseBatchedDummyTask and RazorsEdgeDummyTask Throughput vs Parallelism](images/BaseBatchedDummyTask%20and%20RazorsEdgeDummyTask%20Throughput%20vs%20Parallelism.png)

Base scheduler throughput vs parallelism.

![BaseBatchedGPUBenchmarkTask Throughput vs Parallelism](images/BaseBatchedGPUBenchmarkTask%20Throughput%20vs%20Parallelism.png)

Razor's Edge scheduler throughput vs parallelism.

![RazorsEdgeGPUBenchmarkTask Throughput vs Parallelism](images/RazorsEdgeGPUBenchmarkTask%20Throughput%20vs%20Parallelism.png)

Direct throughput comparison of baseline and Razor's Edge.

![BaseBatchedGPUBenchmarkTask and RazorsEdgeGPUBenchmarkTask Throughput vs Parallelism](images/BaseBatchedGPUBenchmarkTask%20and%20RazorsEdgeGPUBenchmarkTask%20Throughput%20vs%20Parallelism.png)

### 7.5 Throughput under Increasing Load

We next evaluate system throughput as a function of load by varying the number of concurrent requests.

Figure X compares Razor's Edge with the baseline dynamic batching strategy. The baseline exhibits approximately constant throughput as load increases, reflecting its reliance on fixed-size aggregation without improved batching quality. In contrast, Razor's Edge shows increasing throughput with load up to saturation.

Improvement from allowing variable batch sizes (synthetic task performance contours).

![Improvement by Allowing Different Batch Sizes for RazorsEdgeDummyTask](images/Improvement%20by%20Allowing%20Different%20Batch%20Sizes%20for%20RazorsEdgeDummyTask.png)

Improvement from allowing variable batch sizes (actual model performance contours).

![Improvement by Allowing Different Batch Sizes for RazorsEdgeGPUBenchmarkTask](images/Improvement%20by%20Allowing%20Different%20Batch%20Sizes%20for%20RazorsEdgeGPUBenchmarkTask.png)

This effect arises because larger queue sizes enable better sorting of input sizes, allowing the scheduler to construct more efficient batches. As concurrency increases, the dynamic program has greater flexibility to select near-optimal partitions, improving overall hardware utilization.

We observe this trend clearly in both the synthetic workload and the real workload. These results demonstrate that Razor's Edge can exploit queue depth to improve batching efficiency, partially offsetting the effects of increased load. This is a property that may be desirable in distributed systems.

### 7.6 Summary of Findings

Across both workloads, we observe:
- Razors Edge outperforms simple batching strategies
- Razors Edge enables performance gains even when simple batching strategies cause performance degradation compared to no batching
- Input size dependent batching structure. Batching efficiency exhibits a transition driven by memory vs compute-bound regimes.
- Throughput improvement under load. Razor's Edge increases effective throughput as concurrency grows by enabling better batch construction.

Together, these results demonstrate that Razor's Edge not only improves batching efficiency but also exhibits load adaptive behavior that differs fundamentally from conventional batching strategies.


## 8. Limitations and Future Work
- Partitioning and rms calculations are expensive and cause CPU overhead. This may cause slowdowns in the case of small models, low cpu environments, or large batch sizes.
- Timing Estimators could be created directly from model architecture and benchmarking basic operations

Future work includes:
1. Better estimators by analysis of model components
2. estimator uncertainty modeling and online recalibration.
3. Selection of batches could be done by selecting the most "efficient" batch.


## 9. Conclusion
Razor's Edge provides a practical batching framework that unifies throughput optimization and RMS latency for variable-size inference workloads. This makes the approach suitable for production serving environments where both latency behavior and utilization matter.

## 10. Reproducibility Notes
Repo <https://github.com/arrmansa/Razors-Edge-batching-scheduler>
The paper text describes implemented behavior and references concrete function names in:
- `src/razors_edge/optimal_batching.py`
- `src/razors_edge/optimal_benchmarking.py`
- `src/razors_edge/razors_edge_compute_task.py`
- `src/executor/base_batched_compute_task.py`
- `src/executor/process_manager.py`


## 11. References

[1] W. E. Smith, "Various optimizers for single-stage production," *Naval Research Logistics Quarterly*, vol. 3, no. 1-2, pp. 59-66, 1956.

[2] U. Bagchi, Y. L. Chang, and R. S. Sullivan, "Minimizing squared deviation of completion times about a common due date," Manag. Sci., vol. 33, no. 7, pp. 894–906, Jul. 1987.

[3] S. K. Gupta and T. Sen, "Minimizing a quadratic function of job lateness on a single machine," Eng. Costs Prod. Econ., vol. 7, no. 3, pp. 187–194, Sep. 1983.

[4] NVIDIA, "Dynamic Batcher," Triton Inference Server Documentation. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html

[5] mixedbread-ai, "batched" (GitHub repository). https://github.com/mixedbread-ai/batched

[6] michaelfeil, "infinity" integrations (GitHub repository). https://github.com/michaelfeil/infinity?tab=readme-ov-file#integrations

[7] Hugging Face, "Pipeline batch inference" documentation. https://huggingface.co/docs/transformers/pipeline_tutorial#batch-inference
