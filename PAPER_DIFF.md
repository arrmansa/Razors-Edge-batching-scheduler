# Markdown vs LaTeX Diff Check

Generated on 2026-03-31.

## Structural counts
- Markdown lines: 1172
- LaTeX lines (abstract + main): 2275
- Markdown images: 19
- LaTeX `\includegraphics`: 19
- Markdown fenced code blocks: 46
- LaTeX verbatim blocks: 46

## Unified diff sample (first 220 lines)
```diff
--- PAPER.md

+++ PAPER_(abstract+main).tex

@@ -1,543 +1,1157 @@

-# Razor's Edge: Throughput Optimized Dynamic Batching with Latency Objectives
-
-**Author:** Arrman Anicket Saha  
-**Affiliation:** Independent researcher (unaffiliated)  
-**Contact (GitHub):** <https://github.com/arrmansa>  
-**Email:** arrmansa99@gmail.com  
-**ORCID:** 0009-0004-6884-0644  
-**Date:** March 26, 2026
-
-## Abstract
-Serving systems for embedding, LLM, and other matrix-multiplication-dominated inference workloads rely on batching for efficient hardware utilization. We observe that batching efficiency exhibits a sharp input-size-dependent structure driven by the transition between memory-bound and compute-bound regimes: small inputs can be batched flexibly across heterogeneous sizes, while large inputs require near-uniformity, leading to a rapid collapse in batching efficiency. This produces a characteristic blade-like ("razor's edge") shape in the batch performance landscape.
-
-We present the Razor's Edge batching scheduler, a practical framework that combines (i) dynamic-programming-based throughput optimization over sorted requests, (ii) production-oriented next-batch selection strategies (`FIFO`, `MINMAX`, and `GUARDED_BATCH_SIZE`), and (iii) startup-time-efficient model benchmarking that builds batch timing estimators from direct measurements on the same hardware where the model is deployed. A central novelty claim in this paper is this measurement-to-optimizer bridge: instead of relying on analytic proxy cost models, we benchmark the deployed model/hardware pair and feed those empirical timings directly into the DP cost table used for scheduling decisions. We also introduce a practical visualization method for quantifying batching efficiency improvements when expanding the allowed maximum batch size from \(N-1\) to \(N\), producing the characteristic "razor's edge" contour plots. The approach is designed for real-time online serving with queueing. Our claims are scoped to "ahead-of-time variable-size batching for encoder-style inference" evaluated in this paper, not to universal superiority across all serving stacks. We demonstrate the scheduler's efficacy through a 47% throughput increase on a CPU embedding workload (`jina-embeddings-v2-base-en`), a 26% throughput increase on a GPU embedding workload (`BAAI/bge-m3`), and controllable latency/throughput trade-offs across the final strategy set.
-
-## Claim Language Freeze
-To avoid claim drift, this manuscript uses the following fixed claim scope wording everywhere it appears: **"ahead-of-time variable-size batching for encoder-style inference."**
-
-## Publication Scope Sign-off
-- Sign-off date (UTC): 2026-03-26
-- Sign-off commit hash: bc5855c
-
-## 1. Introduction
-Requests cannot be processed instantaneously at arrival in any real serving system. Under load, queueing is inevitable. A first-come, first-served single-request policy is simple, but it fails to exploit throughput gains available through batching. Simply batching the first `n` requests in a queue is common, but it is often not throughput-optimal. Sorting requests by size and then batching can create high latency for large requests that may not be selected quickly under sustained load.
-
-This work addresses a common inference regime where workloads are dominated by batched matrix multiplication with variable input sizes (for example, variable token lengths in embedding or classification models). The scheduler pursues two goals simultaneously:
-
-1. **Maximum throughput**: Sort and choose batch partitions that minimize total completion time for queued work.
-2. **Latency objective**: choose near-term execution order with three production strategies:
-   - (FIFO) choose the batch with the oldest input
-   - (MINMAX) choose the batch with the highest prospective latency (waiting time + processing time)
-   - (GUARDED_BATCH_SIZE) apply a MINMAX guardrail, then prefer larger eligible batches for throughput
-
-In addition to runtime scheduling, we include startup benchmarking and estimator construction techniques that reduce calibration overhead while preserving scheduling quality. Concretely, the scheduler is parameterized by deployed-hardware benchmark data, so optimization is grounded in measured batch inference timings rather than assumed proxy costs.
-
-## 2. Related Work
-This paper builds on two strands of prior work: (1) classical single-machine scheduling theory for completion-time style objectives, and (2) practical production batching systems used in modern ML inference.
-
-### 2.1 Classical Scheduling Background
-- Smith's foundational sequencing rule for weighted completion-time minimization gives the interchange-argument template used throughout modern scheduling analysis [1].
-- For quadratic completion/lateness penalties, prior work studies single-machine objectives where squared terms increase the penalty for tail latency and variability [2], [3].
-
-Our historical RMS ordering experiment is in this family: it used pairwise interchange logic under a squared-latency objective, adapted to grouped/batched online inference decisions. In final evaluation, we retain it only as a negative result for transparency.
-
-### 2.2 Production ML Batching Systems
-- **NVIDIA Triton dynamic batching** provides queue-delay windows and preferred batch sizes for online serving [4].
-- **Infinity / batched integrations** expose practical dynamic batching controls for embedding and inference workloads [5], [6].
-- **Hugging Face pipeline batching** documents practical throughput-oriented batch inference usage [7].
-
-Unlike fixed-threshold accumulation approaches, Razor's Edge combines throughput-optimal partitioning (DP on sorted requests) with a second latency-aware selection pass from the final strategy set (`FIFO`, `MINMAX`, `GUARDED_BATCH_SIZE`).
-
-### 2.3 Positioning Relative to Existing Batching Practice
-Most production dynamic batchers expose preferred batch sizes, and simple first-ready policies [4]–[7]. These policies are robust and easy to deploy, but they typically do not solve an explicit global partitioning objective on the current queue state.
-
-The key distinction of Razor's Edge is its two-stage decision process:
-1. **Throughput-optimal contiguous partitioning** on size-sorted requests (DP over candidate cuts).
-2. **Latency-aware first-batch selection** among DP-consistent candidates.
-
-This makes Razor's Edge closer in spirit to objective-driven scheduling than to size-only batching heuristics, while remaining practical for online serving.
-
-### 2.4 Scope of Novelty Claim
-The contribution of this work is a systems-oriented synthesis with strictly bounded claims:
-
-Algorithmic Synthesis: A practical sorted-queue batching model combining DP-based partitioning with a latency-aware ordering pass.
-
-Deployment-Grounded Cost Modeling: The DP objective is instantiated with empirical timing values collected from the actual deployment target (model + runtime + hardware), rather than only synthetic or analytic estimates. This measurement-driven parameterization is treated as a first-class contribution of the framework.
-
-Workload Specialization: This framework is explicitly designed for padding-heavy, fixed-window inference (e.g., BERT-style embeddings or classification) where the cost of a batch is dominated by its longest member.
-
-Exclusion of Continuous Batching: This method is not intended for, nor applicable to, "continuous batching" or "iteration-level scheduling" (e.g., vLLM or TGI) used in causal LLM generation. It does not account for KV-cache management or mid-execution request insertion. The system uses "ahead-of-time variable-size batching for encoder-style inference" where a batch is fully formed before being sent to the engine.
-
-Empirical Gain: We validate that this synthesis yields operational gains in specific "static" batching environments common in embedding and encoder-only transformer deployments.
-
-What is **not** claimed:
-- no proof of global optimality for the multi-group online or request ordering pass.
-- no universal dominance over all dynamic batching policies, model families, or hardware environments.
-
-What is **validated in this paper**:
-- DP-based sorted contiguous partitioning and multiple first-batch heuristics (`FIFO`, `MINMAX`, `GUARDED_BATCH_SIZE`) can be integrated in an online executor with bounded per-decision overhead on the tested workload class.
-- estimator calibration plus the scheduler improves measured throughput relative to the tested FIFO/fixed-cap baselines in Section 7 under the reported setups.
-
-The empirical sections evaluate whether this synthesis yields operational gains in realistic serving settings under these scope conditions.
-
-
-## 3. Problem Setting and Model Assumptions
-
-Let 
-```math
+\begin{abstract}
+Serving systems for embedding, LLM, and other
+matrix-multiplication-dominated inference workloads rely on batching for
+efficient hardware utilization. We observe that batching efficiency
+exhibits a sharp input-size-dependent structure driven by the transition
+between memory-bound and compute-bound regimes: small inputs can be
+batched flexibly across heterogeneous sizes, while large inputs require
+near-uniformity, leading to a rapid collapse in batching efficiency.
+This produces a characteristic blade-like ("razor\textquotesingle s
+edge") shape in the batch performance landscape.
+
+We present the Razor\textquotesingle s Edge batching scheduler, a
+practical framework that combines (i) dynamic-programming-based
+throughput optimization over sorted requests, (ii) production-oriented
+next-batch selection strategies (\texttt{FIFO}, \texttt{MINMAX}, and
+\texttt{GUARDED\_BATCH\_SIZE}), and (iii) startup-time-efficient model
+benchmarking that builds batch timing estimators from direct
+measurements on the same hardware where the model is deployed. A central
+novelty claim in this paper is this measurement-to-optimizer bridge:
+instead of relying on analytic proxy cost models, we benchmark the
+deployed model/hardware pair and feed those empirical timings directly
+into the DP cost table used for scheduling decisions. We also introduce
+a practical visualization method for quantifying batching efficiency
+improvements when expanding the allowed maximum batch size from (N-1) to
+(N), producing the characteristic "razor\textquotesingle s edge" contour
+plots. The approach is designed for real-time online serving with
+queueing. Our claims are scoped to "ahead-of-time variable-size batching
+for encoder-style inference" evaluated in this paper, not to universal
+superiority across all serving stacks. We demonstrate the
+scheduler\textquotesingle s efficacy through a 47\% throughput increase
+on a CPU embedding workload (\texttt{jina-embeddings-v2-base-en}), a
+26\% throughput increase on a GPU embedding workload
+(\texttt{BAAI/bge-m3}), and controllable latency/throughput trade-offs
+across the final strategy set.
+
+\end{abstract}
+
+\section{Claim Language Freeze}\label{claim-language-freeze}
+
+
+To avoid claim drift, this manuscript uses the following fixed claim
+scope wording everywhere it appears: \textbf{"ahead-of-time
+variable-size batching for encoder-style inference."
+
+\section{Publication Scope
+Sign-off}\label{publication-scope-sign-off}
+
+\begin{itemize}
+\tightlist
+\item
+  Sign-off date (UTC): 2026-03-26
+\item
+  Sign-off commit hash: bc5855c
+\end{itemize
+
+\section{Introduction}\label{1-introduction}
+
+
+Requests cannot be processed instantaneously at arrival in any real
+serving system. Under load, queueing is inevitable. A first-come,
+first-served single-request policy is simple, but it fails to exploit
+throughput gains available through batching. Simply batching the first
+\texttt{n} requests in a queue is common, but it is often not
+throughput-optimal. Sorting requests by size and then batching can
+create high latency for large requests that may not be selected quickly
+under sustained load.
+
+This work addresses a common inference regime where workloads are
+dominated by batched matrix multiplication with variable input sizes
+(for example, variable token lengths in embedding or classification
+models). The scheduler pursues two goals simultaneously:
+
+\begin{enumerate}
+\def\labelenumi{\arabic{enumi}.}
+\tightlist
+\item
+  \textbf{Maximum throughput}: Sort and choose batch partitions that
+  minimize total completion time for queued work.
+\item
+  \textbf{Latency objective}: choose near-term execution order with
+  three production strategies:
+
+  \begin{itemize}
+  \tightlist
+  \item
+    (FIFO) choose the batch with the oldest input
+  \item
+    (MINMAX) choose the batch with the highest prospective latency
+    (waiting time + processing time)
+  \item
+    (GUARDED\_BATCH\_SIZE) apply a MINMAX guardrail, then prefer larger
+    eligible batches for throughput
+  \end{itemize}
+\end{enumerate
+
+In addition to runtime scheduling, we include startup benchmarking and
+estimator construction techniques that reduce calibration overhead while
+preserving scheduling quality. Concretely, the scheduler is
+parameterized by deployed-hardware benchmark data, so optimization is
+grounded in measured batch inference timings rather than assumed proxy
+costs.
+
+\section{Related Work}\label{2-related-work}
+
+
+This paper builds on two strands of prior work: (1) classical
+single-machine scheduling theory for completion-time style objectives,
+and (2) practical production batching systems used in modern ML
+inference.
+
+\subsection{Classical Scheduling
+Background}\label{21-classical-scheduling-background}
+
+\begin{itemize}
+\tightlist
+\item
+  Smith\textquotesingle s foundational sequencing rule for weighted
+  completion-time minimization gives the interchange-argument template
+  used throughout modern scheduling analysis {[}1{]}.
+\item
+  For quadratic completion/lateness penalties, prior work studies
+  single-machine objectives where squared terms increase the penalty for
+  tail latency and variability {[}2{]}, {[}3{]}.
+\end{itemize
+
+Our historical RMS ordering experiment is in this family: it used
+pairwise interchange logic under a squared-latency objective, adapted to
```
