# Baseline Parity Rules

Use these rules when comparing Razor's Edge batching against external baselines (Triton dynamic batching, vLLM, and TGI).

## Required parity dimensions

1. **Same model weights**
   - Use the exact same model revision/checkpoint for all compared systems.
   - Pin model IDs and revisions in config snapshots.

2. **Same tokenizer**
   - Use the same tokenizer files and tokenizer settings (`padding_side`, `truncation`, max sequence lengths).
   - Record tokenizer source and revision in each run artifact.

3. **Same request trace**
   - All systems must replay the same arrival trace and request payload set.
   - Keep the trace immutable and include its digest (`sha256`) in artifacts.

4. **Same warmup window**
   - Exclude warmup from reported metrics using the same warmup duration/count.
   - Warmup start/end timestamps must be captured in run metadata.

5. **Same metric windows**
   - Throughput, latency percentiles, and utilization must use identical measurement windows.
   - Report window boundaries explicitly in structured output.

## Enforcement checklist

- [ ] Model ID + revision match across all compared engines.
- [ ] Tokenizer path + revision match.
- [ ] Trace path + digest match.
- [ ] Warmup window config match.
- [ ] Metric window config match.

## Output contract

Every baseline suite run should write structured artifacts under:

`artifacts/baselines/<timestamp>/`

Each run directory must include:

- `results/` for per-engine measurements
- `metadata/` for run context and checksums
- `snapshots/` for copied configs that explain tuning choices
