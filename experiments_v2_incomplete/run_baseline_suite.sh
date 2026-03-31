#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/artifacts/baselines/$TIMESTAMP"
RESULTS_DIR="$RUN_DIR/results"
METADATA_DIR="$RUN_DIR/metadata"
SNAPSHOTS_DIR="$RUN_DIR/snapshots"

mkdir -p "$RESULTS_DIR/triton" "$RESULTS_DIR/continuous" "$METADATA_DIR" "$SNAPSHOTS_DIR"

# Snapshot baseline and tuning config for auditability.
cp "$ROOT_DIR/experiments_v2_incomplete/baselines/README.md" "$SNAPSHOTS_DIR/parity_rules.md"
cp "$ROOT_DIR/deploy/triton/model_repository/synthetic_dummy_model/config.pbtxt" \
  "$SNAPSHOTS_DIR/synthetic_dummy_model.config.pbtxt"
cp "$ROOT_DIR/deploy/triton/model_repository/real_gpu_benchmark_model/config.pbtxt" \
  "$SNAPSHOTS_DIR/real_gpu_benchmark_model.config.pbtxt"
cp "$ROOT_DIR/experiments_v2_incomplete/baselines/continuous/README.md" \
  "$SNAPSHOTS_DIR/continuous_comparator_readme.md"

# Record immutable metadata for run context.
cat > "$METADATA_DIR/run_metadata.json" <<JSON
{
  "timestamp_utc": "$TIMESTAMP",
  "suite": "baseline_comparison",
  "artifact_root": "artifacts/baselines/$TIMESTAMP",
  "parity_rules": "experiments_v2_incomplete/baselines/README.md"
}
JSON

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$SNAPSHOTS_DIR"/* > "$METADATA_DIR/snapshot_sha256.txt"
fi

# Triton path stub (replace with real benchmark invocation).
cat > "$RESULTS_DIR/triton/dynamic_batching_metrics.json" <<JSON
{
  "engine": "triton",
  "status": "not_executed",
  "reason": "replace stub with benchmark runner invocation",
  "models": [
    "synthetic_dummy_model",
    "real_gpu_benchmark_model"
  ]
}
JSON

# Continuous batching comparator path for applicable workloads.
"$ROOT_DIR/experiments_v2_incomplete/baselines/continuous/continuous_comparator.sh" "$RESULTS_DIR/continuous"

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "run_dir": "artifacts/baselines/$TIMESTAMP",
  "results_dir": "artifacts/baselines/$TIMESTAMP/results",
  "metadata_dir": "artifacts/baselines/$TIMESTAMP/metadata",
  "snapshots_dir": "artifacts/baselines/$TIMESTAMP/snapshots"
}
JSON

echo "Baseline suite artifacts written to: $RUN_DIR"
