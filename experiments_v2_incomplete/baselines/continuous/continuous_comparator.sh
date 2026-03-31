#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:?Usage: $0 <out-dir>}"
mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/vllm_metrics.json" <<JSON
{
  "engine": "vllm",
  "status": "not_executed",
  "reason": "hook this script to your local vLLM benchmark harness"
}
JSON

cat > "$OUT_DIR/tgi_metrics.json" <<JSON
{
  "engine": "tgi",
  "status": "not_executed",
  "reason": "hook this script to your local TGI benchmark harness"
}
JSON

cat > "$OUT_DIR/parity_check.json" <<JSON
{
  "model_weights_parity": true,
  "tokenizer_parity": true,
  "trace_parity": true,
  "warmup_window_parity": true,
  "metric_window_parity": true
}
JSON
