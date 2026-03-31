#!/usr/bin/env python3
"""Run matrix experiment cells with deterministic repetitions and traceable metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml

from experiments_v2_incomplete.traces.loader import load_trace


class TransientInfraError(RuntimeError):
    """Failure caused by infrastructure conditions that may resolve on retry."""


class ModelOrSchedulerError(RuntimeError):
    """Failure caused by model logic or scheduler behavior."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_backoff_seconds: float = 1.0


def _run_command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def collect_runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
    }

    try:
        metadata["git_hash"] = _run_command(["git", "rev-parse", "HEAD"])
    except Exception as exc:  # pragma: no cover - best-effort metadata
        metadata["git_hash_error"] = str(exc)

    try:
        metadata["nvidia_driver_version"] = _run_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()
        metadata["nvidia_devices"] = _run_command(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).splitlines()
    except Exception as exc:  # pragma: no cover - optional in CPU-only envs
        metadata["nvidia_info_error"] = str(exc)

    try:
        metadata["cuda_runtime_version"] = _run_command(["nvcc", "--version"]).splitlines()[-1]
    except Exception as exc:  # pragma: no cover - optional in CPU-only envs
        metadata["cuda_runtime_error"] = str(exc)

    return metadata


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)

    repetitions = int(matrix.get("repetitions", 0))
    seeds = matrix.get("seeds", [])
    if repetitions < 10:
        raise ValueError("Matrix must enforce at least 10 repetitions")
    if len(seeds) < repetitions:
        raise ValueError("Matrix must provide at least one fixed seed per repetition")

    return matrix


def execute_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Placeholder execution hook.

    Replace this function body with actual benchmark invocation to run the model/scheduler.
    """
    _ = load_trace(cell["trace_path"])
    return {"status": "not_executed", "reason": "replace execute_cell with benchmark invocation"}


def run_with_retries(
    cell: dict[str, Any],
    retry_policy: RetryPolicy,
    infra_log: Path,
    model_scheduler_log: Path,
) -> dict[str, Any]:
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            result = execute_cell(cell)
            result["attempt"] = attempt
            return result
        except TransientInfraError as exc:
            record = {
                "attempt": attempt,
                "run_id": cell["run_id"],
                "error": str(exc),
                "type": "transient_infra",
            }
            with infra_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if attempt >= retry_policy.max_attempts:
                return {"status": "failed", "failure_type": "transient_infra", "attempt": attempt}
            time.sleep(retry_policy.base_backoff_seconds * attempt)
        except ModelOrSchedulerError as exc:
            record = {
                "attempt": attempt,
                "run_id": cell["run_id"],
                "error": str(exc),
                "type": "model_or_scheduler",
            }
            with model_scheduler_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            return {"status": "failed", "failure_type": "model_or_scheduler", "attempt": attempt}

    return {"status": "failed", "failure_type": "unknown"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    matrix_path = root / "experiments" / "matrix.yaml"
    artifacts_root = root / "artifacts" / "raw_runs"
    suite_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_dir = artifacts_root / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_matrix(matrix_path)
    runtime_metadata = collect_runtime_metadata()

    infra_log = suite_dir / "infra_failures.jsonl"
    model_scheduler_log = suite_dir / "model_scheduler_failures.jsonl"
    retry_policy = RetryPolicy()

    results: list[dict[str, Any]] = []
    repetitions = int(matrix["repetitions"])

    cells = itertools.product(matrix["hardware"], matrix["models"], matrix["traces"])
    for hardware, model, trace in cells:
        for rep in range(repetitions):
            run_id = f"{suite_id}-{hardware['id']}-{model['id']}-{trace['id']}-r{rep + 1:02d}"
            run_dir = artifacts_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            cell = {
                "run_id": run_id,
                "suite_id": suite_id,
                "hardware": hardware,
                "model": model,
                "trace_id": trace["id"],
                "trace_path": str(root / trace["path"]),
                "repetition": rep + 1,
                "seed": matrix["seeds"][rep],
            }

            with (run_dir / "meta.json").open("w", encoding="utf-8") as handle:
                json.dump({**cell, "runtime": runtime_metadata}, handle, indent=2)

            result = run_with_retries(cell, retry_policy, infra_log, model_scheduler_log)
            results.append({**cell, **result})

    with (suite_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
