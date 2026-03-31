"""Load and validate experiment traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"timestamp_ms", "seq_len", "tenant", "traffic_class"}


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"Invalid trace format at {trace_path}: 'events' must be a list")

    prev_ts = -1
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"Invalid event at index {idx}: expected object")

        missing = REQUIRED_FIELDS - set(event.keys())
        if missing:
            missing_csv = ", ".join(sorted(missing))
            raise ValueError(f"Invalid event at index {idx}: missing fields [{missing_csv}]")

        if event["timestamp_ms"] < prev_ts:
            raise ValueError("Trace timestamps must be monotonic non-decreasing")
        prev_ts = event["timestamp_ms"]

    return events
