"""Trace generation utilities for matrix experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable


@dataclass(frozen=True)
class TraceEvent:
    timestamp_ms: int
    seq_len: int
    tenant: str
    traffic_class: str


def _write_trace(path: Path, events: Iterable[TraceEvent]) -> None:
    payload = {"events": [asdict(event) for event in events]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def generate_bursty_arrivals(
    path: Path,
    *,
    seed: int,
    n_events: int = 1200,
    base_interval_ms: int = 9,
    burst_probability: float = 0.12,
) -> None:
    """Generate arrivals with occasional dense bursts."""
    rng = random.Random(seed)
    current_t = 0
    events: list[TraceEvent] = []

    for _ in range(n_events):
        if rng.random() < burst_probability:
            current_t += rng.randint(0, max(1, base_interval_ms // 3))
        else:
            current_t += rng.randint(base_interval_ms // 2, base_interval_ms * 2)

        events.append(
            TraceEvent(
                timestamp_ms=current_t,
                seq_len=rng.randint(32, 384),
                tenant="tenant_a",
                traffic_class="interactive",
            )
        )

    _write_trace(path, events)


def generate_heavy_tail_sequence_lengths(
    path: Path,
    *,
    seed: int,
    n_events: int = 1400,
    inter_arrival_ms: int = 7,
) -> None:
    """Generate requests with a long-tail length distribution."""
    rng = random.Random(seed)
    current_t = 0
    events: list[TraceEvent] = []

    for _ in range(n_events):
        current_t += rng.randint(1, inter_arrival_ms * 2)
        # Log-normal gives frequent short sequences and rare long outliers.
        seq_len = int(min(4096, max(16, rng.lognormvariate(4.4, 1.0))))
        events.append(
            TraceEvent(
                timestamp_ms=current_t,
                seq_len=seq_len,
                tenant="tenant_b",
                traffic_class="analytics",
            )
        )

    _write_trace(path, events)


def generate_mixed_tenant_traffic_classes(
    path: Path,
    *,
    seed: int,
    n_events: int = 1600,
) -> None:
    """Generate arrivals from mixed tenants with class-dependent sequence lengths."""
    rng = random.Random(seed)
    current_t = 0

    tenants = ["gold", "silver", "bronze"]
    traffic_classes = {
        "interactive": (16, 192),
        "batch": (192, 1536),
        "rerank": (64, 768),
    }

    events: list[TraceEvent] = []
    for _ in range(n_events):
        tenant = rng.choices(tenants, weights=[0.25, 0.35, 0.40], k=1)[0]
        traffic_class = rng.choices(
            list(traffic_classes.keys()),
            weights=[0.5, 0.3, 0.2],
            k=1,
        )[0]
        low, high = traffic_classes[traffic_class]
        current_t += rng.randint(1, 16)
        events.append(
            TraceEvent(
                timestamp_ms=current_t,
                seq_len=rng.randint(low, high),
                tenant=tenant,
                traffic_class=traffic_class,
            )
        )

    _write_trace(path, events)
