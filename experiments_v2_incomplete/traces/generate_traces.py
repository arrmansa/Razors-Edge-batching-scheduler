"""Generate canonical trace files consumed by the experiment matrix."""

from __future__ import annotations

from pathlib import Path

from experiments_v2_incomplete.traces.generators import (
    generate_bursty_arrivals,
    generate_heavy_tail_sequence_lengths,
    generate_mixed_tenant_traffic_classes,
)


def main() -> None:
    root = Path(__file__).resolve().parent

    generate_bursty_arrivals(root / "bursty_arrivals.json", seed=7)
    generate_heavy_tail_sequence_lengths(root / "heavy_tail_sequences.json", seed=11)
    generate_mixed_tenant_traffic_classes(root / "mixed_tenant_traffic.json", seed=13)


if __name__ == "__main__":
    main()
