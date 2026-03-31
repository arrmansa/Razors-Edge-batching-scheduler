# Experiment traces

This directory contains canonical synthetic trace generators and a strict trace loader used by matrix runs.

## Included trace types

- `bursty_arrivals.json`: request arrivals with burst periods.
- `heavy_tail_sequences.json`: long-tail sequence length distribution.
- `mixed_tenant_traffic.json`: mixed tenant and traffic class composition.

Generate or refresh traces:

```bash
python -m experiments_v2_incomplete.traces.generate_traces
```
