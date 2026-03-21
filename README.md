# Razors-Edge-batching-scheduler
A scheduler to maximize throughput and minimize RMSE latency for ML requests.

## What this is
This project experiments with a smarter way to group requests into batches so you can:

- get **more throughput**
- keep **latency lower**
- handle **different input sizes** better than simple batching

It focuses on workloads like embeddings / classification where batched compute is much faster than one-by-one processing.

## How it works

When batching inputs for AI, there is usually padding. This padding creates inefficiency. Therefore to maximize throughput, inputs with very different sizes should not be batched.
This repo describes a scheduler which takes this detail into account.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run demos from `demos/` (synthetic and real benchmark notebooks/scripts are included).

## Project layout

- `src/` → scheduler + task logic
- `tests/` → test coverage
- `demos/` → experiments and benchmark notebooks
- `images/` → generated benchmark plots
- `PAPER.md` → full deep-dive explanation

## Result Images

Benchmark plots are in `images/`.

- Synthetic throughput comparisons
- Gains from variable batch sizing
- Real workload benchmarks

## To run tests

Simply use 
```bash
python -m coverage run --source=src -m unittest discover -v
coverage html
```

## Recommended Background Music
When using these methods, it is recommended that you listen to this for better code.

[Razor's Edge (Official Nightcore)](https://www.youtube.com/watch?v=UkqYk8INnq8)
