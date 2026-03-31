# Continuous-Batching Comparator (vLLM / TGI)

This directory hosts the comparator path for workloads where continuous batching is applicable (token-generation / decoder-style serving).

## Scope

Use this comparator for:
- `demos/real/*` GPU text-generation style workloads
- synthetic decode workloads that mimic autoregressive generation

Do not use this comparator for pure non-generative kernels where continuous batching is not meaningful.

## Engines

- **vLLM**: OpenAI-compatible server path
- **TGI**: Hugging Face Text Generation Inference path

## Required parity knobs

Both engines must be configured with:
- identical model + revision
- identical tokenizer + revision
- identical prompt/completion lengths
- identical request arrival trace
- identical warmup + metric windows

## Structured outputs

The baseline suite runner writes:
- `results/continuous/vllm_metrics.json`
- `results/continuous/tgi_metrics.json`
- `results/continuous/parity_check.json`

## Comparator command template

Use `continuous_comparator.sh` to stamp a run and output stub metrics in the expected schema.
