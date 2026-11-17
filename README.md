# LLM Inference Optimisation Lab

Reproducible benchmarks of LLM serving: vLLM, SGLang and TensorRT-LLM; FP16/BF16 vs FP8 vs INT4;
tensor-parallel vs replicated layouts. Reports TTFT, inter-token latency, throughput and **energy per
token**, measured with NVML.

Every number in this repo traces back to a line in `results/runs.jsonl`. Anything not yet measured is
shown as "not measured".

## Status

| Experiment | GPU | Result |
|---|---|---|
| vLLM FP16 baseline, concurrency 1→256 | 1× T4 | 18.9 → 792 tok/s; 3.46 → 0.082 J/token (STORIES S-006) |
| FP16 vs AWQ-INT4 | 1× T4 | c=1: 18.9 → 74.2 tok/s, 3.46 → 0.89 J/token; c=256: +2% (S-011) |
| Prefix caching off vs on, 1,024-token shared prompt | 1× T4 | off: KV 98% full, TTFT p50 152 s at c=256 (S-014); on: pending |
| TP=2 vs 2 replicas | 2× T4 (PCIe) | TP wins c≤64 (c=1: 61.0 vs 27.0 tok/s); DP wins c=256 (1,376 vs 1,179 tok/s) (S-012) |
| vLLM vs SGLang vs TensorRT-LLM, BF16 vs FP8 | 1× H100 | not measured |

## Quickstart

```bash
make setup                                              # on a GPU box (Kaggle T4 or RunPod)
make env                                                # GPU, driver, CUDA, runtime versions
make test                                               # unit tests, no GPU needed
make smoke EXP=configs/experiments/t4_qwen3b_fp16.yaml  # plumbing check
make bench EXP=configs/experiments/t4_qwen3b_fp16.yaml  # full sweep → results/runs.jsonl
```

On Kaggle, `notebooks/kaggle_driver.ipynb` runs the same make targets.

## Layout

- `src/infer_lab/`: server launcher, load generator, metrics, NVML power sampler, run schema
- `configs/experiments/`: one YAML file per experiment
- `results/runs.jsonl`: one validated record per measured run; `results/raw/` holds per-request and power CSVs
- `recipes/`: best configs, packaged to reproduce
- `DECISIONS.md`: why things are the way they are
- `STORIES.md`: findings worth telling (surprise, evidence, reasoning, outcome)
