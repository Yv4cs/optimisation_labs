# LLM Inference Optimisation Lab

Reproducible benchmarks of LLM serving: vLLM, SGLang and TensorRT-LLM; FP16/BF16 vs FP8 vs INT4;
tensor-parallel vs replicated layouts. Reports TTFT, inter-token latency, throughput and **energy per
token**, measured with NVML.

Every number in this repo traces back to a line in `results/runs.jsonl`. Anything not yet measured is
shown as "not measured".

## Status

| Experiment | GPU | Result |
|---|---|---|
| vLLM FP16 baseline, concurrency 1→256 | 1× T4 | not measured |
| FP16 vs AWQ-INT4, prefix caching | 1× T4 | not measured |
| TP=2 vs 2 replicas | 2× T4 (PCIe) | not measured |
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
