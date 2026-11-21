# LLM Inference Optimisation Lab

A benchmark harness, and the measurements made with it, showing how serving choices change the throughput,
latency, cost and energy of LLM inference. The choices covered are batch size, precision, serving engine, prefix
caching and multi-GPU layout. Measurements were made on one NVIDIA H100 SXM (vLLM, SGLang and TensorRT-LLM
serving Qwen3-8B) and on NVIDIA T4s.

Every number here traces back to a record in `results/runs.jsonl` and the raw per-request and power files
in `results/raw/`. Anything not measured is labelled "not measured".

## Headline results (1× H100 SXM, Qwen3-8B)

| Setup | Throughput | Cost per 1M output tokens | Energy per token |
|---|---:|---:|---:|
| vLLM, BF16, 1 request at a time | 146 tok/s | USD 8.36 | 3.12 J |
| vLLM, BF16, 256 concurrent requests | 12,398 tok/s | USD 0.098 | 0.054 J |
| vLLM, FP8, 256 concurrent requests | 16,432 tok/s | USD 0.074 | 0.0395 J |

On the same GPU, batching and FP8 together cut the cost per token **113×** (85× from batching, 1.33× from FP8).
Cost assumes USD 4.39/h (RunPod Secure Cloud) with the GPU busy for the whole hour.

## Findings

| Finding | Evidence |
|---|---|
| [Batching is the biggest lever](docs/findings.md#f1): 74× throughput and 47× less energy per token from 1 to 256 requests, while power rose only 1.6× | H100, T4 |
| [FP8 adds 33–53% on H100](docs/findings.md#f2) at every load; AWQ-INT4 on T4 helps a single user (~3.5×) but not a full GPU (+2%) | H100, T4 |
| [vLLM and SGLang are within about 0.5%](docs/findings.md#f3) at 256 requests; precision and batch size matter far more than the engine | H100 |
| [TensorRT-LLM 1.2.1 halves at 256 requests](docs/findings.md#f4) with default settings: CUDA graphs are captured only up to batch 128, and the GPU is busy 43% of the time | H100 |
| [Prefix caching doubles throughput](docs/findings.md#f5) with a shared 1,024-token prompt: TTFT p50 152 s → 9.4 s, energy per token halved | T4 |
| [KV-cache becomes the limit only with long prompts](docs/findings.md#f6): 44% full with short chat prompts at 256 requests, 94–98% with long ones | T4 |
| [Tensor parallel beats replicas until the GPU is compute-bound](docs/findings.md#f7): 2.3× faster for one user, 17% slower at 256 requests | 2× T4 |
| [The H100 is 12–29× faster than a T4 but only 3× more energy-efficient](docs/findings.md#f8); both run at their power cap under full load | T4, H100 |

## What is measured

- **Workload:** ShareGPT prompts, exactly 256 output tokens per request (`temperature=0`, `ignore_eos`), sent at a
  fixed number of concurrent requests (1 to 256). Warm-up requests are excluded.
- **Throughput:** output tokens per second over the whole run.
- **Latency:** time to first token (TTFT, p50 and p99) and time per output token (ITL, or TPOT on H100).
- **Energy:** GPU power from NVML sampled every 100 ms, integrated over the run and divided by output tokens.
  It is cross-checked against NVML's own energy counter (median difference −0.8% over 90 H100 runs).
- **Hardware state:** GPU utilisation, SM clock, temperature and throttle reasons with every power sample.
- **Repeatability:** 3 repeats per level with fresh prompts. A level is rerun if the repeats differ by more
  than 5% (coefficient of variation).

Details and the reasons behind each choice are in [docs/methodology.md](docs/methodology.md).

## Environment

| | H100 runs | T4 runs |
|---|---|---|
| GPU | 1× H100 SXM 80 GB (RunPod) | 1× or 2× Tesla T4 16 GB, PCIe (Kaggle) |
| Driver / CUDA | 595.112.07 / 13.0 | 580.178.04 / 13.0 |
| Engines | vLLM 0.33.0, SGLang 0.5.23, TensorRT-LLM 1.2.1 | vLLM 0.33.0 |
| Models | `Qwen/Qwen3-8B` (BF16), `nvidia/Qwen3-8B-FP8`, `Qwen/Qwen2.5-3B-Instruct` (FP16) | `Qwen/Qwen2.5-3B-Instruct` (FP16 and AWQ-INT4) |

Model revisions and exact commands are in [docs/recipes.md](docs/recipes.md).

## Reproduce

```bash
make setup                                              # on a GPU box (Kaggle T4 or RunPod)
make env                                                # GPU, driver, CUDA, runtime versions
make test                                               # unit tests, no GPU needed
make smoke EXP=configs/experiments/t4_qwen3b_fp16.yaml  # plumbing check
make bench EXP=configs/experiments/t4_qwen3b_fp16.yaml  # full sweep → results/runs.jsonl
```

On Kaggle, `notebooks/kaggle_driver.ipynb` runs the same make targets.

## Repository layout

- `src/infer_lab/`: server launcher, load generator, metrics, NVML power sampler, run schema
- `configs/experiments/`: one YAML file per experiment
- `results/runs.jsonl`: one record per measured run; `results/raw/` holds per-request and power CSVs
- `docs/findings.md`: what the measurements showed, with evidence
- `docs/recipes.md`: which configuration to run for which goal, with commands and expected numbers
- `docs/methodology.md`: how and why things are measured the way they are
- `tests/`: unit tests for the load generator, metrics and power integration

## Limitations

- One H100 in one session; multi-GPU results are on T4s over PCIe only.
- Output quality was not measured for FP8 or INT4. Check accuracy before relying on either.
- The TensorRT-LLM CUDA-graph fix is prepared but not yet measured.
- Energy covers the GPU board only, not the host, power supply losses or cooling.
- A single-process load generator limits latency resolution above about 20,000 tok/s.
