# TN4 checklist: 1× H100 on RunPod

Goal: compare vLLM, SGLang and TensorRT-LLM in BF16 and FP8 on the same model, workload and harness. Also run one bridge sweep that repeats TN1 on H100.
Budget: about 3 GPU-hours, with a **hard cap of USD 15**. Nothing is rented until the approval box at the bottom is ticked.

## 1. Decisions (verified 2026-11-18)

| Decision | Choice | Why / evidence |
|---|---|---|
| BF16 model | `Qwen/Qwen3-8B` @ `b968826d…` | 8B class (D-002). Ungated on Hugging Face (`gated: false`). Listed in TRT-LLM 1.2.1 supported models. |
| FP8 model | `nvidia/Qwen3-8B-FP8` @ `2cebc4c8…` | Per-tensor ModelOpt FP8, which all three engines support. Qwen's own `Qwen3-8B-FP8` is block-wise FP8, and TRT-LLM 1.2.1 does not list block FP8 for Qwen-3 (D-011). Ungated. |
| Bridge to T4 | `Qwen/Qwen2.5-3B-Instruct` FP16, vLLM | Same model and workload as TN1, so "H100 vs T4" means something (about 10 min). |
| GPU | H100 **SXM** 80 GB, else PCIe | SXM has ~3.35 TB/s HBM3, PCIe ~2 TB/s. Write the one you get into `interconnect` in the configs. |
| Concurrency | `[1, 16, 64, 256]`, 3 repeats | Single user, the T4 knee, and the plateau. Keeps each sweep around 10 minutes. |
| Workload | Same as TN1: ShareGPT, 256 output tokens, `temperature=0`, `ignore_eos` | Comparable with the T4 numbers (D-003). All three engines accept `ignore_eos` and `stream_options.include_usage` (checked in their source). |
| Order | vLLM → SGLang → TensorRT-LLM | Most familiar first. TRT-LLM is the riskiest, so it goes last with a time box. |

**Images** (all exist on their registries and were released at least two weeks ago):

| Engine | Image | Released |
|---|---|---|
| vLLM 0.33.0 | `vllm/vllm-openai:v0.33.0` | 2026-11-02 |
| SGLang 0.5.23 | `lmsysorg/sglang:v0.5.23` | 2026-10-30 |
| TensorRT-LLM 1.2.1 (latest stable; 1.4 is still rc) | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` | 2026-04-20 |

**Price** (RunPod, checked 2026-11-18; recheck in the console): H100 SXM costs about **USD 2.69/h on Community Cloud** and **USD 3.99/h on Secure Cloud**.
Three hours therefore cost about **USD 8** on Community or **USD 12** on Secure.

## 2. Before renting (free)

- [x] Qwen3-8B, the FP8 checkpoint and the image tags verified (section 1).
- [x] `trtllm` runtime: `trtllm-serve serve MODEL --tp_size … --extra_llm_api_options <yaml>`. The YAML carries
      `kv_cache_config` (block reuse = prefix caching, FP8 KV). Flags were checked against the v1.2.1 source.
- [x] FP8 mapping: no `--quantization` flag, because the engines read `quant_method: modelopt`. FP8 KV-cache is set explicitly on every engine (D-011).
- [x] Configs: `h100_qwen3-8b_{vllm,sglang,trtllm}_{bf16,fp8}.yaml` and `h100_qwen3b_fp16_vllm.yaml`. Tests are green (44).
- [x] Push all commits.
- [ ] Optional: smoke the SGLang path on Kaggle T4 (`runtime: sglang` on the TN1 config). SGLang may not support
      sm_75 well. If it fails there, the first real SGLang run is on the H100, and that is acceptable.
- [ ] Choose Community or Secure Cloud.
  - RunPod network volumes are, as far as we know, **Secure Cloud only** (verify in the console).
  - Without a volume, each pod downloads its own weights (about 16 GB BF16 and 9 GB FP8, a few minutes) and the results must be copied out before termination (section 3).

## 3. One pod, one virtualenv per engine (decided 2026-11-19)

We use a single pod from RunPod's PyTorch template, which comes with a working terminal. Each engine gets its own `uv` virtualenv on `/workspace`.
Why not one pod per engine image: those images may lack the terminal/SSH setup, and every new pod means another image pull and another weights download.
Engines are installed one at a time, never while a benchmark runs, because a pip install competes for CPU with the engine's scheduler.

**Deploy:** Secure Cloud → **H100 SXM ×1** → template "RunPod PyTorch".
- Filter **CUDA version ≥ 13.0**: vLLM 0.33 wheels need it; Kaggle ran driver 580 / CUDA 13.0.
- Volume disk 120 GB at `/workspace`, container disk 30 GB.

```bash
# --- once per pod
nvidia-smi                                  # driver/CUDA must be >= 13.0; note SXM vs PCIe
apt-get update -qq && apt-get install -y -qq tmux libopenmpi-dev >/dev/null
pip install -q uv
export HF_HOME=/workspace/hf
cd /workspace && git clone https://github.com/Yv4cs/optimisation_labs.git lab && cd lab
sed -i 's/^pod_hourly_usd: 0.0.*/pod_hourly_usd: 3.99/' configs/experiments/h100_*.yaml   # use the console price
# if nvidia-smi says PCIe: sed -i 's/^interconnect: SXM.*/interconnect: PCIe/' configs/experiments/h100_*.yaml

# --- per engine (E = vllm | sglang | trtllm)
uv venv /workspace/venv-$E --python 3.12 && source /workspace/venv-$E/bin/activate
uv pip install "vllm==0.33.0" -e ".[dev]"            # sglang: "sglang==0.5.23"   trtllm: see below
make env && make test
tmux new -s $E                                       # benchmarks survive a closed browser tab
make smoke EXP=configs/experiments/h100_qwen3-8b_${E}_bf16.yaml
make bench EXP=configs/experiments/h100_qwen3-8b_${E}_bf16.yaml
make bench EXP=configs/experiments/h100_qwen3-8b_${E}_fp8.yaml
[ $E = vllm ] && make bench EXP=configs/experiments/h100_qwen3b_fp16_vllm.yaml
uv pip freeze --python /workspace/venv-$E/bin/python > env/lock-h100-$E.txt   # NOT pip freeze: uv venvs have no pip (S-005)
grep -iE "^(vllm|sglang|tensorrt.llm)==" env/lock-h100-$E.txt                   # must print the engine
tar czf /workspace/results_h100.tar.gz results env  # re-run after every engine; download it each time
```

TensorRT-LLM needs two extras, found on the pod:
```bash
uv pip install "tensorrt_llm==1.2.1" "pyarrow<20" -e ".[dev]" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match   # newest pyarrow needs NumPy 2; TRT-LLM pins 1.26
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1                                 # MPI refuses root by default
```

Download `results_h100.tar.gz` after every engine block, through the pod's Jupyter file browser or `runpodctl send`.
**Terminate the pod** after the last block, once the tarball is on the laptop.

## 4. Time plan and stop rules

| Block | Budget | Stop rule |
|---|---|---|
| vLLM: setup, smoke, BF16, FP8, 3B bridge | 55 min | |
| SGLang: setup, smoke, BF16, FP8 | 45 min | If the server is not healthy after 20 min of debugging, skip SGLang FP8 and record why. |
| TensorRT-LLM: setup, smoke, BF16, FP8 | 60 min | **Time box: 30 min to a healthy server.** Otherwise stop, save the logs, and write it up honestly in STORIES. |
| Buffer | 20 min | |

- Check the spend at the end of every block. At USD 12 spent, finish the current sweep and stop.
- Never leave a pod running unattended. If a sweep runs long, terminate rather than wait idle.
- Optional, only if time is left: a GSM8K accuracy check of BF16 vs FP8 (lm-evaluation-harness against the running vLLM server, `--limit 250`).

## 5. What to record (feeds STORIES, the README table and RECIPES R7+)

- Exact image tags, `nvidia-smi` output, SXM vs PCIe, hourly price.
- Each engine's resolved config from its server log: attention backend, CUDA graph batch sizes, chunked prefill, KV pool size and dtype (S-002, S-003).
- Clock and throttle columns in `power.csv`. An H100 at 700 W behaves very differently from a T4 capped at 70 W (S-001).
- Tokens per streamed chunk for TRT-LLM (D-008). If it sends several tokens per chunk, its ITL is per chunk.
- Anything that failed, with the log tail.

## Approval

- [x] Model choice confirmed (Qwen3-8B BF16 + `nvidia/Qwen3-8B-FP8`, plus the 3B bridge)
- [x] Secure Cloud chosen (≈ USD 3.99/h → ≈ USD 12 for 3 h); cap USD 15 accepted
- [x] Prerequisites in section 2 done (SGLang T4 smoke skipped)

Approved by: Charles Tran · Date: 2026-11-19

## Outcome (2026-11-18)

- H100 80GB HBM3 (SXM), driver 595.112.07, CUDA 13.2, 700 W limit. Secure Cloud, **USD 4.39/h compute** (USD 4.41/h with storage, from the console; the 3.99 planning price was wrong).
- All 7 planned sweeps completed on the first try, plus one TRT-LLM diagnostic (`…_trtllm_fp8_pp8.yaml`). TRT-LLM came
  up well inside its 30-minute time box. The CUDA-graph test for S-018 was prepared and skipped to save budget.
- Time on the pod: about 16:45–19:15 UTC. Spend: ≈ USD 11 in total, plus volume storage while the pod was stopped.
- Problems hit: TRT-LLM import failed on pyarrow/NumPy versions (fixed with `pyarrow<20`); per-engine lock files
  captured the system env because `pip` was not the venv's (files dropped, versions recovered from `env.json`); HF
  weights download into `hub/blobs`, so `du` on a model folder shows only a few MB.
