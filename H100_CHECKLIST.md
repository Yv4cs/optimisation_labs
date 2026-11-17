# TN4 checklist: 1× H100 on RunPod

Goal: vLLM vs SGLang vs TensorRT-LLM, BF16 vs FP8, same model, same workload, same harness.
Budget: about 3 GPU-hours, **hard cap USD 15**. Nothing is rented until the approval box at the bottom is ticked.

## 1. Decisions to confirm before renting

| Decision | Proposal | Why |
|---|---|---|
| Model | `Qwen/Qwen3-8B` (BF16) and the official `Qwen/Qwen3-8B-FP8` | 8B class as planned in D-002. An official FP8 checkpoint means all three engines load the *same* FP8 weights, so the comparison is about the engines, not about three different quantisation recipes. **Verify both repos exist and are ungated on Hugging Face.** Fallback: `Qwen/Qwen2.5-7B-Instruct` with on-the-fly FP8 in vLLM/SGLang only. |
| GPU | H100 **SXM** 80 GB if available, else PCIe | SXM has ~3.35 TB/s HBM3, PCIe ~2 TB/s: a 1.7× difference that would dominate decode speed. Record which one in `interconnect`. |
| Concurrency | `[1, 16, 64, 256]`, 3 repeats | Covers single-user latency, the knee and the plateau seen in TN1 while keeping each sweep around 10 minutes. |
| Workload | Same as TN1: ShareGPT, 256 output tokens, `temperature=0`, `ignore_eos` | Comparable with T4 numbers (D-003). |
| Order | vLLM → SGLang → TensorRT-LLM | Most-known first. TRT-LLM is the riskiest and goes last, time-boxed. |

## 2. Before renting (free, on laptop and Kaggle)

- [ ] Push all commits; `make test` green on the laptop.
- [ ] Add a `trtllm` runtime to `config.py` / `server.py` (launch with `trtllm-serve`, OpenAI-compatible).
      Check the flags against the TensorRT-LLM docs for the exact image tag chosen below. *(Claude, 18/11 morning)*
- [ ] Decide how `fp8` maps per engine for a pre-quantised checkpoint (likely: no extra flag, the engine reads
      the checkpoint config, as with AWQ in D-010). Update `_PRECISION_ARGS` and add a test.
- [ ] Write the 6 configs `h100_qwen3-8b_{vllm,sglang,trtllm}_{bf16,fp8}.yaml` with `pod_hourly_usd` left as a
      placeholder to fill on the pod.
- [ ] Smoke the SGLang path once on Kaggle T4 (FP16, Qwen2.5-3B) if SGLang installs there, so the first SGLang
      run is not on paid time.
- [ ] Pick image tags and write them here (verify each exists on the registry):
      vLLM `vllm/vllm-openai:v0.33.0` · SGLang `lmsysorg/sglang:<tag>` · TRT-LLM `nvcr.io/nvidia/tensorrt-llm/release:<tag>`.
- [ ] Create a RunPod **network volume** (~50 GB) for the Hugging Face cache and `results/`, so weights download
      once and results survive pod termination.
- [ ] Check the current H100 hourly price and write it down: ______ USD/h → 3 h ≈ ______ USD.

## 3. On each pod (one pod per engine image, same network volume)

```bash
export HF_HOME=/workspace/hf            # on the network volume
cd /workspace && git clone https://github.com/Yv4cs/optimisation_labs.git lab 2>/dev/null; cd lab && git pull
pip install -e .                         # light deps only; must not replace the image's torch
nvidia-smi && nvidia-smi topo -m         # record SXM/PCIe, driver
make env && make test
# fill pod_hourly_usd and interconnect in the two configs for this engine, then:
make smoke EXP=configs/experiments/h100_qwen3-8b_<engine>_bf16.yaml
make bench EXP=configs/experiments/h100_qwen3-8b_<engine>_bf16.yaml
make bench EXP=configs/experiments/h100_qwen3-8b_<engine>_fp8.yaml
pip freeze --exclude-editable > env/lock-h100-<engine>.txt
tar czf /workspace/results_<engine>.tar.gz results env
```

Then **stop and terminate the pod** before starting the next engine.

## 4. Time plan and stop rules

| Block | Budget | Stop rule |
|---|---|---|
| vLLM (setup, smoke, BF16, FP8) | 45 min | |
| SGLang | 45 min | If the server is not healthy after 20 min of debugging, skip SGLang FP8 and record why. |
| TensorRT-LLM | 60 min | **Time box: 30 min to a healthy server.** Otherwise stop, save the logs, and write the attempt up honestly in STORIES. |
| Buffer | 30 min | |

- Check the spend at the end of every block. At USD 12 spent, finish the current sweep and stop.
- Never leave a pod running unattended. If a sweep runs long, terminate rather than wait idle.
- Optional, only with time left: GSM8K accuracy check, BF16 vs FP8 (lm-evaluation-harness against the running
  vLLM server, `--limit 250`), so "FP8 is faster" comes with "and loses X points" or "loses nothing measurable".

## 5. What to record (feeds STORIES and the README table)

- Exact image tags, `nvidia-smi` output, SXM vs PCIe, hourly price.
- Each engine's resolved config from its server log (attention backend, CUDA graphs, chunked prefill: see S-002).
- Clock and throttle columns in `power.csv` (H100 at 700 W behaves very differently from a 70 W T4: S-001).
- Anything that failed, with the log tail.

## Approval

- [ ] Model choice confirmed
- [ ] Price checked, cap USD 15 accepted
- [ ] Prerequisites in section 2 done

Approved by: __________ Date: __________
