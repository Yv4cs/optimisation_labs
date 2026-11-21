# Methodology

How the benchmarks are run and why. Each entry records a choice and the reason for it. Results are in
[findings.md](findings.md).

## D-001 Environment pinning without Docker on Kaggle
Kaggle notebooks cannot run Docker. On Kaggle we install from `requirements-kaggle.txt`, verify the
server works on T4, then `make freeze` writes `env/lock-kaggle.txt`. That lock file is the source of truth.
Docker is used only on RunPod (H100), where the image tag pins everything.

## D-002 Models
T4 (16 GB, no BF16, no FP8): `Qwen/Qwen2.5-3B-Instruct` in FP16 plus the official `-AWQ` checkpoint.
These checkpoints are ungated (no HF token needed), and 3B FP16 weights leave room for KV-cache on one T4.
H100: `Qwen/Qwen3-8B` (BF16) and `nvidia/Qwen3-8B-FP8` (D-011), both ungated (checked 2026-11-18), plus
`Qwen/Qwen2.5-3B-Instruct` FP16 once on vLLM as the bridge to the T4 numbers (same model and workload).

## D-003 Controlled workload
Requests use `/v1/completions` with raw text (no chat template), `temperature=0`, `ignore_eos=true` and
a fixed `max_tokens`. Every runtime and precision therefore generates exactly the same number of tokens,
so throughput differences come from the system and not from answer length. ShareGPT supplies realistic
prompt lengths. Fixed-length synthetic prompts cover the long-context case.

## D-004 One runs.jsonl line per repeat
Each repeat is its own record, tagged with `experiment` and `repeat_idx`, and mean ± std is computed at
report time. If the repeats' tokens/s CV exceeds `max_cv` (5%), the level is re-run once. The rejected
set stays in the log with `notes: discarded`, so nothing measured is silently dropped.
Fields added to the original schema: `experiment`, `repeat_idx`, `model_revision`.

## D-005 Fresh prompts for every warm-up and repeat
Re-sending the same prompts would let prefix caching serve later repeats from cache. Long-context prompts
start with a unique header for the same reason. Prefix caching gets its own experiment with a
deliberately shared prefix.

## D-006 Requests per concurrency level
`max(min_requests, requests_per_slot × concurrency)`, defaults 32 and 4. With too few requests the run
is mostly ramp-up and ramp-down rather than steady state at the target concurrency.

## D-007 Throughput definition
`tokens_per_s = successful output tokens / wall time of the whole load run` (first send → last finish).
This is the system's output rate, not the per-user streaming speed (that is 1000 / ITL).

## D-008 ITL is measured per streamed chunk
ITL is the gap between consecutive SSE chunks. vLLM and SGLang usually send one token per chunk. When a
runtime batches tokens per chunk, ITL becomes per-chunk; check `output_tokens / len(chunk_times_s)` in
the raw CSV before comparing runtimes.

## D-009 Energy
A background thread samples NVML power every 100 ms, and energy is the trapezoid integral per GPU,
summed over the GPUs the experiment uses. The integral is cross-checked against NVML's own cumulative
energy counter, and both values go in `env.json`. NVML power readings are averaged by the driver over a
short window, so very short runs are less accurate. `vram_gb` is peak usage *including* the runtime's
up-front KV-cache reservation (vLLM pre-allocates most of the memory), so it is not "memory the model
needs".
Each sample also records the SM clock, temperature and NVML's clock-event (throttle) reasons. Kaggle does
not allow locking clocks (`nvidia-smi -lgc` is refused), so the clock state is recorded instead of fixed.

## D-010 AWQ kernel selection
For `awq-int4` we pass only `--dtype float16` and let the runtime read the checkpoint's quantization
config. It then picks the fastest AWQ kernel the GPU supports, which differs between T4 and H100. The
server log records which kernel was chosen.

## D-011 FP8 on H100: one ModelOpt checkpoint for all engines, FP8 KV-cache everywhere
Qwen's own `Qwen/Qwen3-8B-FP8` uses block-wise FP8 (128×128 weight blocks). TensorRT-LLM 1.2.1's support
matrix lists Qwen-3 with per-tensor FP8 but not block-scaling FP8, so that checkpoint could fail or fall back
on one engine only. `nvidia/Qwen3-8B-FP8` (ModelOpt 0.35, per-tensor W8A8 with static scales, FP8 KV scales,
`lm_head` kept in BF16) is supported by all three, so every engine loads the same weights and the FP8
comparison is about engines, not quantization recipes.
No `--quantization` flag is passed; each engine reads `quant_method: modelopt` from the checkpoint. The KV-cache
dtype is set explicitly to FP8 on every engine (vLLM `--kv-cache-dtype fp8`, SGLang `fp8_e4m3`, TRT-LLM
`kv_cache_config.dtype: fp8`), because the engines' defaults differ. "fp8" in results therefore means FP8
weights, activations and KV-cache; the BF16 runs keep a BF16 KV-cache.
TensorRT-LLM's prefix caching is KV block reuse, which is on by default; `prefix_caching: false` turns it off
through `--extra_llm_api_options`, as for the other engines. Other engine defaults (memory fraction, CUDA graph
batch sizes, scheduler) stay as shipped and are read from each server log ([findings F9](findings.md#f9)).

## D-012 Load generator
A semaphore keeps exactly `concurrency` requests in flight, so the server sees a steady number of users
instead of one burst. The arrival time of each streamed chunk is recorded before the JSON is parsed, so client
work is not counted as server latency. A stream with no text or no usage chunk counts as a failure rather than
a fast zero-token success. If no request succeeds, every metric is `None` ("not measured"), never 0.

## D-013 TPOT for latency per token above ~20k tok/s
At H100 throughput one Python client reading 256 streams falls behind and reads chunks in bursts, so the gap
between chunks (ITL, D-008) can read as 0 ms. H100 results therefore report TPOT = (end − first chunk) ÷
(output tokens − 1) per request, taking the median, which does not depend on when each chunk was read.
Throughput is cross-checked against the server's own counters ([findings F9](findings.md#f9)).
