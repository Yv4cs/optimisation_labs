# Design decisions

Each entry: the choice, why, and what would make us revisit it.

## D-001 Environment pinning without Docker on Kaggle
Kaggle notebooks cannot run Docker. On Kaggle we install from `requirements-kaggle.txt`, verify the
server works on T4, then `make freeze` writes `env/lock-kaggle.txt`. That lock file is the source of truth.
Docker is used only on RunPod (H100), where the image tag pins everything.

## D-002 Models
T4 (16 GB, no BF16, no FP8): `Qwen/Qwen2.5-3B-Instruct` in FP16 plus the official `-AWQ` checkpoint.
These checkpoints are ungated (no HF token needed), and 3B FP16 weights leave room for KV-cache on one T4.
H100: an 8B-class model from the same family, chosen on 17/11 together with the H100 checklist.

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

## D-010 AWQ kernel selection
For `awq-int4` we pass only `--dtype float16` and let the runtime read the checkpoint's quantization
config. It then picks the fastest AWQ kernel the GPU supports, which differs between T4 and H100. The
server log records which kernel was chosen.
