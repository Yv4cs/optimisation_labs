# Stories

Findings worth telling: each one has a surprise, the evidence, the reasoning, and what we did about it.
They feed the talk with Tom and, in the final phase, an animated explainer website.

Rules: numbers are copied from logs or `results/`, never rounded into something nicer. Smoke-test numbers
(8 requests, 1 repeat) are labelled as such. A story stays `open` until its cause is confirmed by data.

Template:

```
## S-00X Title
Status: open | confirmed | resolved · Date: YYYY-MM-DD · Experiment: TNx / smoke
Context      what we were doing and why
Observation  what we saw, with exact numbers and where they came from
Reasoning    hypotheses and how each was tested
Outcome      what we concluded / changed
Pitch        2-3 sentences, ready to say out loud
Follow-ups   questions Tom might ask, with short answers
Animation    how the website could show it
```

---

## S-001 Same test, same GPU, 45% apart: the T4 is the noisy part

Status: **open** (clock/temperature data pending) · Date: 2026-11-16 · Experiment: smoke, TN1 config

**Context.** First end-to-end smoke test of TN1: vLLM 0.33.0, Qwen2.5-3B-Instruct FP16, 1× Tesla T4 on
Kaggle, concurrency 1, 8 ShareGPT requests × 256 output tokens.

**Observation.**

| Source | Value |
|---|---|
| Paper estimate, decode at batch 1 | ≈ 50 tok/s upper bound (6 GB of FP16 weights ÷ 320 GB/s ≈ 19 ms per token) |
| Smoke run 1, our `tokens_per_s` | **13.1 tok/s** |
| Smoke run 1, our `itl_p50_ms` | 63.6 ms (≈ 15.7 tok/s per stream) |
| Smoke run 1, vLLM's own log, 10 s windows | 15.0 → 10.6 → 8.8 → 9.1 → 9.2 tok/s, with `Running: 1 reqs`, KV-cache usage 0.1–0.3% |
| Smoke run 1, `gpu_util_pct` | 99.9% |
| Smoke run 1, `avg_power_w` | 60.0 W (T4 power limit is 70 W) |
| Smoke run 1, `joules_per_token` | 4.58 J |
| Smoke run 2, identical command | **19.0 tok/s** |

**Reasoning.**
- *A. Our measurement code is wrong.* Rejected: vLLM's server-side count is independent of our client
  and lands in the same range, and our tokens/s is consistent with our own ITL.
- *B. The GPU throttles (heat or power cap).* Likely: vLLM's speed drops over time while the load stays
  constant at one request, power sits near the 70 W cap, and two identical runs differ by 45%. The T4 is a
  passively cooled 70 W card. To confirm: log `clocks.sm`, `power.draw`, `temperature.gpu` during a run.
- *C. Turing kernels are simply slow.* Still possible as part of the gap to 50 tok/s; it would not explain
  the drift within one run or the gap between runs.

Side lesson: 99.9% "GPU utilisation" only means a kernel was running 99.9% of the time. It does not say how
efficiently. At batch 1 the GPU is busy waiting on memory.

**Outcome (so far).** The CV ≤ 5% gate in `runner.py` would reject most repeats in this state.
- 2026-11-17: `nvidia-smi -i 0 -lgc 1200,1200` → "The current user does not have permission to change
  clocks". Clocks cannot be locked on Kaggle, so they must be recorded instead.
- 2026-11-17: a separate `nvidia-smi -l 1 > clocks.csv` logger only kept the first ~80 s (idle: 300 MHz SM,
  10 W, 39 °C). nvidia-smi buffers its file output and was killed before flushing, so the loaded part was
  lost. Lesson: measure inside the instrument you already trust, not with a side process.
- Fix: `GpuSampler` now records SM clock, temperature and NVML clock-event reasons every 100 ms, so every
  run's `power.csv` shows whether it was throttled and why.
- Next: run TN1 and read the clock columns per run to confirm or reject hypothesis B.

**Pitch.** "My first smoke test gave 13 tokens/s where the roofline said about 50, and a second identical run
gave 19. I cross-checked against vLLM's own counters to rule out my client, then saw throughput drifting
down at constant load, which pointed to the T4 throttling. So before collecting numbers I had to control the
hardware state, not just the software config."

**Follow-ups.**
- *Why is decode at batch 1 memory-bound?* Every token reads all weights once; the math per byte is tiny.
- *Why not just average more runs?* Averaging hides a drifting variable; you want to remove or record it.
- *How would this matter at Firmus?* Thermal and power state change tokens/s and J/token directly; a
  benchmark that ignores them is not comparable across machines or cooling setups.

**Animation.** A thermometer next to a token stream: as the needle rises, the clock dial drops and tokens
come out slower. Two side-by-side runs of the "same" test diverging, with a 50 tok/s roofline line above both.

---

## S-002 The engine picks things you never asked for

Status: **confirmed** · Date: 2026-11-15 · Experiment: smoke

**Context.** Reading vLLM's startup log during the first smoke test.

**Observation.** Our command set only dtype, max model length and prefix caching off. The resolved config
also contained choices we never made:
- `Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION']`
- `enable_chunked_prefill=True`
- `--gpu-memory-utilization=0.9200`
- `compilation_config mode=VLLM_COMPILE`, `cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes 1…512

**Reasoning.** FlashAttention 2 needs Ampere (sm_80) or newer; the T4 is Turing (sm_75), so vLLM falls back to
a Triton kernel. Chunked prefill changes how prompt processing is interleaved with decoding, which moves
TTFT and ITL. These defaults also differ between engine versions.

**Outcome.** Record the resolved config from the server log, not just our flags (to do in `DECISIONS.md`).
When comparing T4 with H100 later, part of the difference is the attention kernel, not only the hardware.

**Pitch.** "A benchmark is defined by the engine's resolved config, not by the flags you passed. On T4, vLLM
silently used a Triton attention kernel because FlashAttention 2 needs Ampere, and chunked prefill was on by
default, so I log the effective config next to every result."

**Follow-ups.**
- *What does chunked prefill do?* Splits a long prompt's prefill into chunks so decode steps for other users
  are not blocked; better ITL under load, possibly slower TTFT for the long prompt.
- *Why do CUDA graphs need a warm-up?* Capture and compilation happen up front; the first requests after
  start-up are not representative.

**Animation.** A "flags you passed" card next to a much longer "config the engine actually used" card, with
the hidden defaults highlighted one by one.

---

## S-003 KV-cache capacity predicted on paper, then confirmed by the log

Status: **confirmed** · Date: 2026-11-16 · Experiment: smoke

**Context.** Before running, we estimated how many tokens of KV-cache fit on one T4.

**Observation.**
- Estimate: 2 (K and V) × 36 layers × 2 KV heads × 128 head dim × 2 bytes ≈ 36 KB per token; with roughly
  5–6 GB left after weights, about 150k tokens.
- vLLM log: `GPU KV cache size: 184,496 tokens, Maximum concurrency for 4,096 tokens per request: 45.04x`.
- `vram_gb` = 14.7 GB at batch 1, because vLLM reserves the KV pool up front.

**Reasoning.** Same order of magnitude. The estimate assumed slightly less free memory than vLLM found.
GQA matters: with 2 KV heads instead of 16, the cache is 8× smaller than full multi-head attention.

**Outcome.** "45x" is the worst case (every request 4096 tokens). ShareGPT prompts are much shorter, so the
point where requests start queueing should be well above c=45. TN1's sweep will show where it is.

**Pitch.** "I sized the KV-cache by hand at about 36 KB per token, so roughly 150k tokens on a T4, and vLLM
reported 184k. That number is what caps concurrency, which is why VRAM, not compute, is often the real limit
on how many users one GPU can serve."

**Follow-ups.**
- *Why is reported VRAM 14.7 GB for a 6 GB model?* Pre-allocated KV pool; it is not what the model needs.
- *How does GQA change this?* Fewer KV heads means proportionally less cache per token, so more concurrent users.

**Animation.** A GPU memory bar: a 6 GB weights block, then the KV pool filling up as users arrive; each user
is a growing stack of 36 KB tiles, and the 46th long user has to wait in a queue.

---

## S-004 Building the load generator so the numbers mean something

Status: **resolved** · Date: 2026-11-15 · Experiment: all

**Context.** Writing `loadgen.py` and `metrics.py`.

**Observation / decisions.**
- A semaphore keeps exactly N requests in flight, so the server sees N steady users instead of one burst.
- Requests per level = max(32, 4 × concurrency), so a run is mostly steady state, not ramp-up (D-006).
- The clock is read when a chunk arrives, before JSON parsing, so client work is not timed as server latency.
- A stream with no text or no usage chunk counts as a failure; otherwise it would count as a success with
  0 tokens and quietly drag throughput down.
- With zero successful requests every metric is `None` ("not measured"), never 0.
- Every request sends `temperature=0`, `ignore_eos=true` and a fixed `max_tokens`, so every engine generates
  the same number of tokens (D-003).

**Pitch.** "The load generator models N concurrent users with a semaphore, and it is strict about what counts
as a success, because a benchmark that silently records failures as fast zero-token requests lies."

**Follow-ups.**
- *Client-side TTFT vs server-side TTFT?* Ours includes HTTP and network time on localhost; small but nonzero.
- *Why throughput over wall time, not the mean of per-request speeds?* The system's output rate is what pays
  for the GPU; per-user speed is 1000 / ITL (D-007).

**Animation.** N lanes with ticket gates: a new request enters only when one leaves; compare with a burst
where all requests rush in and the lanes then drain empty.

---

## S-005 Reproducibility on a machine that forgets everything

Status: **resolved** · Date: 2026-11-16 · Experiment: environment

**Context.** Kaggle has no Docker and wipes `/kaggle/working` between sessions.

**Observation.** `pip install vllm` replaced Kaggle's torch with 2.14.0. A new session lost the repo, vLLM,
the 673 MB ShareGPT download and the compile cache. An unpinned `requirements-kaggle.txt` could install a
different vLLM next time.

**Outcome.** Install `vllm==0.33.0` explicitly, then `make freeze` writes `env/lock-kaggle.txt` with every
package version (D-001). Docker image tags play that role on RunPod.

**Pitch.** "On Kaggle I could not use containers, so I pinned the engine version that I had verified on the
T4 and froze the full environment to a lock file, so every number can be reproduced on a fresh session."

**Animation.** A notebook session that resets to empty, then rebuilds itself from the lock file into the same state.
