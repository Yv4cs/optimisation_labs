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
- 2026-11-17, TN1 bench (old sampler, no clock columns): c=1 was stable across 3 repeats at 19.8 / 18.4 /
  18.6 tok/s (CV 4.2%), ITL p50 51–52 ms, power 65.3–65.5 W. Average power stayed at 64–66 W at every
  concurrency from 1 to 256, i.e. the card sat just under its 70 W cap the whole time. The 13.1 tok/s smoke
  run now looks like the outlier, and the run-to-run noise in TN1 is mostly workload, not hardware (S-010).
- 2026-11-17, the TN2/TN3 runs with the new sampler (143,223 samples over 5 experiments): the NVML reason
  bitmask is **4 (SW power cap) in 136,205 samples and 0 in the other 7,018**. No thermal-slowdown bit ever
  appears, although the cards reach 81 °C. The SM clock under load ranges from **390 MHz to 1,335 MHz**
  depending on the kernels running, while power stays at 66–67 W. So the T4 is not overheating: it is
  pinned at its 70 W budget, and the clock it can afford depends on how power-hungry the current work is.
  This is why the same work can run at different speeds (S-007, S-013).

Status update: **confirmed** (power cap, not thermal). Kept for the pitch: you cannot lock clocks on Kaggle,
so record them.

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

**Correction (2026-11-17).** The pool size is not a constant. Same model, same flags, different sessions:
184,496 tokens (smoke run, DP=2 replicas), 194,160 (TN1, prefix-off), 206,560 with `--enforce-eager`
(no memory reserved for CUDA graphs), 296,288 with AWQ (smaller weights), 557,248 with TP=2 (two cards, half
the weights each). Always read it from the log of the run you are analysing.

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

---

## S-006 One GPU, 42× cheaper per token: the TN1 curve

Status: **confirmed** · Date: 2026-11-17 · Experiment: TN1 (`results/runs.jsonl`, 42 runs, 11,904 requests, 0 errors)

**Context.** Full TN1 sweep, concurrency 1 → 256, 3 repeats per level, run 2026-11-16 16:55–18:38 UTC
(1 h 43 min), git commit `e33b9c7`, vLLM 0.33.0, driver 580.178.04, CUDA 13.0.

**Observation.** Means over the accepted repeats (c=128 had no accepted set; its 6 runs are shown):

| c | tokens/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | J/token | avg W |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.9 | 56 | 2,194 | 51.9 | 3.461 | 65.4 |
| 2 | 59.0 | 113 | 3,571 | 30.9 | 1.116 | 65.9 |
| 4 | 117.0 | 194 | 1,269 | 32.6 | 0.563 | 65.8 |
| 8 | 195.7 | 687 | 2,835 | 35.5 | 0.336 | 65.8 |
| 16 | 339.3 | 666 | 4,201 | 39.2 | 0.194 | 65.7 |
| 32 | 453.2 | 1,862 | 6,861 | 54.5 | 0.145 | 65.7 |
| 64 | 619.7 | 3,066 | 11,647 | 71.4 | 0.104 | 64.7 |
| 128 (6 runs, all discarded) | 688.9 (628.8–777.7) | | | 123.8 | 0.095 | |
| 256 | 792.0 | 6,581 | 30,120 | 201.6 | 0.082 | 64.6 |

**Reasoning.**
- Throughput grows ~2× per doubling up to c=4, ~1.7× to c=16, ~1.35× to c=64, then only ~1.1–1.15×.
  The knee is around c=32–64.
- Power is flat (64–66 W) at every level, so energy per token is simply power ÷ throughput:
  3.461 → 0.082 J/token, **42× cheaper** at c=256 than at c=1.
- Per-user experience gets worse: ITL 31 ms at c=2 (≈32 tok/s per user) vs 202 ms at c=256 (≈5 tok/s per user),
  and TTFT p50 grows from 56 ms to 6.6 s.

**Outcome.** The operating point depends on the latency target. Example reading (to be checked once SLOs are
chosen): if a user needs ≥ 20 tok/s streaming (ITL ≤ 50 ms), c=16 is the last level that qualifies, at
339 tok/s and 0.194 J/token.

**Pitch.** "On one T4 the card drew about 65 W no matter how many users it served, so batching was the whole
efficiency story: going from 1 to 256 concurrent users cut energy per token 42×. But per-user speed fell from
about 32 to 5 tokens/s, so the right concurrency is set by the latency SLO, not by peak throughput."

**Follow-ups.**
- *Why does power not rise with load?* The GPU is already at the edge of its 70 W cap at batch 1 (99.9% busy);
  more users means more useful work per step, not more watts.
- *What is goodput?* Throughput counted only while the latency SLO holds; it picks the operating point.

**Animation.** Two linked charts with a concurrency slider: throughput rising then flattening, and a battery
icon whose J/token cost shrinks; a per-user "typing speed" meter slows as the slider moves right.

---

## S-007 Two users get faster tokens than one

Status: **open** · Date: 2026-11-17 · Experiment: TN1

**Observation.** ITL p50 is 51.9 ms at c=1 but 30.9 ms at c=2 (and 32.6 at c=4). Throughput jumps 3.1×
(18.9 → 59.0 tok/s) for 2× the users. Power is the same (65.4 vs 65.9 W).

**Reasoning.** Decode at batch 2 should be slightly slower per step than batch 1, never 40% faster. At c=2
the effective weight-read bandwidth is 6 GB ÷ 31 ms ≈ 200 GB/s (≈ 60% of the T4's 320 GB/s); at c=1 it is only
≈ 120 GB/s. Something specific to batch size 1 is slow. Candidates: a poorly tuned kernel for M=1 matrix
multiplies on Turing, the CUDA-graph / compiled path for size 1, or how the engine schedules a lone request.

**Next.** Re-run c=1 and c=2 with `--enforce-eager` (no CUDA graphs, no compile) and compare; if the gap
remains, profile one decode step at batch 1 vs 2.

**Update 2026-11-17 (eager run + clock data; status still open, much narrower).**

| Run (all FP16, batch 1 unless noted) | Session | GPU | ITL per token | SM clock |
|---|---|---|---|---|
| TN1, graphs, c=1 | 16 Nov | 0 | 51.9 ms (p50) | not recorded |
| TN1, graphs, c=2 | 16 Nov | 0 | 30.9 ms (p50) | not recorded |
| eager, c=1 | 17 Nov | 0 | **38.4 ms** (p50) | 1,170 MHz |
| eager, c=2 | 17 Nov | 0 | **39.1 ms** (p50) | 1,320–1,335 MHz |
| DP=2 replica 0, graphs, c=1 | 17 Nov | 0 | 41.9 / 47.1 / 47.1 ms (per-request median, 3 repeats) | 390–555 MHz |
| DP=2 replica 1, graphs, c=1 | 17 Nov | 1 | 27.3 / 27.5 / 27.8 ms | (mixed with idle time) |

- Without CUDA graphs and compile, the c=1 vs c=2 gap is gone (38.4 vs 39.1 ms). So the anomaly lives in the
  graphs/compile path, not in the model or the scheduler.
- On the same card in the same session, the graphs path at batch 1 ran at **390–555 MHz** while eager ran at
  **1,170 MHz**, both at 66 W with reason "SW power cap" (S-001). Hypothesis: the captured batch-1 path keeps
  the GPU busier per cycle (no launch gaps), so it reaches the 70 W budget at a far lower clock and ends up
  slower per token than eager (≈ 42–47 vs 38 ms) on that card.
- The two cards differ on their own: 27 ms vs 42–47 ms per token for identical work (S-013).
- Missing: graphs at c=2 on the same card in the same session as eager. Config
  `t4_qwen3b_fp16_graphs.yaml` (same as the eager config minus the flag) closes that gap; run it next to the
  eager config in one session.

**Pitch (if confirmed).** "My batch-1 numbers were slower per token than batch-2, which should be impossible
for a memory-bound decode. It turned out to be [cause], so on this GPU single-stream latency is a kernel
problem, not a hardware limit."

**Animation.** Two pipes reading the same tank of weights: the single-user pipe is narrower than the two-user one.

---

## S-008 My KV-cache prediction was wrong, and that is the interesting part

Status: **confirmed** · Date: 2026-11-17 · Experiment: TN1 (`server_0.log`)

**Observation.** I predicted TTFT p99 would jump when the KV-cache filled up. At c=256 vLLM's log shows
`Running: 256 reqs, Waiting: 0 reqs` with GPU KV cache usage at most **44.0%**, and 0 preemptions in the log.
Average prompt ≈ 210–230 tokens + 256 output tokens ≈ 480 tokens × 256 users ≈ 123k tokens, under the
194,160-token pool of that run (S-003 correction). S-014 is the counterpart where KV *is* the wall.

**Reasoning.** With realistic (short) chat prompts the KV-cache was never the limit on one T4. The plateau
(~790 tok/s) and the TTFT growth come from step time: at c=256 every decode step takes ~200 ms (ITL), and a
new request's prefill has to share steps with 255 others. With the card pinned at its power cap, the limit is
compute/power, not memory capacity. To be confirmed with clock data (S-001).

**Pitch.** "I expected KV-cache capacity to be the wall. The log showed 44% usage at 256 users, so for chat-length
prompts on a T4 the limit was compute under a 70 W power cap. KV-cache becomes the wall with long contexts,
which is a separate experiment."

**Animation.** The KV memory bar from S-003 filling only to 44% while a clock/power gauge is pegged at the limit.

---

## S-009 Tail TTFT at one user is the prompt length, not a queue

Status: **confirmed** · Date: 2026-11-17 · Experiment: TN1 c=1 (`requests.csv`)

**Observation.** At c=1, TTFT p50 is 56 ms but p99 is 1.6–2.8 s. Across the 96 c=1 requests, TTFT correlates
with prompt length at r = 0.967, slope ≈ 0.83 ms per prompt token: 4-token prompts start in ~46 ms, the
3,486-token prompt took 3,722 ms.

**Reasoning.** With one user there is no queue, so the tail is prefill time. 0.83 ms per token ≈ 1,200 prompt
tokens/s of prefill on a T4.

**Pitch.** "Even with one user, my p99 TTFT was almost 3 seconds. That was not queueing: TTFT tracked prompt length
with r = 0.97, about 0.8 ms per prompt token, so long prompts are a prefill cost you pay regardless of load."

**Animation.** Scatter plot of prompt length vs TTFT with the fitted line; dots light up as each request runs.

---

## S-010 Half the noise came from my own workload

Status: **confirmed** · Date: 2026-11-17 · Experiment: TN1

**Observation.** 5 of 9 levels needed a re-run (CV > 5%), and c=128 was discarded twice. At c=4 the discarded
set had repeats with mean prompt lengths of 203, **450** and **48** tokens, giving 115.1 / 101.6 / 125.2 tok/s
and TTFT p99 from 169 ms to 2,580 ms.

**Reasoning.** Each repeat uses fresh ShareGPT prompts (D-005, to avoid prefix-cache reuse). With only 32
requests per level, the mean prompt length varies a lot between repeats, and the numbers follow it. The
hardware was steady (power 64–66 W throughout).

**Outcome / options.** Keep fresh prompts but make repeats comparable: draw each repeat from the same
length distribution (stratify by prompt length) or raise `min_requests`. Report CV next to every number.

**Pitch.** "My CV gate kept rejecting repeats, and the cause was not the GPU: with 32 fresh prompts per repeat,
one repeat averaged 48 prompt tokens and another 450. Controlling the workload distribution matters as much
as controlling the hardware."

**Animation.** Three repeat "buckets" of prompts with visibly different sizes, each feeding a throughput bar.

---

## S-011 INT4 weights: 4× faster for one user, nothing gained at 256

Status: **confirmed** (speed), quality **not measured** · Date: 2026-11-17 · Experiment: TN2a
(`t4-qwen2.5-3b-awq-vllm`, 21 runs, same ShareGPT prompts as TN1)

**Context.** Same model in its official AWQ-INT4 checkpoint (`Qwen/Qwen2.5-3B-Instruct-AWQ`), same engine and
workload as TN1. Prediction: fewer weight bytes help when decode is memory-bound (low concurrency) and stop
helping once it is compute-bound (high concurrency).

**Observation.** Means over accepted repeats (AWQ c=16: both attempts had CV > 5%, last attempt shown).

| c | AWQ tok/s | FP16 tok/s (TN1) | AWQ ITL p50 | FP16 ITL p50 | AWQ J/token | FP16 J/token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **74.2** | 18.9 | **12.6 ms** | 51.9 ms | **0.893** | 3.461 |
| 4 | 208.7 | 117.0 | 14.4 ms | 32.6 ms | 0.317 | 0.563 |
| 16 | 519.1 (unstable) | 339.3 | 21.3 ms | 39.2 ms | 0.127 | 0.194 |
| 64 | 682.2 | 619.7 | 55.4 ms | 71.4 ms | 0.096 | 0.104 |
| 256 | 809.9 | 792.0 | 190.9 ms | 201.6 ms | 0.081 | 0.082 |

- TN1 ran in an earlier session. A same-session, same-card FP16 batch-1 reference is DP=2 replica 0 at
  42–47 ms per token (S-007), so the fair batch-1 gain is ≈ 3.3–3.7×, still large.
- KV pool: 296,288 tokens (FP16: 194,160), because the weights take less memory.
- SM clock at c=1: 1,020–1,110 MHz (FP16 graphs on the same card: 390–555 MHz).

**Reasoning.** At batch 1 each token reads every weight once; 4-bit weights are roughly a third of the bytes
(embeddings and some layers stay 16-bit), so the read takes roughly a third of the time. At c=256 the step is
dominated by compute on 256 rows, and AWQ must dequantize weights inside the kernel, so the byte saving no
longer shows: +2% throughput, same J/token.

**Outcome.** INT4 is a latency and single-user energy optimisation on this card (4× less energy per token at
c=1), not a peak-throughput one. Accuracy cost is unknown: needs a GSM8K or similar check before recommending it.

**Pitch.** "Quantising to INT4 made one user's tokens 4× faster and 4× cheaper in energy, because batch-1
decode is just reading weights. At 256 users it gave 2%, because by then the GPU is compute-bound. So
quantisation is a latency tool first, and I would still check accuracy before shipping it."

**Follow-ups.**
- *Why not faster at high load?* Compute-bound: the bytes saved are no longer the bottleneck, and
  dequantisation adds work.
- *Would FP8 on H100 behave the same?* FP8 has native tensor-core support on Hopper, so it can also speed up the
  compute-bound regime; that is what TN4 measures.

**Animation.** Two book stacks (16-bit vs 4-bit) being read by a single reader: the thin book is done 4× sooner.
Then 256 readers crowd in and both stacks are bottlenecked at the same desk.

---

## S-012 Two GPUs: splitting the model beat copying it, until the very top

Status: **confirmed** · Date: 2026-11-17 · Experiment: TN3 (`t4x2-…-tp2`, `t4x2-…-dp2`, 2× T4 over PCIe, same session)

**Context.** Two T4s, no NVLink. TP=2 splits every layer across both cards and all-reduces activations after
each layer. DP=2 runs two independent copies, and the load generator alternates requests between them. My
prediction: since the 3B model fits on one card, DP wins because it never talks over PCIe.

**Observation.** Means over accepted repeats:

| c | TP=2 tok/s | DP=2 tok/s | TP ITL p50 | DP ITL p50 | TP J/token | DP J/token | TP W | DP W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **61.0** | 27.0 | **15.8 ms** | 32.4 ms | 2.191 | 3.656 | 133.7 | 98.7 |
| 4 | **191.6** | 117.7 | 18.0 ms | 29.8 ms | 0.705 | 1.131 | 135.0 | 133.2 |
| 16 | **549.9** | 380.6 | 23.6 ms | 34.0 ms | 0.246 | 0.347 | 135.0 | 132.3 |
| 64 | **1,044.1** | 838.3 | 43.9 ms | 52.6 ms | 0.129 | 0.158 | 134.4 | 132.4 |
| 256 | 1,178.5 | **1,376.0** | 145.3 ms | 114.4 ms | 0.114 | **0.095** | 134.1 | 131.0 |

TTFT p99 at c=256: TP 22.1 s, DP 14.3 s (one T4 in TN1: 30.1 s). TP's KV pool is 557,248 tokens.

**Reasoning.**
- My prediction was wrong for c ≤ 64. With TP each card reads only half the weights per step, and in the
  memory-bound regime that halves step time. The all-reduce payload is tiny (one 2,048-wide vector per user,
  twice per layer), so even PCIe carries it cheaply at low batch.
- At c=256 the step is compute-bound. TP pays the per-layer synchronisation on every step and each card
  multiplies narrower matrices; DP runs two independent batches of 128 with no coordination. DP wins by 17%
  and has lower energy per token.
- DP scales almost perfectly: DP=2 at c ≈ 2× single-card TN1 at c/2 (e.g. 380.6 vs 2 × 195.7 at c=16), to
  within 3–8% across sessions.
- TP keeps both cards busy for one user (133.7 W at c=1); DP leaves the second card mostly idle (98.7 W).

**Outcome.** Interactive, latency-bound serving on this hardware: TP=2. Saturated batch throughput or
lowest J/token at full load: DP=2. The crossover is somewhere between c=64 and c=256 (not measured in between).
On H100 with NVLink, TP's communication cost is lower still.

**Pitch.** "I predicted two replicas would beat tensor parallelism because the model fits on one card and PCIe
is slow. I was wrong up to 64 users: TP halves the weight reads per card, and the all-reduce payload is tiny,
so single-user latency was 2× better. Only at 256 users, when the step is compute-bound, did replicas win by
17%. The right layout depends on where your traffic sits on that curve."

**Follow-ups.**
- *Why does TP not need NVLink here?* Per token the all-reduce moves kilobytes; the cost is the number of
  synchronisations (two per layer), not bandwidth. With bigger batches and models, bandwidth starts to matter.
- *What would you do in production?* Mix: TP groups for latency-sensitive tiers, replicas behind a load
  balancer for bulk traffic; pick per SLO.

**Animation.** Two kitchens. TP: one recipe book torn in half, cooks shouting across a corridor after every
step, dishes come out fast. DP: two full kitchens, each with its own queue. A slider raises the number of
diners and the "winner" flag flips from TP to DP near the right edge.

---

## S-013 Two identical T4s in one box, 1.7× apart

Status: **confirmed** (observation), cause **open** · Date: 2026-11-17 · Experiment: TN3 DP=2, c=1 (`requests.csv`)

**Observation.** DP=2 at c=1 sends even request IDs to GPU 0 and odd ones to GPU 1. Same model, same flags,
same session, one request at a time. Median per-request time per token over 3 repeats:
GPU 0 **41.9 / 47.1 / 47.1 ms**, GPU 1 **27.3 / 27.5 / 27.8 ms**. GPU 0 reached 80 °C at c=1, GPU 1 73–75 °C;
both report only the SW power-cap reason (S-001). TN1 at c=1 (also GPU 0, earlier session) shows no such
split between even and odd requests (51–55 ms for both), which confirms the difference is the card.

**Reasoning.** Same model of card, same power limit, different behaviour: silicon variation (leakage, which
changes how much clock each watt buys) and cooling position in the chassis both fit. We cannot open the box,
so the cause stays open.

**Outcome.** Every single-GPU experiment (TN1, AWQ, eager, prefix) ran on GPU 0, the slower card. Comparisons
within one session and one card are fair; absolute single-GPU numbers are on the pessimistic side. Record
the GPU index and its clocks with every run (now done), and in production expect per-card spread.

**Pitch.** "Two identical T4s in the same machine served the same request at 27 and 45 milliseconds per token.
Both were power-capped at 70 W, but one bought far less clock per watt. So in a fleet, per-GPU telemetry
matters: the same SKU is not the same performance."

**Animation.** Two identical GPU icons with a power meter fixed at 70 W; one clock dial spins fast, the other
slow, and tokens drip out at visibly different rates.

---

## S-014 Long shared prompts: now the KV-cache is the wall

Status: **open** (prefix caching ON run pending) · Date: 2026-11-17 · Experiment: TN2b, prefix caching OFF
(`t4-qwen2.5-3b-fp16-vllm-prefix-off`)

**Context.** Every prompt starts with the same 1,024-token block (think: a long system prompt) followed by a
ShareGPT question, so mean prompt length is 1,192–1,259 tokens. vLLM log confirms `enable_prefix_caching=False`.

**Observation.**

| c | tok/s | TTFT p50 | TTFT p99 | ITL p50 | J/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.1 | 540 ms | 2.3 s | 39.5 ms | 2.748 |
| 4 (unstable) | 84.7 | 1.6 s | 4.2 s | 35.5 ms | 0.782 |
| 16 | 171.2 | 3.4 s | 10.0 s | 54.1 ms | 0.383 |
| 64 | 225.6 | 4.5 s | 44.2 s | 125.3 ms | 0.286 |
| 256 | 214.9 | **152.5 s** | **244.1 s** | 257.3 ms | 0.300 |

vLLM log at c=256: `Running: 125–150 reqs, Waiting: 106–128 reqs, GPU KV cache usage: 94.0–98.2%`, with prompt
throughput 1,260–1,640 tokens/s and generation ~99 tokens/s in those windows; 0 preemptions.

**Reasoning.** Pool 194,160 tokens ÷ (≈1,220 prompt + 256 output) ≈ 131 requests at full length, which
matches the ~125–150 running. Unlike TN1 (S-008, 44% KV at c=256), half the users now wait outside the cache,
and the GPU spends its time on prefill: each new user costs ~1,220 prompt tokens at ~1,200–1,600 tokens/s
(the same prefill rate as S-009). Throughput peaks at c=64 and falls slightly at 256; 3.7× lower than TN1 at
the same concurrency.

**Prediction for prefix caching ON.** The shared 1,024 tokens are prefilled once and stored once, so per-user
prefill drops to ~200 tokens and per-user KV to ~450 tokens. TTFT should fall sharply and all 256 users should fit.

**Pitch (to finish).** "With a 1,000-token system prompt and caching off, 256 users meant 150-second TTFTs:
half of them waited because the KV-cache was 98% full, and the GPU spent its time re-reading the same system
prompt. Turning prefix caching on [result]."

**Animation.** The S-003 memory bar, now filled to the brim with identical blue "system prompt" blocks, one
per user, and a queue outside; then the caching switch flips and the blue blocks collapse into one shared block.
