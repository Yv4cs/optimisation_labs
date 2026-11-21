# Findings

What the measurements showed, with the evidence and the explanation. Every number comes from
`results/runs.jsonl`, the per-run files in `results/raw/`, or the server logs of those runs. Values are means
over 3 accepted repeats unless noted. How things were measured is in [methodology.md](methodology.md);
which configuration to run for which goal is in [recipes.md](recipes.md).

| # | Finding | Hardware |
|---|---|---|
| [F1](#f1) | Batching is the biggest lever: 74× throughput, 47× less energy per token | H100, T4 |
| [F2](#f2) | FP8 adds 33–53% on H100; INT4 on T4 helps one user only | H100, T4 |
| [F3](#f3) | vLLM and SGLang are within a few percent of each other | H100 |
| [F4](#f4) | TensorRT-LLM halves at 256 users with default CUDA graph sizes | H100 |
| [F5](#f5) | Prefix caching doubles throughput when prompts share a long prefix | T4 |
| [F6](#f6) | KV-cache is the limit only with long prompts | T4 |
| [F7](#f7) | On two GPUs, tensor parallel wins until the GPU is compute-bound | 2× T4 |
| [F8](#f8) | T4 to H100: 12–29× faster, only 3× less energy per token | T4, H100 |
| [F9](#f9) | Measurement lessons | all |

---

<a id="f1"></a>
## F1. Batching is the biggest lever

On one H100 with Qwen3-8B FP8 on vLLM, going from 1 to 256 concurrent requests raised throughput 74× and cut
energy per token 47×, while power rose only 1.6×.

| Concurrency | Throughput (tok/s) | Per-request speed (tok/s, 1000/TPOT) | TTFT p50 | Power | J/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 223 | 227 | 16 ms | 416 W | 1.866 |
| 16 | 3,044 | 204 | 76 ms | 460 W | 0.150 |
| 64 | 9,146 | 163 | 198 ms | 541 W | 0.059 |
| 256 | 16,432 | 79 | 687 ms | 649 W | 0.0395 |

**Why.** Each decode step reads every weight from memory once, whether it serves 1 request or 256. At batch 1
the step is memory-bound and the compute units mostly wait. Batching spreads the same weight read over more
tokens, so output grows much faster than power.

**Trade-off.** Each request gets slower (227 → 79 tok/s) and waits longer for its first token (16 → 687 ms).
The right concurrency is set by the latency target, not by peak throughput.

**Same shape on a T4** (Qwen2.5-3B FP16): 18.9 → 792 tok/s from c=1 to c=256, and 3.461 → 0.082 J/token (42×).
The T4 drew 64–66 W at every level, because it sits at its 70 W cap even at batch 1.

**Cost.** At USD 4.39/h, cost per million output tokens on the H100 falls from USD 8.36 (BF16, c=1) to 0.098
(BF16, c=256) and 0.074 (FP8, c=256): 113× in total, about 85× from batching and 1.33× from FP8.

---

<a id="f2"></a>
## F2. Precision: FP8 helps at every load on H100; INT4 on T4 helps one user only

**H100, Qwen3-8B, vLLM, FP8 vs BF16** (FP8 weights, activations and KV-cache; see methodology D-011):

| Concurrency | BF16 tok/s | FP8 tok/s | Gain | BF16 J/token | FP8 J/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 145.8 | 222.6 | +53% | 3.117 | 1.866 |
| 16 | 2,034 | 3,044 | +50% | 0.248 | 0.150 |
| 64 | 6,483 | 9,146 | +41% | 0.099 | 0.059 |
| 256 | 12,398 | 16,432 | +33% | 0.054 | 0.0395 |

**T4, Qwen2.5-3B, AWQ-INT4 vs FP16:** about 3.5× faster per token at c=1 (12.6 ms vs 42–47 ms on the same card
in the same session) but only +2% at c=256 (809.9 vs 792.0 tok/s).

**Why they differ.** At batch 1 both halve or quarter the bytes read per token, so both help. At high load the
step is compute-bound. Hopper has FP8 tensor cores, so FP8 also speeds up the math. The T4 has no INT4 tensor
path for AWQ: the weights are dequantised inside the kernel, so the byte saving stops mattering.

**Not measured:** accuracy of FP8 vs BF16 and of INT4 vs FP16. Run an eval (e.g. GSM8K) before shipping either.

---

<a id="f3"></a>
## F3. Engines: vLLM and SGLang are within a few percent of each other

| Concurrency | vLLM BF16 | SGLang BF16 | TRT-LLM BF16 | vLLM FP8 | SGLang FP8 | TRT-LLM FP8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (tok/s) | 145.8 | 150.4 | 139.6 | 222.6 | 214.6 | 222.1 |
| 16 | 2,034 | 2,034 | 1,860 | 3,044 | 2,851 | 3,045 |
| 64 | 6,483 | 6,441 | 5,960 | 9,146 | 8,989 | 8,991 |
| 256 | 12,398 | 12,333 | **8,882** | 16,432 | 16,361 | **8,530** |

vLLM and SGLang agree within 4% everywhere except FP8 at c=16 (vLLM +6.8%), and within about 0.5% at c=256, which is
inside the repeat-to-repeat variation.

**Why.** For a dense model with plain decoding, all three engines run similar kernels (FlashAttention-class
attention, cuBLAS/CUTLASS GEMMs, CUDA graphs) close to the hardware limit. Precision and batch size change the
work per token by much more than the engine does. Choose an engine on features and operations, and check its
defaults (F4).

---

<a id="f4"></a>
## F4. TensorRT-LLM halves at 256 users with default CUDA graph sizes

**Observation.** TensorRT-LLM 1.2.1 matched vLLM up to c=64 (FP8: 8,991 vs 9,146 tok/s). At c=256 it fell to
8,530 tok/s, below its own c=64 and half of vLLM's 16,432. The GPU was busy only 43% of the time (BF16: 71%),
drew 375–385 W against 649 W for vLLM, and ran at its maximum SM clock of 1,980 MHz. It was idle, not throttled.

**Diagnosis.**
- *Queueing or KV-cache admission:* ruled out. TTFT p50 was 0.5–0.9 s.
- *Single-process detokenisation:* ruled out. Re-running with `--num_postprocess_workers 8` gave 8,561 tok/s.
- *CUDA graphs not covering the batch:* most likely. The server log shows `Creating CUDA graph instances for 34
  batch sizes`, covering 1–32, 64 and 128, with `enable_padding=False`. A decode batch near 256 has no graph,
  so each step launches its kernels one by one from the CPU and the GPU waits between them. vLLM captures
  graphs up to 512.

**Fix, prepared but not yet run:**
```yaml
cuda_graph_config:
  max_batch_size: 256
  enable_padding: true
```
Until it is measured, TensorRT-LLM's c=256 numbers are "defaults, untuned", not its capability.

**Note.** Energy per token hides this (0.044 vs 0.039 J/token), because an idle GPU draws less power. Cost per
token does not: USD 0.143 vs 0.074 per million tokens, because the GPU is paid for by the hour.

---

<a id="f5"></a>
## F5. Prefix caching doubles throughput when prompts share a long prefix

Workload: every prompt is the same 1,024-token block followed by a different ShareGPT question (about 1,220
prompt tokens in total). T4, Qwen2.5-3B FP16, vLLM. The two runs are identical except for prefix caching.

| Concurrency | tok/s off → on | TTFT p50 off → on | TTFT p99 off → on | J/token off → on |
|---:|---:|---:|---:|---:|
| 16 | 171.2 → 254.9 | 3.4 → 1.7 s | 10.0 → 4.4 s | 0.383 → 0.266 |
| 64 | 225.6 → 399.0 | 4.5 → 3.9 s | 44.2 → 14.1 s | 0.286 → 0.169 |
| 256 | 214.9 → 442.9 | 152.5 → 9.4 s | 244.1 → 53.3 s | 0.300 → 0.151 |

**Why.** With caching off, each request re-processes the shared 1,024 tokens and stores its own copy in the
KV-cache. At c=256 the cache was 94–98% full, only 125–150 requests could run, and the rest waited
(`Waiting: 106–128 reqs` in the log). With caching on, the shared block is computed and stored once: the cache
hit rate was 81–84%, all 256 requests ran, and almost no prefill was left in steady state.

**Trade-off.** Per-token latency at c=256 rose from 257 to 387 ms, because every step now decodes 256 requests
instead of about 130.

**Caveat.** The two runs were in different Kaggle sessions. Batch-1 numbers vary between sessions (F9), but at
c ≥ 16 the effects (2× throughput, 16× TTFT p50) are far larger than any session-to-session noise measured.

---

<a id="f6"></a>
## F6. KV-cache is the limit only with long prompts

**Sizing.** Qwen2.5-3B uses about 36 KB of KV-cache per token (2 × 36 layers × 2 KV heads × 128 dims × 2 bytes).
The T4's pool was 184,496–194,160 tokens depending on the session. Read it from the server log
(`GPU KV cache size`): it also changes with flags (206,560 with `--enforce-eager`, 296,288 with AWQ,
557,248 with TP=2).

- **Short chat prompts (about 220 tokens):** KV usage peaked at 44% at c=256, with 0 requests waiting.
  The limit was compute under the 70 W power cap.
- **Long shared prompts, caching off (about 1,220 tokens):** 194,160 ÷ (1,220 + 256) ≈ 131 requests fit, which
  matches the 125–150 running in the log. Half the users queued and TTFT p50 reached 152 s (F5).

**Rule.** Maximum concurrent requests ≈ KV pool tokens ÷ (prompt + output tokens).

**Tail TTFT at low load is prefill, not queueing.** At c=1, TTFT tracked prompt length with r = 0.967, at about
0.83 ms per prompt token on a T4 (a 3,486-token prompt took 3.7 s). A p99 TTFT target needs a maximum prompt
length to go with it.

---

<a id="f7"></a>
## F7. On two GPUs, tensor parallel wins until the GPU is compute-bound

Two T4s over PCIe, Qwen2.5-3B FP16. TP=2 splits each layer across both cards. DP=2 runs two independent copies
with requests alternated between them.

| Concurrency | TP=2 tok/s | DP=2 tok/s | TP per-token latency | DP per-token latency | TP J/token | DP J/token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **61.0** | 27.0 | 15.8 ms | 32.4 ms | 2.191 | 3.656 |
| 16 | **549.9** | 380.6 | 23.6 ms | 34.0 ms | 0.246 | 0.347 |
| 64 | **1,044.1** | 838.3 | 43.9 ms | 52.6 ms | 0.129 | 0.158 |
| 256 | 1,178.5 | **1,376.0** | 145.3 ms | 114.4 ms | 0.114 | **0.095** |

**Why.** At low load each step is memory-bound, and TP halves the weights each card reads. The all-reduce
carries one 2,048-wide vector per request twice per layer, which even PCIe handles cheaply. At c=256 the step is
compute-bound: TP pays for synchronising every layer, while DP's replicas never coordinate. DP scaled almost
perfectly, matching two single cards at half the load each within 3–8%.

**Recommendation.** TP for latency-sensitive serving; replicas for bulk throughput and lowest energy at full
load. The crossover lies between c=64 and c=256 (not measured in between).

---

<a id="f8"></a>
## F8. T4 to H100: 12–29× faster, only 3× less energy per token

Same model (Qwen2.5-3B FP16), workload and engine (vLLM 0.33.0) on both GPUs.

| Concurrency | T4 tok/s | H100 tok/s | Speed-up | T4 J/token | H100 J/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.9 | 273.5 | 8–14× | 3.461 | 1.419 |
| 16 | 339.3 | 4,039.7 | 11.9× | 0.194 | 0.107 |
| 64 | 619.7 | 12,622.3 | 20.4× | 0.104 | 0.042 |
| 256 | 792.0 | 22,972.4 | 29× | 0.082 | 0.027 |

(The c=1 range reflects the T4's session-to-session variation at batch 1; see F9.)

**Why the energy gain is smaller.** The H100 draws about 10× the power (about 390–610 W against 65 W here), so
energy per token improves 2.4–3× while speed improves 12–29×. The gap in speed grows with load because the T4
flattens after c≈64 under its power cap, while the H100 is still scaling at 256.

**Both GPUs hit their power limit.** At c=256 the H100 BF16 runs report NVML's "SW power cap" reason in almost
every sample, at about 692 W, with the SM clock down to 1,635–1,680 MHz (maximum 1,980). FP8 does the same work
at about 671 W and keeps about 1,965 MHz.

---

<a id="f9"></a>
## F9. Measurement lessons

- **Record the hardware state, because you cannot fix it on cloud GPUs.** Two identical T4 smoke runs gave 13.1
  and 19.0 tok/s. Clocks cannot be locked on Kaggle (`nvidia-smi -lgc` is refused), so every 100 ms sample now
  records SM clock, temperature and NVML throttle reasons. Across 143,223 T4 samples the reason was "SW power
  cap" in 136,205 and none in the rest, with no thermal throttling: the T4 is power-limited, and the clock it
  can afford depends on how power-hungry the current kernels are.
- **The same model of GPU is not the same performance.** Two T4s in one machine, same request, same session:
  41.9–47.1 ms per token on GPU 0 against 27.3–27.8 ms on GPU 1, both at the 70 W cap. Each run now records the
  GPU UUID, and single-GPU comparisons are made within one session on one card.
- **Batch-1 latency with CUDA graphs is the least reproducible number.** The same config gave 51.9, 42–47 and
  28.7 ms per token in three sessions, while c ≥ 2 and eager mode stayed within a few percent. An apparent
  "two users get faster tokens than one" effect disappeared in a controlled same-session re-run.
- **Workload variance between repeats.** With 32 fresh prompts per repeat, one repeat averaged 48 prompt tokens
  and another 450, and throughput followed. Hence at least `4 × concurrency` requests per level and a 5% CV gate
  on every level (methodology D-004, D-006).
- **The client can become the bottleneck.** Above about 20,000 tok/s, one Python client reading 256 streams fell
  behind and measured inter-token latency as 0.0 ms. H100 results therefore use TPOT = (end − first token) ÷
  (tokens − 1), which does not depend on when chunks are read, and throughput is cross-checked against the
  server's own counters. A multi-process load generator is the real fix (not done).
- **The engine's resolved config is the benchmark, not the flags passed.** vLLM on T4 silently chose a Triton
  attention backend (FlashAttention 2 needs sm_80+), chunked prefill and CUDA graphs. TensorRT-LLM's default
  graph sizes caused F4. Each server log is kept with the results.
- **Check that every engine supports the same checkpoint.** Qwen's own FP8 release uses block-wise scaling,
  which TensorRT-LLM 1.2.1 does not list for Qwen-3. All engines therefore load NVIDIA's per-tensor
  `nvidia/Qwen3-8B-FP8`, with the KV-cache dtype set explicitly (methodology D-011).
- **A lock file is evidence only if it contains what it locks.** On the H100, `pip freeze` in `uv` venvs fell
  back to the system pip, so all three per-engine lock files described the same environment. They were dropped.
  The engine versions are in each run's `env.json`, written from inside the venv.
- **Read the price from the running machine.** The H100 was planned at USD 3.99/h, but the console showed 4.39.
  The `cost_usd` field in `results/runs.jsonl` uses 3.99 (multiply by 1.10). All cost figures in the docs use
  4.39. Ratios between setups are unaffected.
