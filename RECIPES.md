# Recipes

Each recipe answers a serving question: *for this goal, on this hardware, run this command, and expect these numbers.*
The configs are the ones we measured, and the numbers come from `results/runs.jsonl`. The reasoning behind them is in `STORIES.md`.

Status: T4 recipes R1–R6 (2026-11-18), H100 recipes R7–R9 (2026-11-19).

## How to read the numbers

- **Workload.** The prompt is the first human turn of a ShareGPT conversation, with a mean of about 200–300 prompt tokens and a long tail up to about 3.5k.
  Every request gets exactly 256 output tokens (`temperature=0`, `ignore_eos`). The *shared-prefix* workload puts one fixed 1,024-token block before each question.
- **Concurrency `c`** is the number of requests in flight. The load generator keeps exactly `c` requests open (STORIES S-004).
- **Values** are means over the accepted repeats: 3 repeats, CV ≤ 5%. Unstable levels are marked.
- **Interactive SLO** used below: **TTFT p50 ≤ 1 s and ITL p50 ≤ 50 ms**, where 50 ms per token is about 20 tok/s per user, faster than reading speed.
  This SLO was set before reading the tables. Change it, and the chosen operating point changes too.
- **Comparisons across sessions.** Kaggle sessions can land on different cards. Batch-1 latency moved from 28.7 to 51.9 ms between sessions, while numbers at c ≥ 2 stayed within a few percent (S-007, S-013).
  So batch-1 comparisons are made within one session wherever possible.

## Environment (pinned)

| | |
|---|---|
| GPU | Tesla T4 16 GB (sm_75, 70 W cap), Kaggle; 2× T4 over PCIe for the two-GPU recipes |
| Driver / CUDA | 580.178.04 / 13.0 |
| Engine | vLLM 0.33.0, torch 2.14.0 (`requirements-kaggle.txt`, full lock in `env/lock-kaggle.txt`) |
| FP16 model | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| INT4 model | `Qwen/Qwen2.5-3B-Instruct-AWQ` @ `3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd` |

The commands below are the ones the harness ran, minus `--host`/`--port`. We added `--revision` to pin the checkpoint we recorded; the benchmark itself downloaded the same revision without passing the flag.
To reproduce any row, run `make setup && make bench EXP=<config>`.

## Which recipe?

| Your goal | Hardware | Recipe |
|---|---|---|
| One user, lowest latency per token | 1× T4 | **R1** INT4 |
| One user, lowest latency, no accuracy risk | 2× T4 | **R2** TP=2 |
| Many chat users within the interactive SLO | 1× or 2× T4 | **R3** |
| Maximum throughput or lowest energy per token, no latency target (batch, offline) | 1× or 2× T4 | **R4** |
| Many users sharing a long system prompt (RAG, agents, chat products) | any | **R5** prefix caching |
| Long, *different* prompts per user | any | **R6** (KV-cache limit) |
| Highest throughput / lowest cost per token on H100 | 1× H100 | **R7** FP8, vLLM or SGLang |
| Lowest single-user latency on H100 | 1× H100 | **R8** FP8, any engine |
| Serving TensorRT-LLM above ~128 concurrent requests | 1× H100 | **R9** check CUDA graph sizes |

---

## R1: Lowest single-user latency on one T4, using AWQ-INT4

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --revision 3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd \
  --tensor-parallel-size 1 --dtype float16 --no-enable-prefix-caching --max-model-len 4096
```
Config: `configs/experiments/t4_qwen3b_awq.yaml`

| c | tok/s | TTFT p50 | ITL p50 | J/token |
|---:|---:|---:|---:|---:|
| 1 | 74.2 | 33 ms | **12.6 ms** (≈ 79 tok/s per user) | 0.893 |
| 4 | 208.7 | 178 ms | 14.4 ms | 0.317 |

- **Why it works.** Batch-1 decode is limited by reading the weights. INT4 weights are about a third of the bytes, so the read takes about a third of the time.
  Measured in the same session against FP16 on the same card (42–47 ms per token), INT4 is **about 3.5× faster** (S-011).
- **When not to use it.** Under heavy load: at c=256 the gain is only +2% (809.9 vs 792.0 tok/s).
  Also, **accuracy has not been measured**, so run an eval such as GSM8K before shipping.
- No `--quantization` flag is needed. vLLM reads the AWQ config from the checkpoint and picks the kernel.

## R2: Lowest single-user latency on two T4s, using FP16 with TP=2

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --tensor-parallel-size 2 --dtype float16 --no-enable-prefix-caching --max-model-len 4096
```
Config: `configs/experiments/t4x2_qwen3b_fp16_tp2.yaml`

| c | tok/s | TTFT p50 | ITL p50 | J/token | Power (2 cards) |
|---:|---:|---:|---:|---:|---:|
| 1 | 61.0 | 44 ms | **15.8 ms** | 2.191 | 133.7 W |

- **Why it works.** Each card reads only half the weights per step. The all-reduce traffic over PCIe is a few KB per token (S-012).
- **Trade-off against R1.** This uses full precision, so there is no accuracy risk, but it ties up two GPUs and about twice the power for one user.
  R1 is slightly faster (12.6 vs 15.8 ms) on one card.

## R3: Most chat users within the interactive SLO

Every measured level that meets **TTFT p50 ≤ 1 s and ITL p50 ≤ 50 ms**, taking the highest passing `c` per setup:

| Setup | Highest passing c | tok/s | per GPU | TTFT p50 | TTFT p99 | ITL p50 | J/token | Next level fails on |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1× T4 FP16 | 16 | 339.3 | **339** | 666 ms | 4.2 s | 39.2 ms | 0.194 | ITL (c=32: 54.5 ms) |
| 1× T4 AWQ | 4 | 208.7 | 209 | 178 ms | 3.5 s | 14.4 ms | 0.317 | TTFT (c=16: 1.2 s, unstable) |
| 2× T4 TP=2 | 16 | **549.9** | 275 | 680 ms | 2.0 s | 23.6 ms | 0.246 | TTFT (c=64: 1.65 s) |
| 2× T4 DP=2 | 16 | 380.6 | 190 | 251 ms | 2.4 s | 34.0 ms | 0.347 | ITL (c=64: 52.6 ms) |

**Recipe today:**
- **One T4:** FP16 at up to 16 concurrent users. Use the FP16 command from R2 with `--tensor-parallel-size 1`, or run `configs/experiments/t4_qwen3b_fp16.yaml`.
- **Two T4s:** TP=2 at up to 16 users, which is the **best measured** setup at 549.9 tok/s.

**Open point that could change the two-GPU answer.** The sweep jumps from c=16 to c=64 on two GPUs, so c=32 was never measured.
DP=2 scaled almost perfectly in TN3: at concurrency c it matched two single cards running c/2 each, within 8% (S-012). So DP=2 at c=32 should look like two single cards at c=16 each, about **680 tok/s at ~39 ms ITL**, which would beat TP=2.
That is a prediction, not a measurement. The configs `t4x2_qwen3b_fp16_{tp2,dp2}_c32.yaml` measure it in about 30 minutes on Kaggle.

**Enforcing it in production.** The benchmark capped concurrency on the client side. On a server, the equivalent is admission control: cap the running batch with `--max-num-seqs N` and let extra requests queue. Queued requests add to TTFT. We did not benchmark the server-side cap.

**About TTFT p99.** It is 2–4 s even when the median looks fine. At low load the cause is long prompts, not queueing: about 0.83 ms per prompt token of prefill on a T4 (S-009).
A p99 TTFT SLO has to be set together with a maximum prompt length.

## R4: Maximum throughput or lowest energy per token, no latency target

| Setup | c | tok/s | J/token | TTFT p99 | ITL p50 |
|---|---:|---:|---:|---:|---:|
| 2× T4 DP=2 | 256 | **1,376.0** | 0.095 | 14.3 s | 114.4 ms |
| 2× T4 TP=2 | 256 | 1,178.5 | 0.114 | 22.1 s | 145.3 ms |
| 1× T4 AWQ | 256 | 809.9 | **0.081** | 29.9 s | 190.9 ms |
| 1× T4 FP16 | 256 | 792.0 | 0.082 | 30.1 s | 201.6 ms |

- **Throughput: DP=2.** Run two independent servers behind a load balancer.
  The benchmark started one server per card and round-robined requests on the client side; vLLM's own `--data-parallel-size` was not used.
  ```bash
  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
    --dtype float16 --no-enable-prefix-caching --max-model-len 4096 --port 8000 &
  CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
    --dtype float16 --no-enable-prefix-caching --max-model-len 4096 --port 8001 &
  ```
  Config: `configs/experiments/t4x2_qwen3b_fp16_dp2.yaml`. Once compute-bound, DP beats TP by 17% because the replicas never synchronise (S-012).
- **Energy: one card loaded to c=256.** The T4 draws about 65 W at every load (S-006), so J/token only falls as concurrency per card rises.
  Two cards at c=128 each (DP=2) cost more energy per token than one card at c=256. More GPUs buy throughput, not efficiency.
- INT4 adds nothing here (+2%). Pick FP16 unless you also need R1's single-user latency from the same deployment.

## R5: Many users sharing a long system prompt, so keep prefix caching on

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --tensor-parallel-size 1 --dtype float16 --enable-prefix-caching --max-model-len 4096
```
Configs: `t4_qwen3b_fp16_prefix_on.yaml` vs `t4_qwen3b_fp16_prefix_off.yaml` (1,024 shared tokens followed by a ShareGPT question).

| c | tok/s off → on | TTFT p50 off → on | TTFT p99 off → on | J/token off → on |
|---:|---:|---:|---:|---:|
| 16 | 171.2 → 254.9 | 3.4 → 1.7 s | 10.0 → 4.4 s | 0.383 → 0.266 |
| 64 | 225.6 → 399.0 | 4.5 → 3.9 s | 44.2 → 14.1 s | 0.286 → 0.169 |
| 256 | 214.9 → **442.9** | 152.5 → **9.4 s** | 244.1 → 53.3 s | 0.300 → **0.151** |

- **Result.** The cache hit rate was 81–84%. At c=256, 0 requests waited instead of about 125, throughput doubled and energy per token halved (S-014).
- Prefix caching is **on by default** in vLLM 0.33. The only lesson here is *don't turn it off*. Our other benchmarks did turn it off, so that repeats could not reuse each other's cache (D-005).
- The shared part must be **byte-identical and at the start** of the prompt. Put per-user content after it.
- Even with the cache on, this workload does not meet the interactive SLO above c≈4 on a T4: the questions still need prefill, and 256 users share every step.

## R6: Long, different prompts per user, where KV-cache is the limit

There is no clean recipe yet; this is a sizing rule plus a warning.
With prefix caching off and prompts around 1,220 tokens, the KV pool (194,160 tokens) held only about 131 full requests.
At c=256 half the users queued, TTFT p50 reached 152 s, and throughput fell to 215 tok/s (S-014). Short chat prompts never hit this limit: KV usage peaked at 44% at c=256 (S-008).

- **Sizing rule:** max concurrent users ≈ KV pool tokens ÷ (prompt + output tokens). Read the KV pool size from the server log (`GPU KV cache size`). It varies by session and by flags (S-003).
- **Ways to get more KV, not measured with long prompts:**
  - AWQ frees weight memory: pool 296,288 tokens.
  - TP=2 spreads the load over two cards: pool 557,248 tokens.
  - A shorter `--max-model-len` does not grow the pool, but it does bound the worst case.

---

## H100 recipes (TN4: 1× H100 SXM 80 GB, Secure Cloud USD 3.99/h, 2026-11-18)

Environment per engine (from each run's `env.json`): vLLM 0.33.0 + torch 2.14.0; SGLang 0.5.23 + torch 2.14.1;
TensorRT-LLM 1.2.1 + torch 2.9.1+cu130 (plus `pyarrow<20`); Python 3.12.3, CUDA 13.0, driver 595.112.07.
Models: `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` (BF16), `nvidia/Qwen3-8B-FP8` @
`2cebc4c89e25abc17668c81b01dceaf3d8b914d5` (FP8 weights, activations and KV-cache, D-011). Same ShareGPT workload as T4.
Latency per token is **TPOT** (per-request median), because ITL breaks above ~20k tok/s (S-016).

**On H100 the T4 interactive SLO (TTFT p50 ≤ 1 s, per-token ≤ 50 ms) holds at every measured level up to c=256 on all
three engines**, so it no longer picks the operating point. Throughput and cost per token do.

### R7: Highest throughput and lowest cost per token: FP8, vLLM or SGLang, c=256

```bash
vllm serve nvidia/Qwen3-8B-FP8 --revision 2cebc4c89e25abc17668c81b01dceaf3d8b914d5 \
  --tensor-parallel-size 1 --kv-cache-dtype fp8 --no-enable-prefix-caching --max-model-len 4096
# or
python -m sglang.launch_server --model-path nvidia/Qwen3-8B-FP8 --revision 2cebc4c89e25abc17668c81b01dceaf3d8b914d5 \
  --tp-size 1 --kv-cache-dtype fp8_e4m3 --disable-radix-cache --context-length 4096
```
Configs: `h100_qwen3-8b_{vllm,sglang}_fp8.yaml`.

| Setup at c=256 | tok/s | TTFT p50 / p99 | TPOT p50 | J/token | USD per 1M output tokens (at 3.99/h) |
|---|---:|---:|---:|---:|---:|
| vLLM FP8 | **16,432** | 0.69 / 1.21 s | 12.7 ms | **0.039** | **0.067** |
| SGLang FP8 | 16,361 | 0.72 / 1.22 s | 12.7 ms | 0.039 | 0.068 |
| vLLM BF16 | 12,398 | 0.96 / 1.66 s | 17.0 ms | 0.054 | 0.089 |
| SGLang BF16 | 12,333 | 0.90 / 1.64 s | 17.2 ms | 0.055 | 0.090 |
| TRT-LLM FP8 (defaults, see R9) | 8,530 | 0.58 / 1.40 s | 27.5 ms | 0.044 | 0.130 |
| TRT-LLM BF16 (defaults, see R9) | 8,882 | 0.90 / 1.87 s | 25.3 ms | 0.064 | 0.125 |

- FP8 is +33% throughput over BF16 at c=256 and +53% at c=1. Unlike INT4 on T4, it still helps when compute-bound,
  because Hopper has FP8 tensor cores (S-017). Accuracy of FP8 vs BF16 was **not measured**.
- vLLM and SGLang are within 0.6% at c=256. Pick on features and operations, not speed (S-017).
- Each user still gets ~79 tok/s (12.7 ms per token) at 256 concurrent users. The H100 was still scaling at 256, so the
  real ceiling is higher and unmeasured.
- At c=256 BF16 runs at the 700 W power cap (clock 1,635–1,680 MHz); FP8 does the same work at ~671 W (S-019).

### R8: Lowest single-user latency on H100: FP8, any engine

| c=1 | tok/s | TPOT p50 | TTFT p50 | J/token |
|---|---:|---:|---:|---:|
| TRT-LLM FP8 | 222.1 | **4.36 ms** | 32 ms | 1.852 |
| vLLM FP8 | **222.6** | 4.41 ms | **16 ms** | 1.866 |
| SGLang FP8 | 214.6 | 4.59 ms | 15 ms | 1.899 |
| vLLM BF16 | 145.8 | 6.78 ms | 18 ms | 3.117 |

All FP8 engines are within 5%. TRT-LLM's first token took about twice as long at c=1 (30–32 vs 15–18 ms). Single-user
serving is the most expensive way to use an H100: USD 4.98 per million tokens on vLLM FP8, 74× the c=256 cost.

### R9: TensorRT-LLM above ~128 concurrent requests: size the CUDA graphs (untested)

With defaults, TRT-LLM 1.2.1 matched vLLM up to c=64 and then fell to about half at c=256, with the GPU only 37% busy
(S-018). Its log shows CUDA graphs captured for batch sizes 1–32, 64 and 128 only, with padding off. The suspected fix,
**prepared but not run**, is to add to the `--extra_llm_api_options` YAML:
```yaml
cuda_graph_config:
  max_batch_size: 256      # at least the decode batch you expect
  enable_padding: true     # let in-between batch sizes use the next captured graph
```
Until that is measured, treat TRT-LLM's c=256 numbers as "defaults, untuned", not as its capability.

## Things that made results worse (don't)

| Don't | Measured effect |
|---|---|
| `--enforce-eager` | 25% lower throughput at c=1 (25.1 vs 33.4 tok/s), ITL 38.6 vs 28.7 ms, same session (S-007) |
| `--no-enable-prefix-caching` when prompts share a prefix | 2× lower throughput, 16× higher TTFT p50 at c=256 (S-014) |
| Quote batch-1 latency measured in another session or on another card | The same config gave 28.7–51.9 ms per token across sessions (S-007, S-013) |
| Add a second GPU to cut energy per token | DP=2 at c=256: 0.095 J/token vs one card at c=256: 0.082 (R4) |
| Run TRT-LLM 1.2.1 above batch 128 with default CUDA graph sizes | Half of vLLM's throughput at c=256, GPU 37% busy (R9, S-018) |
| Read ITL from one Python client above ~20k tok/s | ITL p50 measured as 0.0 ms; use TPOT and the server's counters (S-016) |

## Not measured yet

- Accuracy of AWQ-INT4 against FP16.
- c=32 on two GPUs (R3 open point).
- Long *unique* prompts with AWQ or TP=2 (R6).
- Server-side `--max-num-seqs` as admission control.
- Accuracy of FP8 against BF16 on H100.
- TRT-LLM with `cuda_graph_config` sized for c=256 (R9).
- H100 above c=256, and H100 with TP/DP or NVLink (one GPU only).
- Per-engine lock files on H100: lost to a `pip freeze` mistake (S-005); versions are in each run's `env.json`.
