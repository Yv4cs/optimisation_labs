# Learning log

Target: explain every "Tom" topic without notes by 20/11. Tick a box only after passing that day's quiz.

## Syllabus

### Must know (Tom will ask)
- [ ] Token, parameters, weight memory = params × bytes per param — 13/11
- [ ] Prefill vs decode; compute-bound vs memory-bandwidth-bound — 13/11
- [ ] TTFT, ITL/TPOT, end-to-end latency, throughput, p50 vs p99 — 14/11
- [ ] KV-cache: what it stores, size formula, GQA, why it limits batch size — 14/11
- [ ] PagedAttention, continuous batching, chunked prefill — 15/11
- [ ] Energy: watts vs joules, joules/token, cost per 1M tokens — 15/11
- [ ] Quantization: FP16, BF16, FP8, INT8, INT4; weight-only (AWQ/GPTQ) vs weights+activations — 16/11
- [ ] Prefix caching / RadixAttention — 16/11
- [ ] TP, PP, DP, EP; all-reduce; NCCL; PCIe vs NVLink — 17/11
- [ ] Why replicating can beat distributing ("Less is More") — 17/11
- [ ] vLLM vs SGLang vs TensorRT-LLM: design and strengths of each — 17/11
- [ ] T4 vs H100: memory bandwidth, tensor cores, FP8 support — 18/11
- [ ] Speculative decoding — 18/11
- [ ] Accuracy check after quantization (lm-evaluation-harness, GSM8K) — 18/11

### Good to know
- [ ] MoE and DP-attention (DeepSeek-style serving)
- [ ] Roofline model and arithmetic intensity
- [ ] Goodput and SLOs (throughput that still meets a latency target)
- [ ] Little's law: in-flight requests = arrival rate × latency
- [ ] Disaggregated prefill/decode

## Quiz log

| Date | Milestone | Question | My answer | Correct? | Gap to review |
|---|---|---|---|---|---|
