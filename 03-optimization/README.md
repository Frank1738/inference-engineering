# 03 — Optimization

**Goal:** make inference measurably faster or cheaper, and be able to explain
exactly which bottleneck each technique attacks.

Never apply a technique before measuring. Every entry here should be paired
with a before/after from stage 02 — and with an honest note when it made
things worse.

## The one idea behind all of this

LLM **decode** is memory-bandwidth-bound: for each token you read the entire
model's weights from VRAM and do relatively little math with them. So most wins
come from either **moving fewer bytes** (quantization, better cache layout) or
**doing more useful work per byte moved** (batching, fused kernels).

## What to learn here

### Quantization
Representing weights in fewer bits: FP16/BF16 → INT8 → INT4. Fewer bytes means
less bandwidth and less VRAM, so it's usually a speedup *and* a memory win.
Learn the vocabulary: post-training quantization vs quantization-aware
training; weight-only vs weight+activation; GPTQ, AWQ, GGUF, bitsandbytes.
Always measure the quality cost, not just the speed win.

### GPU memory and bandwidth
The memory hierarchy: HBM/VRAM → L2 → SRAM/shared memory → registers. Know your
GPU's peak bandwidth (GB/s) and use it to compute a roofline: theoretical
max tokens/sec ≈ bandwidth ÷ bytes-read-per-token. If you're near that ceiling,
stop optimizing kernels and start reducing bytes.

### FlashAttention
Attention done without materializing the full N×N score matrix — tiled,
fused, kept in SRAM. Saves memory and time, most dramatically at long context.
Learn what "IO-aware kernel" means, and check whether your stack is actually
using it (it's often available but off).

### Continuous batching
Static batching wastes the GPU: everyone waits for the longest sequence.
Continuous (in-flight) batching swaps finished sequences out and new ones in
every step. This is the biggest throughput win in real serving, and it's why
vLLM and TGI exist. Pair it with **PagedAttention** — KV cache in fixed-size
pages instead of one contiguous block, which removes the fragmentation that
otherwise caps concurrency.

### Speculative decoding
A small draft model proposes k tokens; the big model verifies them in one
pass. Accepted tokens are free-ish. Exploits the fact that verification is
parallel while generation is sequential. Learn what acceptance rate you need
for it to pay off, and when it doesn't.

### Profiling
Stop guessing. Learn `torch.profiler`, Nsight Systems, and `nvidia-smi dmon`.
Read a trace: which kernels dominate, where the gaps are, whether the GPU is
even busy. Check GPU utilization *and* memory bandwidth utilization — high util
with low bandwidth means you're launch-bound, not compute-bound.

## Also worth touching

- `torch.compile` and CUDA graphs (kernel launch overhead at small batch).
- Paged / chunked prefill, and prefix caching for shared system prompts.
- Tensor parallelism, when one GPU isn't enough.

## Suggested experiments

- Quantize one model to 8-bit and 4-bit. Table: VRAM, tok/s, and an output
  quality spot-check.
- Compute the roofline for your GPU. Compare to measured tok/s. Explain the gap.
- Toggle FlashAttention/SDPA backends at 512 vs 8192 context. Measure.
- Profile one generation call. Find the top 5 kernels by time.
- Try `torch.compile`. Measure compile cost vs steady-state gain.

## What goes in this folder

Optimization experiments, profiler traces, before/after tables, plots.

## What I learned

_(fill in as I go)_

## Before / after

| Technique | Baseline | After | Δ tok/s | Δ VRAM | Quality cost |
|-----------|----------|-------|---------|--------|--------------|
|           |          |       |         |        |              |

## Notes

_(what didn't work, and why I think it didn't)_
