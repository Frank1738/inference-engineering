# 01 — Foundations

**Goal:** understand what actually happens when a model produces a token, and
be able to run one myself in PyTorch.

Don't try to master ML here. The aim is the mental model an inference engineer
needs: tensors move through memory, matmuls run on hardware, and the shape of
the transformer explains why inference is slow in the ways it is.

## What to learn here

### Python
Just the parts that matter for this work: generators, context managers,
`dataclasses`, typing, virtual environments, `pip` vs `uv`, and timing code
properly (`time.perf_counter`, not `time.time`).

### NumPy
Arrays, dtypes, shapes, broadcasting, matrix multiplication. Understand that a
tensor is a block of memory plus a shape and a stride — this is the idea
everything else is built on.

### PyTorch
Tensors, `dtype` (`float32` / `float16` / `bfloat16`), devices, `.to("cuda")`,
`torch.no_grad()` / `inference_mode()`, loading a model from Hugging Face, and
running a forward pass. Learn why `torch.cuda.synchronize()` matters before you
time anything on GPU.

### CPU vs GPU
Why a GPU is faster: thousands of cores, high memory bandwidth. Why it isn't
always: kernel launch overhead, data transfer over PCIe, small batches.
Learn the distinction between **compute-bound** and **memory-bandwidth-bound**
work — LLM decoding is usually the latter, and that one fact explains most of
inference optimization.

### Transformers
Tokenization, embeddings, the stack of decoder blocks, the LM head, and how a
next-token distribution becomes a token (greedy, temperature, top-p).
Understand the two phases:
- **Prefill** — process the whole prompt at once, compute-heavy, parallel.
- **Decode** — one token at a time, memory-bandwidth-heavy, sequential.

### Attention
Q, K, V; the `softmax(QKᵀ/√d)V` formula; multi-head attention; causal masking.
Understand why attention cost grows with sequence length, and why that hurts
long contexts.

### KV Cache
The single most important inference concept. Why recomputing K and V for every
past token each step is wasteful, what gets cached, how to size the cache
(`2 × layers × heads × head_dim × seq_len × batch × bytes_per_element`), and
why the KV cache — not the weights — is what usually runs you out of VRAM.

## Suggested experiments

- Multiply two large matrices on CPU and GPU. Time both. Explain the gap.
- Load a small model (e.g. GPT-2 or TinyLlama) and generate 100 tokens.
- Generate with and without the KV cache (`use_cache=False`). Compare speed.
- Print tensor shapes at each step of a forward pass.
- Compute by hand how much VRAM the KV cache needs at 512 vs 4096 tokens.

## What goes in this folder

Scratch scripts and notebooks — `tensors.py`, `pytorch-basics.ipynb`,
`kv-cache-math.md`, whatever I actually run.

## What I learned

_(fill in as I go)_

## Experiments

| What | Setup | Result |
|------|-------|--------|
|      |       |        |

## Notes

_(gotchas, things that confused me, links worth keeping)_
