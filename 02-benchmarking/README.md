# 02 — Benchmarking

**Goal:** be able to say, with numbers I trust, how fast a model is and why.

This is the stage that separates "I read about inference" from "I do
inference." An inference engineer's real job starts with measurement — you
can't optimize what you haven't measured, and most published numbers are
measured badly.

## What to learn here

### Run an LLM locally
Get a model running end to end with Hugging Face `transformers`, and ideally
also `llama.cpp` / Ollama for comparison. Note model size, quantization, and
where the weights live (GPU vs CPU vs offloaded).

### TTFT — Time To First Token
How long from request to the first token appearing. Dominated by **prefill**,
so it scales with prompt length. This is what a user perceives as "is it
alive?" Measure it separately from everything else.

### Tokens/sec — throughput
Two different numbers, and conflating them is a classic mistake:
- **Per-request tokens/sec** (decode speed for one user) — latency.
- **Total system tokens/sec** (all concurrent requests) — throughput.
Batching improves the second while often hurting the first.

### VRAM usage
Measure with `torch.cuda.max_memory_allocated()` and `nvidia-smi`. Break it
down: weights + KV cache + activations + framework overhead. Watch the KV cache
grow as context length and batch size grow.

### P50 / P95 / P99 latency
Averages lie. Run many requests, sort the latencies, report percentiles. P95 is
what your unhappiest users actually experience. Always report the number of
samples alongside.

### Batching
Run 1, 2, 4, 8, 16 concurrent requests. Watch throughput climb and per-request
latency degrade. Find the knee of the curve — that tradeoff is the core of
serving economics.

## Benchmarking discipline

Rules to follow every single time, or the numbers are worthless:

- **Warm up first.** Discard the first few runs (CUDA init, kernel autotuning,
  lazy loading).
- **Synchronize before timing on GPU** (`torch.cuda.synchronize()`), otherwise
  you're timing how fast Python queues work, not how fast it runs.
- **Fix the token count.** Set `max_new_tokens` and ignore EOS, or runs aren't
  comparable.
- **Fix the seed and the prompt set.** Vary one thing at a time.
- **Record the environment**: GPU, driver, CUDA, torch version, model, dtype,
  quantization, batch size, context length.
- **Repeat.** Report median and spread, not a single lucky run.

## Suggested experiments

- TTFT vs prompt length: 128 / 512 / 2048 / 8192 tokens. Plot it.
- Tokens/sec vs batch size: 1 / 2 / 4 / 8 / 16. Plot throughput and P95 together.
- VRAM vs context length. Compare measured against your KV-cache math from 01.
- Same model, `float16` vs `bfloat16` vs 8-bit vs 4-bit: speed, VRAM, quality.
- Same model on two backends (transformers vs llama.cpp). Explain the gap.

## What goes in this folder

Benchmark scripts, raw result CSVs, plots, and a `results/` directory. Keep the
raw data — future-me will want to re-plot it.

## What I learned

_(fill in as I go)_

## Results

| Model | Hardware | Setup | TTFT | tok/s | VRAM | P95 |
|-------|----------|-------|------|-------|------|-----|
|       |          |       |      |       |      |     |

## Notes

_(where my measurements were wrong the first time, and why)_
