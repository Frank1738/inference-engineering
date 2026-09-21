# 05 — Final Project

**Goal:** one artifact that proves the whole journey. Something a stranger can
read in five minutes and conclude: this person does inference engineering.

Everything before this was practice. This is the portfolio piece.

## The project

### 1. Build a benchmark CLI
A real tool, not a script. Something like:

```
llmbench run --model <model> --backend <transformers|vllm|onnx|trtllm> \
             --prompt-len 512 --max-tokens 128 --concurrency 1,8,32 \
             --repeat 5 --out results/run-01.json
```

It should:
- warm up, then measure
- record TTFT, per-request tok/s, total tok/s, P50/P95/P99, peak VRAM
- capture the full environment (GPU, driver, CUDA, versions, model, dtype)
  into every result file, so results stay interpretable months later
- emit machine-readable JSON/CSV **and** a human-readable summary
- be reproducible: same command, same numbers, within noise

### 2. Optimize a model
Take one model and push it, using stage 03. Quantization, the right attention
kernel, the right engine, sensible batching. Keep every intermediate
measurement — the path matters as much as the destination, and the failed
attempts are the most convincing part.

### 3. Document the results
A write-up that includes:
- the hardware and exact setup
- the baseline
- each change, what it cost, and what it bought
- plots (throughput vs concurrency, latency percentiles, VRAM vs context)
- **the things that didn't work**, and your explanation why
- a clear recommendation for a given workload

### 4. Publish
Push the CLI, the raw results, and the write-up. Make the top-level README of
this repo point at it. Optionally write it up as a blog post — being able to
explain it in prose is half the skill.

## What makes this good instead of average

- Reproducible: someone else can run your command and get your numbers.
- Honest: includes regressions, noise, and uncertainty.
- Explained: every result has a *why* rooted in bandwidth, cache, or batching.
- Decision-oriented: ends with "use X when Y", not just a table.

## What goes in this folder

The CLI source, `results/`, plots, and the final write-up.

## Status

- [ ] CLI scaffold
- [ ] Measurement core (TTFT, tok/s, percentiles, VRAM)
- [ ] Multi-backend support
- [ ] Optimization run
- [ ] Plots
- [ ] Write-up
- [ ] Published

## Notes

_(scope decisions, what I cut, what I'd do next)_
