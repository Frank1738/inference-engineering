# 04 — Serving

**Goal:** run the same model behind four different engines, and be able to say
which one to pick for a given situation — with numbers, not vibes.

A model that runs in a notebook isn't a service. Serving adds concurrency,
queuing, streaming, memory limits, and failure modes. This stage is where the
job title stops being theoretical.

## What to learn here

### Transformers (the baseline)
Hugging Face `transformers` + `generate()`, maybe wrapped in FastAPI. Slow and
naive under load — no continuous batching, no paged KV cache — but it's the
reference implementation and the number every other engine has to beat. Build
this first so you have something to compare against.

### vLLM
The default answer for GPU serving today. PagedAttention plus continuous
batching gives large throughput gains at high concurrency. Learn the OpenAI-
compatible server, `--max-model-len`, `--gpu-memory-utilization`,
tensor parallelism, and how to read its scheduler metrics. Understand *why*
it's fast — that's the interview question.

### ONNX Runtime
Export the model to ONNX, run it through a graph-optimizing runtime. Strong on
CPU and on non-NVIDIA hardware, and good for smaller models and edge
deployment. Learn about export quirks, opset versions, and why dynamic shapes
are awkward.

### TensorRT-LLM
NVIDIA's compiled approach: build a hardware-specific engine ahead of time,
with fused kernels and aggressive optimization. Usually the fastest on NVIDIA,
at the cost of a slow build step and much less flexibility. Learn the engine
build workflow and when that tradeoff is worth it.

## Beyond the engines

The operational layer that turns an engine into a service:

- **Streaming** responses (SSE) and why TTFT dominates perceived quality.
- **Queuing and admission control** — what happens when requests exceed capacity.
- **Concurrency limits**, timeouts, backpressure.
- **Observability** — latency percentiles, queue depth, KV cache utilization,
  GPU util, tokens/sec per replica.
- **Cost per million tokens** — the metric the business actually cares about.

## Suggested experiments

- Serve the same model on all four. Same prompts, same token count, same
  hardware. Build one comparison table.
- Load-test each at 1 / 8 / 32 concurrent requests. Record throughput and P95.
- Find where each engine falls over, and describe how it fails.
- Compute cost per million tokens for each, using a real GPU hourly rate.
- Write down, in three sentences, when you'd choose each one.

## What goes in this folder

One subfolder per engine, each with setup notes, the exact commands used, and
its results. Note the versions — these projects move fast and numbers age.

## What I learned

_(fill in as I go)_

## Engine comparison

| Engine | Setup effort | TTFT | tok/s @ c=1 | tok/s @ c=32 | P95 @ c=32 | VRAM | Best for |
|--------|--------------|------|-------------|--------------|------------|------|----------|
|        |              |      |             |              |            |      |          |

## Notes

_(install pain, version pins, things that only worked on the third try)_
