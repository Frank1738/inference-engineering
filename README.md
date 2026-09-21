# Inference Engineering Roadmap



### [01 — Foundations](01-foundations/)
- [ ] Python
- [ ] NumPy
- [ ] PyTorch
- [ ] CPU vs GPU
- [ ] Transformers
- [ ] Attention
- [ ] KV Cache

### [02 — Benchmarking](02-benchmarking/)
- [ ] Run LLM locally
- [ ] Measure TTFT
- [ ] Measure tokens/sec
- [ ] Measure VRAM
- [ ] Measure P50/P95
- [ ] Benchmark batching

### [03 — Optimization](03-optimization/)
- [ ] Quantization
- [ ] GPU memory/bandwidth
- [ ] FlashAttention
- [ ] Continuous batching
- [ ] Speculative decoding
- [ ] Profiling

### [04 — Serving](04-serving/)
- [ ] Transformers
- [ ] vLLM
- [ ] ONNX Runtime
- [ ] TensorRT-LLM

### [05 — Final Project](05-project/)
- [ ] Build benchmark CLI
- [ ] Optimize model
- [ ] Document results
- [ ] Publish benchmarks

## How I work in here

1. Pick the next unchecked item.
2. Read/watch just enough to attempt it.
3. Write code in that stage's folder.
4. Run it and record real numbers (hardware, model, settings, result).
5. Update that folder's README with what I learned and what surprised me.
6. Commit with a message that says what I learned, e.g.
   `git commit -m "Learn PyTorch tensors and GPU inference"`.
7. Tick the box above.

## My hardware

Fill this in once and reference it from every benchmark — numbers are
meaningless without it.

- CPU:
- RAM:
- GPU / VRAM:
- OS:
- CUDA / driver version:
- Python version:

## Progress log

| Date | Stage | What I did |
|------|-------|------------|
|      |       |            |
