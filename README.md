# vLLM Inference Lab

Small lab for getting hands-on with vLLM inference serving, benchmarking, quantization, and GPT-2 comparison work.

The goal was to move past just running `vllm serve` and build a reusable benchmark harness for measuring basic LLM serving behavior: TTFT, latency, output tokens/sec, aggregate throughput, concurrency scaling, prefix caching, and AWQ quantization.

## What This Covers

- Local vLLM online serving with the OpenAI-compatible API
- Streaming chat clients and token/latency measurement
- Single-request and concurrent request benchmarks
- Prefix caching enabled/disabled comparison
- AWQ INT4 vs FP16 serving comparison
- Offline vLLM GPT-2 benchmark for comparison with my own CUDA GPT-2 inference engine

## Layout

```text
scripts/   benchmark and client scripts
results/   saved benchmark JSON outputs
plots/     concurrency plots
notes/     short lab notes for each experiment
```

Start with the notes:

```text
notes/01_first_server.md
notes/02_streaming_and_single_request_benchmark.md
notes/03_concurrency_benchmark.md
notes/04_prefix_cache.md
notes/05_quantization_awq.md
```

## Main Scripts

```text
scripts/simple_completion.py
scripts/stream_chat.py
scripts/benchmark_single.py
scripts/benchmark_concurrency.py
scripts/benchmark_prefix_cache.py
scripts/benchmark_gpt2_offline.py
scripts/plot_concurrency.py
```
