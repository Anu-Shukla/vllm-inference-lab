# 03 - Concurrency Benchmark

## What I did

Wrote an async benchmark script:

```text
scripts/benchmark_concurrency.py
```

It sends multiple streaming chat requests at the same time and measures:

```text
TTFT
total latency
completion tokens
per-request output tok/s
aggregate output tok/s
```

The workload is intentionally identical for every request so the first test isolates concurrency/batching behavior.

Prompt:

```text
Explain in one sentence why LLM decode is memory-bound.
```

Command:

```bash
python scripts/benchmark_concurrency.py \
  --concurrency 1 2 4 8 16 32 \
  --requests-per-level 32 \
  --max-tokens 64 \
  --output results/concurrency_baseline.json
```

## Small fix

The first version of the script computed aggregate throughput incorrectly. It divided by max request latency instead of the full benchmark wall time.

Fixed it to use:

```text
total completion tokens / benchmark wall time
```

Then reran the benchmark.

## Result

Final summary:

```text
conc  TTFT p50  TTFT p95  lat p50   lat p95   agg tok/s  per-req tok/s p50
1     10.9 ms   11.2 ms   175.4 ms  176.3 ms  210        211.0
2     10.8 ms   11.1 ms   196.0 ms  196.2 ms  378        188.8
4     11.3 ms   11.6 ms   201.0 ms  201.2 ms  736        184.1
8     17.4 ms   17.9 ms   208.5 ms  209.5 ms  1414       181.1
16    23.8 ms   24.2 ms   218.4 ms  218.7 ms  2684       169.9
32    40.9 ms   42.2 ms   229.6 ms  231.0 ms  5108       163.5
```

Saved result:

```text
results/concurrency_baseline.json
```

## Notes

Latency here means request start to final token received.

TTFT means request start to first token received.

Main takeaway:

```text
higher concurrency -> much higher total throughput
higher concurrency -> worse TTFT and per-request latency
```

This is the first clear vLLM serving tradeoff in the lab.
