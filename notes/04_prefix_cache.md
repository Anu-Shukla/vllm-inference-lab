# 04 - Prefix Caching

## What I did

Wrote:

```text
scripts/benchmark_prefix_cache.py
```

The script tests two workloads:

```text
shared_prefix: same long prefix, different final questions
unrelated: different short prompts
```

I ran each workload twice:

```text
prefix caching enabled
prefix caching disabled
```

Enabled was the default server config. Disabled server command added:

```bash
--no-enable-prefix-caching
```

## Commands

Enabled:

```bash
python scripts/benchmark_prefix_cache.py \
  --workload shared_prefix \
  --requests 32 \
  --concurrency 8 \
  --max-tokens 64 \
  --output results/prefix_cache_enabled_shared.json

python scripts/benchmark_prefix_cache.py \
  --workload unrelated \
  --requests 32 \
  --concurrency 8 \
  --max-tokens 64 \
  --output results/prefix_cache_enabled_unrelated.json
```

Disabled:

```bash
python scripts/benchmark_prefix_cache.py \
  --workload shared_prefix \
  --requests 32 \
  --concurrency 8 \
  --max-tokens 64 \
  --output results/prefix_cache_disabled_shared.json

python scripts/benchmark_prefix_cache.py \
  --workload unrelated \
  --requests 32 \
  --concurrency 8 \
  --max-tokens 64 \
  --output results/prefix_cache_disabled_unrelated.json
```

## Result

```text
workload        cache      TTFT p50  latency p50  agg tok/s
shared_prefix   enabled    19.1 ms   308.2 ms     1194.9
shared_prefix   disabled   67.6 ms   357.6 ms     1054.4
unrelated       enabled    17.1 ms   187.7 ms     1083.2
unrelated       disabled   23.7 ms   195.7 ms     1062.0
```

## Notes

The shared-prefix workload showed the real effect:

```text
TTFT p50: 67.6 ms -> 19.1 ms
latency p50: 357.6 ms -> 308.2 ms
```

The unrelated workload changed much less, which is what I expected.

Main takeaway:

```text
prefix caching helps when many requests share a long prompt prefix
```

This is useful, but I am treating it as a smaller serving feature compared to batching, kernel profiling, and quantization.
