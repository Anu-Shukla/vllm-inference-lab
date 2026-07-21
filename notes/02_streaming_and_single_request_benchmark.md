# 02 - Streaming and Single-Request Benchmark

## What I did

After getting the vLLM server running, I moved from `curl` to small Python clients.

I made:

```text
scripts/simple_completion.py
scripts/stream_chat.py
scripts/benchmark_single.py
```

I also deleted `scripts/stream_completion.py` because the timing looked buffered and did not give a useful TTFT measurement.

## Simple Python Client

First I wrote `simple_completion.py` to hit:

```text
/v1/completions
```

It worked and returned the same kind of result as `curl`:

```text
completion_tokens: 32
prompt_tokens: 8
total_tokens: 40
```

This was just a sanity check that Python could talk to the local server.

## Streaming Client

Then I wrote `stream_chat.py` using:

```text
/v1/chat/completions
```

with:

```json
{
  "stream": true,
  "stream_options": {"include_usage": true},
  "temperature": 0,
  "max_tokens": 64
}
```

I used streaming because TTFT only really makes sense when tokens are streamed back as they are generated.

The prompt was:

```text
Explain in one sentence why LLM decode is memory-bound.
```

One run gave:

```json
{
  "ttft_sec": 0.023314201273024082,
  "total_latency_sec": 0.19146018009632826,
  "generated_chars": 218,
  "usage": {
    "prompt_tokens": 42,
    "total_tokens": 79,
    "completion_tokens": 37
  },
  "output_tokens_per_sec": 193.25167239153544,
  "total_tokens_per_sec": 412.61843564679185
}
```

Important detail: `max_tokens` was 64, but the model only generated 37 completion tokens. `max_tokens` is a cap, not a guarantee.

## Repeated Benchmark

Then I wrote `benchmark_single.py` to run the same streaming request 30 times:

```bash
python scripts/benchmark_single.py --runs 30 --max-tokens 64
```

It records:

```text
TTFT
total latency
output tokens/sec
```

and summarizes mean, p50, p95, min, and max.

## Result

For 30 single-request runs:

```text
completion tokens per request: 37
TTFT p50:        ~10.8 ms
TTFT p95:        ~11.6 ms
latency p50:     ~175 ms
latency p95:     ~180 ms
output tok/s:    ~210 tok/s
```

Full summary:

```json
{
  "runs": 30,
  "max_tokens": 64,
  "ttft_sec_mean": 0.011162824183702468,
  "ttft_sec_p50": 0.01079470943659544,
  "ttft_sec_p95": 0.011628900654613972,
  "ttft_sec_min": 0.00981078390032053,
  "ttft_sec_max": 0.021755476482212543,
  "latency_sec_mean": 0.17593418083464105,
  "latency_sec_p50": 0.1753695672377944,
  "latency_sec_p95": 0.17976410314440727,
  "latency_sec_min": 0.17407564911991358,
  "latency_sec_max": 0.19055444654077291,
  "output_tokens_per_sec_mean": 210.35860135565352,
  "output_tokens_per_sec_p50": 210.9614046082016,
  "output_tokens_per_sec_p95": 212.2641173603036,
  "output_tokens_per_sec_min": 194.1702262617268,
  "output_tokens_per_sec_max": 212.55126829664852
}
```

The first run was slower than the rest:

```text
TTFT:    21.8 ms
latency: 190.6 ms
```

After that, the numbers were very stable. This was probably warmup / cache / CUDA graph / GPU clock behavior.

## Notes

TTFT = time from sending the request to receiving the first generated token.

p95 means 95% of requests were at or below that value.

This is only a single-request baseline. It does not really test vLLM's scheduler or batching yet.

