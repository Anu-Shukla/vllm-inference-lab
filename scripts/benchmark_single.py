import argparse
import json
import statistics
import time

import requests

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain in one sentence why LLM decode is memory-bound."


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = round((pct / 100) * (len(ordered) - 1))
    return ordered[index]


def run_once(max_tokens):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_time = None
    usage = None

    with requests.post(URL, json=payload, stream=True) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(chunk_size=1):
            if not raw_line:
                continue

            line = raw_line.decode("utf-8")
            if line.startswith("data: "):
                line = line[len("data: "):]
            if line == "[DONE]":
                break

            chunk = json.loads(line)

            if chunk.get("usage") is not None:
                usage = chunk["usage"]
                continue

            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta and first_token_time is None:
                first_token_time = time.perf_counter()

    end = time.perf_counter()
    total_latency = end - start
    ttft = None if first_token_time is None else first_token_time - start
    completion_tokens = None if usage is None else usage.get("completion_tokens")

    return {
        "ttft_sec": ttft,
        "total_latency_sec": total_latency,
        "completion_tokens": completion_tokens,
        "output_tokens_per_sec": None
        if not completion_tokens
        else completion_tokens / total_latency,
        "usage": usage,
    }


def summarize(name, values):
    values = [v for v in values if v is not None]
    return {
        f"{name}_mean": statistics.mean(values) if values else None,
        f"{name}_p50": percentile(values, 50),
        f"{name}_p95": percentile(values, 95),
        f"{name}_min": min(values) if values else None,
        f"{name}_max": max(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    results = []
    for i in range(args.runs):
        result = run_once(args.max_tokens)
        results.append(result)
        print(
            f"run={i + 1:02d} "
            f"ttft={result['ttft_sec']:.4f}s "
            f"latency={result['total_latency_sec']:.4f}s "
            f"out_tok_s={result['output_tokens_per_sec']:.2f}"
        )

    ttfts = [r["ttft_sec"] for r in results]
    latencies = [r["total_latency_sec"] for r in results]
    output_tok_s = [r["output_tokens_per_sec"] for r in results]

    summary = {
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        **summarize("ttft_sec", ttfts),
        **summarize("latency_sec", latencies),
        **summarize("output_tokens_per_sec", output_tok_s),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
