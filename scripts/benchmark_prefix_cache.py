import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SHARED_PREFIX = """
You are helping analyze LLM inference performance. Use the following context:

LLM serving has two major phases: prefill and decode. Prefill processes the
input prompt and is usually compute-heavy because it performs large matrix
multiplications over all prompt tokens. Decode generates one token at a time and
is often memory-bandwidth-bound because each step reloads model weights and
reads the KV cache. vLLM improves serving throughput with continuous batching,
PagedAttention, prefix caching, and careful GPU memory management. Prefix
caching reuses previously computed KV cache blocks when multiple requests share
the same prompt prefix.

Answer the final question briefly.
""".strip()

QUESTIONS = [
    "Why can prefix caching reduce TTFT?",
    "Why is decode often memory-bandwidth-bound?",
    "What does PagedAttention optimize?",
    "Why does batching improve throughput?",
    "What tradeoff appears as concurrency increases?",
    "Why does KV cache memory matter for serving?",
    "What is the difference between prefill and decode?",
    "Why can quantization improve decode throughput?",
]

UNRELATED_PROMPTS = [
    "Explain in one sentence why ocean tides happen.",
    "Explain in one sentence how photosynthesis works.",
    "Explain in one sentence what a hash table does.",
    "Explain in one sentence why airplanes generate lift.",
    "Explain in one sentence what TCP congestion control does.",
    "Explain in one sentence why GPUs are good for matrix multiplication.",
    "Explain in one sentence what entropy means in information theory.",
    "Explain in one sentence why caches improve CPU performance.",
]


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = round((pct / 100) * (len(ordered) - 1))
    return ordered[index]


def summarize(name, values):
    values = [v for v in values if v is not None]
    return {
        f"{name}_mean": statistics.mean(values) if values else None,
        f"{name}_p50": percentile(values, 50),
        f"{name}_p95": percentile(values, 95),
        f"{name}_min": min(values) if values else None,
        f"{name}_max": max(values) if values else None,
    }


def make_prompt(workload, index):
    if workload == "shared_prefix":
        question = QUESTIONS[index % len(QUESTIONS)]
        return f"{SHARED_PREFIX}\n\nQuestion: {question}"
    if workload == "unrelated":
        return UNRELATED_PROMPTS[index % len(UNRELATED_PROMPTS)]
    raise ValueError(f"unknown workload: {workload}")


async def run_one(session, request_id, prompt, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_time = None
    usage = None
    generated_chars = 0

    async with session.post(URL, json=payload) as response:
        response.raise_for_status()

        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

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
            generated_chars += len(delta)

    end = time.perf_counter()
    total_latency = end - start
    ttft = None if first_token_time is None else first_token_time - start
    completion_tokens = None if usage is None else usage.get("completion_tokens")

    return {
        "request_id": request_id,
        "ttft_sec": ttft,
        "total_latency_sec": total_latency,
        "completion_tokens": completion_tokens,
        "generated_chars": generated_chars,
        "output_tokens_per_sec": None
        if not completion_tokens
        else completion_tokens / total_latency,
        "usage": usage,
    }


async def run_benchmark(args):
    prompts = [make_prompt(args.workload, i) for i in range(args.requests)]
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None)

    started = time.perf_counter()
    results = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for offset in range(0, args.requests, args.concurrency):
            batch = prompts[offset : offset + args.concurrency]
            tasks = [
                run_one(session, offset + i + 1, prompt, args.max_tokens)
                for i, prompt in enumerate(batch)
            ]
            results.extend(await asyncio.gather(*tasks))

    ended = time.perf_counter()
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in results)
    ttfts = [r["ttft_sec"] for r in results]
    latencies = [r["total_latency_sec"] for r in results]
    per_request_tok_s = [r["output_tokens_per_sec"] for r in results]

    summary = {
        "workload": args.workload,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "benchmark_wall_time_sec": ended - started,
        "total_completion_tokens": total_completion_tokens,
        "aggregate_output_tokens_per_sec": total_completion_tokens / (ended - started),
        **summarize("ttft_sec", ttfts),
        **summarize("latency_sec", latencies),
        **summarize("per_request_output_tokens_per_sec", per_request_tok_s),
    }

    return {
        "model": MODEL,
        "summary": summary,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["shared_prefix", "unrelated"], required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = asyncio.run(run_benchmark(args))
    print(json.dumps(output["summary"], indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
