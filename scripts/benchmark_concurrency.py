import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain in one sentence why LLM decode is memory-bound."


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


async def run_one(session, request_id, model, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
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


async def run_concurrency(concurrency, requests_per_level, model, max_tokens):
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = []
        remaining = requests_per_level
        next_request_id = 0

        while remaining > 0:
            batch_size = min(concurrency, remaining)
            tasks = []
            for _ in range(batch_size):
                next_request_id += 1
                tasks.append(run_one(session, next_request_id, model, max_tokens))

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            remaining -= batch_size

    return results


def summarize_concurrency(concurrency, results):
    ttfts = [r["ttft_sec"] for r in results]
    latencies = [r["total_latency_sec"] for r in results]
    per_request_tok_s = [r["output_tokens_per_sec"] for r in results]
    completion_tokens = [r["completion_tokens"] or 0 for r in results]
    total_completion_tokens = sum(completion_tokens)

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "total_completion_tokens": total_completion_tokens,
        **summarize("ttft_sec", ttfts),
        **summarize("latency_sec", latencies),
        **summarize("per_request_output_tokens_per_sec", per_request_tok_s),
    }


async def main_async(args):
    all_summaries = []
    all_results = {}

    for concurrency in args.concurrency:
        print(f"\nconcurrency={concurrency}")
        started = time.perf_counter()
        results = await run_concurrency(
            concurrency=concurrency,
            requests_per_level=args.requests_per_level,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        ended = time.perf_counter()

        summary = summarize_concurrency(concurrency, results)
        summary["benchmark_wall_time_sec"] = ended - started
        summary["aggregate_output_tokens_per_sec"] = (
            summary["total_completion_tokens"] / summary["benchmark_wall_time_sec"]
        )
        all_summaries.append(summary)
        all_results[str(concurrency)] = results

        print(json.dumps(summary, indent=2))

    output = {
        "model": args.model,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "requests_per_level": args.requests_per_level,
        "summaries": all_summaries,
        "results": all_results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nwrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/concurrency_baseline.json")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
