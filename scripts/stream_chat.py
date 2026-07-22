import argparse
import json
import time

import requests

URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Explain in one sentence why LLM decode is memory-bound.",
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_time = None
    text = ""
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

            text += delta
            print(delta, end="", flush=True)

    end = time.perf_counter()

    total_latency = end - start
    ttft = None if first_token_time is None else first_token_time - start
    completion_tokens = None if usage is None else usage.get("completion_tokens")
    total_tokens = None if usage is None else usage.get("total_tokens")

    metrics = {
        "model": args.model,
        "ttft_sec": ttft,
        "total_latency_sec": total_latency,
        "generated_chars": len(text),
        "usage": usage,
        "output_tokens_per_sec": None
        if not completion_tokens
        else completion_tokens / total_latency,
        "total_tokens_per_sec": None if not total_tokens else total_tokens / total_latency,
    }

    print()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
