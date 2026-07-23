import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams

DEFAULT_MODEL = "gpt2"
DEFAULT_PROMPT = "The best GPU programming language is"


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


def generate_once(llm, prompt, max_tokens):
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        detokenize=True,
    )

    start = time.perf_counter()
    outputs = llm.generate([prompt], params, use_tqdm=False)
    end = time.perf_counter()

    output = outputs[0].outputs[0]
    token_ids = list(output.token_ids)

    return {
        "latency_sec": end - start,
        "generated_tokens": len(token_ids),
        "token_ids": token_ids,
        "text": output.text,
    }


def warmup(llm, prompt, warmup_runs):
    for _ in range(warmup_runs):
        generate_once(llm, prompt, 1)
        generate_once(llm, prompt, 15)


def benchmark_config(llm, prompt, max_tokens, runs):
    pair_results = []

    for _ in range(runs):
        one_token = generate_once(llm, prompt, 1)
        full = generate_once(llm, prompt, max_tokens)

        ttft = one_token["latency_sec"]
        total_latency = full["latency_sec"]
        generated_tokens = full["generated_tokens"]
        avg_decode_step = None
        if generated_tokens > 1:
            avg_decode_step = max(0.0, total_latency - ttft) / (generated_tokens - 1)

        pair_results.append(
            {
                "ttft_sec": ttft,
                "total_latency_sec": total_latency,
                "generated_tokens": generated_tokens,
                "avg_decode_step_sec": avg_decode_step,
                "token_ids": full["token_ids"],
                "text": full["text"],
            }
        )

    representative = pair_results[-1]

    return {
        "max_tokens": max_tokens,
        "runs": runs,
        "representative_generated_tokens": representative["generated_tokens"],
        "representative_token_ids": representative["token_ids"],
        "representative_text": representative["text"],
        **summarize("ttft_sec", [r["ttft_sec"] for r in pair_results]),
        **summarize("total_latency_sec", [r["total_latency_sec"] for r in pair_results]),
        **summarize(
            "avg_decode_step_sec", [r["avg_decode_step_sec"] for r in pair_results]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[15, 100])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", default="results/gpt2_vllm_offline.json")
    args = parser.parse_args()

    llm = LLM(model=args.model, dtype=args.dtype)
    warmup(llm, args.prompt, args.warmup_runs)

    summaries = []
    for max_tokens in args.max_tokens:
        print(f"\nmax_tokens={max_tokens}")
        summary = benchmark_config(llm, args.prompt, max_tokens, args.runs)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    output = {
        "model": args.model,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "method": "offline vllm.LLM.generate",
        "greedy_decoding": True,
        "warmup_runs": args.warmup_runs,
        "notes": [
            "TTFT is approximated by timing max_tokens=1 because offline generate is not streaming.",
            "Average decode step is estimated as max(0, full generation latency - TTFT) / (generated_tokens - 1).",
            "vLLM uses its normal cached/PagedAttention path; there is no no-cache comparison here.",
            "This compares full stacks, not isolated kernels.",
        ],
        "summaries": summaries,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
