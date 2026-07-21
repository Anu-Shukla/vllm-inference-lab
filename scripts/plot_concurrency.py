import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summaries(path):
    data = json.loads(Path(path).read_text())
    return data["summaries"]


def save_plot(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_throughput(summaries, output_dir):
    conc = [s["concurrency"] for s in summaries]
    aggregate = [s["aggregate_output_tokens_per_sec"] for s in summaries]
    per_request = [s["per_request_output_tokens_per_sec_p50"] for s in summaries]

    plt.figure(figsize=(7, 4))
    plt.plot(conc, aggregate, marker="o", label="aggregate output tok/s")
    plt.plot(conc, per_request, marker="o", label="per-request output tok/s p50")
    plt.xscale("log", base=2)
    plt.xticks(conc, [str(c) for c in conc])
    plt.xlabel("concurrency")
    plt.ylabel("tokens/sec")
    plt.title("Throughput vs concurrency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(output_dir / "concurrency_throughput.png")


def plot_ttft(summaries, output_dir):
    conc = [s["concurrency"] for s in summaries]
    p50 = [s["ttft_sec_p50"] * 1000 for s in summaries]
    p95 = [s["ttft_sec_p95"] * 1000 for s in summaries]

    plt.figure(figsize=(7, 4))
    plt.plot(conc, p50, marker="o", label="TTFT p50")
    plt.plot(conc, p95, marker="o", label="TTFT p95")
    plt.xscale("log", base=2)
    plt.xticks(conc, [str(c) for c in conc])
    plt.xlabel("concurrency")
    plt.ylabel("milliseconds")
    plt.title("TTFT vs concurrency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(output_dir / "concurrency_ttft.png")


def plot_latency(summaries, output_dir):
    conc = [s["concurrency"] for s in summaries]
    p50 = [s["latency_sec_p50"] * 1000 for s in summaries]
    p95 = [s["latency_sec_p95"] * 1000 for s in summaries]

    plt.figure(figsize=(7, 4))
    plt.plot(conc, p50, marker="o", label="latency p50")
    plt.plot(conc, p95, marker="o", label="latency p95")
    plt.xscale("log", base=2)
    plt.xticks(conc, [str(c) for c in conc])
    plt.xlabel("concurrency")
    plt.ylabel("milliseconds")
    plt.title("End-to-end latency vs concurrency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(output_dir / "concurrency_latency.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/concurrency_baseline.json")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    summaries = load_summaries(args.input)
    output_dir = Path(args.output_dir)

    plot_throughput(summaries, output_dir)
    plot_ttft(summaries, output_dir)
    plot_latency(summaries, output_dir)

    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
