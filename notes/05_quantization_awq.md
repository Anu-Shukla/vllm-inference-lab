# 05 - AWQ Quantization

## What I did

Tested a pre-quantized AWQ INT4 model against the FP16 baseline.

Baseline model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Quantized model:

```text
Qwen/Qwen2.5-1.5B-Instruct-AWQ
```

AWQ server command:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --dtype float16
```

I also updated the benchmark scripts to accept:

```bash
--model
```

so the same scripts can benchmark FP16 and AWQ.

## Single-request result

FP16 baseline:

```text
TTFT p50:        ~10.8 ms
latency p50:     ~175 ms
output tok/s:    ~211 tok/s
```

AWQ:

```text
TTFT p50:        ~6.2 ms
latency p50:     ~101.7 ms
output tok/s:    ~374 tok/s
```

Roughly:

```text
~1.77x output tok/s
~42% lower latency
```

## Concurrency result

Command:

```bash
python scripts/benchmark_concurrency.py \
  --model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --concurrency 1 2 4 8 16 32 \
  --requests-per-level 32 \
  --max-tokens 64 \
  --output results/concurrency_awq.json
```

Comparison:

```text
conc  FP16 agg tok/s  AWQ agg tok/s  speedup  FP16 lat p50  AWQ lat p50
1     210             372            1.77x    175.4 ms      101.4 ms
2     377             711            1.88x    196.0 ms      106.2 ms
4     736             1411           1.92x    201.0 ms      107.2 ms
8     1414            2715           1.92x    208.5 ms      111.4 ms
16    2684            4863           1.81x    218.4 ms      123.6 ms
32    5108            7261           1.42x    229.6 ms      165.2 ms
```

The speedup is strongest through concurrency 16. At concurrency 32, the speedup drops, probably because another bottleneck starts to dominate.

## VRAM

Measured with:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Idle loaded-server VRAM:

```text
FP16: 20344 MiB
AWQ:  20064 MiB
```

The measured idle VRAM difference was only ~280 MiB.

This is smaller than the raw weight-size difference because vLLM process memory includes more than just model weights:

```text
KV cache reservation
CUDA context
runtime buffers
workspace / compiled kernels
```

The performance gain was much clearer than the idle VRAM difference in this setup.

## Notes

AWQ stores main linear weights in INT4, while activations and other runtime pieces are still usually FP16/BF16.

`--dtype float16` controls the non-quantized/default runtime dtype. It does not mean the AWQ weights stop being INT4.

Main takeaway:

```text
AWQ INT4 gave a large serving speedup on this workload by reducing weight bandwidth pressure during decode.
```
