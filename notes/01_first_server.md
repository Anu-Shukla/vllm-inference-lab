# 01 - First vLLM Server Run

## What I did

Set up the first local vLLM server for the lab using:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Environment:

```text
Python: 3.12.3
uv: 0.11.30
vLLM: 0.25.1
```

I started with the basic serve command:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

The model started loading correctly. vLLM resolved the model as `Qwen2ForCausalLM`, used `FLASH_ATTN` for attention, and defaulted to a max context length of `32768`.

## What broke

Startup failed during engine initialization because FlashInfer tried to compile a sampling kernel and hit a CUDA/CUB compile error:

```text
flashinfer/sampling.cuh: error:
BlockAdjacentDifference<...> has no member "FlagHeads"
```

This was not a model loading issue and not an attention issue. The attention backend was FlashAttention and that part was fine.

The failure was specifically in the sampling path:

```text
logits -> sampler -> next token
```

## Fix

I disabled the FlashInfer sampler and reduced the context/memory settings:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --dtype float16
```

The important fix was:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

The other flags just made the run more reasonable for my local GPU.

## Result

The server started successfully:

```text
Application startup complete.
```

I checked the model endpoint:

```bash
curl http://localhost:8000/v1/models
```

That returned the loaded model and confirmed `max_model_len: 4096`.

Then I sent a test completion:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The main bottleneck in LLM decode is",
    "max_tokens": 32,
    "temperature": 0
  }'
```

Result:

```text
prompt_tokens: 8
completion_tokens: 32
total_tokens: 40
finish_reason: length
```

The first milestone is done: vLLM is serving a local model through the OpenAI-compatible API.

## Notes

OpenAI-compatible just means vLLM exposes the same HTTP request/response format as OpenAI's API. The model is still running locally.

Sampling here means choosing the next token from logits. Since I used `temperature: 0`, this was greedy decoding.

Disabling FlashInfer sampling did not change the decoding policy. It only changed the backend implementation used for the next-token selection step.
