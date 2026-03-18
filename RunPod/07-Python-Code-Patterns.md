# Python Code Patterns

## Complete Serverless Handler — Model at Module Level

```python
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load ONCE at module level (not per-request)
MODEL_PATH = "/runpod-volume/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda"
)

def handler(event):
    prompt = event["input"]["prompt"]
    max_tokens = event["input"].get("max_tokens", 256)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"text": text}

runpod.serverless.start({"handler": handler})
```

## OOM-Safe Handler with Recovery

```python
import runpod
import torch
import logging

model = load_model()  # module-level

def handler(event):
    try:
        result = model.generate(event["input"]["prompt"])
        return {"text": result}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        logging.error(f"OOM error: {e}")
        raise RuntimeError("GPU OOM — try shorter prompt or lower max_tokens") from e
    except Exception as e:
        logging.error(f"Handler error: {e}", exc_info=True)
        raise  # re-raise so RunPod marks job FAILED with error message

runpod.serverless.start({"handler": handler})
```

Never use bare `except: pass` — it silently swallows errors and job appears COMPLETED with no output.

## Streaming Handler — TextIteratorStreamer + Background Thread

```python
import runpod
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL_PATH = "/runpod-volume/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cuda")

def handler(event):
    prompt = event["input"]["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=event["input"].get("max_tokens", 256),
    )

    # Generate in background thread so we can yield tokens
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token

    thread.join()

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
```

## Async Generator Streaming Handler

```python
import runpod
from vllm import AsyncLLMEngine, SamplingParams
import asyncio

engine = AsyncLLMEngine.from_engine_args(...)  # module-level

async def handler(event):
    prompt = event["input"]["prompt"]
    sampling_params = SamplingParams(
        max_tokens=event["input"].get("max_tokens", 256),
        temperature=event["input"].get("temperature", 0.7)
    )
    request_id = event["id"]

    async for output in engine.generate(prompt, sampling_params, request_id):
        yield output.outputs[0].text  # yield each partial text

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
```

## Client: `run()` + Polling Loop

```python
import runpod
import time

runpod.api_key = "your_api_key"
endpoint = runpod.Endpoint("your_endpoint_id")

def run_with_polling(payload, poll_interval=1.0, max_wait=300):
    job = endpoint.run(payload)
    print(f"Job submitted: {job.job_id}")

    elapsed = 0
    while elapsed < max_wait:
        status = job.status()
        if status == "COMPLETED":
            return job.output()
        elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"Job {job.job_id} ended with status: {status}")
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Job {job.job_id} did not complete in {max_wait}s")

result = run_with_polling({"prompt": "Explain transformers briefly"})
print(result)
```

## Client: `run_sync()` with Timeout

```python
import runpod

runpod.api_key = "your_api_key"
endpoint = runpod.Endpoint("your_endpoint_id")

# Blocks until done; 90s hard max (SDK limitation)
# Only use for jobs you know complete quickly
result = endpoint.run_sync(
    {"prompt": "hello world", "max_tokens": 100},
    timeout=60
)
print(result)
```

Use `run_sync` only for jobs that reliably complete in under 90 seconds. For longer jobs, use `run()` + polling.

## Client: Async Batch Runner with Semaphore

```python
import runpod
import asyncio

runpod.api_key = "your_api_key"

async def run_job(endpoint, payload, semaphore):
    async with semaphore:
        job = await endpoint.run_async(payload)
        # poll until done
        while True:
            status = await job.status_async()
            if status == "COMPLETED":
                return await job.output_async()
            elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"Job failed: {status}")
            await asyncio.sleep(1)

async def batch_run(payloads, endpoint_id, max_concurrent=5):
    endpoint = runpod.Endpoint(endpoint_id)
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [run_job(endpoint, p, semaphore) for p in payloads]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

prompts = [{"prompt": f"Question {i}"} for i in range(20)]
results = asyncio.run(batch_run(prompts, "your_endpoint_id", max_concurrent=5))
```

## Endpoint Health Check Before Sending Traffic

```python
import runpod
import requests

def check_endpoint_health(endpoint_id, api_key):
    endpoint = runpod.Endpoint(endpoint_id)
    try:
        health = endpoint.health()
        workers = health.get("workers", {})
        if workers.get("idle", 0) + workers.get("running", 0) > 0:
            return True, health
        # Fallback to REST if SDK health returns empty
    except Exception:
        pass

    # REST fallback
    url = f"https://api.runpod.io/v2/{endpoint_id}/health"
    resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
    return resp.status_code == 200, resp.json()

healthy, status = check_endpoint_health("endpoint_id", "api_key")
if not healthy:
    print("Endpoint not ready:", status)
```

## Network Volume Readiness Check

```python
import os
import time
import runpod

VOLUME_PATH = "/runpod-volume"
MODEL_PATH = f"{VOLUME_PATH}/llama-7b"

def wait_for_volume(path, timeout=60, check_interval=2):
    """Wait until network volume is mounted and model directory exists."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path) and os.listdir(path):
            return True
        time.sleep(check_interval)
    raise RuntimeError(f"Volume not ready at {path} after {timeout}s")

# At module level — runs before handler is registered
wait_for_volume(MODEL_PATH)
model = load_model(MODEL_PATH)  # safe to load now

def handler(event):
    return model.generate(event["input"]["prompt"])

runpod.serverless.start({"handler": handler})
```

## SIGTERM Handler for Spot Pod Checkpoint Saves

```python
import signal
import torch
import os
import runpod

CHECKPOINT_PATH = "/runpod-volume/checkpoints"
model = None
optimizer = None
current_step = 0

def save_checkpoint():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    torch.save({
        "step": current_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"{CHECKPOINT_PATH}/checkpoint_step_{current_step}.pt")
    print(f"Checkpoint saved at step {current_step}")

def sigterm_handler(signum, frame):
    print("SIGTERM received — saving checkpoint before shutdown")
    save_checkpoint()
    exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

# Training loop
for step in range(resume_step, total_steps):
    current_step = step
    train_step(model, optimizer, batch)
    if step % 100 == 0:
        save_checkpoint()  # periodic save, not just on interrupt
```

## Docker Build + Push One-liner with Tags

```bash
#!/bin/bash
set -e

IMAGE="yourusername/your-model-server"
VERSION="v1.2"
GIT_SHA=$(git rev-parse --short HEAD)

docker build \
  --platform linux/amd64 \
  -t ${IMAGE}:latest \
  -t ${IMAGE}:${VERSION} \
  -t ${IMAGE}:${VERSION}-${GIT_SHA} \
  .

docker push ${IMAGE}:latest
docker push ${IMAGE}:${VERSION}
docker push ${IMAGE}:${VERSION}-${GIT_SHA}

echo "Pushed ${IMAGE}:${VERSION}-${GIT_SHA}"
```

## Calling Streaming Endpoint via curl

```bash
curl -X POST \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"input": {"prompt": "Explain attention mechanisms", "max_tokens": 200}}'

# For streaming (server-sent events):
curl -N \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/stream/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Streaming workflow:
1. POST to `/run` → get `job_id`
2. GET `/stream/{job_id}` → server-sent events with chunks
3. Final event has `status: "COMPLETED"` and full output
