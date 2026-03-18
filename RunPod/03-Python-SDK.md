# RunPod Python SDK

## Installation and Setup

```bash
pip install runpod  # Python 3.8+
```

```python
import runpod
runpod.api_key = "your_api_key"  # global config
```

Per-endpoint override:
```python
endpoint = runpod.Endpoint("endpoint_id", api_key="override_key")
```

## Handler Pattern — `runpod.serverless.start()`

The entry point for every Serverless worker. RunPod calls your `handler` function for each job.

```python
import runpod

def handler(event):
    job_input = event["input"]       # dict from caller
    job_id    = event["id"]          # RunPod job ID string

    # do work
    result = process(job_input["prompt"])

    return {"output": result}        # must return serializable dict or str

runpod.serverless.start({"handler": handler})
```

Return contract:
- Return a dict → RunPod wraps it in `{"output": <your_dict>}`
- Return a string → wrapped in `{"output": "<your_string>"}`
- Raise an exception → job status becomes FAILED, error stored

## Async Handler

```python
import runpod
import asyncio

async def handler(event):
    result = await async_model_call(event["input"]["prompt"])
    return {"text": result}

runpod.serverless.start({"handler": handler})
```

RunPod detects `async def` and runs it with an event loop automatically.

## Generator / Streaming Handler

Use `yield` to stream chunks back to caller. Set `return_aggregate_stream=True` to also return full text in final status.

```python
import runpod

def handler(event):
    prompt = event["input"]["prompt"]
    for token in model.stream(prompt):
        yield token                  # each yield → one chunk sent to caller

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True  # final status also includes full text
})
```

## Async Generator Streaming

```python
import runpod

async def handler(event):
    async for token in async_model_stream(event["input"]["prompt"]):
        yield token

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
```

## `runpod.Endpoint` Class — Client Side

```python
endpoint = runpod.Endpoint("your_endpoint_id")
```

### `run()` — fire and forget, returns job handle
```python
job = endpoint.run({"prompt": "hello world"})
print(job.job_id)
```

### `run_sync()` — blocks until done or timeout
```python
result = endpoint.run_sync(
    {"prompt": "hello world"},
    timeout=60  # seconds; default 90, hard max 90
)
print(result)  # {"output": ...}
```

### `health()` — endpoint status
```python
status = endpoint.health()
# {"workers": {"idle": 2, "running": 1}, "jobs": {"inQueue": 0, ...}}
```

## Job Status Polling Loop

```python
import time
import runpod

endpoint = runpod.Endpoint("endpoint_id")
job = endpoint.run({"prompt": "hello"})

while True:
    status = job.status()
    if status == "COMPLETED":
        print(job.output())
        break
    elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        print("Job ended with:", status)
        break
    time.sleep(1)
```

## All Job States

| State | Meaning |
|---|---|
| `IN_QUEUE` | Waiting for available worker |
| `IN_PROGRESS` | Worker is executing handler |
| `COMPLETED` | Handler returned successfully |
| `FAILED` | Handler raised exception |
| `CANCELLED` | Cancelled by caller before execution |
| `TIMED_OUT` | Exceeded endpoint timeout setting |

## `progress_update()` — Mid-Job Status

```python
import runpod

def handler(event):
    runpod.serverless.progress_update(event, {"step": 1, "status": "loading model"})
    model = load_model()
    runpod.serverless.progress_update(event, {"step": 2, "status": "generating"})
    result = model.generate(event["input"]["prompt"])
    return {"text": result}
```

Caller can poll `job.status()` and see the progress payload before completion.

## `concurrency_modifier` — Multiple Jobs per Worker

```python
import runpod

def concurrency_modifier(current_concurrency):
    # called by RunPod to decide max concurrent jobs for this worker
    # return new concurrency limit based on current load / resources
    gpu_free_gb = get_free_vram()
    if gpu_free_gb > 20:
        return 4
    elif gpu_free_gb > 10:
        return 2
    return 1

def handler(event):
    return {"output": process(event["input"])}

runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})
```

Default concurrency per worker = 1. Useful for lighter models where one GPU can handle multiple requests simultaneously.
