# Debugging and Pitfalls

## OOM Crashes

**Symptoms**: Job status FAILED, error contains `CUDA out of memory` or `RuntimeError: CUDA error`

**Root causes**:
- Input too long → KV cache grows beyond VRAM
- Batch size too large for model + KV cache
- KV cache explosion on very long sequences
- Not reserving headroom for activations

**Fixes (priority order)**:
1. Add `max_new_tokens` cap in handler to prevent unbounded generation
2. Set `max_memory` headroom: `{0: "20GiB"}` leaves 4GB free on 24GB GPU
3. Call `torch.cuda.empty_cache()` in except block and retry
4. Quantize model (INT4/INT8) to reduce base footprint
5. Enable Flash Attention 2 — reduces KV cache memory complexity
6. Reduce batch size or use dynamic batching

```python
# Set max memory limit in from_pretrained
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    max_memory={0: "20GiB"},  # leave ~4GB headroom on 24GB GPU
    torch_dtype=torch.float16,
)
```

---

## Cold Start Too Slow

**Symptoms**: First request after idle takes 30-120+ seconds

**Root causes (in order of frequency)**:
1. No Network Volume → model downloading from HuggingFace/S3 on every cold start
2. Large Docker image → image pull adds startup time
3. Slow model loading code (tokenizer + model sequential loads)
4. No FlashBoot cache (new endpoint or low traffic volume)

**Solutions (highest impact first)**:
1. Store model weights on Network Volume at `/runpod-volume/` — eliminates download
2. Set `min_workers=1` — eliminates cold start entirely (costs money)
3. Use multi-stage Docker build to minimize image size
4. Load tokenizer and model concurrently if possible
5. Use smaller quantized model (faster to deserialize)
6. Let FlashBoot cache warm up over first few requests

---

## Spot Pod Interruption

**Symptoms**: Pod terminates mid-training, job lost

**SIGTERM handling pattern**:
```python
import signal
import sys

def graceful_shutdown(signum, frame):
    print("Received SIGTERM — saving checkpoint")
    save_checkpoint(current_step)  # save to /runpod-volume/checkpoints/
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
```

**When not to use Spot**:
- Real-time inference serving (user-facing latency)
- Jobs with no natural checkpoint boundary
- Short jobs where restart overhead > savings

---

## Network Volume Mount Timing

**Symptom**: `FileNotFoundError: /runpod-volume/model not found` at startup

**Cause**: Network volume takes 2-10s to mount after container starts; code runs before mount is ready.

**Fix — readiness wait loop**:
```python
import os, time

def wait_for_path(path, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path) and os.listdir(path):
            return
        time.sleep(2)
    raise RuntimeError(f"Path {path} not available after {timeout}s")

wait_for_path("/runpod-volume/model")
model = load_model("/runpod-volume/model")
```

---

## Volume Attached But Model Path Not Found

**Symptom**: Volume shows as attached in UI, but `FileNotFoundError` for model files.

**Cause**: Assuming wrong mount path. Network volume always mounts at `/runpod-volume/` regardless of what you name the volume.

**Fix**: Always use `/runpod-volume/your_subdir` as path:
```python
MODEL_PATH = "/runpod-volume/llama-7b"  # correct
MODEL_PATH = "/mnt/storage/llama-7b"    # wrong — volume is not here
```

---

## Handler Silently Succeeds With No Output

**Symptom**: Job status COMPLETED, but output is `None` or empty `{}`.

**Cause**: Broad `except` clause swallowing exception before it propagates:
```python
# BAD
def handler(event):
    try:
        result = model.generate(event["input"]["prompt"])
        return {"text": result}
    except Exception:
        pass  # silently swallows — job completes with None output
```

**Fix**: Always re-raise or return error information:
```python
# GOOD
def handler(event):
    try:
        result = model.generate(event["input"]["prompt"])
        return {"text": result}
    except Exception as e:
        raise  # re-raise → RunPod marks job FAILED with error message
```

---

## `run_sync` 90-Second Hard Timeout

**Symptom**: `TimeoutError` or empty result for jobs that take longer than 90s.

**Cause**: `endpoint.run_sync()` has a 90-second hard timeout imposed by the SDK.

**Fix**: Switch to `run()` + polling for long-running jobs:
```python
# Instead of:
result = endpoint.run_sync(payload, timeout=300)  # ignores timeout > 90

# Use:
job = endpoint.run(payload)
while job.status() not in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
    time.sleep(2)
result = job.output()
```

---

## `docker build` Without `--platform linux/amd64` on Apple Silicon

**Symptom**: Image builds fine locally on M1/M2/M3 Mac, but crashes on RunPod with `exec format error` or `Illegal instruction`.

**Cause**: Apple Silicon builds default to `linux/arm64`. RunPod runs on x86_64.

**Fix**: Always specify platform:
```bash
docker build --platform linux/amd64 -t myimage:latest .
```

Add to CI/CD and local scripts to prevent forgetting.

---

## Worker Retried Endlessly

**Symptom**: Worker shows "Initializing" then immediately retries in a loop; logs show crash before any job is processed.

**Cause**: Module-level crash (outside the handler function). RunPod retries worker startup, but every attempt hits the same crash.

**Debug steps**:
1. Pull image locally and run it: `docker run --rm myimage:latest`
2. Look for import errors, model loading failures, path errors
3. Check that `wait_for_volume()` doesn't time out (volume not attached in test)

Common module-level crashes:
- `ModuleNotFoundError` — missing pip install in Dockerfile
- `FileNotFoundError` — model path wrong before volume is mounted
- CUDA error — CUDA version mismatch between image and GPU driver

---

## `RUNPOD_JOB_ID` Not Injected

**Symptom**: `os.environ.get("RUNPOD_JOB_ID")` returns `None` in handler.

**Cause**: RunPod does not inject a per-job `RUNPOD_JOB_ID` env var.

**Fix**: Get job ID from the event object:
```python
def handler(event):
    job_id = event["id"]  # correct
    job_id = os.environ.get("RUNPOD_JOB_ID")  # wrong — always None
```

---

## FlashBoot Not Effective for New Endpoints

**Symptom**: FlashBoot is enabled but cold starts still take 30+ seconds.

**Cause**: FlashBoot cache is seeded by actual traffic. A brand-new endpoint has no cached checkpoint; first N requests must do full boot.

**Fix**:
- Accept slower starts for first few requests after deploy
- Warm up with `min_workers=1` during initial deployment period, then reduce
- FlashBoot improves automatically as traffic accumulates
- Do not expect FlashBoot to help on endpoints receiving <1 request/hour
