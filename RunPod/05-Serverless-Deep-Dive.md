# Serverless Deep Dive

## Worker Lifecycle States

```
Initializing → Ready/Idle → Running → Throttled → Shutdown
```

| State | Description |
|---|---|
| Initializing | Container starting, pulling image, loading model |
| Ready/Idle | Handler loaded, waiting for job |
| Running | Executing a job's handler function |
| Throttled | Worker at concurrency limit, no new jobs accepted |
| Shutdown | idle_timeout elapsed, worker terminated |

## Worker Configuration

```
min_workers: 0   → scale to zero when idle (default; zero idle cost)
min_workers: 1   → always 1 active worker (billed continuously, eliminates cold start)
max_workers: N   → hard cap on parallel workers
```

Billing implication: `min_workers > 0` workers are billed per-second of uptime, even when idle.
Set `min_workers=1` only if cold start latency is unacceptable and traffic is frequent enough to justify cost.

## FlashBoot

RunPod's proprietary cold start optimization:
- After first successful boot, RunPod snapshots worker memory state at the "ready" checkpoint
- Subsequent cold starts restore from snapshot instead of full boot + model load
- 48% of cold starts complete in under 200ms (per RunPod docs)
- Requires a traffic volume to seed the cache — new or rarely-used endpoints won't benefit immediately
- Free — no configuration required
- Works best when model loading is the slow step (vs image pull)

## Scaling Strategies

### Queue Delay (default)
Scale up when jobs wait longer than `scalerValue` milliseconds.
- Good for latency-sensitive APIs
- Conservative — doesn't overprovision

### Request Count
Scale based on job volume:
```
target_workers = ceil((inQueue + inProgress) / scalerValue)
```
- `scalerValue=1` → 1 worker per pending/running job (aggressive)
- `scalerValue=4` → scale up when 4 jobs per worker
- Good for batch/throughput workloads

## Idle Timeout Tuning

Default: 5 seconds.

```
Low idle_timeout (1-5s)  → aggressive scale-down, lower cost, more cold starts
High idle_timeout (60s+) → workers stay warm, better latency, higher idle cost
```

Tune based on traffic pattern:
- Bursty, infrequent: low timeout or min_workers=1
- Steady stream: low timeout fine, workers stay busy
- Occasional single requests: increase timeout or set min_workers=1

## GPU Tier Selection — Fallback List

In the Serverless UI, you can specify multiple GPU types in priority order:
```
Priority 1: H100 SXM
Priority 2: H100 PCIe
Priority 3: A100 SXM 80GB
```

RunPod tries each in order based on availability. Fallback list pattern prevents job queue stalls when preferred GPU is unavailable.

## Dockerfile Requirements

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04  # CUDA base image

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY handler.py .

# REQUIRED: linux/amd64 platform, -u flag for unbuffered Python output
CMD ["python3", "-u", "handler.py"]
```

Critical requirements:
- **`linux/amd64` platform** — RunPod runs on x86_64; Apple Silicon builds default to `arm64` and will crash
- **`-u` flag** — unbuffered stdout, required for RunPod to capture logs in real time
- **CUDA base image** — must match CUDA version your packages expect (check with `nvidia-smi`)
- CMD runs your handler script directly

## Build and Push Workflow

```bash
# Build for linux/amd64 (required — even on Apple Silicon)
docker build --platform linux/amd64 -t yourusername/your-image:latest .

# Tag with version + git SHA for reproducibility
git_sha=$(git rev-parse --short HEAD)
docker tag yourusername/your-image:latest yourusername/your-image:v1.0-${git_sha}

# Push both tags
docker push yourusername/your-image:latest
docker push yourusername/your-image:v1.0-${git_sha}
```

Then in RunPod UI:
1. Create Serverless Endpoint
2. Set container image to `yourusername/your-image:latest`
3. Set GPU type(s), min/max workers, idle timeout
4. Deploy

## RunPod-Injected Environment Variables

| Variable | Value |
|---|---|
| `RUNPOD_POD_ID` | Unique ID of this worker pod |
| `RUNPOD_API_KEY` | API key associated with the endpoint |
| `RUNPOD_WEBHOOK_GET_JOB` | Internal job fetch URL (do not call manually) |
| `RUNPOD_WEBHOOK_POST_OUTPUT` | Internal result submission URL |

Access job ID from `event["id"]` in handler — do NOT rely on `RUNPOD_JOB_ID` env var (not injected per-job).

## Concurrency per Worker

Default: 1 job per worker at a time.

Override with `concurrency_modifier` (see `03-Python-SDK.md`).
Useful when:
- Model is small relative to GPU VRAM
- Each request does little GPU compute
- You want to pack multiple small requests onto one worker

## Cold Start Minimization Checklist

Priority order (highest impact first):

1. **`min_workers=1`** — eliminates cold start entirely (costs money)
2. **Network Volume for weights** — model loads from local NVMe instead of downloading from HF/S3
3. **FlashBoot** — automatic, but only kicks in after traffic seeds it
4. **Small Docker images** — multi-stage builds, only install what's needed
5. **Module-level model loading** — load model at module import, not inside handler function
6. **Smaller model or quantized** — quantized 7B loads faster than FP16 13B
7. **Increase idle_timeout** — keep workers warm between requests

```python
# GOOD: load at module level (happens once per worker lifecycle)
model = load_model("/runpod-volume/model")

def handler(event):
    return model.generate(event["input"]["prompt"])

# BAD: load inside handler (happens on every job)
def handler(event):
    model = load_model("/runpod-volume/model")  # slow every request
    return model.generate(event["input"]["prompt"])
```
