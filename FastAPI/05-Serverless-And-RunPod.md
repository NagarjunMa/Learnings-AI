# FastAPI: Backend Management and Serverless Deployment

## FastAPI on RunPod Pods (Always-On)

A RunPod **Pod** is a persistent container with a reserved GPU. Unlike Serverless, it stays running — no cold start, no RunPod handler SDK required.

Dockerfile CMD:
```dockerfile
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

RunPod exposes the port via its proxy URL. The endpoint is live as long as the Pod is running.

Contrast with Serverless:
| | Pod (always-on) | Serverless |
|---|---|---|
| Cold start | None | Yes (can be seconds) |
| RunPod SDK | Not needed | Required (`runpod.serverless.start`) |
| Billing | Per hour (even idle) | Per request / compute second |
| FastAPI | Drop-in, standard | Wrapped in SDK handler |

## Model Loading at Startup — Lifespan Pattern

Load once, reuse on every request:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import torch
from diffusers import StableDiffusionPipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    print("Loading model...")
    app.state.pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")
    print("Model ready")
    yield
    # --- shutdown ---
    del app.state.pipe
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
```

The `yield` separates startup (before) from shutdown (after). Both sides run in the same async context.

## Passing State to Routes

Access model via `request.app.state` — avoids module-level globals:

```python
from fastapi import Request
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 20

@app.post("/generate")
def generate(request: Request, req: GenerateRequest):
    pipe = request.app.state.pipe   # model loaded at startup
    image = pipe(req.prompt, num_inference_steps=req.steps).images[0]
    # convert + return
    return {"status": "ok"}
```

Can also use `Depends()` to wrap the state access:

```python
def get_pipeline(request: Request):
    return request.app.state.pipe

@app.post("/generate")
def generate(req: GenerateRequest, pipe=Depends(get_pipeline)):
    ...
```

## Async vs Sync Route Handlers

FastAPI handles both — choice depends on the work type:

```python
# I/O-bound: use async def
@app.get("/data")
async def fetch_data():
    result = await some_db_call()   # non-blocking
    return result

# CPU-bound (GPU inference): use def
@app.post("/generate")
def generate(req: GenerateRequest):
    # FastAPI automatically runs this in a thread pool
    # so it doesn't block the event loop
    return run_gpu_inference(req.prompt)
```

**Key rule**: GPU inference is CPU-bound from asyncio's perspective (it blocks a thread, not I/O). Use `def` — FastAPI runs it via `asyncio.get_event_loop().run_in_executor()` automatically.

Explicit executor if needed:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)  # 1 GPU → 1 worker

@app.post("/generate")
async def generate(request: Request, req: GenerateRequest):
    pipe = request.app.state.pipe
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: pipe(req.prompt).images[0]
    )
    return {"image": encode_image(result)}
```

## Mangum Adapter (True Serverless — AWS Lambda)

Wraps FastAPI as an AWS Lambda handler:

```python
from mangum import Mangum

app = FastAPI()
# ... routes ...

handler = Mangum(app)   # AWS Lambda calls handler(event, context)
```

Not needed on RunPod Pods (persistent containers). Useful if deploying to AWS Lambda + API Gateway. Mangum translates the Lambda event format to ASGI scope.

## Health Check Endpoint

RunPod, load balancers, and orchestrators poll a health endpoint:

```python
@app.get("/health")
def health():
    return {"status": "ok"}

# With model readiness check
@app.get("/health")
def health(request: Request):
    model_loaded = hasattr(request.app.state, "pipe")
    return {
        "status": "ok" if model_loaded else "loading",
        "model": "stable-diffusion-v1-5",
    }
```

Return `200` when ready, `503` when not. Configure RunPod health check to poll `/health` with a startup grace period (models can take 30–60s to load).

## Worker Management with Gunicorn + Uvicorn Workers

Single Uvicorn process for development. Production multi-process setup:

```bash
gunicorn app:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

- `-w 4` — 4 worker processes (each has its own event loop)
- `-k uvicorn.workers.UvicornWorker` — Gunicorn manages processes, Uvicorn handles ASGI within each
- Gunicorn handles process crashes, restarts, graceful shutdown signals

**GPU servers**: typically `-w 1` per GPU. Multiple workers = multiple processes each trying to load the model = OOM. If you have 2 GPUs, run 2 workers with `CUDA_VISIBLE_DEVICES` set per process, or use separate containers.

```bash
# Single GPU server
gunicorn app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Dev (no Gunicorn)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
