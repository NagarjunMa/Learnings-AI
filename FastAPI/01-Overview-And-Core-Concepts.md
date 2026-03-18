# FastAPI: Overview and Core Concepts

## What FastAPI is

Async Python web framework built on:
- **Starlette** — ASGI toolkit (routing, middleware, WebSockets)
- **Pydantic** — data validation and serialization via Python type hints

Ships with automatic **OpenAPI** spec generation → `/docs` (Swagger UI) and `/redoc` (ReDoc) available at startup, zero config.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello"}
```

## ASGI vs WSGI

| | WSGI (Flask, Django) | ASGI (FastAPI, Starlette) |
|---|---|---|
| Execution model | Synchronous | Async (event loop) |
| Concurrency unit | Thread per request | Coroutine per request |
| I/O blocking | Blocks thread | Yields to event loop |
| Server | Gunicorn, uWSGI | Uvicorn, Hypercorn |

**Why it matters for AI endpoints**: GPU inference is slow (100ms–10s). WSGI blocks the thread for that entire duration. ASGI releases the event loop during I/O waits — database calls, HTTP fan-outs, file reads — while inference runs. Under concurrent load this is a significant throughput difference.

## Path Operations (Route Decorators)

HTTP method → decorator mapping:

```python
@app.get("/items/{item_id}")      # READ
@app.post("/items/")              # CREATE
@app.put("/items/{item_id}")      # FULL UPDATE
@app.patch("/items/{item_id}")    # PARTIAL UPDATE
@app.delete("/items/{item_id}")   # DELETE
```

The decorated function's **return value is automatically serialized to JSON**. Return a dict, Pydantic model, or list — FastAPI handles the rest.

## Pydantic Models

Define request/response shape as Python classes:

```python
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
def generate(req: InferenceRequest):
    # req.prompt, req.max_tokens, req.temperature are validated + typed
    return {"output": run_model(req.prompt, req.max_tokens)}
```

- **Type mismatch or missing required field** → `422 Unprocessable Entity` with detailed error body, no custom code needed
- **Nested models**, **lists**, **optional fields** all work with standard Python type hints
- Same models used for serialization on output

## Auto-Generated Docs

| URL | UI | Source |
|---|---|---|
| `/docs` | Swagger UI (interactive) | OpenAPI JSON at `/openapi.json` |
| `/redoc` | ReDoc (read-only, clean) | Same OpenAPI spec |

FastAPI builds the spec from:
- Route decorators (paths, methods)
- Function signatures (params, types)
- Pydantic models (schemas)
- Docstrings (descriptions)

Disable with `FastAPI(docs_url=None, redoc_url=None)` in production if needed.

## Dependency Injection

`Depends()` — declarative injection of reusable components into route handlers:

```python
from fastapi import Depends, HTTPException

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401)
    return x_api_key

@app.post("/generate")
def generate(req: InferenceRequest, _: str = Depends(verify_api_key)):
    ...
```

Dependencies can depend on other dependencies (chains). FastAPI handles caching within a request — a dependency called multiple times in one request runs once.

Common uses: auth checks, DB session injection, config access, rate limiting.

## Lifespan / Startup Events

Load ML models once at startup using the lifespan context manager (FastAPI 0.93+):

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    app.state.model = load_model("/models/stable-diffusion")
    print("Model loaded")
    yield
    # --- shutdown ---
    del app.state.model

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
def generate(request: Request, req: InferenceRequest):
    model = request.app.state.model
    return {"output": model.run(req.prompt)}
```

**Why not a global variable**: module-level globals work but are less testable and harder to mock. `app.state` is the idiomatic FastAPI pattern.

Deprecated alternative (still seen in older codebases):
```python
@app.on_event("startup")   # deprecated since 0.93
async def startup():
    app.state.model = load_model()
```
