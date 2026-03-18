# FastAPI: Request and Response Handling

## Path Parameters

Declared in the route path, typed in the function signature:

```python
@app.get("/models/{model_id}/versions/{version}")
def get_version(model_id: str, version: int):
    # FastAPI validates: version must be int, 422 if not
    return {"model": model_id, "version": version}
```

FastAPI coerces strings from the URL to the declared type. Invalid coercion → `422`.

## Query Parameters

Any function parameter NOT in the path string becomes a query param:

```python
@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10, search: str | None = None):
    # GET /items/?skip=20&limit=5&search=cat
    ...
```

- `= None` → optional, omitted if not sent
- No default → required query param (returns 422 if missing)
- Fully validated and type-coerced, same as path params

## Request Body

Pass a Pydantic model as the type hint — FastAPI reads + validates the JSON body:

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    stream: bool = False

@app.post("/generate")
def generate(req: GenerateRequest):
    return run_inference(req.prompt, req.max_tokens)
```

Can combine path params, query params, and body in the same function — FastAPI figures out which is which by where the name appears.

## Raw Request Object

For headers, cookies, client IP, raw bytes, form data:

```python
from fastapi import Request

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()           # raw bytes
    headers = request.headers            # dict-like
    client_ip = request.client.host
    form = await request.form()          # multipart/form-data
    return {}
```

## Response Model

Declare what the response should look like — FastAPI filters output and validates it:

```python
class UserPublic(BaseModel):
    id: int
    username: str
    # no password field

class UserInternal(UserPublic):
    password_hash: str

@app.get("/users/{user_id}", response_model=UserPublic)
def get_user(user_id: int) -> UserInternal:
    user = db.get(user_id)   # returns UserInternal with password
    return user              # FastAPI strips password_hash automatically
```

`response_model` is the contract to callers. Even if you return extra fields internally, they're stripped before the response goes out.

## Status Codes

```python
from fastapi import HTTPException

@app.post("/items/", status_code=201)   # success status on decorator
def create_item(item: Item):
    ...

@app.get("/items/{item_id}")
def get_item(item_id: int):
    item = db.find(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

Common codes: `200` (default), `201` (created), `204` (no content), `400` (bad request), `401` (unauthorized), `403` (forbidden), `404` (not found), `422` (validation error, auto), `500` (server error).

## Background Tasks

Fire-and-forget work that runs after the response is sent:

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(message + "\n")

@app.post("/generate")
def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    result = run_inference(req.prompt)
    background_tasks.add_task(write_log, f"Generated: {req.prompt[:50]}")
    return {"output": result}   # response sent immediately; log written after
```

Good for: logging, analytics, sending notifications, cache invalidation.
Not good for: work the response depends on, or tasks needing reliable execution (use a task queue like Celery instead).

## Streaming Responses

Return large payloads without buffering the full body in memory:

```python
from fastapi.responses import StreamingResponse
import io

@app.get("/image/{image_id}")
def serve_image(image_id: str):
    image_bytes = load_image_from_disk(image_id)
    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png"
    )
```

For LLM token streaming:

```python
def generate_tokens(prompt: str):
    for token in model.stream(prompt):
        yield token

@app.post("/stream")
def stream_generate(req: GenerateRequest):
    return StreamingResponse(
        generate_tokens(req.prompt),
        media_type="text/plain"
    )
```

## Middleware

Runs before and after every request. Use for cross-cutting concerns:

```python
from fastapi.middleware.cors import CORSMiddleware
import time

# CORS (see 04-Frontend-Integration.md for details)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom timing middleware
from starlette.middleware.base import BaseHTTPMiddleware

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        response.headers["X-Process-Time"] = str(duration)
        return response

app.add_middleware(TimingMiddleware)
```

Middleware order: added last → runs first (LIFO).
