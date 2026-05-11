# Exception Handling and Error Patterns

## Default FastAPI Error Behavior

- Validation errors (wrong types, missing fields) → `422 Unprocessable Entity` with Pydantic error detail
- `HTTPException` → whatever status_code you set
- Unhandled Python exceptions → `500 Internal Server Error`

All defaults are overridable.

---

## HTTPException — Basic Usage

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = await fetch_user(user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
            headers={"X-Error": "user-not-found"},  # optional custom headers
        )
    return user
```

`detail` can be a string, dict, or list — serialized as JSON in response body.

```python
raise HTTPException(
    status_code=400,
    detail={
        "error": "invalid_input",
        "message": "Username already taken",
        "field": "username",
    }
)
```

---

## Custom Exception Classes

```python
# exceptions.py
class AppException(Exception):
    def __init__(self, status_code: int, error_code: str, message: str):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message

class NotFoundException(AppException):
    def __init__(self, resource: str, resource_id: int | str):
        super().__init__(
            status_code=404,
            error_code="not_found",
            message=f"{resource} with id '{resource_id}' not found",
        )

class UnauthorizedException(AppException):
    def __init__(self, message: str = "Authentication required"):
        super().__init__(status_code=401, error_code="unauthorized", message=message)

class ForbiddenException(AppException):
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(status_code=403, error_code="forbidden", message=message)

class ConflictException(AppException):
    def __init__(self, message: str):
        super().__init__(status_code=409, error_code="conflict", message=message)
```

---

## @app.exception_handler — Custom Handlers

```python
# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from exceptions import AppException

app = FastAPI()

# Handle your custom exceptions with consistent format
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "path": str(request.url.path),
        },
    )

# Usage in routes — no HTTPException needed
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = await fetch_user(user_id)
    if not user:
        raise NotFoundException("User", user_id)
    return user
```

---

## Override Default 422 Validation Error Format

FastAPI's default 422 is verbose. Override for consistent API error format.

```python
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"] if loc != "body"),
            "message": error["msg"],
            "type": error["type"],
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": errors,
        },
    )
```

Before override:
```json
{"detail": [{"type": "missing", "loc": ["body", "email"], "msg": "Field required"}]}
```

After override:
```json
{"error": "validation_error", "message": "Request validation failed", "details": [{"field": "email", "message": "Field required", "type": "missing"}]}
```

---

## Override Default 500 Handler

```python
from fastapi.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
        },
    )

# Also override Starlette's HTTPException to use your format
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": exc.detail},
    )
```

---

## Retry with Exponential Backoff

For external API calls (LLM APIs, payment gateways, etc.) that may be rate-limited.

```python
import asyncio
import random
import logging
from typing import TypeVar, Callable, Any

logger = logging.getLogger(__name__)
T = TypeVar("T")

async def retry_with_backoff(
    fn: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    **kwargs,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except exceptions as e:
            if attempt == max_retries:
                logger.error(f"All {max_retries} retries exhausted: {e}")
                raise

            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
            await asyncio.sleep(delay)

# Usage
async def call_llm_with_retry(prompt: str) -> str:
    return await retry_with_backoff(
        call_llm_api,
        prompt,
        max_retries=3,
        base_delay=2.0,
        exceptions=(RateLimitError, TimeoutError),
    )
```

---

## Circuit Breaker Pattern

Prevents cascading failures when a downstream service is down.

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"       # normal — requests pass through
    OPEN = "open"           # failing — requests rejected immediately
    HALF_OPEN = "half_open" # testing — one request allowed through

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def call_allowed(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True  # let one through to test
            return False
        return True  # HALF_OPEN: one test request

    def on_success(self):
        self.failures = 0
        self.state = CircuitState.CLOSED

    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
llm_breaker = CircuitBreaker(failure_threshold=5, timeout=30.0)

async def call_llm_safe(prompt: str) -> str:
    if not llm_breaker.call_allowed():
        raise HTTPException(status_code=503, detail="LLM service temporarily unavailable")
    try:
        result = await call_llm_api(prompt)
        llm_breaker.on_success()
        return result
    except Exception as e:
        llm_breaker.on_failure()
        raise
```

---

## Timeout Handling

```python
import asyncio

@app.post("/generate")
async def generate(prompt: str):
    try:
        result = await asyncio.wait_for(
            call_llm_api(prompt),
            timeout=30.0,  # 30 second timeout
        )
        return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM request timed out")
```

---

## Error Response Schema (Pydantic)

```python
from pydantic import BaseModel

class ErrorDetail(BaseModel):
    field: str
    message: str
    type: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[ErrorDetail] | None = None
    path: str | None = None

# Use in OpenAPI docs
@app.get(
    "/users/{id}",
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    }
)
async def get_user(id: int):
    ...
```

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| Raise HTTP error | `raise HTTPException(status_code=404, detail="msg")` |
| Custom exception class | `class NotFoundException(AppException): ...` |
| Register handler | `@app.exception_handler(ExcType) async def handler(req, exc): return JSONResponse(...)` |
| Override 422 | `@app.exception_handler(RequestValidationError)` |
| Override 500 | `@app.exception_handler(Exception)` |
| Retry backoff | `asyncio.sleep(base * 2**attempt + jitter)` |
| Timeout | `await asyncio.wait_for(coro(), timeout=30.0)` |
| Circuit breaker state | CLOSED → OPEN (after N failures) → HALF_OPEN (after timeout) → CLOSED |
