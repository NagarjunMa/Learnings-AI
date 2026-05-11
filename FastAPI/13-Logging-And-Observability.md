# Logging and Observability

## Why Structured Logging

Plain text logs are grep-unfriendly at scale. JSON logs are parseable by Datadog, CloudWatch, Loki, Splunk. Every log entry a machine can filter/aggregate.

---

## Python Logging Configuration

```python
# logging_config.py
import logging
import json
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include any extra fields
        for key, value in record.__dict__.items():
            if key not in ("msg", "args", "exc_info", "exc_text", "stack_info",
                          "levelname", "levelno", "pathname", "filename", "module",
                          "name", "funcName", "lineno", "created", "msecs",
                          "relativeCreated", "thread", "threadName", "processName",
                          "process", "message"):
                log_entry[key] = value
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO"):
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[handler],
    )

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
```

```python
# main.py
from logging_config import setup_logging
from config import get_settings

settings = get_settings()
setup_logging(log_level=settings.log_level)
```

---

## Request/Response Logging Middleware

Logs every incoming request and outgoing response with timing.

```python
import logging
import time
import uuid
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("access")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Attach request_id to request state for use in route handlers
        request.state.request_id = request_id

        # Log incoming request
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(exc),
                    "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                },
                exc_info=True,
            )
            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )

        response.headers["X-Request-ID"] = request_id
        return response

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)
```

---

## Correlation ID — Trace Requests Across Services

Propagate a single ID through all logs for a request — critical for microservices debugging.

```python
import contextvars
import uuid

# Context variable — per-async-task, not per-thread
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Use incoming header or generate new ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        correlation_id_var.set(correlation_id)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

# In any logger anywhere in codebase
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

class CorrelationFormatter(JSONFormatter):
    def format(self, record: logging.LogRecord) -> str:
        record.correlation_id = correlation_id_var.get()
        return super().format(record)
```

Usage in route handlers:
```python
logger = logging.getLogger(__name__)

@app.get("/users/{id}")
async def get_user(id: int, request: Request):
    logger.info(
        "Fetching user",
        extra={"user_id": id, "correlation_id": request.state.correlation_id}
    )
    ...
```

---

## Logging in Route Handlers

```python
import logging

logger = logging.getLogger(__name__)  # one per module

@router.post("/generate")
async def generate(prompt: str, request: Request):
    logger.info(
        "LLM generation started",
        extra={
            "prompt_length": len(prompt),
            "correlation_id": getattr(request.state, "correlation_id", ""),
        }
    )

    try:
        result = await call_llm(prompt)
        logger.info(
            "LLM generation complete",
            extra={"output_tokens": len(result.split())}
        )
        return {"result": result}
    except Exception as e:
        logger.error(
            "LLM generation failed",
            extra={"error_type": type(e).__name__},
            exc_info=True,  # includes full traceback in log
        )
        raise
```

---

## Uvicorn Access Log Format

```bash
# Development — human readable
uvicorn main:app --log-level debug

# Production — disable uvicorn access log (use your middleware instead)
uvicorn main:app \
    --log-level warning \
    --no-access-log \
    --workers 4
```

```python
# programmatic uvicorn config
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="warning",
        access_log=False,   # handled by middleware
        workers=4,
    )
```

---

## Structured Log Output Examples

Request log (JSON):
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "access",
  "message": "Request completed",
  "request_id": "a1b2c3d4",
  "method": "POST",
  "path": "/api/generate",
  "status_code": 200,
  "duration_ms": 1243.5,
  "client_ip": "10.0.0.1"
}
```

---

## Log Levels — When to Use

| Level | Use Case |
|-------|----------|
| `DEBUG` | Variable values, function entry/exit, SQL queries — dev only |
| `INFO` | Request start/end, business events (user created, payment processed) |
| `WARNING` | Degraded state, retry attempt, deprecated feature used |
| `ERROR` | Request failed, exception caught and handled |
| `CRITICAL` | Service cannot start, data corruption detected |

Production: `WARNING` and above. Development: `DEBUG`.

---

## OpenTelemetry Integration (Concept)

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Setup
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Instrument FastAPI — auto-traces all requests
FastAPIInstrumentor.instrument_app(app)

# Manual spans
@app.post("/generate")
async def generate(prompt: str):
    with tracer.start_as_current_span("llm-generate") as span:
        span.set_attribute("prompt.length", len(prompt))
        result = await call_llm(prompt)
        span.set_attribute("output.tokens", len(result.split()))
        return {"result": result}
```

Traces appear in Jaeger, Zipkin, or any OTLP-compatible backend.

---

## Health Check Endpoint

```python
import logging
from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from dependencies import get_db

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/health/detailed")
async def health_detailed(db: AsyncSession = Depends(get_db)):
    checks = {"api": "ok", "database": "unknown"}
    try:
        await db.execute(select(1))
        checks["database"] = "ok"
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        checks["database"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", "checks": checks}
    )
```

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| JSON formatter | `class JSONFormatter(logging.Formatter): def format(self, record): return json.dumps({...})` |
| Logger per module | `logger = logging.getLogger(__name__)` |
| Request middleware | `class Middleware(BaseHTTPMiddleware): async def dispatch(req, call_next)` |
| Attach to request state | `request.state.request_id = id` |
| Context var (async-safe) | `var: ContextVar[str] = ContextVar("name", default="")` |
| Log with extra fields | `logger.info("msg", extra={"key": "value"})` |
| Include traceback | `logger.error("msg", exc_info=True)` |
| Timing | `start = time.perf_counter(); duration_ms = (time.perf_counter() - start) * 1000` |
