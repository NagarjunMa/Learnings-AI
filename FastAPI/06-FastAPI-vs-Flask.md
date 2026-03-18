# FastAPI vs Flask: Detailed Comparison

## Architecture

| | Flask | FastAPI |
|---|---|---|
| Interface standard | WSGI | ASGI |
| Execution model | Synchronous | Async-first |
| Concurrency | One thread per request | Event loop, coroutines |
| Built on | Werkzeug | Starlette + Pydantic |
| First release | 2010 | 2018 |

Flask processes one request per thread. Under load, you need more threads (or processes), each consuming memory. FastAPI's event loop handles concurrent I/O without spawning extra threads — more efficient under high concurrency.

## Performance

FastAPI benchmarks **3–5× faster than Flask** under concurrent load in most synthetic benchmarks (TechEmpower, etc.). The gap is most pronounced when:
- Requests involve I/O waits (DB, external HTTP, file reads)
- Many concurrent connections

Flask can approach FastAPI's throughput with **gevent monkey-patching** (cooperative multitasking via coroutines), but:
- Monkey-patching is fragile — some libraries break
- Not idiomatic; debugging is harder
- Still not true ASGI

For CPU-bound-only workloads with no concurrency (e.g., single-user script), the difference is negligible.

## Type System

```python
# Flask — manual, no validation
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")            # could be None, no validation
    max_tokens = data.get("max_tokens", 100)  # no type coercion
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    ...

# FastAPI — Pydantic, automatic
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
def generate(req: GenerateRequest):  # validated, typed, 422 if invalid
    ...
```

FastAPI type hints are **functional** (enforced at runtime), not just documentation. Flask type hints are decorative — they don't affect behavior.

## Auto-Documentation

| | Flask | FastAPI |
|---|---|---|
| Built-in docs | None | `/docs` + `/redoc` |
| OpenAPI spec | Requires extension | Auto-generated |
| Extensions needed | Flask-RESTX, Flasgger | None |

FastAPI builds the full OpenAPI spec from your route decorators, type hints, and Pydantic models at startup — no extra annotations needed. In Flask, you'd write docstrings in a specific YAML format or use decorators from Flask-RESTX.

## Validation

```python
# Flask — manual validation
@app.route("/items", methods=["POST"])
def create_item():
    data = request.get_json()
    if "name" not in data:
        return jsonify({"error": "name required"}), 400
    if not isinstance(data.get("price"), (int, float)):
        return jsonify({"error": "price must be numeric"}), 400
    if data["price"] < 0:
        return jsonify({"error": "price must be positive"}), 400
    ...

# FastAPI — Pydantic
from pydantic import BaseModel, validator, Field

class Item(BaseModel):
    name: str
    price: float = Field(..., gt=0)

@app.post("/items")
def create_item(item: Item):   # all validation above = 3 lines
    ...
```

FastAPI with Pydantic: validators, field constraints (`gt`, `lt`, `min_length`, `regex`), nested models, custom error messages — all declarative.

## Dependency Injection

```python
# Flask — no DI system
# Common pattern: g object + before_request
@app.before_request
def load_user():
    g.user = get_user_from_token(request.headers.get("Authorization"))

@app.route("/data")
def get_data():
    if not g.user:
        abort(401)
    ...

# FastAPI — first-class Depends()
def get_current_user(token: str = Depends(oauth2_scheme)):
    return decode_token(token)

@app.get("/data")
def get_data(user: User = Depends(get_current_user)):
    ...
```

FastAPI's `Depends()` is composable, testable (can override in tests), and cached within a request. Flask's `g`/`before_request` pattern works but is implicit and harder to test in isolation.

## Ecosystem Maturity

| | Flask | FastAPI |
|---|---|---|
| Age | 15+ years | ~6 years |
| GitHub stars (approx) | ~67k | ~80k |
| Stack Overflow Q&A | Massive | Growing fast |
| Extensions/plugins | Vast (Flask-SQLAlchemy, Flask-Login, etc.) | Smaller but modern equivalents exist |
| Production adoption | Huge (Netflix, Airbnb, Pinterest have used it) | Growing (Uber, Microsoft, others) |

Flask wins on raw tutorial count and legacy integration examples. FastAPI's docs are exceptionally well-written and often sufficient on their own.

## Learning Curve

**Flask**:
- Easier to start — no async concepts, no Pydantic required
- "Hello world" in 5 lines
- Gets complex in production: session handling, CSRF, auth all manual

**FastAPI**:
- Steeper start — need to understand async/await and Pydantic
- Production patterns are built-in, not bolted on
- Once you understand the abstractions, boilerplate is minimal

## When to Use Which

| Scenario | Flask | FastAPI |
|---|---|---|
| Quick prototype / simple CRUD app | ✓ | ✓ |
| High-concurrency AI inference API | | ✓ |
| Async DB (SQLAlchemy async, Motor) | | ✓ |
| Team already deeply knows Flask | ✓ | |
| Auto-generated API docs needed | | ✓ |
| Streaming responses (LLM tokens) | Awkward | ✓ |
| Legacy codebase integration | ✓ | |
| Strict input validation without boilerplate | | ✓ |
| Serverless (AWS Lambda via Mangum) | ✓ (via serverless-wsgi) | ✓ (via Mangum) |
| WebSocket support | Extensions needed | Built-in |

**Bottom line for AI/ML backends**: FastAPI is the better default. The async model, Pydantic validation, streaming support, and built-in docs directly address the common pain points of model-serving APIs. Flask is still fine for simple use cases or when the team is heavily invested in the Flask ecosystem.
