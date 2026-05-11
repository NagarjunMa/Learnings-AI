# Performance and Optimization

## response_model_exclude_unset / exclude_none

Critical for PATCH endpoints and sparse responses — don't serialize fields the client didn't send or that are null.

```python
from pydantic import BaseModel

class UserUpdate(BaseModel):
    username: str | None = None
    email: str | None = None
    bio: str | None = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    bio: str | None = None

# PATCH — only return fields that were actually updated
@app.patch(
    "/users/{id}",
    response_model=UserResponse,
    response_model_exclude_unset=True,   # fields not set by caller are excluded
)
async def update_user(id: int, update: UserUpdate):
    # Client sends: {"email": "new@example.com"}
    # Response includes only: {"email": "new@example.com"} — no id, username, bio
    ...

# GET — strip null fields
@app.get(
    "/users/{id}",
    response_model=UserResponse,
    response_model_exclude_none=True,    # None fields excluded from response
)
async def get_user(id: int):
    # bio=None → stripped from response JSON
    ...
```

---

## Pagination — Offset/Limit

```python
from pydantic import BaseModel, Field
from typing import TypeVar, Generic

T = TypeVar("T")

class PaginationParams:
    def __init__(
        self,
        skip: int = Query(default=0, ge=0, description="Items to skip"),
        limit: int = Query(default=20, ge=1, le=100, description="Max items to return"),
    ):
        self.skip = skip
        self.limit = limit

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    skip: int
    limit: int
    has_more: bool

@router.get("/users/", response_model=PaginatedResponse[UserResponse])
async def list_users(
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_db),
):
    # Count total
    total_result = await db.execute(select(func.count(User.id)))
    total = total_result.scalar()

    # Fetch page
    result = await db.execute(
        select(User)
        .offset(pagination.skip)
        .limit(pagination.limit)
        .order_by(User.created_at.desc())
    )
    users = list(result.scalars().all())

    return PaginatedResponse(
        items=users,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
        has_more=(pagination.skip + pagination.limit) < total,
    )
```

---

## Cursor-Based Pagination

Better than offset for large datasets — offset scans discarded rows, cursor does not.

```python
from datetime import datetime

@router.get("/feed/")
async def get_feed(
    cursor: str | None = Query(default=None, description="Pagination cursor (last item ID)"),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    query = select(Post).order_by(Post.id.desc()).limit(limit + 1)

    if cursor:
        last_id = int(cursor)
        query = query.where(Post.id < last_id)

    result = await db.execute(query)
    posts = list(result.scalars().all())

    has_more = len(posts) > limit
    if has_more:
        posts = posts[:limit]

    next_cursor = str(posts[-1].id) if has_more and posts else None

    return {
        "items": posts,
        "next_cursor": next_cursor,
        "has_more": has_more,
    }
```

---

## Redis Caching

```bash
pip install redis[asyncio]
```

```python
import json
import redis.asyncio as aioredis
from functools import wraps

redis_client = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)

async def cache_get(key: str):
    value = await redis_client.get(key)
    return json.loads(value) if value else None

async def cache_set(key: str, value, ttl: int = 300):
    await redis_client.setex(key, ttl, json.dumps(value))

async def cache_delete(key: str):
    await redis_client.delete(key)

# Usage in route
@router.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    cache_key = f"user:{user_id}"

    # Check cache first
    cached = await cache_get(cache_key)
    if cached:
        return cached

    # Cache miss — hit DB
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_dict = UserResponse.model_validate(user).model_dump()
    await cache_set(cache_key, user_dict, ttl=300)  # 5 min cache
    return user_dict

# Invalidate on update
@router.put("/users/{user_id}")
async def update_user(user_id: int, update: UserUpdate, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    # ... update logic ...
    await cache_delete(f"user:{user_id}")  # invalidate stale cache
    return user
```

---

## HTTP Caching Headers

For public, rarely-changing data — let CDN/browser cache.

```python
from fastapi.responses import Response

@router.get("/config")
async def get_config(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"  # CDN caches 1 hour
    return {"theme": "dark", "locale": "en"}

@router.get("/users/{user_id}")
async def get_user(user_id: int, response: Response):
    user = await fetch_user(user_id)
    # ETag — client sends If-None-Match; return 304 if unchanged
    etag = f'"{hash(str(user.updated_at))}"'
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=60"
    return user
```

---

## GZip Compression

```python
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # only compress responses > 1KB
    compresslevel=5,    # 1-9, higher = more compression, more CPU
)
```

Reduces response size 60-80% for JSON. Free performance for text responses.

---

## Async Connection Pool Tuning

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,           # persistent connections (= number of workers)
    max_overflow=10,        # burst capacity above pool_size
    pool_timeout=30,        # wait this long for a connection before error
    pool_recycle=1800,      # recycle connections after 30 min (prevent stale)
    pool_pre_ping=True,     # test connection health before use
)

# Rule: pool_size ≈ number of uvicorn workers
# For 4 workers: pool_size=5 (each worker gets ~5 connections)
# Total DB connections = pool_size + max_overflow = 30
```

---

## Avoid Blocking the Event Loop

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

# CPU-bound work — run in thread pool to avoid blocking event loop
@app.post("/process")
async def process_image(file: UploadFile):
    image_data = await file.read()

    # This is CPU-bound — don't await in event loop directly
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        heavy_cpu_processing,  # sync function
        image_data,
    )
    return {"result": result}

# For ML inference (GPU-bound): use def (not async def) — FastAPI runs it in thread pool
@app.post("/predict")
def predict(data: InputData):  # sync handler — FastAPI handles threading
    return model.inference(data)
```

---

## Response Model Performance

```python
# Avoid: loading entire ORM object and serializing all fields
@app.get("/users/")
async def list_users(db: AsyncSession = Depends(get_db)):
    users = await db.execute(select(User))  # loads all columns
    return users.scalars().all()

# Better: select only needed columns
@app.get("/users/")
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User.id, User.username, User.email)  # only 3 columns
        .limit(100)
    )
    return [{"id": r.id, "username": r.username, "email": r.email} for r in result.all()]
```

---

## Profiling

```bash
pip install pyinstrument
```

```python
from pyinstrument import Profiler

@app.middleware("http")
async def profile_middleware(request: Request, call_next):
    if request.query_params.get("profile") != "true":
        return await call_next(request)

    profiler = Profiler(async_mode="enabled")
    profiler.start()
    response = await call_next(request)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    return response
```

Access: `GET /api/slow-endpoint?profile=true` — prints call stack with timing.

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| Exclude unset in response | `response_model_exclude_unset=True` on route |
| Exclude None in response | `response_model_exclude_none=True` on route |
| Offset pagination | `select(Model).offset(skip).limit(limit)` |
| Cursor pagination | `select(Model).where(Model.id < cursor).limit(limit)` |
| Redis cache check | `cached = await redis.get(key); if cached: return json.loads(cached)` |
| Cache invalidation | `await redis.delete(f"user:{id}")` after write |
| Gzip middleware | `app.add_middleware(GZipMiddleware, minimum_size=1000)` |
| CPU work off event loop | `await loop.run_in_executor(executor, sync_fn, args)` |
| Pool sizing | `pool_size ≈ workers, add max_overflow for burst` |
| Profile endpoint | `pyinstrument.Profiler(async_mode="enabled")` |
