# APIRouter and Project Structure

## Why APIRouter Exists

Flat `main.py` with 30 routes is unmanageable. `APIRouter` lets you split routes into modules with their own prefix, tags, dependencies, and responses — then compose them in `main.py`. Same effect as Express Router or Django urlconf.

---

## Basic APIRouter

```python
# routers/users.py
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(
    prefix="/users",          # prepended to every route in this file
    tags=["users"],           # groups routes in /docs
    dependencies=[],          # applied to EVERY route in this router
    responses={404: {"description": "Not found"}},  # shown in OpenAPI for all routes
)

@router.get("/")
async def list_users():
    return [{"id": 1, "name": "Alice"}]

@router.get("/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id}

@router.post("/", status_code=201)
async def create_user():
    return {"id": 2}
```

```python
# main.py
from fastapi import FastAPI
from routers import users, items, auth

app = FastAPI()

app.include_router(users.router)
app.include_router(items.router)
app.include_router(auth.router)
```

Result: `/users/`, `/users/{user_id}`, `/users/` POST — all registered on `app`.

---

## Router-Level Dependencies

Dependencies declared on `APIRouter` apply to **every route** in that router. Critical pattern for auth guards on entire feature area.

```python
# routers/admin.py
from fastapi import APIRouter, Depends
from dependencies import require_admin

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],  # all admin routes require this
)

@router.get("/dashboard")
async def dashboard():
    # require_admin already ran — no need to add Depends here
    return {"data": "sensitive"}

@router.delete("/user/{user_id}")
async def delete_user(user_id: int):
    return {"deleted": user_id}
```

```python
# dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def require_admin(token: str = Depends(security)):
    if token.credentials != "admin-secret":
        raise HTTPException(status_code=403, detail="Admin only")
```

---

## Nested Routers / API Versioning

```python
# main.py
from fastapi import FastAPI
from routers.v1 import users as v1_users
from routers.v2 import users as v2_users

app = FastAPI()

# v1 prefix added at include time — routers stay clean
app.include_router(v1_users.router, prefix="/v1")
app.include_router(v2_users.router, prefix="/v2")
```

```python
# routers/v1/users.py
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["v1/users"])

@router.get("/")
async def list_users():
    return {"version": "v1", "users": []}
```

Routes: `/v1/users/`, `/v2/users/` — no duplication.

---

## include_router Options

```python
app.include_router(
    router,
    prefix="/api",            # additional prefix on top of router's own prefix
    tags=["extra-tag"],       # merged with router's tags
    dependencies=[Depends(rate_limiter)],  # merged with router's dependencies
    responses={500: {"description": "Server error"}},
    deprecated=True,          # marks ALL routes in router as deprecated in docs
)
```

---

## Production Project Structure

```
project/
├── main.py                   # FastAPI app init, include_router calls, lifespan
├── config.py                 # pydantic-settings BaseSettings
├── dependencies.py           # shared Depends functions (auth, db session, pagination)
├── models/                   # SQLAlchemy ORM models
│   ├── __init__.py
│   ├── user.py
│   └── item.py
├── schemas/                  # Pydantic request/response models
│   ├── __init__.py
│   ├── user.py
│   └── item.py
├── routers/                  # APIRouter modules
│   ├── __init__.py
│   ├── auth.py
│   ├── users.py
│   ├── items.py
│   └── admin.py
├── services/                 # Business logic (no HTTP concerns)
│   ├── user_service.py
│   └── item_service.py
├── db/                       # Database setup
│   ├── session.py            # engine, AsyncSession factory
│   └── migrations/           # Alembic
├── middleware/
│   └── logging.py
└── tests/
    ├── conftest.py           # shared fixtures
    ├── test_users.py
    └── test_items.py
```

**Key constraint**: Routers only import from `schemas/`, `services/`, `dependencies.py`. Services import from `models/` and `db/`. No circular imports.

---

## Router Separation by Domain (not HTTP method)

Wrong — split by HTTP method:
```
routers/
├── get_routes.py
├── post_routes.py
```

Correct — split by domain:
```
routers/
├── users.py      # GET /users, POST /users, GET /users/{id}, DELETE /users/{id}
├── items.py
├── auth.py
```

Each domain file owns all CRUD for that resource.

---

## Tags for OpenAPI Grouping

```python
# main.py — define tag metadata for docs
app = FastAPI(
    openapi_tags=[
        {"name": "users", "description": "User management"},
        {"name": "items", "description": "Item catalog"},
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "admin", "description": "Admin-only operations — requires admin role"},
    ]
)
```

Tags declared here appear in `/docs` as collapsible groups with descriptions. Tag order here controls display order.

---

## Overriding Responses Per Route

```python
router = APIRouter(
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    }
)

# This route adds 404 on top of router's 401/403
@router.get("/{id}", responses={404: {"description": "Not found"}})
async def get_item(id: int):
    ...
```

---

## Route Deprecation

```python
@router.get("/old-endpoint", deprecated=True)
async def old_endpoint():
    """Use /new-endpoint instead."""
    return {"status": "deprecated"}
```

Shows strikethrough in `/docs`. Use for phased API migrations — don't delete routes immediately.

---

## Interview Cheat Sheet

| Concept | Pattern |
|---------|---------|
| Router with prefix | `APIRouter(prefix="/users", tags=["users"])` |
| Include router | `app.include_router(router)` |
| Auth on all routes | `APIRouter(dependencies=[Depends(auth)])` |
| Versioning | `app.include_router(v2.router, prefix="/v2")` |
| Group in docs | `openapi_tags=[{"name": "users", "description": "..."}]` |
| Deprecate route | `@router.get("/path", deprecated=True)` |
| Extra prefix at include | `app.include_router(router, prefix="/api")` |
