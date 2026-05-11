# OpenAPI Customization

FastAPI auto-generates OpenAPI schema from your code. Customize it to make `/docs` useful for API consumers and frontend teams.

---

## App-Level Metadata

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="""
## Overview

Production API for the MyApp platform.

### Features
- User management
- Item catalog
- Real-time notifications
    """,
    version="2.1.0",
    terms_of_service="https://myapp.com/terms",
    contact={
        "name": "API Support",
        "url": "https://myapp.com/support",
        "email": "api@myapp.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",          # Swagger UI (default: /docs)
    redoc_url="/redoc",        # ReDoc (default: /redoc)
    openapi_url="/openapi.json",   # Schema endpoint
)
```

---

## Tags with Descriptions

```python
app = FastAPI(
    openapi_tags=[
        {
            "name": "users",
            "description": "User registration, profile management, account settings.",
        },
        {
            "name": "auth",
            "description": "Login, logout, token refresh. JWT-based auth.",
            "externalDocs": {
                "description": "Auth documentation",
                "url": "https://myapp.com/docs/auth",
            },
        },
        {
            "name": "items",
            "description": "Item catalog. Requires authenticated user.",
        },
        {
            "name": "admin",
            "description": "Admin-only operations. Requires `admin` role.",
        },
    ]
)
```

Tags appear in display order. Untagged routes appear under a default group.

---

## Route-Level Documentation

```python
from fastapi import APIRouter, Path, Query

router = APIRouter(prefix="/users", tags=["users"])

@router.get(
    "/{user_id}",
    summary="Get user by ID",           # short title in docs
    description="""
Retrieve a user's public profile by their numeric ID.

Returns 404 if the user does not exist or has been deactivated.
    """,
    response_description="The user's public profile",
    responses={
        200: {"description": "User found"},
        404: {"description": "User not found"},
        422: {"description": "Invalid user_id format"},
    },
    operation_id="get_user_by_id",       # used in generated client SDKs
    deprecated=False,
)
async def get_user(
    user_id: int = Path(
        ...,
        description="Numeric user ID",
        ge=1,
        example=42,
    )
):
    ...
```

---

## Field-Level Examples in Pydantic

```python
from pydantic import BaseModel, Field, ConfigDict

class UserCreate(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=50,
        description="Unique username, alphanumeric + underscores",
        examples=["alice_42"],
    )
    email: str = Field(
        description="Valid email address",
        examples=["alice@example.com"],
    )
    age: int = Field(
        ge=0,
        le=150,
        description="User age in years",
        examples=[30],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "alice_42",
                "email": "alice@example.com",
                "age": 30,
            }
        }
    )
```

---

## Multiple Request Body Examples

```python
from fastapi import Body

@app.post("/users/")
async def create_user(
    user: UserCreate = Body(
        openapi_examples={
            "standard_user": {
                "summary": "Standard user registration",
                "description": "Typical user with all required fields",
                "value": {
                    "username": "alice_42",
                    "email": "alice@example.com",
                    "age": 30,
                },
            },
            "minimal_user": {
                "summary": "Minimal registration",
                "description": "Only required fields",
                "value": {
                    "username": "bob",
                    "email": "bob@example.com",
                },
            },
        }
    )
):
    ...
```

---

## Query Parameter Documentation

```python
from fastapi import Query

@router.get("/users/")
async def list_users(
    skip: int = Query(default=0, ge=0, description="Number of items to skip", example=0),
    limit: int = Query(default=20, ge=1, le=100, description="Max items to return", example=20),
    search: str | None = Query(default=None, description="Filter by username (partial match)", example="ali"),
    sort_by: str = Query(default="created_at", description="Sort field", enum=["created_at", "username", "email"]),
):
    ...
```

---

## Deprecated Endpoints

```python
@router.get(
    "/users/search",
    deprecated=True,
    summary="[DEPRECATED] Search users",
    description="Use GET /users/?search=query instead. This endpoint will be removed in v3.0.",
    tags=["users", "deprecated"],
)
async def search_users_old(q: str):
    # Redirect internally or maintain for backward compat
    ...
```

---

## Custom Response Models per Status Code

```python
from pydantic import BaseModel

class UserResponse(BaseModel):
    id: int
    username: str

class ErrorResponse(BaseModel):
    error: str
    message: str

@router.post(
    "/users/",
    response_model=UserResponse,
    status_code=201,
    responses={
        201: {"model": UserResponse, "description": "User created successfully"},
        400: {"model": ErrorResponse, "description": "Username already taken"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def create_user(user: UserCreate):
    ...
```

---

## Hide Endpoints from Docs

```python
# Exclude specific routes from OpenAPI schema and /docs
@router.get("/internal/health", include_in_schema=False)
async def internal_health():
    return {"status": "ok"}

# Or disable docs globally (production security)
app = FastAPI(
    docs_url=None,     # disables /docs
    redoc_url=None,    # disables /redoc
    openapi_url=None,  # disables /openapi.json entirely
)
```

Conditional based on environment:
```python
from config import get_settings

settings = get_settings()

app = FastAPI(
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)
```

---

## Custom OpenAPI Schema

For edge cases not handled by auto-generation:

```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="My API",
        version="2.0.0",
        description="Custom description",
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    # Apply security globally
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])

    app.openapi_schema = openapi_schema
    return openapi_schema

app.openapi = custom_openapi
```

---

## Docstring as Description

FastAPI uses the route function's docstring as the `description` field:

```python
@router.get("/{user_id}")
async def get_user(user_id: int):
    """
    Retrieve a user by their ID.

    - **user_id**: Must be a positive integer
    - Returns full public profile
    - Use `/users/me` to get the authenticated user's profile
    """
    ...
```

Markdown in docstrings renders in Swagger UI.

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| App metadata | `FastAPI(title=, description=, version=, contact={})` |
| Tag groups | `FastAPI(openapi_tags=[{"name": "users", "description": "..."}])` |
| Route summary | `@router.get("/path", summary="Short title")` |
| Route description | `@router.get("/path", description="Markdown text")` |
| Field example | `Field(..., examples=["value"])` |
| Multiple body examples | `Body(openapi_examples={"name": {"value": {...}}})` |
| Deprecated route | `@router.get("/path", deprecated=True)` |
| Hide from docs | `@router.get("/path", include_in_schema=False)` |
| Custom responses | `responses={404: {"model": ErrorModel}}` |
| Disable docs in prod | `FastAPI(docs_url=None, redoc_url=None)` |
