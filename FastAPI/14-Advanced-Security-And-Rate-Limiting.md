# Advanced Security and Rate Limiting

## JWT Refresh Token Flow

Access tokens expire quickly (15-60 min). Refresh tokens are long-lived (7-30 days) and used only to get new access tokens — never sent to API endpoints.

```python
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

router = APIRouter(prefix="/auth", tags=["auth"])

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire, "type": "access"},
        SECRET_KEY, algorithm=ALGORITHM,
    )

def create_refresh_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire, "type": "refresh"},
        SECRET_KEY, algorithm=ALGORITHM,
    )

@router.post("/login", response_model=TokenPair)
async def login(credentials: LoginCredentials):
    user = await authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenPair(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )

@router.post("/refresh", response_model=TokenPair)
async def refresh(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = int(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return TokenPair(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),  # rotate refresh token
    )
```

---

## Token Revocation (Blacklist)

JWTs are stateless — can't invalidate without a blocklist. Use Redis for O(1) lookup.

```python
import redis.asyncio as aioredis
from datetime import datetime, timezone

redis_client = aioredis.from_url("redis://localhost:6379/0")

async def revoke_token(token: str, expires_at: datetime):
    ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds())
    if ttl > 0:
        await redis_client.setex(f"revoked:{token}", ttl, "1")

async def is_token_revoked(token: str) -> bool:
    return await redis_client.exists(f"revoked:{token}") > 0

# Updated auth dependency
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if await is_token_revoked(token):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Logout endpoint
@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    await revoke_token(token, exp)
    return {"message": "Logged out"}
```

---

## Rate Limiting with slowapi

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/data")
@limiter.limit("100/minute")          # 100 requests per minute per IP
async def get_data(request: Request):
    return {"data": "..."}

@app.post("/auth/login")
@limiter.limit("5/minute")            # stricter limit for auth endpoints
async def login(request: Request):
    ...

@app.post("/api/generate")
@limiter.limit("10/hour;3/minute")    # compound limit
async def generate(request: Request):
    ...
```

**Rate limit by user instead of IP:**
```python
async def get_user_id(request: Request) -> str:
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return f"user:{payload['sub']}"
    except Exception:
        return get_remote_address(request)

limiter = Limiter(key_func=get_user_id)
```

---

## Security Headers Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        # Remove server info
        response.headers.pop("server", None)
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

---

## HTTPS Enforcement

```python
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Redirect HTTP → HTTPS (production only)
if settings.is_production:
    app.add_middleware(HTTPSRedirectMiddleware)

# Reject requests with invalid Host header (prevent host header injection)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["myapp.com", "www.myapp.com", "api.myapp.com"]
)
```

---

## Input Sanitization

```python
import re
import html
from pydantic import BaseModel, field_validator

class UserInput(BaseModel):
    username: str
    bio: str | None = None

    @field_validator("username")
    @classmethod
    def sanitize_username(cls, v: str) -> str:
        # Only allow alphanumeric + underscore + hyphen
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username contains invalid characters")
        return v

    @field_validator("bio")
    @classmethod
    def sanitize_bio(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Escape HTML to prevent XSS if bio is ever rendered in HTML
        return html.escape(v)
```

---

## SQL Injection Prevention

ORM (SQLAlchemy) prevents SQL injection by using parameterized queries automatically.

```python
# SAFE — ORM parameterizes this
result = await db.execute(
    select(User).where(User.username == username)  # username is a parameter
)

# SAFE — explicit parameter binding
from sqlalchemy import text
result = await db.execute(
    text("SELECT * FROM users WHERE username = :username"),
    {"username": username}  # properly parameterized
)

# UNSAFE — never do this
username = request.query_params.get("username")
result = await db.execute(
    text(f"SELECT * FROM users WHERE username = '{username}'")  # SQL injection risk
)
```

---

## API Key Rotation Pattern

```python
from enum import Enum

class APIKeyStatus(str, Enum):
    active = "active"
    deprecated = "deprecated"   # still works, warn client
    revoked = "revoked"         # rejected

# DB model: APIKey(key_hash, status, expires_at, user_id)

async def validate_api_key(
    x_api_key: str = Header(...),
    db: AsyncSession = Depends(get_db),
) -> User:
    import hashlib
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    api_key = await db.execute(
        select(APIKey).where(APIKey.key_hash == key_hash)
    )
    api_key = api_key.scalar_one_or_none()

    if not api_key or api_key.status == APIKeyStatus.revoked:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if api_key.status == APIKeyStatus.deprecated:
        # Warn but allow
        # Log deprecation usage for migration tracking
        logger.warning(f"Deprecated API key used: user_id={api_key.user_id}")

    user = await db.get(User, api_key.user_id)
    return user
```

---

## CORS — Detailed Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://myapp.com",
        "https://staging.myapp.com",
    ],
    allow_origin_regex=r"https://.*\.myapp\.com",  # match subdomains
    allow_credentials=True,      # Allow cookies / Authorization headers
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Correlation-ID"],
    max_age=600,                 # preflight cache duration in seconds
)
```

**`allow_credentials=True` + `allow_origins=["*"]` is invalid** — browser blocks it. Must list explicit origins when using credentials.

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| Access token (short) | `jwt.encode({"sub": id, "exp": now+30min, "type": "access"}, key)` |
| Refresh token (long) | `jwt.encode({"sub": id, "exp": now+7days, "type": "refresh"}, key)` |
| Revoke token | `redis.setex(f"revoked:{token}", ttl, "1")` |
| Check revocation | `redis.exists(f"revoked:{token}")` |
| Rate limit | `@limiter.limit("100/minute")` on route |
| Rate by user | `Limiter(key_func=get_user_id)` |
| Security headers | Custom `BaseHTTPMiddleware` setting X-Frame-Options, CSP, HSTS |
| HTTPS redirect | `app.add_middleware(HTTPSRedirectMiddleware)` |
| Trusted hosts | `app.add_middleware(TrustedHostMiddleware, allowed_hosts=[...])` |
| Safe SQL | Always use ORM or parameterized `text("... :param", {"param": value})` |
