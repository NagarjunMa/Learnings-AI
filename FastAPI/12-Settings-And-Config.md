# Settings and Configuration Management

## Why pydantic-settings

Environment variables are strings — need parsing and validation. `pydantic-settings` reads env vars / `.env` files and validates them with Pydantic models. One source of truth for all config.

```bash
pip install pydantic-settings
```

---

## Basic BaseSettings

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",           # load from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,       # DATABASE_URL == database_url
        extra="ignore",            # ignore unknown env vars
    )

    # App
    app_name: str = "MyAPI"
    app_version: str = "1.0.0"
    debug: bool = False
    allowed_hosts: list[str] = ["*"]

    # Database
    database_url: str                       # required — no default
    db_pool_size: int = Field(default=20, ge=1, le=100)

    # Auth
    secret_key: str                         # required
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"

    # External services
    openai_api_key: str | None = None       # optional
    anthropic_api_key: str | None = None

    # Redis
    redis_url: str = "redis://localhost:6379/0"

settings = Settings()  # reads from env + .env file at import time
```

```bash
# .env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mydb
SECRET_KEY=supersecretkey123
DEBUG=true
DB_POOL_SIZE=10
```

---

## Settings as Dependency

Don't use global `settings` import everywhere — inject via `Depends` for testability.

```python
# config.py
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()

# routes
from fastapi import Depends
from config import Settings, get_settings

@app.get("/info")
async def info(settings: Settings = Depends(get_settings)):
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
    }
```

`lru_cache` means `Settings()` is instantiated once — subsequent `Depends(get_settings)` calls return the cached instance. Override in tests:
```python
def test_info():
    def mock_settings():
        return Settings(app_name="TestApp", database_url="sqlite:///:memory:", secret_key="test")

    app.dependency_overrides[get_settings] = mock_settings
    response = client.get("/info")
    assert response.json()["app_name"] == "TestApp"
    app.dependency_overrides.clear()
```

---

## Multi-Environment Config

```python
# config.py
from enum import Enum

class Environment(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    environment: Environment = Environment.development
    database_url: str
    secret_key: str

    # Derived settings based on environment
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.production

    @property
    def cors_origins(self) -> list[str]:
        if self.is_production:
            return ["https://myapp.com"]
        return ["http://localhost:3000", "http://localhost:5173"]

    @property
    def log_level(self) -> str:
        return "WARNING" if self.is_production else "DEBUG"
```

```bash
# .env.development
ENVIRONMENT=development
DATABASE_URL=sqlite+aiosqlite:///./dev.db
SECRET_KEY=dev-secret-not-secure

# .env.production
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://user:pass@prod-db:5432/mydb
SECRET_KEY=<real-secret-from-vault>
```

Load by environment:
```python
import os

env = os.getenv("ENVIRONMENT", "development")
settings = Settings(_env_file=f".env.{env}")
```

---

## Secrets Management

### AWS Secrets Manager

```python
import json
import boto3
from functools import lru_cache

@lru_cache
def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

class Settings(BaseSettings):
    database_url: str = ""
    secret_key: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with secrets from Secrets Manager in production
        if not self.debug:
            secrets = get_secret("myapp/production")
            self.database_url = secrets["database_url"]
            self.secret_key = secrets["secret_key"]
```

### Environment Variable Best Practices

```python
class Settings(BaseSettings):
    # Never log these
    secret_key: str
    database_password: str
    api_key: str

    def __repr__(self) -> str:
        # Safe repr — never exposes secrets in logs
        return f"Settings(app_name={self.app_name!r}, env={self.environment!r})"

    class Config:
        # Mark fields as secret to prevent accidental logging
        @classmethod
        def customise_sources(cls, init_settings, env_settings, dotenv_settings, default_settings):
            return init_settings, env_settings, dotenv_settings, default_settings
```

---

## Settings Validation

```python
from pydantic import field_validator, model_validator

class Settings(BaseSettings):
    database_url: str
    db_pool_size: int = 20
    db_max_overflow: int = 0
    secret_key: str
    environment: str = "development"

    @field_validator("database_url")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith(("postgresql", "sqlite", "mysql")):
            raise ValueError("Unsupported database URL scheme")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        env = info.data.get("environment", "development")
        if env == "production" and len(v) < 32:
            raise ValueError("Production secret_key must be at least 32 characters")
        return v

    @model_validator(mode="after")
    def validate_pool_config(self) -> "Settings":
        if self.db_pool_size + self.db_max_overflow > 100:
            raise ValueError("Total DB connections cannot exceed 100")
        return self
```

---

## Settings in main.py

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use settings during startup
    print(f"Starting {settings.app_name} in {settings.environment} mode")
    yield

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    # Disable docs in production
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)
```

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| BaseSettings | `class Settings(BaseSettings): field: type = default` |
| Load .env file | `SettingsConfigDict(env_file=".env")` |
| Required field | `field: str` (no default = required) |
| Optional field | `field: str \| None = None` |
| Cached singleton | `@lru_cache def get_settings(): return Settings()` |
| Inject in routes | `settings: Settings = Depends(get_settings)` |
| Override in tests | `app.dependency_overrides[get_settings] = lambda: Settings(...)` |
| Multi-env | `Settings(_env_file=f".env.{env}")` |
| List from env | `ALLOWED_HOSTS=localhost,example.com` → `allowed_hosts: list[str]` |
