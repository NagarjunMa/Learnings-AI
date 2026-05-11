# Testing FastAPI Applications

## Setup

```bash
pip install pytest pytest-asyncio httpx
```

```python
# conftest.py — shared fixtures for all tests
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
```

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
```

---

## TestClient (Sync)

`TestClient` wraps `requests` — works for sync and async route handlers.

```python
# tests/test_users.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_users():
    response = client.get("/users/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_user():
    payload = {"username": "alice", "email": "alice@example.com"}
    response = client.post("/users/", json=payload)
    assert response.status_code == 201
    assert response.json()["username"] == "alice"

def test_get_user_not_found():
    response = client.get("/users/99999")
    assert response.status_code == 404

def test_validation_error():
    response = client.post("/users/", json={"username": ""})  # missing email
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(e["loc"] == ["body", "email"] for e in errors)
```

---

## AsyncClient (Async Routes)

Use `httpx.AsyncClient` with `ASGITransport` for async test functions.

```python
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/items/")
        assert response.status_code == 200

# Or with fixture from conftest.py
async def test_create_item(async_client: AsyncClient):
    response = await async_client.post("/items/", json={"name": "Widget", "price": 9.99})
    assert response.status_code == 201
```

---

## dependency_overrides — Mocking Dependencies

Override any `Depends()` dependency for testing. Critical for isolating DB, auth, external services.

```python
# main.py
from fastapi import FastAPI, Depends

app = FastAPI()

async def get_current_user():
    # Real implementation hits DB
    ...

@app.get("/profile")
async def profile(user=Depends(get_current_user)):
    return {"user": user}
```

```python
# tests/test_profile.py
from fastapi.testclient import TestClient
from main import app, get_current_user

def mock_current_user():
    return {"id": 1, "username": "test_user", "role": "user"}

def mock_admin_user():
    return {"id": 2, "username": "admin", "role": "admin"}

def test_profile_authenticated():
    app.dependency_overrides[get_current_user] = mock_current_user
    client = TestClient(app)
    response = client.get("/profile")
    assert response.status_code == 200
    assert response.json()["user"]["username"] == "test_user"
    app.dependency_overrides.clear()  # always clean up

def test_profile_as_admin():
    app.dependency_overrides[get_current_user] = mock_admin_user
    client = TestClient(app)
    response = client.get("/profile")
    assert response.status_code == 200
    app.dependency_overrides.clear()
```

**Fixture-based cleanup (preferred):**
```python
# conftest.py
@pytest.fixture(autouse=False)
def override_auth():
    app.dependency_overrides[get_current_user] = lambda: {"id": 1, "username": "test"}
    yield
    app.dependency_overrides.clear()

# test uses fixture
def test_protected_route(client, override_auth):
    response = client.get("/protected")
    assert response.status_code == 200
```

---

## Testing Auth-Protected Routes

```python
def test_unauthorized_access():
    client = TestClient(app)
    response = client.get("/admin/dashboard")
    assert response.status_code == 401  # or 403

def test_with_api_key():
    client = TestClient(app)
    response = client.get(
        "/admin/dashboard",
        headers={"X-API-Key": "test-api-key"}
    )
    assert response.status_code == 200

def test_with_bearer_token():
    client = TestClient(app)
    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer valid-test-token"}
    )
    assert response.status_code == 200
```

---

## pytest Fixtures Pattern

```python
# conftest.py — full example
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db
from main import app

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def client(db):
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

---

## Testing File Uploads

```python
import io

def test_upload_file(client):
    file_content = b"fake image data"
    response = client.post(
        "/upload/",
        files={"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "test.jpg"

def test_upload_multiple_files(client):
    response = client.post(
        "/upload-multiple/",
        files=[
            ("files", ("a.txt", b"content a", "text/plain")),
            ("files", ("b.txt", b"content b", "text/plain")),
        ],
    )
    assert response.status_code == 200
```

---

## Testing WebSockets

```python
def test_websocket(client):
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("hello")
        data = websocket.receive_text()
        assert data == "echo: hello"

def test_websocket_json(client):
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json({"message": "ping"})
        response = websocket.receive_json()
        assert response["message"] == "pong"

def test_websocket_disconnect(client):
    with client.websocket_connect("/ws") as websocket:
        websocket.close()
        # verify clean disconnect handling
```

---

## Testing Background Tasks

Background tasks run after response — use `TestClient` which waits for them by default.

```python
from unittest.mock import patch, MagicMock

def test_background_task_called(client):
    with patch("routers.notifications.send_email") as mock_email:
        response = client.post("/register", json={"email": "user@test.com"})
        assert response.status_code == 201
        # TestClient finishes background tasks before returning
        mock_email.assert_called_once_with("user@test.com")
```

---

## Testing Exception Handlers

```python
def test_custom_404(client):
    response = client.get("/nonexistent-route")
    assert response.status_code == 404
    # Test your custom error format
    body = response.json()
    assert "error" in body
    assert "message" in body

def test_validation_error_format(client):
    response = client.post("/users/", json={"username": 123})  # wrong type
    assert response.status_code == 422
    # Verify your custom 422 format
    assert response.json()["error"] == "validation_error"
```

---

## Testing Streaming Responses

```python
def test_streaming_response(client):
    with client.stream("GET", "/stream/data") as response:
        assert response.status_code == 200
        chunks = list(response.iter_text())
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "data:" in full_text  # SSE format
```

---

## Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("username,expected_status", [
    ("alice", 201),
    ("", 422),          # empty string fails validation
    ("a" * 51, 422),    # too long
    ("user@invalid", 422),  # special chars
])
def test_username_validation(client, username, expected_status):
    response = client.post("/users/", json={"username": username, "email": "test@test.com"})
    assert response.status_code == expected_status
```

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| Sync test client | `TestClient(app)` |
| Async test client | `AsyncClient(transport=ASGITransport(app=app), base_url="http://test")` |
| Mock dependency | `app.dependency_overrides[dep_fn] = mock_fn` |
| Clear overrides | `app.dependency_overrides.clear()` |
| Auth header | `client.get("/path", headers={"Authorization": "Bearer token"})` |
| File upload | `client.post("/upload", files={"file": ("name.jpg", bytes, "image/jpeg")})` |
| WebSocket | `with client.websocket_connect("/ws") as ws: ws.send_text("msg")` |
| Async test | `@pytest.mark.asyncio async def test_fn():` |
| Auto asyncio mode | `asyncio_mode = auto` in pytest.ini |
