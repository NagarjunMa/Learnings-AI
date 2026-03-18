# FastAPI: Authentication Patterns

## API Key Auth

Simplest pattern. Caller sends key in a header or query param:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = "super-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key

@app.post("/generate")
def generate(req: GenerateRequest, _: str = Depends(verify_api_key)):
    ...
```

Client sends: `X-API-Key: super-secret-key` header.

For query param variant: `APIKeyQuery(name="api_key")` — less preferred (key ends up in server logs).

## HTTP Basic Auth

```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, "admin")
    correct_pass = secrets.compare_digest(credentials.password, "password123")
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
```

`secrets.compare_digest` prevents timing attacks. Use only over HTTPS — credentials are base64-encoded, not encrypted.

## Bearer Token / JWT

Most common pattern for production APIs. Flow:
1. Client authenticates → receives JWT
2. Client sends `Authorization: Bearer <token>` on every subsequent request
3. Server decodes + verifies JWT → extracts user identity

```python
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id

@app.get("/me")
def get_me(user_id: str = Depends(get_current_user)):
    return {"user_id": user_id}
```

Libraries: `python-jose[cryptography]` or `PyJWT`.

## OAuth2 Password Flow

Full `/token` endpoint — username + password → JWT:

```python
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    access_token = jwt.encode(
        {"sub": user.id, "exp": datetime.utcnow() + timedelta(hours=24)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {"access_token": access_token, "token_type": "bearer"}
```

`OAuth2PasswordRequestForm` expects `application/x-www-form-urlencoded` (not JSON) — this is the OAuth2 spec.

Full flow:
```
Client → POST /token (username, password form data)
Server → { access_token: "eyJ...", token_type: "bearer" }
Client → POST /generate { Authorization: Bearer eyJ... }
Server → validates JWT → runs inference → returns result
```

## Dependency Chain

Authentication is a natural dependency chain:

```
OAuth2PasswordBearer (extracts raw token string from header)
    ↓
get_current_user (decodes JWT, fetches user from DB)
    ↓
Route handler (receives typed User object)
```

```python
# Route just declares what it needs — chain runs automatically
@app.post("/generate")
def generate(req: GenerateRequest, user: User = Depends(get_current_user)):
    log_usage(user.id, req.prompt)
    return run_inference(req.prompt)
```

## Scopes / Role-Based Access

```python
from fastapi import Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/token",
    scopes={"admin": "Full access", "inference": "Run inference only"}
)

def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    token_scopes = payload.get("scopes", [])
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
    return payload

@app.delete("/models/{model_id}")
def delete_model(model_id: str, user = Security(get_current_user, scopes=["admin"])):
    ...
```

## RunPod Context

RunPod Pods expose a port directly to the internet. Without auth:
- Anyone who discovers the URL can run your GPU and rack up costs
- No rate limiting by default

**Recommended**: Bearer + JWT, or API key via header.

```python
# Minimal protection for a RunPod Pod
API_KEY = os.environ["RUNPOD_API_SECRET"]   # set in Pod env vars

def check_auth(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403)

@app.post("/generate")
def generate(req: GenerateRequest, _=Depends(check_auth)):
    ...
```

Never hardcode secrets — use environment variables injected at Pod startup.
