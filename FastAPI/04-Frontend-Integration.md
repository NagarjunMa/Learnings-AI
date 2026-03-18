# FastAPI: Connecting to a Frontend

## CORS

Browsers block cross-origin requests by default (same-origin policy). If your FastAPI backend is at `api.example.com` and your frontend is at `app.example.com`, requests are blocked without CORS headers.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],  # explicit for prod
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,   # needed if sending cookies or Auth headers
)
```

Dev shortcut (never in prod):
```python
allow_origins=["*"]   # wildcard: accept any origin
```

CORS applies to browser clients only — `curl`, Postman, `requests.post()` are not affected.

## How a JS/React Frontend Calls FastAPI

```javascript
// fetch API
const response = await fetch("https://api.example.com/generate", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify({ prompt: "a cat in space", max_tokens: 200 }),
});
const data = await response.json();
console.log(data.output);
```

```javascript
// axios
import axios from "axios";

const { data } = await axios.post("https://api.example.com/generate", {
    prompt: "a cat in space",
    max_tokens: 200,
}, {
    headers: { Authorization: `Bearer ${token}` }
});
```

Same pattern as the `requests.post()` demo in `10-Serverless-Codebase-Reference.md` — just in-browser JavaScript instead of Python.

## Gradio / Streamlit Frontends

Python-native UIs that call FastAPI via `requests` — no JavaScript required:

```python
# Gradio frontend calling FastAPI backend
import gradio as gr
import requests

BACKEND = "https://your-runpod-pod-url:8000"

def generate_image(prompt: str) -> str:
    response = requests.post(
        f"{BACKEND}/generate",
        json={"prompt": prompt},
        headers={"X-API-Key": "your-key"}
    )
    return response.json()["image_url"]

demo = gr.Interface(fn=generate_image, inputs="text", outputs="text")
demo.launch()
```

Gradio/Streamlit can be a separate service or mounted on FastAPI itself (less common). Useful for internal tooling and demos without needing a full React build.

## File Uploads

```python
from fastapi import File, UploadFile, Form

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form("Describe this image"),
):
    contents = await file.read()   # bytes
    # process contents with model
    return {"filename": file.filename, "size": len(contents)}
```

Client sends `multipart/form-data`. In JS:

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("prompt", "Describe this image");

await fetch("/analyze-image", { method: "POST", body: formData });
```

`UploadFile` gives: `.filename`, `.content_type`, `.read()`, `.seek()`, `.close()`. Large files: use `shutil.copyfileobj(file.file, dest)` to stream instead of reading all into memory.

## WebSockets

Real-time bidirectional communication — ideal for streaming LLM token output:

```python
from fastapi import WebSocket

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        prompt = data["prompt"]

        async for token in model.astream(prompt):
            await websocket.send_text(token)   # stream each token as it's generated

        await websocket.send_json({"done": True})
    except Exception:
        await websocket.close()
```

JS client:

```javascript
const ws = new WebSocket("wss://api.example.com/ws/generate");
ws.onopen = () => ws.send(JSON.stringify({ prompt: "Tell me a story" }));
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.done) ws.close();
    else document.getElementById("output").textContent += data;
};
```

WebSockets bypass CORS entirely — controlled by the `websocket.accept()` call.

## Serving Static Files

Serve a built React/Next.js app from the same FastAPI process:

```python
from fastapi.staticfiles import StaticFiles

# Mount AFTER all API routes to avoid shadowing them
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
```

`html=True` enables SPA fallback — returns `index.html` for unknown paths, so React Router works correctly.

In development: run FastAPI and the React dev server separately (different ports), use CORS. In production: build the React app (`npm run build`), copy output to `frontend/build/`, serve via `StaticFiles`.
