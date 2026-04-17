# Bedrock Invoke Functions & Streaming — Complete Reference

This file is your interview insurance. If asked "how do you stream a Bedrock response?" you now have the answer. If asked "which invoke function for RAG?" you can explain the difference in 30 seconds.

---

## Decision Tree: Which Invoke Function?

```
┌─ Need model text output?
│  ├─ One-shot, no history, sync acceptable?     → invoke_model
│  ├─ One-shot, streaming required?              → invoke_model_with_response_stream
│  ├─ Multi-turn conversation, sync?             → converse
│  ├─ Multi-turn conversation, streaming?        → converse_stream
│  │
│  └─ RAG required?
│     ├─ Retrieve chunks only (your generation)?  → retrieve
│     ├─ Retrieve + generate (one call, sync)?    → retrieve_and_generate
│     └─ Retrieve + generate (streaming)?         → retrieve_and_generate_stream
│
├─ Need Bedrock Agent orchestration?
│  └─ Tool calling, knowledge base search, etc?   → invoke_agent (bedrock-agent-runtime)
│
└─ Need AgentCore runtime (new GA Oct 2025)?
   └─ Custom agent framework in microVM?          → invoke_agent_runtime (via Control Plane API)
```

---

## 1. invoke_model — Synchronous One-Shot

**Client:** `bedrock-runtime`
**Blocks:** Until complete
**Use:** Batch processing, offline, where latency doesn't matter

### Request Shape

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Claude body schema
body = {
    "anthropic_version": "bedrock-2023-06-01",
    "max_tokens": 1024,
    "system": "You are a financial analyst.",
    "messages": [
        {
            "role": "user",
            "content": "What is our Q3 revenue?"
        }
    ]
}

response = client.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body)
)

# Parse response
result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

### Response Parsing

```python
# CloudWatch note: response["body"] is a StreamingBody (boto3 internals)
# You MUST call .read() to get bytes, then parse

body_bytes = response["body"].read()  # Get all bytes at once
result = json.loads(body_bytes)        # Parse JSON

# Claude response structure
print(result["content"][0]["text"])        # The assistant message
print(result["stop_reason"])               # "end_turn" or "max_tokens"
print(result["usage"])                     # {"input_tokens": X, "output_tokens": Y}
```

### Per-Model Body Schemas

**Claude (anthropic.claude-3-5-sonnet-*):**
```python
{
    "anthropic_version": "bedrock-2023-06-01",
    "max_tokens": 1024,
    "system": "optional string or list of content blocks",
    "messages": [{"role": "user|assistant", "content": "..."}]
}
```

**Llama 2 (meta.llama2-13b-chat-v1):**
```python
{
    "prompt": "<s>[INST] What is 2+2? [/INST]",
    "max_gen_len": 512,
    "temperature": 0.7,
    "top_p": 0.9
}
```

**Titan Text (amazon.titan-text-express-v1):**
```python
{
    "inputText": "What is 2+2?",
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.7
    }
}
```

**Cohere Command (cohere.command-light-text-v14):**
```python
{
    "prompt": "What is 2+2?",
    "max_tokens": 512,
    "temperature": 0.7
}
```

**Gotcha:** Each model family has a different body schema. You MUST know which model you're using. Use `converse` instead to avoid this entirely.

---

## 2. invoke_model_with_response_stream — Streaming One-Shot

**Client:** `bedrock-runtime`
**Blocks:** No, returns EventStream immediately
**Use:** UI responses, real-time feedback, where latency matters

### Request and EventStream Iteration

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

body = {
    "anthropic_version": "bedrock-2023-06-01",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Explain quantum computing in 3 sentences"}]
}

response = client.invoke_model_with_response_stream(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body)
)

# response["body"] is an EventStream
for event in response["body"]:
    if "chunk" in event:
        # This is a content chunk
        chunk_data = json.loads(event["chunk"]["bytes"])

        # Claude format
        if chunk_data.get("type") == "content_block_delta":
            delta = chunk_data.get("delta", {})
            if delta.get("type") == "text_delta":
                print(delta["text"], end="", flush=True)

        # messageStop signals end (no more chunks coming)
        if chunk_data.get("type") == "message_stop":
            usage = chunk_data.get("message", {}).get("usage", {})
            print(f"\n[Tokens: {usage}]")
```

### EventStream Event Types

Each event in the stream has exactly one of these keys:

| Event Type | Contains | Meaning |
|---|---|---|
| `chunk` | `{"bytes": b"..."}` | Content delta or metadata |
| `internalServerException` | Error details | Model crashed |
| `modelStreamErrorException` | Error details | Model error mid-stream |
| `throttlingException` | Error details | Rate limited |
| `validationException` | Error details | Bad request |

### Decoding Chunks — Per Provider

**Claude:**
```python
chunk_data = json.loads(event["chunk"]["bytes"])
# Types: "message_start", "content_block_start", "content_block_delta", "message_stop"

if chunk_data["type"] == "content_block_delta":
    text = chunk_data["delta"]["text"]  # incremental text
```

**Llama 2, Titan, Cohere:**
Each has different chunk format. **Do not mix**. Use `converse_stream` instead.

### Full Example with Error Handling

```python
def stream_claude_response(prompt, max_tokens=1024):
    """Stream Claude response, accumulating until end."""
    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    body = {
        "anthropic_version": "bedrock-2023-06-01",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = client.invoke_model_with_response_stream(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    full_text = ""
    usage = {}

    try:
        for event in response["body"]:
            if "chunk" in event:
                chunk_data = json.loads(event["chunk"]["bytes"])

                if chunk_data["type"] == "content_block_delta":
                    full_text += chunk_data["delta"]["text"]
                    print(chunk_data["delta"]["text"], end="", flush=True)

                elif chunk_data["type"] == "message_stop":
                    usage = chunk_data["message"]["usage"]

            elif "modelStreamErrorException" in event:
                raise Exception(f"Model error: {event['modelStreamErrorException']}")

            elif "throttlingException" in event:
                raise Exception("Rate limited. Retry with exponential backoff.")

    except Exception as e:
        print(f"\nStream error: {e}")
        # NOTE: partial content in full_text is lost if stream fails mid-response
        raise

    return full_text, usage
```

### Gotcha: Stream Failure = Lost Partial Content

If the stream breaks mid-response (network error, model crash), `full_text` contains only what arrived before the break. There is no way to recover. Always assume a streaming response is incomplete if an exception occurs.

---

## 3. converse — Modern Unified Synchronous API

**Client:** `bedrock-runtime`
**Blocks:** Until complete
**Use:** When you want the same API for Claude, Llama, Titan, Cohere, Mistral without per-model body differences

### Request Shape (Same for All Models)

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-west-2")

response = client.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    # or "meta.llama3-8b-instruct-v1:0"
    # or "amazon.titan-text-express-v1"
    # or "cohere.command-light-text-v14"
    # All work with identical API

    system=[
        {
            "text": "You are a financial analyst. Be concise."
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "What was our Q3 2025 revenue?"
                }
            ]
        }
    ],
    inferenceConfig={
        "maxTokens": 1024,
        "temperature": 0.7,
        "topP": 0.9,
        "stopSequences": ["END"]
    }
)

# Normalized response
text = response["output"]["message"]["content"][0]["text"]
usage = response["usage"]  # {"inputTokens": X, "outputTokens": Y}
stop_reason = response["stopReason"]  # "end_turn", "max_tokens", "stop_sequence"

print(text)
```

### Why converse > invoke_model

| Aspect | invoke_model | converse |
|---|---|---|
| Body schema | Per-model (Claude ≠ Llama ≠ Titan) | Identical across all |
| Response parsing | Model-specific | Normalized |
| System prompt | In body.system (string or list per model) | Unified list format |
| Temperature/topP | In inferenceConfig per model | Unified inferenceConfig |
| Multi-turn | Must manage messages manually | Same API |
| Tool calling | Partial support | Full support (future-proof) |

**Recommendation:** Use `converse` / `converse_stream` for everything. Only use `invoke_model` if:
- You need model-specific low-level control
- You're batch processing and already abstracted the body schema
- You're tightly tied to a legacy integration

---

## 4. converse_stream — Production Streaming API

**Client:** `bedrock-runtime`
**Blocks:** No, returns EventStream
**Use:** Real-time UI, Server-Sent Events, chat applications

### Request (Same as converse, returns stream)

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

response = client.converse_stream(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    system=[{"text": "You are a helpful assistant."}],
    messages=[{"role": "user", "content": [{"text": "Explain black holes"}]}],
    inferenceConfig={
        "maxTokens": 2048,
        "temperature": 0.7
    }
)

# response["stream"] is an EventStream
stream = response["stream"]

for event in stream:
    if "messageStart" in event:
        print(f"[Message from {event['messageStart']['message']['role']}]")

    elif "contentBlockStart" in event:
        block_index = event["contentBlockStart"]["contentBlockIndex"]
        print(f"[Block {block_index}]", end="")

    elif "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]["delta"]
        if delta.get("type") == "text_delta":
            print(delta["text"], end="", flush=True)

    elif "contentBlockStop" in event:
        print()

    elif "messageStop" in event:
        stop_reason = event["messageStop"]["stopReason"]
        print(f"[Stop: {stop_reason}]")

    elif "metadata" in event:
        usage = event["metadata"].get("usage", {})
        metrics = event["metadata"].get("metrics", {})
        print(f"[Tokens: input={usage.get('inputTokens')}, output={usage.get('outputTokens')}]")
        print(f"[Latency: {metrics.get('latencyMs')}ms]")
```

### Event Types in the Stream

| Event Type | Key | Purpose |
|---|---|---|
| Message Start | `messageStart` | Role of responder (assistant) |
| Content Block Start | `contentBlockStart` | Block type (text, tool use) and index |
| Content Block Delta | `contentBlockDelta` | Incremental text/tool delta |
| Content Block Stop | `contentBlockStop` | End of block |
| Message Stop | `messageStop` | `stopReason`: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use` |
| Metadata | `metadata` | Token usage, latency metrics |
| Errors | `internalServerException`, etc. | Same as `invoke_model_with_response_stream` |

### Production Pattern: Buffer Accumulation

```python
def stream_chat_response_buffered(prompt, system_text=""):
    """Stream response but only yield complete message at end."""
    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    full_text = ""
    usage_info = {}

    try:
        response = client.converse_stream(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system=[{"text": system_text}] if system_text else [],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2048}
        )

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if delta.get("type") == "text_delta":
                    full_text += delta["text"]

            elif "messageStop" in event:
                # Message complete, safe to yield
                stop_reason = event["messageStop"]["stopReason"]
                if stop_reason == "max_tokens":
                    print("[WARNING: Response truncated by max_tokens]")

            elif "metadata" in event:
                usage_info = event["metadata"].get("usage", {})

    except Exception as e:
        print(f"Stream error: {e}")
        # Partial content in full_text; you decide if it's returnable
        if len(full_text) > 0:
            print(f"[Partial response: {len(full_text)} chars before error]")
        raise

    return full_text, usage_info
```

### FastAPI SSE Integration

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import boto3
import json

app = FastAPI()
client = boto3.client("bedrock-runtime", region_name="us-west-2")

async def generate_sse_events(prompt: str):
    """Generator for Server-Sent Events."""
    try:
        response = client.converse_stream(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system=[{"text": "You are a helpful assistant."}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2048}
        )

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if delta.get("type") == "text_delta":
                    # SSE format: "data: <json>\n\n"
                    sse_data = json.dumps({"type": "text", "content": delta["text"]})
                    yield f"data: {sse_data}\n\n"

            elif "messageStop" in event:
                yield 'data: {"type": "done"}\n\n'

    except Exception as e:
        yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'

@app.get("/chat/stream")
async def chat_stream(prompt: str):
    """Stream chat response to browser."""
    return StreamingResponse(
        generate_sse_events(prompt),
        media_type="text/event-stream"
    )

# Browser JavaScript:
# const es = new EventSource(`/chat/stream?prompt=${encodeURIComponent(prompt)}`);
# es.addEventListener("message", (event) => {
#   const data = JSON.parse(event.data);
#   if (data.type === "text") {
#     document.getElementById("response").innerText += data.content;
#   }
# });
```

### Async Streaming with aioboto3

```python
import aioboto3
import json
import asyncio

async def stream_chat_async(prompt: str):
    """Non-blocking stream using asyncio."""
    session = aioboto3.Session()

    async with session.client("bedrock-runtime", region_name="us-west-2") as client:
        response = await client.converse_stream(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2048}
        )

        full_text = ""
        async for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if delta.get("type") == "text_delta":
                    full_text += delta["text"]
                    print(delta["text"], end="", flush=True)
                    await asyncio.sleep(0)  # Yield to event loop

        return full_text

# Usage
result = asyncio.run(stream_chat_async("Explain quantum computing"))
```

---

## 5. retrieve — Knowledge Base Query Only (No Generation)

**Client:** `bedrock-agent-runtime` (NOT bedrock-runtime!)
**Returns:** List of retrieval results (chunks + scores)
**Use:** When you want to implement your own generation logic or use a different model

### Request

```python
import boto3

# NOTE: Different client!
client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")

response = client.retrieve(
    knowledgeBaseId="XXXXXX",  # From KB creation
    retrievalQuery={
        "text": "What was our Q3 2025 revenue?"
    },
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 5,  # Return top 5 chunks
            "overrideSearchType": "HYBRID"  # HYBRID (keyword + semantic) or SEMANTIC
        }
    }
)

# Response contains retrieved chunks
for result in response["retrievalResults"]:
    content_text = result["content"]["text"]
    relevance_score = result["score"]  # 0.0-1.0
    source_location = result["location"]["s3Location"]["uri"]

    print(f"[Score: {relevance_score:.2f}] {source_location}")
    print(f"{content_text}\n")
```

### When to Use retrieve

- You want custom generation (e.g., different model, custom prompt)
- You need post-processing of chunks before passing to LLM (filtering, ranking, synthesis)
- You're building a research system where retrieval and generation are separate concerns
- You want to implement your own caching/deduplication of chunks

### When NOT to Use retrieve

- You just want Q&A over documents → use `retrieve_and_generate`
- You want one API call for RAG → use `retrieve_and_generate`

---

## 6. retrieve_and_generate — RAG in One Call

**Client:** `bedrock-agent-runtime`
**Returns:** Generated text + citations
**Use:** Standard RAG: retrieve documents, generate answer, in one call

### Request

```python
import boto3

client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")

response = client.retrieve_and_generate(
    input={
        "text": "What was our Q3 2025 revenue and profit margins?"
    },
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "XXXXXX",
            "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "overrideSearchType": "HYBRID"
                }
            },
            "generationConfiguration": {
                "inferenceConfig": {
                    "maxTokens": 2048,
                    "temperature": 0.7
                },
                "promptTemplate": {
                    "textPromptTemplate": "Answer based on these documents: {{CONTEXT}}\n\nQuestion: {{INPUT}}"
                }
            }
        }
    }
)

# Structured response
answer_text = response["output"]["text"]
citations = response["citations"]  # List of sources used

print(answer_text)
print("\nSources:")
for citation in citations:
    for source in citation["retrievedReferences"]:
        print(f"  - {source['location']['s3Location']['uri']}")
```

### EXTERNAL_SOURCES Configuration

If you want to use an external vector DB (Pinecone, Milvus, etc.) instead of Bedrock KB:

```python
response = client.retrieve_and_generate(
    input={"text": "..."},
    retrieveAndGenerateConfiguration={
        "type": "EXTERNAL_SOURCES",
        "externalSourcesConfiguration": {
            "sources": [
                {
                    "sourceType": "S3",
                    "s3Location": {
                        "uri": "s3://my-bucket/documents/"
                    }
                },
                {
                    "sourceType": "CUSTOM",
                    "customExternalSourcesConfiguration": {
                        "authenticationType": "API_KEY",
                        # Your vector DB connection details
                    }
                }
            ],
            "modelArn": "arn:aws:bedrock:...",
            "generationConfiguration": {...}
        }
    }
)
```

---

## 7. retrieve_and_generate_stream — RAG Streaming

**Client:** `bedrock-agent-runtime`
**Returns:** EventStream (like converse_stream)
**Use:** When you want streaming response in RAG (don't wait for full answer)

### Request

```python
import boto3
import json

client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")

response = client.retrieve_and_generate_stream(
    input={
        "text": "What was our Q3 revenue?"
    },
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "XXXXXX",
            "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        }
    }
)

# Stream the response
for event in response["stream"]:
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]["delta"]
        if delta.get("type") == "text_delta":
            print(delta["text"], end="", flush=True)

    elif "generatedResponsePart" in event:
        # Final event with citations
        citations = event["generatedResponsePart"].get("citations", [])
        for citation in citations:
            for ref in citation["retrievedReferences"]:
                print(f"\nSource: {ref['location']['s3Location']['uri']}")
```

### Gotcha: Streaming RAG is Slower

Streaming starts before retrieval completes. The model generates while chunks are still being retrieved. This can cause:
- Hallucination (LLM has incomplete context)
- Regeneration (LLM refines answer as more chunks arrive)

Use `retrieve_and_generate` (non-streaming) if you need a single coherent answer.

---

## 8. invoke_agent — Bedrock Agents (Always Streaming)

**Client:** `bedrock-agent-runtime` (NOT bedrock-runtime)
**Returns:** EventStream (always, no sync option)
**Use:** When you want the LLM to orchestrate tool calls, knowledge base search, and reasoning

### Mental Model

```
User Query
   ↓
[Bedrock Agent]
   ├─ Thinks: "Do I need to search KB?"
   ├─ Calls: search_knowledge_base tool
   ├─ Gets: Retrieved chunks
   ├─ Thinks: "Do I need to calculate?"
   ├─ Calls: calculator tool
   ├─ Gets: Result
   ├─ Thinks: "I can now answer"
   └─ Responds: Final answer

You iterate the EventStream:
   ├─ chunk events: pieces of the final answer
   ├─ trace events: reasoning steps (tool calls, KB searches)
   └─ returnControl events: agent pauses, needs your help
```

### Request and EventStream

```python
import boto3
import json

client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")

response = client.invoke_agent(
    agentId="XXXXXX",          # From agent creation
    agentAliasId="YYYYYYY",    # From agent alias creation
    sessionId="user-123",      # Conversation thread ID
    inputText="What was our revenue and is it growing?",
    enableTrace=True  # Include reasoning steps in trace events
)

full_answer = ""
traces = []

for event in response["output"]:
    if "chunk" in event:
        # Final answer text (streaming)
        chunk_data = json.loads(event["chunk"]["bytes"])
        text = chunk_data.get("text", "")
        full_answer += text
        print(text, end="", flush=True)

    elif "trace" in event:
        # Reasoning: tool calls, KB searches, decisions
        trace = event["trace"]["trace"]
        traces.append(trace)
        # trace format: {"type": "tool_use", "toolName": "...", "input": {...}, "output": {...}}

    elif "returnControl" in event:
        # Agent needs your help: execute a tool and resume
        invoke_input = event["returnControl"]["invocationInputs"]
        tool_name = invoke_input["toolUseBlock"]["name"]
        tool_input = invoke_input["toolUseBlock"]["input"]

        print(f"[Agent requests: {tool_name}({tool_input})]")

        # You execute the tool
        tool_result = execute_custom_tool(tool_name, tool_input)

        # Resume agent with the result
        response2 = client.invoke_agent(
            agentId="XXXXXX",
            agentAliasId="YYYYYYY",
            sessionId="user-123",
            inputText=full_answer,  # Continue from where we paused
            sessionState={
                "invocationId": event["returnControl"]["invocationId"],
                "returnControlInvocationResults": [
                    {
                        "functionResult": {
                            "actionGroup": "custom_tools",
                            "function": tool_name,
                            "functionResponse": {
                                "responseBody": {
                                    "text": json.dumps(tool_result)
                                }
                            }
                        }
                    }
                ]
            }
        )

        # Continue iterating the new EventStream
        for event2 in response2["output"]:
            # ... process like before

print(f"\n\nFull answer:\n{full_answer}")
print(f"\nReasoning trace ({len(traces)} steps)")
```

### returnControl Pattern (Advanced)

When the agent encounters a tool it cannot execute (custom tool not implemented in Bedrock), it pauses and yields a `returnControl` event:

```python
if "returnControl" in event:
    # Structure:
    event["returnControl"] = {
        "invocationId": "step-3",  # Unique ID for this pause
        "invocationInputs": {
            "toolUseBlock": {
                "toolUseId": "tbid-123",
                "name": "query_customer_database",
                "input": {"customer_id": "cust-456"}
            }
        }
    }

    # You execute the tool
    customer = query_customer_db({"customer_id": "cust-456"})

    # Resume with the result
    response2 = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        inputText=original_query,
        sessionState={
            "invocationId": event["returnControl"]["invocationId"],
            "returnControlInvocationResults": [
                {
                    "functionResult": {
                        "actionGroup": "custom",  # Or your action group
                        "function": "query_customer_database",
                        "functionResponse": {
                            "responseBody": {
                                "text": json.dumps(customer)
                            }
                        }
                    }
                }
            ]
        }
    )
```

### Client vs Model Responsibility

| Responsibility | Example |
|---|---|
| **Bedrock Agent** | Decides when to call tools, interprets results, reasons |
| **You (returnControl)** | Execute custom DB queries, call private APIs |

Bedrock Agents cannot call arbitrary APIs (security + isolation). They can call:
- Bedrock Knowledge Base (built-in)
- Bedrock Guardrails (built-in)
- Tools defined in agent (custom, but you implement via returnControl)

---

## 9. Streaming Error Handling

### EventStream Exceptions

When iterating an EventStream, you can hit these:

```python
for event in response_stream:
    if "internalServerException" in event:
        # Bedrock crashed
        raise Exception(f"500: {event['internalServerException']}")

    elif "modelStreamErrorException" in event:
        # Model crashed mid-generation
        raise Exception(f"Model error: {event['modelStreamErrorException']}")

    elif "throttlingException" in event:
        # Rate limited
        raise Exception("Rate limited. Retry with exponential backoff.")

    elif "validationException" in event:
        # Bad request (malformed, invalid model ID, etc.)
        raise Exception(f"Validation error: {event['validationException']}")
```

### Retry Pattern (Exponential Backoff)

```python
import time
import random

def stream_with_retry(prompt, max_retries=3):
    """Stream with exponential backoff on throttling."""
    for attempt in range(max_retries):
        try:
            response = client.converse_stream(
                modelId="...",
                messages=[{"role": "user", "content": [{"text": prompt}]}]
            )

            for event in response["stream"]:
                if "throttlingException" in event:
                    raise ThrottlingException()
                # ... process event

            return  # Success

        except ThrottlingException:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"Throttled. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

        except Exception as e:
            # Non-retryable error
            raise
```

### Partial Responses

**Critical gotcha:** If a stream fails mid-response, you have accumulated partial content but no way to know if it's complete.

```python
full_text = ""
try:
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            full_text += event["contentBlockDelta"]["delta"]["text"]
except Exception as e:
    # full_text now contains only the part before the error
    if full_text:
        print(f"[Partial response ({len(full_text)} chars): {full_text[:100]}...]")
    # Decision: return partial? retry? depends on your UX
    raise
```

---

## 10. Interview Cheat Sheet

| Function | Client | Streaming | Model Schema | Use Case | Gotcha |
|---|---|---|---|---|---|
| `invoke_model` | bedrock-runtime | No | Per-model (Claude ≠ Llama) | Batch, offline | Sync blocks; must know body schema |
| `invoke_model_with_response_stream` | bedrock-runtime | Yes | Per-model | Real-time UI | Partial content on error; per-model chunk format |
| `converse` | bedrock-runtime | No | Unified | Multi-turn, any model | Normalizes everything (recommended) |
| `converse_stream` | bedrock-runtime | Yes | Unified | Production chat | Same API for all models; cleanest |
| `retrieve` | bedrock-agent-runtime | No | N/A | Custom generation logic | Different client; retrieval only |
| `retrieve_and_generate` | bedrock-agent-runtime | No | N/A | Standard RAG | Different client; one-call |
| `retrieve_and_generate_stream` | bedrock-agent-runtime | Yes | N/A | Streaming RAG | Can hallucinate (incomplete context) |
| `invoke_agent` | bedrock-agent-runtime | Always | N/A | Agent orchestration | Always returns EventStream; returnControl for custom tools |

---

## 11. Quick Study Checklist for Interviews

### Concepts
- [ ] Bedrock has 7+ invoke functions; each solves a different problem
- [ ] `converse` and `converse_stream` are the modern APIs (use these)
- [ ] Streaming returns `EventStream` (not strings); requires iteration
- [ ] Different clients: `bedrock-runtime` vs `bedrock-agent-runtime`
- [ ] Knowledge Base APIs use `bedrock-agent-runtime`, not `bedrock-runtime`
- [ ] `invoke_agent` always streams; no sync option
- [ ] `invoke_agent` yields `returnControl` for custom tool execution
- [ ] Streaming failures lose partial content (no recovery)
- [ ] `retrieve_and_generate_stream` can hallucinate (incomplete KB context)

### Code Patterns
- [ ] Claude body schema for `invoke_model`: `anthropic_version`, `messages`, `max_tokens`
- [ ] EventStream iteration: `for event in response["body"]`; check `"chunk"` key
- [ ] `converse_stream` event types: `messageStart`, `contentBlockDelta`, `messageStop`, `metadata`
- [ ] FastAPI SSE: yield `f"data: {json}\n\n"`
- [ ] Async streaming with `aioboto3`: `async for event in response["stream"]`
- [ ] `retrieve_and_generate` + citations: `response["citations"]`
- [ ] `invoke_agent` with custom tools: catch `returnControl`, execute tool, resume with `sessionState.returnControlInvocationResults`

### Decision Flowchart
- [ ] Just want text output? → `converse` or `converse_stream`
- [ ] Need RAG? → `retrieve_and_generate` or `retrieve_and_generate_stream`
- [ ] Need KB retrieval only (custom generation)? → `retrieve`
- [ ] Need agent orchestration? → `invoke_agent`
- [ ] Need streaming? → Append `_stream` to any function; or use `invoke_model_with_response_stream`

---

## 12. Production Patterns

### Pattern 1: Financial Analysis Agent with Guardrails

```python
# Setup
kb_id = "xxxxxx"
agent_id = "xxxxxx"
agent_alias_id = "yyyyyy"
guardrail_id = "zzzzzz"

# User query
user_query = "What is our total debt and interest coverage ratio?"

# Invoke agent with guardrails (if Bedrock Agents supports guardrails)
response = client.invoke_agent(
    agentId=agent_id,
    agentAliasId=agent_alias_id,
    sessionId=user_id,
    inputText=user_query,
    enableTrace=True,
    # Note: Classic Bedrock Agents don't support guardrails directly
    # Use converse_stream + manual KB retrieval if guardrails required
)

full_answer = ""
for event in response["output"]:
    if "chunk" in event:
        chunk_data = json.loads(event["chunk"]["bytes"])
        full_answer += chunk_data.get("text", "")
        print(chunk_data.get("text", ""), end="", flush=True)

# Log for compliance
audit_log(user_id, user_query, full_answer, timestamp=now)
```

### Pattern 2: Streaming Chat with SSE

```python
# FastAPI endpoint
@app.get("/chat/stream")
async def chat_stream(prompt: str):
    async def generate():
        async with aioboto3.Session().client("bedrock-runtime", region_name="us-west-2") as client:
            response = await client.converse_stream(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 2048}
            )

            async for event in response["stream"]:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if delta.get("type") == "text_delta":
                        yield f"data: {json.dumps({'text': delta['text']})}\n\n"

                elif "messageStop" in event:
                    yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Pattern 3: RAG with Custom Tool Fallback

```python
# Attempt RAG first; fall back to web search if confidence is low
def rag_with_fallback(query: str, confidence_threshold: float = 0.7):
    # Try Knowledge Base
    rag_response = client.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {...}
        }
    )

    answer = rag_response["output"]["text"]

    # Check if we should fall back
    # (Rough heuristic: if answer is very short, confidence is low)
    if len(answer) < 100:
        print("[Low confidence. Attempting web search...]")
        answer = web_search_fallback(query)

    return answer
```

---

## Interview Talking Points

**Q: "How do you stream a Bedrock response?"**

A: Use `converse_stream`. It returns an EventStream; iterate it with `for event in response["stream"]`. Each event has keys like `contentBlockDelta` (text), `messageStop` (end), `metadata` (usage). Yield deltas to the client in real-time. If the stream fails mid-response, partial content is lost — there's no recovery.

**Q: "What's the difference between `invoke_model` and `converse`?"**

A: `invoke_model` requires per-model body schemas (Claude ≠ Llama ≠ Titan). `converse` unifies the API across all models. Use `converse` for everything unless you need low-level model-specific control.

**Q: "How does RAG work in Bedrock?"**

A: Three options:
1. `retrieve`: Get KB chunks, then generate yourself (custom model, prompt)
2. `retrieve_and_generate`: One-call RAG (retrieves, generates, returns citations)
3. `retrieve_and_generate_stream`: Same as #2 but streaming (watch for hallucination — incomplete KB context)

**Q: "What is `returnControl` in `invoke_agent`?"**

A: When Bedrock Agent encounters a tool it can't execute (custom DB query, private API), it pauses and yields a `returnControl` event with tool details. You execute the tool, then resume the agent with `sessionState.returnControlInvocationResults` containing the result. The agent continues reasoning.

**Q: "How do you handle streaming errors?"**

A: Catch exceptions in the EventStream loop. Throttling → retry with exponential backoff. Model crash → unrecoverable (partial content lost). Best practice: accumulate in a buffer and only return after `messageStop` event.
