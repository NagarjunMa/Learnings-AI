# Full-Stack Patterns: Persistence, Databases, Production Deployment

## Full-Stack Architecture for AI Systems

```
┌──────────────────────────────────────────────────┐
│ Frontend (React, Vue)                            │
│ - User input form                                │
│ - Results display                                │
│ - Conversation history UI                        │
└────────────────────┬─────────────────────────────┘
                     │ REST/WebSocket
┌────────────────────▼─────────────────────────────┐
│ API Layer (FastAPI, Express)                     │
│ - Request validation (Pydantic)                  │
│ - Rate limiting, auth                            │
│ - Response formatting                            │
└────────────────────┬─────────────────────────────┘
                     │ 
┌────────────────────▼─────────────────────────────┐
│ LLM Orchestration (LangGraph, LangChain)         │
│ - Agent logic                                    │
│ - Tool calling                                   │
│ - State management                               │
└────────────────────┬─────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
[Database]      [Cache]           [Vector DB]
[PostgreSQL]    [Redis]           [Pinecone]
(persistent)    (session)         (retrieval)
```

---

## PostgreSQL Integration for Persistence

### Why PostgreSQL for LLM Systems

- Store conversation history (JSONB columns for flexibility)
- Store embeddings (pgvector extension)
- Store agent state checkpoints (LangGraph StateSaver)
- ACID guarantees (transactions don't lose data on crash)

### Schema Design

```sql
-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    metadata JSONB DEFAULT '{}'
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tokens INT,  -- Track token usage
    created_at TIMESTAMP DEFAULT now(),
    embedding vector(1536)  -- For similarity search (requires pgvector extension)
);

-- Agent state snapshots (checkpointing)
CREATE TABLE agent_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    step_number INT NOT NULL,
    state JSONB NOT NULL,  -- LangGraph StateGraph serialized as JSON
    created_at TIMESTAMP DEFAULT now()
);

-- Tool call logs (debugging + cost tracking)
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tool_name VARCHAR(255) NOT NULL,
    input_tokens INT,
    output_tokens INT,
    latency_ms INT,
    status VARCHAR(50) CHECK (status IN ('success', 'failure', 'timeout')),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- Indexes for fast queries
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_checkpoint_conversation ON agent_checkpoints(conversation_id);
CREATE INDEX idx_tool_calls_conversation ON tool_calls(conversation_id);
```

### Python Integration with SQLAlchemy

```python
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSONB, default={})

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    tokens = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

# Initialize
engine = create_engine("postgresql://user:pass@localhost/ai_app")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Usage
session = Session()
conv = Conversation(user_id=uuid.uuid4())
session.add(conv)
session.commit()

msg = Message(conversation_id=conv.id, role="user", content="Hello")
session.add(msg)
session.commit()

# Retrieve conversation
messages = session.query(Message).filter_by(conversation_id=conv.id).all()
print([m.content for m in messages])
```

---

## Redis for Caching and Sessions

### Use Cases

```
1. Session cache: Store active conversation state
2. Rate limiting: Track API calls per user (sliding window)
3. Prompt cache: Cache frequent prompts + responses
4. LLM response cache: Semantic caching
```

### Example: Session Cache

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def get_or_create_conversation(user_id):
    # Check cache first
    cached = redis_client.get(f"conv:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Cache miss: query database
    conv = db.query(Conversation).filter_by(user_id=user_id).first()
    
    # Store in cache (1 hour expiration)
    redis_client.setex(
        f"conv:{user_id}",
        3600,
        json.dumps({"id": str(conv.id), "created_at": str(conv.created_at)})
    )
    
    return conv

# Usage in FastAPI
@app.post("/chat")
async def chat(user_id: str, message: str):
    conv = await get_or_create_conversation(user_id)
    
    # Add message to database AND cache
    msg = Message(conversation_id=conv.id, role="user", content=message)
    session.add(msg)
    session.commit()
    
    # Invalidate cache
    redis_client.delete(f"conv:{user_id}")
    
    return {"response": "..."}
```

### Rate Limiting with Redis

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")

@app.post("/chat")
@limiter.limit("10/minute")  # 10 requests per minute
async def chat(request: Request, message: str):
    # Rate limit automatically enforced
    return {"response": "..."}

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Try again later."}
    )
```

---

## Vector Database Integration (Pinecone/Weaviate)

### Semantic Caching with Vector DB

Store embeddings of requests + responses. For similar requests, return cached response.

```python
from pinecone import Pinecone
from openai import OpenAI
import hashlib

pinecone_client = Pinecone(api_key="...")
index = pinecone_client.Index("semantic-cache")
openai_client = OpenAI()

async def cached_llm_call(prompt: str, similarity_threshold=0.95):
    # Embed the prompt
    embedding = openai_client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    ).data[0].embedding
    
    # Search similar prompts in cache
    results = index.query(vector=embedding, top_k=1, include_metadata=True)
    
    if results.matches and results.matches[0].score > similarity_threshold:
        # Cache hit: return cached response
        return results.matches[0].metadata["response"]
    
    # Cache miss: call LLM
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
    
    # Store in cache
    index.upsert(
        vectors=[{
            "id": hashlib.md5(prompt.encode()).hexdigest(),
            "values": embedding,
            "metadata": {"prompt": prompt, "response": response}
        }]
    )
    
    return response

# Savings: 50%+ reduction in LLM calls for FAQ-heavy systems
```

---

## Full-Stack Error Handling and Retries

### Exponential Backoff for API Calls

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def llm_call_with_retry(prompt: str):
    try:
        response = await llm.ainvoke(prompt)
        return response
    except RateLimitError:
        raise  # Retry
    except APIError:
        raise  # Retry
    except ValidationError:
        return None  # Don't retry (client error)

# Usage
response = await llm_call_with_retry("What is 2+2?")
```

### Graceful Degradation

```python
async def chat(user_input: str, conversation_id: str):
    try:
        # Try premium model (expensive)
        response = await llm_premium.ainvoke(user_input)
    except RateLimitError:
        try:
            # Fallback to cheaper model
            response = await llm_cheap.ainvoke(user_input)
            logger.warning(f"Rate limited, using fallback model for {conversation_id}")
        except Exception as e:
            # Last resort: return cached response or error message
            cached = await get_cached_response(user_input)
            if cached:
                return cached
            return "I'm experiencing issues. Please try again later."
    
    return response
```

---

## Transaction Management

```python
from sqlalchemy import event
from sqlalchemy.orm import Session

def save_conversation_with_transaction(conv_id: str, messages: list, tools: list):
    """
    Atomicity: Either ALL or NOTHING is saved.
    If error mid-way, database reverts to previous state.
    """
    session = Session()
    
    try:
        # Add all messages
        for msg in messages:
            session.add(Message(conversation_id=conv_id, content=msg["content"]))
        
        # Add all tool calls
        for tool in tools:
            session.add(ToolCall(conversation_id=conv_id, tool_name=tool["name"]))
        
        # Commit atomically
        session.commit()
        
    except Exception as e:
        # Automatic rollback (NEVER partially save)
        session.rollback()
        logger.error(f"Transaction failed for {conv_id}: {e}")
        raise
    finally:
        session.close()

# Usage
save_conversation_with_transaction(conv_id, messages, tools)
# If any save fails, none of them are saved to DB
```

---

## Logging and Debugging

### Structured Logging

```python
import structlog
from pythonjsonlogger import jsonlogger

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# Usage
log.info("chat_request", user_id="user123", conversation_id="conv456", tokens=125)
log.error("llm_failure", tool="search", error="timeout", retry_count=3)

# Outputs JSON:
# {"event": "chat_request", "user_id": "user123", "conversation_id": "conv456", "tokens": 125, "timestamp": "2025-04-22T..."}
```

### Observability Integration

```python
import langsmith
from langsmith import Client

os.environ["LANGSMITH_API_KEY"] = "..."

# Automatic tracing
client = Client()

with client.trace_as_chain_run("chat_request") as run:
    response = await agent.ainvoke({"query": user_input})
    run.add_feedback(
        key="rating",
        value=user_rating,
        comment="User feedback"
    )

# View traces at: https://smith.langchain.com/hub
```

---

## API Response Format Best Practice

```python
from pydantic import BaseModel

class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    tokens_used: int
    latency_ms: int
    status: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start = time.time()
    
    response = await agent.ainvoke(request.message)
    
    return ChatResponse(
        conversation_id=request.conversation_id,
        message_id=str(uuid4()),
        content=response,
        tokens_used=count_tokens(response),
        latency_ms=int((time.time() - start) * 1000),
        status="success"
    )

# Standardized response format, validates outgoing data
```

---

## Production Deployment Checklist

- [ ] Database: PostgreSQL with proper schema, indexes, and ACID guarantees
- [ ] Cache: Redis for sessions, rate limiting, semantic caching
- [ ] Vector DB: Pinecone/Weaviate for retrieval, semantic caching
- [ ] Error handling: Exponential backoff, graceful degradation, circuit breakers
- [ ] Transactions: All-or-nothing saves (atomic commits)
- [ ] Logging: Structured JSON logs, integration with LangSmith/observability
- [ ] Rate limiting: Per-user, per-API-key limits
- [ ] Auth: JWT tokens, refresh rotation
- [ ] Response validation: Pydantic schemas on all endpoints
- [ ] Monitoring: Latency, error rate, cost per request tracked

