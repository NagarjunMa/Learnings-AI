# LLM Cost Optimization — Caching, Routing, Token Budgeting

At scale, LLM costs dominate (OpenAI gpt-4-turbo = $10-$30 per 1M tokens input). Cost must be engineered, not hoped.

## Token Economics

Anthropic pricing (Claude 3):
- Sonnet: $3/$15 per 1M tokens (input/output)
- Opus: $15/$75 per 1M tokens (input/output)
- Haiku: $0.80/$4 per 1M tokens (input/output)

1000-user app, 10 queries/user/day, 500 input tokens avg, 100 output tokens avg:

```
1000 users × 10 queries × 500 input = 5M input tokens/day
1000 users × 10 queries × 100 output = 1M output tokens/day

Cost (Sonnet): (5M × $3 + 1M × $15) / 1M = $18/day = $540/month
Cost (Opus): (5M × $15 + 1M × $75) / 1M = $90/day = $2,700/month
```

**Every 10% reduction in tokens = 10% cost savings.**

## Strategy 1: Prompt Caching

Anthropic/OpenAI offer built-in prompt caching. Cache the context, pay 10% of input token cost for cache hits.

### Anthropic API (claude-3-5-sonnet)

```python
import anthropic

client = anthropic.Anthropic()

# System prompt (e.g., financial regulations doc) goes in cache_control
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": """You are a financial compliance expert. 
            
REGULATIONS:
[1000-token regulations document]
            
AUDIT_SCHEMA:
[500-token audit schema]
            """,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "Is this transaction compliant?"}
    ]
)

# Response headers show:
# cache_creation_input_tokens: 1500 (first request)
# cache_read_input_tokens: 0
# input_tokens: 1500

# Second request same user (within 5 min):
response2 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}],
    messages=[
        {"role": "user", "content": "Different compliance question"}
    ]
)

# Response headers:
# cache_creation_input_tokens: 0
# cache_read_input_tokens: 1500  (charged 150 tokens = 1500 × 10%)
# input_tokens: 150
```

**Cost reduction**: 1500 tokens → 150 tokens = 90% savings on cache hits.

Cache types:
- **ephemeral**: 5-minute lifetime, per conversation
- **static**: persistent cache across requests (for large system prompts, knowledge bases)

### OpenAI API (gpt-4-turbo, gpt-4o)

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {"role": "user", "content": "Hello"}
    ]
)

print(response.usage.cache_creation_input_tokens)  # First hit
print(response.usage.cache_read_input_tokens)      # Subsequent hits cost 10%
```

**Best for**: Large static system prompts (regulatory docs, company policies, few-shot examples).

## Strategy 2: Semantic Caching

Don't cache by prompt hash — cache by semantic meaning. "What is AI safety?" and "Tell me about AI safety" are identical semantically.

### Implementation with Redis + Embedding

```python
import redis
import anthropic
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

redis_client = redis.Redis(host='localhost', port=6379)
llm = anthropic.Anthropic()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_cache_get(user_query: str, threshold=0.95):
    """Return cached response if semantically similar query exists."""
    query_embedding = embedder.encode(user_query)
    
    # Scan Redis for similar embeddings
    for key in redis_client.scan_iter("cache:*"):
        cached_embedding = np.array(redis_client.hget(key, "embedding"), dtype=float)
        similarity = cosine_similarity(
            [query_embedding], 
            [cached_embedding]
        )[0][0]
        
        if similarity > threshold:
            return redis_client.hget(key, "response").decode()
    
    return None

def semantic_cache_set(user_query: str, response: str):
    """Store response with semantic embedding for future retrieval."""
    query_embedding = embedder.encode(user_query).tolist()
    cache_key = f"cache:{hash(user_query)}"
    
    redis_client.hset(
        cache_key,
        mapping={
            "query": user_query,
            "response": response,
            "embedding": json.dumps(query_embedding)
        }
    )
    redis_client.expire(cache_key, 3600)  # 1-hour TTL

# Usage
query = "What is AI safety?"
cached = semantic_cache_get(query, threshold=0.90)

if cached:
    response = cached
else:
    response = llm.messages.create(
        model="claude-3-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}]
    ).content[0].text
    semantic_cache_set(query, response)

print(response)
```

**Cost reduction**: Typical semantic cache hit rate 30-50% in FAQ-heavy systems (customer support, onboarding).

### Trade-offs

| Strategy | Hit Rate | Cost Savings | Latency | Staleness Risk |
|---|---|---|---|---|
| Prompt hash | 10-20% | low | best | none |
| Semantic cache | 30-50% | high | +50ms (embedding) | moderate (fix with TTL) |
| Anthropic prompt cache | 60-90% (same conversation) | very high (90% off) | best | none (managed by API) |

**Use semantic cache for**: Customer support, FAQ answering, repeated patterns.
**Use prompt cache for**: Large system prompts, few-shot examples, reference docs.

## Strategy 3: Model Routing

Route queries to cheapest model that still works. Don't always use Opus.

```python
import anthropic

llm_cheap = anthropic.Anthropic(model="claude-3-haiku")  # $0.80/$4
llm_fast = anthropic.Anthropic(model="claude-3-sonnet")  # $3/$15
llm_smart = anthropic.Anthropic(model="claude-3-opus")   # $15/$75

def route_query(user_query: str, user_tier: str):
    """Route to appropriate model based on query complexity + user tier."""
    
    # Classify query complexity (low-effort heuristic)
    if len(user_query) < 100 and "simple" in user_query.lower():
        # Simple factual question → Haiku
        return llm_cheap.messages.create(
            model="claude-3-haiku",
            max_tokens=512,
            messages=[{"role": "user", "content": user_query}]
        )
    
    elif "reasoning" in user_query.lower() or len(user_query) > 500:
        # Complex reasoning or long context → Opus (for paid tiers)
        if user_tier in ["premium", "enterprise"]:
            return llm_smart.messages.create(
                model="claude-3-opus",
                max_tokens=2048,
                messages=[{"role": "user", "content": user_query}]
            )
        else:
            # Free tier gets Sonnet (middle ground)
            return llm_fast.messages.create(
                model="claude-3-sonnet",
                max_tokens=1024,
                messages=[{"role": "user", "content": user_query}]
            )
    
    else:
        # Default: Sonnet (good balance)
        return llm_fast.messages.create(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": user_query}]
        )

response = route_query("What is 2+2?", user_tier="free")
```

**Cost reduction**: 50-70% by routing to Haiku for simple queries.

### Advanced Routing

Use a fast classifier (Haiku) to decide which expensive model to use:

```python
def smart_route(user_query: str):
    """Use Haiku to classify, then route to appropriate model."""
    
    # Classify query (cheap, fast)
    classification = llm_cheap.messages.create(
        model="claude-3-haiku",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"Classify as: [simple_factual, reasoning, creative, other]\n\nQuery: {user_query}"
        }]
    )
    
    query_type = classification.content[0].text.strip().lower()
    
    if "simple" in query_type:
        return llm_cheap.messages.create(...)  # Haiku
    elif "reasoning" in query_type:
        return llm_smart.messages.create(...)  # Opus
    else:
        return llm_fast.messages.create(...)   # Sonnet
```

Cost: 50 tokens (Haiku) + response tokens (Sonnet/Opus) = marginal overhead for smart routing.

## Strategy 4: Token Optimization

### Compress Context

Don't pass entire documents. Summarize or extract relevant sections.

```python
def extract_relevant_context(documents: list[str], query: str, max_tokens=500):
    """Use Haiku to extract only relevant sections."""
    
    extracted = []
    token_budget = max_tokens
    
    for doc in documents:
        if token_budget <= 0:
            break
        
        extraction = llm_cheap.messages.create(
            model="claude-3-haiku",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"Extract 2-3 most relevant sentences for query: '{query}'\n\nDocument: {doc[:1000]}"
            }]
        )
        
        extracted_text = extraction.content[0].text
        extracted.append(extracted_text)
        token_budget -= len(extracted_text.split())
    
    return "\n".join(extracted)

# Usage in RAG
query = "What is the compliance requirement?"
relevant = extract_relevant_context(large_docs, query, max_tokens=500)

response = llm_fast.messages.create(
    model="claude-3-sonnet",
    messages=[
        {"role": "user", "content": f"Docs:\n{relevant}\n\nQuestion: {query}"}
    ]
)
```

**Cost reduction**: 10-30% by filtering irrelevant context.

### Output Token Optimization

Constrain output.

```python
# Don't do this (unbounded output):
response = client.messages.create(
    model="claude-3-sonnet",
    messages=[{"role": "user", "content": "Tell me everything about AI"}],
    max_tokens=4096  # Could generate 4K tokens at $15/1M output
)

# Do this (constrained):
response = client.messages.create(
    model="claude-3-sonnet",
    messages=[{"role": "user", "content": "Summarize in 100 words: ..."}],
    max_tokens=150  # Tight bound
)
```

**Cost reduction**: 50-80% by explicit output constraints.

## Strategy 5: Batching

Don't call LLM per-user request. Batch if possible.

```python
from concurrent.futures import ThreadPoolExecutor

def batch_process_queries(queries: list[str], batch_size=10):
    """Process queries in batches instead of 1-by-1."""
    
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        
        # Single request with multiple messages (if API supports)
        # Or parallel requests with thread pool
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    lambda q: client.messages.create(
                        model="claude-3-sonnet",
                        messages=[{"role": "user", "content": q}]
                    ),
                    query
                )
                for query in batch
            ]
            batch_results = [f.result() for f in futures]
        
        results.extend(batch_results)
    
    return results
```

**Cost reduction**: None per token, but 10-50% latency improvement via parallelization.

## Cost Budgeting

Set hard limits per user, time window.

```python
import time
from collections import defaultdict

class CostLimiter:
    def __init__(self, daily_budget_cents: int = 500):  # $5/day per user
        self.daily_budget = daily_budget_cents
        self.usage = defaultdict(list)  # user_id → [(timestamp, cost)]
    
    def record_usage(self, user_id: str, cost_cents: float):
        """Record LLM call cost for user."""
        self.usage[user_id].append((time.time(), cost_cents))
    
    def get_daily_spend(self, user_id: str) -> float:
        """Sum costs in last 24h."""
        now = time.time()
        day_ago = now - 86400
        return sum(
            cost for ts, cost in self.usage[user_id]
            if ts > day_ago
        )
    
    def can_afford(self, user_id: str, estimated_cost: float) -> bool:
        """Check if user has budget for this request."""
        spent = self.get_daily_spend(user_id)
        return (spent + estimated_cost) < self.daily_budget

limiter = CostLimiter(daily_budget_cents=500)

query = "Tell me about AI"
estimated_cost = 0.05  # $0.05 estimate

if limiter.can_afford("user123", estimated_cost):
    response = client.messages.create(...)
    actual_cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000 * 100
    limiter.record_usage("user123", actual_cost)
else:
    return {"error": "Daily budget exceeded"}
```

## Interview Talking Points

- "Prompt caching (Anthropic/OpenAI) = 90% cost savings on cache hit. Use for large system prompts, regulatory docs, few-shot examples."
- "Semantic caching + embedding = 30-50% hit rate for FAQ/customer support. Trade-off: +50ms latency for embedding lookup."
- "Model routing: classify query complexity (fast with Haiku), route to appropriate model. 50-70% cost reduction overall."
- "Context extraction: use cheap model (Haiku) to filter relevant docs before expensive model (Opus) call."
- "Set hard cost budgets per user. Graceful degrade (queue request, return cached answer, use cheaper model)."
