# Production and Observability

## LangSmith Tracing

LangSmith is the observability platform for agentic systems. It tracks every step of your agent.

### Setup

```bash
pip install langsmith

# Set environment variables
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="my-agent"
```

### Automatic Tracing

```python
from langchain_core.callbacks import LangChainTracer

# Traces happen automatically if LANGCHAIN_TRACING_V2=true
result = agent.invoke({"input": "What's 5+5?"})

# View traces at: https://smith.langchain.com/projects/my-agent
```

### What Gets Tracked

```
Run ID: abc123def456
├─ Inputs: {"input": "What's 5+5?"}
├─ Outputs: {"output": "10"}
├─ Traces (events):
│  ├─ llm_call
│  │  ├─ Input: "Q: 5+5?"
│  │  ├─ Output: "I need to calculate..."
│  │  └─ Duration: 850ms
│  ├─ tool_call (calculator)
│  │  ├─ Input: {"a": 5, "b": 5}
│  │  ├─ Output: 10
│  │  └─ Duration: 50ms
│  └─ llm_call
│     ├─ Input: "Tool returned: 10"
│     ├─ Output: "The answer is 10"
│     └─ Duration: 300ms
├─ Total Duration: 1200ms
├─ Token Usage: 245 tokens
└─ Cost: $0.01
```

### Feedback Loop

Add feedback to learn what works:

```python
from langsmith import Client

client = Client()

# After getting a result, add feedback
run_id = trace_id  # From the trace

client.create_feedback(
    run_id=run_id,
    key="user_rating",
    score=1.0,  # 1.0 = good, 0.0 = bad
    comment="Correct answer, fast response"
)

# Over time, view aggregate feedback in LangSmith UI
# See which runs got positive/negative feedback
```

## Key Metrics to Monitor

| Metric | Why It Matters | Healthy Value |
|--------|----------------|----------------|
| **Latency per node** | Bottleneck identification | <2s per node |
| **Token usage** | Cost control | <5K tokens per task |
| **Tool call success rate** | Agent reliability | >95% |
| **Tool call latency** | External API speed | <1s per call |
| **Hallucination rate** | Output quality | <5% |
| **Max iterations hit** | Infinite loop detection | <1% of runs |
| **Error rate** | System health | <2% |
| **Cost per task** | Budget tracking | Varies by task |

### Implementing Metrics

```python
from datetime import datetime

class AgentMetrics:
    def __init__(self):
        self.runs = []

    def log_run(self, run):
        self.runs.append({
            "run_id": run.id,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "duration": (run.end_time - run.start_time).total_seconds(),
            "tokens": run.token_usage,
            "tool_calls": len(run.tool_calls),
            "succeeded": run.status == "success",
            "error": run.error_message if run.status == "error" else None
        })

    def get_average_latency(self):
        if not self.runs: return 0
        total = sum(run["duration"] for run in self.runs)
        return total / len(self.runs)

    def get_success_rate(self):
        if not self.runs: return 0
        successes = sum(1 for run in self.runs if run["succeeded"])
        return successes / len(self.runs)

    def get_average_cost(self, token_cost_per_k=0.01):
        if not self.runs: return 0
        total_tokens = sum(run["tokens"] for run in self.runs)
        return (total_tokens / 1000) * token_cost_per_k
```

## Cost Management

Agents are expensive. Minimize cost:

### Strategy 1: Caching

Cache tool results to avoid redundant calls:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_web_search(query: str):
    """Cache web search results for 1 hour"""
    return web_search(query)

# Same query → cached result (no API cost)
```

### Strategy 2: Model Tiering

Use cheaper models for simple tasks:

```python
def smart_model_choice(task_complexity: str):
    if task_complexity == "simple":
        return ChatOpenAI(model="gpt-3.5-turbo")  # $0.001 per 1K tokens
    elif task_complexity == "medium":
        return ChatOpenAI(model="gpt-4")  # $0.015 per 1K tokens
    else:
        return ChatOpenAI(model="gpt-4-turbo")  # $0.03 per 1K tokens

llm = smart_model_choice(determine_complexity(user_query))
```

### Strategy 3: Tool Call Batching

Call multiple tools at once instead of sequentially:

```python
# Bad: One by one
price_apple = get_stock_price("AAPL")  # 1 API call
price_google = get_stock_price("GOOGL")  # 1 API call
price_meta = get_stock_price("META")  # 1 API call
# Total: 3 calls

# Good: Batch
prices = get_stock_prices(["AAPL", "GOOGL", "META"])  # 1 API call
# Total: 1 call
```

### Strategy 4: Token Optimization

Compress prompts and results:

```python
# Bad: Full conversation history (5K tokens)
messages = [all previous messages] + [new query]

# Good: Summarized history (1K tokens)
summary = summarize_old_messages(all[:-10])
messages = [summary] + [last 10 messages] + [new query]

# Saves 4K tokens per call = $0.04 per call
```

## Deployment Patterns

### Pattern 1: Serverless (AWS Lambda / Google Cloud Functions)

```python
# deployment/handler.py
from langgraph.graph import StateGraph
import json

graph = StateGraph(AgentState)
# ... define graph ...
agent = graph.compile()

def lambda_handler(event, context):
    body = json.loads(event["body"])
    query = body["query"]

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    return {
        "statusCode": 200,
        "body": json.dumps({"answer": result["output"]})
    }
```

### Pattern 2: FastAPI Server

```python
from fastapi import FastAPI
from langgraph.graph import StateGraph

app = FastAPI()

agent = create_agent()  # Your agent

@app.post("/agent")
async def agent_endpoint(query: str):
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return {"answer": result["output"]}

# Run: uvicorn main:app --reload
```

### Pattern 3: Streaming Responses

For real-time feedback, stream results:

```python
@app.post("/agent-stream")
async def agent_stream(query: str):
    def event_generator():
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="updates"
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### Pattern 4: Job Queue (Celery / RQ)

For long-running tasks:

```python
from celery import Celery

celery = Celery()

@celery.task
def run_agent_task(query: str):
    """Long-running agent task"""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result

# Frontend polls for completion
@app.post("/agent-async")
async def agent_async(query: str):
    task = run_agent_task.delay(query)
    return {"task_id": task.id}

@app.get("/agent-result/{task_id}")
async def agent_result(task_id: str):
    task = run_agent_task.AsyncResult(task_id)
    if task.ready():
        return {"result": task.result}
    else:
        return {"status": "pending"}
```

## Error Handling Patterns

### Pattern 1: Retry with Exponential Backoff

```python
import time

def with_retry(func, max_attempts=3, base_wait=1):
    attempt = 1

    while attempt <= max_attempts:
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait_time = base_wait * (2 ** (attempt - 1))
            time.sleep(wait_time)
            attempt += 1
```

### Pattern 2: Circuit Breaker

Stop calling a failing service temporarily:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func()
            self.failure_count = 0
            self.state = "closed"
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

            raise
```

### Pattern 3: Fallback Response

```python
def agent_with_fallback(query: str):
    try:
        return agent.invoke({"messages": [{"role": "user", "content": query}]})
    except Exception as e:
        logging.error(f"Agent failed: {e}")
        return {
            "output": "I encountered an error. Please try again.",
            "error": str(e)
        }
```

## Testing Agents

### Unit Tests: Individual Nodes

```python
def test_llm_node():
    state = {
        "messages": [{"role": "user", "content": "What's 2+2?"}]
    }
    result = llm_node(state)
    assert len(result["messages"]) > 1  # LLM added a message
```

### Integration Tests: Full Graph

```python
def test_full_agent():
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What's 2+2?"}]
    })
    assert "4" in result["output"]
```

### Evaluation: Comparing Agents

```python
from langsmith import Client

client = Client()

# Create evaluation dataset
test_cases = [
    {"query": "What's 2+2?", "expected": "4"},
    {"query": "Who founded OpenAI?", "expected": "Sam Altman"},
]

# Run agent on each test case
results = []
for test in test_cases:
    result = agent.invoke({"messages": [{"role": "user", "content": test["query"]}]})
    results.append({
        "test": test,
        "result": result,
        "success": test["expected"] in result["output"]
    })

# View results in LangSmith UI
```

## Monitoring in Production

### Health Checks

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent_loaded": agent is not None,
        "latency_ms": measure_latency()
    }
```

### Alerting

```python
def check_metrics():
    metrics = AgentMetrics()

    if metrics.get_error_rate() > 0.05:
        alert("High error rate: {}%".format(metrics.get_error_rate() * 100))

    if metrics.get_average_cost() > daily_budget:
        alert("Daily cost exceeded budget")

    if metrics.get_average_latency() > 10:
        alert("High latency detected")
```

## Summary

Production agents require:
- **Tracing**: LangSmith for visibility
- **Metrics**: Monitor latency, cost, success rate
- **Error handling**: Retry, circuit breaker, fallback
- **Testing**: Unit, integration, evaluation
- **Deployment**: Serverless, FastAPI, job queue
- **Monitoring**: Alerts, health checks, dashboards

Don't deploy without these in place.
