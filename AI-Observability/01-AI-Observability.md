# AI Observability — Tracing, Metrics, Evals

LLM systems are black boxes. Observability = tracing (call chains), metrics (latency/cost/hallucination), and evals (systematic quality checks).

## OpenTelemetry for GenAI

OpenTelemetry (OTEL) is the standard instrumentation framework. GenAI semantic conventions define what to trace.

### OTEL Concepts

- **Span**: single operation (LLM call, embedding, retrieval). Start time, duration, attributes, events.
- **Trace**: chain of spans (user request → retrieve docs → call LLM → parse response). Root span has trace_id; child spans have parent_id.
- **Exporter**: sends traces to backend (Jaeger, Datadog, Arize, Helicone, etc.)
- **Sampler**: decide which traces to keep (sample_rate=0.1 = 10% of traffic).

### GenAI Semantic Conventions

OTEL defines attributes for LLM operations:

```
Span: "llm.completion"
Attributes:
  llm.model = "claude-3-sonnet"
  llm.system = "anthropic"
  llm.request.max_tokens = 1024
  llm.response.stop_reason = "end_turn"
  llm.usage.input_tokens = 150
  llm.usage.output_tokens = 200
  llm.usage.total_tokens = 350
  
Span: "llm.embedding"
Attributes:
  llm.model = "text-embedding-3-small"
  llm.embedding.vector_size = 1536
  llm.request.batch_size = 100

Span: "db.query"  (for retrieval)
Attributes:
  db.system = "postgres"
  db.query = "SELECT ... FROM documents WHERE embedding <-> $1 LIMIT 5"
  db.rows_returned = 5
```

Events capture tool calls:

```
span.add_event("tool_call", {
  "tool.name": "search",
  "tool.args": "{'query': 'AI safety'}",
  "tool.result": "[result1, result2, ...]"
})
```

## LangSmith (LangChain Native)

LangSmith is the first-class LangChain tracing platform (same org — LangChain).

### Setup

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

from langchain import llm
chain = llm | parser
result = chain.invoke({"input": "..."})  # Automatically traced
```

### What Gets Traced

- Every chain, agent, tool call
- Inputs/outputs, tokens, latency
- Tool arguments and results
- Model parameters (temperature, max_tokens)
- Errors and exceptions

### Evaluation & Feedback

Post-production feedback loop:

```python
from langsmith import Client
client = Client()

run = client.read_run(run_id)
# Mark run as good/bad
client.create_feedback(
    run_id=run.id,
    key="user_feedback",
    score=1.0,  # or 0.0 for bad
    comment="Correct answer"
)
```

Use feedback to: (1) detect regressions, (2) curate golden datasets, (3) train eval models.

### Comparison Table

| Platform | Model Coverage | Integration | Cost | Best For |
|---|---|---|---|---|
| LangSmith | LangChain native | Auto with env var | Per-trace | LangChain workflows |
| Arize AI | Multi-model | SDK or API | Per-event | Production LLM monitoring |
| Helicone | OpenAI/Anthropic/others | Proxy or SDK | Per-request | Cost tracking, latency SLA |
| W&B Weave | Multi-model | SDK | Per-trace | Experiment tracking + monitoring |

## Arize AI

Arize specializes in LLM observability at production scale.

### Setup with Anthropic

```python
import anthropic
from arize.openinference.instrumentation.anthropic import setup_anthropic_instrumentation

setup_anthropic_instrumentation()

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

Traces auto-send to Arize dashboard.

### Key Metrics

Arize tracks LLM-specific metrics:

- **Latency (p50, p95, p99)**: track SLA breaches
- **Token cost**: input_tokens × input_cost_per_1k + output_tokens × output_cost_per_1k
- **Error rate**: % of requests that failed
- **Hallucination rate**: detected via semantic similarity (LLM response vs ground truth)
- **Token efficiency**: output_tokens / input_tokens (lower = better compression)
- **Model drift**: when model output distribution changes unexpectedly

## Helicone

Helicone proxies OpenAI/Anthropic requests and logs everything.

### Setup

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-...",
    base_url="https://api.helicone.ai/anthropic/v1"  # proxy
)

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[...],
    extra_headers={"Helicone-Auth": "Bearer <key>"}
)
```

Every request logged to Helicone dashboard.

### Strong Points

- **Cost tracking**: see per-request, per-user, per-model costs
- **Caching integration**: Helicone caches responses by prompt + model (Redis-backed), reduces cost 10-30%
- **Latency monitoring**: p50/p95/p99 per model
- **User analytics**: track cost per user, usage patterns
- **Request replay**: re-run requests to compare model versions

## W&B Weave

Weave integrates experiment tracking + production monitoring.

### Setup

```python
import anthropic
from weave import init, client as weave_client

weave_init("project-name")

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[...],
    extra_headers={"Weave-Trace": "true"}
)
```

### Unified Dashboard

Single pane of glass:
- Training/experiment runs (metric history, hyperparams)
- Production traces (request → response → feedback)
- A/B test results (model1 vs model2 win rate)
- Drift alerts (input/output distribution change)

## Structured Logging for AI Pipelines

Beyond vendor platforms, log everything to a structured sink (Datadog, CloudWatch, ELK).

```python
import structlog
import json

logger = structlog.get_logger()

logger.info("llm_call",
    model="claude-3-sonnet",
    input_tokens=150,
    output_tokens=200,
    latency_ms=1200,
    cost_cents=0.15,
    user_id="user123",
    session_id="sess_xyz",
    tool_calls=["search", "retrieve"],
    error=None
)
```

Schema:
- **request_id / trace_id**: link spans together
- **timestamp**: when
- **operation**: llm_call, embedding, retrieval, tool_call
- **model**: which LLM or service
- **input_tokens, output_tokens**: accounting
- **latency_ms**: p99 tracking
- **cost_cents**: per-request cost
- **user_id, session_id**: aggregation
- **tool_calls**: which tools invoked
- **error**: if failed, the exception

Store in JSON format for easy querying.

## Key Metrics Dashboard

Build dashboards with:

1. **Latency (SLA tracking)**
   - Histogram: p50, p95, p99 per model
   - Alert if p99 > 2s (example)

2. **Cost (budget control)**
   - Daily spend by model, by user
   - Cost per successful request vs cost per error
   - Alert if daily spend > $X

3. **Quality (hallucination detection)**
   - % of responses with detected hallucinations (via semantic similarity)
   - % of responses flagged by human feedback
   - % of tool calls that succeeded

4. **Tool Success Rate**
   - % of tool calls that completed without error
   - Latency per tool

5. **Model Drift**
   - Input token distribution change (sudden influx of long prompts?)
   - Output token distribution change (model becoming verbose?)
   - Stop reason distribution (are we hitting max_tokens?)

## Shadow Mode & Canary Deployments

**Shadow mode**: run new model in parallel, log results, don't return to user.

```python
# Production model
response_prod = anthropic_prod.messages.create(
    model="claude-3-sonnet",
    messages=[...]
)

# Shadow (new model) — log but discard
try:
    response_shadow = anthropic_shadow.messages.create(
        model="claude-3-opus",
        messages=[...]
    )
    logger.info("shadow_test",
        prod_model="sonnet",
        shadow_model="opus",
        prod_response=response_prod.content,
        shadow_response=response_shadow.content,
        semantic_similarity=compute_similarity(...)
    )
except Exception as e:
    logger.error("shadow_error", model="opus", error=str(e))

return response_prod  # Always return production
```

Use shadow logs to: (1) detect quality regression, (2) compare cost, (3) gather evidence for rollout decision.

**Canary**: route X% of traffic to new model, monitor metrics before full rollout.

```python
import random

model = "claude-3-opus" if random.random() < 0.1 else "claude-3-sonnet"
response = client.messages.create(model=model, messages=[...])
logger.info("request", model=model, ...)
```

Monitor: do canary requests have higher error rate? higher latency? If metrics OK after 24h, roll out 100%.

## Interview Talking Points

- "Observability ≠ monitoring. Observability is the ability to ask any question about system state. For LLMs, that means tracing every tool call, token usage, latency."
- "OTEL semantic conventions give you vendor-agnostic trace format — swap backends without code change."
- "LangSmith for dev/testing (auto-integration with LangChain), Arize for production (model drift detection), Helicone for cost control."
- "Hallucination detection at scale: log ground truth in feedback loop, use semantic similarity to score responses, set alerts at 5% hallucination rate."
- "Shadow mode > A/B test for high-stakes models (loan approval). Run new model unseen, compare systematically, then canary."
