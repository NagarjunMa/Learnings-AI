# AWS Bedrock AgentCore — Production Agent Infrastructure

AgentCore is NOT a framework. It's **managed infrastructure** for running AI agents. You bring code written in any framework (Strands, LangGraph, CrewAI, OpenAI SDK), AgentCore provides the serverless runtime, memory, tool connectivity, security, and observability.

**GA since October 2025. Announced at AWS Summit NYC, July 2025.**

---

## Mental Model

```
Without AgentCore:
Your agent code → You manage: servers, memory, sessions, auth, tracing, guardrails

With AgentCore:
Your agent code → AgentCore handles: serverless execution, memory, sessions,
                                     OAuth, tracing, policy enforcement
```

**Framework agnostic:** Works with Strands, LangGraph, CrewAI, OpenAI Agents SDK, Google ADK.
**Model agnostic:** Works with Claude, GPT-4, Gemini, Llama, Nova — not just Bedrock models.

---

## Architecture

```
User Request
     │
     ▼
AgentCore Runtime ←── Identity (IAM / OAuth2)
     │
     ├──→ Memory (Short-term + Long-term strategies)
     ├──→ Gateway (MCP hub → Lambda / REST APIs / MCP servers)
     ├──→ Code Interpreter (sandboxed Python/JS/TS)
     ├──→ Browser (cloud Playwright automation)
     └──→ Policy (Cedar-based guardrail layer)
          │
          ▼
   Observability → CloudWatch (OTEL traces, spans, logs)
   Evaluations  → 13 built-in quality evaluators
   Registry     → Centralized org-wide tool/agent catalog
```

### Core Components

| Component | What It Does | Key Use Case |
|---|---|---|
| **Runtime** | Serverless microVM per session | Isolate and execute your agent code |
| **Memory** | Short + long-term managed storage | Persist user preferences, facts across sessions |
| **Gateway** | REST/Lambda/MCP → unified MCP endpoint | Give agents tool access without writing glue code |
| **Identity** | OAuth2 + IAM inbound/outbound auth | Secure who can call the agent + what services it can call |
| **Code Interpreter** | Isolated Python/JS/TS sandbox | Agent-generated code execution |
| **Browser** | Cloud Playwright runtime | Web automation tasks |
| **Policy** | Cedar-based access control | Deterministic rules that intercept tool calls |
| **Observability** | OpenTelemetry → CloudWatch | Traces, spans, metrics, logs — zero-code |
| **Evaluations** | 13 auto-evaluators | Quality scoring without writing eval harness |

---

## Session Model

Each session runs in a **dedicated microVM** with isolated CPU, memory, and filesystem.

```python
# Session is identified by runtimeSessionId
# - Reuse same ID for multi-turn conversations
# - New ID = fresh session

client.invoke_agent_runtime(
    agentRuntimeArn="arn:...",
    runtimeSessionId="user-123-session-456",  # Your ID; reuse for continuity
    payload=json.dumps({"prompt": "..."}),
    qualifier="DEFAULT"
)
```

**Session limits:**
- Max lifetime: 8 hours
- Idle timeout: 15 minutes (configurable)
- Storage: 1 GB per session
- On termination: full microVM teardown, memory sanitization

---

## Installation & Setup

```bash
# Python SDK
pip install bedrock-agentcore

# With protocol extras
pip install "bedrock-agentcore[ag-ui]"   # AG-UI protocol (SSE/WebSocket)
pip install "bedrock-agentcore[a2a]"     # Agent-to-Agent protocol

# CLI
npm install -g @aws/agentcore
```

**Requirements:** Python 3.9+, AWS credentials configured, Bedrock model access enabled.

---

## Building an Agent — Option A: CLI (Recommended)

```bash
# Step 1: Create project (interactive wizard)
agentcore create
# Choose: framework (Strands/LangGraph/CrewAI), model provider, memory, build type

# Step 2: Local dev
agentcore dev
agentcore dev "Hello, what can you do?"
agentcore dev "Tell me a joke" --stream

# Step 3: Deploy to AWS
agentcore deploy
agentcore deploy --plan    # Dry run first

# Step 4: Invoke deployed agent
agentcore invoke "Hello"
agentcore invoke "Continue" --session-id my-session  # Multi-turn
agentcore invoke "Hello" --stream

# Step 5: Observe
agentcore logs
agentcore logs --since 30m --level error
agentcore traces list
agentcore traces get <trace-id>
```

**Generated `agentcore.json`:**
```json
{
  "name": "MyAgent",
  "version": 1,
  "agents": [
    {
      "type": "AgentCoreRuntime",
      "name": "MyAgent",
      "build": "CodeZip",
      "entrypoint": "main.py",
      "codeLocation": "app/MyAgent/",
      "runtimeVersion": "PYTHON_3_13",
      "networkMode": "PUBLIC",
      "modelProvider": "Bedrock",
      "protocol": "HTTP"
    }
  ],
  "memories": [],
  "credentials": [],
  "evaluators": [],
  "agentCoreGateways": [],
  "policyEngines": []
}
```

---

## Building an Agent — Option B: SDK (Code)

### Basic HTTP Agent

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel

app = BedrockAgentCoreApp()

model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-5-20251001-v1:0")
agent = Agent(model=model, system_prompt="You are a helpful assistant.")

@app.entrypoint
def invoke(payload):
    user_input = payload.get("prompt")
    response = agent(user_input)
    return response.message['content'][0]['text']

if __name__ == "__main__":
    app.run()  # HTTP server on :8080
```

### Streaming Agent

```python
@app.entrypoint
async def invoke(request):
    prompt = request.get("prompt")
    async for event in agent.stream_async(prompt):
        yield event
```

### AG-UI Protocol (SSE + WebSocket)

```python
from bedrock_agentcore.runtime import AGUIApp
from ag_ui.core import RunAgentInput, RunStartedEvent, TextMessageStartEvent, \
                       TextMessageContentEvent, TextMessageEndEvent, RunFinishedEvent
import uuid

app = AGUIApp()

@app.entrypoint
async def my_agent(input_data: RunAgentInput):
    yield RunStartedEvent(
        thread_id=input_data.thread_id,
        run_id=input_data.run_id
    )

    msg_id = str(uuid.uuid4())
    yield TextMessageStartEvent(message_id=msg_id, role="assistant")

    response = agent(input_data.messages[-1].content)
    for chunk in response.stream():
        yield TextMessageContentEvent(message_id=msg_id, delta=chunk)

    yield TextMessageEndEvent(message_id=msg_id)
    yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

app.run()
# POST /invocations → SSE stream
# /ws              → WebSocket
```

### Agent-to-Agent Protocol

```python
from bedrock_agentcore.runtime import serve_a2a
from strands.a2a import StrandsA2AExecutor

serve_a2a(StrandsA2AExecutor(agent))
# Now other agents can call this agent across frameworks
```

---

## Control Plane API (Manage Runtimes)

```python
import boto3

client = boto3.client('bedrock-agentcore-control', region_name='us-west-2')

# Create Runtime — CodeZip
response = client.create_agent_runtime(
    agentRuntimeName="my-agent",
    agentRuntimeArtifact={
        'codeConfiguration': {
            'code': {
                's3': {
                    'bucket': f"bedrock-agentcore-code-{account_id}-us-west-2",
                    'prefix': "my-agent/deployment_package.zip"
                }
            },
            'runtime': 'PYTHON_3_13',
            'entryPoint': ['opentelemetry-instrument', 'main.py']
            # entryPoint with OTEL enables automatic tracing
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},  # or "PRIVATE"
    roleArn="arn:aws:iam::ACCOUNT_ID:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2",
    lifecycleConfiguration={
        'idleRuntimeSessionTimeout': 300,  # 5 min idle → terminate session
        'maxLifetime': 28800               # 8 hour max session
    }
)

# Create Runtime — Container
response = client.create_agent_runtime(
    agentRuntimeName="my-container-agent",
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': f'{account}.dkr.ecr.{region}.amazonaws.com/my-agent:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn="arn:aws:iam::ACCOUNT_ID:role/..."
)

# Update Runtime
client.update_agent_runtime(
    agentRuntimeId='<agent-id>',
    agentRuntimeArtifact={...},
    networkConfiguration={...},
    roleArn="..."
)

# Delete
client.delete_agent_runtime(agentRuntimeId='<agent-id>')
```

---

## Data Plane API (Invoke Runtimes)

```python
import boto3, json

client = boto3.client('bedrock-agentcore', region_name='us-west-2')

# Invoke — synchronous
response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:123456789:agent-runtime/...",
    runtimeSessionId="session-123",
    payload=json.dumps({"prompt": "What is 2+2?"}),
    qualifier="DEFAULT"          # or specific version alias
)
result = json.loads(response['response'].read())

# Stop a session (free up resources early)
client.stop_runtime_session(
    agentRuntimeArn="arn:...",
    runtimeSessionId="session-123",
    qualifier="DEFAULT"
)
```

---

## Memory

### Two Memory Types

**Short-term:** Within a single session. Auto-managed. Cleared on session end.

**Long-term:** Across sessions. Persisted via knowledge strategies. Semantically searchable.

### Memory Strategies

| Strategy | What It Extracts | Search Method |
|---|---|---|
| `SEMANTIC` | Factual statements → vectors | Cosine similarity |
| `SUMMARIZATION` | Rolling summaries | Text retrieval |
| `USER_PREFERENCE` | Behavioral patterns | Key lookup |
| `EPISODIC` | Task sequences + reflections | Intent search |
| `SELF_MANAGED` | Custom logic | Your query |

### Add Memory via CLI

```bash
agentcore add memory \
  --name CustomerMemory \
  --strategies SEMANTIC,SUMMARIZATION

agentcore deploy
# MEMORY_CUSTOMERMEMORY_ID env var auto-injected into runtime
```

### Memory SDK

```python
from bedrock_agentcore.memory import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
import boto3

session_manager = MemorySessionManager(
    memory_id="<memory-id>",
    region_name="us-west-2"
)

# Create session
session = session_manager.create_memory_session(
    actor_id="user-123",
    session_id="support-session-456"
)

# Write conversation turns
session.add_turns(messages=[
    ConversationalMessage("How can I help?", MessageRole.ASSISTANT),
    ConversationalMessage("I need a refund for order 999", MessageRole.USER)
])

# Read recent turns (short-term)
recent = session.get_last_k_turns(k=5)

# Semantic search over long-term memory
results = session.search_long_term_memories(
    query="what was the customer's issue?",
    namespace_prefix="/",
    top_k=3
)

# List all long-term records
records = session.list_long_term_memory_records(namespace_prefix="/")
```

### Memory + Strands Integration

```python
import os
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig, RetrievalConfig
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager
)

MEMORY_ID = os.getenv("MEMORY_CUSTOMERMEMORY_ID")  # Auto-injected after deploy
REGION = os.getenv("AWS_REGION", "us-west-2")

def get_memory_session_manager(session_id: str, actor_id: str):
    return AgentCoreMemorySessionManager(
        AgentCoreMemoryConfig(
            memory_id=MEMORY_ID,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config={
                f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
                f"/summaries/{actor_id}/{session_id}": RetrievalConfig(top_k=3, relevance_score=0.5)
            }
        ),
        REGION
    )

@app.entrypoint
async def invoke(payload, context):
    session_id = getattr(context, 'session_id', 'default')
    user_id = getattr(context, 'user_id', 'default')

    agent = Agent(
        model=load_model(),
        session_manager=get_memory_session_manager(session_id, user_id),
        system_prompt="You are a helpful assistant.",
    )
    return agent(payload.get("prompt"))
```

### Memory Limits

| Limit | Value |
|---|---|
| Memory resources per account per region | 150 |
| Strategies per memory resource | 6 |
| Messages per CreateEvent call | 100 |
| Max message size | 100 KB |
| Max event size | 10 MB |
| Event expiration range | 7–365 days |
| Extraction rate (long-term) | 150,000 tokens/min |
| Extraction rate (episodic) | 50,000 tokens/min |
| CreateEvent rate | 10 TPS |
| RetrieveMemoryRecords rate | 30 TPS |

---

## Gateway (Tool Hub)

Gateway converts REST APIs, Lambda functions, and MCP servers into unified MCP-compatible tools.

```
Agent → Gateway (MCP endpoint) → Target (REST API / Lambda / MCP server)
```

```bash
# Add gateway via CLI
agentcore add gateway --name MyAPIGateway
agentcore deploy
```

### Invoke a Tool via MCP Protocol

```bash
curl -X POST https://<gateway-mcp-url> \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "searchProducts",
      "arguments": {
        "query": "laptop",
        "maxResults": 10,
        "priceRange": {"min": 500, "max": 2000}
      }
    }
  }'
```

### Gateway Limits

| Limit | Value |
|---|---|
| Gateways per account | 1,000 |
| Targets per gateway | 100 |
| Tools per target | 1,000 |
| Invocation timeout | 15 minutes |
| Concurrent connections | 1,000 |
| Inline schema size | 1 MB |
| S3 schema size | 10 MB |

---

## Policy (Cedar-Based Guardrails)

Cedar is a **deterministic** policy language. Unlike LLM guardrails (probabilistic), Cedar rules always enforce exactly. `forbid` always beats `permit`.

Default: **deny all**. Must explicitly permit.

### Basic Permit

```cedar
permit(
  principal is AgentCore::OAuthUser,
  action in [
    AgentCore::Action::"InsuranceAPI__get_policy",
    AgentCore::Action::"InsuranceAPI__get_claim_status"
  ],
  resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:us-west-2:123456789012:gateway/insurance"
);
```

### Role-Based (Senior Only Can Update)

```cedar
forbid(
  principal is AgentCore::OAuthUser,
  action == AgentCore::Action::"InsuranceAPI__update_coverage",
  resource == AgentCore::Gateway::"arn:..."
) unless {
  principal.hasTag("role") &&
  (principal.getTag("role") == "senior-adjuster" ||
   principal.getTag("role") == "manager")
};
```

### Scope-Based (JWT Claims)

```cedar
permit(
  principal is AgentCore::OAuthUser,
  action == AgentCore::Action::"InsuranceAPI__file_claim",
  resource == AgentCore::Gateway::"arn:..."
) when {
  principal.hasTag("scope") &&
  principal.getTag("scope") like "*insurance:claim*"
};
```

### Conditional on Tool Input Value

```cedar
permit(
  principal is AgentCore::IamEntity,
  action == AgentCore::Action::"RefundAPI__process_refund",
  resource == AgentCore::Gateway::"arn:..."
) when {
  principal.id like "*:111122223333:*" &&
  context.input has amount &&
  context.input.amount < 1000
};
```

### Cedar Syntax Reference

| Syntax | Description |
|---|---|
| `permit` / `forbid` | Policy effect |
| `principal is AgentCore::OAuthUser` | Principal type check |
| `action in [...]` | Multiple actions |
| `when { ... }` | Condition that must be true |
| `unless { ... }` | Condition that must NOT be true (forbid only) |
| `principal.hasTag("x")` | Check JWT claim exists |
| `principal.getTag("x")` | Read JWT claim value |
| `context.input has field` | Check tool input field exists |
| `context.input.field` | Access tool input parameter value |
| `like "*pattern*"` | Wildcard pattern match |

**Cedar constraints:**
- No floating-point (use integers or decimals max 4 decimal places)
- Pattern matching: `like` only — no regex
- `forbid` always wins

---

## Observability

Zero-code OTEL tracing via entrypoint wrapper:

```python
# In agentcore.json / create_agent_runtime()
'entryPoint': ['opentelemetry-instrument', 'main.py']
```

```bash
# Environment variables for observability
export AGENT_OBSERVABILITY_ENABLED=true
export OTEL_PYTHON_DISTRO=aws_distro
export OTEL_PYTHON_CONFIGURATOR=aws_configurator
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_RESOURCE_ATTRIBUTES=service.name=MyAgent
```

**Where data goes:**
- Logs: `/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint}/`
- Spans: `/aws/spans/default`
- Metrics: `bedrock-agentcore` namespace in CloudWatch

**CLI commands:**
```bash
agentcore logs
agentcore logs --since 30m --level error
agentcore traces list
agentcore traces get <trace-id>
```

---

## Code Interpreter

Isolated Python/JS/TS sandbox — 2 vCPU, 8 GB RAM, 10 GB disk per session.

```python
client = boto3.client('bedrock-agentcore', region_name='us-west-2')

# Execute Python
client.invoke_code_interpreter(
    codeInterpreterIdentifier="aws.codeinterpreter.v1",
    sessionId="my-session",
    name="executeCode",
    arguments={"language": "python", "code": "import pandas as pd; print(pd.__version__)"}
)

# Shell command
client.invoke_code_interpreter(
    sessionId="my-session",
    name="executeCommand",
    arguments={"command": "ls -la /tmp"}
)

# Long-running async
response = client.invoke_code_interpreter(
    sessionId="my-session",
    name="startCommandExecution",
    arguments={"command": "sleep 30 && echo done"}
)
task_id = response['taskId']

# Poll
client.invoke_code_interpreter(
    sessionId="my-session",
    name="getTask",
    arguments={"taskId": task_id}
)

# File operations
client.invoke_code_interpreter(
    sessionId="my-session",
    name="writeFiles",
    arguments={"content": [{"path": "data.csv", "text": "a,b,c\n1,2,3"}]}
)

client.invoke_code_interpreter(
    sessionId="my-session",
    name="readFiles",
    arguments={"paths": ["data.csv"]}
)
```

---

## Deployment

### CodeZip Deployment (ARM64 required)

```bash
# Package for ARM64
uv pip install \
  --python-platform aarch64-manylinux2014 \
  --python-version 3.13 \
  --target=deployment_package \
  --only-binary=:all: \
  -r pyproject.toml

cd deployment_package && zip -r ../deployment_package.zip . && cd ..
zip deployment_package.zip main.py

# Upload to the specific AgentCore S3 bucket
s3_client.upload_file(
    'deployment_package.zip',
    f"bedrock-agentcore-code-{account_id}-us-west-2",
    f"{agent_name}/deployment_package.zip"
)
```

**Max sizes:** 250 MB compressed, 750 MB uncompressed.

### Container Deployment

```bash
docker build -t my-agent .
aws ecr get-login-password | docker login --username AWS \
  --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
docker tag my-agent ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/my-agent:latest
docker push ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/my-agent:latest
```

**Max image size:** 2 GB.

### Versioning

```
Create Runtime → Version 1 auto-created → DEFAULT endpoint auto-created → READY
Update Runtime → New Version created     → DEFAULT endpoint updates (zero-downtime)
Custom Endpoint → Pin to specific version → A/B testing / canary / rollback
```

### Supported Python Runtimes

| ID | EOL |
|---|---|
| `PYTHON_3_14` | June 2030 |
| `PYTHON_3_13` | June 2029 |
| `PYTHON_3_12` | Oct 2028 |
| `PYTHON_3_11` | June 2026 |
| `PYTHON_3_10` | June 2026 |

---

## AgentCore vs Classic Bedrock Agents

| Dimension | Bedrock Agents (classic) | Bedrock AgentCore |
|---|---|---|
| Nature | Fully managed no-code | Infrastructure layer; you write code |
| Framework | AWS-only orchestration | Any (LangGraph, Strands, CrewAI, etc.) |
| Model support | Bedrock-hosted only | Any model (Bedrock, OpenAI, Gemini) |
| Tool integration | Action Groups (Lambda + OpenAPI) | Gateway (MCP protocol) |
| Policy | Bedrock Guardrails (probabilistic LLM) | Cedar (deterministic rules) |
| Customization | Config-driven | Full code control |
| Observability | CloudWatch logs | Full OpenTelemetry traces + dashboards |
| Target use case | Rapid prototyping | Enterprise production |

**Use Bedrock Agents:** Quick prototyping, config-based, Bedrock-only models.
**Use AgentCore:** Custom code, multi-framework, external models, compliance, production scale.

---

## Hard Limits

| Limit | Value |
|---|---|
| Session CPU/RAM | 2 vCPU / 8 GB (non-adjustable) |
| Session storage | 1 GB |
| Max Docker image | 2 GB |
| Max CodeZip (compressed) | 250 MB |
| Max CodeZip (uncompressed) | 750 MB |
| Max payload | 100 MB |
| Request timeout | 15 minutes |
| Streaming max | 60 minutes |
| Async job max | 8 hours |
| Idle session timeout | 15 minutes (default) |
| Max session lifetime | 8 hours |
| WebSocket frame size | 64 KB |
| WebSocket frame rate | 250 frames/second |
| Gateways per account | 1,000 |
| Memory resources per account | 150 |

---

## Pricing

- **Runtime:** $0.0895/vCPU-hour + $0.00945/GB-hour
- **Billing start:** when agent code starts executing
- **I/O wait is free:** Time waiting for LLM/API responses not billed
- **Per-second increments** (not per-minute)
- **Typical savings:** Agents spend 30–70% in I/O wait → significantly cheaper than provisioned EC2

Gateway, Memory, Policy, Browser, Code Interpreter each priced separately. Foundation model inference billed separately.

---

## IAM — Required Permissions

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateAgentRuntime",
        "bedrock-agentcore:InvokeAgentRuntime",
        "bedrock-agentcore:CreateGateway",
        "bedrock-agentcore:CreateGatewayTarget",
        "bedrock-agentcore:SynchronizeGatewayTargets"
      ],
      "Resource": "*"
    }
  ]
}
```

Trust policy for execution role:
```json
{
  "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
  "Condition": {
    "StringEquals": {"aws:SourceAccount": "ACCOUNT_ID"},
    "ArnLike": {"aws:SourceArn": "arn:aws:bedrock-agentcore:REGION:ACCOUNT:*"}
  }
}
```

---

## GitHub References

- SDK: `github.com/aws/bedrock-agentcore-sdk-python`
- Samples: `github.com/awslabs/amazon-bedrock-agentcore-samples`
- Onboarding: `github.com/aws-samples/sample-amazon-bedrock-agentcore-onboarding`
- Official docs: `docs.aws.amazon.com/bedrock-agentcore/latest/devguide/`
