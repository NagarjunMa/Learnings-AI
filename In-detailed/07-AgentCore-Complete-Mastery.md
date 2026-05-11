# AWS Bedrock AgentCore — Complete Mastery Deep-Dive

**Objective:** Every minute detail needed to master AgentCore for production systems and interview scenarios.

---

## Part 1: Foundation & Mental Models

### What AgentCore Is (and Isn't)

AgentCore ≠ framework. It's **managed serverless infrastructure** for running AI agents written in any framework.

```
Old way:  Agent code + DIY: servers, sessions, memory, auth, policies, observability
New way:  Agent code + AgentCore: handles all ops concerns
```

**Invariant:** AgentCore is **framework-agnostic** and **model-agnostic**. You choose:
- Framework: Strands, LangGraph, CrewAI, OpenAI Agents SDK, Google ADK
- Model: Claude, GPT-4, Gemini, Llama, Nova (any LLM)
- Provider: Bedrock (AWS), OpenAI API, Anthropic API, open-source

**GA timeline:** Announced July 2025 (AWS Summit NYC). GA October 2025.

### Architectural Invariants

```
Principle 1: Isolation
  Every session = dedicated microVM (CPU, memory, filesystem isolated)
  → No crosstalk between users/sessions
  → Perfect multi-tenancy at scale

Principle 2: Serverless Consumption
  You pay for execution time only
  → I/O wait time (LLM, API, DB calls) is FREE
  → Not billed for idle sessions
  → Per-second granularity (not per-minute)

Principle 3: Framework Agnosticity
  AgentCore runtime doesn't care if you use Strands or LangGraph
  → Your code runs as-is
  → All components (memory, gateway, policy, observability) available

Principle 4: Deterministic Security (Cedar)
  Policies are rules, not probabilistic guardrails
  → forbid always beats permit
  → No LLM filtering (which can be bypassed)
```

---

## Part 2: Architecture Deep-Dive

### Core Components (8 Layers)

```
┌────────────────────────────────────────────────────────────────────┐
│                        Your Agent Code                             │
│                   (Strands / LangGraph / etc.)                      │
└────────────────────┬───────────────────────────────────────────────┘
                     │
        ┌────────────▼────────────────────────────────────────┐
        │          AgentCore Runtime                          │
        │  (Serverless microVM execution environment)         │
        └────────┬──────────────────────────────────────────┬─┘
                 │                                          │
    ┌────────────▼─────────────┐         ┌────────────────▼────┐
    │  Data Plane              │         │  Control Plane       │
    │  (Execution APIs)        │         │  (Management APIs)   │
    │                          │         │                      │
    │ • InvokeAgentRuntime     │         │ • CreateAgentRuntime │
    │ • StopRuntimeSession     │         │ • UpdateAgentRuntime │
    │ • InvokeCodeInterpreter  │         │ • DeleteAgentRuntime │
    │                          │         │ • Describe*          │
    └──────────────────────────┘         └──────────────────────┘
                 │
    ┌────────────┴───────────────────────────────────────────┐
    │                                                         │
    │  Internal Components (Auto-Managed)                   │
    │                                                         │
    ├─ Memory (short-term + long-term)                      │
    ├─ Gateway (tool hub → REST/Lambda/MCP)               │
    ├─ Code Interpreter (Python/JS/TS sandbox)            │
    ├─ Browser (cloud Playwright)                          │
    ├─ Policy Engine (Cedar rules evaluator)              │
    ├─ Identity (OAuth2 + IAM)                             │
    ├─ Observability (OTEL → CloudWatch)                  │
    └─ Evaluations (13 built-in quality scorers)          │
```

### Component 1: Runtime

**Purpose:** Serverless execution environment for agent code.

**Per-session specs:**
- CPU: 2 vCPU (Intel/ARM, non-adjustable)
- RAM: 8 GB (non-adjustable)
- Disk: 1 GB ephemeral storage
- Lifecycle: 15 min idle → auto-terminate, 8 hr max lifetime

**Session model:**
```
session = physical microVM instance
session.id = runtimeSessionId (you provide)
reuse same ID = reuse same microVM (state persists)
new ID = fresh microVM (clean slate)
```

**Key insight:** Session = stateful connection. Reuse sessionId for multi-turn conversations. Change sessionId only when you need isolation.

### Component 2: Memory

**Two-tier architecture:**

| Tier | Scope | Persistence | Use Case |
|---|---|---|---|
| **Short-term** | Single session | Session ephemeral | Current conversation |
| **Long-term** | Across sessions | Permanent | Cross-session context |

**Short-term memory:**
- Automatic conversation history tracking
- Token-aware pruning (respects context window)
- Cleared on session termination

**Long-term memory:**
- Survives session end
- Searched via 5 extraction strategies
- Indexed for semantic retrieval

#### Memory Strategies (How Long-Term Works)

| Strategy | Extracts | Search | Use Case |
|---|---|---|---|
| `SEMANTIC` | Factual statements | Vector cosine similarity | "What was mentioned about X?" |
| `SUMMARIZATION` | Rolled-up summaries | Text keyword match | "What happened last week?" |
| `USER_PREFERENCE` | Behavioral patterns | Exact key lookup | "Does user prefer async?" |
| `EPISODIC` | Task sequences + reflections | Intent matching | "What tasks did we attempt?" |
| `SELF_MANAGED` | Custom logic (your code) | Your query logic | Domain-specific retrieval |

**Memory API (Strands integration example):**

```python
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager
)

manager = AgentCoreMemorySessionManager(
    AgentCoreMemoryConfig(
        memory_id="mem-xyz",
        session_id="session-123",
        actor_id="user-456",
        # Namespace = logical partition in memory
        retrieval_config={
            "/users/user-456/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
            "/sessions/session-123/turns": RetrievalConfig(top_k=10),
        }
    ),
    region="us-west-2"
)

# Write to long-term
manager.store(
    namespace="/users/user-456/facts",
    content="User prefers async patterns"
)

# Read from long-term
results = manager.retrieve(
    query="What does user prefer?",
    namespace="/users/user-456/facts",
    top_k=3
)
```

**Memory limits (per account/region):**
- 150 memory resources total
- 6 strategies per resource
- 100 messages per CreateEvent
- 100 KB per message
- 10 MB per event
- 7–365 day expiration window
- 150,000 tokens/min extraction (semantic)
- 50,000 tokens/min extraction (episodic)
- 10 TPS CreateEvent rate
- 30 TPS RetrieveMemoryRecords rate

### Component 3: Gateway (Tool Hub)

**Purpose:** Unify REST APIs, Lambda functions, MCP servers behind single MCP endpoint.

**Architecture:**
```
Your agent code
     │
     ├─→ Gateway (MCP endpoint)
     │        │
     │        ├─→ REST API (e.g., Stripe, Twilio)
     │        ├─→ Lambda function (custom logic)
     │        ├─→ MCP server (e.g., file ops, GitHub)
     │        └─→ Internal tool (code interpreter)
     │
Your agent code never knows how tool is actually invoked
```

**Tool invocation protocol (MCP):**
```bash
curl -X POST https://<gateway-url>/tools/call \
  -H "Authorization: Bearer <token>" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "transfer_funds",
      "arguments": {
        "amount": 1000,
        "recipient": "user-456"
      }
    }
  }'
```

**Key insight:** Agent code calls tools via MCP protocol. Gateway abstracts implementation details (REST vs Lambda vs MCP).

**Gateway limits:**
- 1,000 gateways per account
- 100 targets per gateway
- 1,000 tools per target
- 15-minute invocation timeout
- 1,000 concurrent connections
- 1 MB inline schema, 10 MB S3 schema

### Component 4: Code Interpreter

**Purpose:** Sandboxed Python/JS/TS execution for agent-generated code.

**Specs:**
- Language: Python 3.9+, JavaScript (Node 18+), TypeScript
- Isolation: Complete sandbox (no host access)
- Capabilities:
  - Math libraries (numpy, pandas)
  - Data processing
  - File operations (read/write within sandbox)
  - Shell commands (limited)

**API:**
```python
import boto3

client = boto3.client('bedrock-agentcore')

# Execute Python code
response = client.invoke_code_interpreter(
    sessionId="my-session",
    name="executeCode",
    arguments={
        "language": "python",
        "code": "import pandas; df = pandas.DataFrame({'a': [1,2,3]}); print(df)"
    }
)

# Async long-running task
response = client.invoke_code_interpreter(
    sessionId="my-session",
    name="startCommandExecution",
    arguments={"command": "python train_model.py"}
)
task_id = response['taskId']

# Poll for status
status = client.invoke_code_interpreter(
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

files = client.invoke_code_interpreter(
    sessionId="my-session",
    name="readFiles",
    arguments={"paths": ["data.csv"]}
)
```

**Key constraint:** 10 GB disk per session, 8h max execution time.

### Component 5: Browser (Cloud Playwright)

**Purpose:** Cloud-hosted browser automation for web scraping/testing.

**Not documented in detail in GA release** but available as agent tool:
```python
# Via built-in tool (Strands)
from strands.tools import BrowserAutomation

agent.tools = [BrowserAutomation()]

# Agent can:
# - Navigate URLs
# - Fill forms
# - Click buttons
# - Extract text
# - Take screenshots
```

### Component 6: Policy Engine (Cedar)

**Cedar language:** Deterministic access control. **forbid always beats permit** (no probabilistic filtering).

**Evaluation model:**
```
Agent calls tool
  │
  ├─→ Cedar policy engine
  │    ├─→ Check all permit rules
  │    ├─→ Check all forbid rules
  │    ├─→ forbid matched? → DENY
  │    └─→ Any permit matched? → ALLOW (default: DENY)
  │
  └─→ Tool execution OR error
```

**Default:** Deny all. Must explicitly permit.

#### Cedar Syntax Reference

**Basic permit rule:**
```cedar
permit(
  principal is AgentCore::OAuthUser,
  action in [
    AgentCore::Action::"InsuranceAPI__get_policy",
    AgentCore::Action::"InsuranceAPI__claim_status"
  ],
  resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:us-west-2:123:gateway/insurance"
);
```

**Role-based access (forbid pattern):**
```cedar
forbid(
  principal is AgentCore::OAuthUser,
  action == AgentCore::Action::"InsuranceAPI__delete_policy",
  resource == AgentCore::Gateway::"arn:..."
) unless {
  principal.hasTag("role") &&
  (principal.getTag("role") == "admin" ||
   principal.getTag("role") == "claims-manager")
};
```

**Scope-based (OAuth scopes in JWT):**
```cedar
permit(
  principal is AgentCore::OAuthUser,
  action == AgentCore::Action::"InsuranceAPI__file_claim",
  resource == AgentCore::Gateway::"arn:..."
) when {
  principal.hasTag("scope") &&
  principal.getTag("scope") like "*insurance:claim:write*"
};
```

**Input-value conditional:**
```cedar
permit(
  principal is AgentCore::IamEntity,
  action == AgentCore::Action::"PaymentAPI__process_refund",
  resource == AgentCore::Gateway::"arn:..."
) when {
  principal.id like "*:111122223333:*" &&
  context.input has amount &&
  context.input.amount <= 10000  # Refunds max $10k
};
```

**Cedar constraints:**
- No floating-point arithmetic
- Pattern matching: `like` only (no regex)
- Principal types: `AgentCore::OAuthUser`, `AgentCore::IamEntity`
- forbid rule with unsatisfied `unless` condition = DENY

**Key insight:** Cedar is your deterministic guardrail layer. Use it to enforce hard business rules (max refund, role-based access, rate limits encoded in policy).

### Component 7: Identity & Authentication

**Inbound (who calls agent):**
- OAuth2 (e.g., Google, Okta, Auth0)
- IAM roles (AWS account principals)
- API keys (agent-specific)

**Outbound (agent calls external services):**
- OAuth2 token retrieval (pre-configured)
- API key injection (via SecureString parameter)
- IAM role assumption (for AWS services)

**Setup via CLI:**
```bash
agentcore add credential --type oauth2 --provider google --client-id xxx --client-secret yyy
agentcore deploy
# OAuth token auto-injected into agent code
```

### Component 8: Observability (OTEL + CloudWatch)

**Zero-code tracing:**
```bash
# In agentcore.json
'entryPoint': ['opentelemetry-instrument', 'main.py']
```

**Auto-captured data:**
- Request/response traces
- Tool invocation spans
- Model call latency
- Error stack traces
- Custom metrics (via OTEL SDK)

**Output destinations:**
- CloudWatch Logs: `/aws/bedrock-agentcore/runtimes/{agent_id}/`
- CloudWatch Metrics: `bedrock-agentcore` namespace
- CloudWatch Traces: `/aws/spans/default` (X-Ray compatible)

**CLI access:**
```bash
agentcore logs --since 30m --level error
agentcore traces list --limit 100
agentcore traces get <trace-id>
```

**Built-in evaluations (13 auto-scorers):**
- Faithfulness (does response match source?)
- Relevance (is answer on-topic?)
- Toxicity (harmful content?)
- Latency (performance ok?)
- Cost (token usage reasonable?)
- Tool accuracy (did tool work?)
- Hallucination (fabricated facts?)
- etc.

All automatic. No custom eval harness needed.

---

## Part 3: Session Model (Lifecycle & Limits)

### Session Lifecycle

```
Create → Idle → Active → Idle → Timeout (OR explicit stop)
         15m max
                8h max total lifetime
```

**State transitions:**
```
CREATING      → Setup microVM, prepare environment
ACTIVE        → Agent code executing
IDLE          → Waiting for next invocation
STOPPING      → Graceful shutdown, memory flush
STOPPED       → Session ended (microVM terminated)
ERROR         → Unrecoverable failure
```

### runtimeSessionId (Your Session Key)

```python
client.invoke_agent_runtime(
    agentRuntimeArn="arn:...",
    runtimeSessionId="user-123-session-456",  # You choose this
    payload=json.dumps({"prompt": "What's my balance?"}),
    qualifier="DEFAULT"
)

# Reuse same runtimeSessionId for multi-turn:
# Turn 1: runtimeSessionId="user-123-session-456" → agent sees empty history
# Turn 2: runtimeSessionId="user-123-session-456" → agent sees Turn 1 in memory
# Turn 3: runtimeSessionId="user-123-session-456" → agent sees Turns 1 + 2
```

**Design pattern:** `runtimeSessionId = "{user_id}-{conversation_id}"`

### Limits (Hard)

| Limit | Value | Notes |
|---|---|---|
| Session CPU | 2 vCPU (Intel/ARM) | Non-adjustable. Sufficient for most agents. |
| Session RAM | 8 GB | Non-adjustable. Includes agent code + libraries + working set. |
| Session storage | 1 GB ephemeral | Cleared on session termination. |
| Session lifetime | 8 hours max | Hard limit. Plan for session handoff. |
| Idle timeout | 15 minutes (default) | Configurable. Shorter = less cost. |
| Payload size | 100 MB | InvokeAgentRuntime request limit. |
| Request timeout | 15 minutes | Synchronous invoke max wait. |
| Streaming max | 60 minutes | For streaming responses. |
| Concurrent sessions | Account-level | Depends on provisioning. |

### Session Termination & Cleanup

```python
# Explicit stop (free up resources early)
client.stop_runtime_session(
    agentRuntimeArn="arn:...",
    runtimeSessionId="user-123-session-456",
    qualifier="DEFAULT"
)

# Automatic on:
# - Idle timeout exceeded
# - Max lifetime exceeded
# - Explicit stop
# - Unhandled error
```

**On termination:**
- Microvm destroyed
- 1 GB storage cleaned
- Short-term memory cleared
- Long-term memory persisted
- Logs flushed to CloudWatch

---

## Part 4: Building Agents

### Approach A: CLI (Recommended for Beginners)

**Interactive setup:**
```bash
agentcore create
# Prompts:
# - What framework? (Strands/LangGraph/CrewAI)
# - Model provider? (Bedrock/OpenAI/Anthropic)
# - Need memory? (Yes/No)
# - Need tools? (Yes/No)
```

**Generated structure:**
```
my-agent/
├── agentcore.json        # Config
├── main.py              # Entrypoint
├── agents/
│   └── search_agent.py  # Your framework code
├── requirements.txt
├── Dockerfile           # Optional (container deploy)
└── pyproject.toml
```

**Generated agentcore.json:**
```json
{
  "name": "MyAgent",
  "version": 1,
  "agents": [{
    "type": "AgentCoreRuntime",
    "name": "MyAgent",
    "build": "CodeZip",      // or "Container"
    "entrypoint": "main.py",
    "runtimeVersion": "PYTHON_3_13",
    "networkMode": "PUBLIC", // or "PRIVATE"
    "modelProvider": "Bedrock",
    "protocol": "HTTP"       // or "AGUI" (SSE/WebSocket)
  }],
  "memories": [],
  "credentials": [],
  "evaluators": [],
  "agentCoreGateways": [],
  "policyEngines": []
}
```

**Local dev → deploy:**
```bash
# Dev
agentcore dev --agent main.py
agentcore dev "What time is it?" --stream

# Dry-run
agentcore deploy --plan

# Deploy
agentcore deploy

# Invoke
agentcore invoke "Hello"
agentcore invoke "Continue" --session-id user-123-session-1 --stream
```

### Approach B: SDK (Code-First)

#### Minimal HTTP Agent

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel

app = BedrockAgentCoreApp()

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-5-20251001-v1:0"
)

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant."
)

@app.entrypoint
def invoke(payload):
    user_input = payload.get("prompt")
    response = agent(user_input)
    return response.message['content'][0]['text']

if __name__ == "__main__":
    app.run()  # HTTP server on :8080
```

#### Streaming Agent

```python
@app.entrypoint
async def invoke(request):
    prompt = request.get("prompt")
    async for event in agent.stream_async(prompt):
        yield event
```

#### AG-UI Protocol (SSE + WebSocket for Real-Time UI)

```python
from bedrock_agentcore.runtime import AGUIApp
from ag_ui.core import (
    RunAgentInput,
    RunStartedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    RunFinishedEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
)
import uuid

app = AGUIApp()

@app.entrypoint
async def my_agent(input_data: RunAgentInput):
    # Signal run started
    yield RunStartedEvent(
        thread_id=input_data.thread_id,
        run_id=input_data.run_id
    )

    msg_id = str(uuid.uuid4())
    yield TextMessageStartEvent(message_id=msg_id, role="assistant")

    # Get last user message
    user_msg = input_data.messages[-1].content

    # Tool calls appear as events
    for tool_name in agent.get_planned_tools(user_msg):
        tool_id = str(uuid.uuid4())
        yield ToolCallStartEvent(tool_id=tool_id, tool_name=tool_name)
        tool_result = agent.call_tool(tool_name, ...)
        yield ToolCallEndEvent(tool_id=tool_id, result=tool_result)

    # Stream response tokens
    response = agent(user_msg)
    for chunk in response.stream():
        yield TextMessageContentEvent(message_id=msg_id, delta=chunk)

    yield TextMessageEndEvent(message_id=msg_id)
    yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

app.run()
```

#### A2A Protocol (Agent-to-Agent Communication)

```python
from bedrock_agentcore.runtime import serve_a2a
from strands.a2a import StrandsA2AExecutor

serve_a2a(StrandsA2AExecutor(agent))

# Now other agents (Strands, LangGraph, etc.) can call this agent
# as if it were a local tool
```

---

## Part 5: Control & Data Plane APIs

### Control Plane (Management)

**Purpose:** Create, update, delete, describe runtimes.

#### Create Agent Runtime (CodeZip)

```python
import boto3

client = boto3.client('bedrock-agentcore-control', region_name='us-west-2')

response = client.create_agent_runtime(
    agentRuntimeName="my-agent-prod",
    agentRuntimeArtifact={
        'codeConfiguration': {
            'code': {
                's3': {
                    'bucket': f'bedrock-agentcore-code-{account_id}-us-west-2',
                    'prefix': 'my-agent/deployment_package.zip'
                }
            },
            'runtime': 'PYTHON_3_13',
            'entryPoint': [
                'opentelemetry-instrument',  # Enable tracing
                'main.py'
            ]
        }
    },
    networkConfiguration={'networkMode': 'PUBLIC'},  # or PRIVATE
    roleArn='arn:aws:iam::ACCOUNT_ID:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2',
    lifecycleConfiguration={
        'idleRuntimeSessionTimeout': 300,   # 5 min (configurable)
        'maxLifetime': 28800                # 8 hour (fixed)
    },
    tags={'Environment': 'production', 'Team': 'AI'}
)

agent_id = response['agentRuntimeId']
agent_arn = response['agentRuntimeArn']
```

#### Create Agent Runtime (Container)

```python
response = client.create_agent_runtime(
    agentRuntimeName="my-container-agent",
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': f'{account}.dkr.ecr.{region}.amazonaws.com/my-agent:latest',
            'imageTag': 'latest',  # Optional: pin specific version
        }
    },
    networkConfiguration={'networkMode': 'PUBLIC'},
    roleArn='arn:aws:iam::ACCOUNT_ID:role/...'
)
```

#### Update Agent Runtime

```python
client.update_agent_runtime(
    agentRuntimeId='agent-xyz',
    agentRuntimeArtifact={
        'codeConfiguration': {
            'code': {'s3': {'bucket': '...', 'prefix': 'v2/...'}},
            'runtime': 'PYTHON_3_13',
            'entryPoint': ['opentelemetry-instrument', 'main.py']
        }
    }
)
# → New version created, DEFAULT endpoint updated (zero-downtime)
```

#### Versioning & Endpoints

```
Initial: CreateAgentRuntime → Version 1 (auto) → DEFAULT endpoint (auto)
Update:  UpdateAgentRuntime → Version 2 (auto) → DEFAULT endpoint points to v2
Custom:  CreateEndpoint(versionId=1, name="canary") → v1 still available

Patterns:
- Blue-green: NEW endpoint points to v2, old clients still on v1
- Canary: 10% traffic to v2, 90% to v1 via load balancer
- Rollback: CreateEndpoint pointing to v1 if v2 fails
```

#### List & Describe

```python
# List all runtimes
runtimes = client.list_agent_runtimes(maxResults=50)

# Describe specific runtime
runtime = client.describe_agent_runtime(agentRuntimeId='agent-xyz')
# Returns: name, ARN, status, versions, endpoints, created_time, etc.

# Describe version
version = client.describe_agent_runtime_version(
    agentRuntimeId='agent-xyz',
    version='2'
)
```

#### Delete

```python
client.delete_agent_runtime(agentRuntimeId='agent-xyz')
# Cascades: deletes all versions, endpoints, sessions
```

### Data Plane (Invocation)

**Purpose:** Invoke agents, manage sessions.

#### Synchronous Invoke

```python
import boto3, json

client = boto3.client('bedrock-agentcore', region_name='us-west-2')

response = client.invoke_agent_runtime(
    agentRuntimeArn='arn:aws:bedrock-agentcore:us-west-2:123:...',
    runtimeSessionId='user-123-session-1',
    payload=json.dumps({'prompt': 'What is my account balance?'}),
    qualifier='DEFAULT'  # or specific version alias
)

# Result is streamed
result = json.loads(response['response'].read())
```

#### Streaming Invoke

```python
response = client.invoke_agent_runtime(
    agentRuntimeArn='...',
    runtimeSessionId='...',
    payload=json.dumps({'prompt': '...'}),
    responseStream=True  # Enable streaming
)

# Handle streaming
for event in response['response']:
    if event['type'] == 'message':
        print(event['message'], end='', flush=True)
    elif event['type'] == 'tool_call':
        print(f"\n[Tool: {event['tool_name']}]")
```

#### Stop Session

```python
client.stop_runtime_session(
    agentRuntimeArn='...',
    runtimeSessionId='user-123-session-1',
    qualifier='DEFAULT'
)
# Frees microVM immediately (no 15-min idle wait)
```

---

## Part 6: Deployment Patterns

### CodeZip Deployment (for Python)

**Step 1: Package for ARM64 (required)**
```bash
# Use uv (faster, more reliable than pip)
uv pip install \
  --python-platform aarch64-manylinux2014 \
  --python-version 3.13 \
  --target=deployment_package \
  --only-binary=:all: \
  -r pyproject.toml

# Manual binary download if uv fails
pip install --platform manylinux2014_aarch64 --target=pkg ...
```

**Step 2: Package**
```bash
cd deployment_package
zip -r ../deployment_package.zip .
cd ..
zip deployment_package.zip main.py agents/

# Verify (ARM64 only)
unzip -l deployment_package.zip | grep '\.so' | head -5
# Should show: libc.so.6, libm.so.6 (manylinux tag)
```

**Step 3: Upload to S3**
```python
import boto3

s3 = boto3.client('s3')

account_id = boto3.client('sts').get_caller_identity()['Account']
bucket = f'bedrock-agentcore-code-{account_id}-us-west-2'
prefix = 'my-agent/deployment_package.zip'

s3.upload_file(
    'deployment_package.zip',
    bucket,
    prefix
)
```

**Constraints:**
- Max compressed: 250 MB
- Max uncompressed: 750 MB
- ARM64 only (no x86_64)

### Container Deployment (for Complex Dependencies)

**Dockerfile:**
```dockerfile
FROM public.ecr.aws/lambda/python:3.13

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agents/ agents/
COPY main.py .

# AgentCore expects this entrypoint
ENTRYPOINT ["python", "main.py"]
```

**Build & Push:**
```bash
docker build -t my-agent:latest .

# Authenticate with ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  ${ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com

# Push
docker tag my-agent:latest \
  ${ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest
docker push ${ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest
```

**Constraints:**
- Max image: 2 GB
- Must support arm64 (if using ARM)
- Startup time ~30s (vs CodeZip ~5s)

### Versioning & Deployment Strategy

```
Manual version tracking:
  agentcore.json version field = your SemVer (1.0.0, 1.1.0, etc.)
  Control Plane assigns numeric versions (1, 2, 3, ...) on each update
  You create named endpoints (prod, staging, canary) pointing to versions
```

**Production strategy:**
```
Deployment flow:
1. Update code locally
2. Create CodeZip or build container
3. agentcore deploy (creates new version automatically)
4. Test on DEFAULT endpoint (usually stagingcanary)
5. Create new endpoint "prod-v2" pointing to new version
6. Gradual traffic shift (load balancer: 10% → 25% → 50% → 100%)
7. Keep old endpoint for quick rollback
```

---

## Part 7: Pricing & Costs

### Compute Pricing

**Runtime execution:**
- $0.0895/vCPU-hour
- $0.00945/GB-hour
- Per-second granularity (not per-minute)

**Example cost calculation:**
```
Agent running 60 seconds:
  CPU: 2 vCPU × $0.0895/hour = $0.1790/hour
  RAM: 8 GB × $0.00945/hour = $0.0756/hour
  Total/hour: $0.2546
  Per minute: $0.2546 / 60 = $0.00424
  Per 60 seconds: $0.00424 ✓

Agent running 1 hour continuously:
  CPU: 2 × $0.0895 = $0.1790
  RAM: 8 × $0.00945 = $0.0756
  Total: $0.2546
```

**I/O wait is FREE:**
- Agent waiting for LLM response (5 seconds)
- Agent waiting for API call (2 seconds)
- Agent waiting for database query (3 seconds)
→ None of these count toward billing

**Real-world savings:** Agents spend 30–70% in I/O wait → ~50% cheaper than provisioned EC2.

### A La Carte Components

| Component | Pricing |
|---|---|
| Memory | Included (up to 150 per account) |
| Gateway | $X/tool-invocation (details TBD) |
| Code Interpreter | Included (up to 10GB/session) |
| Browser | $X/session-minute (details TBD) |
| Policy | Included (Cedar evaluation) |
| Observability | Included (OTEL spans → CloudWatch) |
| Evaluations | Included (13 auto-scorers) |
| **LLM Inference** | **Separate** (pay Claude, GPT-4, etc.) |

**Key insight:** AgentCore charges for infrastructure. LLM calls are billed separately by the LLM provider.

---

## Part 8: Limits, Constraints, & Gotchas

### Hard Limits (Non-Negotiable)

| Resource | Limit | Impact |
|---|---|---|
| Session CPU | 2 vCPU | Can't run CPU-intensive ML models inside agent. Use external API. |
| Session RAM | 8 GB | Keep agent code + libraries lightweight. Large vector DBs offline. |
| Session storage | 1 GB | Cache to S3, not local disk. |
| Session lifetime | 8 hours | Plan for handoff. Don't start long-running batch jobs in session. |
| Payload | 100 MB | Large file uploads? Use S3 pre-signed URLs. |
| Concurrent sessions | Account-limit | Vertical scaling via more parallel clients. |
| Gateway timeout | 15 min | Tool calls can't exceed 15 minutes. |
| Cedar pattern match | No regex | Use `like "*pattern*"` only. |

### Soft Limits (Can Be Increased)

| Resource | Default | How to Increase |
|---|---|---|
| Memory resources/account | 150 | Contact AWS Support |
| Gateways/account | 1,000 | Contact AWS Support |
| Idle timeout | 15 min | UpdateAgentRuntime → lifecycleConfiguration |

### Design Gotchas

1. **Session isolation is good; state sharing is hard**
   - Each runtimeSessionId = separate microVM
   - Shared state must go through external service (Redis, DynamoDB)
   - No in-process global state across sessions

2. **Short-term memory is per-agent, not per-user**
   - Memory tied to Agent instance in Strands
   - Multiple agents = multiple memory instances
   - Design: one Agent per user, or external session manager

3. **I/O wait is free, but model inference is expensive**
   - Agent paused waiting for Claude response = free
   - Claude generating tokens = billed by Anthropic
   - Design: batch requests, reuse conversations (cheaper than repeated invokes)

4. **Cedar forbid beats permit; be careful with deny all**
   - Default: deny all (must explicitly permit)
   - Typo in permit rule = agent can't call any tools
   - Test Cedar policies locally before deploy

5. **Session termination clears short-term memory**
   - Long-term memory survives
   - If you need history after session ends, query long-term memory
   - Design: summarize + store to long-term before session ends

---

## Part 9: Interview Questions & Answers

### Q1: Design a customer support agent with AgentCore. What's your architecture?

**Answer outline:**
```
Components:
- Strands Agent (model-driven orchestration)
- Tools: Bedrock KB (doc retrieval), database (customer lookup), email (send resolution)
- Memory: short-term (conversation), long-term (customer history via SEMANTIC strategy)
- Gateway: wraps customer DB + email API
- Policy: Cedar rule forbids refunds > $5k without manager approval
- Session model: runtimeSessionId = user_id + ticket_id (reuse for multi-turn)

Flow:
1. Customer question → Agent invoked with runtimeSessionId
2. Agent sees short-term history (this conversation)
3. Tool retrieval: KB search + customer DB lookup
4. Reasoning: LLM decides if refund, escalation, or standard answer
5. If refund < $5k: Cedar permits → Gateway executes
6. If refund > $5k: Cedar forbids → Agent suggests escalation
7. Session ends → long-term memory saves customer preference for next time

Cost: ~$0.0025/min execution + LLM tokens (Claude Haiku for fast response)
```

### Q2: How do you handle a 24-hour customer conversation spanning multiple sessions?

**Answer:**
```
Problem: 8-hour session limit, conversation needs 24 hours.

Solution: Multiple sessions with memory handoff
- Session 1 (8h): runtimeSessionId = "user-123-session-1"
  - Conversation turns 1–200
  - At 8h: summarize conversation → store to long-term memory (SUMMARIZATION strategy)
  - Call client.stop_runtime_session() → save resources

- Session 2 (next day): runtimeSessionId = "user-123-session-2"
  - Agent retrieves summary from long-term memory
  - Loads context: "Previous session: customer asked X, we resolved Y"
  - Conversation continues from turn 201
  - No loss of context; just different microVM

- Session 3 (next day): repeat

Key: Long-term memory bridges 8-hour session boundary.
```

### Q3: Strands vs LangGraph in AgentCore. Which and why?

**Answer:**
```
Use Strands when:
- Agent needs to be autonomous (LLM decides tool calls, not dev)
- Multi-agent coordination desired (swarm, hierarchical patterns)
- Rapid prototyping (docstring → tool spec)
- Real-time streaming critical (built-in)
- Memory auto-management important

Use LangGraph when:
- Fixed workflow critical (e.g., validation → processing → review → publish)
- Full control over state transitions needed
- Branching logic complex (explicit edges, conditions)
- Already invested in LangChain ecosystem
- Deterministic output required

AgentCore doesn't care. Both frameworks run in same microVM.
Recommendation: Strands for most agents (simpler). LangGraph for ETL-like workflows.
```

### Q4: You accidentally deny all tools with broken Cedar policy. How do you fix?

**Answer:**
```
Scenario: Cedar rule has typo. Agent can't call any tools.

Quick fix (don't do this):
  remove Cedar policy rule → all tools now permitted (default deny all broken)
  
Correct fix:
1. Check policy via AWS console or CLI
2. Identify typo (e.g., hasTag("role") → hasTag("role_type"))
3. Update policy (not via agent code, via control plane)
4. Test locally with cedar-cli before deploy
5. Deploy new version

Prevention:
- Store Cedar policies in version control (YAML/JSON)
- Test with cedar-cli locally:
  cedar validate -s schema.json -p policies.json
- Gradual rollout (canary endpoint with new policy)
- Metrics: track tool call success rate (alert if drops)
```

### Q5: Cost optimization for high-volume agent. What's your strategy?

**Answer:**
```
Problem: Agent serving 1000 requests/day. Cost too high.

Optimization layers:

1. Model selection
  - Use Haiku (cheap) for classification → Sonnet for complex reasoning
  - Fallback chain: Haiku → Sonnet → GPT-4 (only if Haiku fails)

2. Execution efficiency
  - Batch requests: combine 10 questions → single invocation
  - Cache tool results: memoize identical queries
  - Reuse sessions: same runtimeSessionId for multi-turn (cheaper than new session)

3. I/O wait free
  - LLM thinking time doesn't count (free pause while waiting)
  - Design agent to parallelize tool calls (gather all data, then reason once)

4. Memory strategy
  - EPISODIC (task sequences) cheaper than SEMANTIC (vector search)
  - Limit long-term retrieval (top_k=3, not top_k=20)

5. Session tuning
  - Shorter idleTimeout (e.g., 5 min) for short-lived agents
  - Don't hold sessions open waiting for user input (pause before invoke)

Estimate: 1000 req/day × 30 sec execution = 500 min/day
         2 vCPU × $0.0895/h ÷ 60 = $0.00298/min
         500 min × $0.00298 = $1.49/day compute
         + LLM cost (~$0.01–$0.10 per request depending on model)
         Total: ~$10–$20/day compute + LLM
```

---

## Part 10: Common Patterns & Best Practices

### Pattern 1: Customer Support with Escalation

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

@app.entrypoint
def support_agent(payload, context):
    user_id = context.user_id
    session_id = f"{user_id}-support-{context.timestamp}"
    
    agent = Agent(
        model="claude-3-5-sonnet-20241022",
        tools=[
            CustomerDB(),
            KnowledgeBase(),
            EmailClient(),
            RefundProcessor(),
        ],
        system_prompt="""
        You are a support agent. Resolve customer issues with tools.
        If refund needed:
        - <$500: approve via RefundProcessor
        - ≥$500: summarize issue + recommend manager escalation
        """
    )
    
    response = agent(payload["query"])
    
    # Store outcome in long-term memory
    session_manager.store(
        namespace=f"/customers/{user_id}/issues",
        content=f"Issue: {payload['query']}, Resolution: {response}"
    )
    
    return response
```

### Pattern 2: Data Processing Pipeline (LangGraph)

```python
from langgraph.graph import StateGraph
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

state_graph = StateGraph()

# Nodes
async def validate(state):
    # Validate input
    return {"status": "validated", ...}

async def process(state):
    # Process data
    return {"result": processed_data, ...}

async def store(state):
    # Save to database
    return {"stored": true, ...}

# Edges
state_graph.add_node("validate", validate)
state_graph.add_node("process", process)
state_graph.add_node("store", store)
state_graph.add_edge("validate", "process")
state_graph.add_edge("process", "store")

@app.entrypoint
def process_data(payload):
    result = state_graph.invoke({"input": payload["data"]})
    return result

app.run()
```

### Pattern 3: Multi-Session Conversation (Strands)

```python
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager
)

def get_or_create_agent(user_id):
    # Always create fresh agent, memory persists across sessions
    memory_mgr = AgentCoreMemorySessionManager(
        config=AgentCoreMemoryConfig(
            memory_id=os.getenv("MEMORY_ID"),
            session_id=str(uuid.uuid4()),  # New session each invoke
            actor_id=user_id,
            retrieval_config={
                f"/users/{user_id}/preferences": RetrievalConfig(top_k=5),
                f"/users/{user_id}/history": RetrievalConfig(top_k=10),
            }
        ),
        region="us-west-2"
    )
    
    agent = Agent(
        model="claude-3-5-sonnet",
        session_manager=memory_mgr,
        tools=[...]
    )
    
    return agent

@app.entrypoint
def chat(payload, context):
    agent = get_or_create_agent(context.user_id)
    # Memory manager loads user history automatically
    response = agent(payload["message"])
    return response
```

---

## Part 11: Summary Table (Quick Reference)

| Aspect | Detail |
|---|---|
| **What is AgentCore?** | Managed serverless infrastructure for agents (not a framework) |
| **Cost model** | $0.0895/vCPU-hr + $0.00945/GB-hr; I/O wait free; per-second billing |
| **Session scope** | 2 vCPU, 8 GB RAM, 1 GB disk, 8h max, 15m idle timeout |
| **Memory** | Short-term (session), Long-term (cross-session, 5 strategies) |
| **Security** | Cedar policies (deterministic), OAuth2 + IAM |
| **Observability** | OTEL traces → CloudWatch, 13 auto-evaluators |
| **Tools** | Gateway (REST/Lambda/MCP), Code Interpreter, Browser |
| **Deployment** | CodeZip (250 MB) or Container (2 GB) |
| **Frameworks** | Strands, LangGraph, CrewAI, OpenAI SDK, Google ADK |
| **Models** | Any (Claude, GPT-4, Gemini, Llama, Nova) |
| **Multi-agent** | A2A protocol (cross-framework orchestration) |

---

**Created for:** Complete mastery before interview. Every detail needed for production AgentCore agents.
