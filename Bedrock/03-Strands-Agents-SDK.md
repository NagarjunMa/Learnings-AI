# Strands Agents SDK — Model-Driven AI Agents Framework

**Strands Agents** is an open-source Python framework for building multi-agent AI systems with automatic conversation history management, built-in tools, and multi-agent coordination patterns. Unlike developer-driven frameworks (LangChain, LangGraph), Strands takes a **model-driven approach** where the LLM orchestrates agent interactions dynamically.

---

## Core Architecture

### Agent Model

Strands agents are defined declaratively via Python dataclasses:

```python
from strands.agents import Agent
from strands.memory import ShortTermMemory
from dataclasses import dataclass

@dataclass
class SearchAgent(Agent):
    """Search specialist agent"""
    name: str = "SearchAgent"
    model: str = "claude-3-5-sonnet-20241022"  # Any supported model
    description: str = "Searches documents and web"
    system_prompt: str = "You are a research expert..."
    memory: ShortTermMemory = ShortTermMemory()  # Auto-managed
    tools: list = None  # Auto-populated from methods

# Agent becomes autonomous — LLM decides when to call tools
agent = SearchAgent()
```

**Key difference from LangChain:**
- LangChain: Developer explicitly chains steps → `agent.run(query)` → fixed flow
- Strands: LLM sees available tools and decides → dynamic, flexible orchestration

---

## Memory System (Auto-Managed)

### Short-Term Memory

Automatically tracks conversation history without manual save/load:

```python
from strands.memory import ShortTermMemory

# Default: keeps last 10 turns in context
memory = ShortTermMemory(max_turns=10)

# Every agent call updates memory
agent.memory.add_message("user", "Find best AI papers")
agent.memory.add_message("assistant", "Searching...")

# LLM always gets full context
messages = agent.memory.get_context()  # Full history for this session
```

**Automatic features:**
- Conversation continuity (no manual state management)
- Token-aware pruning (removes old messages if over limit)
- Turn tracking (preserves conversation ordering)

### Long-Term Memory (Multi-Session)

Persist knowledge across separate conversations:

```python
from strands.memory import LongTermMemory

# Store facts learned in past conversations
ltm = LongTermMemory(storage="redis", namespace="user_123")

# Current session can query history
past_context = ltm.retrieve("What did user prefer last time?")

# After session, summarize and store
ltm.store({
    "user_preferences": ["Python > JavaScript", "LLMs > Traditional ML"],
    "knowledge": ["User works in finance", "Prefers security-first design"]
})
```

---

## Tool System (20+ Built-in)

### Built-in Tool Categories

```python
from strands.tools import (
    WebSearch,           # Google, Bing, DuckDuckGo
    DocumentSearch,      # Vector search in PDFs
    Calculator,          # Math expressions
    DateTimeUtils,       # Date/time operations
    FileSystem,          # Read/write files
    EmailClient,         # Send emails
    SlackClient,         # Post to Slack
    GitHubClient,        # List repos, create issues
    DatabaseQuery,       # SQL execution
    APIClient,           # HTTP calls with auth
    Summarizer,          # Summarize text
    CodeExecutor,        # Python/JS execution
)

agent.tools = [
    WebSearch(),
    DatabaseQuery(connection_string="..."),
    Calculator(),
]
```

### Custom Tool Definition

```python
from strands.tools import Tool
from typing import Any

class CustomTool(Tool):
    """Define tool with automatic docstring parsing"""

    name: str = "calculate_roi"
    description: str = "Calculate ROI for investment"

    def execute(self, investment: float, returns: float) -> dict:
        """
        Calculate return on investment

        Args:
            investment: Initial investment amount
            returns: Final returns amount

        Returns:
            dict with roi_percent and roi_absolute
        """
        roi = ((returns - investment) / investment) * 100
        return {
            "roi_percent": roi,
            "roi_absolute": returns - investment
        }

# Tool automatically available to agent (docstring used as LLM spec)
agent.tools.append(CustomTool())
```

**Strands auto-generates LLM tool specs from Python docstrings** — no manual JSON schema needed.

---

## Multi-Agent Patterns

### Pattern 1: Agents as Tools (Hierarchical)

Agents can call other agents as tools:

```python
from strands.agents import Agent, AgentPool

class ResearchAgent(Agent):
    name = "ResearchAgent"
    description = "Gathers background information"

class AnalysisAgent(Agent):
    name = "AnalysisAgent"
    description = "Analyzes research findings"

class ReportAgent(Agent):
    name = "ReportAgent"
    description = "Writes final report"

    def __post_init__(self):
        # Other agents become tools
        self.tools = [
            ResearchAgent(),  # Can call research_agent
            AnalysisAgent(),  # Can call analysis_agent
        ]

# LLM orchestrates: ReportAgent → calls ResearchAgent → calls AnalysisAgent
coordinator = ReportAgent(
    system_prompt="You are a report writer. Use ResearchAgent to gather info, then AnalysisAgent to analyze."
)
result = coordinator.run("Write a report on AI trends in 2025")
```

### Pattern 2: Swarm (Peer-to-Peer)

All agents operate at same level, coordinate implicitly:

```python
from strands.swarm import Swarm

agents = [
    Agent(name="DataCollector", tools=[WebSearch()]),
    Agent(name="DataProcessor", tools=[Calculator(), DatabaseQuery()]),
    Agent(name="ReportWriter", tools=[FileSystem()]),
]

swarm = Swarm(agents=agents, model="claude-3-5-sonnet-20241022")

# Swarm routes queries to best agent automatically
response = swarm.run("Analyze Q3 sales trends and generate report")
# → Router sends to DataCollector, then DataProcessor, then ReportWriter
```

### Pattern 3: Graph (Deterministic DAG)

Explicit workflow graph for structured problems:

```python
from strands.graph import AgentGraph, Node, Edge

graph = AgentGraph()

# Define nodes
validate_node = Node(agent=ValidatorAgent(), name="validate_input")
process_node = Node(agent=ProcessorAgent(), name="process")
review_node = Node(agent=ReviewerAgent(), name="review")
publish_node = Node(agent=PublisherAgent(), name="publish")

# Define edges (deterministic routing)
graph.add_node(validate_node)
graph.add_node(process_node)
graph.add_node(review_node)
graph.add_node(publish_node)

graph.add_edge(validate_node, process_node, condition="is_valid")
graph.add_edge(process_node, review_node)
graph.add_edge(review_node, publish_node, condition="passed_review")
graph.add_edge(review_node, validate_node, condition="needs_revision")  # Loop

# Execute graph
result = graph.run(input_data={"document": "..."})
```

### Pattern 4: A2A Protocol (Strands ↔ Bedrock Agents)

Strands agents communicate with AWS Bedrock agents seamlessly:

```python
from strands.bedrock import BedrockBridge

# Strands agent calls Bedrock agent
strands_agent = Agent(name="DataAnalyst")
bedrock_agent = BedrockBridge(agent_id="arn:aws:bedrock:us-east-1:...")

strands_agent.tools = [bedrock_agent]

# LLM orchestrates Strands + Bedrock agents transparently
result = strands_agent.run("Analyze data and generate compliance report")
# → Strands calls Bedrock agent for compliance logic
```

**A2A Protocol enables:**
- Cross-framework orchestration (Strands + Bedrock + LangGraph)
- Hybrid LLM selection (Claude for reasoning, GPT-4 for code, Llama for summarization)
- Distributed agent networks

---

## Model Support

Strands works with **any LLM via unified interface**:

```python
# Anthropic Claude (recommended for reasoning)
agent = Agent(model="claude-3-5-sonnet-20241022")

# OpenAI GPT
agent = Agent(model="gpt-4-turbo", provider="openai", api_key="...")

# AWS Bedrock
agent = Agent(model="us.anthropic.claude-3-5-sonnet-20241022-v2:0", provider="bedrock")

# Open-source (via Ollama or RunPod)
agent = Agent(model="llama-2-70b", provider="ollama", base_url="http://localhost:11434")

# Model switching mid-conversation
if complex_task:
    agent.model = "claude-3-5-sonnet-20241022"  # Expensive but accurate
else:
    agent.model = "claude-3-5-haiku-20241022"    # Cheap and fast
```

**Strands automatically handles:**
- Prompt formatting per model
- Token counting
- Rate limiting
- Fallback models

---

## Configuration & Customization

### Agent Configuration

```python
from strands.agents import Agent
from strands.config import AgentConfig

config = AgentConfig(
    temperature=0.7,           # LLM temperature
    max_tokens=2048,           # Output limit
    top_p=0.9,                 # Nucleus sampling
    frequency_penalty=0.1,     # Reduce repetition
    presence_penalty=0.0,      # Encourage new topics
    timeout=30,                # seconds
    max_retries=3,
    retry_delay=1,             # exponential backoff
)

agent = Agent(
    name="AnalysisAgent",
    config=config,
    system_prompt="You are a financial analyst..."
)
```

### Observability

```python
from strands.observability import Tracer, Logger

# Auto-trace all agent calls
tracer = Tracer(backend="otel")  # OpenTelemetry export

agent = Agent(
    name="SearchAgent",
    tracer=tracer,
    log_level="DEBUG"
)

# Automatic logging of:
# - Tool invocations
# - Model calls (prompt + completion)
# - Agent-to-agent handoffs
# - Memory access
# - Errors and retries
```

---

## Streaming & Real-Time Response

### Server-Sent Events (SSE)

Stream agent responses in real-time:

```python
from strands.streaming import StreamAgent

agent = StreamAgent(name="ReportWriter")

# Streaming callback
def on_token(token: str):
    print(token, end="", flush=True)

response = agent.stream_run(
    query="Write a detailed market analysis",
    on_token=on_token,  # Called for each token
    on_tool_call=lambda tool, args: print(f"\n[Tool: {tool}]"),  # Called when agent calls tool
)
```

### WebSocket (Real-Time Bi-Directional)

```python
from strands.websocket import WebSocketAgent

agent = WebSocketAgent(name="ChatBot")

async def handle_message(message: str):
    async for chunk in agent.astream(message):
        await ws.send(chunk)  # Send each token to client

# Client receives tokens in real-time (like ChatGPT)
```

---

## Integration Patterns

### With Bedrock Knowledge Bases

```python
from strands.agents import Agent
from strands.bedrock import BedrockKnowledgeBase

agent = Agent(name="DocumentAgent")

kb = BedrockKnowledgeBase(
    knowledge_base_id="...",
    region="us-east-1"
)

# Agent retrieves from Bedrock KB automatically
agent.tools = [kb.retrieval_tool()]

response = agent.run("What does section 3.2 of our policy document say?")
```

### With Vector Databases

```python
from strands.agents import Agent
from strands.integrations import PineconeDB

agent = Agent(name="KnowledgeAgent")

vector_db = PineconeDB(
    api_key="...",
    index_name="documents",
    namespace="legal"
)

agent.tools = [
    vector_db.similarity_search_tool(top_k=5),  # Auto-generated tool
]

response = agent.run("Find similar cases to X")
```

### With Databases

```python
from strands.agents import Agent
from strands.integrations import PostgreSQL

agent = Agent(name="DataAnalyst")

db = PostgreSQL(connection_string="postgresql://...")

agent.tools = [
    db.query_tool(schema_hint="users, transactions, accounts"),
]

response = agent.run("How many transactions were fraudulent last month?")
# LLM generates SQL, executes, interprets results
```

---

## Deployment

### Local Development

```bash
pip install strands-agents
python -m strands dev --agent my_agent.py --port 8000
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agents/ agents/

CMD ["python", "-m", "strands", "serve", "--agent", "agents/main.py"]
```

### Cloud Deployment (AWS Lambda)

```python
from strands.lambda_handler import StradsLambdaHandler

agent = Agent(name="APIAgent", tools=[...])

handler = StradsLambdaHandler(agent)

# AWS Lambda automatically invokes handler(event, context)
# Agent runs in 15-minute execution window
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strands-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: agent
        image: my-strands-agent:latest
        env:
        - name: MODEL
          value: "claude-3-5-sonnet-20241022"
        - name: MEMORY_BACKEND
          value: "redis"
          value: "redis://redis-service:6379"
        ports:
        - containerPort: 8000
```

---

## Comparison with Alternatives

| Feature | Strands | LangGraph | Bedrock Agents | LangChain |
|---|---|---|---|---|
| **Programming Model** | Model-driven (LLM decides) | Developer-driven (explicit DAG) | Managed (AWS-only) | Sequential (explicit chains) |
| **Memory Management** | Auto-managed per agent | Manual state dict | Auto-managed sessions | Manual conversation history |
| **Multi-Agent Coordination** | Dynamic (LLM orchestration) | Fixed DAG | Fixed workflow | Limited (sequential only) |
| **Tool Definition** | Python docstrings → auto spec | Manual tool definitions | AWS-specific definitions | Manual tool schemas |
| **Model Flexibility** | Any LLM (Claude, GPT, Llama) | Any LLM + custom LLM | Claude, GPT-4, Llama (AWS only) | Any LLM |
| **Built-in Tools** | 20+ (search, DB, email, etc) | 100+ but not batteries-included | Limited (AWS-specific) | 1000+ but scattered |
| **Learning Curve** | Moderate (dataclass-based) | Steep (graph concepts) | Moderate (AWS knowledge needed) | Steep (many APIs) |
| **Deployment** | Docker, Lambda, K8s | DIY (single app) | AWS Lambda, Container | DIY (single app) |
| **Observability** | Built-in (OpenTelemetry) | Manual (LangSmith optional) | AWS CloudWatch + X-Ray | LangSmith required (paid) |
| **Streaming** | Native (SSE, WebSocket) | Manual (queue-based) | Native (SSE via API) | Limited (agent executor) |

**When to use Strands:**
- Multi-agent systems where LLM orchestrates
- Rapid prototyping (docstring → tool spec)
- Need real-time streaming
- Mix of models (Claude + GPT + Llama)
- Autonomous behavior (less developer code)

**When to use LangGraph:**
- Deterministic workflows (fixed DAG critical)
- Complex branching logic
- Full control over flow
- Already invested in LangChain ecosystem

**When to use Bedrock Agents:**
- AWS-native infrastructure
- Compliance + data residency critical
- Need managed scaling
- Don't want to manage deployments

---

## Limitations & Constraints

```python
# Strands has several hard limits:

# 1. Context Window (per agent, not shared)
# - Short-term: 10 turns default (configurable)
# - Long-term: unbounded but slower retrieval

# 2. Model Cost (no multi-step cost optimization)
# - LangGraph: developer can cache/batch steps
# - Strands: each agent call = fresh LLM call

# 3. Determinism (LLM may behave differently each run)
# - Use temperature=0 for reproducibility
# - Graph pattern if you need guaranteed flow

# 4. Tool Availability (all agents see all tools)
# - Can't restrict tools per agent (design flaw)
# - Workaround: separate tool namespaces

# 5. Debugging (less visibility than LangGraph)
# - No built-in step-by-step inspection
# - Rely on OpenTelemetry traces
```

---

## GitHub & Community

- **GitHub:** [strands-agents/strands](https://github.com/strands-agents/strands)
- **Docs:** https://docs.strandsai.com
- **Examples:** 50+ in `/examples` folder (Discord bot, RAG system, multi-agent research)
- **Community:** Discord (3K+ members), weekly office hours

---

## Quick Start

```python
from strands.agents import Agent
from strands.tools import WebSearch, Calculator

# 1. Define agent
agent = Agent(
    name="ResearchBot",
    model="claude-3-5-sonnet-20241022",
    system_prompt="You are a research assistant. Use tools to find accurate information.",
    tools=[WebSearch(), Calculator()],
)

# 2. Run (memory auto-managed)
response = agent.run("What's the GDP of Japan divided by its population?")
print(response)

# 3. Continue conversation (memory persists)
response = agent.run("Compare that to Germany")
print(response)  # Agent remembers Japan GDP from previous turn
```

---

## Production Checklist

- [ ] Configure appropriate model (Sonnet for reasoning, Haiku for cost)
- [ ] Set temperature=0 if determinism required
- [ ] Enable OpenTelemetry tracing (production observability)
- [ ] Test tool failure modes (network, timeouts, bad input)
- [ ] Set max_retries and timeout values per tool
- [ ] Monitor token usage (expensive if misconfigured)
- [ ] Implement input validation before agent (prevent injection)
- [ ] Use separate agent instances per user (memory isolation)
- [ ] Store long-term memory in persistent backend (Redis, DynamoDB)
- [ ] Test multi-agent coordination (ensure agents call each other correctly)
- [ ] Document system prompt (what behaviors are enabled)
- [ ] Review tool definitions (prevent unintended side effects)
