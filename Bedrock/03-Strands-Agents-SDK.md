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

## Hooks / Lifecycle Events (Extensibility)

Hooks are composable event subscriptions to agent lifecycle events. Use for tool interception, result modification, invocation resumption, and exception handling.

### Event Types

```python
from strands.hooks import BeforeInvocationEvent, AfterInvocationEvent
from strands.hooks import BeforeModelCallEvent, AfterModelCallEvent
from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent

# Single-agent events
@agent.on(BeforeInvocationEvent)
def before_invoke(event: BeforeInvocationEvent):
    """Fires before agent.invoke() starts"""
    print(f"Invoking with: {event.message}")

@agent.on(BeforeToolCallEvent)
def intercept_tool(event: BeforeToolCallEvent):
    """Fires before any tool execution"""
    if event.selected_tool == "dangerous_tool":
        event.cancel_tool = True  # Block execution
        print("Tool cancelled")

@agent.on(AfterToolCallEvent)
def inspect_result(event: AfterToolCallEvent):
    """Fires after tool completes"""
    print(f"Tool returned: {event.result}")
    event.result = f"Modified: {event.result}"  # Transform output

@agent.on(AfterModelCallEvent)
def inspect_model(event: AfterModelCallEvent):
    """Fires after LLM responds"""
    print(f"Model selected tools: {event.selected_tool}")
```

### Event Properties & Mutation

```python
# BeforeToolCallEvent properties (mutable)
event.retry = True                    # Retry after all hooks
event.cancel_tool = True              # Block execution
event.selected_tool = "new_tool"      # Replace tool choice

# AfterToolCallEvent properties (mutable)
event.result = new_result             # Transform tool output
event.resume = True                   # Resume reasoning with modified result

# BeforeModelCallEvent properties (read-only)
event.messages                        # Current conversation history

# AfterModelCallEvent properties (mutable)
event.selected_tool                   # Tool the model picked
event.resume = True                   # Resume agent loop
event.exception                       # Errors (read-only)
```

### Multi-Agent Events

```python
from strands.hooks import BeforeNodeCallEvent, MultiAgentHandoffEvent

@graph.on(BeforeNodeCallEvent)
def intercept_node(event: BeforeNodeCallEvent):
    """Intercept Graph node execution"""
    if event.node_id == "review":
        event.cancel_node = True  # Skip node

@swarm.on(MultiAgentHandoffEvent)
def track_handoff(event: MultiAgentHandoffEvent):
    """Track agent-to-agent transfers"""
    print(f"{event.from_agent} → {event.to_agent}")
    print(f"Context: {event.transfer_message}")
```

### Hook Registration & Patterns

```python
from strands.hooks import HookProvider

# Method 1: Decorator-based (clean)
@agent.on(BeforeToolCallEvent)
def my_hook(event):
    pass

# Method 2: Explicit registration
def my_hook(event: BeforeToolCallEvent):
    pass
agent.add_hook(BeforeToolCallEvent, my_hook)

# Method 3: Plugin-based (reusable)
class MyPlugin(HookProvider):
    def register_hooks(self, agent):
        agent.add_hook(BeforeToolCallEvent, self.intercept)
    
    def intercept(self, event):
        pass

agent.add_plugin(MyPlugin())

# Hook ordering: after-event callbacks execute in reverse registration order
```

---

## Session Management (Persistence)

Persist conversation history, agent state, and conversation manager state across sessions.

### File-Based Sessions

```python
from strands.session import FileSessionManager, FileStorage

storage = FileStorage(directory="/tmp/agent_sessions")
session_mgr = FileSessionManager(storage=storage)

agent = Agent(
    name="PersistentAgent",
    session_manager=session_mgr,
    session_id="user_123_session"
)

# Session auto-saved after: init, message add, invocation, message redact
response = agent.invoke("What's the weather?")

# Later: reload session
agent2 = Agent(
    name="PersistentAgent",
    session_manager=session_mgr,
    session_id="user_123_session"  # Loads previous state
)
response = agent2.invoke("Compare with yesterday")  # Agent remembers previous context
```

### S3-Based Sessions (Cloud Persistence)

```python
from strands.session import S3SessionManager, S3Storage
import boto3

s3_client = boto3.client("s3")
storage = S3Storage(
    bucket="my-agent-sessions",
    prefix="sessions/",
    s3_client=s3_client
)

session_mgr = S3SessionManager(storage=storage)
agent = Agent(
    name="CloudAgent",
    session_manager=session_mgr,
    session_id="user_456"
)

# Sessions persisted to S3 (requires s3:PutObject, s3:GetObject, s3:DeleteObject, s3:ListBucket)
```

### Amazon AgentCore Memory

Integrates with AWS Bedrock AgentCore for short and long-term memory:

```python
from strands.session import AgentCoreMemoryManager

session_mgr = AgentCoreMemoryManager(
    agent_id="arn:aws:bedrock:us-east-1:123456789:agent/ABC123",
    session_id="user_123"
)

agent = Agent(
    name="AgentCoreAgent",
    session_manager=session_mgr
)
# Automatically uses AgentCore's managed memory (short + long term with TTL)
```

### Custom Session Backend

```python
from strands.session import SessionRepository

class RedisSessionRepository(SessionRepository):
    def save_session(self, session_id: str, state: dict, ttl: int = 3600):
        """Save session state to Redis"""
        redis_client.setex(f"session:{session_id}", ttl, json.dumps(state))
    
    def load_session(self, session_id: str) -> dict:
        """Load session state from Redis"""
        return json.loads(redis_client.get(f"session:{session_id}"))
    
    def delete_session(self, session_id: str):
        """Delete session from Redis"""
        redis_client.delete(f"session:{session_id}")

session_mgr = SessionRepository(backend=RedisSessionRepository())
```

---

## Interrupts (Human-in-the-Loop)

Pause agent execution to request human approval before proceeding.

### Hook-Based Interrupts

```python
from strands.hooks import BeforeToolCallEvent
from strands.interrupt import InterruptRequest

@agent.on(BeforeToolCallEvent)
def require_approval(event: BeforeToolCallEvent):
    """Pause before potentially destructive tools"""
    if event.selected_tool in ["delete_database", "send_email", "transfer_funds"]:
        event.interrupt()  # Pause execution, return result.stop_reason = "interrupt"

result = agent.invoke("Delete all inactive users from the database")
# Returns: AgentResult with stop_reason="interrupt"
# result.interrupts = [InterruptRequest(tool="delete_database", ...)]

# Human reviews, then resume
if user_approved:
    result = agent.resume(interrupt_id=result.interrupts[0].id)
else:
    agent.cancel()
```

### Tool-Based Interrupts

```python
from strands import tool
from strands.tools import ToolContext

@tool
def send_email(recipient: str, subject: str, context: ToolContext = None):
    """
    Send email with human approval
    
    Args:
        recipient: Email address
        subject: Email subject
        context: Tool execution context (allows interrupts)
    """
    if context:
        context.interrupt()  # Pause before email sent
    
    send_via_smtp(recipient, subject)
    return f"Email sent to {recipient}"
```

### Session Persistence of Interrupts

```python
from strands.session import FileSessionManager

# Interrupt state persists in session
session_mgr = FileSessionManager(...)
agent = Agent(session_manager=session_mgr, session_id="user_123")

result = agent.invoke("Delete user account")
# Session now stores: stop_reason="interrupt", interrupt_id, tool_name, etc.

# Later session reconnects
agent2 = Agent(session_manager=session_mgr, session_id="user_123")
# Can inspect pending interrupts
pending = agent2.get_pending_interrupts()  # [InterruptRequest(...)]

# Resume or cancel
agent2.resume(pending[0].id)  # Continue with approval
```

---

## Plugins System

Plugins are modular behavioral modifications via composable hooks and tools.

### Plugins Overview

```python
from strands.plugins import Plugin

class MyPlugin(Plugin):
    name: str = "my_plugin"
    
    @hook(BeforeToolCallEvent)
    def intercept_tools(self, event):
        """Hook into tool execution"""
        pass
    
    @tool
    def custom_tool(self, input_data: str):
        """Define tools as part of plugin"""
        return f"Processed: {input_data}"
    
    async def init_agent(self, agent):
        """Called when plugin attaches to agent"""
        print(f"Plugin {self.name} loaded for {agent.name}")

# Use plugin
agent = Agent(
    name="PluginAgent",
    plugins=[MyPlugin()]
)
```

### Skills Plugin (On-Demand Instructions)

Skills are modular, on-demand instructions that load on-demand instead of bloating system prompts.

```python
# SKILL.md format
"""
---
name: "Financial Advisor"
description: "Expert advice on personal finance"
allowed-tools:
  - calculator
  - web_search
metadata:
  domain: finance
  version: "1.0"
---

## When to Use This Skill
You are a certified financial advisor...

## Key Concepts
1. Compound interest
2. Dollar-cost averaging
...
"""

from strands.plugins import SkillsPlugin

skills = SkillsPlugin(
    skills_dir="./skills/",  # Load from directory
    # Or: skills=[Skill(...), Skill(...)]
)

agent = Agent(
    name="FinancialAdvisor",
    plugins=[skills]
)

# Three-phase operation:
# 1. Discovery: Skill metadata in system prompt
# 2. Activation: Agent calls "skills" tool to load full instructions
# 3. Execution: Agent uses loaded skill context
```

### Steering Plugin (Context-Aware Guidance)

Steering solves "prompt bloat" with just-in-time contextual guidance.

```python
from strands.plugins import SteeringPlugin
from strands.plugins.steering import SteeringContextCallback, LedgerProvider

class FinanceSteeringCallback(SteeringContextCallback):
    async def get_steering_context(self, agent) -> dict:
        """Provide contextual guidance based on current state"""
        return {
            "ledger": LedgerProvider(agent).get_tool_history(),
            "compliance_rules": fetch_current_compliance_rules(),
            "user_risk_profile": fetch_user_risk_profile()
        }

steering = SteeringPlugin(
    context_callbacks=[FinanceSteeringCallback()]
)

agent = Agent(
    name="SteeringAgent",
    plugins=[steering]
)

# Two steering modes:
# 1. steer_before_tool(option="Guide"): Cancel + inject feedback
# 2. steer_after_model(option="Guide"): Discard response + retry with guidance

# Results: 100% pass rate on eval (vs 82.5% simple, 80.8% workflow)
```

---

## Structured Output

Type-safe, validated responses from LLMs using schema definitions.

```python
from pydantic import BaseModel
from strands.agents import Agent

class ResearchReport(BaseModel):
    """Schema for structured agent output"""
    title: str
    summary: str
    key_findings: list[str]
    recommendation: str
    confidence_score: float  # 0.0 - 1.0

agent = Agent(
    name="ResearchAgent",
    model="claude-3-5-sonnet-20241022"
)

# Get validated response matching schema
result = agent.structured_output(
    prompt="Research latest AI safety developments",
    model=ResearchReport
)

# Returns ResearchReport instance (automatically validated)
print(result.recommendation)  # Type-safe access

# Streaming compatible
for partial in agent.stream_structured_output(
    prompt="...",
    model=ResearchReport
):
    print(partial)  # Incremental updates
```

**Benefits:**
- Type safety (IDE autocomplete)
- Automatic validation (Pydantic raises `StructuredOutputException` on failure)
- Schema as documentation
- Agent-level defaults + per-invocation overrides

---

## Agent State Types (3 Distinct Layers)

Strands distinguishes between three types of state:

### 1. Conversation History (Passed to Model)

```python
# Auto-managed by agent
agent.messages  # Full conversation history

# Accessible to LLM
for message in agent.messages:
    print(f"Role: {message.role}, Content: {message.content}")

# Example:
# agent.messages = [
#     {"role": "user", "content": "Find AI papers"},
#     {"role": "assistant", "content": "I'll search..."},
#     {"role": "tool", "content": "Found 5 papers"},
#     ...
# ]
```

### 2. Agent App State (NOT Passed to Model)

```python
# Key-value store for agent internal state
agent.state.set("user_id", "user_123")
agent.state.set("session_start", datetime.now())

# Accessible in tools via ToolContext
@tool
def my_tool(context: ToolContext = None):
    user_id = context.state.get("user_id")  # "user_123"
    session_start = context.state.get("session_start")
    return f"User {user_id} session"

# Properties:
agent.state.get("key")          # Retrieve
agent.state.set("key", value)   # Set
agent.state.delete("key")       # Delete
agent.state                      # JSON-serializable dict
```

### 3. Request State (Single Event Loop)

```python
# Persists during one agent.invoke() call only
result = agent.invoke("...")

# Accessible in request_state parameter during execution
result.state  # {"temp_calculation": 42, ...}

# Example use: temporary values computed during reasoning
@agent.on(BeforeToolCallEvent)
def capture_state(event):
    event.request_state["tool_order"] = event.selected_tool
```

---

## Conversation Manager Variants

Control how conversation history is preserved and managed.

### SlidingWindowConversationManager (Default)

```python
from strands.agents import SlidingWindowConversationManager

mgr = SlidingWindowConversationManager(
    max_messages=20,          # Keep last 20 messages
    truncate_tool_results=True  # Shorten verbose outputs
)

agent = Agent(
    name="WindowAgent",
    conversation_manager=mgr
)

# Auto-removes oldest messages when limit exceeded
# Preserves complete tool-use sequences (doesn't split chains)
```

### NullConversationManager (No History)

```python
from strands.agents import NullConversationManager

mgr = NullConversationManager()

agent = Agent(
    name="StatelessAgent",
    conversation_manager=mgr
)

# No history modification
# Each invoke() starts fresh
# Use for: debugging, short one-shot queries, testing
```

### SummarizingConversationManager (Smart Summarization)

```python
from strands.agents import SummarizingConversationManager

mgr = SummarizingConversationManager(
    summary_ratio=0.3,          # Summarize oldest 30% of messages
    preserve_recent_messages=10,  # Always keep last 10
    summarization_model="claude-3-5-haiku-20241022",  # Cheap model for summaries
    summary_prompt="Summarize this conversation concisely..."
)

agent = Agent(
    name="SmartAgent",
    conversation_manager=mgr
)

# Automatically summarizes old messages to save tokens
# Recent context always preserved in full
# Use for: long multi-turn conversations
```

---

## Tool Executors

Control how multiple tool calls are executed (parallel vs sequential).

### ConcurrentToolExecutor (Default)

```python
from strands.tools import ConcurrentToolExecutor

executor = ConcurrentToolExecutor(
    max_parallel=5,  # Max concurrent tool executions
    timeout_per_tool=30  # Seconds
)

agent = Agent(
    name="ParallelAgent",
    tool_executor=executor
)

# When model returns multiple tool-use requests:
# 1. Execute all in parallel (faster)
# 2. Combine results
# 3. Resume reasoning

# Example: Model decides to call [calculator, web_search, database] at once
# → All execute simultaneously
```

### SequentialToolExecutor

```python
from strands.tools import SequentialToolExecutor

executor = SequentialToolExecutor()

agent = Agent(
    name="SequentialAgent",
    tool_executor=executor
)

# Tools execute in order returned by model
# Use for: dependent workflows where tool_B needs tool_A's result
# Trade-off: slower but deterministic ordering
```

---

## Evals SDK (Evaluation Framework)

Systematic evaluation of agent behavior using multiple evaluator types.

### Setup & Installation

```bash
pip install strands-agents[evals]
```

### OutputEvaluator (LLM-as-a-Judge)

```python
from strands.evals import OutputEvaluator

evaluator = OutputEvaluator(
    rubric="""
    1. Accuracy: Is the response factually correct?
    2. Completeness: Does it address all aspects of the question?
    3. Clarity: Is it clearly written?
    """,
    model="claude-3-5-sonnet-20241022"
)

test_cases = [
    {"input": "What's 2+2?", "expected_output": "4"},
    {"input": "Capital of France?", "expected_output": "Paris"},
]

scores = evaluator.evaluate(
    agent_responses=[(tc["input"], agent.invoke(tc["input"]).text) for tc in test_cases],
    rubric=evaluator.rubric
)

print(f"Average score: {sum(scores) / len(scores)}")
```

### TrajectoryEvaluator (Tool Selection Analysis)

```python
from strands.evals import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()

# Evaluate: Did agent pick the right tools in the right order?
trajectory_score = evaluator.evaluate(
    agent_result=result,  # AgentResult with full trace
    expected_tools=["web_search", "calculator"],  # Ideal sequence
    expected_order=True   # Must be in this order
)

print(f"Tool selection quality: {trajectory_score}")
```

### HelpfulnessEvaluator (OpenTelemetry Traces)

```python
from strands.evals import HelpfulnessEvaluator

evaluator = HelpfulnessEvaluator(
    trace_backend="opentelemetry"  # Requires OTEL-exported traces
)

# Evaluate based on execution traces (not just final output)
helpfulness_score = evaluator.evaluate(
    session_id="session_123",  # Look up traces in OTEL backend
    user_feedback="Was the response helpful?"
)
```

### Run Evaluations Asynchronously

```python
from strands.evals import run_evaluations_async

results = await run_evaluations_async(
    agent=agent,
    evaluators=[OutputEvaluator(...), TrajectoryEvaluator(...)],
    test_cases=test_cases,
    batch_size=10  # Concurrent evaluations
)

# Save for versioning
import json
with open("eval_results.json", "w") as f:
    json.dump(results, f)
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

## Workflow Pattern (5th Multi-Agent Pattern)

Structured multi-agent coordination for complex multi-step processes with dependency management.

```python
from strands.workflow import Workflow, Task, Dependency

workflow = Workflow(
    name="DocumentProcessing",
    description="Process documents with validation and review"
)

# Define tasks (each with its own agent)
validate_task = Task(
    id="validate",
    agent=ValidatorAgent(),
    description="Validate document format"
)

extract_task = Task(
    id="extract",
    agent=ExtractorAgent(),
    description="Extract key data",
    dependencies=[Dependency(task_id="validate")]  # Wait for validate
)

review_task = Task(
    id="review",
    agent=ReviewerAgent(),
    description="Review extracted data",
    dependencies=[Dependency(task_id="extract")]
)

publish_task = Task(
    id="publish",
    agent=PublisherAgent(),
    description="Publish final output",
    dependencies=[Dependency(task_id="review")]
)

workflow.add_tasks([validate_task, extract_task, review_task, publish_task])

# Execute with automatic dependency resolution
result = workflow.execute(
    input_data={"document": doc_content},
    retry_strategy="exponential",      # Auto-retry on failure
    rate_limit="1/sec",                # Throttle execution
    parallel_tasks=2                   # Parallel when possible
)

print(result.final_output)  # Output from last task
```

**Workflow Features:**
- Sequential + parallel mixed execution
- Built-in dependency resolution
- Retry mechanisms (exponential backoff)
- Rate limiting + task quotas
- Persistent task state for resumability
- Compare with Swarm (peer-based) and Graph (DAG-based)

---

## Additional Model Providers

### Amazon Nova

```python
from strands.models import NovaAPIModel

agent = Agent(
    name="NovaAgent",
    model=NovaAPIModel(
        api_key="nova_api_key",
        model_id="nova-2-lite",        # nova-2-lite, nova-2-pro, nova-3, etc.
        max_completion_tokens=4096,
        temperature=0.7,
        reasoning_effort="medium",      # low, medium, high (extended thinking)
        system_tools=["grounding", "code_interpreter"]  # Bedrock system tools
    )
)
```

### LiteLLM (Unified Provider Proxy)

```python
from strands.models import LiteLLMModel

# Unified interface for 100+ models
agent = Agent(
    name="LiteLLMAgent",
    model=LiteLLMModel(
        model_id="gpt-4-turbo",  # Or: "claude-3-sonnet", "gemini-pro", etc.
        api_key="litellm_key",
        cache_config={
            "cache_provider": "redis",  # Provider-agnostic caching
            "ttl": 3600
        }
    )
)
```

### Mistral AI

```python
from strands.models import MistralModel

agent = Agent(
    name="MistralAgent",
    model=MistralModel(
        api_key="mistral_api_key",
        model_id="mistral-large",      # mistral-large, mistral-medium, open-mistral-7b
        max_tokens=8000,
        temperature=0.7,
        top_p=0.95,
        stream=True
    )
)
# Specializes in: multilingual, code generation, math
```

### Amazon SageMaker

```python
from strands.models import SageMakerModel

agent = Agent(
    name="SageMakerAgent",
    model=SageMakerModel(
        endpoint_name="my-sagemaker-endpoint",
        region_name="us-east-1",
        inference_component_name="component-1",  # Optional
        target_model="mistral-small-24b",        # Model on endpoint
        max_tokens=2048,
        temperature=0.7
    )
)
# Requires OpenAI-compatible chat completion API on endpoint
```

### Writer (Palmyra)

```python
from strands.models import WriterModel

agent = Agent(
    name="WriterAgent",
    model=WriterModel(
        api_key="writer_api_key",
        model_id="palmyra-x5",          # X5 (1M tokens), X4, Fin, Med, Creative
        max_tokens=4096,
        temperature=1.0,  # 0-2 range
        stream_options={"include_usage": True}
    )
)
# Palmyra X5: 1M token context, vision, financial/medical specialized
```

### llama.cpp (Local GGUF)

```python
from strands.models import LlamaCppModel

agent = Agent(
    name="LocalAgent",
    model=LlamaCppModel(
        base_url="http://localhost:8000",  # llama-server must be running
        model_id="llama-2-7b",
        params={
            "repeat_penalty": 1.1,
            "top_k": 40,
            "min_p": 0.05,
            "mirostat": 2,
            "json_schema": {  # Constrained output via GBNF grammar
                "type": "object",
                "properties": {...}
            }
        }
    )
)
# Local execution, no external API, structured output via grammar
```

### LlamaAPI

```python
from strands.models import LlamaAPIModel

agent = Agent(
    name="LlamaAPIAgent",
    model=LlamaAPIModel(
        api_key="llamaapi_key",
        model_id="llama-2-70b-chat",    # Meta's hosted Llama inference
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        max_completion_tokens=2048
    )
)
```

---

## Bedrock Advanced Features

### Prompt Caching (Token Savings)

```python
from strands.models import BedrockModel
from strands.bedrock import CacheConfig

agent = Agent(
    name="CachedAgent",
    model=BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        cache_config=CacheConfig(
            cache_system_prompt=True,    # Cache static system prompt
            cache_tools=True,             # Cache tool definitions
            cache_messages=True           # Cache conversation history
        )
    )
)

result = agent.invoke("Analyze this document")
print(result.metrics)  # Shows:
# {
#   "cacheWriteInputTokens": 500,   # Tokens written to cache
#   "cacheReadInputTokens": 1000,   # Tokens read from cache (no charge)
#   "inputTokens": 100              # New input tokens
# }
```

### Extended Thinking / Reasoning (Multi-Step)

```python
agent = Agent(
    name="ThinkingAgent",
    model=BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        additional_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 5000  # Min 1,024 tokens for reasoning
            }
        }
    )
)

result = agent.invoke("Solve this complex problem: ...")
# Model internally reasons with extended tokens before responding
```

### Multimodal Support (Documents, Images, Videos)

```python
from strands.models import BedrockModel

agent = Agent(
    name="MultimodalAgent",
    model=BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
)

# Multimodal message
result = agent.invoke({
    "text": "Analyze this document and image",
    "documents": [
        {
            "format": "pdf",
            "source": {
                "bytes": pdf_bytes  # Raw bytes
                # OR: "s3Location": {"bucket": "...", "key": "..."}
            }
        }
    ],
    "images": [
        {
            "format": "png",
            "source": {
                "bytes": image_bytes
                # OR: "s3Location": {"bucket": "...", "key": "..."}
            }
        }
    ],
    "videos": [
        {
            "format": "mp4",
            "source": {
                "s3Location": {"bucket": "...", "key": "..."}
            }
        }
    ]
})
```

### Guardrails Integration (Content Safety)

```python
from strands.models import BedrockModel

agent = Agent(
    name="SafeAgent",
    model=BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        guardrail_id="arn:aws:bedrock:us-east-1:...:guardrail/...",
        guardrail_redact_input=True,      # Redact harmful input before model
        guardrail_redact_output=True,     # Redact harmful output before user
        guardrail_latest_message=False    # Evaluate full conversation
    )
)

# Shadow-mode monitoring via hooks
from strands.hooks import AfterModelCallEvent

@agent.on(AfterModelCallEvent)
def audit_guardrails(event):
    # Access guardrail evaluation results
    print(f"Guardrail action: {event.guardrail_action}")
    print(f"Redacted: {event.guardrail_redacted}")
```

---

## AgentResult Structure (Response Details)

Complete breakdown of what an agent invocation returns:

```python
result = agent.invoke("What's the capital of France?")

# result is AgentResult with:
result.text                           # Final response text
result.stop_reason                    # "end_turn" | "tool_use" | "max_tokens" | "interrupt" | "guardrail_intervention"
result.messages                       # Full conversation history
result.interrupts                     # List of InterruptRequest (if interrupted)
result.state                          # Request state (request-scoped data)

# Metrics (detailed execution stats)
result.metrics.latency_ms             # Total execution time
result.metrics.input_tokens           # Prompt tokens
result.metrics.output_tokens          # Completion tokens
result.metrics.cache_write_tokens     # Tokens written to cache
result.metrics.cache_read_tokens      # Tokens read from cache (free)
result.metrics.get_summary()          # Aggregated summary

# Tool execution stats
result.metrics.tool_calls             # [{"name": "web_search", "duration_ms": 250}, ...]
result.metrics.total_tool_calls       # Count of tool executions

# Traces (OpenTelemetry)
result.traces                         # Execution traces (if OTEL configured)

# Example inspection
print(f"Response: {result.text}")
print(f"Tokens: {result.metrics.input_tokens} → {result.metrics.output_tokens}")
print(f"Tools called: {len(result.metrics.tool_calls)}")
print(f"Cache hit: {result.metrics.cache_read_tokens} tokens free")
if result.stop_reason == "interrupt":
    print(f"Awaiting approval for: {result.interrupts[0].tool}")
```

---

## MCP Tools (Model Context Protocol)

Advanced integration with the open MCP standard for tool discovery and execution.

### Transport Options

```python
from strands.mcp import MCPClient, StdioTransport, SSETransport, HTTPTransport

# 1. Standard I/O (stdio) - for local processes
mcp_client = MCPClient(
    transport=StdioTransport(
        command="python",
        args=["-m", "mcp_server"]
    )
)

# 2. Server-Sent Events (SSE) - for HTTP servers
mcp_client = MCPClient(
    transport=SSETransport(
        url="http://mcp-server:8000/sse"
    )
)

# 3. Streamable HTTP with auth
mcp_client = MCPClient(
    transport=HTTPTransport(
        url="http://mcp-server:8000",
        headers={"Authorization": "Bearer token"}
    )
)
```

### Tool Filtering & Prefixing

```python
agent = Agent(
    name="FilteredAgent",
    tools=mcp_client,
    tool_config={
        "mcp_tool_filter": {
            "allowed_patterns": ["read_*", "calculate_*"],  # Include
            "rejected_patterns": ["delete_*", "drop_*"],     # Exclude
            "name_prefix": "mcp_"  # Rename: read_file → mcp_read_file
        }
    }
)
```

### Custom MCP Server (FastMCP)

```python
from fastmcp import FastMCP

mcp = FastMCP("my-mcp-server")

@mcp.tool()
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text"""
    return {
        "sentiment": "positive",
        "score": 0.92,
        "details": "Highly positive language"
    }

@mcp.resource()
def knowledge_base(resource_id: str) -> str:
    """Access knowledge base articles"""
    return fetch_from_kb(resource_id)

# Serve MCP server
mcp.run_server()  # Listens for stdio or HTTP requests
```

---

## Community Tools Package (Extended)

Additional tools beyond the 20+ built-ins:

### Computer Use & Automation
- `use_computer` — Control mouse, keyboard, take screenshots
- `cron` — Schedule background tasks

### Multimedia & Generation
- `generate_image` — DALL-E integration
- `generate_image_stability` — Stability AI API
- `nova_reels` — Video generation (AWS Nova)
- `speak` — Text-to-speech
- `diagram` — Create diagrams (Mermaid, PlantUML)

### Workflow & Orchestration
- `journal` — Maintain execution journal
- `workflow` — Workflow pattern tool (task dependency resolution)
- `batch` — Batch operation processing
- `think` — Explicit reasoning step (like extended thinking)

### Alternative Memory Backends
- `agent_core_memory` — AWS AgentCore memory integration
- `mem0_memory` — Mem0 external memory (hybrid storage)

### Handoff Patterns
- `handoff_to_user` — Transfer control to human (interactive or complete handoff)
- `a2a_client` — Call remote agents via A2A protocol

### Code & Data
- `python_repl` — Execute Python code
- `code_interpreter` — Run code in sandbox
- `shell` — Execute shell commands
- `editor` — Edit files interactively
- `file_read`, `file_write` — File operations

### External Services
- `http_request` — HTTP calls (all methods)
- `slack` — Slack integration
- `browser` — Web browsing
- `rss` — RSS feed consumption
- `email` — Email sending/receiving

### Tool Consent & Approval

```python
# Require user approval for sensitive tools
import os
os.environ["BYPASS_TOOL_CONSENT"] = "false"  # Default: require consent

# Or programmatically
agent = Agent(
    name="ConsentAgent",
    tool_config={"require_consent_for": ["delete_*", "send_email"]}
)

# User sees: "Tool 'send_email' requires approval. Approve? [y/n]"
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
