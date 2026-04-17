# AI Agent Frameworks Comparison — Interview Deep Dive

**Context:** Research into "EasyChat AI" revealed it exists as a Windows desktop app, iOS app, and PyPI stub, but **not as a developer framework**. This guide instead provides comprehensive comparison of production-ready agent frameworks for technical interviews.

---

## The Three Production Frameworks You Need to Know

### 1. AWS Bedrock Agents (Managed Service)

**What it is:** AWS-hosted agent runtime with integrated tools, memory, guardrails, and observability.

**Best for:** Financial services, compliance-critical systems, AWS-native infrastructure.

**Key features:**
```python
# AWS Bedrock = Infrastructure Layer
# Runs agents in managed microVMs
# Built-in memory (short + long-term)
# Built-in guardrails (Cedar policy engine)
# Built-in observability (CloudWatch + X-Ray)

agent = bedrock.create_agent(
    name="FinancialAgent",
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="You analyze financial data securely",
    tools=[...]  # AWS-native tool definitions
)

# One command = fully managed production agent
response = bedrock.invoke_agent(agent_id="...", input="...")
```

**Pros:**
- Zero infrastructure management
- Compliance-grade security (data never leaves AWS)
- Session management (8-hour max, 15-min idle timeout)
- Cost transparency ($0.0895/vCPU-hour)

**Cons:**
- AWS lock-in
- Limited to Claude, GPT-4, Llama (no custom models)
- Fixed resource allocation (2 vCPU/8GB)
- Slower cold starts (~2-3 seconds)

**Interview angle:** "How would you deploy an agentic system in a financial services company where data residency and compliance are non-negotiable?"

---

### 2. Strands Agents (Open-Source SDK)

**What it is:** Python framework where LLM dynamically orchestrates other agents and tools.

**Best for:** Multi-agent systems, rapid prototyping, model flexibility, autonomous behavior.

**Key features:**
```python
# Strands = Programming Model
# Developer defines agents
# LLM orchestrates interactions
# Auto-managed memory + tools

from strands.agents import Agent

research_agent = Agent(
    name="Researcher",
    model="claude-3-5-sonnet-20241022",
    tools=[WebSearch(), DocumentSearch()],
)

analysis_agent = Agent(
    name="Analyst",
    model="gpt-4",  # Different model!
    tools=[Calculator(), DatabaseQuery()],
)

coordinator = Agent(
    name="Coordinator",
    tools=[research_agent, analysis_agent],  # Agents as tools
    system_prompt="Orchestrate research and analysis to answer the question"
)

result = coordinator.run("Analyze market trends for Q3")
# LLM decides: first call researcher, then analyst, then synthesize
```

**Pros:**
- Any LLM (Claude, GPT-4, Llama, custom)
- Dynamic orchestration (LLM decides flow)
- Rich tool ecosystem (20+ built-in)
- Streaming support (real-time responses)

**Cons:**
- Developer manages deployment
- Manual memory persistence
- Less observability than Bedrock
- LLM cost not optimized (each agent call = fresh model call)

**Interview angle:** "Walk me through how you'd use an LLM to orchestrate multiple specialized agents, each with different models and tools, to solve a complex problem."

---

### 3. LangGraph (Developer-Driven Workflows)

**What it is:** Framework for building explicit agent workflows as deterministic graphs.

**Best for:** Complex business logic, audit trails, reproducible behavior, traditional software engineering patterns.

**Key features:**
```python
# LangGraph = Explicit State Machine
# Developer codes every step
# Graph structure guarantees flow
# Full control over decisions

from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query: str
    research_findings: list
    analysis: str
    final_report: str

graph = StateGraph(AgentState)

# Define nodes (steps)
graph.add_node("research", research_step)
graph.add_node("analyze", analyze_step)
graph.add_node("review", review_step)
graph.add_node("publish", publish_step)

# Define edges (deterministic routing)
graph.add_edge("research", "analyze")
graph.add_edge("analyze", "review")
graph.add_conditional_edge(
    "review",
    lambda state: "publish" if state["approved"] else "research",
    {"publish": "publish", "research": "research"}
)
graph.add_edge("publish", END)

# Compile and run
compiled_graph = graph.compile()
result = compiled_graph.invoke({"query": "..."})
```

**Pros:**
- Explicit control (no LLM surprises)
- Full audit trail
- Testable (each step isolated)
- Cost-optimized (developer batches operations)
- Cycle detection

**Cons:**
- More code (every step explicit)
- Steep learning curve
- Less flexible (fixed DAG)
- Requires thinking about state management

**Interview angle:** "Design an agentic system for document processing where every decision must be auditable and reproducible for regulatory compliance."

---

## Side-by-Side Comparison

```
┌─────────────────────┬────────────────────┬──────────────────┬────────────────────┐
│ Dimension           │ Bedrock Agents     │ Strands Agents   │ LangGraph          │
├─────────────────────┼────────────────────┼──────────────────┼────────────────────┤
│ Deployment Model    │ Managed (AWS)      │ DIY (Docker/K8s) │ DIY (single app)   │
│ Programming Model   │ Declarative (YAML) │ Model-Driven     │ Developer-Driven   │
│ Flow Control        │ Fixed workflow     │ LLM-orchestrated │ Explicit DAG       │
│ Observability       │ CloudWatch native  │ OpenTelemetry    │ LangSmith (paid)   │
│ Memory Management   │ Auto (sessions)    │ Auto (per-agent) │ Manual (state dict)│
│ Model Flexibility   │ AWS models only    │ Any LLM          │ Any LLM            │
│ Tool Definition     │ AWS console        │ Python docstring │ Manual JSON schema │
│ Multi-Agent Pattern │ Agents in workflow │ Agents as tools  │ Explicit nodes     │
│ Cost Model          │ Per vCPU-hour      │ Per token (model)│ Per token (model)  │
│ Learning Curve      │ Moderate (AWS)     │ Moderate         │ Steep              │
│ Production Ready    │ Yes (AWS-native)   │ Yes (with work)  │ Yes (popular)      │
│ Cold Start          │ 2-3 seconds        │ <100ms           │ <10ms              │
│ Scaling             │ Horizontal (AWS)   │ K8s auto-scale   │ App-level scaling  │
└─────────────────────┴────────────────────┴──────────────────┴────────────────────┘
```

---

## When to Use Each (Decision Tree)

```python
def choose_framework(requirements):
    """Choose the right agent framework based on requirements"""

    # Rule 1: Compliance + AWS-native?
    if "data_residency" in requirements and "aws" in requirements:
        return "Bedrock Agents"  # Non-negotiable: use managed service

    # Rule 2: Multiple models needed (Claude + GPT + Llama)?
    if "model_diversity" in requirements:
        return "Strands Agents"  # Only Strands supports multi-model out-of-box

    # Rule 3: Audit trail + deterministic flow critical?
    if "reproducibility" in requirements and "audit_trail" in requirements:
        return "LangGraph"  # Must be explicit DAG for compliance

    # Rule 4: Quick prototype?
    if "prototype" in requirements and "rapid" in requirements:
        return "Strands Agents"  # Docstring → tool spec is fastest

    # Rule 5: Complex business logic + control?
    if "control" in requirements and "complexity" in requirements:
        return "LangGraph"  # Explicit is better than implicit

    # Default: use Strands for flexibility
    return "Strands Agents"

# Examples:

# Financial services: audit logging, PII redaction, data residency
choose_framework({
    "data_residency": True,
    "audit_trail": True,
    "pii_handling": True,
    "aws": True
})
# → "Bedrock Agents"

# Multi-LLM research system: use best model for each task
choose_framework({
    "model_diversity": True,
    "rapid": True,
    "openai": True,
    "anthropic": True,
    "opensource": True
})
# → "Strands Agents"

# Loan underwriting workflow: repeatable, auditable, testable
choose_framework({
    "compliance": True,
    "reproducibility": True,
    "audit_trail": True,
    "testability": True
})
# → "LangGraph"
```

---

## Architecture Comparison (High-Level)

### Bedrock Agents (Managed)

```
User Request
    ↓
AWS Bedrock API Gateway
    ↓
Session Manager (8-hour max, 15-min idle timeout)
    ↓
Agent Runtime (microVM: 2 vCPU, 8GB RAM)
    ├─ LLM Call (Claude/GPT/Llama)
    ├─ Memory Retrieval (short + long-term)
    ├─ Tool Invocation (MCP protocol)
    ├─ Policy Enforcement (Cedar)
    └─ Observability (OpenTelemetry → CloudWatch)
    ↓
Response

Key: Fully managed — you worry about business logic only
```

### Strands Agents (Open-Source)

```
User Request
    ↓
Strands Agent (Python process)
    ├─ Short-term Memory (conversation history)
    ├─ LLM Call (any model via unified interface)
    ├─ Tool Execution (docstring → spec)
    ├─ Agent-to-Agent Communication (hierarchical/swarm)
    └─ Long-term Memory (optional, Redis/DynamoDB)
    ↓
Streaming Response (SSE/WebSocket)

Key: Flexible but you manage infra (Docker/K8s)
```

### LangGraph (Developer-Driven)

```
User Request
    ↓
LangGraph Compiled Graph
    ├─ Node 1: Research Step
    │   └─ LLM Call + Tool Use
    ├─ Node 2: Analyze Step (conditional routing)
    │   └─ If approved → Node 3, else → Node 1
    ├─ Node 3: Review Step
    │   └─ Human approval?
    └─ Node 4: Publish Step
    ↓
Deterministic Response

Key: Explicit control — every path coded, every decision explicit
```

---

## Interview Preparation: What to Study

### For "Tell me about agentic AI" questions:

**Bedrock Agents:**
- Session model (8-hour max, idle timeout)
- Memory strategies (SEMANTIC, SUMMARIZATION, USER_PREFERENCE)
- Cedar policy engine for guardrails
- MCP protocol for tool integration
- Data residency guarantees

**Strands Agents:**
- Model-driven orchestration (LLM decides flow)
- Docstring → tool spec (no manual JSON)
- Multi-agent patterns (hierarchical, swarm, graph)
- Memory auto-management
- Tool ecosystem (20+ built-in)

**LangGraph:**
- Graph-based state machine
- Explicit conditional routing
- Cycle detection
- Testability (each node isolated)
- Cost optimization (developer batches)

### Example Interview Questions:

**Q: "Design an AI system for financial document processing. What framework would you choose?"**

A: "I'd use Bedrock Agents because:
- Data residency is critical (never leaves AWS)
- Compliance requires audit trails (CloudWatch native)
- PII handling is built-in (guardrails)
- Cost is predictable (per vCPU-hour)
- Scale is guaranteed (AWS infrastructure)

Session model ensures 8-hour max runtime, perfect for batch processing overnight."

**Q: "How would you build a research system that uses Claude for reasoning and GPT-4 for coding?"**

A: "Strands Agents is ideal:
- Unified interface for multiple models
- Agents as tools (researcher agent → calls coder agent)
- LLM orchestrates: 'Try reasoning with Claude first, if complex coding needed, delegate to GPT-4'
- Memory auto-managed (no manual history tracking)
- Streaming support for real-time feedback"

**Q: "Walk me through an agentic workflow that must be reproducible and testable."**

A: "LangGraph:
```python
graph.add_node('validate', validation_step)
graph.add_node('process', processing_step)
graph.add_node('review', review_step)

graph.add_edge('validate', 'process')
graph.add_conditional_edge('process', check_if_approved, ...)

# Each step is a pure function → testable
# Every execution follows same path → reproducible
# Full audit trail → compliant
```"

---

## Note on "EasyChat AI"

Research into "EasyChat AI" revealed:
- **Windows Desktop App:** Consumer chat interface
- **iOS App:** Mobile chat client
- **PyPI Package:** Empty stub (no actual package)
- **FastAPI Server:** Open-source RWKV inference backend

**None are developer frameworks for building agentic systems.**

**Likely intended:** You may have meant:
- **OpenAI Swarm** (similar to Strands, models agents dynamically)
- **AutoGen (Microsoft)** (multi-agent conversation)
- **Crew AI** (task-based agent orchestration)

For your interview prep, focus on **Bedrock + Strands + LangGraph** — these three cover all production patterns you'll encounter in technical interviews.

---

## Quick Study Guide

**Before your interview, know this cold:**

### Bedrock Agents
- [ ] Session model (8 hours, 15 min idle, 1 GB storage)
- [ ] Memory types (short-term + long-term with strategies)
- [ ] Cedar policy engine (guardrails)
- [ ] MCP protocol (tool gateway)
- [ ] Pricing ($0.0895/vCPU-hour + $0.00945/GB-hour)
- [ ] When to use (compliance, AWS-native, data residency)

### Strands Agents
- [ ] Model-driven vs developer-driven
- [ ] Docstring → tool spec (no manual JSON)
- [ ] Multi-agent patterns (4 types: hierarchical, swarm, graph, A2A)
- [ ] Memory auto-management
- [ ] Built-in tools (20+ categories)
- [ ] Streaming (SSE, WebSocket)
- [ ] When to use (rapid prototyping, multi-model, autonomous)

### LangGraph
- [ ] State graph (nodes + edges)
- [ ] Conditional routing
- [ ] Cycle detection
- [ ] Testing patterns (unit test each node)
- [ ] Cost optimization (batch operations)
- [ ] When to use (audit trails, reproducibility, control)

---

## Production Patterns (Real-World Examples)

### Pattern 1: Financial Document Processing (Bedrock)

```python
# Use case: Bank must process loan applications with audit trail

agent = bedrock.create_agent(
    name="LoanUnderwriting",
    instructions="...",
    tools=[
        DatabaseQuery(table="applicants"),
        DocumentSearch(index="regulations"),
        EmailClient(),  # Notify applicant
    ],
    guardrails={
        "policy": cedar_redaction_policy,  # Redact PII
        "max_session_duration": 3600,
    }
)

# Every action logged to CloudWatch → Regulators can audit
# Data never leaves AWS → Compliance guaranteed
# Memory auto-managed → No manual state tracking
```

**Why Bedrock:** Compliance + data residency + audit logging.

### Pattern 2: Multi-Model Research System (Strands)

```python
# Use case: Research team needs different models for different tasks

research = Agent(name="Researcher", model="claude-3-5-sonnet", tools=[WebSearch()])
analysis = Agent(name="Analyst", model="gpt-4", tools=[Calculator()])
reporting = Agent(name="Reporter", model="claude-3-5-haiku", tools=[FileSystem()])

coordinator = Agent(
    name="Coordinator",
    model="claude-3-5-sonnet",
    tools=[research, analysis, reporting],  # Agents as tools
    system_prompt="Use best model for each subtask"
)

# LLM orchestrates: research → analyze → report
# Cost-effective: Haiku for cheap reporting
# Optimal quality: GPT-4 for complex analysis
```

**Why Strands:** Multi-model support + LLM orchestration + cost optimization.

### Pattern 3: Loan Underwriting Workflow (LangGraph)

```python
# Use case: Must be reproducible for regulatory audits

graph = StateGraph(UnderwritingState)

graph.add_node("extract", extract_application_info)
graph.add_node("validate", validate_applicant_data)
graph.add_node("score", calculate_credit_score)
graph.add_node("review", manual_review)
graph.add_node("approve", approve_or_deny)

graph.add_edge("extract", "validate")
graph.add_edge("validate", "score")
graph.add_conditional_edge("score", lambda s: "review" if s["risk_score"] > 50 else "approve")
graph.add_edge("review", "approve")

# Every execution follows same path
# Each step is a function → testable
# No LLM surprises → predictable outcomes
# Full audit trail → compliant
```

**Why LangGraph:** Deterministic flow + audit trail + testability.

---

## Closing: For Your Interview

The interviewer will likely ask variants of:

1. **"Tell me about your experience with AI agents"** → Discuss Bedrock/Strands/LangGraph with specific use cases
2. **"How would you architect an agentic system for [X]?"** → Use decision tree above
3. **"What are the tradeoffs?"** → Reference comparison table
4. **"Walk me through a production deployment"** → Pick one framework and explain session management, memory, tools, observability

**Pro tip:** Be ready to draw the architecture diagram on a whiteboard. Be able to explain why you chose one framework over another for a specific problem.

