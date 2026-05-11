# LangChain & LangGraph Production: API Versioning, Stability, Design Patterns

## LangChain Package Architecture Post-Split

### Pre-Split Era (v0.0.x)
Single monolithic `langchain` package. Everything in one import.
```python
from langchain import LLMChain, PromptTemplate, OpenAI
```

### Post-Split Architecture (v0.2+)

**Core packages (stable):**
- `langchain-core`: base abstractions (LLM, Chain, Tool, Message, OutputParser) — **stable, rarely breaking**
- `langchain`: orchestration layer (agents, chains, memory) — **moderate change**

**Integration packages (volatile):**
- `langchain-community`: community integrations (all vector DBs, APIs) — **high breaking change risk**
- `langchain-openai`, `langchain-anthropic`: provider-specific — **stable within package, may change signatures**
- `langchain-text-splitters`, `langchain-retrieval`: specialized tools — **low change**

**LangGraph:**
- `langgraph`: separate package, state machine framework — **relatively stable, v0.1+**

### Why This Matters for Production

```python
# ✅ PORTABLE (use in library code)
from langchain_core.llms import BaseLanguageModel
from langchain_core.tools import StructuredTool

# ❌ RISKY (use only in application code, pin exact version)
from langchain_community.vectorstores import Pinecone
from langchain.agents import AgentExecutor
```

**Rule:** If you're building a reusable package (library, SDK), depend on `langchain-core` only. If you're building application code, you can depend on `langchain` + community integrations, but pin exact versions.

---

## API Versioning Cadence and Breaking Changes

### Release Frequency
- `langchain-core`: stable, updates every 2-4 weeks (minor versions)
- `langchain`: fast-moving, updates weekly (minor versions)
- `langchain-community`: **unpredictable, breaking changes common**

### V0.1 → V0.2 → V0.3 Key Breaks

**V0.1 → V0.2: Agent Deprecation**
```python
# V0.1 (deprecated)
from langchain.agents import AgentExecutor, initialize_agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# ❌ This code will error in V0.2+

# V0.2+ (correct)
from langgraph.agents import create_react_agent
agent = create_react_agent(llm, tools)  # LangGraph, different API
# or use the compatibility wrapper:
from langchain.agents import AgentExecutor
agent = AgentExecutor.from_agent_and_tools(...)  # More verbose
```

**V0.1 → V0.2: ConversationChain Deprecation**
```python
# V0.1 (deprecated)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
# ❌ No longer idiomatic in V0.2+

# V0.2+ (correct)
# Explicit memory management, no ConversationChain
messages = []
response = await llm.ainvoke(messages)
messages.append({"role": "user", "content": input})
messages.append({"role": "assistant", "content": response})
```

**V0.2 → V0.3: Message Format Changes**
```python
# V0.2
from langchain.schema import HumanMessage, AIMessage
messages = [HumanMessage(content="..."), AIMessage(content="...")]

# V0.3
from langchain_core.messages import HumanMessage, AIMessage
# Same API, different import (moved to core)
```

**Tool Calling API Changes (V0.1 → V0.2)**
```python
# V0.1 (deprecated, tool invocation inside Agent)
agent = initialize_agent(tools, llm, agent="openai-functions", verbose=True)

# V0.2+ (explicit tool binding on LLM)
llm_with_tools = llm.bind_tools(tools)
# Now LLM handles tool calling, you control the loop
```

### Pattern: Tool Invocation Evolution
- **V0.0.x:** Agent abstracts tool loop (black box)
- **V0.1:** Still black box, but `AgentExecutor` exposed
- **V0.2:** Tool binding explicit, you control the loop (more portable)
- **V0.3+:** LangGraph is the recommended pattern (state machine replaces agent executor)

---

## Version Pinning Strategy for Production

### DO NOT:
```python
# ❌ This
langchain>=0.1
langchain-community>=0.0.1
```
These will auto-upgrade, breaking your code.

### DO:
```python
# ✅ This
langchain==0.2.15
langchain-core==0.2.5
langchain-community==0.2.4
langchain-openai==0.1.8
langgraph==0.1.7
```

### Separate Core from Community:
```toml
# pyproject.toml
dependencies = [
    # Core (stable, can update monthly)
    "langchain-core==0.2.5",
    "langgraph==0.1.7",
    
    # Community (volatile, pin exact)
    "langchain==0.2.15",
    "langchain-community==0.2.4",
    "langchain-openai==0.1.8",
]
```

### Version Upgrade Process:
1. Test in isolated environment (virtual env)
2. Run full test suite against new versions
3. Check migration guide: `langchain` → `langchain/BREAKING_CHANGES.md` in repo
4. Update one version at a time (core first, then community)
5. Deploy to staging, monitor for 24h before prod

---

## LangGraph API Stability

### StateGraph API (Stable)
```python
from langgraph.graph import StateGraph

class State(TypedDict):
    query: str
    history: list[str]

graph = StateGraph(State)
graph.add_node("think", think_node)
graph.add_edge("start", "think")
# ^^^ This API is stable (v0.1+)
```

### Recent Changes (V0.1 → V0.2)

**Added: `Interrupt` for human-in-the-loop (v0.2)**
```python
# V0.2+ only
from langgraph.types import Interrupt

def review_node(state):
    decision = state["decision"]
    # Pause, ask human
    raise Interrupt(f"Review: {decision}")
    # Human resumes with input
```

**Changed: `create_react_agent` parameters (v0.1 → v0.2)**
```python
# V0.1
from langgraph.agents import create_react_agent
agent = create_react_agent(llm, tools, debug=True)

# V0.2 (parameter renamed)
agent = create_react_agent(llm, tools)  # no debug param
# Use LangSmith or explicit logging instead
```

**Conclusion:** LangGraph API is more stable than `langchain` core, but still expect minor parameter changes.

---

## Impact on Existing Systems

### Silent Behavior Changes
**Tool routing behavior changed (V0.1 → V0.2):**
```python
# V0.1: Tool name "search" matches "search_web", "search_docs" (fuzzy matching)
tools = [search_web_tool, search_docs_tool]
# If user says "search", which runs?
# ❌ Ambiguous, undefined behavior

# V0.2: Tool names must be exact, bindings explicit
llm = llm.bind_tools([search_web_tool, search_docs_tool])
# If user says "search", model must pick exactly one
# ✅ Clear contract
```
**Impact:** Agents that worked in V0.1 may call different tools in V0.2.

### Checkpointer Schema Migrations
```python
# If you upgrade LangGraph with existing Postgres checkpointer:
# Old schema: thread_id (text), checkpoint_ns (text), checkpoint_id (uuid)
# New schema: ... + new columns ...
# ❌ May cause runtime errors until migration runs

# Solution: Before upgrading, back up DB. After upgrade, run:
db.run_migration()  # Explicit migration script
```

### Serialization Breaking
```python
# V0.1 Message serialization
message = HumanMessage(content="...", metadata={"source": "api"})

# V0.2 serialization changed slightly (fields reordered)
# ❌ If you pickled V0.1 messages, unpickling in V0.2 may fail

# Solution: Don't pickle. Use JSON (to_json, from_json)
message_json = message.to_json()  # Safe across versions
```

---

## Production Design Patterns

### 1. Use `langchain-core` Abstractions at Boundaries

```python
# Wrapper interface (library code — use only langchain-core)
from langchain_core.llms import BaseLanguageModel
from langchain_core.tools import StructuredTool

class MyAgent:
    def __init__(self, llm: BaseLanguageModel, tools: list[StructuredTool]):
        self.llm = llm
        self.tools = tools
    
    async def run(self, query: str) -> str:
        # Implementation can use langchain + community internally
        # But interface is portable (langchain-core)
        ...
```

**Benefit:** This wrapper can be used with any LLM provider (OpenAI, Claude, Bedrock) without code changes.

### 2. Wrap LLM Calls with Retry + Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def llm_call_safe(llm, prompt):
    response = await llm.ainvoke(prompt)
    return response

# Or using LangChain's built-in:
from langchain_core.runnables import Runnable

llm_with_retry = llm.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1
)
```

**Why:** Rate limits, transient errors, API outages are common. Retry with backoff is essential.

### 3. Structured Output with Pydantic on Every Tool Input/Output

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, description="Max results")

class SearchOutput(BaseModel):
    results: list[str]
    total_count: int

@tool("search", args_schema=SearchInput)
async def search(query: str, limit: int = 5) -> SearchOutput:
    # Pydantic validates input before this runs
    results = await db.search(query, limit)
    return SearchOutput(results=results, total_count=len(results))
```

**Why:** Prevents invalid inputs from reaching the tool. Pydantic validation happens before tool execution, saving API calls.

### 4. Separate Concerns: Retrieval / Reasoning / Action Nodes

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    retrieved_docs: list[str]
    reasoning: str
    action_plan: str

graph = StateGraph(State)

def retrieval_node(state):
    docs = vector_db.search(state["query"], top_k=5)
    state["retrieved_docs"] = docs
    return state

def reasoning_node(state):
    reasoning = llm.predict(
        f"Given these docs: {state['retrieved_docs']}\n\nReason about: {state['query']}"
    )
    state["reasoning"] = reasoning
    return state

def action_node(state):
    action = llm.predict(f"Plan action based on: {state['reasoning']}")
    state["action_plan"] = action
    return state

graph.add_node("retrieve", retrieval_node)
graph.add_node("reason", reasoning_node)
graph.add_node("act", action_node)
graph.add_edge("retrieve", "reason")
graph.add_edge("reason", "act")
```

**Why:** Clear separation enables monitoring per node (latency, cost, errors). Easy to debug which node failed. Easy to replace one node without affecting others.

### 5. Checkpointer in Postgres for Stateful Agents

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Create checkpointer (creates tables automatically if missing)
checkpointer = PostgresSaver.from_conn_string("postgresql://user:pass@localhost/langgraph_db")

# Compile graph with checkpointer
agent = graph.compile(checkpointer=checkpointer)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "user_12345"}}
result = agent.invoke({"query": "..."}, config)

# Later: resume from same thread
result2 = agent.invoke({"query": "follow-up"}, config)
# ^^^ This automatically resumes from last checkpoint
```

**Why:** If agent crashes, restart from last checkpoint. Multi-turn persistence. Essential for long-running agents.

### 6. LangSmith for Trace-Level Debugging

```python
import os
os.environ["LANGCHAIN_API_KEY"] = "..."
os.environ["LANGSMITH_TRACING"] = "true"

# Every LangChain call automatically logged
response = llm.invoke("What is 2+2?")
# Appears in LangSmith UI with full trace

# Also works with LangGraph
result = agent.invoke({"query": "..."})
# Full agent execution graph visible in UI
```

**Why:** Production visibility without custom logging. See token usage, latency, errors per tool call. Invaluable for debugging production issues.

### 7. Dependency Injection for Swappable LLM Providers

```python
# Don't hardcode provider
class Agent:
    def __init__(self, llm_provider: str = "anthropic"):
        if llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model="claude-3-5-sonnet")
        elif llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")

# Usage
agent_claude = Agent("anthropic")
agent_openai = Agent("openai")  # Easy swap for testing, cost optimization
```

**Why:** Easy A/B test different models. Switch providers without code changes. Cost optimization: route cheap queries to Haiku, expensive to Opus.

### 8. Max Iteration / Recursion Limit Guards

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    iterations: int
    max_iterations: int = 10

def reasoning_node(state):
    if state["iterations"] >= state["max_iterations"]:
        return {"result": "Max iterations reached, stopping"}
    
    # ... reason ...
    state["iterations"] += 1
    return state

# Or use LangGraph built-in:
graph.compile(max_iterations=10)
```

**Why:** Prevent infinite loops. Uncontrolled agents can spiral, burning tokens and latency.

### 9. Graceful Degradation When Tool Calls Fail

```python
async def tool_execution_node(state):
    tool_calls = state["tool_calls"]
    results = []
    
    for call in tool_calls:
        try:
            result = await execute_tool(call)
            results.append(result)
        except Exception as e:
            # Don't crash; gracefully degrade
            results.append({
                "error": str(e),
                "fallback": "Skipped due to error, continuing..."
            })
    
    state["tool_results"] = results
    return state
```

**Why:** One failed tool shouldn't crash entire agent. Return partial results, let reasoning node decide next step.

### 10. Async Everywhere (`ainvoke`, `astream`)

```python
# ✅ Good (async, non-blocking)
response = await llm.ainvoke(prompt)
async for chunk in agent.astream(input):
    print(chunk)

# ❌ Bad (blocking, slow)
response = llm.invoke(prompt)
for chunk in agent.stream(input):
    print(chunk)
```

**Why:** Production apps need async for concurrency. Multiple users, multiple agents, no blocking.

---

## When NOT to Use LangChain

### Use LangChain if:
- Building multi-step agents (reasoning → tool call → next step)
- Need integrations (20+ vector DBs, 100+ LLM providers)
- State management across turns

### Use pure Anthropic SDK if:
```python
# Simple single-call use case
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet",
    max_tokens=1000,
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response.content[0].text)

# ✅ Simpler, faster, fewer dependencies
# LangChain overhead not needed
```

### Use LangChain over Anthropic SDK if:
- Multi-step agents with tools
- Switching between providers (test GPT-4 vs Claude)
- Need memory management, retrieval

---

## Production Checklist

- [ ] Pin exact versions for all `langchain*` packages
- [ ] Use `langchain-core` abstractions at API boundaries (library code)
- [ ] Wrap LLM calls with retry + exponential backoff
- [ ] Pydantic validation on all tool inputs/outputs
- [ ] Use LangGraph StateGraph for multi-step agents, not ConversationChain
- [ ] Checkpointer in Postgres for persistence, not in-memory
- [ ] LangSmith tracing enabled, dashboard monitored
- [ ] Dependency injection pattern for LLM provider swapping
- [ ] Max iteration guards (prevent infinite loops)
- [ ] Graceful degradation for tool failures
- [ ] Async (`ainvoke`, `astream`) for concurrency
- [ ] Load test agent under expected QPS before prod
- [ ] Monitor token usage per tool call, alert on spikes
- [ ] Document which versions of `langchain*` are tested and supported

---

## Interview Talking Points

**"How would you design a production-grade LangGraph agent?"**

Use StateGraph with TypedDict state (explicit field names). Separate concerns: retrieval node, reasoning node, action node. Each node has single responsibility, easy to test and debug.

Checkpointer in Postgres for persistence. If agent crashes, resume from last checkpoint. LangSmith for tracing every LLM call and tool execution.

Structured tool inputs with Pydantic, max iteration guards, graceful degradation if a tool fails. Async everywhere for concurrency.

**"How do you handle LangChain API versioning?"**

Pin exact versions in requirements. LangChain core is stable (few breaking changes), but `langchain` and `langchain-community` move fast (breaking changes common).

Test upgrades in staging before prod. Check migration guide for breaking changes. Separate core dependencies from community integrations in version pinning.

**"What's the difference between ConversationChain and LangGraph for production?"**

ConversationChain is deprecated (v0.2+). It's simple but opaque: you can't see the loop, hard to add custom logic.

LangGraph is the modern pattern: you control the state machine explicitly. Add conditions, branching, human-in-the-loop. Better for production because it's debuggable and extensible.

**"How do you prevent an agent from looping infinitely?"**

Max iteration guard: `graph.compile(max_iterations=10)`. Also add explicit state counter: if `iterations > max_iterations`, return early.

Additionally, design the agent so there's always a terminal node (no cycles back to reasoning). Use conditional edges that lead to `END` node.

**"When should you NOT use LangChain?"**

For simple single-call use cases, pure Anthropic SDK is cleaner and faster. No dependency bloat. LangChain's value is in multi-step orchestration and integrations.

