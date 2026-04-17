# Multi-Agent Systems

## Core Patterns

Multi-agent systems coordinate multiple specialized agents to solve complex problems.

| Pattern | Structure | Best For | Complexity |
|---------|-----------|----------|-----------|
| **Supervisor** | Orchestrator → Specialists | Task routing, domain experts | Medium |
| **Swarm** | Peer agents, shared state | Brainstorming, consensus | Medium to High |
| **Pipeline** | Sequential agents (A → B → C) | Data processing workflows | Low |
| **Debate** | Multiple agents propose, then vote | Decision making, fact-checking | High |
| **Hierarchical** | Tree of agents (supervisors + workers) | Large-scale problems | High |

## Supervisor Pattern (Most Common)

One orchestrator decides which specialist handles each task.

```
User Query
    ↓
┌─────────────────┐
│  Dispatcher     │  (LLM decides: which specialist?)
└────────┬────────┘
         ↓
   ┌─────┴─────┐
   ↓           ↓
┌─────────┐ ┌──────────┐
│ Math    │ │Research  │
│Agent    │ │Agent     │
└────┬────┘ └────┬─────┘
     │           │
     └─────┬─────┘
           ↓
        Result
```

### Implementation

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class SupervisorState(TypedDict):
    messages: list
    next_agent: str

# Define specialist agents
math_agent = create_math_agent()  # Handles math questions
research_agent = create_research_agent()  # Handles research

# Supervisor node: decides which agent to call
def supervisor_node(state: SupervisorState):
    query = state["messages"][-1]["content"]

    # Use LLM to classify
    classification = llm.invoke(f"""
Classify this query:
- "math" if it's a math problem
- "research" if it needs web search
- "end" if I can answer directly

Query: {query}
""")

    next_agent = classification.strip().lower()
    return {"next_agent": next_agent}

# Build graph
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("math", math_agent)
graph.add_node("research", research_agent)

# Route supervisor output to appropriate agent
def route_from_supervisor(state: SupervisorState):
    next_agent = state["next_agent"]
    return next_agent

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "math": "math",
        "research": "research",
        "end": END
    }
)
graph.add_edge("math", END)
graph.add_edge("research", END)

# Compile and run
supervisor = graph.compile()
result = supervisor.invoke({"messages": [{"role": "user", "content": "What's 5*10?"}]})
```

## Swarm Pattern

Peer agents communicate directly, sharing state. No central orchestrator.

```
┌──────────┐
│  Agent A │
└────┬─────┘
     │
   Shared State
   (messages, findings)
     │
┌────┴─────┐
│  Agent B │
└────┬─────┘
     │
   Shared State
     │
┌────┴─────┐
│  Agent C │
└──────────┘
```

Each agent can read and modify shared state:

```python
class SwarmState(TypedDict):
    messages: list
    research_findings: list  # Shared results
    current_speaker: str

def agent_a(state: SwarmState):
    # Agent A does research, adds findings to shared state
    findings = search("AI agents")
    state["research_findings"].append(findings)
    state["current_speaker"] = "agent_b"
    return state

def agent_b(state: SwarmState):
    # Agent B reads findings from Agent A, adds analysis
    findings = state["research_findings"]
    analysis = llm.invoke(f"Analyze: {findings}")
    state["messages"].append({"role": "assistant", "content": analysis})
    state["current_speaker"] = "agent_c"
    return state

# Swarm: all agents see same state
graph = StateGraph(SwarmState)
graph.add_node("agent_a", agent_a)
graph.add_node("agent_b", agent_b)
graph.add_node("agent_c", agent_c)

# Each agent triggers the next
graph.add_edge(START, "agent_a")
graph.add_edge("agent_a", "agent_b")
graph.add_edge("agent_b", "agent_c")
graph.add_edge("agent_c", END)

swarm = graph.compile()
```

## Pipeline Pattern

Sequential execution: A's output becomes B's input.

```
Input → Agent A → Agent B → Agent C → Output
```

Each agent transforms the state:

```python
class PipelineState(TypedDict):
    raw_data: str
    processed_data: str
    analyzed_data: str

def extract_node(state: PipelineState):
    # Extract data from raw text
    extracted = extract_entities(state["raw_data"])
    return {"processed_data": extracted}

def analyze_node(state: PipelineState):
    # Analyze processed data
    analysis = llm.invoke(f"Analyze: {state['processed_data']}")
    return {"analyzed_data": analysis}

def report_node(state: PipelineState):
    # Generate report
    report = f"Analysis: {state['analyzed_data']}"
    return {"report": report}

# Simple pipeline
graph = StateGraph(PipelineState)
graph.add_node("extract", extract_node)
graph.add_node("analyze", analyze_node)
graph.add_node("report", report_node)

graph.add_edge(START, "extract")
graph.add_edge("extract", "analyze")
graph.add_edge("analyze", "report")
graph.add_edge("report", END)

pipeline = graph.compile()
```

## Debate Pattern

Multiple agents propose different answers, then vote:

```
Query
  ↓
┌─────────────────┐
│ Agent 1: "Yes"  │
│ Agent 2: "No"   │  (Each reasons independently)
│ Agent 3: "Yes"  │
└────────┬────────┘
         ↓
    Vote: 2 vs 1
    Majority wins: "Yes"
```

```python
def debate_node(state: AgentState):
    query = state["messages"][-1]["content"]

    # Each agent proposes an answer
    agent1_answer = llm1.invoke(f"Answer this: {query}")
    agent2_answer = llm2.invoke(f"Answer this: {query}")
    agent3_answer = llm3.invoke(f"Answer this: {query}")

    # Count votes (simplified: if answer contains "yes", it's a yes vote)
    votes = {
        "yes": sum([1 for a in [agent1_answer, agent2_answer, agent3_answer] if "yes" in a.lower()]),
        "no": sum([1 for a in [agent1_answer, agent2_answer, agent3_answer] if "no" in a.lower()])
    }

    # Majority wins
    final_answer = "yes" if votes["yes"] > votes["no"] else "no"

    return {"debate_result": final_answer, "votes": votes}
```

## Agent Communication Protocols

Agents coordinate via:

### Message Passing

```python
# Agent A sends message to Agent B
state["messages"].append({
    "role": "agent_a",
    "content": "Agent B, I found important data: ...",
    "recipient": "agent_b"
})

# Agent B receives and processes
if message["recipient"] == "agent_b":
    process_message(message)
```

### Shared State

```python
# All agents read/write to same state dictionary
state["findings"].append("Alice likes Python")
state["findings"].append("Bob likes Go")

# Any agent can access
if "Alice" in state["findings"]:
    print("Found info about Alice")
```

### Callback Hooks

```python
# Agent A does work, then calls a hook
def on_agent_a_complete(result):
    # Trigger Agent B
    agent_b.invoke({"input": result})

# Chain events
agent_a.on_complete(on_agent_a_complete)
```

## Supervisor Multi-Agent with LangGraph (Full Example)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class MultiAgentState(TypedDict):
    messages: list
    supervisor_messages: list

def supervisor_agent(state: MultiAgentState):
    """Orchestrator that routes to specialists"""
    messages = state["messages"]

    # Ask supervisor which specialist to call
    response = llm.invoke(f"""
You are a supervisor. Route this query to the right specialist:
- "research": for web searches
- "code": for code problems
- "math": for calculations

Messages: {messages}

Choose specialist: """)

    specialist = response.strip().lower()
    state["supervisor_messages"].append(f"Routing to: {specialist}")

    return state

def research_specialist(state: MultiAgentState):
    """Handles research tasks"""
    query = state["messages"][-1]["content"]
    results = web_search(query)
    state["messages"].append({"role": "research_specialist", "content": results})
    return state

def code_specialist(state: MultiAgentState):
    """Handles code tasks"""
    query = state["messages"][-1]["content"]
    code_solution = generate_code(query)
    state["messages"].append({"role": "code_specialist", "content": code_solution})
    return state

def math_specialist(state: MultiAgentState):
    """Handles math tasks"""
    query = state["messages"][-1]["content"]
    solution = solve_math(query)
    state["messages"].append({"role": "math_specialist", "content": solution})
    return state

# Build graph
graph = StateGraph(MultiAgentState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("research", research_specialist)
graph.add_node("code", code_specialist)
graph.add_node("math", math_specialist)

# Route from supervisor
def route_specialist(state: MultiAgentState):
    specialist = state["supervisor_messages"][-1].split(": ")[1]
    return specialist

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_specialist,
    {"research": "research", "code": "code", "math": "math"}
)
graph.add_edge("research", END)
graph.add_edge("code", END)
graph.add_edge("math", END)

# Compile and run
multi_agent = graph.compile()
result = multi_agent.invoke({
    "messages": [{"role": "user", "content": "Search for AI news"}],
    "supervisor_messages": []
})
print(result["messages"][-1])
```

## Failure Handling in Multi-Agent Systems

### Retry with Backoff

```python
def agent_with_retry(agent_func, max_retries=3, backoff_factor=2):
    attempt = 1
    delay = 1

    while attempt <= max_retries:
        try:
            return agent_func()
        except Exception as e:
            if attempt == max_retries:
                return f"ERROR: {e}"
            wait(delay)
            delay *= backoff_factor
            attempt += 1
```

### Fallback Agents

```python
def route_with_fallback(state: MultiAgentState):
    try:
        specialist = determine_specialist(state)
        return specialist
    except:
        # Fallback to general agent if classification fails
        return "general_agent"
```

### Human Escalation

```python
def critical_decision_node(state: MultiAgentState):
    decision = llm.invoke("Should we proceed with this payment?")

    if decision == "critical":
        # Interrupt: wait for human approval
        raise Interrupt("Approve this action?")

    return state
```

## Pattern Comparison

| Pattern | Best For | Pros | Cons |
|---------|----------|------|------|
| **Supervisor** | Task routing | Simple, clear control | Single point of failure |
| **Swarm** | Collaborative tasks | Resilient, flexible | Harder to coordinate |
| **Pipeline** | Data processing | Simple, sequential | Not adaptive |
| **Debate** | Decisions | Multiple perspectives | Slow (N agents) |
| **Hierarchical** | Complex organizations | Scalable, modular | Complex to implement |

Most production systems use **Supervisor** (easiest to debug) or **Hierarchical** (most scalable).

## Key Insights

1. **Communication overhead**: More agents = more coordination cost. Limit to 3-5 specialists per supervisor.
2. **State management**: Shared state gets messy fast. Use explicit message passing when possible.
3. **Failure propagation**: One failing agent can break the system. Always add error handling.
4. **Cost multiplies**: 3 agents = 3x the LLM calls. Monitor costs carefully.
5. **Debugging is hard**: Multi-agent systems are harder to trace. Use LangSmith for visibility.

Start with Supervisor pattern. Upgrade to more complex patterns only if needed.
