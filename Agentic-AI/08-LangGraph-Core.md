# LangGraph Core

## What LangGraph Adds Over LangChain

LangChain is great for chains and basic agents, but LangGraph adds:

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| **State management** | Manual (memory object) | Built-in StateGraph |
| **Graph visualization** | No | Yes, `draw_mermaid_png()` |
| **Conditional routing** | Limited | Full conditional edges |
| **Checkpointing** | No | Yes, persistence + resumption |
| **Human-in-the-loop** | Manual | Built-in interrupts |
| **Multi-agent** | Possible but messy | Natural with subgraphs |
| **Streaming** | Partial | Full streaming support |

LangGraph is for **stateful, long-running agents**. LangChain is for simpler chains.

## Core Concepts

### StateGraph

A graph where nodes are functions and edges are transitions. State flows through the graph.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Define state schema
class AgentState(TypedDict):
    messages: list      # Conversation history
    tools_used: list    # Track which tools were called
    iterations: int

# 2. Create graph
graph = StateGraph(AgentState)

# 3. Add nodes (functions)
def llm_node(state: AgentState) -> dict:
    """Call LLM to decide what to do next"""
    # ... LLM logic ...
    return {"messages": [...], "iterations": state["iterations"] + 1}

def tool_node(state: AgentState) -> dict:
    """Execute tools"""
    # ... tool execution ...
    return {"tools_used": [...]}

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

# 4. Add edges
graph.add_edge(START, "llm")          # Start by calling LLM
graph.add_edge("tools", "llm")        # After tools, call LLM again

# 5. Add conditional edges (branching)
def should_continue(state: AgentState) -> str:
    if state["iterations"] > 5:
        return END
    return "tools"

graph.add_conditional_edges("llm", should_continue)

# 6. Compile
agent = graph.compile()
```

### Nodes

A node is a function that takes state and returns a partial state update:

```python
def my_node(state: AgentState) -> dict:
    # Receive current state
    messages = state["messages"]
    tools_used = state["tools_used"]

    # Process
    new_message = f"I've used {len(tools_used)} tools so far"
    messages.append({"role": "assistant", "content": new_message})

    # Return partial update (not the full state)
    return {"messages": messages}

# LangGraph merges the return dict into the state:
# state = {**state, **node_return}
```

### Edges

Edges connect nodes. There are two types:

1. **Static edge**: Always go from A to B
   ```python
   graph.add_edge("llm", "tools")  # Always: llm → tools
   ```

2. **Conditional edge**: Route based on state
   ```python
   def route_based_on_message(state):
       last_message = state["messages"][-1]
       if "tool" in last_message.content:
           return "tools"
       else:
           return END

   graph.add_conditional_edges("llm", route_based_on_message)
   ```

## Full Working LangGraph Agent

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from anthropic import Anthropic
from typing import TypedDict, Annotated, Any
import json

# State schema
class AgentState(TypedDict):
    messages: list

# Tool definitions
tools = [
    {
        "name": "calculator",
        "description": "Perform math",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "multiply"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }
    }
]

# Tool implementations
def calculator(operation: str, a: float, b: float) -> float:
    if operation == "add": return a + b
    if operation == "multiply": return a * b
    return 0

# Tool execution node (LangGraph's ToolNode)
tool_node = ToolNode([
    {"name": "calculator", "func": lambda x: str(calculator(**x))}
])

# LLM node
client = Anthropic()

def llm_node(state: AgentState) -> AgentState:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=state["messages"]
    )

    # Add LLM's response to messages
    state["messages"].append({"role": "assistant", "content": response.content})

    return state

# Routing logic
def should_continue(state: AgentState) -> str:
    """Check if there are tool calls in the last message"""
    last_message = state["messages"][-1]

    # If LLM's response contains tool calls, go to tool_node
    for block in last_message.get("content", []):
        if hasattr(block, "type") and block.type == "tool_use":
            return "tools"

    # Otherwise, we're done
    return END

# Build graph
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")

# Compile and run
agent = graph.compile()

# Run agent
initial_state = {"messages": [{"role": "user", "content": "What's 5 * 10?"}]}
final_state = agent.invoke(initial_state)
print(final_state["messages"][-1])
```

## Conditional Edges (Branching Logic)

Edges can split based on state:

```python
def route_to_specialist(state: AgentState) -> str:
    query = state["messages"][-1]["content"]

    if "math" in query.lower():
        return "math_specialist"
    elif "code" in query.lower():
        return "code_specialist"
    else:
        return "general"

graph.add_conditional_edges("dispatcher", route_to_specialist)
```

This creates a multi-agent system in which specialists handle distinct tasks.

## Checkpointing: Persistence and Resumption

Checkpointing saves the graph's state so you can resume later:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# In-memory checkpoint (for testing)
memory_saver = InMemorySaver()
agent = graph.compile(checkpointer=memory_saver)

# Or Postgres (for production)
pg_saver = PostgresSaver(connection_string="postgresql://...")
agent = graph.compile(checkpointer=pg_saver)

# Run with thread ID (like a conversation session)
config = {"configurable": {"thread_id": "user-alice-123"}}
result = agent.invoke(initial_state, config=config)

# Later, resume the same conversation
# The checkpointer automatically loads the previous state
result = agent.invoke({"messages": [{"role": "user", "content": "Continue..."}]}, config=config)
```

The `thread_id` acts like a conversation session. Each invocation in the same thread continues from the previous state.

## Human-in-the-Loop (Interrupts)

Pause the graph and wait for human approval:

```python
def tool_node_with_interrupt(state: AgentState) -> AgentState:
    # Execute tool
    result = execute_tool(state["tool_call"])

    # For critical tools (like charging money), wait for approval
    if state["tool_call"]["name"] == "charge_payment":
        # Raise an interrupt
        from langgraph.types import Interrupt
        raise Interrupt(f"Approve payment: {result}")

    return state

# When interrupted:
# 1. User reviews the interrupt message
# 2. User approves or rejects
# 3. Execution resumes from that point (with checkpointer)
```

## Multi-Turn Conversation with State Persistence

```python
agent = graph.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "conv-1"}}

# Turn 1
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config=config
)

# Turn 2 - State automatically includes previous messages
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config
)
# Agent knows: "Your name is Alice" (from Turn 1)
```

The checkpointer automatically accumulates state across calls with the same `thread_id`.

## Graph Visualization

```python
from IPython.display import Image

# Generate graph diagram
image = agent.get_graph().draw_mermaid_png()
Image(image)

# ASCII version
print(agent.get_graph().draw_ascii())

# Output:
#     ┌─────────┐
#     │ START   │
#     └────┬────┘
#          │
#          ↓
#     ┌─────────┐
#     │  llm    │
#     └────┬────┘
#          │
#      ┌───┴───┐
#      ↓       ↓
#   tools    END
#      ↑
#      │
#      └───┐
#          │
#      ┌───┴───┐
#      │ llm   │
#      └───────┘
```

Visualization helps you verify the graph structure before running.

## Subgraphs (Hierarchical Agents)

Create a graph within a graph for modular agent systems:

```python
# Specialist graph
math_graph = StateGraph(AgentState)
math_graph.add_node("calculator", calculator_node)
math_graph.add_edge(START, "calculator")
math_compiled = math_graph.compile()

# Main graph
main_graph = StateGraph(AgentState)
main_graph.add_node("dispatcher", dispatcher_node)
main_graph.add_node("math_specialist", math_compiled)  # Subgraph as a node
main_graph.add_edge(START, "dispatcher")
main_graph.add_conditional_edges("dispatcher", route_to_specialist)

# Routing maps to subgraph
# "math_specialist" → math_compiled
```

Subgraphs allow you to build complex multi-agent systems hierarchically.

## Streaming

LangGraph supports streaming results as they're generated:

```python
# Stream results as they're computed
for event in agent.stream(initial_state, config=config):
    print(event)

# Useful for real-time updates (show user what agent is thinking)
```

## Prebuilt Agent (create_react_agent)

Most engineers use `langgraph.prebuilt.create_react_agent` instead of building StateGraph manually for simple tool-calling agents.

### One-Line Agent Creation

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

# Define tools
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate math expression."""
    return str(eval(expression))

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
tools = [search, calculate]

# One line — no manual StateGraph, nodes, edges needed
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver()  # Optional: enable persistence
)

# Run
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 25 * 48?"}]},
    config={"configurable": {"thread_id": "session-1"}}
)

print(result["messages"][-1].content)  # Output: "25 * 48 = 1200"
```

### What's Inside: MessagesState

`create_react_agent` uses `MessagesState` internally:

```python
from langgraph.graph import MessagesState
from typing_extensions import Annotated
from operator import add

# MessagesState is roughly:
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]  # Appends new messages, doesn't overwrite

# add_messages reducer automatically merges new messages into history
# This prevents the "replace entire message list" problem of manual state

# That's why agent.invoke() works with {"messages": [...]}
# and result["messages"] contains full conversation history
```

### When to Use Prebuilt vs Manual StateGraph

| Scenario | Use Prebuilt | Use Manual StateGraph |
|---|---|---|
| **Simple tool-calling agent** | ✅ Yes | ❌ Overkill |
| **Agent with 2-3 tools, no special routing** | ✅ Yes | ❌ Overkill |
| **Agent that needs custom nodes** | ❌ No | ✅ Yes |
| **Multi-step workflow with custom logic** | ❌ No | ✅ Yes |
| **Non-message state** (custom TypedDict) | ❌ No | ✅ Yes |
| **Conditional routing based on state** | ❌ Limited | ✅ Yes |
| **Rapid prototype** | ✅ Yes | ❌ Too verbose |
| **Production system with complex logic** | ❌ No | ✅ Yes |

### Prebuilt + Custom Routing

You can use prebuilt agent but customize routing:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph

# Create base agent
agent = create_react_agent(llm, tools, checkpointer=MemorySaver())

# Wrap in custom StateGraph if you need special handling
# (Most cases don't need this — prebuilt is enough)

# Example: Add logging around agent
def logged_agent_wrapper(state):
    print(f"Calling agent with {len(state['messages'])} messages")
    result = agent.invoke(state)
    print(f"Agent returned {len(result['messages'])} messages")
    return result
```

---

## Key Differences: LangGraph vs LangChain Agent

| Aspect               | LangChain Agent           | LangGraph                 |
| -------------------- | ------------------------- | ------------------------- |
| **State**            | Manual memory object      | First-class StateGraph    |
| **Persistence**      | Not built-in              | Built-in checkpointing    |
| **Routing**          | Limited conditional logic | Full graph with edges     |
| **Visualization**    | No                        | Yes, draw_mermaid_png()   |
| **Multi-agent**      | Possible but complicated  | Natural with subgraphs    |
| **Resume/interrupt** | Manual logic              | Built-in interrupts       |
| **Learning curve**   | Easier                    | Steeper but more powerful |

**When to use LangChain Agent**: Simple single-agent tasks, quick prototypes
**When to use LangGraph**: Production agents, multi-agent systems, persistent state, complex workflows

## Production Checklist

- [ ] Define AgentState with all required fields
- [ ] Add error handling in nodes
- [ ] Set max_iterations to prevent infinite loops
- [ ] Use checkpointer for persistence (Postgres in production)
- [ ] Test graph visualization with `draw_mermaid_png()`
- [ ] Add logging and tracing (LangSmith)
- [ ] Test with checkpointer: verify thread_id persistence
- [ ] Load test: ensure graph handles concurrent invocations
- [ ] Monitor latency per node (see traces in LangSmith)

LangGraph is the framework for production-grade agentic AI.
