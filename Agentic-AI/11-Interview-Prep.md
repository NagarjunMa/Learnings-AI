# Interview Prep

## Concept Questions

### Q: Explain the difference between an agent and a chain.

**A**: A chain is deterministic (fixed path: A → B → C). An agent is adaptive (observes outcomes, decides next step).

Chain: Prompt → LLM → Parser (always same path)
Agent: LLM → [Should I use tool X?] → (If yes) Execute X → LLM → [Done?]

Agents adapt based on tool results. Chains follow a predetermined path.

### Q: What is ReAct and why does it matter?

**A**: ReAct (Reason + Act) is an agent pattern that makes reasoning explicit and traceable.

Pattern: Thought → Action → Observation → Thought (repeat)

Why it matters:
1. Transparency: You can see every reasoning step
2. Self-correction: Agent can fix mistakes mid-loop
3. Reliability: Explicit reasoning prevents hallucination

### Q: Compare LangChain and LangGraph. When would you use each?

**A**:
- **LangChain**: Simpler chains and basic agents. Good for prototyping.
- **LangGraph**: Stateful, complex agents. Production-grade.

|Aspect|LangChain|LangGraph|
|------|---------|---------|
|State management|Manual|First-class|
|Persistence|Not built-in|Built-in checkpointing|
|Multi-agent|Possible but messy|Natural with subgraphs|
|Visualization|No|Yes|

Use LangChain to get started. Move to LangGraph for production complexity.

### Q: How would you prevent an agent from entering an infinite loop?

**A**: Multiple strategies:

1. **Max iterations**: `max_iterations=10` in executor
2. **Token budget**: Stop if tokens exceed budget
3. **Repeated action detection**: If same tool called 3 times, stop
4. **Timeout**: Kill execution after N seconds

```python
def should_continue(state):
    if state["iterations"] > 10:
        return END
    return "continue"
```

### Q: Explain memory in agents. What types exist?

**A**: Four types:

1. **In-Context**: Messages in current context window (short-term, free)
2. **External (Vector DB)**: Embeddings stored in database (long-term, searchable)
3. **Episodic**: Timestamped events (long-term, specific)
4. **Semantic**: Learned patterns (long-term, conceptual)

Most agents use in-context + external memory (vector DB).

### Q: What's the difference between supervised and unsupervised agent architectures?

**A**:
- **Supervised**: Orchestrator controls which specialist agent handles each task (Supervisor pattern)
- **Unsupervised**: Peer agents self-coordinate without central control (Swarm pattern)

Supervised is easier to debug. Unsupervised is more resilient.

### Q: How do you handle tool hallucination?

**A**: Tool hallucination: LLM invents tools that don't exist.

Solutions:
1. **Schema validation**: Check tool exists before execution
2. **Clear tool descriptions**: Reduce confusion about what tools do
3. **Error feedback**: When tool call fails, tell LLM and retry
4. **Guardrails**: Filter LLM output, only allow known tools

```python
allowed_tools = {"calculator", "search", "code_executor"}
if llm_output["tool"] not in allowed_tools:
    # Tell LLM it's invalid, ask to retry
```

### Q: What's a conditional edge in LangGraph?

**A**: A conditional edge routes based on state. Instead of always going A → B, you check state and decide.

```python
def should_continue(state):
    if state["tool_calls"]:
        return "execute_tools"
    else:
        return END

graph.add_conditional_edges("llm", should_continue)
```

Routes to different nodes based on state, enabling branching logic.

## System Design Questions

### Q: Design a research agent that browses the web and writes a report.

**A**:

**Requirements**:
- Takes a research topic
- Searches the web for information
- Gathers sources
- Synthesizes findings
- Writes a 3-section report

**Architecture**:

```
┌─────────────────┐
│  START: Topic   │
└────────┬────────┘
         ↓
┌─────────────────────────┐
│  Search Node            │
│  - web_search(topic)    │
│  - Returns top 10 links │
└────────┬────────────────┘
         ↓
┌──────────────────────────┐
│  Scrape Node             │
│  - Extract content       │
│  - Cache URLs            │
└────────┬─────────────────┘
         ↓
┌──────────────────────────┐
│  Analyze Node            │
│  - Summarize findings    │
│  - Extract key points    │
└────────┬─────────────────┘
         ↓
┌──────────────────────────┐
│  Write Node              │
│  - 3-section report      │
│  - Cite sources          │
└────────┬─────────────────┘
         ↓
┌──────────────────────┐
│  END: Return Report  │
└──────────────────────┘
```

**State**:
```python
class ResearchState(TypedDict):
    topic: str
    queries: list  # Search queries executed
    documents: list  # Retrieved documents
    findings: list  # Extracted key points
    report: str  # Final report
```

**Nodes**:
1. **Search**: Use web_search tool for top results
2. **Scrape**: Extract text from URLs (rate-limited)
3. **Analyze**: LLM summarizes documents, extracts findings
4. **Write**: LLM writes report with citations

**Checkpointing**: Use SQLite saver so you can resume if failed

### Q: Design a multi-agent code review system.

**A**:

**Architecture**: Supervisor routes to specialists

```
Code Submission
    ↓
┌──────────────────┐
│  Dispatcher      │  Determines: Security? Performance? Style?
└────────┬─────────┘
    ┌─────┼─────┐
    ↓     ↓     ↓
┌────────┐┌───────┐┌─────────┐
│Security││Perf   ││ Style   │
│Agent   ││Agent  ││ Agent   │
└────┬───┘└───┬───┘└────┬────┘
     └─────┬──┴────┬─────┘
           ↓
      ┌─────────────┐
      │ Aggregator  │  Combines reviews
      └──────┬──────┘
             ↓
        Final Report
```

**Specialists**:
- **Security**: Check for SQL injection, XSS, auth issues
- **Performance**: Check for inefficient queries, N+1 problems
- **Style**: Check naming, code organization, duplicates

**Multi-turn**: Developer can ask clarifying questions, agents refine feedback

### Q: How would you optimize cost for a high-volume agent application?

**A**:

**Cost drivers**:
- Token usage (biggest)
- LLM API calls (3-5 per task)
- Tool calls (search, APIs)

**Optimizations**:

1. **Caching**: LRU cache for frequent queries
2. **Batching**: Combine multiple tool calls
3. **Model tiering**: Cheap models for simple tasks
4. **Token compression**: Summarize old messages
5. **Early exit**: Stop if confidence is high

**Example**:
```
10,000 queries/day
→ 2 LLM calls per query = 20,000 API calls
→ 5K tokens per call = 100M tokens
→ At $0.01 per 1K tokens = $1,000/day

With caching (50% cache hit): $500/day (50% savings)
With model tiering (use GPT-3.5 for 70% of tasks): $300/day
With token compression: $200/day
Total: 80% cost reduction
```

## Coding Challenges

### Challenge 1: Implement a Tool-Calling Loop

Implement the agent loop that:
1. Takes user input
2. Calls LLM with tools
3. Executes tool
4. Feeds result back to LLM
5. Repeats until no more tool calls

```python
def agent_loop(user_query: str, tools: list, max_iterations: int = 5):
    messages = [{"role": "user", "content": user_query}]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Call LLM with tools
        response = call_llm_with_tools(messages, tools)

        # Check for tool calls
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            # No more tools → return final answer
            return response.get("final_answer")

        # Execute tools
        messages.append({"role": "assistant", "content": response})

        tool_results = []
        for tool_call in tool_calls:
            result = execute_tool(tool_call["name"], tool_call["args"])
            tool_results.append({
                "tool": tool_call["name"],
                "result": result
            })

        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached"

# Usage
answer = agent_loop("What's 5 * 10?", tools=[calculator_tool, search_tool])
```

### Challenge 2: Build a State Machine for a Task Agent

```python
from enum import Enum
from typing import TypedDict

class TaskState(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    DONE = "done"
    FAILED = "failed"

class TaskAgentState(TypedDict):
    task: str
    plan: list
    results: list
    state: TaskState
    attempt: int

def planning_node(state: TaskAgentState) -> dict:
    """Create plan for task"""
    plan = llm.invoke(f"Create a 3-step plan for: {state['task']}")
    return {
        "plan": parse_plan(plan),
        "state": TaskState.EXECUTING
    }

def executing_node(state: TaskAgentState) -> dict:
    """Execute each step of plan"""
    results = []
    for step in state["plan"]:
        result = execute_step(step)
        results.append(result)
    return {
        "results": results,
        "state": TaskState.REVIEWING
    }

def reviewing_node(state: TaskAgentState) -> dict:
    """Review results and decide: done or retry"""
    review = llm.invoke(f"Review results: {state['results']}")

    if "success" in review.lower():
        return {"state": TaskState.DONE}
    else:
        if state["attempt"] < 3:
            return {
                "state": TaskState.EXECUTING,
                "attempt": state["attempt"] + 1
            }
        else:
            return {"state": TaskState.FAILED}

# Build graph
graph = StateGraph(TaskAgentState)
graph.add_node("planning", planning_node)
graph.add_node("executing", executing_node)
graph.add_node("reviewing", reviewing_node)

graph.add_edge(START, "planning")
graph.add_edge("planning", "executing")
graph.add_edge("executing", "reviewing")
graph.add_conditional_edges("reviewing", lambda s: s["state"], {...})

agent = graph.compile()
```

## Interview Key Insights

### Insight 1: Agent vs Chain
Agents are **adaptive** (respond to tool results).
Chains are **deterministic** (follow fixed path).

### Insight 2: Cost Matters
Agent loops cost 5-10x more than single LLM call.
Always optimize: caching, batching, model selection.

### Insight 3: State is Everything
StateGraph makes state explicit and manageable.
Without clear state, agents become unmaintainable.

### Insight 4: Tools are the Execution Layer
LLMs are great at reasoning, bad at acting.
Tools bridge that gap (search, APIs, code, DB).

### Insight 5: Observability is Essential
You can't debug what you can't see.
Use LangSmith from day one.

## Sample Follow-Up Questions (To Ask Back)

1. "What's the expected latency budget?" (Helps choose between agent vs simple LLM)
2. "What's the monthly budget?" (Determines cost optimizations needed)
3. "How important is observability/debugging?" (Determines if LangSmith is needed)
4. "Is this multi-user or single-user?" (Affects state management approach)
5. "What's the failure recovery strategy?" (Determines checkpointing needs)

## Red Flags to Avoid

❌ "Agents don't hallucinate" — Agents still hallucinate, especially with tools
❌ "Just use the most capable model" — Overkill for simple tasks, expensive
❌ "We don't need monitoring" — Agents fail silently without tracing
❌ "Infinite loop protection is unnecessary" — Happens more than you think
❌ "We'll optimize cost later" — By then you're bleeding money

## Green Flags

✅ You mention LangSmith for observability
✅ You consider cost optimization early
✅ You discuss failure modes (infinite loops, hallucination)
✅ You have a clear state management strategy
✅ You understand the difference between LangChain and LangGraph
✅ You discuss testing and evaluation strategies
