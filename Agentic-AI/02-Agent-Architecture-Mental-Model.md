# Agent Architecture and Mental Model
## The Six-Layer Agent Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: PERCEPTION                                        │
│  └─ Input: User query, sensor data, previous state         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: PLANNING & REASONING                              │
│  └─ Brain: LLM decides: "What tools do I need?"            │
│            "What's my next action?"                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: TOOL SELECTION & ROUTING                          │
│  └─ Logic: Parse tool name from LLM, validate arguments     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: EXECUTION (Tools, APIs, Code)                     │
│  └─ Execute: search_tool(), code_executor(), api_call()    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: MEMORY & STATE STORAGE                            │
│  └─ Store: Observation result in context/DB                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 6: OUTPUT / NEXT ITERATION                           │
│  └─ Loop back to Layer 2, OR return final response          │
└─────────────────────────────────────────────────────────────┘
```

## Single Query Flow (Step by Step)

**User Input**: "Research who was the first CEO of OpenAI"

1. **Perception**: Input enters the agent context. State = `{"messages": [HumanMessage("Research...")], "tools_used": []}`
2. **Planning**: LLM reads the query and thinks: "I don't have real-time info. I should search."
3. **Tool Selection**: LLM outputs `tool_call(name="web_search", query="first CEO of OpenAI")`
4. **Execution**: `web_search()` returns: "Sam Altman founded OpenAI and served as first CEO from 2015-2018"
5. **Memory**: Add observation to state: `{"observation": "Sam Altman, 2015-2018"}`
6. **Reasoning**: LLM reads observation and decides: "I have enough info, I can answer now" (or: "I need to search for more details")
7. **Output**: Return final answer: "Sam Altman was the first CEO of OpenAI from 2015 to 2018"

## The ReAct Loop (Explicit Reasoning Pattern)

```
User: "What's 47 * 23?"

┌─────────────────────────────────────────┐
│ Thought: I need to multiply 47 * 23     │
│ I should use the calculator tool        │
└─────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ Action: calculator    │
        │ args: a=47, b=23      │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ Observation: 1081     │
        └───────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Thought: I have the answer. Return it   │
└─────────────────────────────────────────┘
                    ↓
        Final Answer: 1081
```

This pattern forces transparency. The LLM's reasoning is visible at each step, and failures are traceable.

## Context Window as a Constraint

The agent's context window is finite. As the loop continues, the message history grows:

```
┌─ Initial context: 128K tokens available
├─ Query: 50 tokens
├─ Tool description: 2K tokens
├─ First tool result: 3K tokens
├─ Second tool result: 5K tokens
├─ Third tool result: 4K tokens
└─ Remaining: 114K tokens left
```

**Context management strategies**:
- **Sliding window**: Keep only the last N messages
- **Summarization**: Compress old observations into a summary
- **Compression**: Use specialized compression APIs before re-using old results
- **External memory**: For very long tasks, offload results to a vector DB and retrieve by relevance

## Token Budget and Cost

Each tool call costs tokens:

| Action | Tokens | Cost |
|--------|--------|------|
| Send query + system prompt | 500 | ~$0.01 |
| Tool definition (30 tools) | 5000 | ~$0.10 |
| Call LLM, get tool result | 2000 | ~$0.04 |
| Loop iteration (x5 loops) | 10000 | ~$0.20 |
| **Total**: Single task with 5 loops | **~17.5K tokens** | **~$0.35** |

**Why agents are expensive**: If a simple LLM call takes 1 pass (~1K tokens = $0.02), an agent loop takes 5-10 passes. Plan accordingly.

## Scratchpad Pattern

Many agents use an internal "scratchpad" — a section of context reserved for the LLM to think through steps without outputting them to the user:

```
<scratchpad>
Task: Research and write a blog post about AI agents
1. I need background material
2. I need examples
3. I need recent developments
4. Outline the post structure
5. Write each section

Step 1: Search for "AI agents intro"
</scratchpad>

Now that I have my plan, let me research...
```

The scratchpad is never shown to the user, but it helps the LLM organize thoughts and avoid hallucination. It's especially useful in Plan-and-Execute agents.

## State Graph Concept (Preview of `08-LangGraph-Core.md`)

Instead of a linear loop, agents are often modeled as graphs:

```
START
  ↓
[LLM Node] ──→ Tool calls found? ──Yes──→ [Tool Execution Node] ──┐
  ↑                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  No tool calls found → END
```

LangGraph makes this explicit. You define states, nodes (functions), and edges (transitions), then compile it into a runnable agent. See `08-LangGraph-Core.md` for the implementation.

## Common Pitfalls in Architecture

**1. Infinite loops**: Agent keeps calling tools in a circle. Mitigation: max iteration limit, token budget check.
**2. Context overflow**: Too much history, new query gets lost. Mitigation: memory compression, sliding window.
**3. Hallucinated tool calls**: LLM invents tools that don't exist. Mitigation: strict tool schema validation, guardrails.
**4. Slow response time**: Multiple loops take 10-30 seconds. Mitigation: parallel tool execution, caching.
**5. Tool ordering matters**: If you define tools in wrong order, LLM uses wrong one. Mitigation: sorted tool list, clear descriptions.
