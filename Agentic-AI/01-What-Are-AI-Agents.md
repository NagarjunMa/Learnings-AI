# What Are AI Agents

## Definition
An AI agent is a software system that combines an LLM with a control loop, tools, memory, and reasoning capabilities. Core equation: **Agent = LLM + Tools + Memory + Loop**. The agent perceives its environment (user query, previous state), reasons about the best action, executes tools, and refines its approach iteratively.

## The Four Core Properties

| Property | Meaning | Example |
|----------|---------|---------|
| **Perception** | Ability to receive and understand input (queries, sensor data, API responses) | User asks: "Research the CEO of OpenAI" |
| **Reasoning** | Chain-of-thought planning about what to do next | Agent thinks: "I need to search for information first" |
| **Action** | Execute tools, APIs, or code to affect the world | Agent calls: `search_tool("CEO of OpenAI")` |
| **Learning** | Update internal state (memory, context) based on observations | Agent stores: "Sam Altman is CEO, founded in 2015" |

## Major Agent Architectures

### ReAct (Reasoning + Acting)
Pattern: Thought → Action → Observation → Thought (repeat). The agent explicitly writes reasoning steps visible in its scratchpad before calling tools. Best for tasks requiring transparent decision-making. Forces the LLM to slow down and think through each step.

### Plan-and-Execute
The agent creates a high-level plan first, then executes each step. Reduces hallucination by committing to a plan upfront. Good for multi-step tasks with fixed scope (e.g., "Write a 3-section blog post about X").

### Reflexion (Self-Critique)
Agent executes → observes outcome → critiques own response → retries. Builds a "self-reflection memory" of past failures. Stronger for tasks where trial-and-error improves performance (code generation, research).

### MRKL (Modular Reasoning, Knowledge, and Language)
Specialized tools for specialized problems. Multiple expert subagents each handling their domain (calculator for math, code executor for programming, search for research). Orchestrated by a supervisor agent.

## When to Use Agents vs. Plain LLM

| Scenario | Plain LLM Call | Agent Loop |
|----------|----------------|-----------|
| **Simple Q&A** (e.g., "What is Python?") | ✓ Use LLM | Too heavy |
| **Long-horizon task** (e.g., "Research and write a report") | Fails without tools | ✓ Use Agent |
| **API integration** (call Slack, GitHub, Stripe) | Can't execute APIs | ✓ Use Agent |
| **Real-time data** (stock prices, weather) | Hallucinations | ✓ Use Agent with search |
| **Code execution** | Output only, no execution | ✓ Use Agent with code executor |
| **Persistence** (remember prior context) | Stateless | ✓ Use Agent with memory |
| **Latency budget < 200ms** | ✓ One pass is fast | Too slow (multiple loops) |
| **Cost critical** | ✓ One forward pass | Expensive (multiple calls) |

## Agent vs Chatbot vs LLM API

| Aspect | LLM API | Chatbot | Agentic Agent |
|--------|---------|---------|---------------|
| **Statefulness** | Stateless | Single session state | Multi-turn + persistent memory |
| **Tool use** | None | Limited (buttons, quick replies) | Full tool ecosystem |
| **Autonomy** | Passive (user commands) | Semi-passive (scripted) | Active (self-directed) |
| **Failure recovery** | Retry manually | User retries | Auto-retry with different approach |
| **Latency** | 1 pass (~1 sec) | Per-turn (~2 sec) | Multiple loops (~5-30 sec) |
| **Example** | "Write a poem" API | Customer support bot | Research bot building a full report |

## Key Distinctions

**Agent ≠ Chain**: A chain is deterministic (Step A → Step B → Step C, fixed path). An agent is adaptive (observes outcome, decides next step). Chains are for workflows you control; agents are for tasks you can't fully specify upfront.

**Agent ≠ Retrieval-Augmented Generation (RAG)**: RAG fetches documents then answers. Agents use RAG as *one* tool among many and decide whether to use it based on reasoning.

**Agent ≠ Agentic Loop**: The agent is the system; the agentic loop is the control mechanism (observe → reason → act → observe). Not all agents have explicit loops (some use implicit routing).
