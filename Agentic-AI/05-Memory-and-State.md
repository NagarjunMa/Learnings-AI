# Memory and State in Agents

## Four Types of Agent Memory

| Type | Persistence | Retrieval | Use Case | Examples |
|------|-------------|-----------|----------|----------|
| **In-Context** | During conversation only | All messages in context window | Short-term reasoning within one task | Sliding window of last 10 messages |
| **External (Vector DB)** | Permanent (database) | Semantic similarity search | Long-term facts, learned from interactions | Vector store with embeddings |
| **Episodic** | Permanent (database) | By timestamp or relevance | Record of past events/conversations | "On Jan 15, user asked about..." |
| **Semantic/Procedural** | Permanent | By concept matching | General knowledge, learned patterns | "When user says X, do Y" |

## In-Context Memory

The simplest form: all relevant messages are in the current context window.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What did I tell you my name was?"},
    {"role": "assistant", "content": "You told me your name is Alice."}
]

# Alice's name is retrieved from message history, not a database
```

**Pros**: Simple, no external storage, complete conversation context
**Cons**: Limited by context window size (~4K to 128K tokens), expensive for large histories

## Sliding Window Strategy

As conversations grow, keep only recent messages to save tokens:

```python
def maintain_sliding_window(messages: list, window_size: int = 10):
    """Keep system prompt + last N messages"""
    if len(messages) <= window_size + 1:  # +1 for system prompt
        return messages

    system_prompt = messages[0]
    recent = messages[-(window_size):]
    return [system_prompt] + recent

# Before: 100 messages
# After: system prompt + 10 recent messages = ~5K tokens instead of 50K
```

**Trade-off**: You lose context about earlier interactions, but save significant tokens.

## Summarization Strategy

Instead of dropping old messages, compress them:

```python
def summarize_messages(messages: list, summary_start: int = 1, summary_end: int = -10):
    """Summarize old messages, keep recent ones verbatim"""
    old_messages = messages[summary_start:summary_end]
    recent_messages = messages[summary_end:]

    # Use LLM to summarize
    summary_prompt = f"""Summarize this conversation in 100 words:
{format_messages(old_messages)}"""

    summary = llm.call(summary_prompt)

    return [
        messages[0],  # System prompt
        {"role": "system", "content": f"Previous context: {summary}"},
        *recent_messages
    ]

# Before: 50 messages (~20K tokens)
# After: system + summary + 10 recent (~5K tokens)
```

**Trade-off**: Some detail is lost in summarization, but you preserve more context than sliding window.

## External Memory with Vector Databases

For truly long-term memory, store facts separately and retrieve them when relevant.

### How It Works

1. **Embedding**: Convert text to a vector (high-dimensional number)
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')

   text = "Alice works at Acme Corp and loves machine learning"
   vector = model.encode(text)  # [0.2, -0.1, 0.5, ..., 0.9]
   ```

2. **Store**: Save the vector and original text to a vector DB
   ```python
   vector_db.insert(
       text="Alice works at Acme Corp and loves machine learning",
       vector=vector,
       metadata={"user": "alice", "date": "2025-01-15"}
   )
   ```

3. **Retrieve**: When needed, find similar memories by vector similarity
   ```python
   query = "What company does Alice work for?"
   query_vector = model.encode(query)

   results = vector_db.search(query_vector, top_k=3)
   # Returns: ["Alice works at Acme Corp..."]
   ```

4. **Use in Agent**: Inject retrieved memories into context
   ```python
   memories = vector_db.search(user_query, top_k=5)

   messages = [
       {"role": "system", "content": "You are a helpful assistant"},
       {"role": "system", "content": f"User context:\n{format_memories(memories)}"},
       {"role": "user", "content": user_query}
   ]
   ```

### Full Vector Memory Loop

```python
class AgentWithMemory:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.conversation_history = []

    def chat(self, user_query: str) -> str:
        # Step 1: Retrieve relevant memories
        memories = self.vector_db.search(user_query, top_k=5)

        # Step 2: Build context
        context_message = f"""
						Relevant past interactions:
						{format_memories(memories)}

						New query: {user_query}
						"""

        # Step 3: Call LLM
        response = llm.call([
            {"role": "system", "content": "You are helpful and remember past interactions"},
            *self.conversation_history,
            {"role": "user", "content": context_message}
        ])

        # Step 4: Store new interaction in memory
        self.vector_db.insert(
            text=f"User asked: {user_query}\nAssistant said: {response}",
            metadata={"date": now()}
        )

        # Step 5: Update conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response
```

## State Graph for Multi-Turn Conversations

Agents maintain state across multiple turns. State = all relevant information.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: list  # Conversation history
    user_id: str
    current_task: str
    memory: dict  # Short-term facts
    tools_used: list

# State evolves:
# Turn 1: user asks "Book a flight to NYC on Friday"
# State.messages = [query]
# State.current_task = "flight_booking"
# State.tools_used = []

# Turn 2: Agent calls flight_search tool
# State.tools_used = ["flight_search"]
# State.messages = [query, assistant_thought, tool_result]

# Turn 3: Agent books the flight
# State.messages = [..., confirmation]
# State.current_task = "completed"
```

See `08-LangGraph-Core.md` for implementation with LangGraph.

## Episodic Memory

Store conversations as discrete episodes:

```python
class Episode:
    def __init__(self, user_id: str, topic: str):
        self.user_id = user_id
        self.topic = topic
        self.messages = []
        self.created_at = now()
        self.summary = None

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content, "timestamp": now()})

    def summarize(self):
        # At end of conversation, create a summary
        prompt = f"Summarize this conversation: {format_messages(self.messages)}"
        self.summary = llm.call(prompt)

# Retrieval
episodes = db.find_episodes(user_id="alice", topic="flight_booking")
# Returns all past flight booking conversations for Alice
```

## Semantic / Procedural Memory

Learn patterns and rules from interactions:

```python
semantic_memory = {
    "When user asks for weather": {
        "action": "call web_search tool",
        "example": "Q: What's the weather? → A: Use web_search"
    },
    “When user mentions deadline”: {
        “action": “help prioritize tasks",
        "example": "Q: Deadline is Friday → A: Suggest timeline"
    }
}

# Used to prime the LLM:
system_prompt = """You are helpful. Remember these patterns:
- When user asks for weather, search the web
- When user mentions a deadline, help prioritize
"""
```

## Context Window Management Trade-Offs

| Strategy         | Context Saved                 | Data Loss                | When to Use                        |
| ---------------- | ----------------------------- | ------------------------ | ---------------------------------- |
| Sliding Window   | 80% (keep last 10 msgs)       | Recent context preserved | Short conversations (<50 msgs)     |
| Summarization    | 75% (compress old msgs)       | Details lost in summary  | Medium conversations (50-500 msgs) |
| Vector Retrieval | 90% (retrieve relevant only)  | Lose irrelevant context  | Long conversations (1000+ msgs)    |
| Hybrid           | 85% (sliding window + vector) | Balanced trade-off       | Production agents                  |

## Hybrid Approach (Recommended)

Combine multiple strategies:

```python
def prepare_agent_context(user_query: str, conversation_history: list):
    """
    1. Keep last 5 messages (recent context)
    2. Summarize messages 5-20 (old context)
    3. Vector search for relevant past memories
    """

    # Recent: last 5 messages
    recent = conversation_history[-5:]

    # Summarized: messages 5-20
    if len(conversation_history) > 20:
        old = conversation_history[:-5]
        summary = summarize_messages(old)
    else:
        summary = None

    # Vector search: find relevant memories
    memories = vector_db.search(user_query, top_k=3)

    # Construct final context
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "system", "content": f"Previous context: {summary}"},
        {"role": "system", "content": f"Relevant memories:\n{format_memories(memories)}"},
        *recent
    ]

    return messages
```

This reduces token usage while maintaining context quality.

## Why Memory Matters

Without memory: Agent is stateless, can't learn, repeats mistakes.
With memory: Agent maintains context, improves over time, provides personalized responses.

The challenge is balancing:
- **Completeness**: How much history to keep?
- **Cost**: Every token in context costs money
- **Relevance**: Old messages may not be relevant
- **Accuracy**: Old summaries may distort facts

Choose the strategy based on your use case: short-term conversational agents (sliding window) vs. long-term learning agents (vector DB + hybrid).
