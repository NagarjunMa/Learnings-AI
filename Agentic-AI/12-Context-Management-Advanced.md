# Context Management Advanced: Context Rot, Compression, Production Strategies

## What is Context Rot?

### Definition
Accumulated irrelevant, stale, or contradictory context degrading LLM reasoning quality over long conversations or multi-turn interactions.

### Root Cause
LLMs use attention mechanism: all tokens compete equally for model capacity. No built-in concept of "recency" or "relevance."

Example attention in a 10-turn conversation:
```
Turn 1: "I want to build a web app"
Turn 2-9: [discussions about architecture, database choices, etc.]
Turn 10: "What should I use for authentication?"

LLM attention:
- Token "web app" from turn 1 competes with "database choice" from turn 5
- Both carry equal weight despite turn 10's focus on "authentication"
- Older irrelevant context can drown out new intent
```

### How Context Rot Manifests

**Symptom 1: Wrong answers to clarifications**
```
User: "Actually, I changed my mind. Use React instead."
Assistant: [Ignores recent change, suggests Vue based on 5-turn-old discussion]
```

**Symptom 2: Ignoring recent system instructions**
```
System prompt (turn 1): "Always output JSON"
Turn 15: [User asks for formatted output]
Assistant: [Returns plaintext, attended to old instruction context, not recent request]
```

**Symptom 3: Hallucinating from old context**
```
Turn 1: "User database schema has id, name, email"
Turn 20: [User asks for "phone field"]
Assistant: [Hallucinates based on turn-1 schema which didn't have phone]
```

**Symptom 4: Contradictions in long agent runs**
```
Agent reasons at step 1: "User has budget of $100"
Agent reasons at step 10: "User has budget of $1000" [contradicts old context]
```

### Context Rot vs Context Length Exhaustion

**Context Length Exhaustion:**
- Problem: total tokens exceed model limit (Claude 3.5: 200K, GPT-4o: 128K)
- Solution: truncate or summarize
- Symptom: model stops responding or errors out

**Context Rot:**
- Problem: relevant info is buried in noise, attention diluted
- Solution: selective pruning, not length-based truncation
- Symptom: wrong answers even within context window

Example: 4K token conversation (far below 200K limit) can have context rot. 100K token conversation with structured state (LangGraph TypedDict) may not.

---

## Why Context Rot Happens

### Attention Mechanism Limitation
```
Transformer attention = softmax(Q * K^T / sqrt(d_k))

All tokens K compete equally in dot product.
No built-in recency bias or relevance weighting.
```

Old context with high token embedding similarity can "steal" attention.

### Increased Loss in Multi-Agent Systems
Agent A inserts context. Agent B inserts context. Agent C reads all. Noise compounds.

---

## Strategies to Prevent/Fix Context Rot

### 1. Sliding Window (Simple, Lossy)

**What:** Keep system prompt + last N messages, discard oldest.

```python
def sliding_window(messages, max_messages=10):
    system = [m for m in messages if m['role'] == 'system']
    others = [m for m in messages if m['role'] != 'system']
    
    # Keep system + last N
    return system + others[-max_messages:]
```

**Pros:**
- Simple, zero cost
- Stateless (no summarization model required)

**Cons:**
- Loses early context that's still relevant (e.g., initial instructions, user preferences from turn 1)
- Risk of re-explaining things already discussed

**When to use:**
- Short conversations (<50 turns)
- FAQ/support bot (early context less valuable over time)
- Budget-constrained (no spare tokens for summarization)

---

### 2. Summarization (Lossy Compression)

**What:** LLM summarizes old messages into compact summary, replace old messages with summary.

```python
async def summarize_old_messages(messages, summary_threshold=5):
    # Keep recent N messages
    recent = messages[-summary_threshold:]
    old = messages[:-summary_threshold]
    
    if len(old) < 2:
        return messages
    
    # Summarize old
    summary_text = old_context_str = format_messages(old)
    summary = await llm.apredict(
        "Summarize these messages in 2-3 sentences:\n" + summary_text
    )
    
    # Replace old with summary
    return [
        {"role": "system", "content": f"Previous context: {summary}"}
    ] + recent
```

**Pros:**
- Preserves essential facts from early conversation
- Reduces token count (200 turn history → 50 tokens)
- Prevents "re-explanation" loss of sliding window

**Cons:**
- Summarization itself is lossy (nuance lost)
- Cost: extra LLM call every N turns
- Latency: blocking or async?

**When to use:**
- Long conversations with factual accumulation (research, code review)
- Can amortize summarization cost (do every 10 turns, not every turn)
- Trade: 200 tokens saved vs 50 tokens spent on summarization = 150 net tokens saved

---

### 3. Context Compression (Selective Token Removal)

**What:** Remove low-signal tokens (stopwords, redundant phrases) without LLM-based summarization.

```python
def compress_context(text, keep_ratio=0.7):
    # Simple: remove stopwords, collapse multiple spaces
    import nltk
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    words = text.split()
    important = [w for w in words if w.lower() not in stop_words]
    
    # Aggressive: keep top-n tokens by TF-IDF or importance score
    # ...
    
    return ' '.join(important[:int(len(important) * keep_ratio)])
```

**Pros:**
- Zero cost (no LLM call)
- Deterministic (same input → same output)
- Preserves semantic tokens

**Cons:**
- Crude: loses grammatical structure
- Works only for certain text types (not code, structured data)
- Recall loss if important stopword is critical ("NOT", "DON'T")

**When to use:**
- Pre-processing old context before summarization (reduce summarization input)
- Filtering search results before injection (remove boilerplate)
- Not for critical data (code, instructions)

---

### 4. Episodic Injection (Vector-Based Retrieval)

**What:** Don't keep all old messages in context. Use vector search to retrieve only relevant old messages.

```python
# During conversation:
conversation_history = []

async def respond(user_input):
    # Retrieve relevant past messages (top-3) via embedding similarity
    relevant = await vector_store.search(user_input, top_k=3)
    
    # Inject only relevant old messages + recent messages
    context_messages = [
        recent_messages[-5:],  # Last 5
        relevant,               # Top 3 by similarity
    ]
    
    response = await llm.apredict(context_messages)
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})
    
    return response
```

**Pros:**
- Keep all history accessible without bloating context
- Relevance-driven: only inject what matters for current query
- Scales to 1000+ turn conversations

**Cons:**
- Requires vector embeddings of all past messages (memory, cost)
- Vector search retrieval can miss important context (semantic similarity ≠ current need)
- Adds latency (~10-50ms for search)

**When to use:**
- Long-running agents (100+ turns)
- Multi-session conversations (user A, then user B, retrieve A's relevant history)
- Knowledge-heavy: support bot, research assistant

---

### 5. Structured State (LangGraph TypedDict)

**What:** Instead of passing all conversation history, pass only relevant fields in state.

```python
# LangChain Chain: passes all history
chain = ConversationChain(memory=ConversationBufferMemory())

# LangGraph Agent: structured state
class AgentState(TypedDict):
    user_query: str           # Current turn only
    conversation_history: list[str]  # Last 5 only
    retrieved_facts: list[str]       # From vector search, not full history
    current_task: str         # What we're working on right now
    decision_context: str     # What led to this node

# State passed between nodes
def reasoning_node(state: AgentState) -> AgentState:
    # Only sees structured fields, not 100 turns of history
    # Cleaner attention, less context rot
    decision = reason(state['user_query'], state['retrieved_facts'])
    state['decision_context'] = decision
    return state
```

**Pros:**
- Explicitly manage what each node sees (no hidden context)
- LangGraph ensures only typed fields passed (no surprise history bleed)
- Reduces attention competition

**Cons:**
- Requires careful state design upfront
- More code than simple ConversationChain
- Risk: forgetting to include a critical field

**When to use:**
- Production agents (multi-step reasoning, tool calling)
- Multi-agent systems (prevent context pollution between agents)
- High-stakes: financial, medical, legal reasoning

---

### 6. System Prompt Refresh

**What:** Periodically re-assert system instructions to prevent drift.

```python
# Every N turns, inject fresh system prompt
if turn_count % 10 == 0:
    context_messages.insert(0, {
        "role": "system",
        "content": original_system_prompt
    })
```

**Pros:**
- Simple, free
- Counters instruction forgetting in long contexts

**Cons:**
- Redundant (system prompt already at top)
- Competes for attention like any other token
- Not a substitute for structured state

**When to use:**
- Safety-critical (instructions like "always verify before executing")
- Fallback when other strategies fail

---

## Token Budget Allocation Strategy

Production agents have finite context. Allocate strategically:

```
Total context = 200K tokens (Claude 3.5)

System prompt:          2K (1%)
Working memory (recent): 20K (10%)
Retrieved context:      50K (25%)
Retrieved facts (vector): 20K (10%)
Output buffer:          10K (5%)
RESERVED (safety):      98K (49%) <- prevents exhaustion
────────────────────────────
```

**Allocation principle:**
- System prompt: fixed, minimal
- Working memory: last N turns (5–10)
- Retrieved context: vector search + IVF, not all history
- Output buffer: reserved for model's response
- Never use >50% budget (leaves headroom for surprise context growth)

---

## Context Window Sizes Across Models

| Model | Window | Strategy |
|-------|--------|----------|
| Claude 3.5 Sonnet | 200K | Episodic injection, structured state, no compression needed |
| GPT-4o | 128K | Same as above, slightly tighter budget |
| Gemini 1.5 Pro | 1M | Episodic injection still recommended (even large windows have rot) |
| Llama 3.1 | 128K | Same as GPT-4o |
| Mistral Large | 128K | Same |

**Note:** Large context ≠ no context rot. Even 1M window can suffer rot. Use structured strategies regardless of window size.

---

## Production Context Architecture for Long-Running Agents

### Ideal architecture:

```
┌─────────────────────────────────────────┐
│ LangGraph Agent (StatefulGraph)         │
├─────────────────────────────────────────┤
│                                         │
│ Reasoning Node ────┐                    │
│                    ├─→ Tool Node        │
│ Planning Node  ────┤                    │
│                    └─→ Action Node      │
│                                         │
│ State (TypedDict):                      │
│  - user_query: str                      │
│  - recent_history: list[5 turns]        │
│  - retrieved_facts: list[vector search] │
│  - tool_results: list[recent tools]     │
│  - reasoning_trace: str (current step)  │
│                                         │
└─────────────────────────────────────────┘
         ↓
    Vector Store (all messages, indexed)
         ↓
    Checkpointer (Postgres, thread_id)
```

**Flow:**
1. User input arrives
2. Vector search retrieves top-5 relevant past messages
3. LangGraph passes typed state (recent + retrieved)
4. Each node sees only its needed context, not full history
5. Tool results added to state
6. State checkpointed for recovery

**Result:**
- Full history preserved (in vector store)
- Per-turn context limited (prevents rot)
- Scalable to 100+ turns without degradation

---

## Interview Talking Points

**"What is context rot and how do you prevent it in production agents?"**

Context rot: old, irrelevant tokens compete for model attention, degrading reasoning quality. Not a length problem (even short conversations can rot), a relevance problem.

Prevention depends on conversation length:
- Short (<20 turns): sliding window, system prompt refresh
- Medium (20-100 turns): episodic injection (vector search recent, retrieve relevant old)
- Long (100+ turns): structured state (LangGraph TypedDict), careful field design

Most important: use structured state in multi-node agents. Each node only sees what it needs.

**"How would you design context for a production LangGraph agent?"**

Use TypedDict state:
- `user_query`: current turn only (don't accumulate)
- `conversation_history`: last 5-10 turns (sliding window)
- `retrieved_facts`: vector search results (episodic, not full history)
- `task_state`: what we're doing right now (prevents attention drift)

Each node is explicit about context it consumes. No hidden history bleed. No context rot.

**"How do you handle a 1000-turn conversation?"**

1. Store all turns in vector DB, indexed by embedding
2. Retrieve top-k relevant turns via similarity to current query
3. Pass to agent: recent 5 turns + top 5 retrieved turns (total ~10 turns in context)
4. Never include all 1000 in context, even if window allows
5. Checkpointer stores full state (for recovery/debugging), but model attention is on ~10 most relevant turns

**"Context rot vs context exhaustion?"**

Exhaustion: total tokens exceed limit (e.g., 200K limit reached). Solution: truncate/summarize.

Rot: relevant info buried in noise, attention diluted. Can happen at 4K tokens (below 200K limit). Solution: selective pruning, relevance-based retrieval.

Different problems, different solutions.

