# Provider Playbook — Claude, OpenAI, Gemini

Techniques are mostly model-agnostic. Implementation details are not. This file covers provider-specific nuances that unlock full power.

---

## Claude (Anthropic)

### Effort Parameter (Opus 4.7, Sonnet 4.6)

**What:** Control reasoning depth with `budget_tokens` (via `max_tokens`) or system-level "effort" (internal parameter).

**Levels:**
- **low:** 5K-10K internal reasoning tokens. Fast, good for classification
- **medium:** 10K-25K. Balanced. Recommended default
- **high:** 25K-50K. Deep reasoning. Use for complex problems
- **xhigh:** 50K-100K. Very deep. Use for coding/agentic tasks
- **max:** Unrestricted. Use max_tokens 64K+. For hardest problems

**When to use each:**
```
Classification task → low or medium
Math/logic → high
Coding → xhigh (let model think more)
Agentic workflow → xhigh (multi-step planning)
Very hard problem → max (set max_tokens=64000)
```

**Implementation:**

```python
# Low effort (fast, for simple tasks)
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,  # Implicit low effort
    messages=[{"role": "user", "content": "Classify this: ..."}]
)

# High effort (deep reasoning)
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,  # Signals high effort
    messages=[{"role": "user", "content": "Solve this complex logic puzzle: ..."}]
)

# Opus 4.7: Adaptive thinking (automatic effort selection)
# No explicit effort param needed; model auto-allocates reasoning budget
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=8192,  # Model decides internal effort
    messages=[...]
)
```

**Measured impact:**
- Low effort on hard problem: 30% accuracy
- High effort on same problem: 80% accuracy
- Cost: +2-3x for high effort

### Adaptive Thinking (Opus 4.7 only)

**What:** Model automatically decides when to think deeply. No explicit trigger needed.

**Why:** Some problems need thinking, others don't. Adaptive avoids wasted thinking on easy problems.

**Implementation:**

```python
# Just set max_tokens high, model handles rest
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=8192,  # Give room for thinking
    messages=[{"role": "user", "content": "Solve this problem"}]
)

# Response may have thinking tags
for block in response.content:
    if block.type == "thinking":
        print(f"Internal reasoning: {block.thinking}")
    elif block.type == "text":
        print(f"Answer: {block.text}")
```

**Cost:** Only pay for tokens model actually uses (thinking + text).

### Structured Outputs (Claude 3.5 Sonnet+)

**What:** Guarantee JSON output matching a schema. Model cannot deviate.

**Why:** No parsing surprises. Works with complex nested schemas.

**Implementation:**

```python
from anthropic import Anthropic

client = Anthropic()

# Define schema
class ExtractedData(BaseModel):
    name: str
    age: int
    occupation: str
    skills: list[str]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Extract info from this text: John, 28, Software Engineer, Python, Go"}
    ]
)

# No direct structured output yet in Claude (as of April 2026)
# Workaround: Use JSON mode in system prompt
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You must respond with ONLY valid JSON matching: {\"name\": \"...\", \"age\": int, \"occupation\": \"...\", \"skills\": []}",
    messages=[
        {"role": "user", "content": "Extract from: John, 28, Software Engineer, Python, Go"}
    ]
)
```

### Long-Context Best Practices (Anthropic)

**1. Document placement:** Put long documents at TOP, queries at BOTTOM.

**Measured impact:** +30% quality on complex multi-document retrieval.

**Why:** Model's attention is freshest on most recent tokens.

```python
# Bad order
prompt = f"""
Question: {question}  ← Short query at top

Document 1: {very_long_doc1}
Document 2: {very_long_doc2}
Document 3: {very_long_doc3}
"""

# Good order
prompt = f"""
Document 1: {very_long_doc1}
Document 2: {very_long_doc2}
Document 3: {very_long_doc3}

Question: {question}  ← Query at bottom
"""
```

**2. Grounding in quotes:** Ask model to quote relevant passages before reasoning.

```python
quote_and_reason_prompt = f"""
Documents:
{all_documents}

Question: {question}

First, quote the most relevant passage from the documents.
Then, answer the question based on that passage.
"""
```

**3. Batch processing:** Use batch API for 50% cost reduction (slower, but cheaper).

```python
# Use batch API for non-urgent work
# (Anthropic Messages Batch API — async processing)
```

### Parallel Tool Calls

**What:** Execute independent tool calls simultaneously, not sequentially.

**How:** Add explicit instruction in system prompt.

```python
system_prompt = """
When you intend to call multiple tools and there are no dependencies between them, make all independent calls in parallel.

Example:
If you need to search for "Paris" and search for "London", don't search for Paris, wait for result, then search London.
Instead: search for both Paris and London in one step.
"""

# Claude will batch independent tool calls
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system=system_prompt,
    tools=tools,
    messages=[...]
)
```

### Subagent Control (Opus 4.7)

**What:** Primary agent can spawn sub-agents for parallel/isolated work.

**When to spawn:**
- Fanning out across 5+ independent items
- Reading 3+ unrelated files
- Parallel branches in decision tree

**When NOT to spawn:**
- Single task you can complete directly
- Sequential dependencies (agent1 output feeds agent2)

```python
decision_prompt = """
Spawn multiple subagents when:
1. Working on 5+ independent items (array processing)
2. Reading/analyzing multiple unrelated files
3. Parallel decision tree branches

Do NOT spawn for:
- Sequential workflows (output of step 1 feeds step 2)
- Single-step work you can do directly
"""
```

### Common Pitfalls

**Pitfall 1: Avoid word "think" in prompts (Claude 3.5)**
```python
# Bad (triggers over-thinking on 3.5)
prompt = "Think carefully about this problem"

# Good
prompt = "Work through this systematically" or "Reason through this step-by-step"
```

**Pitfall 2: Prefilling responses (deprecated in 4.6+)**
```python
# Old Claude 3.0 pattern (DON'T use on 4.6+)
messages=[
    ...,
    {"role": "assistant", "content": "Let me work through this:\n1."}  # Prefill — don't use
]

# Use explicitly with Opus 4.7 adaptive thinking instead
```

**Pitfall 3: XML tag inconsistency**
```python
# Bad
prompt = """
<instructions> ... </instructions>
<context>...</context>
<question>...</question>
"""

# Good (close all tags explicitly)
prompt = """
<instructions>...</instructions>

<context>...</context>

<question>...</question>
"""
```

---

## OpenAI (GPT series)

### 6 Official Strategies

1. **Write clear, specific instructions**
   - Be explicit about desired output format
   - Example: "Output ONLY JSON, no markdown"

2. **Provide reference text**
   - Reduces hallucinations by grounding answers
   - Impact: 15-30% accuracy improvement on factual tasks

3. **Split complex tasks into subtasks**
   - Equivalent to prompt chaining
   - Use CoT for multi-step reasoning

4. **Give models time to think**
   - Equivalent to CoT: "Let's think step by step"
   - Impact: 20-40% on reasoning tasks

5. **Use external tools**
   - Equivalent to ReAct pattern
   - Function calling for tool-use loops

6. **Test changes systematically**
   - A/B test prompt versions
   - Measure on 50+ examples

### Structured Outputs API

**What:** Guarantee JSON output matching a schema (like Claude's structured outputs).

```python
from openai import OpenAI

client = OpenAI()

response = client.beta.messages.create(
    model="gpt-4-turbo",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Extract company info"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "CompanyInfo",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "industry": {"type": "string"},
                    "employees": {"type": "integer"}
                },
                "required": ["name", "industry", "employees"]
            }
        }
    }
)
```

**Benefit:** JSON guaranteed valid and matches schema. No parsing errors.

### Prompt Caching (gpt-4-turbo, gpt-4o)

**What:** Cache the system prompt and first N messages. Reuse cache, pay 10% on cache hit.

**How:** Automatically done by API. No explicit control needed.

```python
# Cache automatically triggered on repeated prompts
large_system_prompt = "You are an expert in financial regulation..." * 1000  # Big prompt

# First call: cache created
response1 = client.messages.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "user", "content": "Question 1"}
    ],
    system=large_system_prompt
)

# Second call with same system prompt: 90% cost savings on cache hit
response2 = client.messages.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "user", "content": "Question 2"}  # Different question
    ],
    system=large_system_prompt  # Same system prompt → cache hit
)
```

**Measured impact:** 90% savings on cache hit (pay 10% of input tokens).

### Reasoning Effort (o-series)

**What:** Control reasoning depth on o1/o3 models (analogous to Claude's effort).

```python
# o1 (default medium thinking)
response = client.messages.create(
    model="o1-preview",
    messages=[...],
    temperature=1  # o-series uses temp=1 always
)

# o3 (control thinking effort)
response = client.messages.create(
    model="o3-mini",
    messages=[...],
    temperature=1,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Control reasoning depth
    }
)
```

---

## Google Gemini

### Temperature Default

**Gemini 3:** Temperature defaults to 1.0 (vs 0.7 for Claude/GPT)

**Implication:** Gemini's "balanced" mode is more creative than competitors.

```python
# For classification (deterministic), override to 0.0
response = genai.generate_content(
    "Classify this sentiment",
    generation_config=genai.types.GenerationConfig(
        temperature=0.0,  # Override default 1.0
        top_p=0.9
    )
)
```

### Built-in Grounding Tools

**What:** Gemini has native Google Search and code execution.

**Why:** Use instead of building ReAct agent with separate search tool.

```python
import google.generativeai as genai

client = genai.GenerativeModel('gemini-3-5-pro')

# Enable Google Search grounding
tools = [genai.tools.GoogleSearch()]

response = client.generate_content(
    "What's the latest news about AI?",
    tools=tools
)

# Gemini automatically grounds answer in real-time search results
```

**Advantage:** No need for separate search API; integrated in model.

### Internal Reasoning (Don't request explicit CoT)

**What:** Gemini 3 reasons internally automatically. Don't ask for explicit CoT output.

**Bad:**
```python
prompt = "Solve this step by step: ..."
# Gemini outputs reasoning steps in response (verbose)
```

**Good:**
```python
prompt = "Think very hard before answering: ..."
# Gemini reasons internally, outputs only final answer (concise)
```

**Why:** Gemini's internal reasoning is invisible. Asking for explicit output wastes tokens without benefit.

### Markdown Headers for Structure

**What:** Use Markdown structure instead of XML tags.

```python
prompt = """
# Task
Classify customer sentiment.

## Examples
- "I love this!" → positive
- "Terrible product" → negative

## Constraints
- If confidence < 0.7, return "uncertain"
- No explaining, just output category

## Input
{customer_feedback}
"""
```

**Why:** Gemini's training heavily uses Markdown.

### Multimodal Prompting

**Built-in:** Gemini natively handles text + images + video.

```python
import base64

# Load image
with open("chart.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.generate_content([
    {
        "type": "image",
        "image": {
            "mime_type": "image/png",
            "data": image_data
        }
    },
    {
        "type": "text",
        "text": "Analyze this chart and summarize trends"
    }
])
```

**Best for:** Document analysis, images, diagrams, charts.

---

## Model-Agnostic vs Provider-Specific Comparison

| Technique | Model-Agnostic | Provider-Specific |
|---|---|---|
| Zero-shot, few-shot | ✅ Yes | ❌ No |
| CoT (basic) | ✅ Yes | ❌ No |
| ReAct tool-use | ✅ Conceptually | 🟡 API syntax differs |
| XML structure | 🟡 Mostly (Anthropic docs it) | ❌ No |
| Prompt caching | ❌ No | ✅ Anthropic/OpenAI only |
| Structured outputs API | ❌ No | ✅ Anthropic/OpenAI only |
| Effort/thinking parameter | ✅ Concept | ❌ Implementation per-provider |
| Grounding tools | ✅ Concept | ❌ Gemini Google Search only |
| Long-context (docs at top) | ✅ Yes | 🟡 Impact varies by model |
| Parallel tool calls | ✅ Conceptually | 🟡 API-level per-provider |
| Function calling | ✅ Concept | ❌ Syntax differs (Anthropic tools vs OpenAI functions) |
| Temperature/top-p tuning | ✅ Yes | 🟡 Different defaults (Gemini temp=1.0 vs others 0.7) |

---

## Quick Reference by Provider

### Claude (Anthropic)
```
✅ Use: XML tags, long-context (docs first), parallel tool calls, structured output
✅ Effort: Set via max_tokens (512=low, 4096=high, 64000=max)
❌ Avoid: Word "think" in 3.5, prefill responses in 4.6+
🎯 Best for: Complex reasoning, agentic systems, long contexts
```

### OpenAI (GPT)
```
✅ Use: Function calling for tools, prompt caching, structured outputs
✅ Reasoning: o-series with thinking budget
✅ 6 strategies: clear instructions, reference text, subtasks, time to think, tools, test systematically
❌ Default: No long-context optimization, standard temperature 0.7
🎯 Best for: Production, cost-sensitive, function calling
```

### Gemini (Google)
```
✅ Use: Built-in Google Search, multimodal (images/video), Markdown structure
✅ Reasoning: Auto-internal, trigger with "Think very hard"
❌ Avoid: Explicit CoT output requests, high temperature for classification
⚠️ Note: Temperature defaults 1.0 (not 0.7)
🎯 Best for: Multimodal tasks, real-time grounding, document analysis
```

---

## Production Checklist

- [ ] **Provider chosen:** Know if using Claude, OpenAI, or Gemini
- [ ] **Model selected:** Pick model matching task (3.5 for speed, 4x for power)
- [ ] **Temperature set:** 0.0 for deterministic, 0.7-1.0 for creative
- [ ] **Effort tuned:** If Claude, set max_tokens appropriate to task complexity
- [ ] **Long-context:** If >4K tokens, document at top, query at bottom
- [ ] **Tools used:** If Claude, use tool_use. If OpenAI, use function calling.
- [ ] **Cost tracked:** Log tokens, monitor per-request cost
- [ ] **Error handling:** Graceful fallback on rate limits, timeouts

