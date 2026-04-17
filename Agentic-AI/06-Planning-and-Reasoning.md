# Planning and Reasoning

## ReAct Pattern (Reasoning + Acting)

ReAct makes the agent's reasoning explicit and traceable. The LLM outputs structured steps: Thought → Action → Observation → Thought (repeat).

### Full Annotated Trace

```
User: "Find out how many people live in Tokyo and calculate the population density per square km"

Thought: I need to find:
1. Population of Tokyo
2. Area of Tokyo in km²
3. Calculate density = population / area

I should search for this information.

Action: web_search
Query: "Tokyo population 2024"

Observation: Tokyo population is approximately 37.4 million people

Thought: Good. Now I need the area of Tokyo.

Action: web_search
Query: "Tokyo area square kilometers"

Observation: Tokyo metropolitan area is about 13,960 km²

Thought: Now I can calculate the density.

Action: calculator
Operation: divide
a: 37400000
b: 13960

Observation: 2,679.54 people per km²

Thought: I have all the information. I can answer the user's question.

Final Answer: Tokyo has approximately 37.4 million people in an area of about 13,960 km², giving a population density of approximately 2,680 people per square km.
```

### Why ReAct Works

- **Transparency**: Every step is visible. Failures are traceable to specific reasoning or actions.
- **Self-correction**: LLM can see its own mistakes and fix them in the next iteration.
- **Token efficiency**: Explicit reasoning often prevents hallucination loops.

### ReAct Prompt Template

```
You are a helpful assistant. Follow this format exactly:

Thought: What do I need to do? What's my next step?
Action: [tool_name]
Input: [tool_input in JSON]
Observation: [tool result will be here]

Repeat until you have enough information, then:
Final Answer: [your response to the user]

Available tools:
- web_search: Search the web
- calculator: Perform math
- code_executor: Run Python code

User query: {user_query}
```

## Chain-of-Thought vs ReAct vs Plan-and-Execute

| Pattern | How It Works | Best For | Overhead |
|---------|------------|----------|----------|
| **Chain-of-Thought** | LLM thinks step-by-step before answering | Reasoning tasks (math, logic) | Minimal (1 LLM call) |
| **ReAct** | LLM thinks → acts → observes → repeats | Multi-step tasks with tools | Medium (multiple iterations) |
| **Plan-and-Execute** | LLM makes a plan first, then executes each step | Long-horizon tasks with clear steps | Medium to high |

### Example: Write a blog post

**Chain-of-Thought** (not enough):
```
LLM: "To write a good post, I should think about:
1. Topic research
2. Outline structure
3. Write each section
4. Proofread

Now let me write the post..."
```
Problem: LLM doesn't actually do research, just imagines it.

**ReAct** (better):
```
Thought: I need to research the topic first
Action: web_search
Input: {"query": "AI agents latest trends 2025"}
Observation: [real search results]

Thought: Now I'll outline the post
Action: code_executor
Input: {"code": "print(outline_structure)"}
...
```

**Plan-and-Execute** (best for this case):
```
Plan:
1. Research "AI agents" (search)
2. Create 5-section outline
3. Write intro (500 words)
4. Write body section 1 (800 words)
5. Write body section 2 (800 words)
6. Write conclusion (500 words)
7. Proofread

Now executing plan step by step...
```

## Tree-of-Thought: Branching Reasoning

For complex problems, explore multiple reasoning paths in parallel:

```
Question: "Should I start a business or pursue grad school?"

                    Root
                  /      \
            Business      Grad School
            /      \        /      \
         Startup   Join   PhD    Masters
         /    \    / \    / \     / \
       ...   ...  ... ... ... ...  ... ...
```

Instead of one linear reasoning path, the agent explores multiple branches:

```python
def tree_of_thought(question: str, depth: int = 3):
    """
    Explore multiple reasoning paths
    """
    paths = []

    for i in range(depth):
        for option in ["Start business", "Grad school"]:
            thought = llm.think(
                f"Option {i+1}: {option}. Next step?"
            )
            paths.append(thought)

    # Evaluate all paths and pick the best
    best_path = max(paths, key=lambda p: evaluate_reasoning(p))
    return best_path
```

**When to use**: High-stakes decisions, complex problems, ambiguous tasks.

## Reflexion (Self-Critique and Improvement)

The agent critiques its own output and retries with the feedback:

```
Initial Attempt:
Agent: "To solve this calculus problem, I'll use the power rule..."
Code: [generates solution]
Result: "Wrong answer"

Reflection:
Agent: "My approach was too simple. I need to use integration by parts instead."

Second Attempt:
Agent: "Let me use integration by parts..."
Code: [revised solution]
Result: "Correct!"
```

### Reflexion Loop

```python
def reflexion_loop(task: str, max_retries: int = 3):
    attempt = 1

    while attempt <= max_retries:
        # Generate solution
        solution = llm.generate(task)
        result = execute(solution)

        if is_correct(result):
            return solution

        # Critique
        critique = llm.critique(
            f"Task: {task}\nSolution: {solution}\nResult: {result}\nWhat went wrong?"
        )

        # Update task with feedback
        task = f"{task}\n\nPrevious attempt failed. Feedback: {critique}"
        attempt += 1

    return None  # Max retries exceeded
```

**Best for**: Tasks with measurable feedback (code generation, math, testing).

## Prompt Engineering for Planning

A good system prompt guides the agent toward effective planning:

### Bad System Prompt

```
You are an AI assistant. Help the user.
```

Problem: No guidance on reasoning structure or planning.

### Good System Prompt

```
You are a helpful AI assistant with access to tools. Follow these rules:

1. ALWAYS read the user's question carefully. Identify what information is needed.
2. PLAN: Before taking action, write out your plan in 2-3 steps.
3. EXECUTE: Call tools to gather information.
4. REASON: Analyze the results. Do you have enough info? If not, gather more.
5. ANSWER: Provide a clear, complete answer with sources.

Use the ReAct format:
Thought: [What I'm thinking]
Action: [Tool I'll use]
Observation: [Result]

Available tools: web_search, calculator, code_executor
```

## Reasoning Strategy × Task Type × Latency

| Reasoning Strategy | Best For | Latency | Cost |
|-------------------|----------|---------|------|
| **Simple prompt** | Straightforward Q&A | <1s | $0.01 |
| **Chain-of-Thought** | Logic, math, analysis | 1-2s | $0.02 |
| **ReAct** | Multi-step, tools | 5-15s | $0.20 |
| **Plan-and-Execute** | Long-horizon projects | 10-30s | $0.50 |
| **Tree-of-Thought** | Complex decisions, brainstorm | 15-60s | $1.00+ |
| **Reflexion** | Iterative refinement | 10-30s per cycle | $0.50/cycle |

## Practical Decision Tree

```
Task requires reasoning?
├─ No (simple lookup) → Use simple prompt
└─ Yes → Does it need tools?
    ├─ No (pure reasoning) → Use Chain-of-Thought
    └─ Yes → Is task scope fixed?
        ├─ Yes → Use Plan-and-Execute
        ├─ No → Use ReAct
        └─ High stakes? → Add Tree-of-Thought or Reflexion
```

## Common Pitfalls in Reasoning

### 1. Over-Planning
```
Bad: Agent spends 5 iterations planning a simple task, never executes
Fix: Set max_planning_iterations = 2, force execution after that
```

### 2. Hallucinated Reasoning
```
Bad: Agent thinks "I'll search for X" but X doesn't make sense
LLM output: "Thought: I'll calculate the emotional impact of colors"
Fix: Use schema validation, force tool_use in output format
```

### 3. Infinite Reasoning Loop
```
Bad: Thought → Action → Thought → Action → ... never ends
Fix: Max iterations limit, check for repeated actions
```

### 4. Reasoning Quality Dependent on Model
```
GPT-3.5: Reasoning often shallow, misses steps
GPT-4: More deliberate, better planning
Claude 3.5 Sonnet: Excellent reasoning, good planning
Fix: Choose the right model for the reasoning complexity
```

## When Reasoning is Overkill

```
Simple Q&A ("What's the capital of France?")
→ Just return answer, skip reasoning

Multi-step but deterministic (ETL pipeline)
→ Use traditional software logic, not reasoning

User needs speed (<200ms)
→ Skip reasoning, trade accuracy for latency

Cost is critical (processing 1M queries/day)
→ Use simple prompts, add reasoning only where needed
```

## Prompt Examples

### ReAct System Prompt

```
You are an agent that solves problems step by step.

Format your response EXACTLY like this:

Thought: [What you're thinking]
Action: [tool_name]
Input: [{"arg1": "value1", ...}]
Observation: [You don't write this; tools write it]

Repeat until solved, then:
Final Answer: [Your answer to the user]

Tools available:
{tool_definitions}

Always be specific about tool inputs. For example:
- Bad: Input: {"query": "stuff"}
- Good: Input: {"query": "population of Tokyo 2025"}
```

### Plan-and-Execute System Prompt

```
You are a planning agent. For complex tasks:

STEP 1 - PLAN:
List the exact steps you will take. Be specific:
- What tool will you use?
- What input will you provide?
- How will you use the result?

STEP 2 - EXECUTE:
Follow your plan exactly. After each step, verify the result makes sense.

STEP 3 - FINALIZE:
Review all results and provide a complete answer.

Now, here's the task: {user_task}
```

Remember: **Explicit reasoning prevents hallucination.** The more you make the agent "think out loud," the better its decisions.
