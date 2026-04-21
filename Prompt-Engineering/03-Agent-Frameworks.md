# Agent Frameworks — ReAct, Tree of Thoughts, Reflexion, PAL

Reasoning alone isn't enough. Agents need to **act** (use tools), **observe** (get feedback), and **reflect** (learn from mistakes). This file covers the frameworks that make agentic systems work.

---

## ReAct (Reason + Act)

**What:** Loop of Thought → Action → Observation → (repeat). Model reasons about what to do next, executes an action (tool call), observes the result, and repeats.

**Key paper:** "ReAct: Synergizing Reasoning and Acting in Language Models" — Yao et al. 2022 (arXiv:2210.11610)

**Why it matters:** Pure reasoning (CoT) hallucinates facts. Pure tool use (no reasoning) is trial-and-error. ReAct combines both.

**Measurable impact:**
- HotpotQA (multi-hop reasoning): 34.1% (CoT) → 77.9% (ReAct)
- FEVER (fact verification): 51.6% (CoT) → 82.3% (ReAct)
- Outperforms both pure reasoning and pure tool-use on knowledge-intensive tasks

**The Loop:**

```
Thought: "I need to find the population of Tokyo."
Action: search(query="Tokyo population 2024")
Observation: "Tokyo has approximately 14 million people (metro area)."

Thought: "Good. Now I need to compare to London."
Action: search(query="London population 2024")
Observation: "London has approximately 9 million people."

Thought: "Tokyo is larger. I can answer the question."
Final Answer: Tokyo (14M) is larger than London (9M).
```

**Implementation with Claude:**

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "search",
        "description": "Search the web for information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform arithmetic calculation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '5 + 3'"
                }
            },
            "required": ["expression"]
        }
    }
]

def execute_tool(tool_name: str, tool_input: dict):
    """Simulate tool execution."""
    if tool_name == "search":
        # Simulate search results
        results = {
            "Tokyo population": "Tokyo: 14M (metro), 37.4M (greater Tokyo area)",
            "London population": "London: 9M (city), 14.5M (metro area)",
        }
        for key, val in results.items():
            if key.lower() in tool_input["query"].lower():
                return val
        return "No results found."
    
    elif tool_name == "calculate":
        try:
            result = eval(tool_input["expression"])  # Dangerous in production, use ast.literal_eval
            return str(result)
        except:
            return "Calculation error."

def react_loop(user_query: str, max_iterations: int = 10):
    """ReAct loop: Thought → Action → Observation."""
    messages = [{"role": "user", "content": user_query}]
    
    for i in range(max_iterations):
        # Get response from Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check stop reason
        if response.stop_reason == "end_turn":
            # Model finished reasoning
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "No answer generated."
        
        elif response.stop_reason == "tool_use":
            # Model wants to use a tool
            # Collect all tool calls
            tool_calls = [block for block in response.content if block.type == "tool_use"]
            
            # Execute all tool calls
            tool_results = []
            for tool_call in tool_calls:
                observation = execute_tool(tool_call.name, tool_call.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": observation
                })
            
            # Append assistant response (with tool calls) and tool results to message history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        
        else:
            # Unexpected stop reason
            break
    
    return "Max iterations reached."

# Example
query = "Which is larger, Tokyo or London? Find populations and compare."
answer = react_loop(query)
print(answer)
# Output: "Tokyo is larger than London. Tokyo has a population of 14 million..."
```

**Key insight:** The `tool_use_id` is critical. When you call tool, you must include the exact ID in the `tool_result` response. This ties the observation back to the specific tool call.

**Why ReAct works:**
1. **Reasoning catches errors early:** "I need to search for..." prevents hallucination
2. **Observation grounds facts:** Real search results, not model memory
3. **Iteration allows correction:** If first search fails, try again with better query

**In production:**
- Set `max_iterations` to prevent infinite loops (default 5-10)
- Add cost limits (too many API calls for tools)
- Timeout after N seconds
- Log every thought-action-observation triple for debugging

---

## Tree of Thoughts (ToT)

**What:** Instead of a single linear reasoning chain (CoT), build a tree of candidate "thoughts" (intermediate steps). At each node, evaluate branches as "sure/maybe/impossible" and prune low-scoring ones. Explore promising branches with BFS or DFS.

**Key paper:** "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" — Yao et al. NeurIPS 2023

**Why:** Linear CoT sometimes hits dead ends. ToT backtracks and explores alternative paths.

**Measurable impact (Game of 24 — arithmetic puzzle):**
- IO (direct answer): 4% success
- CoT: 10% success
- ToT: 74% success

**The algorithm:**

```
1. Start with problem statement
2. Generate k candidate next steps (thoughts)
3. Evaluate each thought: "Is this leading to solution? (sure/maybe/impossible)"
4. Prune impossible branches
5. Expand promising branches (repeat step 2-4)
6. If leaf node is solution, return path
7. Backtrack if dead-end
```

**Implementation:**

```python
from collections import deque

def tree_of_thoughts(problem: str, max_depth: int = 5, branching_factor: int = 3):
    """Explore problem as a tree of thoughts."""
    
    class Node:
        def __init__(self, thought: str, depth: int, path: list):
            self.thought = thought
            self.depth = depth
            self.path = path  # Breadcrumb trail
            self.score = None
            self.children = []
    
    root = Node(problem, 0, [problem])
    queue = deque([root])
    solutions = []
    
    while queue and len(solutions) < 3:
        node = queue.popleft()
        
        if node.depth >= max_depth:
            continue
        
        # Generate next thoughts (alternatives)
        generate_prompt = f"""
You're solving: {problem}

Progress so far:
{' → '.join(node.path)}

Generate {branching_factor} alternative next steps. For each, provide:
- Thought: a concrete next step
- Score: (sure/maybe/impossible) — will this lead to solution?

Format:
Thought 1: ...
Score: sure

Thought 2: ...
Score: maybe
"""
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": generate_prompt}]
        )
        
        # Parse thoughts and scores
        lines = response.content[0].text.split('\n')
        thoughts = []
        current_thought = None
        
        for line in lines:
            if line.startswith("Thought"):
                current_thought = line.split(": ", 1)[1] if ": " in line else line
            elif line.startswith("Score"):
                score = line.split(": ", 1)[1].strip().lower() if ": " in line else ""
                if current_thought:
                    thoughts.append((current_thought, score))
        
        # Create child nodes, prune impossible
        for thought, score in thoughts:
            if score == "impossible":
                continue
            
            child = Node(thought, node.depth + 1, node.path + [thought])
            child.score = score
            node.children.append(child)
            
            # Check if this is a solution
            if "solution" in thought.lower() or "answer" in thought.lower():
                solutions.append(child.path)
            else:
                queue.append(child)
    
    if solutions:
        return " → ".join(solutions[0])
    else:
        return "No solution found."

# Example: Game of 24 (arrange 1, 2, 3, 4 with +/-/* to make 24)
problem = "Using 1, 2, 3, 4, make 24. You can +, -, *, /"
answer = tree_of_thoughts(problem, max_depth=4, branching_factor=3)
print(answer)
# Output: "Step 1: (1+2+3)*4 = 6*4 = 24. Solution found."
```

**Cost analysis:**
- CoT: 1 API call
- ToT: 5-20 API calls (depends on tree depth/branching)
- 10-20x more expensive, but 10-20x better on hard problems

**When to use:**
- Hard reasoning problems (math proofs, logic puzzles, planning)
- When you have time/budget
- Avoid for simple classification or real-time tasks

---

## Reflexion

**What:** Framework with three components:
1. **Actor:** Generates solution using ReAct or CoT
2. **Evaluator:** Scores solution quality (correct/incorrect, with reasoning)
3. **Self-Reflection:** Generates verbal critique, stores in episodic memory

On retry, model uses past reflections to improve.

**Key paper:** "Reflexion: Language Agents with Verbal Reinforcement Learning" — Shinn et al. 2023 (arXiv:2303.11366)

**Why:** RL usually requires fine-tuning. Reflexion learns from failures using only prompting + memory.

**Measurable impact:**
- AlfWorld (embodied reasoning): 34% (ReAct) → 71% (Reflexion) — agents complete more tasks after learning from failures
- HumanEval (coding): 67% → 86%
- MBPP (code): 61% → 74%

**Implementation:**

```python
def reflexion_loop(task: str, max_attempts: int = 3):
    """Actor → Evaluate → Reflect → Retry with memory."""
    
    episodic_memory = []  # Store past failures and reflections
    
    for attempt in range(max_attempts):
        # Stage 1: Actor generates solution
        actor_prompt = f"""
Task: {task}

Past attempts and lessons learned:
{chr(10).join(episodic_memory) if episodic_memory else 'None yet.'}

Now attempt the task:
"""
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": actor_prompt}]
        )
        solution = response.content[0].text
        
        # Stage 2: Evaluator scores solution
        evaluator_prompt = f"""
Task: {task}

Proposed solution:
{solution}

Is this solution correct? Rate: correct/incorrect/partial.
If incorrect, what's wrong?
"""
        
        eval_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": evaluator_prompt}]
        )
        evaluation = eval_response.content[0].text
        
        # Check if correct
        if "correct" in evaluation.lower() and "incorrect" not in evaluation.lower():
            return solution
        
        # Stage 3: Self-Reflection generates critique
        reflection_prompt = f"""
My solution:
{solution}

Evaluation:
{evaluation}

What did I do wrong? What should I try differently?
Be specific. This is for my next attempt.
"""
        
        reflection_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": reflection_prompt}]
        )
        reflection = reflection_response.content[0].text
        
        # Store in episodic memory
        episodic_memory.append(f"Attempt {attempt + 1}: {reflection}")
    
    return f"Failed after {max_attempts} attempts."

# Example
task = "Write Python code to check if a number is prime."
code = reflexion_loop(task, max_attempts=3)
print(code)
```

**Key advantage:** Model learns from mistakes without fine-tuning. Each reflection is a lesson stored in memory.

**In production:**
- Use evaluator that's reliable (could use test cases instead of LLM evaluation)
- Store reflections in a database for later analysis
- Add diversity: different actor models, different evaluators

---

## Program-Aided Language Models (PAL)

**What:** For math/logic problems, don't ask the LLM to compute the answer. Ask it to **write code** that solves the problem, then execute.

**Why:** Models are bad at arithmetic. Code execution is deterministic. Model handles the algorithm, runtime handles the math.

**Measurable impact (GSM8K math word problems):**
- CoT (direct): 40.7%
- PAL (generate code + execute): 57.3%
- PAL + self-consistency: 84.5%

**Implementation:**

```python
import subprocess
import tempfile

def pal_solve(problem: str):
    """Generate code, execute, return result."""
    
    # Stage 1: Generate Python code
    code_prompt = f"""
Problem: {problem}

Write Python code to solve this problem. 
Print the final answer.

Code:
"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": code_prompt}]
    )
    
    code = response.content[0].text
    
    # Stage 2: Execute code
    try:
        # Extract Python code from response (may contain markdown)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Run in isolated environment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    
    except Exception as e:
        return f"Execution failed: {str(e)}"

# Example
problem = "A store has 100 items. 20% are sold. 10% of remaining are defective. How many are usable?"
answer = pal_solve(problem)
print(answer)  # Output: 72
```

**Safety note:** Never execute untrusted code. Use sandboxing (Docker, restricted Python environments).

**When to use:**
- Math word problems
- Any problem with numerical computation
- Logic puzzles where computation is bottleneck

---

## Comparison Table

| Framework | Cost | Accuracy | Latency | Best For | Complexity |
|---|---|---|---|---|---|
| ReAct | 2-10x | +30-50% | Medium | Knowledge-intensive, tool-use | Medium |
| ToT | 10-20x | +30-70% | High | Hard reasoning, planning | High |
| Reflexion | 3-5x per retry | +10-30% per retry | High | Learning from failures | Medium |
| PAL | 1x (code) + exec | +20-40% | Low | Math, logic | Low |

**Decision tree:**
```
Do you need external tools (search, APIs)?
  ├─ Yes → ReAct
  └─ No → Is problem hard (game of 24, proofs)?
         ├─ Yes → ToT
         ├─ No → Is it math/logic?
         │      ├─ Yes → PAL
         │      └─ No → CoT (from file 02)
         └─ Do you want self-improvement?
            ├─ Yes → Reflexion
            └─ No → Done
```

---

## Production Checklist

- [ ] **Tool definitions:** All tool schemas have clear `name`, `description`, `input_schema`
- [ ] **Tool execution:** Tool IDs matched in responses (prevent mismatches)
- [ ] **Max iterations:** Set limit to prevent infinite loops
- [ ] **Cost tracking:** Log API calls, set budget alerts
- [ ] **Error handling:** Graceful fallback if tool fails
- [ ] **Evaluator quality:** If using LLM evaluation, validate on known cases first
- [ ] **Memory storage:** If using Reflexion, persist episodic memory
- [ ] **Sandboxing:** If executing code (PAL), use isolated environment

