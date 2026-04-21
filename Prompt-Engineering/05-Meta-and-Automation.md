# Meta-Prompting and Automated Prompt Engineering

Manual prompting is slow. APE (Automatic Prompt Engineer) beats human-written prompts. DSPy autotunes parameters. This file covers techniques that engineer prompts automatically or structurally.

---

## Meta-Prompting

**What:** Define the reusable structural **template** for a reasoning process, rather than providing content-specific examples.

**Five properties:**
1. **Structure-oriented:** Focuses on process (how to think), not content (what to know)
2. **Syntax-focused:** Uses abstract, symbolic syntax (placeholders, operators)
3. **Abstract examples:** Uses generic problem classes, not concrete instances
4. **Versatile:** One template works across many tasks
5. **Categorical:** Type-theoretic — organizes thinking by problem structure

**Why it works:** Humans use mental templates. "Multi-step reasoning" is a template. "Process of elimination" is a template. Meta-prompting extracts these and makes them explicit.

**Example 1 — Symbolic Reasoning Template:**

```
For any problem P with variables V:
1. Identify what you know: Extract all given facts into set K = {fact1, fact2, ...}
2. Identify what you want: State goal G clearly
3. Find connections: For each fact in K, ask "Does this relate to G?"
4. Build chain: Connect related facts with logical operators (AND, OR, IF-THEN)
5. Evaluate: Check if chain leads to G

Apply this process to: {problem}
```

This template works for:
- Logic puzzles
- Theorem proving
- Detective stories
- Legal reasoning

**Example 2 — Mathematical Problem-Solving Template:**

```
For any math problem:
1. Translate: Convert words to symbols/equations
   word problem → {equation1, equation2, ...}
2. Identify unknowns: What variable(s) need solving?
3. Select operations: Which operations (±*/) connect unknowns to knowns?
4. Compute: Execute operations step-by-step
5. Verify: Does the answer make sense?

Apply to: {problem}
```

**Implementation:**

```python
def meta_prompting_solve(problem: str, template_category: str):
    """Solve using abstract structural template."""
    
    templates = {
        "logic_puzzle": """
For any logic puzzle:
1. Identify variables and constraints
2. List all given facts
3. For each variable, test each possibility against constraints
4. Eliminate impossibilities
5. Repeat until one possibility remains
""",
        "math": """
For any math problem:
1. Translate words to equations
2. Identify unknowns
3. Rearrange to isolate unknowns
4. Compute numerically
5. Check reasonableness
""",
        "multi_step": """
For any multi-step process:
1. Break into smallest sub-goals
2. Order by dependency (can step X happen before Y?)
3. Solve each sub-goal
4. Verify each sub-goal result
5. Combine results
"""
    }
    
    template = templates.get(template_category, templates["multi_step"])
    
    prompt = f"""{template}

Now apply this template to:
{problem}

Work through each step:
"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Usage
problem = "A farmer has 100 meters of fencing. He wants to enclose a rectangular field with max area. What are the dimensions?"
answer = meta_prompting_solve(problem, "math")
print(answer)
```

**Advantage over few-shot:** Meta-prompting uses 0-2 examples (abstract) vs 5-10 (few-shot). More token-efficient.

**Advantage over CoT:** Meta-prompting shows **structure** explicitly. CoT leaves structure implicit.

---

## Automatic Prompt Engineer (APE)

**What:** Use an LLM to automatically generate prompt candidates, test on target LLM, select best.

**Key paper:** "Large Language Models as Optimizers" — Zhou et al. 2022 (arXiv:2211.01910)

**Key discovery:** APE found *"Let's work this out in a step by step way to be sure we have the right answer"* — outperformed human-written *"Let's think step by step"* on multiple benchmarks.

**Implication:** Humans miss optimal prompts. Exhaustive search finds better ones.

**Algorithm:**

```
1. Generate diverse prompt candidates (LLM or templates)
2. Test each candidate on target LLM with labeled data
3. Evaluate: Calculate accuracy/F1/loss
4. Select top K candidates
5. (Optional) Iterate: Mutate top candidates, re-test, select best
```

**Implementation:**

```python
def auto_prompt_engineer(
    task_description: str,
    labeled_examples: list[dict],  # [{"input": "...", "output": "..."}, ...]
    num_candidates: int = 10,
    num_rounds: int = 3
):
    """Generate and optimize prompts automatically."""
    
    def evaluate_prompt(prompt_instruction: str, examples: list[dict], k_test: int = 5):
        """Test prompt on k examples, return accuracy."""
        correct = 0
        
        for example in examples[:k_test]:
            full_prompt = f"""{prompt_instruction}

Input: {example['input']}

Output:"""
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            output = response.content[0].text.strip()
            expected = example['output'].strip()
            
            if output.lower() == expected.lower():
                correct += 1
        
        return correct / k_test
    
    # Round 1: Generate candidate prompts
    generation_prompt = f"""
Task: {task_description}

Examples:
{chr(10).join([f"Input: {ex['input']}, Output: {ex['output']}" for ex in labeled_examples[:3]])}

Generate {num_candidates} diverse instruction prompts that would help an LLM solve this task.
Each instruction should be 1-2 sentences.

Prompts:
"""
    
    gen_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": generation_prompt}]
    )
    
    candidates = gen_response.content[0].text.split('\n')
    candidates = [c.strip() for c in candidates if c.strip() and len(c) > 10][:num_candidates]
    
    best_prompt = None
    best_score = -1
    
    for round_num in range(num_rounds):
        # Evaluate all candidates
        scores = []
        for candidate in candidates:
            score = evaluate_prompt(candidate, labeled_examples, k_test=3)
            scores.append((score, candidate))
        
        # Select top K
        scores.sort(reverse=True)
        print(f"Round {round_num + 1} best score: {scores[0][0]:.2f}")
        print(f"Best prompt: {scores[0][1]}")
        
        best_prompt = scores[0][1]
        best_score = scores[0][0]
        
        if best_score >= 0.95:  # Good enough
            break
        
        # Mutate top candidates for next round
        if round_num < num_rounds - 1:
            top_3 = [s[1] for s in scores[:3]]
            
            mutation_prompt = f"""
These prompts got good scores:
{chr(10).join(top_3)}

Generate {num_candidates} mutations/variations of these.
Try different phrasings, different instruction styles, different levels of detail.

Mutated prompts:
"""
            
            mutation_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": mutation_prompt}]
            )
            
            new_candidates = mutation_response.content[0].text.split('\n')
            candidates = [c.strip() for c in new_candidates if c.strip() and len(c) > 10][:num_candidates]
    
    return {
        "best_prompt": best_prompt,
        "best_score": best_score
    }

# Usage
examples = [
    {"input": "classify(positive review)", "output": "positive"},
    {"input": "classify(negative review)", "output": "negative"},
    {"input": "classify(mixed review)", "output": "mixed"}
]

result = auto_prompt_engineer("Sentiment classification", examples, num_candidates=10, num_rounds=2)
print(f"Optimal prompt:\n{result['best_prompt']}")
print(f"Score: {result['best_score']:.2f}")
```

**Cost:** High (generates N candidates, evaluates each on K examples). Amortized over many uses, justified.

**When to use:**
- Task is critical (needs highest accuracy)
- Have labeled examples (>50)
- Time-insensitive (batch optimization)

---

## DSPy — Stanford's Automated Prompt Framework

**What:** Framework that treats prompts as differentiable parameters. Automatically optimizes prompts given labeled data using gradient-based search or reinforcement learning.

**Key claim:** Achieved F1 ≈ 0.6 in 10 minutes vs 20 hours of manual prompt engineering.

**Mental model:** Traditional ML:
```
data + architecture + optimization = trained model
```

DSPy:
```
data + prompt structure + DSPy optimizer = optimized prompt + model
```

**Installation:**

```bash
pip install dspy-ai
```

**Simple example:**

```python
import dspy

# Define a task using DSPy signature
class GenerateSummary(dspy.Signature):
    """Generate a short summary of the document."""
    document = dspy.InputField(desc="The document to summarize")
    summary = dspy.OutputField(desc="A 1-2 sentence summary")

# Define a module
class SummaryGenerator(dspy.ChainOfThought):
    def __init__(self):
        super().__init__(GenerateSummary)

# Load examples
train_examples = [
    dspy.Example(
        document="Paris is the capital of France...",
        summary="Paris is France's capital city."
    ),
    # ... more examples
]

# Optimize prompts
optimizer = dspy.BootstrapFewShot(
    metric=lambda pred, gold: gold.summary.lower() in pred.summary.lower()
)

summarizer = SummaryGenerator()
optimized_summarizer = optimizer.compile(
    student=summarizer,
    trainset=train_examples
)

# Use optimized version
result = optimized_summarizer(document="Some document...")
print(result.summary)
```

**What DSPy does:**
1. Tries different numbers of examples (0-5)
2. Tries different example selections
3. Tries different prompt phrasings
4. Tests each configuration on dev set
5. Returns best configuration

**Trade-off:** Requires labeled examples (50-100), but saves manual tuning.

---

## Active-Prompt

**What:** Dynamically select most informative/uncertain examples for few-shot prompting.

**Why:** Not all examples equally useful. Uncertain cases teach more than easy cases.

**Algorithm:**
1. Compute model's uncertainty on all examples
2. Select examples with highest uncertainty
3. Use those as few-shot demonstrations

**Implementation (simplified):**

```python
def active_prompt_selection(test_examples: list[str], num_examples: int = 5):
    """Select most uncertain examples as few-shot demos."""
    
    uncertainties = []
    
    for example in test_examples:
        # Get model's self-rated confidence
        confidence_prompt = f"""
How confident are you in classifying: "{example}"?
Rate 0-10 (0=very uncertain, 10=very confident).
"""
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": confidence_prompt}]
        )
        
        try:
            confidence = int(response.content[0].text.strip())
        except:
            confidence = 5
        
        uncertainty = 10 - confidence  # Invert
        uncertainties.append((uncertainty, example))
    
    # Sort by uncertainty, pick top
    uncertainties.sort(reverse=True)
    uncertain_examples = [ex for _, ex in uncertainties[:num_examples]]
    
    return uncertain_examples

# Use selected examples in few-shot
uncertain = active_prompt_selection(test_data, num_examples=5)

few_shot_prompt = f"""
Examples (carefully selected):
{chr(10).join([f"- {ex}" for ex in uncertain])}

Now classify:
"""
```

---

## Interview Pattern Prompting

**What:** Prompt the LLM to ask clarifying questions before attempting to solve ambiguous tasks.

**Why:** Many tasks are underspecified. Asking questions prevents wasted effort.

**Implementation:**

```python
def interview_pattern(task: str):
    """Model asks clarifying questions before solving."""
    
    # Stage 1: Ask clarifying questions
    clarify_prompt = f"""
Task: {task}

Before solving, ask 3-5 clarifying questions.
These should disambiguate the task.

Questions:
"""
    
    questions_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": clarify_prompt}]
    )
    
    questions = questions_response.content[0].text
    print(f"Questions:\n{questions}")
    
    # User answers (simulated)
    user_answers = """
1. Financial data for 2024
2. Revenue, profit, growth
3. Monthly aggregation
"""
    
    # Stage 2: Solve with clarification
    solve_prompt = f"""
Task: {task}

Clarifications:
{user_answers}

Now solve the task with these details in mind:
"""
    
    solution_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": solve_prompt}]
    )
    
    return solution_response.content[0].text
```

---

## Comparison Table

| Technique | Cost | Effort | When to Use |
|---|---|---|---|
| Meta-prompting | 1x | Low (write template) | Every task type |
| Manual prompting | 1x | High (tune prompts) | One-off tasks |
| APE | 10-50x | Medium (set up + run) | Critical task, have examples |
| DSPy | 5-20x | Medium (label examples) | Production pipelines |
| Active-Prompt | 2x (select + use) | Low | When examples expensive to label |
| Interview Pattern | 2x (ask + solve) | Low | Ambiguous tasks |

---

## Production Checklist

- [ ] **Start manual:** Hand-craft 1-2 good examples first
- [ ] **Measure baseline:** Know zero-shot accuracy
- [ ] **Meta-prompt:** If task has clear structure, extract template
- [ ] **DSPy:** If have 50+ labeled examples, auto-optimize
- [ ] **APE:** If accuracy plateaus and budget allows
- [ ] **Active-prompt:** If labeling examples is expensive
- [ ] **Version control:** Track prompts in git (text files, not notebooks)

