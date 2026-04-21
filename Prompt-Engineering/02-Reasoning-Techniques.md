# Reasoning Techniques — CoT, Self-Consistency, Generate Knowledge, Least-to-Most

Vanilla prompt gets 30% accuracy on math. Add "Let's think step by step" → 95%+ accuracy. Reasoning is the lever.

---

## Zero-Shot Chain-of-Thought (CoT)

**What:** Append "Let's think step by step" (or similar) to ANY prompt. Model outputs reasoning before answer.

**Key paper:** "Large Language Models are Zero-Shot Reasoners" — Kojima et al. 2022 (arXiv:2205.11916)

**Why it works:** Forces intermediate token generation. Without CoT, model tries to leap directly to answer, skipping steps where it could self-correct.

**Measurable impact:**
- MultiArith dataset: 17.7% (direct) → 78.7% (zero-shot CoT)
- GSM8K: 10.7% → 40.7%
- SVAMP: 7.8% → 83.0%

**Implementation:**

```python
# Without CoT
prompt = "Q: A store has 100 items. 20% sold, 10% of remaining defective. Usable?"
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": prompt}]
)
# May output: "72" (often wrong)

# With zero-shot CoT
prompt = """Q: A store has 100 items. 20% sold, 10% of remaining defective. Usable?

Let's think step by step."""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": prompt}]
)
# Output: "Items sold: 100*0.2=20. Remaining: 100-20=80. Defective: 80*0.1=8. Usable: 80-8=72."
```

**APE discovery (2022):** Automatic Prompt Engineer tested 10,000+ trigger phrases. Found: *"Let's work this out in a step by step way to be sure we have the right answer"* outperformed the human-written "Let's think step by step" on some benchmarks.

Implication: Exact wording matters, but the concept is robust.

**Triggers that work:**
- "Let's think step by step."
- "Think carefully."
- "Break down the problem."
- "Work through this systematically."

Model-agnostic. Works on GPT, Claude, Gemini, LLaMA.

---

## Few-Shot Chain-of-Thought

**What:** Provide 2-5 examples that INCLUDE reasoning steps, not just input→output pairs.

**Difference from zero-shot:**
- Zero-shot: Just trigger reasoning with text
- Few-shot: Show examples of reasoning, then model mimics the pattern

**Implementation:**

```python
prompt = """
Examples:

Q: A bakery makes 120 cookies. 25% are chocolate chip. How many chocolate chip?
A: Let me break this down. 120 cookies total. 25% are chocolate chip. 
   120 * 0.25 = 30 cookies. Answer: 30

Q: A student scored 85, 90, 92 on three tests. Average?
A: Sum the scores: 85 + 90 + 92 = 267. Divide by 3 tests: 267 / 3 = 89. Answer: 89

Now solve:

Q: A restaurant has 200 chairs. 30% are reserved. How many unreserved?
A:"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": prompt}]
)
# Output: "200 chairs total. 30% reserved: 200*0.3=60. Unreserved: 200-60=140. Answer: 140"
```

**Best practices:**
- 3-5 examples for complex tasks; 1-2 for simple
- Include reasoning steps in examples (not just answers)
- Diverse examples covering edge cases > homogeneous set
- Format consistency is critical — model learns format as much as task

**Measurable impact on few-shot selection variables** (The Prompt Report, 2024):

| Variable | Impact Range |
|---|---|
| Example order | 10-30% accuracy swing |
| Example quantity | 5-20% (1 vs 5 examples) |
| Example quality | 20-50% (clear vs ambiguous) |
| Format consistency | 15-40% (consistent vs mixed format) |
| Label distribution | 10-25% (balanced vs skewed) |
| Example similarity to test case | 10-35% (similar vs dissimilar) |

**Combined:** Up to 90% accuracy shift by optimizing example selection.

Implication: Spend time on examples. Manual example engineering beats most prompt optimization techniques.

---

## Auto-CoT (Automatic Chain-of-Thought)

**What:** Automatically generate demonstrations instead of hand-crafting them. Three steps:

1. **Clustering:** Group test questions by semantic similarity
2. **Sampling:** Pick one representative question from each cluster
3. **Demonstration generation:** Use zero-shot CoT on sampled questions to generate demos

Then use these auto-generated demos in few-shot CoT.

**Key paper:** "Automatic Chain-of-Thought Prompting in Large Language Models" — Zhang et al. 2023

**Why:** Eliminates manual example crafting. Works for any task.

**Implementation:**

```python
from sklearn.cluster import KMeans
import numpy as np

def auto_cot(test_questions: list[str], num_clusters: int = 5):
    """Generate CoT demonstrations automatically."""
    
    # Step 1: Embed all questions
    embeddings = [embedder.encode(q) for q in test_questions]
    
    # Step 2: Cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    
    # Step 3: Sample one question per cluster (closest to centroid)
    sampled_questions = []
    for cluster_id in range(num_clusters):
        cluster_mask = kmeans.labels_ == cluster_id
        cluster_questions = [q for q, m in zip(test_questions, cluster_mask) if m]
        # Pick most representative (could also pick first)
        sampled_questions.append(cluster_questions[0])
    
    # Step 4: Generate demonstrations using zero-shot CoT
    demonstrations = []
    for q in sampled_questions:
        reasoning = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{
                "role": "user",
                "content": f"{q}\n\nLet's think step by step."
            }]
        )
        demonstrations.append({
            "question": q,
            "reasoning": reasoning.content[0].text
        })
    
    return demonstrations

# Use auto-generated demos
demos = auto_cot(questions, num_clusters=5)

few_shot_prompt = "Examples:\n\n"
for demo in demos:
    few_shot_prompt += f"Q: {demo['question']}\nA: {demo['reasoning']}\n\n"
few_shot_prompt += f"Now solve:\n\nQ: {test_question}\nA:"

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": few_shot_prompt}]
)
```

**Measurable impact:** Outperforms random example selection. Often matches carefully hand-crafted examples with no human effort.

**Trade-off:** Adds one LLM call per sampled question. Amortized over many test cases, negligible cost.

---

## Self-Consistency Prompting

**What:** Generate K independent reasoning chains (high temperature) for the same question, then take majority vote on the final answer.

**Key paper:** "Self-Consistency Improves Chain of Thought Reasoning in Language Models" — Wang et al. 2022 (arXiv:2203.11171)

**Why:** Different reasoning paths can reach the same correct answer. A single greedy decode picks one path; voting on many paths catches errors.

**Measurable impact:**
- GSM8K: 40.7% (single CoT) → 55% (self-consistency, k=5)
- SVAMP: 75.2% → 86.5%
- MAWPS: 92.9% → 94.4%

**Implementation:**

```python
def self_consistency_solve(question: str, num_paths: int = 5, temperature: float = 1.0):
    """Generate multiple reasoning paths, vote on answers."""
    
    final_answers = []
    
    for _ in range(num_paths):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,  # High temp = diverse reasoning
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"{question}\n\nLet's think step by step."
            }]
        )
        
        reasoning = response.content[0].text
        # Extract final answer (last number, or parsed from last sentence)
        answer = extract_final_answer(reasoning)
        final_answers.append(answer)
    
    # Majority vote
    from collections import Counter
    most_common_answer = Counter(final_answers).most_common(1)[0][0]
    return most_common_answer

# Example
question = "If John has 5 apples and buys 3 more, then gives 2 to Mary, how many does he have?"
answer = self_consistency_solve(question, num_paths=5)
```

**Cost analysis:**
- Single CoT: 1 API call
- Self-consistency (k=5): 5 API calls
- 5x cost increase for 10-20% accuracy improvement (task-dependent)

**When to use:**
- High-stakes accuracy needed (medical, legal, financial)
- Time-insensitive (batch processing)
- Avoid if latency critical or cost-constrained

**Why not always use it:** 5x cost multiplier. For simple classification, single CoT often sufficient.

---

## Generate Knowledge Prompting

**What:** Two-call pattern:
1. **Call 1:** Ask LLM to generate relevant background knowledge about the topic
2. **Call 2:** Feed the generated knowledge + original question to solve the task

**Key paper:** "Generated Knowledge Prompting for Commonsense Reasoning" — Liu et al. 2022 (arXiv:2110.08387, University of Washington)

**Why:** Some questions require commonsense reasoning that the LLM has in pre-training but doesn't activate without prompt. Generate Knowledge forces activation before reasoning.

**Measurable impact:**
- CommonsenseQA: 70.1% (baseline) → 79.4% (generated knowledge)
- QASC: 74.5% → 78.2%
- NumerSense: 72% → 78% (7-10% gains)

**Implementation:**

```python
def generate_knowledge_solve(question: str):
    """Two-call: generate knowledge, then answer."""
    
    # Call 1: Generate background knowledge
    knowledge_prompt = f"""
Generate 3-5 relevant facts or background knowledge about the topic in this question.
Be concise. Focus on facts that would help answer the question.

Question: {question}

Relevant knowledge:
"""
    
    knowledge_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": knowledge_prompt}]
    )
    knowledge = knowledge_response.content[0].text
    
    # Call 2: Answer using generated knowledge
    answer_prompt = f"""
Background knowledge:
{knowledge}

Question: {question}

Answer:"""
    
    answer_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": answer_prompt}]
    )
    return answer_response.content[0].text

# Example: commonsense reasoning
question = "If you want to learn a new language, what would help you?"
answer = generate_knowledge_solve(question)
```

**Cost:** 2 API calls. Worth it for questions that require knowledge activation.

**When to use:**
- Commonsense reasoning (why?, what would?, how does?)
- Domain questions where knowledge exists but isn't primed
- Trade 2x cost for 10-20% accuracy improvement

---

## Least-to-Most Prompting

**What:** Decompose a complex problem into subproblems ordered from simplest to hardest. Solve sequentially, each solution becomes context for the next.

**Two stages:**
1. **Decomposition:** Break problem into ordered subproblems
2. **Sequential solving:** Solve each in order, prepend prior solutions

**Why:** CoT tries to solve everything in one reasoning stream. Least-to-most scales better on compositional problems because:
- Smaller subproblems are easier to reason about
- Each solved subproblem reduces cognitive load for the next
- Avoids the "forgetting early steps" problem in long reasoning chains

**Measurable impact (on SCAN compositional generalization):**
- Standard CoT: 16% accuracy
- Least-to-Most: 99.3% accuracy

**Implementation:**

```python
def least_to_most_solve(problem: str):
    """Decompose → solve in order."""
    
    # Stage 1: Decomposition
    decompose_prompt = f"""
Break this problem into ordered subproblems from simplest to hardest.
Number them 1, 2, 3, etc.

Problem: {problem}

Subproblems:
"""
    
    decompose_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": decompose_prompt}]
    )
    subproblems = decompose_response.content[0].text
    
    # Stage 2: Sequential solving
    solutions = []
    current_context = f"Problem: {problem}\n\nSubproblems:\n{subproblems}\n\nSolutions:\n"
    
    for i in range(1, 10):  # Up to 9 subproblems
        solve_prompt = f"""{current_context}

Solve subproblem {i}:"""
        
        solve_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": solve_prompt}]
        )
        
        solution = solve_response.content[0].text
        solutions.append(solution)
        current_context += f"\nSubproblem {i}: {solution}"
        
        # Check if problem solved (heuristic)
        if "final answer" in solution.lower() or i == 9:
            break
    
    return current_context

# Example: multi-step reasoning
problem = """
I have a robot that can move forward (F), turn left (L), and turn right (R).
Starting at position (0,0) facing north, what's my position after: F F R F L F F?
"""
answer = least_to_most_solve(problem)
```

**Cost:** Higher (multiple API calls), but justified for hard compositional problems.

**When to use:**
- Complex, multi-step reasoning (math proofs, logic puzzles, navigation)
- Tasks where simple CoT fails or produces verbose reasoning
- Acceptable latency/cost constraints

---

## Example Selection Variables — The 90% Shift

Most prompting advice focuses on technique (CoT, few-shot, etc.). The Prompt Report (2024, 1,500+ papers analyzed) discovered: **Example selection variables can shift accuracy by up to 90% more than technique choice.**

**Variables that matter:**

### 1. Example Order
Random order vs best order: 10-30% accuracy swing.

**Why:** Models are sensitive to first/last examples (primacy/recency bias).

**Best practice:** Put hardest examples last (model rehearses them most).

```python
# Bad order: easy examples first
examples = [
    ("simple case", "answer"),
    ("simple case", "answer"),
    ("complex case", "answer")  # Model forgets by time of test
]

# Good order: complexity increasing
examples = [
    ("simple case", "answer"),
    ("medium case", "answer"),
    ("hard case", "answer")  # Most recent in model's context
]
```

### 2. Example Quantity
1 example vs 5 examples: 5-20% improvement (task-dependent).

**Diminishing returns:** 5-10 examples plateau. More examples = more tokens = more cost, little accuracy gain.

### 3. Example Quality
Clear, unambiguous examples vs noisy/confusing examples: 20-50% swing.

**Quality over quantity.** One perfect example beats five mediocre ones.

### 4. Format Consistency
Mixed format (some JSON, some prose) vs consistent format: 15-40% swing.

```python
# Bad: inconsistent format
"Input: foo | Output: bar"
"Input: baz → Output: qux"
"Input: buzz; Output: quux"

# Good: consistent
"Input: foo | Output: bar"
"Input: baz | Output: qux"
"Input: buzz | Output: quux"
```

### 5. Label Distribution
Skewed (90% positive, 10% negative) vs balanced (50/50): 10-25% swing on imbalanced tasks.

```python
# Bad: label skew (if task is balanced)
positive_examples = 9  # 90%
negative_examples = 1  # 10%

# Good: match task distribution
positive_examples = 3  # 50%
negative_examples = 3  # 50%
```

### 6. Example Similarity to Test Case
Similar examples vs dissimilar: 10-35% improvement.

```python
# Use semantic similarity to match examples to test case
test_embedding = embedder.encode(test_case)
example_embeddings = [embedder.encode(ex) for ex in examples]
similarities = cosine_similarity([test_embedding], example_embeddings)[0]
top_k_indices = np.argsort(similarities)[-3:]  # Most similar
selected_examples = [examples[i] for i in top_k_indices]
```

---

## Comparison Table

| Technique | Cost | Accuracy Gain | Latency | Best For |
|---|---|---|---|---|
| Zero-shot CoT | 1x | 30-50% | Low | Simple reasoning |
| Few-shot CoT | 1x | 20-40% | Low | Domain-specific tasks |
| Auto-CoT | 2-6x (amortized) | 20-30% | Medium | Any task, no examples |
| Self-Consistency | 5x | 10-20% | High | High-accuracy needs |
| Generate Knowledge | 2x | 7-20% | Medium | Commonsense, knowledge priming |
| Least-to-Most | 3-5x | 20-50% | High | Compositional/multi-step |
| Example optimization | 1x | Up to 90% | Low | All tasks |

**Key insight:** Example quality/selection beats technique choice. Spend time optimizing examples first.

---

## Production Checklist

- [ ] **Baseline:** Measure zero-shot accuracy first
- [ ] **CoT:** Add "Let's think step by step" — quick 30%+ gain
- [ ] **Examples:** Craft 3-5 high-quality diverse examples (beats most other techniques)
- [ ] **Example order:** Put hardest examples last
- [ ] **Format:** Keep format consistent across examples
- [ ] **Eval:** Benchmark on 50+ test cases (don't optimize on 5)
- [ ] **Cost:** If accuracy plateaus, self-consistency only if budget allows

