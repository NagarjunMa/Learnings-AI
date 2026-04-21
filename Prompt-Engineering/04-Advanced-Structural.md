# Advanced Structural Techniques — Chaining, SoT, XML, Context Management

Not all problems fit a single prompt. Sometimes you need **pipeline architecture** — decompose into stages, each with its own prompt and output validation.

---

## Prompt Chaining

**What:** Decompose a complex task into a sequence of prompts, where each output feeds into the next.

**Why:** Transparency (inspect intermediate outputs), control (branch based on intermediate results), reusability (each stage works independently), debugging (pinpoint which stage fails).

**Three common patterns:**

### Pattern 1: Extract → Answer

Useful for document QA where you first extract relevant sections, then answer.

```python
def extract_answer_chain(document: str, question: str):
    """Extract relevant sections first, then answer."""
    
    # Stage 1: Extract relevant sections
    extract_prompt = f"""
Document:
{document}

Question: {question}

Extract 2-3 most relevant sentences from the document.
Only the sentences, nothing else.
"""
    
    extract_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": extract_prompt}]
    )
    relevant_context = extract_response.content[0].text
    
    # Stage 2: Answer using extracted context
    answer_prompt = f"""
Context:
{relevant_context}

Question: {question}

Answer based ONLY on the context above:
"""
    
    answer_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": answer_prompt}]
    )
    
    return {
        "extracted_context": relevant_context,
        "answer": answer_response.content[0].text
    }

# Usage
doc = "Paris is the capital of France. It has ~2.2M people. The Eiffel Tower is in Paris."
q = "What is the population of Paris?"
result = extract_answer_chain(doc, q)
print(result["answer"])  # Output: "~2.2M people"
```

**Benefit:** Extraction forces the model to be concise. Often better than passing the whole document.

### Pattern 2: Draft → Review → Refine

Useful for creative or technical writing where you need iteration.

```python
def draft_review_refine_chain(prompt: str, num_refinements: int = 2):
    """Draft → Review for errors → Refine based on feedback."""
    
    # Stage 1: Draft
    draft_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    draft = draft_response.content[0].text
    
    feedback_history = []
    
    for i in range(num_refinements):
        # Stage 2: Review
        review_prompt = f"""
Original task: {prompt}

Draft:
{draft}

Review this draft. What's unclear, wrong, or could be better?
Be specific. List 2-3 issues if you find any.
"""
        
        review_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": review_prompt}]
        )
        feedback = review_response.content[0].text
        feedback_history.append(feedback)
        
        # Check if good enough
        if "no issues" in feedback.lower() or "looks good" in feedback.lower():
            break
        
        # Stage 3: Refine based on feedback
        refine_prompt = f"""
Original task: {prompt}

Current draft:
{draft}

Feedback:
{feedback}

Revise the draft to address the feedback. Keep the same overall structure.
"""
        
        refine_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": refine_prompt}]
        )
        draft = refine_response.content[0].text
    
    return {
        "final": draft,
        "feedback_history": feedback_history,
        "iterations": num_refinements
    }

# Usage
prompt = "Write a paragraph explaining blockchain to a 10-year-old."
result = draft_review_refine_chain(prompt, num_refinements=2)
print(result["final"])
```

**Cost:** 3x per refinement iteration (draft + review + refine). Justified for high-quality output.

### Pattern 3: Generate → Verify → Revise

For tasks where correctness matters (code, math, contracts).

```python
def generate_verify_revise_chain(task: str):
    """Generate → Test/Verify → Fix bugs."""
    
    # Stage 1: Generate
    gen_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": task}]
    )
    code = gen_response.content[0].text
    
    # Stage 2: Verify (could be test execution or human review)
    verify_prompt = f"""
Task: {task}

Code:
{code}

Check this code for bugs. Can you spot any errors?
List specific issues line-by-line if found.
"""
    
    verify_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": verify_prompt}]
    )
    issues = verify_response.content[0].text
    
    # Stage 3: Revise if issues found
    if "no issues" not in issues.lower():
        revise_prompt = f"""
Original code:
{code}

Issues found:
{issues}

Fix these issues. Return only the corrected code.
"""
        
        revise_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": revise_prompt}]
        )
        code = revise_response.content[0].text
    
    return {
        "code": code,
        "issues": issues,
        "revised": "no issues" not in issues.lower()
    }
```

---

## Skeleton-of-Thought (SoT)

**What:** First generate an outline/skeleton of the answer, then fill in each point in **parallel** API calls.

**Why:** Sequential decoding is slow. Humans outline first, then write. SoT forces parallelism.

**Key paper:** "Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation" — Ning et al. ICLR 2024 (arXiv:2307.15337)

**Measurable impact:**
- Speedup: 2x-2.39x on 8 of 12 tested models
- Works as black-box technique (no model changes needed)
- Works on GPT-4, LLaMA, Claude

**Implementation:**

```python
import asyncio

async def skeleton_of_thought(question: str):
    """Skeleton generation → parallel filling."""
    
    # Stage 1: Generate skeleton
    skeleton_prompt = f"""
Question: {question}

Generate a skeleton/outline for answering this question.
Use numbered points (1., 2., 3., etc).
Each point should be 1-2 sentences max describing what to cover.

Skeleton:
"""
    
    skeleton_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": skeleton_prompt}]
    )
    
    skeleton_text = skeleton_response.content[0].text
    
    # Parse skeleton into points
    lines = skeleton_text.split('\n')
    points = [line.strip() for line in lines if line.strip() and line[0].isdigit()]
    
    # Stage 2: Fill each point in parallel
    async def fill_point(index: int, point: str):
        fill_prompt = f"""
Question: {question}

You're filling in point {index + 1} of the answer skeleton.

Skeleton point: {point}

Write 2-3 sentences filling in this point. Be specific and substantive.
"""
        
        # Sync call wrapped in async (would be async in real scenario)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": fill_prompt}]
        )
        
        return response.content[0].text
    
    # Run all fills in parallel
    filled_points = await asyncio.gather(*[
        fill_point(i, point) for i, point in enumerate(points)
    ])
    
    # Combine
    final_answer = "\n\n".join([
        f"{i + 1}. {points[i]}\n{filled_points[i]}"
        for i in range(len(points))
    ])
    
    return final_answer

# Usage (async)
# answer = asyncio.run(skeleton_of_thought("Explain photosynthesis"))
```

**Latency:** Skeleton (1 call) + N parallel fills = 2 API calls instead of N sequential.
- Skeleton call: ~500ms
- Parallel fills: ~1s (parallel, not serial)
- **Total: ~1.5s** vs **~5s sequential**

**When to use:**
- List-based answers (10 ways to..., compare A vs B vs C)
- Long-form content where structure is clear upfront
- When latency matters

**Avoid for:**
- Narrative/flowing text (structure unclear upfront)
- Tasks requiring sequential context (each answer depends on previous)

---

## XML / Structured Prompt Architecture

**What:** Use XML tags to separate instruction components within a prompt.

**Why:** Reduces ambiguity. Models trained on structured data parse tagged sections reliably.

**Anthropic strongly recommends this pattern.**

**Template:**

```xml
<instructions>
Your role and what you're doing.

Constraints and boundaries.
</instructions>

<context>
Background information, domain knowledge, or previous conversation.
</context>

<examples>
Example 1:
Input: ...
Output: ...

Example 2:
Input: ...
Output: ...
</examples>

<task>
The specific task the user wants done.
</task>

<input>
The actual user data/question.
</input>

<response_format>
How to structure the output (JSON, Markdown, XML, plain text).
Explicit output format specification.
</response_format>
```

**Example — Financial document classifier:**

```python
def structured_classifier(document: str):
    """Classify financial documents using XML structure."""
    
    prompt = """<instructions>
You are a financial document classifier for regulatory compliance.

Your task:
- Classify documents into exactly one category: 1099-INT, W-2, 1040, Schedule-C, Other
- If confidence < 80%, respond with "NEEDS_REVIEW"
- Never output PII (SSN, account numbers)
- Provide confidence score 0.0-1.0
</instructions>

<context>
Financial document categories:
- 1099-INT: Reports interest income
- W-2: Reports wages/salary income
- 1040: US individual income tax form
- Schedule-C: Self-employment income
</context>

<examples>
Example 1:
Input: "Interest income of $50 from Bank of America, Account 123..."
Output: {"category": "1099-INT", "confidence": 0.98}

Example 2:
Input: "Total wages subject to Medicare: $60,000"
Output: {"category": "W-2", "confidence": 0.96}
</examples>

<task>
Classify the document and return valid JSON.
</task>

<input>
Document:
{document}
</input>

<response_format>
Return ONLY valid JSON:
{{"category": "...", "confidence": 0.X}}
</response_format>
"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Usage
result = structured_classifier("Interest income of $25 from Savings Account")
print(result)  # {"category": "1099-INT", "confidence": 0.95}
```

**Key best practices (Anthropic):**

1. **Put long documents at the TOP of the prompt** (if combining many docs)
   - Impact: +30% quality on complex multi-document tasks
   - Reason: Model attends more carefully to early context

2. **Keep user queries at the BOTTOM**
   - Model focus is sharpest on recent content

3. **Use closing tags** `</tag>` explicitly, not implicit
   - Clearer parsing for both human and model

4. **Consistent tag naming** across similar tasks
   - `<instructions>`, `<context>`, `<examples>` become familiar patterns

---

## Context Window Management

### Summarize-as-You-Go

For long agentic tasks spanning multiple API calls, periodically summarize earlier context.

```python
def agentic_task_with_summary(initial_context: str, num_steps: int = 10):
    """Long-horizon task, summarize every 5 steps."""
    
    context = initial_context
    messages = []
    
    for step in range(num_steps):
        # Do work
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": context}] + messages
        )
        
        messages.append({"role": "assistant", "content": response.content[0].text})
        
        # Every 5 steps, summarize
        if (step + 1) % 5 == 0 and step > 0:
            summary_prompt = f"""
Summarize the key progress so far in 2-3 sentences.
Include important decisions and current state.

Progress:
{chr(10).join([m.get('content', '') for m in messages])}
"""
            
            summary_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            
            # Replace old messages with summary
            context += f"\n\nSummary of steps {step-4} to {step}:\n{summary_response.content[0].text}"
            messages = []  # Clear old messages
```

### State File Pattern

For very long tasks (100+ steps), use a JSON state file instead of prompt context.

```python
import json

def persistent_state_task(task_id: str, step: int):
    """State stored in file, not prompt context."""
    
    state_file = f"state/{task_id}.json"
    
    # Load state
    with open(state_file) as f:
        state = json.load(f)
    
    current_context = f"""
Task: {state['task']}
Progress: {state['progress_summary']}
Current step: {step}
Recent decisions:
{chr(10).join(state['recent_decisions'][-5:])}
"""
    
    # Do work
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": current_context}]
    )
    
    # Update state
    state['recent_decisions'].append(response.content[0].text)
    state['progress_summary'] = f"Step {step} complete."
    
    # Save state
    with open(state_file, 'w') as f:
        json.dump(state, f)
```

### "Grounding in Quotes" Technique

For very long documents (10K+ tokens), ask the model to quote relevant passages before reasoning.

```python
def grounded_qa(document: str, question: str):
    """Quote from document, then reason."""
    
    quote_prompt = f"""
Document:
{document}

Question: {question}

Find and quote the most relevant passage from the document.
Quote: "..."
"""
    
    quote_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": quote_prompt}]
    )
    
    quote = quote_response.content[0].text
    
    reason_prompt = f"""
Question: {question}

Relevant passage: {quote}

Answer based on this passage:
"""
    
    answer_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": reason_prompt}]
    )
    
    return answer_response.content[0].text
```

**Why:** Quoting forces the model to ground its reasoning in the document, reducing hallucination on long contexts.

---

## Directional Stimulus Prompting (DSP)

**What:** A small tunable policy model generates hints/keywords (stimulus) added to main LLM's prompt.

**Key paper:** "Guiding Large Language Models via Directional Stimulus Prompting" — Li et al. NeurIPS 2023 (arXiv:2302.11520)

**Measurable impact:** 41.4% relative improvement on MultiWOZ dialogue task using only 80 labeled examples.

**Why:** Agents often lack task-specific nudges. DSP learns what nudges work.

**Implementation (simplified):**

```python
def dsp_prompt_generation(task_description: str, task_data: list[dict]):
    """Train small policy model to generate hints."""
    
    # Stage 1: Train hint generator on labeled examples
    # (In practice, use gradient descent or RL)
    
    def learned_hint_generator(input_text: str) -> str:
        """Returns task-specific keywords to include."""
        # This would be trained on examples
        # For now, simulate with rule-based logic
        
        if "summarize" in task_description.lower():
            return "Key points: concise, main ideas, structured"
        elif "dialogue" in task_description.lower():
            return "Natural flow: polite tone, acknowledge user, provide info"
        else:
            return "Clear communication: structured, specific"
    
    # Stage 2: Use hints in main LLM prompt
    def dsp_solve(user_input: str):
        hints = learned_hint_generator(user_input)
        
        prompt = f"""
Task: {task_description}

Helpful hints for this task:
{hints}

User input:
{user_input}

Response:
"""
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    return dsp_solve
```

**Use case:** When you have 50+ examples of the task and want to automatically extract what makes a good response.

---

## Production Checklist

- [ ] **Chaining:** Use 2-3 stage pipelines for complex tasks
- [ ] **XML structure:** Tag instructions, context, examples, input separately
- [ ] **Long docs:** Place at TOP of prompt, queries at BOTTOM
- [ ] **SoT for speed:** Use parallel filling for list-based answers
- [ ] **Context management:** Summarize every 5-10 steps for long tasks
- [ ] **Grounding:** Quote sources before reasoning on very long contexts
- [ ] **Cost tracking:** Chaining costs N times baseline; budget accordingly

