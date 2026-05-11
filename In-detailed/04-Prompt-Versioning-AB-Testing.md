# Prompt Versioning and A/B Testing: Strategies and Best Practices

## Prompt Versioning: Why and How

### Why Version Prompts?

Prompts are code. They need version control, testing, and rollback.

```
Analogy: Code releases
V1.0: Original code
V1.1: Bug fix (backward compatible)
V2.0: New feature (breaking change)

Same for prompts:
V1.0: "You are a helpful assistant"
V1.1: "You are a helpful assistant. Be concise."  (improvement, backward compat)
V2.0: New system prompt, different behavior (breaking)
```

### Simple Version Control Approach

```bash
# Git repository for prompts

prompts/
├── summarization/
│   ├── v1.txt          # Original
│   ├── v2.txt          # Added "focus on key points"
│   ├── v3.txt          # Added "max 3 sentences"
│   └── CHANGELOG.md
├── code_review/
│   ├── v1.txt
│   └── v2.txt
└── README.md

# Commit each change
git add prompts/summarization/v3.txt
git commit -m "feat: Add max sentence limit to summarization prompt"

# Tag releases
git tag summarization/v3
```

### Structured Prompt Format (For Reuse)

```yaml
# prompts/summarization/v3.yaml

name: "Summarization Prompt v3"
version: "3"
created_at: "2025-04-22"
author: "alice@company.com"
description: "Summarize text with focus on key points, max 3 sentences"

system_prompt: |
  You are a summarization expert. Your task is to:
  1. Extract key points from the given text
  2. Summarize in maximum 3 sentences
  3. Preserve specific details (names, numbers, dates)

examples:
  - input: "The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree..."
    output: "A brown fox jumps over a dog sleeping under a tree."
  
  - input: "Alice founded Company X in 2020. She raised $10M in Series A..."
    output: "Alice founded Company X in 2020 and raised $10M in Series A."

metrics:
  - rouge1: 0.65
  - rouge2: 0.45
  - human_score: 4.2/5.0
  - latency_ms: 850

tags: ["summarization", "concise"]
deployment: "production"
```

---

## A/B Testing Framework for Prompts

### Step 1: Define Hypothesis

```
Null hypothesis (H0): Prompt A and Prompt B perform equally
Alternative hypothesis (H1): Prompt B performs better than Prompt A

Example:
H0: Both prompts achieve 80% accuracy
H1: Prompt B achieves > 85% accuracy

Significance level (α): 0.05 (5% false positive rate acceptable)
Power (1-β): 0.80 (80% chance detect true difference if exists)
```

### Step 2: Sample Size Calculation

```python
# For binary outcomes (correct/incorrect)

def required_sample_size(baseline_p, expected_p, alpha=0.05, beta=0.20):
    """
    baseline_p: Current success rate (e.g., 0.80)
    expected_p: Expected success rate with new prompt (e.g., 0.85)
    """
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(1 - beta)
    
    p_bar = (baseline_p + expected_p) / 2
    
    n = (z_alpha + z_beta)**2 * (2 * p_bar * (1 - p_bar)) / (baseline_p - expected_p)**2
    
    return int(np.ceil(n))

# Example: baseline 80%, expected 85%
n = required_sample_size(0.80, 0.85)
print(f"Need {n} samples per group")
# Output: Need 556 samples per group (1112 total)

# This means: test on ~1000 queries to detect 5% improvement with 80% power
```

### Step 3: Traffic Allocation (Staged Rollout)

```
Phase 1: Canary (5% of traffic)
  Control (A): 97.5% of users (safe baseline)
  Treatment (B): 2.5% of users (test new prompt)
  Duration: 1 hour
  Goal: Ensure no crashes/errors

Phase 2: Increasing Load (25% traffic)
  Control (A): 75% of users
  Treatment (B): 25% of users
  Duration: 24 hours
  Goal: Detect early performance issues

Phase 3: Full A/B (100% traffic split)
  Control (A): 50% of users
  Treatment (B): 50% of users
  Duration: 3-7 days (depends on traffic volume, target sample size)
  Goal: Collect 556+ samples per group for statistical significance
```

### Step 4: Implementation with FastAPI

```python
from fastapi import FastAPI
from enum import Enum
import random
import hashlib

app = FastAPI()

class PromptVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"

def get_variant(user_id: str, traffic_split: float = 0.5) -> PromptVersion:
    """
    Deterministic assignment: same user always gets same variant
    """
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return PromptVersion.v2 if (hash_value % 100) < (traffic_split * 100) else PromptVersion.v1

PROMPTS = {
    PromptVersion.v1: "You are a helpful assistant.",
    PromptVersion.v2: "You are a helpful assistant. Be concise and direct.",
}

@app.post("/chat")
async def chat(user_id: str, message: str):
    # Assign user to variant
    variant = get_variant(user_id, traffic_split=0.5)  # 50/50 split
    prompt = PROMPTS[variant]
    
    # Call LLM with chosen prompt
    response = await llm.ainvoke(prompt + message)
    
    # Log for analysis
    log.info(
        "chat_request",
        user_id=user_id,
        variant=variant,
        message_length=len(message),
        response_length=len(response)
    )
    
    return {
        "response": response,
        "variant": variant,
        "message_id": str(uuid.uuid4())
    }
```

### Step 5: Analysis & Statistical Testing

```python
from scipy.stats import chi2_contingency, norm
import pandas as pd

# Load results from database
results_df = pd.read_sql("SELECT * FROM chat_logs WHERE test_date >= DATE('now') - 7", db_conn)

# Split by variant
v1_data = results_df[results_df['variant'] == 'v1']
v2_data = results_df[results_df['variant'] == 'v2']

print(f"V1 samples: {len(v1_data)}, V2 samples: {len(v2_data)}")

# Metric 1: Success Rate (binary outcome)
v1_success = v1_data['success'].sum()
v2_success = v2_data['success'].sum()

# Two-proportion z-test
def two_prop_z_test(count1, n1, count2, n2):
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p2 - p1) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return z, p_value

z, p_value = two_prop_z_test(v1_success, len(v1_data), v2_success, len(v2_data))

print(f"V1 success rate: {v1_success/len(v1_data):.2%}")
print(f"V2 success rate: {v2_success/len(v2_data):.2%}")
print(f"Z-score: {z:.2f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ STATISTICALLY SIGNIFICANT - Deploy V2")
else:
    print("❌ NOT SIGNIFICANT - Keep V1 or collect more data")

# Metric 2: Average Response Quality (Likert scale 1-5)
v1_quality = v1_data['user_rating'].mean()
v2_quality = v2_data['user_rating'].mean()

print(f"V1 avg rating: {v1_quality:.2f}/5.0")
print(f"V2 avg rating: {v2_quality:.2f}/5.0")

# Metric 3: Latency (continuous outcome)
v1_latency = v1_data['response_latency_ms'].mean()
v2_latency = v2_data['response_latency_ms'].mean()

print(f"V1 latency: {v1_latency:.0f}ms")
print(f"V2 latency: {v2_latency:.0f}ms")

# Decision
if p_value < 0.05 and v2_quality > v1_quality and v2_latency < v1_latency:
    print("🎉 DEPLOY V2 - Better quality, faster, statistically significant")
elif p_value < 0.05 and v2_quality > v1_quality:
    print("✅ DEPLOY V2 - Better quality, statistically significant")
elif p_value >= 0.05:
    print("⏸ INCONCLUSIVE - Run longer or modify prompt further")
```

---

## Bayesian A/B Testing (Alternative)

More robust for small sample sizes or early stopping.

```python
from scipy.stats import beta

def bayesian_bandit(v1_successes, v1_trials, v2_successes, v2_trials, threshold=0.95):
    """
    Bayesian approach: What's probability V2 > V1?
    If P(V2 > V1) > threshold (e.g., 95%), declare winner.
    """
    # Prior: uniform distribution Beta(1, 1)
    # Update with observed data: Beta(successes + 1, failures + 1)
    
    v1_dist = beta(v1_successes + 1, v1_trials - v1_successes + 1)
    v2_dist = beta(v2_successes + 1, v2_trials - v2_successes + 1)
    
    # Monte Carlo: sample from both, estimate P(V2 > V1)
    samples_v1 = v1_dist.rvs(100000)
    samples_v2 = v2_dist.rvs(100000)
    
    prob_v2_better = np.mean(samples_v2 > samples_v1)
    
    print(f"P(V2 > V1) = {prob_v2_better:.2%}")
    
    if prob_v2_better > threshold:
        return "V2 WINS"
    elif prob_v2_better < (1 - threshold):
        return "V1 WINS"
    else:
        return "INCONCLUSIVE"

# Usage (can stop early when probability is high)
result = bayesian_bandit(v1_successes=80, v1_trials=100, v2_successes=90, v2_trials=100)
print(result)  # Output: "V2 WINS" if P(V2 > V1) > 95%
```

---

## Multi-Armed Bandit: Dynamic Allocation

For cases where you want to minimize losses while testing (e.g., expensive LLM calls).

```python
from bandit import BernouilliBandit

class PromptBandit:
    def __init__(self, prompts, epsilon=0.1):
        self.prompts = prompts
        self.epsilon = epsilon
        self.successes = {p: 0 for p in prompts}
        self.trials = {p: 0 for p in prompts}
    
    def select_prompt(self):
        # Epsilon-greedy: 90% best, 10% random exploration
        if random.random() < self.epsilon:
            return random.choice(self.prompts)  # Explore
        else:
            # Exploit: pick best so far
            success_rates = {p: self.successes[p] / (self.trials[p] + 1) for p in self.prompts}
            return max(success_rates, key=success_rates.get)
    
    def update(self, prompt, success: bool):
        self.trials[prompt] += 1
        if success:
            self.successes[prompt] += 1

# Usage
prompts = ["v1", "v2", "v3"]
bandit = PromptBandit(prompts)

for _ in range(1000):
    prompt = bandit.select_prompt()
    response = llm.invoke(prompt + user_input)
    success = evaluate(response)
    bandit.update(prompt, success)

# Result: Explores all prompts, but increasingly favors the best
print(f"Success rates: {bandit.successes}")
# Output: v2 gets ~50% traffic (best), v1 and v3 get less
```

---

## Prompt A/B Testing Checklist

- [ ] Define hypothesis (what success looks like)
- [ ] Calculate sample size needed (for 80% power)
- [ ] Implement traffic allocation (canary → staged → full)
- [ ] Log all requests (variant, outcome, latency, cost)
- [ ] Collect at least target sample size
- [ ] Run statistical test (p-value < 0.05)
- [ ] Check secondary metrics (latency, cost, quality)
- [ ] Document results (which prompt won, by how much, why)
- [ ] Deploy winner to production
- [ ] Monitor for regression (compare to baseline in production)

---

## Version Control + CI/CD for Prompts

### Git Workflow

```bash
# Feature branch for new prompt version
git checkout -b prompts/add-cot-reasoning

# Edit prompt
echo "Let's think step-by-step" >> prompts/chain_of_thought/v2.txt

# Test locally
python test_prompt.py prompts/chain_of_thought/v2.txt

# Push for review
git push origin prompts/add-cot-reasoning
git pull-request

# After approval, merge
git merge prompts/add-cot-reasoning

# Tag release
git tag cot/v2
```

### Automated Testing Pipeline

```yaml
# .github/workflows/test-prompts.yml

name: Test Prompts

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: pip install openai pytest langsmith
      
      - name: Run prompt tests
        run: |
          python -m pytest tests/test_prompts.py \
            --prompt-file=${{ github.event.pull_request.head.ref }} \
            --metric=accuracy \
            --threshold=0.85
      
      - name: Check latency
        run: |
          python benchmark_prompt.py ${{ github.event.pull_request.head.ref }} \
            --target-latency=500ms
      
      - name: Estimate cost
        run: |
          python cost_estimator.py ${{ github.event.pull_request.head.ref }} \
            --monthly-volume=1000000
      
      - name: Comment results on PR
        if: always()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '✅ All tests passed\n- Accuracy: 87%\n- Latency: 450ms\n- Est. cost: $5000/month'
            })
```

---

## Interview Talking Points

**"How would you implement A/B testing for prompts?"**

Define hypothesis: which metric matters (accuracy, latency, cost). Calculate sample size for 80% power (e.g., 500 queries per variant). Allocate traffic: canary 5% first to detect errors, then staged rollout. Collect data, run statistical test (p-value < 0.05 means winner). Deploy winner, monitor for regression.

**"Why version prompts like code?"**

Prompts evolve. Version control lets you track changes, revert if needed, A/B test systematically. Structured format (YAML) makes it easy to attach metrics, metadata, deployment status. CI/CD pipeline auto-tests new prompts before merging.

**"Bayesian vs Frequentist A/B testing?"**

Frequentist: fixed sample size, p-value < 0.05 for significance. Simple, but requires large samples.

Bayesian: continuous monitoring, early stopping when posterior probability is high (e.g., 95% sure V2 > V1). Better for costly experiments.

