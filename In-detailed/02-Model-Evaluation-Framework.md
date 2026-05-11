# Model Evaluation Framework: Metrics, Statistical Methods, Best Practices

## Evaluation Landscape

**Three levels:**
```
Level 1: Automated metrics (BLEU, ROUGE, exact match)     [Fast, cheap, limited insight]
Level 2: Structured evaluation (semantic similarity)      [Medium speed/cost, better insight]
Level 3: Human evaluation (A/B testing, scoring)          [Slow, expensive, ground truth]
```

Most production systems use Level 1 + Level 3. Level 2 (embedding-based) is increasingly common.

---

## Automated Metrics for Different Tasks

### 1. Classification Tasks

#### Accuracy
Simplest: % of correct predictions.

```
Predictions: ["positive", "negative", "positive", "negative", "positive"]
Ground truth: ["positive", "negative", "negative", "negative", "positive"]

Correct: [1, 1, 0, 1, 1] = 4/5 = 80% accuracy
```

**Limitation:** Doesn't account for class imbalance. If 95% samples are "negative":
```
Model: Always predict "negative"
Accuracy: 95% ❌ But useless (no discrimination)
```

#### Precision, Recall, F1

**Confusion matrix:**
```
                Predicted Positive  Predicted Negative
Actual Positive    TP=50              FN=10
Actual Negative    FP=5               TN=935
```

**Definitions:**
```
Precision = TP / (TP + FP) = 50 / (50+5) = 90.9%
→ Of positive predictions, 90.9% are correct

Recall = TP / (TP + FN) = 50 / (50+10) = 83.3%
→ Of actual positives, 83.3% are found

F1 = 2 * (Precision * Recall) / (Precision + Recall) = 86.96%
→ Harmonic mean, balanced metric
```

**When to use:**
- Precision: False positives are costly (spam detection, fraud)
- Recall: False negatives are costly (disease detection, security)
- F1: Balanced importance

**Implementation:**
```python
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

y_true = [1, 0, 1, 0, 1]
y_pred = [1, 0, 0, 0, 1]

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

# For multi-class:
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
```

#### ROC-AUC

Area under Receiver Operating Characteristic curve. Measures ability to distinguish between classes across all thresholds.

```
Model outputs probability, not hard class.
probability > 0.5 → "positive", else "negative"

But threshold is tunable:
threshold = 0.3 → more positives predicted (higher recall, lower precision)
threshold = 0.7 → fewer positives predicted (lower recall, higher precision)

ROC-AUC averages over all thresholds.
AUC = 1.0: Perfect (100% accuracy at all thresholds)
AUC = 0.5: Random (no discriminative power)
AUC = 0.7-0.8: Good
AUC = 0.8-0.9: Excellent
```

**Code:**
```python
from sklearn.metrics import roc_auc_score

y_true = [1, 0, 1, 0, 1, 0]
y_prob = [0.9, 0.3, 0.7, 0.2, 0.8, 0.4]  # Not hard classes, probabilities

auc = roc_auc_score(y_true, y_prob)
print(f"ROC-AUC: {auc:.2f}")
```

---

### 2. Generation Tasks (Summarization, Translation, QA)

#### BLEU (BiLingual Evaluation Understudy)

Measures n-gram overlap with reference output.

```
Reference: "The cat sat on the mat"
Prediction: "The cat was on the mat"

1-gram overlap: ["the", "cat", "on", "the", "mat"] = 5/5 = 100%
2-gram overlap: ["the cat", "on the", "the mat"] = 3/4 = 75%
3-gram overlap: ["cat on the"] = 0/3 = 0%

BLEU = weighted avg(1-gram: 0.25, 2-gram: 0.25, 3-gram: 0.25, 4-gram: 0.25)
     = 0.25 * 1.0 + 0.25 * 0.75 + 0.25 * 0.0 + 0.25 * 0.0
     = 0.44 (44%)
```

**Limitations:**
- Only counts exact n-gram matches. Synonyms = 0 credit. "cat" vs "feline" = different
- Length penalty: if prediction much shorter/longer, BLEU penalizes
- Doesn't measure semantic similarity

**Benchmark:**
```
BLEU < 0.3: Poor
BLEU 0.3-0.5: Acceptable
BLEU 0.5-0.7: Good
BLEU > 0.7: Excellent
```

**Code:**
```python
from nltk.translate.bleu_score import sentence_bleu

reference = [["The", "cat", "sat", "on", "the", "mat"]]
hypothesis = ["The", "cat", "was", "on", "the", "mat"]

score = sentence_bleu(reference, hypothesis)
print(f"BLEU: {score:.2f}")

# For multi-reference (multiple correct answers):
references = [
    ["The", "cat", "sat", "on", "the", "mat"],
    ["The", "cat", "is", "on", "the", "mat"],
]
score = sentence_bleu(references, hypothesis)
print(f"BLEU (multi-ref): {score:.2f}")
```

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Used for summarization. Measures recall of n-grams in reference summary.

```
Reference: "The quick brown fox jumps"
Prediction: "A quick brown fox jumps"

ROUGE-1 (unigram recall):
Common words: ["quick", "brown", "fox", "jumps"] = 4/5 = 80%

ROUGE-2 (bigram recall):
Reference bigrams: ["quick brown", "brown fox", "fox jumps"]
Common bigrams: ["brown fox", "fox jumps"] = 2/3 = 67%
```

**Variants:**
- ROUGE-N: N-gram overlap
- ROUGE-L: Longest common subsequence
- ROUGE-W: Weighted LCS

**Benchmark for summarization:**
```
ROUGE-1 > 0.4: Good
ROUGE-1 > 0.5: Excellent
```

**Code:**
```python
from rouge_score import rouge_scorer

reference = "The quick brown fox jumps"
prediction = "A quick brown fox jumps"

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference, prediction)

print(f"ROUGE-1: {scores['rouge1'].fmeasure:.2f}")
print(f"ROUGE-2: {scores['rouge2'].fmeasure:.2f}")
```

#### METEOR

Better than BLEU: credits synonyms and paraphrases via WordNet.

```
Reference: "The cat sat on the mat"
Prediction: "A feline was on the mat"

BLEU: Low (only "on", "the", "mat" match)
METEOR: Higher (recognizes "feline" ≈ "cat" via WordNet)
```

---

### 3. Semantic Similarity (LLM-Based Evaluation)

#### BERTScore

Compares embeddings, not surface tokens. Credits semantic similarity.

```
Reference: "The cat is on the mat"
Prediction: "A feline is on the rug"

Tokenize both:
Reference tokens: ["the", "cat", "is", "on", "the", "mat"]
Predicted tokens: ["a", "feline", "is", "on", "the", "rug"]

Embed each token → 768-dim vectors
Compare similarity via cosine:
  "cat" vs "feline": 0.92 ✅ (high, synonyms)
  "mat" vs "rug": 0.85 ✅ (high, similar)

Average similarities = BERTScore
```

**Advantages over BLEU:**
- Recognizes synonyms ("cat" ≈ "feline")
- Doesn't penalize paraphrasing
- More aligned with human judgment

**Code:**
```python
from bert_score import score

reference = "The cat is on the mat"
prediction = "A feline is on the rug"

P, R, F1 = score([prediction], [reference], lang="en", verbose=True)
print(f"BERTScore F1: {F1.item():.2f}")

# P = precision, R = recall, F1 = harmonic mean
```

---

## Evaluation Datasets and Train/Val/Test Split

### Proper Dataset Splits

**Critical rule:** Test set must be completely unseen during development.

```
Total dataset: 10,000 examples

Incorrect (data leakage):
[Train: 8000] [Validation: 2000]   ← No test set!
              Model overfits to validation

Correct:
[Train: 6000] [Validation: 2000] [Test: 2000]
              6k for training
              2k for hyperparameter tuning (learning rate, epochs)
              2k for final evaluation (only at end)
```

### Stratification (For Imbalanced Data)

If dataset has class imbalance (e.g., 95% negative, 5% positive):

```python
from sklearn.model_selection import train_test_split

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,  # ← Ensures same ratio in train/test
    random_state=42
)

# Now both train and test have ~95% negative, ~5% positive
```

---

## Statistical Rigor: Confidence Intervals & Effect Sizes

### Confidence Intervals

Don't report single accuracy. Report range.

```python
import numpy as np
from scipy import stats

# 100 test examples
predictions = [1, 0, 1, 1, 0, ...]  # length 100
accuracy = np.mean(predictions)  # e.g., 0.82

# 95% confidence interval
n = len(predictions)
se = np.sqrt(accuracy * (1 - accuracy) / n)  # Standard error
ci = accuracy ± 1.96 * se  # 1.96 for 95% CI

print(f"Accuracy: {accuracy:.2%} [CI: {accuracy - 1.96*se:.2%}, {accuracy + 1.96*se:.2%}]")
# Output: Accuracy: 82.00% [CI: 77.50%, 86.50%]
```

**Interpretation:**
```
Model A: 82% [77%, 87%]
Model B: 80% [75%, 85%]

Overlapping CIs → Not significantly different
Compare more carefully (may need larger test set)
```

### Effect Size (Cohen's d)

How big is the difference between two models?

```
Model A accuracy: 82%
Model B accuracy: 80%

Are they significantly different? Depends on:
- Standard deviation of results
- Sample size
- Not just the 2% difference
```

```python
def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    
    # Cohen's d
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Results of 5 runs each:
model_a = [0.82, 0.81, 0.83, 0.80, 0.82]
model_b = [0.80, 0.79, 0.81, 0.78, 0.80]

d = cohen_d(model_a, model_b)
print(f"Cohen's d: {d:.2f}")

# Interpretation:
# d < 0.2: Negligible
# d 0.2-0.5: Small
# d 0.5-0.8: Medium
# d > 0.8: Large
```

---

## A/B Testing Framework

### Setting Up A/B Test

```
Control: Base model (e.g., GPT-4)
Treatment: Fine-tuned model (e.g., GPT-4 + LoRA)

Hypothesis: Fine-tuned model is better (α = 0.05, 80% power)
Sample size needed: N (calculated below)
```

### Sample Size Calculator

```python
from scipy.stats import norm

def required_sample_size(baseline_rate, expected_rate, alpha=0.05, beta=0.20):
    """
    baseline_rate: Control conversion rate (e.g., 0.82)
    expected_rate: Treatment conversion rate (e.g., 0.87)
    alpha: Type I error (false positive), typically 0.05
    beta: Type II error (false negative), typically 0.20
    """
    z_alpha = norm.ppf(1 - alpha/2)      # Two-tailed
    z_beta = norm.ppf(1 - beta)
    
    p_bar = (baseline_rate + expected_rate) / 2
    
    n = (z_alpha + z_beta)**2 * (2 * p_bar * (1 - p_bar)) / (baseline_rate - expected_rate)**2
    
    return int(np.ceil(n))

# Example
n = required_sample_size(baseline_rate=0.82, expected_rate=0.87)
print(f"Need {n} samples per group")
# Output: Need 372 samples per group (744 total)
```

### Statistical Test (Two-Proportion Z-Test)

```python
from scipy.stats import norm

def two_proportion_ztest(count1, count2, n1, n2):
    """
    count1: Successes in group 1 (control)
    n1: Total samples in group 1
    count2: Successes in group 2 (treatment)
    n2: Total samples in group 2
    """
    p1 = count1 / n1
    p2 = count2 / n2
    
    p_pool = (count1 + count2) / (n1 + n2)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    z = (p2 - p1) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed
    
    return z, p_value

# Example: A/B test results
# Control (base model): 305 correct out of 372
# Treatment (FT model): 328 correct out of 372

z, p_value = two_proportion_ztest(305, 328, 372, 372)
print(f"Z-score: {z:.2f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Statistically significant (reject null hypothesis)")
    print("Treatment is meaningfully better than control")
else:
    print("❌ Not statistically significant")
    print("Difference could be due to chance")
```

---

## Human Evaluation Guidelines

### Setup

1. **Hiring evaluators:** Use Upwork, Scale AI, or internal team (min. 2 per example)
2. **Training:** Show 20 examples with "correct" answers; align evaluators
3. **Scoring rubric:** Define what "good" means (subjective!)

### Simple Rubric (Yes/No)

```
Question: "Is the summarization accurate and complete?"

Example 1:
Original: "Alice went to the store and bought milk, bread, and cheese."
Summary: "Alice bought groceries."
Rating: Yes (accurate, complete for a summary)

Example 2:
Original: "Alice went to the store..."
Summary: "Alice was hungry."
Rating: No (inaccurate, adds information)
```

### Likert Scale (1-5)

```
Rate the quality of this response: 1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent

1: Completely wrong, misleading
2: Mostly wrong, some useful info
3: Adequate, minor issues
4: Good quality, small issues
5: Excellent, no issues
```

### Measuring Evaluator Agreement

```python
from sklearn.metrics import cohen_kappa_score

evaluator1 = [5, 4, 3, 5, 2, ...]  # 50 ratings
evaluator2 = [5, 3, 3, 5, 2, ...]

kappa = cohen_kappa_score(evaluator1, evaluator2)
print(f"Cohen's Kappa: {kappa:.2f}")

# Interpretation:
# kappa > 0.8: Excellent agreement
# kappa 0.6-0.8: Good
# kappa 0.4-0.6: Moderate
# kappa < 0.4: Poor (retrain evaluators)
```

### Win Rate Analysis

```python
# Compare: Base model vs Fine-tuned model
# 100 examples evaluated

base_scores = [5, 4, 3, 5, 2, ...]
ft_scores = [5, 4, 4, 5, 3, ...]

wins = sum([1 for b, f in zip(base_scores, ft_scores) if f > b])
losses = sum([1 for b, f in zip(base_scores, ft_scores) if f < b])
ties = sum([1 for b, f in zip(base_scores, ft_scores) if f == b])

print(f"FT wins: {wins}, Base wins: {losses}, Ties: {ties}")
print(f"Win rate: {100 * wins / (wins + losses):.1f}%")

# Accept FT if win rate > 60% (statistically meaningful)
```

---

## Production Evaluation Checklist

- [ ] Establish baseline (current model or competitor)
- [ ] Choose evaluation metrics (automated + human)
- [ ] Create proper train/val/test splits (no leakage)
- [ ] Calculate required sample size for statistical significance
- [ ] Run automated metrics on test set
- [ ] Run human evaluation (min. 100 examples, 2 evaluators)
- [ ] Calculate confidence intervals (show variability)
- [ ] Run statistical tests (p-value < 0.05 for significance)
- [ ] Document results (metrics, p-values, human agreement)
- [ ] Make decision: deploy if win rate > 60% AND statistically significant

---

## Interview Talking Points

**"How do you evaluate an LLM-based system?"**

Layer it: Automated metrics first (fast, cheap, directional). For generation tasks, use BLEU/ROUGE. For classification, use F1/ROC-AUC. Then human evaluation (ground truth): 100+ examples, 2 evaluators minimum, measure agreement with Cohen's Kappa.

Never deploy on automated metrics alone. Humans are the source of truth.

**"What's the difference between BLEU and BERTScore?"**

BLEU counts exact n-gram matches, so "cat" and "feline" are different (BLEU penalizes). BERTScore compares embeddings, recognizing synonyms. BERTScore is more aligned with human judgment but slower.

For translation: BLEU. For open-ended generation (summarization, QA): BERTScore.

**"How do you know if a model improvement is real or just noise?"**

Confidence intervals + statistical tests. Model A: 82% [79%, 85%], Model B: 84% [81%, 87%]. Overlapping CIs → might be noise. Calculate sample size needed for 80% power, run z-test, check p-value < 0.05. Only then can you claim significance.

