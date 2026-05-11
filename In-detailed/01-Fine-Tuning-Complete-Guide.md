# Fine-Tuning: Complete Guide for Production

## When to Fine-Tune vs Prompt

### Decision Matrix

| Scenario | Fine-Tune | Prompt | Why |
|----------|-----------|--------|-----|
| 1. Few-shot examples work | ❌ | ✅ | Prompting is 95% as good, 100x cheaper |
| 2. Need domain-specific behavior | ✅ | ⚠️ | Prompting adds tokens, fine-tuning bakes it in |
| 3. Latency < 100ms critical | ✅ | ❌ | Fine-tuned model infers faster (no long prompt) |
| 4. Cost-per-inference matters (10M+/month) | ✅ | ❌ | FT amortizes over high volume |
| 5. Proprietary knowledge (cannot leak) | ✅ | ❌ | Fine-tuning on private data, no long context |
| 6. Output format must be exact | ✅ | ⚠️ | FT ensures format 99%+, prompt 85%+ |
| 7. Reasoning style required | ❌ | ✅ | Prompting (CoT, reasoning chains) > FT |
| 8. Model refusal bypassing | ❌ | ✅ | Can't jailbreak via FT (safety still applies) |
| 9. Few examples available (<100) | ❌ | ✅ | FT needs 500+ examples min |
| 10. Large dataset available (10k+) | ✅ | ⚠️ | FT will outperform prompting |

### Rule of Thumb
**Prompt first.** If prompt + few-shot examples reach 90%+ accuracy, stop. Fine-tuning costs (time, money, infrastructure) are not worth 1-5% gains. Only fine-tune if:
- Dataset size: 500+ examples minimum
- Accuracy improvement: expected >10% lift
- Cost justification: savings > training cost within 6 months

---

## Fine-Tuning Data Preparation

### Step 1: Data Collection & Labeling

**Minimum dataset sizes:**
```
Task Type              Min Examples  Recommended   Production
─────────────────────────────────────────────────────────────
Classification        100           500           2000+
Summarization         200           1000          5000+
Code generation       300           1000          10000+
Chat/conversational   500           2000          20000+
Instruction-following 500           2000          10000+
```

**Data format (OpenAI / Anthropic):**
```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a code reviewer."},
      {"role": "user", "content": "Review this function:\n\nfunction add(a, b) {\n  return a + b;\n}"},
      {"role": "assistant", "content": "Good: Simple, readable. Missing: Type hints, edge case handling for null."}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "You are a code reviewer."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

### Step 2: Data Quality Checks

**Before training:**
- [ ] No duplicate examples (exact matches)
- [ ] No near-duplicates (similar input, different output → model confusion)
- [ ] Output length distribution check (outliers? median, p95?)
- [ ] Class balance check (for classification tasks)
- [ ] PII redaction (names, emails, SSNs removed)
- [ ] Encoding consistent (UTF-8, no control characters)
- [ ] Example count ≥ 500 (minimum)

**Code to check:**
```python
import json
from collections import Counter

with open("training_data.jsonl") as f:
    data = [json.loads(line) for line in f]

# Check 1: Duplicates
inputs = [d["messages"][-2]["content"] for d in data]  # Second-to-last = user message
duplicates = len(inputs) - len(set(inputs))
print(f"Duplicate examples: {duplicates}")

# Check 2: Output length distribution
outputs = [d["messages"][-1]["content"] for d in data]  # Last = assistant
output_lengths = [len(o.split()) for o in outputs]
print(f"Output length - median: {sorted(output_lengths)[len(output_lengths)//2]}, p95: {sorted(output_lengths)[int(len(output_lengths)*0.95)]}")

# Check 3: Class balance (for classification)
labels = [d["messages"][-1]["content"].split()[0] for d in data]  # First word of output
print(f"Label distribution: {Counter(labels)}")

# Check 4: Minimum dataset size
print(f"Total examples: {len(data)} (minimum: 500)")
```

### Step 3: Train/Validation Split

**For fine-tuning:**
```
Dataset size    Train%   Validation%
─────────────────────────────────
500-1000        80%      20%
1000-5000       85%      15%
5000+           90%      10%
```

**Example split:**
```python
import random

with open("training_data.jsonl") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
split = int(len(data) * 0.8)

with open("train.jsonl", "w") as f:
    for example in data[:split]:
        f.write(json.dumps(example) + "\n")

with open("validation.jsonl", "w") as f:
    for example in data[split:]:
        f.write(json.dumps(example) + "\n")

print(f"Train: {split}, Validation: {len(data) - split}")
```

---

## Fine-Tuning Techniques: LoRA and QLoRA

### Full Fine-Tuning (Baseline)

**What:** Update ALL model weights. 7B model = 7B parameters to update.

```
Memory cost:  gradient + weight + optimizer = 4*params*4bytes 
              = 4 * 7B * 4 = 112GB
Training time: Slow (48+ hours on A100)
Cost:         $500+ for single run
```

**When to use:** Only if you have infinite compute. Not practical.

### LoRA: Low-Rank Adaptation

**Core idea:** Don't update all weights. Instead, update a small set of trainable rank matrices.

```
Original layer:  y = W * x           [W is 7B parameters]
LoRA layer:      y = W * x + BA * x  [A is r×d, B is d×r, r << d]

Example: W is 4096×4096 (16M parameters)
With r=64: A is 4096×64 (262K params), B is 64×4096 (262K params)
Total: 524K trainable params (3% of original)
```

**Advantage:**
```
Memory:        4 * params * 4 bytes = 4 * 524K * 4 = 8MB (vs 112GB full)
Training time: 2-4 hours on single GPU (vs 48+ hours full)
Cost:          $20-50 per run (vs $500+ full)
Inference:     Same speed as base model (no inference overhead)
```

**Practical code (Hugging Face + peft):**
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA config
lora_config = LoraConfig(
    r=64,                    # Rank of adaptation matrices
    lora_alpha=16,           # Scaling factor (typically r/4)
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,       # Dropout in LoRA layers
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06

# Training
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    save_steps=100,
    save_total_limit=3,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save LoRA weights (only 4MB, not full 26GB model)
model.save_pretrained("./lora_final")
```

### QLoRA: Quantized LoRA

**Extension of LoRA:** Quantize base model (4-bit) + train LoRA on top.

```
Full FT memory:     112GB
LoRA memory:        8MB
QLoRA memory:       2MB (0.25MB base + 1.75MB LoRA)
```

**Code:**
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on top
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

# Training proceeds same as LoRA
```

**Trade-off:**
- Memory: 100x reduction (112GB → 2MB)
- Quality: 1-2% loss vs full FT (acceptable for most tasks)
- Speed: Slower inference due to dequantization (but still <100ms)

---

## Training Dynamics: Loss Functions and Learning

### Standard Loss: Cross-Entropy

For language models, minimize cross-entropy loss:

```
Loss = -1/N * Σ log(P(y_i | x_i, θ))

where:
- N = number of tokens
- P(y_i | x_i, θ) = probability model assigns to correct token
- Lower loss = better (0 = perfect)
```

**Example:**
```
Input:  "The capital of France is"
Target: "Paris"

Probability distribution (before training):
  "London": 0.3  → loss = -log(0.3) = 1.2
  "Paris": 0.2   → loss = -log(0.2) = 1.6  ❌ High loss (wrong)

After training:
  "London": 0.05 → loss = -log(0.05) = 3.0
  "Paris": 0.90  → loss = -log(0.90) = 0.10  ✅ Low loss (correct)
```

### Training Dynamics: What to Monitor

**Metrics to track during training:**

```
Epoch 1, Step 100:  Loss: 2.45
Epoch 1, Step 200:  Loss: 2.10
Epoch 1, Step 300:  Loss: 1.98  ← Should decrease
...
Epoch 2, Step 100:  Loss: 1.65
Epoch 2, Step 200:  Loss: 1.58
Epoch 3, Step 100:  Loss: 1.52  ← Should converge
Epoch 3, Step 200:  Loss: 1.51

⚠️  WARNING: If loss plateaus or increases, learning rate too high → reduce by 2x
⚠️  WARNING: If loss decreases very slowly, learning rate too low → increase by 2x
```

**Validation metrics (evaluate on held-out set every N steps):**

```python
# Perplexity = e^(average_loss)
perplexity = math.exp(validation_loss)

# Example:
# Validation loss: 1.5 → Perplexity: 4.5
# Interpretation: On average, model is "surprised" by 4.5x compared to uniform distribution

# Lower perplexity = better
# Industry benchmarks:
#   Base model (pre-trained): 10-50
#   Fine-tuned (domain-specific): 5-15
#   Fine-tuned (narrow domain): 2-5
```

### Convergence Indicators

**Good convergence:**
```
Step 100:    Loss 2.5
Step 200:    Loss 2.1   (↓0.4)
Step 300:    Loss 1.9   (↓0.2)
Step 400:    Loss 1.85  (↓0.05)  ← Diminishing returns, safe to stop
```

**Bad convergence (overfitting):**
```
Train Loss   Validation Loss
Step 100:  2.5        2.6
Step 200:  2.0        2.2
Step 300:  1.5        2.0  ← Validation stops improving, overfitting
Step 400:  0.9        2.5  ← Train improves, validation worsens
```

**Fix overfitting:**
- Reduce learning rate
- Add regularization (weight decay, dropout)
- Use fewer epochs
- Increase validation set size

---

## Learning Rate Selection

### Rule of Thumb
```
Base model LR: 5e-5 (from pre-training)
Full FT LR:    2e-5 to 5e-5
LoRA LR:       1e-4 to 5e-4  (higher, since fewer params updating)
QLoRA LR:      5e-4 to 1e-3  (highest, most aggressive)
```

### Learning Rate Finder

```python
from transformers import TrainingArguments, Trainer

# Run with high learning rate to find optimal
training_args = TrainingArguments(
    output_dir="./lr_finder",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=1e-1,  # Start high
    lr_scheduler_type="linear",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Plot loss vs learning rate
# Find steepest descent, use that LR (or 10x lower for safety)
trainer.train()
```

---

## Fine-Tuning Cost vs Quality

### Cost Breakdown

```
Task                 Data Size    Training Time    Cost (A100)    Quality Gain
──────────────────────────────────────────────────────────────────────────────
Classification       1K           30 min           $5             +15%
Summarization        5K           2 hours          $20            +20%
Code generation      10K          4 hours          $40            +25%
Chat (full-param)    20K          24 hours         $500           +30%
Chat (LoRA)          20K          3 hours          $30            +28%
Chat (QLoRA)         20K          1 hour           $10            +25%
```

### ROI Calculation

```
Investment:
  Training cost: $30
  Dev time: 10 hours @ $100/hr = $1000
  Total: $1030

Savings:
  Token cost reduction: 50% (LoRA removes long prompt context)
  Monthly API spend: $10,000
  Monthly savings: $5,000
  Payoff time: $1030 / $5000/month = 0.2 months (6 days)
  ✅ WORTH IT
```

---

## Production Fine-Tuning Workflow

### Step 1: Offline Training (Local or Cloud)

```bash
# Using Hugging Face Trainer
python train.py \
  --model_name meta-llama/Llama-2-7b \
  --train_file train.jsonl \
  --validation_file val.jsonl \
  --output_dir ./checkpoints \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-4 \
  --save_steps 100 \
  --eval_steps 100
```

### Step 2: Evaluation on Test Set

```python
# Load trained model
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("./checkpoints/final")

# Evaluate
test_inputs = [example["messages"][1]["content"] for example in test_data]
test_outputs = [example["messages"][2]["content"] for example in test_data]

predictions = []
for input_text in test_inputs:
    output = model.generate(input_text, max_length=50)
    predictions.append(output)

# Compare predictions vs test_outputs
from bleu_score import sentence_bleu  # or other metric

scores = [sentence_bleu([test_outputs[i]], predictions[i]) for i in range(len(test_outputs))]
print(f"Average BLEU: {sum(scores) / len(scores)}")

# Accept if:
#   - Accuracy > threshold (e.g., 90%)
#   - BLEU > baseline (e.g., 0.7)
#   - Latency < SLA (e.g., 100ms per token)
```

### Step 3: Merge and Deploy

```python
# Merge LoRA weights with base model
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("./checkpoints/final")
merged_model = model.merge_and_unload()

# Save merged model (now a standard transformers model)
merged_model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

# Deploy (e.g., to HuggingFace Hub)
merged_model.push_to_hub("my-username/my-finetuned-model")
```

### Step 4: A/B Test

```python
# Compare original vs fine-tuned
base_model = load_model("meta-llama/Llama-2-7b")
ft_model = load_model("./final_model")

# For N random queries:
for query in test_queries:
    base_output = base_model.generate(query)
    ft_output = ft_model.generate(query)
    
    # Human or automated evaluation
    if evaluate(ft_output) > evaluate(base_output):
        wins += 1

print(f"FT model wins: {wins}/{len(test_queries)} ({100*wins/len(test_queries):.1f}%)")
# Accept if > 60% win rate
```

---

## Common Mistakes and Fixes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Learning rate too high | Loss NaN, diverges | Reduce by 10x |
| Learning rate too low | Loss decreases very slowly | Increase by 2x |
| Overfitting | Train loss ↓, val loss → or ↑ | Reduce epochs, add dropout |
| Data quality issues | Random outputs, repeating tokens | Check for duplicates, PII, encoding |
| Class imbalance | Poor accuracy on rare class | Oversample minority, weight loss |
| Insufficient data | Cannot converge | Collect more examples (500+ minimum) |
| Training diverges | Loss becomes infinite | Reduce batch size, lower learning rate |
| Model forgets base knowledge | Catastrophic forgetting | Use lower learning rate, fewer epochs |

---

## Interview Talking Points

**"When would you recommend fine-tuning over prompting?"**

Fine-tune if:
- Dataset: 500+ examples minimum
- Expected quality gain: >10%
- Latency critical: under 100ms (fine-tuning removes long prompt overhead)
- Cost justification: recurring inference savings exceed training investment within 6 months

Otherwise, prompt first. Fine-tuning is expensive; prompting is cheap and good enough 90% of the time.

**"How do you prepare data for fine-tuning?"**

1. Collect 500+ labeled examples (format: message pairs)
2. Quality checks: deduplicate, balance classes, redact PII
3. Split 80/20 train/validation
4. Validate with ~50 manual examples before training

**"Compare LoRA vs QLoRA vs full fine-tuning."**

Full FT: Best quality (100%), but 112GB memory, 48h training, $500 cost. Only with infinite compute.

LoRA: 95% of quality, 8MB memory, 2h training, $20 cost. Practical for most cases.

QLoRA: 90% of quality, 2MB memory, 1h training, $10 cost. When memory/cost is critical.

**"How do you know fine-tuning worked?"**

Validation loss should decrease smoothly and converge. Validation perplexity < 5 is good. On held-out test set, if accuracy improves >10% over base model, worth it. Finally, A/B test: if ft_model wins >60% of human comparisons vs base model, deploy.

