# AI in Financial Services — Compliance, Risk, and Patterns

Building AI systems for banks, fintech, insurance requires balancing **innovation** with **regulation**. Compliance isn't optional; it's the foundation.

---

## The Compliance Landscape

### Key Regulations

| Regulation | Applies To | What It Says |
|---|---|---|
| **SOX (Sarbanes-Oxley)** | Public companies | Financial systems must be auditable; decisions must be explainable |
| **GDPR** | EU data (everyone with EU users) | Right to deletion, right to explanation, data residency |
| **SEC AI Rules** | Investment advisors | AI recommendations must be explainable; human oversight required |
| **GLBA (Gramm-Leach-Bliley)** | Banks | Customer data is private; breaches = fines up to $43K/day per violation |
| **OCC Bulletin 2023-16** | Banks using AI | Banks must validate AI systems; document risks; test for bias |
| **FINRA 4511** | Investment firms | Algorithmic trading must have human override; must be testable |

**Bottom line:** AI in finance = audit trail mandatory, black boxes forbidden, explainability required.

---

## PII in LLM Pipelines — Detection & Redaction

### The Risk

An LLM trained on or exposed to customer data can leak it.

```
Agent is asked: "Summarize this customer document"

Document contains:
"Customer John Doe, SSN 123-45-6789, account 9876543210, owns 100 shares..."

Without redaction, the LLM might:
- Include the data in its response
- Memorize it (training contamination)
- Leak it in another context later
```

### Solution: Pre-Redaction

Redact PII **before** sending to LLM.

```python
import re
from pii_detect import detect_pii

def redact_pii(text):
    """Remove PII before sending to LLM"""

    # SSN: XXX-XX-XXXX
    text = re.sub(r'\d{3}-\d{2}-\d{4}', '[SSN]', text)

    # Account numbers: 10 consecutive digits
    text = re.sub(r'\b\d{10}\b', '[ACCOUNT]', text)

    # Credit card: 16 digits
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC]', text)

    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Use ML-based detection for harder cases
    entities = detect_pii(text)  # Returns: [{"type": "PERSON", "value": "John Doe", "start": 0, "end": 8}]

    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        if entity['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']:
            text = text[:entity['start']] + f"[{entity['type']}]" + text[entity['end']:]

    return text

# Usage
document = "Customer John Doe, SSN 123-45-6789, owns account 9876543210..."
redacted = redact_pii(document)
print(redacted)
# Output: "Customer [PERSON], SSN [SSN], owns account [ACCOUNT]..."

response = llm.invoke(f"Summarize: {redacted}")
```

**PII to redact:**
- SSN, passport, driver's license
- Account numbers, credit cards, debit cards
- Names (persons, companies)
- Phone numbers, email addresses
- Addresses (physical, IP)
- Birthdate, age
- Health info (medical records, diagnosis codes)

---

### Post-Redaction (Response Validation)

Redact PII from LLM **output** too (belt-and-suspenders).

```python
def validate_output(response_text):
    """Check LLM response for accidental PII leakage"""

    pii_patterns = {
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'account': r'\b\d{10}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    }

    found_pii = {}
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, response_text)
        if matches:
            found_pii[pii_type] = matches
            # Log the leak!
            log_security_incident(
                event="PII_LEAK_DETECTED",
                pii_type=pii_type,
                count=len(matches)
            )

    if found_pii:
        raise ValueError(f"PII detected in output: {found_pii}")

    return response_text

# Usage
response = llm.invoke(redacted_input)
validated_response = validate_output(response)
```

---

## Audit Logging — The Compliance Trail

Every AI decision must be logged for regulators.

```python
import json
from datetime import datetime
import uuid

def log_ai_decision(
    user_id,
    query,
    retrieved_documents,
    ai_decision,
    confidence,
    action,
    metadata=None
):
    """Log all AI decisions for audit trail"""

    audit_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "audit_id": str(uuid.uuid4()),  # Unique per decision
        "user_id": user_id,
        "action": action,  # e.g., "LOAN_APPROVAL_RECOMMENDATION"
        "query": query,  # What the user asked
        "retrieved_sources": [
            {
                "document": doc.get("title"),
                "relevance_score": doc.get("score")
            }
            for doc in retrieved_documents[:5]  # Top 5 sources
        ],
        "decision": ai_decision,
        "confidence": confidence,
        "human_override": None,  # Will be filled if human changes decision
        "metadata": metadata or {},
        "version": {
            "model": "claude-3-5-sonnet",
            "prompt_version": "v2.3",
            "guardrails_version": "v1.0"
        }
    }

    # Write to immutable log (append-only)
    write_to_audit_log(json.dumps(audit_log))

    # Duplicate to compliance database
    compliance_db.insert("ai_decisions", audit_log)

    return audit_log["audit_id"]

# Usage
audit_id = log_ai_decision(
    user_id="customer-123",
    query="Should I approve a $100K loan?",
    retrieved_documents=[
        {"title": "Credit Policy", "score": 0.95},
        {"title": "Risk Assessment Guidelines", "score": 0.88}
    ],
    ai_decision="RECOMMEND_APPROVE",
    confidence=0.92,
    action="LOAN_DECISION",
    metadata={"loan_amount": 100000, "customer_credit_score": 750}
)
```

**What to log:**
- **Who** asked (user_id, timestamp)
- **What** they asked (full query)
- **Why** the AI decided (retrieved sources, confidence score)
- **What** the AI decided (decision + confidence)
- **Evidence** (top sources that influenced decision)
- **Audit trail** (human override, if any)

**How to store:**
- Immutable ledger (write-once, append-only)
- Encrypted at rest (compliance requirement)
- Retention: 7+ years (regulatory requirement)
- Access logging (who accessed the audit trail)

---

## Explainability — Show Your Work

Regulators don't accept black-box decisions. You must explain **why** the AI said something.

### Pattern 1: Source Attribution

```python
def generate_explainable_response(query, retrieved_docs):
    """Generate answer with explicit source attribution"""

    # Claude generates answer
    response = llm.invoke(f"""
    Based on these documents, answer the question.
    After your answer, list which documents you relied on.

    {format_docs(retrieved_docs)}

    Question: {query}

    Format:
    [Your answer here]

    Sources:
    - Document 1: [Why it mattered]
    - Document 2: [Why it mattered]
    """)

    # Parse response
    # Response now looks like:
    # "The loan should be approved because the customer has strong credit history.
    #  Sources:
    #  - Credit Policy (2025): Defines approval criteria
    #  - Customer Credit Report: Shows 750 score (exceeds 700 minimum)"

    return response

# Output explicitly shows WHY each source mattered
```

### Pattern 2: Confidence + Decision Reasoning

```python
def structured_decision_output(query):
    """Return structured output with confidence and reasoning"""

    result = {
        "decision": "APPROVE_LOAN",
        "confidence": 0.92,
        "reasoning": {
            "factors_supporting": [
                {
                    "factor": "Credit score > 700",
                    "value": "750",
                    "weight": "high"
                },
                {
                    "factor": "Debt-to-income < 40%",
                    "value": "35%",
                    "weight": "high"
                }
            ],
            "factors_opposing": [
                {
                    "factor": "Short employment history",
                    "value": "2 years",
                    "weight": "low"
                }
            ],
            "key_policy": "Credit Policy v2.5 Section 3.1.2",
            "human_review_required": False,
            "escalation_reason": None
        }
    }

    return result

# Regulator sees exact reasoning for the decision
```

### Pattern 3: Sensitivity Analysis

```python
def explain_sensitivity(decision_inputs):
    """Show how decision changes with input variations"""

    base_decision = make_decision(decision_inputs)

    sensitivities = {}
    for input_name, input_value in decision_inputs.items():
        # Vary each input +10%, -10%
        for variation in [0.9, 1.1]:
            modified_inputs = decision_inputs.copy()
            modified_inputs[input_name] = input_value * variation

            new_decision = make_decision(modified_inputs)

            sensitivities[input_name] = {
                "base": base_decision,
                f"at_{variation}x": new_decision,
                "impact": "DECISION_CHANGES" if base_decision != new_decision else "DECISION_STABLE"
            }

    return sensitivities

# Output:
# {
#   "credit_score": {
#     "base": "APPROVE",
#     "at_0.9x": "REVIEW",
#     "impact": "DECISION_CHANGES"
#   },
#   "debt_to_income": {
#     "base": "APPROVE",
#     "at_1.1x": "APPROVE",
#     "impact": "DECISION_STABLE"
#   }
# }
```

---

## Data Residency — Keep Data Inside Borders

**GDPR requirement:** EU customer data must stay in EU.

**Financial services requirement:** US customer data shouldn't touch Chinese servers (political risk).

### Solution: Regional Bedrock

```python
import boto3

# US data → Bedrock in us-east-1
us_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# EU data → Bedrock in eu-west-1
eu_client = boto3.client("bedrock-runtime", region_name="eu-west-1")

def process_customer_data(customer_id, region):
    """Route to correct regional endpoint"""

    if region == "US":
        client = us_client
    elif region == "EU":
        client = eu_client
    else:
        raise ValueError(f"Unsupported region: {region}")

    response = client.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps({...})
    )

    return response
```

**Key rules:**
- **GDPR (EU data):** Must process in EU region (eu-west-1, eu-central-1)
- **China:** Never send data to China-based services (compliance risk)
- **Banking:** Keep data in US if regulated by OCC (US banks)
- **Cross-border:** Document residency requirements per customer

---

## Bias Detection & Monitoring

AI systems can inadvertently discriminate.

```python
def monitor_for_bias(decisions_batch):
    """Check if AI decisions are biased by protected attributes"""

    # Protected attributes: race, gender, age, disability
    protected_attributes = ["age", "gender", "zip_code"]

    approval_rates = {}
    for attribute in protected_attributes:
        groups = {}
        for decision in decisions_batch:
            attr_value = decision.get(attribute)
            if attr_value not in groups:
                groups[attr_value] = {"approved": 0, "denied": 0}

            if decision["decision"] == "APPROVE":
                groups[attr_value]["approved"] += 1
            else:
                groups[attr_value]["denied"] += 1

        # Calculate approval rate per group
        approval_rates[attribute] = {}
        for group, counts in groups.items():
            total = counts["approved"] + counts["denied"]
            rate = counts["approved"] / total if total > 0 else 0
            approval_rates[attribute][group] = rate

    # Detect disparate impact (legal threshold: 80% rule)
    for attribute, rates in approval_rates.items():
        if rates:
            max_rate = max(rates.values())
            min_rate = min(rates.values())
            if min_rate > 0 and min_rate / max_rate < 0.80:
                # Disparate impact detected!
                log_bias_alert(
                    attribute=attribute,
                    impact_ratio=min_rate / max_rate,
                    groups=rates
                )

    return approval_rates

# Output:
# {
#   "age": {
#     "18-25": 0.75,   # 75% approval
#     "26-35": 0.92    # 92% approval → Disparate impact!
#   }
# }
```

---

## System Design Interview Pattern

**Question:** "Design a loan approval system for a bank using AI."

**Your answer:**

```
1. System requirements:
   - 1000 loan applications/day
   - Must explain every decision
   - Comply with Fair Lending rules
   - Audit trail for regulators

2. Architecture:
   - Bedrock Agent: Retrieves policies, customer data
   - Guardrails: Block PII, flag suspicious patterns
   - Explainability: Return sources + confidence
   - Logging: Every decision logged with reasoning

3. Workflow:
   Input: Customer info (income, credit score, employment)
   ↓
   Redaction: Remove PII before sending to LLM
   ↓
   Retrieval: Fetch Credit Policy, Risk Guidelines
   ↓
   Decision: LLM determines APPROVE/DENY/REVIEW
   ↓
   Audit Log: Log decision, sources, confidence, model version
   ↓
   Bias Check: Monitor approval rates by age/gender/zip code
   ↓
   Output: Decision + explanation + sources + audit_id
   ↓
   Human Override: Loan officer reviews, can override with audit

4. Compliance:
   - PII redaction (pre + post)
   - Full audit trail (7-year retention)
   - Explainability (every decision justified)
   - Bias monitoring (Fair Lending Act)
   - Data residency (US data stays in US)

5. Guardrails:
   - Block recommendations for illegal discrimination
   - Flag decisions below 75% confidence
   - Escalate to human for policy violations

6. Scaling:
   - Bedrock handles 1K requests/sec
   - Cost: $0.003 per decision
   - Latency: <500ms target
```

This shows: **compliance-first thinking + AI systems + regulatory expertise**.

---

## Production Checklist

- [ ] PII redaction implemented (pre + post LLM)
- [ ] Audit logging mandatory (immutable, encrypted)
- [ ] Explainability built-in (sources, confidence, reasoning)
- [ ] Guardrails configured (PII, discrimination, policy violations)
- [ ] Bias monitoring in place (check approval rates by protected attributes)
- [ ] Data residency compliant (regional endpoints)
- [ ] Model versioning documented (model, prompt, guardrails versions in logs)
- [ ] Human override mechanism available
- [ ] Regular compliance reviews (monthly audits)
- [ ] Documentation for regulators (system description, validation results, risk assessment)
