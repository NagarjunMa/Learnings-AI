# RAG Evals — Comprehensive Framework

> Evaluation is where RAG systems are won or lost. Two systems with identical architecture can differ 100x in reliability based on how rigorously they were evaluated. This note covers design, implementation, and interview framing for RAG eval strategies.

---

## Part 1: Why RAG Evals Matter

### The Hallucination Problem

```
Without evals, you discover failures in production.

User asks: "Can I transfer to the night shift?"
RAG answer: "Yes, transfers happen automatically every month."
Reality: Transfers require manager approval and take 2-3 weeks.

One wrong answer = compliance issue + user frustration.
```

### Why RAG Is Different From LLM Evals

Standard LLM:
```
Input: "What is 2+2?" 
Output: "4"
Correct: Yes/No (binary)
```

RAG system:
```
Query: "Can I transfer to night shift?"
  ↓
Retrieval: Fetch docs (right docs? wrong docs? partial docs?)
  ↓
Generation: LLM answers from docs (uses them? ignores them? hallucinate?)
  ↓
Output: "Yes, transfers happen automatically"

Evaluation requires checking THREE things:
  1. Did we retrieve the right source docs?
  2. Did the answer use only those docs?
  3. Is the answer actually correct per the docs?

Fail at step 1 → garbage in, garbage out (hallucination)
Fail at step 2 → hallucination even with good docs
Fail at step 3 → wrong answer despite good docs + grounding
```

### Eval Tiers (Progressive Quality)

```
Tier 1: Retrieval Quality (Is RAG finding relevant docs?)
  Metric: Precision@K, Recall@K
  Cost: Low (automatic, no LLM calls)
  Scope: 50 queries, 10 minutes
  
Tier 2: Groundedness (Does answer use retrieved docs?)
  Metric: % claims verified against sources
  Cost: Medium (LLM-as-judge)
  Scope: 30 answers, 20 minutes
  
Tier 3: Hallucination Detection (Are there claims outside docs?)
  Metric: % claims without source
  Cost: Medium (LLM extraction + checking)
  Scope: 30 answers, 20 minutes
  
Tier 4: User Satisfaction (Does it actually help?)
  Metric: Human rating (0-3 scale)
  Cost: High (manual, no automation)
  Scope: 20-30 answers, 2-3 hours
  
Cadence: Tier 1 daily. Tier 2-3 weekly. Tier 4 before launch + quarterly.
```

---

## Part 2: Eval Design Methodology

### The 5-Step Framework

**Step 1: Define Requirements (From product + engineering)**

```
Example — Frontline Worker Chatbot:
  ✓ Mobile users, spotty networks
  ✓ Zero tolerance for broken job assignment info
  ✓ Latency < 2 seconds (mobile patience)
  ✓ No jargon (frontline = non-technical audience)
  ✓ Always have CTA (what to do next)

Translate to eval targets:
  Hallucination rate: < 2% (can't afford misinformation)
  Groundedness: > 95% (claims must cite source)
  Latency p95: < 2000ms (network limit)
  Answer readability: 0 jargon, < 300 chars
```

**Step 2: Build Ground Truth Test Set**

```
Test set = (Query, Expected Sources, Expected Answer, Quality Score)

Size: 30-50 queries minimum
  10-20 per major topic/category
  5-10 edge cases / adversarial
  
Source: Real user questions (Slack, email, support logs)
  NOT made-up questions
  
Coverage: All query types system must handle

Example (Frontline Worker chatbot):
  Category: Job Assignments (10 queries)
    "What jobs are available in my area?"
    "Can I pick a job from the app?"
    "What do I do if I can't find jobs?" 
    ...
  
  Category: Certifications (10 queries)
    "What cert do I need for supervisor?"
    "How long does cert training take?"
    "Who pays for the training?"
    ...
  
  Category: Benefits (10 queries)
    "How much vacation time do I get?"
    "When do I get my first paycheck?"
    "Can I enroll in 401k?"
    ...
  
  Edge cases (5 queries):
    "Can I work two jobs simultaneously?" (ambiguous policy)
    "What if my shift gets cancelled?" (rare scenario)
    "Can I request time off retroactively?" (tricky timeline)
```

**Step 3: Implement Evaluators (Automatic Metrics)**

```
Evaluator 1: Retrieval Scorer
  Input: Query, Retrieved docs, Ground truth sources
  Output: Score 0-1 (does retrieved match ground truth?)
  Method: Check if ground truth IDs appear in top-K
  
Evaluator 2: Hallucination Detector
  Input: Answer text, Retrieved doc text
  Output: % of claims not found in docs
  Method: Extract claims via LLM, check each against docs
  
Evaluator 3: Groundedness Scorer  
  Input: Answer, Retrieved docs
  Output: Score 0-5 (does answer use ONLY the docs?)
  Method: LLM-as-judge with rubric
  
Evaluator 4: Latency Tracker
  Input: Execution logs
  Output: p50, p95, p99 latency + per-component breakdown
  Method: Instrument embedding, search, generation steps
```

**Step 4: Establish Baseline**

```
Run all evals on current system.

Example output:
  Retrieval Precision@3: 82% (some false positives)
  Hallucination rate: 8% (unacceptable)
  Groundedness score: 3.2/5.0 (barely acceptable)
  Latency p95: 1800ms (good)
  
Identify failure modes:
  ✗ Hallucinations in benefits questions (lawyer-ish topics)
  ✗ Retrieval misses certification docs (wrong KB chunk size)
  ✓ Latency healthy (no bottleneck)
  
Prioritize fixes:
  Priority 1: Fix retrieval chunking
  Priority 2: Add hallucination detection to prompt
  Priority 3: Monitor latency over time
```

**Step 5: Iterate**

```
Per failure mode, try one change:

Change 1: Improve retrieval
  ├─ Smaller chunks (256 tokens instead of 512)
  ├─ Hybrid search (BM25 + semantic)
  └─ Re-rank with LLM
  Re-run Tier 1 evals
  
Change 2: Improve generation
  ├─ Add prompt: "Only use docs. If not in docs, say so."
  ├─ Extract claims, verify before returning
  └─ Change model (better = less hallucination)
  Re-run Tier 2-3 evals
  
Change 3: Monitor
  ├─ Log all retrievals
  ├─ Track hallucination rate by category
  └─ Alert if > 5%
  
Measure: Did the change improve the metric?
  ✓ Precision@3: 82% → 91% (good fix)
  ✗ Hallucination: 8% → 7% (marginal, try prompt instead)
```

---

## Part 3: Project 1 — Frontline Worker Chatbot

### Requirements & Context

From behavioral interview Story 1:
- Client: Non-tech company, frontline workers (field operations)
- Product: Conversational AI for job assignments, certifications, benefits
- Timeline: 6 weeks to production
- Constraint: Zero tolerance for broken UX (workers can't use if unreliable)
- Platform: Mobile-first, AWS Bedrock + Lambda
- Success metric: Onboarding time drops from 4 hours to 15 minutes

### Why Evals Are Critical Here

```
One wrong answer costs real money:
- Assign worker to wrong job → contract breach + customer complaint
- Wrong benefit info → compliance issue + legal risk
- Latency > 2 sec on mobile → app bounces, users give up

Requirements translate to eval targets:
  Hallucination rate: < 2% (can't afford misinformation on job/benefit facts)
  Groundedness: > 95% (every claim must cite policy)
  Mobile latency: < 2000ms p95 (network + device constraints)
  Answer readability: No jargon, max 300 chars, clear CTA
```

### Test Set (30 Queries)

**Topic 1: Job Assignments (10 queries)**

```
1. "What jobs are available in my area?"
   Expected docs: [jobs-api.md, location-matching.md]
   Expected answer: Lists available jobs, includes CTA: "Tap to apply"
   
2. "Can I pick any job from the list?"
   Expected docs: [job-selection-rules.md, eligibility.md]
   Expected answer: Explains eligibility rules, mentions certifications if relevant
   
3. "What if I can't find jobs near me?"
   Expected docs: [remote-jobs.md, relocation-policy.md]
   Expected answer: Options (remote, relocation, alert for new jobs)
   
4. "How do I apply for a job?"
   Expected docs: [apply-flow.md, system-guide.md]
   Expected answer: Step-by-step (mobile UX), expected confirmation time
   
5. "Can I apply for multiple jobs at once?"
   Expected docs: [application-rules.md, constraints.md]
   Expected answer: Clear yes/no, limit if any, reasoning
   
6. "What happens after I apply?"
   Expected docs: [application-status.md, notifications.md]
   Expected answer: Timeline, where to check status
   
7. "Can I withdraw an application?"
   Expected docs: [withdraw-policy.md]
   Expected answer: Yes/no + when cutoff is
   
8. "What's the pay for this job?"
   Expected docs: [compensation.md, job-details.md]
   Expected answer: Hourly rate or salary, any bonuses
   
9. "Can I get scheduled for multiple jobs same day?"
   Expected docs: [scheduling-rules.md, shift-overlap.md]
   Expected answer: Clear policy, reasoning if no
   
10. "How do I see my scheduled jobs?"
    Expected docs: [calendar-feature.md, schedule-view.md]
    Expected answer: How to access, how to interpret timeline
```

**Topic 2: Certifications (10 queries)**

```
11. "What certifications do I need for the supervisor role?"
    Expected docs: [role-requirements.md, cert-supervisor.md]
    Expected answer: List specific certs, timeline to complete
    
12. "How long does [cert name] training take?"
    Expected docs: [cert-duration.md, training-timeline.md]
    Expected answer: Hours required, how many weeks typical
    
13. "Who pays for certification training?"
    Expected docs: [training-cost-policy.md, benefits.md]
    Expected answer: Company covers, partial, or employee pays
    
14. "Can I do training while working?"
    Expected docs: [training-schedule.md, work-training-balance.md]
    Expected answer: Yes, options (online, evenings, etc.)
    
15. "What if I fail the cert exam?"
    Expected docs: [cert-retake-policy.md, failure-process.md]
    Expected answer: Can retake, wait time, costs
    
16. "How do I enroll in certification?"
    Expected docs: [enroll-cert.md, system-guide.md]
    Expected answer: Steps in app + contact info if questions
    
17. "Do certifications expire?"
    Expected docs: [cert-renewal.md, expiration-policy.md]
    Expected answer: Yes/no, renewal timeline if yes
    
18. "What cert should I get for my career path?"
    Expected docs: [career-paths.md, cert-recommendations.md]
    Expected answer: Path-specific certs + reasoning
    
19. "Can I get certified while not actively working?"
    Expected docs: [inactive-worker-policy.md, training-eligibility.md]
    Expected answer: Yes/no, eligibility rules
    
20. "How do I prove I have my certification?"
    Expected docs: [cert-verification.md, credential-management.md]
    Expected answer: How to see in app + print/share options
```

**Topic 3: Benefits (10 queries)**

```
21. "How much vacation time do I get per year?"
    Expected docs: [pto-policy.md, time-off.md]
    Expected answer: Hours or days, any conditions (tenure, role)
    
22. "When do I get my first paycheck?"
    Expected docs: [payroll-schedule.md, first-payment.md]
    Expected answer: Days after hire, payment method
    
23. "Can I enroll in 401k?"
    Expected docs: [retirement-plans.md, 401k-enrollment.md]
    Expected answer: Yes/no, eligibility (vesting, company match %)
    
24. "What health insurance is available?"
    Expected docs: [health-insurance.md, benefits-comparison.md]
    Expected answer: Plans available, cost per paycheck, coverage
    
25. "Can I request time off retroactively?" (EDGE CASE)
    Expected docs: [pto-policy.md, time-off-rules.md]
    Expected answer: No retroactive, but what to do if missed deadline
    
26. "What happens to unused PTO at year end?"
    Expected docs: [pto-payout.md, unused-time-policy.md]
    Expected answer: Carried over, capped, or forfeited + exceptions
    
27. "Can I work fewer hours to reduce benefits cost?"
    Expected docs: [part-time-policy.md, benefits-eligibility.md]
    Expected answer: No, or reduced benefits tier + costs
    
28. "Do I get paid for holidays?"
    Expected docs: [holiday-policy.md, paid-holidays.md]
    Expected answer: List holidays, pay rate (time-and-a-half?)
    
29. "How do I enroll in dependent care FSA?"
    Expected docs: [fsa-dependent-care.md, benefits-enrollment.md]
    Expected answer: Enrollment window, annual limit, tax advantage
    
30. "What if I'm injured on the job?" (ADVERSARIAL)
    Expected docs: [workers-comp.md, incident-reporting.md]
    Expected answer: Report immediately + steps, coverage, timeline
```

### Evaluator 1: Retrieval Quality

**Metric: Precision@3 (are top 3 retrieved docs relevant?)**

```python
def eval_retrieval_quality(query, retrieved_doc_ids, ground_truth_docs):
    """
    retrieved_doc_ids: [docs returned by vector search, top 3]
    ground_truth_docs: [docs that SHOULD be returned for this query]
    
    Returns: 0.0 (no match), 0.33 (1/3 match), 0.67 (2/3 match), 1.0 (perfect)
    """
    matches = sum(1 for doc in retrieved_doc_ids if doc in ground_truth_docs)
    return matches / 3.0

# Per-category scores:
Job assignment queries avg Precision@3: 0.85 (good)
Certification queries avg Precision@3: 0.78 (needs work → chunking issue)
Benefits queries avg Precision@3: 0.88 (good)

Target: > 85% overall (< 15% error rate acceptable)
```

**Action if below target:**
- Certification docs retrieval failing → check chunk size (maybe too coarse)
- Try semantic chunking for cert docs
- Add keyword boost for cert-related queries

### Evaluator 2: Hallucination Detection

**Metric: % of claims not found in retrieved docs**

```
Answer: "Certifications expire after 3 years and cost $50 each. 
         You can retake if you fail. Training is usually 40 hours."

Extracted claims:
  1. "Certifications expire after 3 years" → ✓ found in cert-renewal.md
  2. "Cost $50 each" → ✗ NOT found (hallucination!)
  3. "Can retake if fail" → ✓ found in cert-retake-policy.md
  4. "Training 40 hours" → ✓ found in training-timeline.md

Hallucination rate: 1/4 = 25% (UNACCEPTABLE)
Target: < 2% (basically zero hallucinations)
```

**Implementation: LLM claim extraction**

```
Prompt (to Claude):
  """
  Extract all factual claims from this answer:
  
  [Answer text]
  
  Respond with a JSON list:
  [
    {"claim": "Certifications expire after 3 years", "category": "policy"},
    {"claim": "Cost $50 each", "category": "pricing"},
    ...
  ]
  """

For each claim:
  - Search docs for claim
  - Exact match? Paraphrase match? No match?
  - Flag unmatched claims as hallucinations
```

### Evaluator 3: Groundedness Score (LLM-as-Judge)

**Rubric (0-5 scale):**

```
5 = Answer uses ONLY retrieved docs, cites specifics, no hallucination
4 = Uses docs primarily, minor detail from training data (acceptable)
3 = Uses docs, but supplements with likely accurate general knowledge
2 = Ignores docs, relies heavily on training data
1 = Contradicts retrieved docs
0 = Nonsense or refusal

Prompt:
  """
  Rate this answer on groundedness (0-5):
  
  Query: "What certifications do I need for supervisor?"
  
  Retrieved documents:
  - cert-supervisor.md: "Required: Safety (Level 2), People Management, Communication"
  - role-requirements.md: "Supervisor must have 2+ years experience and required certs"
  
  Generated answer: "For the supervisor role, you need three certifications: 
  Safety Level 2, People Management, and Communication. These are required by 
  company policy. You'll also need 2+ years of experience. Training typically 
  takes 4-6 weeks total."
  
  Rate 0-5 and explain: [Assistant response]
  """

Target: > 4.5/5.0 average
  (Almost entirely grounded, minor acceptable paraphrasing)
```

### Evaluator 4: Mobile UX Score

**Criteria (all-or-nothing per answer):**

```
✓ Answer length ≤ 300 characters (fits one screen)
✓ No jargon (no "UI", "API", "aggregation", "tenure", etc.)
✓ CTA clear ("Tap to X", "Ask your manager", "Contact HR")
✓ Starts with direct answer (not background story)

Example PASS:
  "You need 3 certs: Safety Level 2, People Management, Communication. 
   Training takes 4-6 weeks total. Tap here to enroll."
   [293 characters] ✓ Length ✓ No jargon ✓ CTA ✓ Direct

Example FAIL:
  "The company has a robust framework for professional development where 
   supervisors must demonstrate competency across three domains..."
   [Too long, jargon-heavy, no CTA] ✗ ✗ ✗ ✗

Target: 95% of answers pass all 4 criteria
```

### Evaluator 5: Latency (Component Breakdown)

```
Instrumentation points:
  t0: Query received
  t1: After embedding query (embedding latency)
  t2: After vector search (retrieval latency)
  t3: After LLM generation (generation latency)
  t4: Response sent (total)

Example latency profile:
  Embedding: 50ms
  Retrieval: 150ms
  Generation: 1200ms
  Total: 1400ms ✓ < 2000ms p95 target

If latency spiked:
  - Check which component. Usually generation (LLM latency)
  - Monitor retrieval quality (bad retrievals → longer generation)
  - Check for cold starts (Lambda)

Target: p95 < 2000ms, p99 < 3000ms
```

### Baseline Results (Example)

```
Retrieval Precision@3: 82%
  ✓ Good for job/benefits
  ✗ Weak for certifications (78%)
  → Action: Check cert doc chunking

Hallucination rate: 6%
  ✗ Unacceptable (target < 2%)
  → Action: Add hallucination detection to prompt
  
Groundedness score: 3.4/5.0
  ✓ Acceptable but low (target > 4.5)
  → Action: Prompt improvement (add "cite sources" instruction)
  
Mobile UX score: 88%
  ~ Acceptable (target 95%)
  → Action: Trim 12% of answers, remove jargon
  
Latency p95: 1650ms
  ✓ Good (target 2000ms)
  → No action needed
```

### Iteration 1: Fix Retrieval

**Change: Reduce chunk size for certifications from 512 to 256 tokens**

```
Precision@3: 82% → 89%
  ✓ Certification retrieval improved to 85%
  
Side effect: More chunks, but retrieval latency still 150ms
```

### Iteration 2: Fix Hallucination via Prompt

**Change: Add instruction to prompt**

```
Original prompt:
  "You are a helpful assistant. Answer the question based on the docs below."

New prompt:
  "You are a helpful assistant. Answer the question based ONLY on the docs 
   below. If the answer is not in the docs, say 'I don't have information 
   about that' instead of guessing. Do not add information from your training."

Hallucination rate: 6% → 2%
  ✓ Acceptable now
  
Groundedness score: 3.4 → 4.7
  ✓ Major improvement
```

### Iteration 3: Fix Mobile UX

**Change: Post-processing to trim and simplify answers**

```python
def simplify_for_mobile(answer):
    # Remove filler words (actually, essentially, basically)
    # Reduce length if > 300 chars (truncate + add "...")
    # Replace jargon (UI → app, aggregation → combined)
    return trimmed_answer

Mobile UX score: 88% → 96%
  ✓ Acceptable now
```

### Final Eval Results

```
Retrieval Precision@3: 89% ✓
Hallucination rate: 2% ✓
Groundedness score: 4.7/5 ✓
Mobile UX score: 96% ✓
Latency p95: 1650ms ✓

All metrics acceptable. Ready for production.
```

---

## Part 4: Project 2 — EV Charging Marketplace RAG

### Requirements & Context

From behavioral interview Story 6:
- Brief: One sentence: "Build something to help EV owners find chargers and homeowners rent their spots"
- No API specs, data schema, or user research (ambiguous requirements)
- Solution: You mapped both user types separately before building
- Outcome: Prevented two major scope changes mid-build

### The Unique Challenge: Dual-Sided System

```
One RAG chatbot, TWO very different users:

EV Owner asks: "Where can I charge?"
  → Needs: Charger locations, pricing, availability, reviews
  → Tone: Practical, quick answers, actionable CTAs
  
Homeowner asks: "How do I rent my driveway?"
  → Needs: Rental process, liability, earnings, equipment setup
  → Tone: Educational, builds confidence, addresses concerns

Critical requirement: System must NOT confuse personas.

If homeowner gets EV owner's answer:
  "You can charge from $0.10-$0.50 per kWh" → useless to homeowner

If EV owner gets homeowner's answer:
  "You earn money by letting others charge at your location" → confuses them

Eval must verify persona-specific routing.
```

### Test Set (30 Queries)

**EV Owner Queries (15)**

```
1. "Where can I charge my Tesla?" → Charger locations
2. "What's the cheapest place to charge?" → Pricing comparison
3. "Can I reserve a charger in advance?" → Availability/booking
4. "Do you have chargers near [location]?" → Geo-specific search
5. "What payment methods do you accept?" → Payment options
6. "How fast is Level 3 charging?" → Technical specs
7. "Are there chargers for non-Tesla cars?" → Compatibility
8. "What if the charger breaks while I'm charging?" → Troubleshooting/liability
9. "Can I see real-time availability?" → Live status
10. "What happens if I park too long after charging?" → Fees/policies
11. "Do you have chargers at airports?" → Specific locations
12. "Can I charge with the app offline?" → Technical capability
13. "What's your cancellation policy?" → Booking rules
14. "Do you offer monthly passes?" → Pricing plans
15. "How do I report a charger not working?" → Support process
```

**Homeowner Queries (15)**

```
16. "How do I list my driveway for charging?" → Setup process
17. "What liability insurance do I need?" → Legal/risk
18. "How much can I earn per month?" → Income expectations
19. "Do you handle all the equipment?" → Hardware setup
20. "What if someone damages my property?" → Risk mitigation
21. "Can I set my own pricing?" → Pricing control
22. "How often does someone want to charge?" → Demand/expectations
23. "When do I get paid?" → Payment schedule
24. "What if I want to remove my listing?" → Offboarding
25. "Do I need a special driveway setup?" → Technical requirements
26. "What taxes do I owe?" → Tax implications
27. "Can I limit charging hours?" → Control/scheduling
28. "Are there any upfront costs?" → Costs to homeowner
29. "How does the app work on my end?" → Homeowner UX
30. "What if a renter is rude?" → Customer service
```

**Adversarial/Ambiguous (5, not counted in the 30)**

```
A1. "What's the cost?" 
    → Ambiguous: Could be EV owner asking charging cost OR 
      homeowner asking what it costs to participate
    
A2. "How do I use your service?"
    → Both personas ask this differently
    
A3. "When will someone come to my/at a charger?"
    → EV owner (when will they be available for charging)
    vs Homeowner (when will someone rent my spot)
    
A4. "Do you have reviews?"
    → EV owner (charger/host reviews)
    vs Homeowner (tenant reviews of me)
    
A5. "What's the payment process?"
    → EV owner (pay to charge)
    vs Homeowner (receive payment)
```

### Evaluator 1: Persona Routing Accuracy

**Metric: % of queries routed to correct persona**

```
Critical: This must be 100% (or very close).
Wrong persona = wrong answer = user confusion.

Implementation:
  1. System identifies persona from query (or user profile)
  2. Routes to persona-specific system prompt
  3. Evaluator checks: Did it use the right prompt?
  
Example:
  Query: "Where can I charge my Tesla?"
  System identifies: EV Owner (keyword "charge Tesla")
  Routes to: EV-Owner-Prompt (retrieves charger locations)
  Answer: "Chargers available in [locations]..."
  
  Verdict: ✓ Correct routing
  
Example FAIL:
  Query: "How do I list my driveway?"
  System identifies: Homeowner (keyword "list driveway")
  Routes to: Homeowner-Prompt
  But answer says: "You can find chargers at..."
  
  Verdict: ✗ Wrong prompt used or LLM ignored persona
```

### Evaluator 2: Cross-Persona Confusion Test

**Metric: Handling of adversarial ambiguous queries**

```
For ambiguous queries (A1-A5), system should either:
  a) Ask clarifying question: "Are you looking to charge or to host?"
  b) Use context (from history): "Based on your profile as homeowner..."
  c) Default to majority persona: Assume EV owner, clarify if needed

Example:
  Query: "What's the cost?"
  
  Bad answer: 
    "Charging costs $0.10-0.50/kWh at most locations."
    (Assumes EV owner, no clarification if they're homeowner)
  
  Good answer:
    "Are you asking about charging costs, or how much you can earn 
     as a host? I can help with either!"
  
  Better answer (if user profile known):
    "As a registered homeowner, you can earn $X-Y per rental. 
     Here's how your earnings work..."

Target: 90%+ of ambiguous queries either:
  - Get a clarifying follow-up, OR
  - Use context correctly, OR
  - Default + offer clarification

This prevents silent persona confusion.
```

### Evaluator 3: Knowledge Base Coverage Matrix

**Metric: Do docs exist + retrieval works for each topic?**

```
Matrix format:

Topic                       EV Owner Coverage    Homeowner Coverage    Status
────────────────────────────────────────────────────────────────────────────
Locations & Availability         ✓ Good               ✗ Not applicable
Pricing & Payment               ✓ Good                ✓ Good
Technical Specs                 ✓ Good                ✗ Not applicable
Liability & Insurance           ✗ Missing             ✓ Good
Income & Earnings               ✗ Not applicable      ✓ Good
Setup & Requirements            ✗ Not applicable      ~ Incomplete
Customer Service               ✓ Good                 ✓ Good

Red flags:
  ✗ Missing docs = retrieval will fail or hallucinate
  ~ Incomplete docs = will get partial answers
  
Action:
  - Write missing docs
  - Improve KB for incomplete topics
  - Re-test retrieval for those topics
```

### Evaluator 4: Answer Consistency (Same Fact, Different Angle)

**Metric: Does the same fact get communicated correctly to both personas?**

```
Fact: "Charging at a premium downtown location costs $0.50/kWh"

EV Owner answer:
  "Premium downtown locations cost $0.50/kWh. That's 25% higher 
   than standard locations ($0.40) but you get faster chargers. 
   Would you like nearby standard options?"

Homeowner answer:
  "If you host at a premium downtown location, you earn $0.50/kWh 
   when EVs charge there. That's a premium rate because the location 
   is valuable (downtown). You keep the earnings after platform fees."

Evaluation:
  ✓ Fact matches ($0.50)
  ✓ Tone appropriate (EV owner: practical choice; Homeowner: opportunity)
  ✓ Context fits (EV owner compares to alternatives; Homeowner explains earning)

Run this test on 10 shared facts:
  Pricing, fees, platform policies, availability, etc.
  
Target: 100% consistency on base facts
         100% appropriateness of tone/framing
```

### Evaluator 5: Persona-Specific Latency

**Metric: Does persona routing add overhead?**

```
Two latency paths:

EV Owner path:
  Query → Identify as EV Owner (10ms) → EV prompt → Retrieve chargers 
  → Generate → Total: 1400ms

Homeowner path:
  Query → Identify as Homeowner (10ms) → Homeowner prompt → Retrieve 
  earnings/setup → Generate → Total: 1600ms

(Homeowner queries slightly slower because they're more complex)

Target: Persona routing < 50ms overhead
        Total latency < 2000ms for both
        
If latency spiked:
  - Check identification logic (ML model if used)
  - Check if homeowner KB is slower to search
  - Monitor for cold starts
```

### Baseline Results Example

```
Persona Routing Accuracy: 92%
  ✗ Below target (need 100%)
  Failure pattern: Ambiguous queries routed incorrectly
  → Action: Add clarifying question for ambiguous cases
  
Cross-Persona Confusion: 85%
  ✗ 15% of ambiguous queries silent fail (no clarification)
  → Action: Implement fallback clarifying question
  
KB Coverage: 
  EV Owner: 95% (one missing topic: insurance)
  Homeowner: 88% (gaps in setup requirements)
  → Action: Add missing docs for both
  
Answer Consistency: 98%
  ✓ Good, fact consistency solid
  ✗ 2 cases where tone was wrong (homeowner got EV-centric tone)
  → Action: Review homeowner prompts
  
Latency (EV Owner): 1450ms ✓
Latency (Homeowner): 1680ms ✓
```

### Iteration 1: Fix Persona Routing

**Change: Add clarifying question for ambiguous queries**

```
Detection rule:
  If query contains ["cost", "payment", "price", "earn", "money"]
  AND not in user profile/history
  → Ask: "Are you asking about charging costs or earning potential?"

Result:
  Routing accuracy: 92% → 98%
  (Clear path now for ambiguous cases)
```

### Iteration 2: Add Missing KB Docs

**Change: Write insurance/liability docs for both personas**

```
New docs:
  - insurance-ev-owner.md (public liability if charger breaks)
  - insurance-homeowner.md (liability if host's property damaged)
  
KB Coverage:
  EV Owner: 95% → 100%
  Homeowner: 88% → 95%
```

### Iteration 3: Review Homeowner Prompts

**Change: Ensure homeowner prompt emphasizes earning/control**

```
Old homeowner prompt:
  "Answer questions about hosting a charger."

New homeowner prompt:
  "Answer questions about earning money by hosting chargers. 
   Emphasize control, earnings, and how homeowners benefit. 
   Frame as opportunity, not obligation."

Tone consistency: 98% → 100%
```

### Final Eval Results

```
Persona Routing Accuracy: 98% ✓
Cross-Persona Confusion: 98% ✓
KB Coverage (EV Owner): 100% ✓
KB Coverage (Homeowner): 95% ✓
Answer Consistency: 100% ✓
Latency (both): < 1700ms ✓

Status: Ready for production.
```

---

## Part 5: Implementation & Tools

### Code Template: Hallucination Detector

```python
from anthropic import Anthropic

def extract_claims(answer_text: str) -> list[dict]:
    """Use Claude to extract factual claims from answer."""
    client = Anthropic()
    
    response = client.messages.create(
        model="claude-opus",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Extract all factual claims from this answer.
            
Answer: {answer_text}

Respond with JSON list of objects with 'claim' and 'category' keys:
[
  {{"claim": "...", "category": "policy"}},
  {{"claim": "...", "category": "pricing"}}
]"""
        }]
    )
    
    import json
    return json.loads(response.content[0].text)

def check_claim_in_docs(claim: str, doc_texts: list[str]) -> bool:
    """Check if claim appears in any retrieved doc."""
    for doc in doc_texts:
        if claim.lower() in doc.lower():
            return True
    return False

def eval_hallucination(answer: str, retrieved_docs: list[str]) -> dict:
    """
    Returns:
    {
      'hallucination_rate': float (0-1),
      'total_claims': int,
      'hallucinated_claims': list[str]
    }
    """
    claims = extract_claims(answer)
    hallucinated = []
    
    for claim_obj in claims:
        if not check_claim_in_docs(claim_obj['claim'], retrieved_docs):
            hallucinated.append(claim_obj['claim'])
    
    return {
        'hallucination_rate': len(hallucinated) / len(claims) if claims else 0,
        'total_claims': len(claims),
        'hallucinated_claims': hallucinated
    }

# Usage:
answer = "Certifications expire after 3 years and cost $50."
docs = ["Certifications must be renewed...", "There is no cost for..."]
result = eval_hallucination(answer, docs)
print(f"Hallucination rate: {result['hallucination_rate']:.1%}")
```

### Code Template: LLM-as-Judge Groundedness

```python
def eval_groundedness_llm_judge(
    query: str,
    retrieved_docs: list[str],
    answer: str,
    rubric: str = None
) -> dict:
    """
    Use LLM to score groundedness on 0-5 scale.
    """
    client = Anthropic()
    
    default_rubric = """
5 = Uses ONLY retrieved docs, cites specifics, zero hallucination
4 = Mostly uses docs, minor training-data supplement (acceptable)
3 = Uses docs but adds likely-correct general knowledge
2 = Mostly ignores docs, relies on training data
1 = Contradicts retrieved docs
0 = Nonsense or refusal
"""
    
    rubric = rubric or default_rubric
    
    doc_context = "\n\n---\n\n".join([
        f"Document {i+1}:\n{doc[:500]}"  # Truncate long docs
        for i, doc in enumerate(retrieved_docs)
    ])
    
    prompt = f"""
Score this answer on groundedness (0-5).

Rubric:
{rubric}

Query: {query}

Retrieved Documents:
{doc_context}

Generated Answer: {answer}

Respond with JSON:
{{
  "score": <0-5>,
  "reasoning": "<explain why this score>"
}}
"""
    
    response = client.messages.create(
        model="claude-opus",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    import json
    result = json.loads(response.content[0].text)
    return {
        'query': query,
        'score': result['score'],
        'reasoning': result['reasoning']
    }

# Usage:
result = eval_groundedness_llm_judge(
    query="What certifications do I need for supervisor?",
    retrieved_docs=["cert-supervisor.md: Required: Safety Level 2..."],
    answer="You need three certifications..."
)
print(f"Score: {result['score']}/5 - {result['reasoning']}")
```

### Test Set Management (CSV Template)

```csv
query_id,query_text,category,expected_sources,expected_answer_summary,target_latency_ms,priority
1,"What jobs are available in my area?",job_assignments,"[jobs-api.md, location-matching.md]","Lists available jobs + CTA: Tap to apply",2000,high
2,"Can I pick any job from the list?",job_assignments,"[job-selection-rules.md, eligibility.md]","Explains eligibility + cert requirements",2000,high
...
25,"Can I get certified while not actively working?",certifications,"[inactive-worker-policy.md, training-eligibility.md]","Yes/No + eligibility rules",2000,high
26,"How do I list my driveway for charging?",homeowner_setup,"[setup-process.md, requirements.md]","Step-by-step setup + timeline",2000,high
...
```

### Test Harness (Run All Evals)

```python
import csv
from typing import Callable

class RAGEvalSuite:
    def __init__(self, test_set_csv: str, rag_system: Callable):
        self.test_cases = self._load_test_set(test_set_csv)
        self.rag = rag_system
        self.results = []
    
    def _load_test_set(self, csv_path: str):
        with open(csv_path) as f:
            return list(csv.DictReader(f))
    
    def run_all_evals(self) -> dict:
        """Run all 5 evaluators on all test cases."""
        
        for test_case in self.test_cases:
            query = test_case['query_text']
            
            # Get RAG output
            rag_output = self.rag.query(query)
            answer = rag_output['answer']
            retrieved_docs = rag_output['retrieved_docs']
            latency_ms = rag_output['latency_ms']
            
            # Run evals
            eval_result = {
                'query_id': test_case['query_id'],
                'query': query,
                'category': test_case['category']
            }
            
            # Eval 1: Retrieval
            eval_result['retrieval_precision'] = self._eval_retrieval(
                retrieved_docs, 
                test_case['expected_sources']
            )
            
            # Eval 2: Hallucination
            halluc = eval_hallucination(answer, retrieved_docs)
            eval_result['hallucination_rate'] = halluc['hallucination_rate']
            
            # Eval 3: Groundedness
            groundedness = eval_groundedness_llm_judge(
                query, retrieved_docs, answer
            )
            eval_result['groundedness_score'] = groundedness['score']
            
            # Eval 4: Mobile UX
            eval_result['mobile_ux_pass'] = self._eval_mobile_ux(answer)
            
            # Eval 5: Latency
            eval_result['latency_ms'] = latency_ms
            
            self.results.append(eval_result)
        
        return self._summarize_results()
    
    def _summarize_results(self) -> dict:
        """Aggregate metrics by category."""
        
        summary = {}
        for category in set(r['category'] for r in self.results):
            cat_results = [r for r in self.results if r['category'] == category]
            
            summary[category] = {
                'count': len(cat_results),
                'retrieval_precision_avg': sum(
                    r['retrieval_precision'] for r in cat_results
                ) / len(cat_results),
                'hallucination_rate_avg': sum(
                    r['hallucination_rate'] for r in cat_results
                ) / len(cat_results),
                'groundedness_score_avg': sum(
                    r['groundedness_score'] for r in cat_results
                ) / len(cat_results),
                'mobile_ux_pass_rate': sum(
                    r['mobile_ux_pass'] for r in cat_results
                ) / len(cat_results),
                'latency_p95': sorted(
                    [r['latency_ms'] for r in cat_results]
                )[int(0.95 * len(cat_results))]
            }
        
        return summary

# Usage:
suite = RAGEvalSuite('test_set.csv', rag_system)
results = suite.run_all_evals()

for category, metrics in results.items():
    print(f"\n{category}:")
    print(f"  Retrieval: {metrics['retrieval_precision_avg']:.1%}")
    print(f"  Hallucination: {metrics['hallucination_rate_avg']:.1%}")
    print(f"  Groundedness: {metrics['groundedness_score_avg']:.1f}/5")
    print(f"  Mobile UX: {metrics['mobile_ux_pass_rate']:.1%}")
    print(f"  Latency p95: {metrics['latency_p95']:.0f}ms")
```

---

## Part 6: Interview Framing

### How to Present RAG Eval Strategy

**Setup (30 seconds):**

"When I built RAG systems for two different projects, I learned that architecture and deployment mean nothing if you can't guarantee quality at scale. So I developed an eval framework that works backwards from requirements to metrics."

**Framework (60 seconds):**

"The framework has 5 tiers:

1. **Retrieval Quality** — Are we fetching the right source docs? I measure Precision@K. Fast, automatic, no LLM calls. If retrieval fails, everything downstream fails.

2. **Groundedness** — Does the answer actually use what we retrieved? I use LLM-as-judge with a rubric. This catches hallucinations that use training data instead of docs.

3. **Hallucination Detection** — What claims does the answer make that aren't in the sources? I extract claims via LLM and verify each one. Target < 2% hallucination rate.

4. **User-Facing Quality** — Does the answer actually help? Depends on the domain. For frontline workers, that's mobile-optimized answers. For a marketplace, it's persona-specific responses without confusion.

5. **Production Metrics** — Latency, cost, error rate. Boring but essential.

Each tier builds on the previous one. Retrieval is the foundation. If that's weak, the rest crumbles."

**Real Example (90 seconds):**

"For a frontline worker chatbot, hallucination could mean telling someone the wrong job assignment or wrong benefits info. So I set a 2% hallucination target — basically zero tolerance.

I built a test set of 30 real user questions: 10 on job assignments, 10 on certifications, 10 on benefits. For each query, I documented what docs should be retrieved and what a correct answer looks like.

Baseline evals showed 6% hallucination rate — too high. I traced it to two things: weak retrieval for certification docs (chunking issue) and the LLM generating plausible-sounding answers outside the docs.

Fix 1 was reducing chunk size. Retrieval went 78% → 85% for certs.

Fix 2 was adding one line to the prompt: 'If the answer is not in the docs, say so.' Hallucination dropped 6% → 2%.

Final state: All metrics green. Shipped with confidence."

**Red Flags / Questions You Might Get:**

Q: "How do you know 2% hallucination is good enough?"

A: "It's a business decision, not technical. For frontline workers, job and benefit info is high-stakes. Legal/compliance implications. I set 2% as the threshold where random hallucinations become statistically unlikely to cause real damage. For low-stakes Q&A, I might tolerate 5-10%. For medical/financial, I'd push toward < 0.5%."

Q: "Don't you need more test cases?"

A: "30 is minimum, 100 is ideal. You need enough to cover all major query types + edge cases. But beyond 100, you're seeing diminishing returns on finding new failure modes. I'd invest in quarterly re-eval with fresh test sets rather than massive initial sets."

Q: "How do you automate this?"

A: "Eval 1-3 are fully automated (code runs nightly). Eval 4 is partially automated (mobile UX checks are code, manual review for semantic tone). Eval 5 is instrumentation (automated). Only human eval (Eval 4, the subjective stuff) requires people. That happens on every release."

Q: "What if evals say the system is good but users complain?"

A: "Evals are a leading indicator, not a guarantee. It means your eval rubric missed something. You need to capture that user feedback and add it to the test set. Re-run evals with the new test case. Iterate. Evals are always incomplete."

### STAR Story Angle (For "Tell me about quality assurance")

**Situation:** 6-week timeline, frontline workers, zero tolerance for broken UX (onboarding had to drop from 4 hours to 15 minutes).

**Task:** Build RAG chatbot for job assignments, certs, benefits. Had to guarantee quality in a short timeline.

**Action:** Rather than ship and see what breaks, I built an eval framework upfront. Defined 5-tier approach: retrieval → groundedness → hallucination → UX → production metrics. Created a test set of 30 real user questions before writing any customer-facing code. Established baselines, identified failures, iterated.

**Result:** Baseline showed 6% hallucination (unacceptable). Two targeted fixes (retrieval chunking + prompt refinement) brought it to 2%. Shipped with high confidence. Received zero user complaints about answer quality. Onboarding time dropped from 4 hours to 15 minutes (exceeding goal).

---

## Summary: Eval-First Mindset

RAG systems without evals are faith-based engineering. You hope they work until they catastrophically fail in production.

RAG systems with evals are data-driven. You know exactly where they break, why, and how to fix it.

The three principles:

1. **Measure what matters.** Hallucination rate, retrieval precision, latency — not vanity metrics.

2. **Automate as much as possible.** Tier 1-3 evals should run nightly. Tier 4-5 at release time.

3. **Iterate ruthlessly.** Every failure in baseline evals is a gift. Fix it. Re-run. Measure improvement. Repeat until metrics pass.

Evals are the difference between a prototype and a production system.

