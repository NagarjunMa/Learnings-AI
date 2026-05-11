# Agentic RAG — Complete Mastery Deep-Dive

**Objective:** Master Agentic RAG for production systems and interview scenarios.

---

## Part 1: Foundation & Philosophy

### RAG vs Agentic RAG

**Traditional RAG:**
```
User query → Embed query → Vector search (top-k) → LLM answers
(Fixed pipeline. No reasoning about retrieval.)
```

**Agentic RAG:**
```
User query → Agent sees tools → Agent decides:
  - What to search?
  - Which vector DB?
  - Validate results?
  - Re-retrieve if poor?
  - Combine multiple sources?
(Agent orchestrates retrieval dynamically.)
```

**Core difference:** Agent is the orchestrator, not the pipeline.

### When to Use Agentic RAG

| Scenario | RAG | Agentic RAG |
|---|---|---|
| Simple Q&A over docs | ✓ | ✗ (overkill) |
| Multi-source retrieval | ✗ | ✓ |
| Complex queries (multiple steps) | ✗ | ✓ |
| Validation needed (hallucination check) | ✗ | ✓ |
| Routing to different DBs | ✗ | ✓ |
| Re-retrieval on bad results | ✗ | ✓ |

---

## Part 2: Architecture Deep-Dive

### Agentic RAG Flow

```
┌──────────────────────────────────────────────┐
│ 1. User Query                                │
│    "How do I set up monitoring in Kubernetes?" │
└────────────┬─────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────┐
│ 2. Agent Context Build                       │
│    - Available tools: docs, KB, logs         │
│    - Agent system prompt: "Route intelligently" │
│    - Memory: past successful queries         │
└────────────┬──────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────┐
│ 3. Query Understanding (Optional)            │
│    Agent decides: decompose or direct search?│
│    Complex query → break into sub-queries    │
│    Simple query → single search              │
└────────────┬──────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────┐
│ 4. Multi-Step Retrieval                      │
│    Agent calls:                              │
│    - Search docs: "monitoring setup"         │
│    - Search KB: "Prometheus config"          │
│    - Query logs: "recent errors"             │
│    - Search examples: "K8s best practices"   │
└────────────┬──────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────┐
│ 5. Retrieval Validation                      │
│    Agent evaluates: "Are results relevant?"  │
│    If poor (score < 0.5): re-retrieve        │
│    If good: continue                         │
│    If mixed: note confidence gaps            │
└────────────┬──────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────┐
│ 6. Answer Generation                         │
│    LLM synthesizes retrieved context         │
│    Agent can: cite sources, flag uncertainty │
└────────────┬──────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────┐
│ 7. Post-Generation Validation                │
│    Check: Does answer cite sources?          │
│    Check: Does answer contradict retrieved?  │
│    Check: Confidence threshold met?          │
│    If fail: trigger re-retrieval with hints  │
└──────────────────────────────────────────────┘
```

### Component 1: Query Understanding

Agent analyzes query to decide strategy:

```python
from strands.agents import Agent

agent = Agent(
    name="rag_router",
    system_prompt="""
    Analyze user queries. Decide:
    1. Is this a simple lookup? (Direct search)
    2. Multi-part question? (Decompose into sub-queries)
    3. Needs validation? (Poor docs have contradictions)
    4. Cross-source? (Multiple DBs needed)
    """
)

query = "How do I set up monitoring in Kubernetes with Prometheus?"
# Agent decides:
# - Decompose: ["Setup Prometheus in K8s", "Configure scrape targets", "Define alerts"]
# - Sources: [docs, KB, examples]
# - Validation: Yes (monitoring configs are critical)
```

### Component 2: Query Decomposition

Complex queries split into sub-queries for targeted retrieval:

```python
class QueryDecomposer:
    def decompose(self, query: str) -> list[dict]:
        """
        Break multi-part question into sub-queries.
        
        Example:
        Input: "How do I set up Kubernetes monitoring, deploy Prometheus, 
                and configure alerts for high CPU?"
        Output: [
            {"sub_query": "Kubernetes monitoring setup", "priority": 1},
            {"sub_query": "Prometheus deployment", "priority": 1},
            {"sub_query": "Alert configuration CPU", "priority": 2},
        ]
        """
        # Agent decides decomposition
        # OR rules-based (regex for "and", "then", "also")
        pass

# Usage in Agentic RAG
decomposer = QueryDecomposer()
sub_queries = decomposer.decompose(user_query)

for sub_q in sub_queries:
    results = vector_db.search(sub_q['sub_query'], top_k=5)
    consolidated_results.extend(results)
```

### Component 3: Multi-Source Retrieval

Agent routes to different data sources:

```python
from strands.agents import Agent
from strands.bedrock import BedrockKnowledgeBase
from strands.integrations import PineconeDB, PostgreSQL

agent = Agent(
    name="multi_source_rag",
    tools=[
        # Multiple retrieval tools
        BedrockKnowledgeBase(...),          # AWS docs
        PineconeDB(...),                    # Custom docs (vector)
        PostgreSQL(...),                    # Structured data (SQL)
        WebSearch(),                        # Real-time info
        CodeExecutor(),                     # Run examples
    ]
)

# Agent decides which tool to call for query
query = "How do I set up monitoring in Kubernetes?"
response = agent.run(query)

# Agent reasoning:
# "Kubernetes setup → docs (Bedrock KB)"
# "Prometheus examples → custom docs (Pinecone)"
# "Real-time errors → PostgreSQL logs"
# Calls all 3 tools, synthesizes response
```

### Component 4: Retrieval Validation

Agent evaluates relevance of retrieved documents:

```python
class RetrievalValidator:
    def validate(self, query: str, documents: list[str]) -> dict:
        """
        Score document relevance to query.
        
        Returns:
        {
            "overall_score": 0.85,
            "docs_relevant": True/False,
            "confidence": "high/medium/low",
            "gaps": ["topic X not covered", "conflicting info on Y"]
        }
        """
        # Agent asks: "Do these docs answer the query?"
        # Uses LLM evaluation or rule-based scoring
        pass

# Usage
validator = RetrievalValidator()
results = vector_db.search("monitoring setup", top_k=5)
validation = validator.validate(query, results)

if validation['overall_score'] < 0.5:
    # Poor retrieval. Re-search with different query
    results = vector_db.search("observability Kubernetes", top_k=5)
    validation = validator.validate(query, results)
```

### Component 5: Re-Ranking

Reorder retrieved documents by relevance or strategy:

```python
class DocumentReranker:
    def rerank(
        self,
        query: str,
        documents: list[dict],
        strategy: str = "relevance"
    ) -> list[dict]:
        """
        Reorder documents by relevance or other signals.
        
        Strategies:
        - relevance: LLM judges fitness for query
        - recency: Newer docs first
        - authority: Trust score first
        - length: Longer (more detailed) first
        - hybrid: Combine above
        """
        if strategy == "relevance":
            # Cross-encoder: LLM scores (query, doc) pairs
            scores = []
            for doc in documents:
                score = cross_encoder(query, doc['content'])
                scores.append((doc, score))
            return sorted(scores, key=lambda x: x[1], reverse=True)
        
        elif strategy == "recency":
            return sorted(documents, key=lambda x: x['timestamp'], reverse=True)
        
        elif strategy == "hybrid":
            # Combine: 0.5 * relevance + 0.3 * recency + 0.2 * authority
            pass

# Usage
reranker = DocumentReranker()
initial_results = vector_db.search(query, top_k=10)
reranked = reranker.rerank(query, initial_results, strategy="relevance")
# Top 3 used for LLM context
```

### Component 6: Answer Generation with Citations

LLM synthesizes answer with source attribution:

```python
class CitationGenerator:
    def generate_answer_with_citations(
        self,
        query: str,
        documents: list[dict],
        model: str = "claude-3-5-sonnet"
    ) -> dict:
        """
        Generate answer that cites sources.
        """
        system_prompt = """
        Answer user query using provided documents.
        For each fact, cite source: [Source: doc_id, page X]
        If info conflicts, note both sources.
        If unsure, say "This is not covered in available docs."
        """
        
        context = "\n".join([
            f"[Doc {i}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        response = llm.invoke(
            system_prompt=system_prompt,
            user_prompt=f"Question: {query}\n\nDocuments:\n{context}"
        )
        
        return {
            "answer": response,
            "citations": extract_citations(response),
            "sources_used": [doc['source'] for doc in documents],
        }

# Usage
answer = CitationGenerator().generate_answer_with_citations(
    query="How to set up Kubernetes monitoring?",
    documents=retrieved_docs
)
print(answer['answer'])
# Output: "To set up Kubernetes monitoring, install Prometheus... 
#          [Source: K8s-docs, page 42]"
```

### Component 7: Hallucination Detection

Validate answer against retrieved documents:

```python
class HallucinationDetector:
    def detect(self, answer: str, documents: list[str]) -> dict:
        """
        Check if answer contains facts not in documents.
        
        Returns:
        {
            "is_hallucinating": True/False,
            "hallucinations": [
                {"claim": "X requires Y", "coverage": "not_in_docs"}
            ],
            "confidence": 0.92
        }
        """
        # Extract facts from answer
        # Check each fact against document text
        # Flag if fact missing from all docs
        pass

# Usage
detector = HallucinationDetector()
result = detector.detect(generated_answer, retrieved_docs)

if result['is_hallucinating']:
    # Trigger re-retrieval with query based on hallucination
    new_query = f"{original_query} {hallucination_topic}"
    new_docs = vector_db.search(new_query, top_k=5)
    # Re-generate answer with new docs
```

---

## Part 3: Integration Patterns

### Pattern 1: Bedrock KB + Strands

```python
from strands.agents import Agent
from strands.bedrock import BedrockKnowledgeBase

agent = Agent(
    name="kb_rag_agent",
    model="claude-3-5-sonnet",
    system_prompt="""
    You are a helpful assistant. Use the knowledge base tool to find
    relevant information. Always cite your sources.
    """
)

kb = BedrockKnowledgeBase(
    knowledge_base_id="kbase-abc123",
    region="us-east-1"
)

agent.tools = [kb.retrieval_tool()]

# Agent auto-decides when to call KB
result = agent.run("How do I set up Kubernetes monitoring?")
# Agent reasoning:
# 1. "I need information about K8s monitoring"
# 2. "Call KB retrieval tool"
# 3. "Retrieve docs about monitoring"
# 4. "Synthesize answer with citations"
```

### Pattern 2: Vector DB + Validation

```python
from strands.agents import Agent
from strands.integrations import PineconeDB
from strands.tools import Tool

class ValidatingVectorSearch(Tool):
    name = "search_docs"
    description = "Search documentation with validation"
    
    def __init__(self, vector_db: PineconeDB):
        super().__init__()
        self.db = vector_db
    
    def execute(self, query: str, validate: bool = True) -> dict:
        """
        Search docs and optionally validate relevance.
        """
        # Initial search
        results = self.db.search(query, top_k=5)
        
        if validate:
            # Validate relevance
            scores = []
            for doc in results:
                relevance_score = self._score_relevance(query, doc['content'])
                scores.append({
                    **doc,
                    'relevance_score': relevance_score
                })
            
            # Filter low-relevance
            results = [d for d in scores if d['relevance_score'] > 0.5]
        
        return {
            "documents": results,
            "count": len(results),
            "validation_passed": len(results) > 0
        }
    
    def _score_relevance(self, query: str, doc: str) -> float:
        # Use cross-encoder or LLM
        prompt = f"Rate relevance 0-1: Query='{query}' Doc='{doc[:200]}...'"
        score = float(llm.invoke(prompt))
        return score

# Usage
agent = Agent(
    tools=[ValidatingVectorSearch(pinecone_db)]
)
result = agent.run("How to deploy Prometheus?")
# Agentic tool auto-validates retrieval
```

### Pattern 3: Multi-Source with Routing

```python
from strands.agents import Agent
from strands.bedrock import BedrockKnowledgeBase
from strands.integrations import PineconeDB, PostgreSQL
from strands.tools import WebSearch, Tool

class SmartRouter(Tool):
    name = "route_query"
    description = "Route query to best information source"
    
    def execute(self, query: str) -> dict:
        """
        Decide which source(s) to query based on question.
        """
        routing_rules = {
            r"setup|install|configure": "docs_kb",
            r"error|debug|troubleshoot": "logs_db",
            r"best practice|pattern|design": "docs_vector",
            r"latest|recent|new": "web_search",
        }
        
        sources = []
        for pattern, source in routing_rules.items():
            if re.search(pattern, query, re.I):
                sources.append(source)
        
        return {"sources": sources or ["docs_kb"]}

agent = Agent(
    name="multi_source_router",
    tools=[
        SmartRouter(),
        BedrockKnowledgeBase(...),
        PineconeDB(...),
        PostgreSQL(...),
        WebSearch(),
    ]
)

result = agent.run("How do I debug Prometheus scrape errors?")
# Router decides: logs_db + docs
# Agent calls both, synthesizes
```

---

## Part 4: Failure Modes & Fixes

### Failure 1: Empty Retrieval

```
Problem: Vector search returns 0 results.

Causes:
1. Query too specific (narrow terminology)
2. Documents missing (KB incomplete)
3. Embedding mismatch (doc language ≠ query language)

Fixes:
1. Expand query: use synonyms, broader terms
   "Prometheus scrape config" → also search "metrics collection"
   
2. Re-embed query with different model
   Dense retrieval fails → use sparse (BM25) as fallback
   
3. Agent retries with agent-generated query:
   Agent: "Try searching 'monitoring setup K8s'"
   Re-retrieve with this new query
```

**Implementation:**
```python
class RobustRetrieval:
    def retrieve(self, query: str) -> list[dict]:
        # Try 1: Exact embedding search
        results = dense_search(query)
        if len(results) == 0:
            # Try 2: Expand query with synonyms
            expanded = expand_query_synonyms(query)
            results = dense_search(expanded)
        
        if len(results) == 0:
            # Try 3: Fallback to sparse search (BM25)
            results = sparse_search(query)
        
        if len(results) == 0:
            # Try 4: Agent-generated query
            agent_query = agent.generate_search_query(query)
            results = dense_search(agent_query)
        
        return results
```

### Failure 2: Poor Retrieval (Low Relevance)

```
Problem: Search returns documents but none answer the question.

Causes:
1. Vector DB has wrong documents
2. Similarity metric misaligned
3. Query ambiguous

Fixes:
1. Validation + re-rank:
   Score retrieved docs (< 0.5 = poor)
   If poor, trigger re-retrieval with refined query
   
2. Cross-encoder reranking:
   Use cross-encoder (fine-tuned for domain) to score (query, doc) pairs
   Reorder by cross-encoder score
   
3. Query refinement:
   Agent: "Initial search poor. Trying 'observability in Kubernetes'"
   Re-retrieve with refined query
```

**Implementation:**
```python
class SmartValidation:
    def retrieve_with_validation(self, query: str) -> list[dict]:
        # Initial retrieval
        results = vector_db.search(query, top_k=10)
        
        # Validate
        valid_results = []
        for doc in results:
            relevance = cross_encoder.score(query, doc['content'])
            if relevance > 0.6:
                valid_results.append({**doc, 'relevance': relevance})
        
        # If too few valid results, re-retrieve
        if len(valid_results) < 3:
            refined_query = self.agent.refine_query(query)
            new_results = vector_db.search(refined_query, top_k=10)
            valid_results.extend([
                {**doc, 'relevance': cross_encoder.score(refined_query, doc['content'])}
                for doc in new_results
            ])
        
        return valid_results
```

### Failure 3: Conflicting Information

```
Problem: Retrieved documents contradict each other.

Causes:
1. KB outdated (old + new versions)
2. Multiple standards (AWS region-specific configs)
3. Docs for different versions

Fixes:
1. Version-aware retrieval:
   Filter docs by version/date
   "K8s 1.28+", "AWS region: us-west-2"
   
2. Conflict flagging:
   Agent detects contradiction
   Returns: "Sources X and Y conflict. X is newer."
   
3. Expert resolution:
   Route to human for clarification
```

**Implementation:**
```python
class ConflictDetector:
    def detect_conflicts(self, query: str, documents: list[dict]) -> dict:
        # Extract claims from each doc
        claims = [extract_claims(doc['content']) for doc in documents]
        
        # Check for contradictions
        conflicts = []
        for i, claim_set_i in enumerate(claims):
            for j, claim_set_j in enumerate(claims[i+1:], i+1):
                contradictions = find_contradictions(claim_set_i, claim_set_j)
                if contradictions:
                    conflicts.append({
                        "doc_i": documents[i]['source'],
                        "doc_j": documents[j]['source'],
                        "contradictions": contradictions
                    })
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts,
            "resolution": self._suggest_resolution(conflicts)
        }
    
    def _suggest_resolution(self, conflicts: list) -> str:
        # Use timestamp, source authority, or ask agent
        pass
```

### Failure 4: Hallucination (Answer Not in Docs)

```
Problem: LLM generates plausible answer not supported by retrieval.

Causes:
1. LLM extrapolates (knowledge cutoff info)
2. Vague retrieval allows LLM inference
3. Missing retrieval context

Fixes:
1. Strict grounding:
   System prompt: "Only use retrieved docs. Do not extrapolate."
   
2. Hallucination detection:
   Extract facts from answer → verify against docs
   If fact missing, mark uncertainty: "Based on available docs..."
   
3. Confidence threshold:
   If LLM confidence < 0.7, return: "This is not covered."
```

**Implementation:**
```python
class GroundedAnswerGenerator:
    def generate(self, query: str, documents: list[str]) -> dict:
        system_prompt = """
        Answer ONLY using provided documents. 
        Do NOT use external knowledge or extrapolate.
        If answer not in docs, say: "This is not covered in available materials."
        """
        
        # Generation
        answer = llm.invoke(
            system_prompt=system_prompt,
            user_prompt=f"Q: {query}\n\nDocs:\n{context}"
        )
        
        # Validation
        hallucinations = self.find_hallucinations(answer, documents)
        
        if hallucinations:
            # Mark unreliable facts
            answer = self.mark_uncertain(answer, hallucinations)
        
        return {
            "answer": answer,
            "hallucinations_detected": len(hallucinations),
            "is_grounded": len(hallucinations) == 0
        }
    
    def find_hallucinations(self, answer: str, documents: list[str]) -> list:
        # Extract facts from answer
        facts = extract_facts(answer)
        
        # Check each fact against docs
        hallucinations = []
        for fact in facts:
            found_in_any_doc = any(fact in doc for doc in documents)
            if not found_in_any_doc:
                hallucinations.append(fact)
        
        return hallucinations
```

---

## Part 5: Production Patterns

### Pattern 1: Feedback Loop (Learn from Usage)

```python
class RAGWithFeedback:
    def __init__(self, vector_db, agent):
        self.db = vector_db
        self.agent = agent
        self.feedback_store = []  # Store user feedback
    
    def run_with_feedback(self, query: str) -> dict:
        # Generate answer
        answer = self.agent.run(query)
        
        # Collect feedback (async)
        feedback = self.collect_user_feedback()
        # "Was this answer helpful?" → Yes/No/Partial
        
        if feedback == "No":
            # Learn: re-retrieve with modified query
            refined_query = self.learn_from_failure(query, answer)
            # Save: (query, refined_query) for future improvement
            self.feedback_store.append({
                "original_query": query,
                "refined_query": refined_query,
                "feedback": "negative"
            })
        
        return answer
    
    def learn_from_failure(self, query: str, answer: str) -> str:
        # Agent reflects: "Why did this fail?"
        reflection = self.agent.run(
            f"User rejected answer to '{query}'. "
            f"Answer was: {answer}. "
            f"Suggest better search query."
        )
        return reflection
```

### Pattern 2: Caching & Deduplication

```python
class CachedAgenticRAG:
    def __init__(self, vector_db, agent):
        self.db = vector_db
        self.agent = agent
        self.cache = {}  # query → answer
    
    def run(self, query: str) -> dict:
        # Check cache
        if query in self.cache:
            return self.cache[query]
        
        # Check semantic similarity to past queries
        similar_past = self.find_similar_queries(query)
        if similar_past:
            # Reuse cached answer (with freshness check)
            cached = self.cache[similar_past]
            if not self.is_stale(cached):
                return cached
        
        # Fresh retrieval
        answer = self.agent.run(query)
        
        # Cache
        self.cache[query] = {
            "answer": answer,
            "timestamp": time.time()
        }
        
        return answer
```

### Pattern 3: Cost Optimization (Tiered Retrieval)

```python
class TieredAgenticRAG:
    def run(self, query: str, budget: float = 0.10) -> dict:
        # Tier 1: Cheap retrieval (keyword BM25)
        tier1 = self.sparse_search(query)
        
        if tier1['confidence'] > 0.8:
            return tier1  # Good enough, return
        
        # Tier 2: More expensive (dense + validation)
        tier2 = self.dense_search_with_validation(query)
        
        if tier2['confidence'] > 0.8:
            return tier2
        
        # Tier 3: Expensive (cross-encoder rerank + multi-source)
        tier3 = self.multi_source_rerank(query)
        
        return tier3

# Cost breakdown:
# Tier 1: ~$0.001 (BM25)
# Tier 2: ~$0.005 (embedding + LLM validation)
# Tier 3: ~$0.02 (cross-encoder + multi-source)
# Most queries stop at Tier 1-2
```

---

## Part 6: Interview Questions & Answers

### Q1: Design Agentic RAG for product documentation. Architecture?

**Answer:**
```
Components:
1. Query Understanding
   - Agent analyzes query type (simple, multi-part, ambiguous)
   - Decides: decompose or direct search

2. Multi-Source Retrieval
   - Bedrock KB (official docs)
   - Pinecone (examples + case studies)
   - PostgreSQL (known issues, resolved)
   - WebSearch (bleeding-edge features)
   - CodeExecutor (run example code)

3. Validation & Re-ranking
   - Cross-encoder scores relevance
   - Agent validates: "Docs answer the query?"
   - If poor: re-retrieve with refined query
   - Rerank by relevance, recency, authority

4. Citation-Aware Generation
   - LLM cites source for each claim
   - Detects hallucinations (facts not in docs)
   - Marks confidence: "This is not covered"

5. Feedback Loop
   - User: "Was this helpful?"
   - If No: learn refined query, cache failure
   - Future similar queries: use refined query

Flow:
  User query → Agent decides strategy
  → Multi-source retrieval (parallel)
  → Validation + re-rank
  → Answer generation with citations
  → Hallucination check
  → Return + collect feedback

Cost: ~$0.005–0.02 per query (LLM + retrieval)
Latency: 2–5 seconds (most time in LLM inference)
```

### Q2: Retrieved docs conflict. How do you handle?

**Answer:**
```
Scenario: User asks "What's best practice for K8s CPU limits?"
Doc A: "Set CPU limit 500m for safety"
Doc B: "Remove CPU limit for performance"

Solution layers:

1. Detect conflict (automatically)
   Extract claims from each doc
   Find contradictions
   Flag: "Sources conflict on this point"

2. Resolve (by authority/freshness)
   - Use timestamps: newer doc wins
   - Check author authority (Kubernetes maintainer?)
   - Use doc source: official docs > community blogs
   - Return: "AWS recommends X, community suggests Y"

3. Return with nuance
   Answer: "Best practice depends on workload:
   - For latency-sensitive: remove limit (Doc B)
   - For stability: set limit (Doc A)
   Sources conflict; choose based on requirements"

4. Escalate if unresolved
   If conflict can't be resolved, flag for human review
   Return: "Conflicting guidance. Manual review needed."

Prevention:
- Version-aware retrieval (filter by K8s version)
- Date-aware: prefer recent docs
- Source prioritization: official > community
```

### Q3: Agentic RAG vs traditional RAG. When to use which?

**Answer:**
```
Traditional RAG (Simple):
- Fixed pipeline: embed → search → answer
- Cost: ~$0.001 (one embedding, one search)
- Latency: fast (500ms)
- Use case: FAQ, simple QA
- Failure: poor query → poor retrieval → poor answer

Agentic RAG (Complex):
- Agent decides: what to search, validate, re-retrieve
- Cost: ~$0.01 (multiple searches, validation, LLM reasoning)
- Latency: slower (3–5 seconds)
- Use case: complex queries, multi-source, validation needed
- Resilience: poor retrieval → agent re-tries with refined query

Decision matrix:
Simple FAQ (100K docs)               → Traditional RAG
  "What's my account balance?"
  
Multi-source (docs + logs + KB)      → Agentic RAG
  "Why is this service slow? Show me errors + config"
  
Hallucination risk (financial, legal) → Agentic RAG (validation layer)
  "What are GDPR requirements for data deletion?"
  
High-volume, cost-sensitive          → Traditional RAG
  "When does store open?" × 10,000 queries/day
  
Complex decomposable questions       → Agentic RAG
  "Compare AWS and GCP networking for our architecture"
  
Recommendation: Start with Traditional RAG. Upgrade to Agentic if:
  - Retrieval failure rate > 10%
  - Multi-source needed
  - Hallucination risk high
  - Cost acceptable
```

### Q4: How do you prevent hallucinations in Agentic RAG?

**Answer:**
```
Three layers:

1. Retrieval Layer
   - Multi-source retrieval (more docs = better coverage)
   - Validation: cross-encoder scores relevance
   - If poor: re-retrieve
   → Result: LLM gets strong grounding

2. Generation Layer
   System prompt: "Use ONLY retrieved docs. Do NOT extrapolate."
   → Strict constraint
   
   Return confidence: LLM outputs confidence score
   If confidence < 0.7: "Not covered in available docs"
   → Uncertainty handling

3. Post-Generation Validation
   Extract facts from answer
   Check each fact against retrieved docs
   If fact missing: mark as uncertain or regenerate
   → Hallucination detection + correction

4. Monitoring & Feedback
   User feedback: "Was this accurate?"
   If No: capture hallucination example
   Retrain on failures
   → Continuous improvement

Example flow:
  User: "Can I use Kubernetes on a Raspberry Pi?"
  
  Agent retrieves:
  - Doc 1: "K8s minimum requirements: 2GB RAM, 2 CPU cores"
  - Doc 2: "RPi 4: 4GB RAM, ARM CPU (different from x86)"
  - Doc 3: "Community: 'K8s on RPi works but not recommended'"
  
  Answer: "Technically yes (RPi 4 has 4GB RAM), 
           but not recommended for production (Doc 3).
           K8s on RPi possible for learning/testing."
  
  Validation: All facts cited + grounded ✓
```

### Q5: Design for 10M+ documents. How do you scale?

**Answer:**
```
Scaling challenges:

1. Vector Search Latency
   Problem: 10M vectors, dense search slow
   Solution: Partition index
   - Pinecone namespaces: separate by topic (K8s, AWS, GCP)
   - Router: send query to relevant namespace only
   - Latency: 100ms (indexed) instead of 10s (full scan)

2. Retrieval Cost
   Problem: Cross-encoder validation expensive with 10M docs
   Solution: Tiered approach
   - Tier 1: Dense search (top 10)
   - Tier 2: Cross-encoder rerank (top 10)
   - → Only rank 10, not 10M

3. Memory & Storage
   Problem: Can't fit 10M embeddings in memory
   Solution: Vector DB service (Pinecone, Weaviate)
   - Managed scaling
   - Auto-replication
   - Cost: ~$500/month for 10M documents

4. Hallucination at Scale
   Problem: More docs → more conflicts
   Solution:
   - Version-aware retrieval (filter by doc version/date)
   - Source prioritization (official docs first)
   - Conflict detection (flag contradictions)

5. Freshness
   Problem: Documents become stale
   Solution:
   - Refresh cycle: re-index every week
   - Versioning: keep old docs marked as "archived"
   - Recency score: boost recent docs in ranking

Architecture for 10M docs:
  
  Query → Router (determines namespaces)
    ↓
  Pinecone (10M vectors, sharded by topic)
    ↓
  Dense search (top 10 per namespace)
    ↓
  Cross-encoder rerank (top 10 total)
    ↓
  Conflict detection (check versions)
    ↓
  Answer generation with citations
    ↓
  Hallucination check
    
Cost: ~$0.01–0.02 per query
Latency: 2–3 seconds
Throughput: ~100 QPS (with Pinecone Pro)
```

---

## Part 7: Common Patterns & Anti-Patterns

### Pattern: Query Expansion

```python
def expand_query(query: str) -> list[str]:
    """Generate alternative queries for failed retrieval."""
    return [
        query,
        # Synonym expansion
        query.replace("CPU limit", "resource limit"),
        # Broader term
        query.replace("Prometheus scrape config", "metrics collection"),
        # Related term
        f"{query} troubleshooting",
        # Agent-generated
        agent.suggest_alternative_query(query),
    ]

# Use case:
# Initial query: "How to configure Prometheus in Kubernetes?"
# Retrieval: 0 results
# Expand → retry with: "metrics collection in K8s"
# Retrieval: 5 results ✓
```

### Anti-Pattern: Always Trust Retrieval

```python
# BAD:
retrieved = vector_db.search(query)
answer = llm.generate(retrieved)  # No validation
return answer

# GOOD:
retrieved = vector_db.search(query)
validation = validate_relevance(query, retrieved)
if validation['score'] < 0.5:
    # Re-retrieve
    retrieved = vector_db.search(expand_query(query))
answer = llm.generate(retrieved)
return answer
```

### Anti-Pattern: No Citation Tracking

```python
# BAD:
answer = "Kubernetes is a container orchestrator."
# No source attribution

# GOOD:
answer = "Kubernetes is a container orchestrator [Source: K8s-docs, page 1]."
sources = [
    {"title": "K8s-docs", "page": 1, "url": "..."}
]
return {"answer": answer, "sources": sources}
```

---

## Part 8: Production Checklist

```
[ ] Query decomposer (for complex queries)
[ ] Multi-source router (Bedrock KB, vector DB, logs, web)
[ ] Relevance validator (cross-encoder or LLM)
[ ] Re-retrieval on poor results
[ ] Citation tracking + source attribution
[ ] Hallucination detection
[ ] Conflict detection (contradictory docs)
[ ] Version-aware filtering
[ ] Caching layer (for repeated queries)
[ ] Feedback collection (user: helpful? Yes/No)
[ ] Monitoring (retrieval success rate > 90%)
[ ] Latency SLA (< 5 seconds)
[ ] Cost tracking (per-query budget)
[ ] Load testing (concurrent users)
[ ] A/B test (Agentic RAG vs traditional RAG)
```

---

**Created for:** Complete mastery of Agentic RAG for production systems and interviews. Every detail needed to design, build, operate production Agentic RAG systems.
