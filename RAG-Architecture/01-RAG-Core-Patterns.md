# RAG (Retrieval-Augmented Generation) — From Basics to Production

RAG is the most deployed LLM architecture in production. Instead of asking the model facts it doesn't know, you retrieve relevant documents **first**, then ask it to answer based on those documents.

---

## The Core Formula

```
answer = LLM(question + retrieved_context + prompt)
```

**Without RAG:**
```
User: "What was our Q3 2025 revenue?"
LLM: "I don't have access to real-time company data. As of my knowledge cutoff..."
→ Hallucination or evasion
```

**With RAG:**
```
1. Retrieve: Search company docs → find Q3 2025 earnings report
2. Pass context: "Q3 2025 revenue: $500M. Grew 20% YoY."
3. LLM answers: "Our Q3 2025 revenue was $500M, up 20% from Q3 2024."
→ Grounded, accurate answer
```

---

## RAG Workflow — 5 Stages

### Stage 1: Document Ingestion & Chunking

**Problem:** Documents are too long. A PDF is 50 pages. If you embed the whole thing, you lose fine-grained retrieval.

**Solution:** Split into **chunks** (passages of 256-512 tokens). Different chunking strategies:

#### Fixed-Size Chunking
```python
chunk_size = 500  # characters
stride = 100      # overlap (prevents cutting mid-sentence)

chunks = []
for i in range(0, len(document), chunk_size - stride):
    chunks.append(document[i:i+chunk_size])
```

**Pros:** Simple, predictable.
**Cons:** Cuts mid-concept; "overlap" doesn't fully solve it.

**When to use:** Quick prototypes, homogeneous documents (logs, code).

#### Semantic Chunking
```python
# Split by sentences. Group sentences until token count hits threshold.
sentences = document.split(". ")
current_chunk = []
current_tokens = 0

for sentence in sentences:
    tokens = count_tokens(sentence)
    if current_tokens + tokens > 500:
        chunks.append(". ".join(current_chunk))
        current_chunk = [sentence]
        current_tokens = tokens
    else:
        current_chunk.append(sentence)
        current_tokens += tokens
```

**Pros:** Respects sentence boundaries; cleaner chunks.
**Cons:** Slightly slower; requires tokenizer.

**When to use:** Most production systems; especially important for content where concepts respect sentence/paragraph boundaries.

#### Recursive Chunking (Langchain `RecursiveCharacterTextSplitter`)
```python
# Try splitting by sentences first. If still too big, split by paragraphs, then words.
separators = ["\n\n", "\n", ". ", " ", ""]

def recursive_split(text, size=500, separators=separators):
    chunks = []
    for sep in separators:
        if len(text) < size:
            return chunks + [text]
        parts = text.split(sep)
        # Recursively split parts...
    return chunks
```

**Pros:** Handles mixed content well (markdown, code, prose).
**Cons:** Most complex to implement.

**When to use:** Multi-format documents (markdown with code blocks, mixed PDFs).

#### Metadata Chunking (LangChain `SemanticSplitter` + embeddings)
```python
# Embed each sentence. Split when embedding distance jumps (topic change).
sentence_embeddings = [embed(s) for s in sentences]

chunks = []
current_chunk = [sentences[0]]

for i in range(1, len(sentences)):
    distance = cosine_distance(
        sentence_embeddings[i-1],
        sentence_embeddings[i]
    )
    if distance > threshold:  # Topic boundary
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i]]
    else:
        current_chunk.append(sentences[i])
```

**Pros:** Splits at actual topic boundaries.
**Cons:** Expensive (needs embeddings for every sentence).

**When to use:** High-quality RAG where chunk coherence matters; cost is acceptable.

---

### Stage 2: Embedding & Storage (Vector DB)

**Embed each chunk:** Convert text to dense vectors (768–3,072 dimensions).

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# small: 1,536 dims, fast, cheap
# large: 3,072 dims, more accurate

chunk_vectors = [embeddings.embed_query(chunk) for chunk in chunks]
```

**Store in Vector DB:**

```python
from langchain_chroma import Chroma

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="financial_docs",
    persist_directory="./chroma_db"
)
```

**Vector DB types:**

| DB | Cost | Latency | Scale | Use Case |
|---|---|---|---|---|
| **Chroma** | Free (in-process) | <100ms | <1M vecs | Prototypes, local dev |
| **Pinecone** | $0.10/month + usage | <50ms | 100B vecs | Production SaaS, multi-tenant |
| **Weaviate** | $0.015/hour self-hosted | <100ms | 100B vecs | On-prem, compliance-heavy |
| **pgvector** | Free (Postgres extension) | 100-500ms | 100M vecs | Existing Postgres infra |
| **FAISS** | Free (library) | <10ms (CPU) | 1B vecs | Local similarity search, research |
| **Milvus** | Free (self-hosted) | <50ms | 10B vecs | Cloud-native, Kubernetes |

**Choosing a vector DB:**
- **Prototyping:** Chroma (free, in-memory)
- **Production SaaS:** Pinecone (managed, serverless)
- **Compliance/On-prem:** Weaviate or pgvector
- **Existing Postgres:** pgvector (no new infrastructure)
- **Cost-sensitive at scale:** FAISS or Milvus (open-source)

---

### Stage 3: Query Encoding & Retrieval

**User asks:** "What was revenue growth?"

**Encode query:**
```python
query_vector = embeddings.embed_query("What was revenue growth?")
```

**Retrieve similar chunks:**
```python
results = vectordb.similarity_search_with_score(
    "What was revenue growth?",
    k=5  # Return top 5 most relevant chunks
)
```

**Similarity metrics:**
- **Cosine similarity:** `sim = (A · B) / (||A|| ||B||)` — angle between vectors
- **Euclidean distance:** `dist = √(Σ(A_i - B_i)²)` — direct distance
- **Dot product:** `sim = A · B` — used in Pinecone (faster on GPU)

**Cosine is standard** because it's scale-invariant (absolute magnitude doesn't matter, only direction).

---

### Stage 4: Re-Ranking (Optional but Critical)

**Problem:** Top-k retrieval can bring irrelevant docs. Example:

```
Query: "What's our insurance coverage?"

Retrieved (top 5):
1. "Our office insurance covers fire, flood, liability..."
2. "Employee health insurance deductible is $500..."
3. "The insurance agent's name is Bob Jones..." ← Not relevant
4. "Cyber insurance premium is $10K/year..."
5. "History of insurance policies from 1990..." ← Outdated
```

**Solution:** Re-rank using a **cross-encoder** (different from embedding models).

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('ms-marco-MiniLM-L-12-v2')
# Takes (query, document) pairs → outputs relevance score

scores = reranker.predict([
    ("What's our insurance coverage?", chunk)
    for chunk in retrieved_chunks
])

# Re-sort by score, take top 3
ranked_chunks = sorted(
    zip(retrieved_chunks, scores),
    key=lambda x: x[1],
    reverse=True
)[:3]
```

**Cross-encoder vs embeddings:**
- **Embeddings:** Independent vectors; fast; less accurate on nuanced relevance
- **Cross-encoder:** Sees both query and doc together; slower; more accurate

**Production pattern:**
```python
# Fast retrieval (embeddings) → get top 10
initial_results = vectordb.similarity_search(query, k=10)

# Accurate re-ranking (cross-encoder) → get top 3
reranked = reranker.predict([
    (query, chunk.page_content) for chunk in initial_results
])

final_results = [
    initial_results[i] for i in np.argsort(reranked)[-3:][::-1]
]
```

**Cost tradeoff:** Re-ranking adds latency but cuts garbage. For financial docs, worth it.

---

### Stage 5: Answer Generation

**Combine context + prompt:**

```python
context = "\n\n".join([chunk.page_content for chunk in final_results])

prompt = f"""
Use the provided documents to answer the question.
If the answer is not in the documents, say "I don't have that information."

Documents:
{context}

Question: {user_query}

Answer:
"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}]
)
```

---

## Advanced RAG Patterns

### Hybrid Search (Keyword + Semantic)

**Problem:** Embeddings work for semantic similarity. But exact phrase matches fail.

Example:
```
Query: "Q3 2025 earnings"

Embedding retrieval: Returns docs about "quarterly financial results"
                     Misses docs that literally say "Q3 2025 earnings"

Keyword search: Finds "Q3 2025 earnings" exactly
               Misses "third quarter revenue growth"
```

**Solution:** Combine both, re-rank by combined score.

```python
# Dense retrieval (embeddings)
dense_results = vectordb.similarity_search(query, k=10)

# Sparse retrieval (BM25 — keyword matching)
bm25 = BM25Okapi([chunk.split() for chunk in all_chunks])
sparse_scores = bm25.get_scores(query.split())
sparse_results = [all_chunks[i] for i in np.argsort(sparse_scores)[-10:]]

# Combine and re-rank
combined = list(set(dense_results + sparse_results))
reranked = reranker.predict([(query, chunk) for chunk in combined])
final = combined[np.argsort(reranked)[-5:][::-1]]
```

**When to use:** Technical docs, financial reports, contracts (exact matches matter).

---

### Query Transformation (HyDE, Multi-Query)

**Problem:** User query is short/vague.

```
Query: "Budget"
Context-less. Does it mean spending? Forecasting? Department budgets?
```

**Solution 1: Hypothetical Document Embeddings (HyDE)**

```python
# Ask LLM to write a hypothetical document that would answer the query
hypothetical_doc = llm.generate(
    f"Write a document that answers this query: {query}"
)

# Embed and retrieve based on the hypothetical doc
results = vectordb.similarity_search(hypothetical_doc, k=5)
```

**Solution 2: Multi-Query Retrieval**

```python
# Ask LLM to rephrase the query in multiple ways
queries = llm.generate(
    f"Rephrase this query 3 different ways: {query}"
)
# "Budget" → ["What are our spending limits?", "How much money is allocated?", "What's our financial plan?"]

# Retrieve for each variant
all_results = []
for q in queries:
    results = vectordb.similarity_search(q, k=5)
    all_results.extend(results)

# Deduplicate and re-rank
final_results = reranker.predict([
    (query, chunk) for chunk in deduplicate(all_results)
])
```

**Production use:** When user queries are vague or domain-specific.

---

## RAG Evaluation — RAGAS

**Problem:** How do you know your RAG is working? Standard metrics don't capture it.

**RAGAS metrics:**

| Metric | What it measures | How |
|---|---|---|
| **Faithfulness** | Does answer use only the context (not hallucinate)? | LLM judges if statements are entailed by context |
| **Answer Relevance** | Does answer address the question? | LLM scores relevance to original query |
| **Context Precision** | Are retrieved docs relevant? | % of retrieved docs that are relevant |
| **Context Recall** | Did we retrieve all necessary info? | % of necessary info in retrieved docs |

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
)

# Prepare test data
dataset = {
    "question": [...],
    "ground_truth": [...],      # Gold-standard answers
    "answer": [...],             # Your RAG system's answers
    "contexts": [...]            # Retrieved chunks per question
}

# Run evaluation
results = evaluate(dataset, metrics=[
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
])

# Results: Scores 0.0–1.0 per metric
print(results)
```

**Interpretation:**
- **Faithfulness 0.9:** Answers are grounded; minimal hallucination
- **Context Recall 0.7:** Missing some relevant docs; retrieval needs tuning
- **Answer Relevance 0.6:** Answers don't fully address queries; prompt or retrieval broken

**Production workflow:**

```
1. Build RAG system
2. Create labeled test set (100 Q&A pairs, each with ground truth)
3. Run RAGAS evaluation
4. Identify bottleneck:
   - Low context precision? → Improve chunking or embedding model
   - Low faithfulness? → Improve prompt or reranking
   - Low answer relevance? → Improve query understanding
5. Iterate
6. Monitor ongoing (evaluate weekly on new queries)
```

---

## System Design Interview Pattern

**Question:** "Design a RAG system for a financial company's internal knowledge base."

**Your answer:**

```
1. Data:
   - 10,000 documents (PDFs, policies, reports)
   - 500 employees querying daily
   - Compliance: PII must be redacted, all queries audited

2. Architecture:
   - Chunking: Semantic chunking (respect concept boundaries)
   - Embeddings: text-embedding-3-small (cost-effective, 1,536 dims)
   - Vector DB: Pinecone (managed, compliant, good for financial)
   - Re-ranking: CrossEncoder (Cohere or local model for compliance)
   - LLM: Claude (via private endpoint if needed for compliance)

3. Retrieval:
   - Hybrid search (dense + BM25) for financial terms
   - Re-rank top 10 → top 3 chunks
   - Fallback to human review if confidence < 0.8

4. Generation:
   - Few-shot prompting (examples of good answers)
   - Explicit guardrails: "Only use provided documents"
   - Temperature: 0.0 (deterministic)
   - JSON output: { answer, confidence, sources }

5. Compliance:
   - PII redaction in chunks (before embedding)
   - Audit logging: who queried what, what was returned
   - Data residency: Use Pinecone's FedRAMP option
   - No sensitive data in vector DB (hash SSN, mask account numbers)

6. Monitoring:
   - RAGAS metrics weekly (faithfulness, relevance)
   - Query latency SLO: <500ms
   - Cost tracking per query
   - Manual review of low-confidence answers

7. Scaling:
   - Batch ingestion for new documents
   - Cache common queries
   - Multi-region Pinecone for HA
```

This shows: **end-to-end thinking + compliance + production patterns + evaluation mindset**.
