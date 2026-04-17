# Vector Databases — Choosing and Using at Scale

A vector database stores **embeddings** (dense vectors) and answers **similarity queries** at scale. The key operations: insert, update, delete, and **nearest neighbor search**.

---

## Why Vector DB, Not Regular DB?

**Regular SQL database:**
```sql
SELECT * FROM documents WHERE content LIKE '%insurance%'
```

Finds exact matches. Misses synonyms ("coverage" vs "protection").

**Vector database:**
```python
vectordb.search(query_embedding, k=5)
```

Finds **semantically similar** documents regardless of exact wording.

---

## Core Concepts

### Embeddings

A vector is a list of numbers representing meaning.

```
Text: "The cat sat on the mat"
Embedding (768 dims): [0.123, -0.456, 0.789, ... 765 more values]

Text: "The kitten rested on the rug"
Embedding (768 dims): [0.120, -0.450, 0.785, ... 765 more values]
                       ↑ Similar numbers (semantically related)
```

**Popular embedding models:**

| Model | Dims | Speed | Cost | Best For |
|---|---|---|---|---|
| **text-embedding-3-small** (OpenAI) | 1,536 | Fast | $0.02/1M | Most use cases; best quality/cost |
| **text-embedding-3-large** (OpenAI) | 3,072 | Medium | $0.13/1M | High-precision; expensive |
| **Cohere Embed** | 4,096 | Fast | $0.10/1M | Commercial; good for retrieval |
| **Mistral Embed** (open-source) | 1,024 | Fast | Free | Self-hosted; lower quality than Ada |
| **nomic-embed-text** (open-source) | 768 | Fast | Free | Local RAG; surprisingly good |

**Tradeoff:** Higher dimensions = more accuracy, higher cost.

**Production choice:**
- **Default:** text-embedding-3-small (OpenAI)
- **Cost-conscious:** nomic-embed-text (local, free)
- **Compliance (no API calls):** Mistral or self-hosted embedding model

---

### Similarity Metrics

Given two embeddings, how similar are they?

**Cosine Similarity** (most common)
```
sim(A, B) = (A · B) / (||A|| ||B||)

Range: -1.0 (opposite) to 1.0 (identical)

Normalized: Doesn't care about magnitude, only direction.

When to use: Almost always. Embedding models are trained for cosine.
```

**Euclidean Distance**
```
dist(A, B) = √(Σ(A_i - B_i)²)

Range: 0.0 (identical) to ∞

When to use: Rarely. Less intuitive than cosine.
```

**Dot Product**
```
dot(A, B) = A · B

Range: -∞ to ∞

When to use: When embeddings are pre-normalized (Pinecone does this).
Faster on GPU (no magnitude calculation).
```

**Production note:** Use cosine by default. If Pinecone forces dot product, normalize embeddings.

---

## Vector Database Comparison

### 1. Pinecone (Managed SaaS)

**Architecture:** Cloud-hosted, managed by Pinecone. No self-hosting.

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("rag-index")

# Insert vectors with metadata
index.upsert(vectors=[
    ("doc-1", [0.123, -0.456, ...], {"source": "earnings.pdf", "date": "2025-Q3"}),
    ("doc-2", [0.120, -0.450, ...], {"source": "policy.md"})
])

# Query
results = index.query(vector=[...], top_k=5, include_metadata=True)
# Returns: [("doc-1", 0.95, {...metadata}), ...]
```

**Pros:**
- Fully managed (no ops burden)
- Instant scaling to billions of vectors
- Built-in replication, backups
- Good for SaaS (multi-tenant)
- FedRAMP option for compliance

**Cons:**
- **Cost:** ~$0.10/month per pod (1M vectors), plus $0.10/million operations
- Vendor lock-in (proprietary API)
- Latency: 50-200ms (network overhead)

**Use case:** Production SaaS with 1M+ vectors, compliance needs.

---

### 2. Weaviate (Self-Hosted & Cloud)

**Architecture:** Open-source. Can self-host or use cloud.

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Define schema
client.schema.create_class({
    "class": "Document",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]}
    ]
})

# Insert
client.batch.add_objects(objects=[
    {
        "class": "Document",
        "properties": {"content": "...", "source": "earnings.pdf"},
        "vector": [0.123, -0.456, ...]
    }
])

# Query
results = client.query.get("Document").with_near_vector({
    "vector": [...],
    "certainty": 0.7  # Confidence threshold
}).do()
```

**Pros:**
- Open-source (no vendor lock-in)
- Self-hostable (data on your servers)
- GraphQL API (powerful, standard)
- Hybrid search (keyword + semantic)
- Good Kubernetes support

**Cons:**
- Requires self-hosting/ops (Docker, Kubernetes)
- Smaller ecosystem than Pinecone
- Cloud version ($15/month base)

**Use case:** On-prem compliance, hybrid search needed.

---

### 3. ChromaDB (Lightweight, Local)

**Architecture:** In-process database. No server needed.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Query
results = vectordb.similarity_search("What's our revenue?", k=5)
```

**Pros:**
- Zero setup (in-memory or local file)
- Perfect for prototyping
- Fast for small datasets (<1M vectors)
- Free

**Cons:**
- **Doesn't scale** to production (single-machine bottleneck)
- No replication, no cloud sync
- Slow for large datasets

**Use case:** Local RAG development, hackathons, tiny internal tools.

---

### 4. pgvector (PostgreSQL Extension)

**Architecture:** Stores vectors in your existing Postgres database.

```python
from langchain.vectorstores.pgvector import PGVector

vectordb = PGVector.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    connection_string="postgresql://user:pass@localhost/mydb"
)

# Query
results = vectordb.similarity_search("...", k=5)
```

**Pros:**
- **No new infrastructure** (use existing Postgres)
- ACID transactions (consistency guarantees)
- Metadata filtering in SQL (powerful)
- ~$15/month (AWS RDS), same as app DB
- SQL familiarity

**Cons:**
- Slower than specialized vector DB (~100-500ms latency)
- Doesn't scale as well (100M vector limit realistically)
- Requires Postgres knowledge
- Indexing requires tuning (HNSW or IVFFlat)

**Use case:** Existing Postgres infrastructure, <10M vectors, compliance/relational data needs.

---

### 5. FAISS (Facebook AI Similarity Search)

**Architecture:** Library, not a service. Runs locally.

```python
import faiss
import numpy as np

# Build index
vectors = np.array([...])  # Shape: (n_vectors, dims)
index = faiss.IndexFlatL2(dims)  # Euclidean distance
index.add(vectors)

# Query
distances, indices = index.search(query_vector.reshape(1, -1), k=5)
```

**Pros:**
- **Super fast** (<10ms for 1B vectors on CPU)
- Free, open-source
- Works offline
- Production-grade (used internally at Meta)

**Cons:**
- **Not a database** (no persistence, metadata, CRUD)
- You manage everything (indexing, updates)
- Requires Python/C++ engineering
- Doesn't scale to multiple machines easily

**Use case:** Research, offline similarity search, high-volume batch processing.

---

## Choosing a Vector DB

```
Decision tree:

Q: Compliance required (HIPAA, FedRAMP)?
├─ YES → Weaviate (self-host) OR Pinecone (FedRAMP)
└─ NO → Continue

Q: Existing Postgres infrastructure?
├─ YES → pgvector (extend what you have)
└─ NO → Continue

Q: <1M vectors, fast prototyping?
├─ YES → ChromaDB (free, quick)
└─ NO → Continue

Q: Production SaaS, need instant scaling?
├─ YES → Pinecone (managed, cheap at scale)
└─ NO → Continue

Q: Hybrid search (keyword + semantic) needed?
├─ YES → Weaviate (built-in BM25)
└─ NO → Continue

Default → Pinecone (best general-purpose) or Weaviate (open-source)
```

---

## Advanced Features

### Metadata Filtering

Not all vectors are created equal. Add metadata (source, date, category) and filter.

```python
# Insert with metadata
pinecone.Index("rag").upsert([
    ("doc-1", [0.123, ...], {
        "source": "earnings.pdf",
        "date": 2025,
        "category": "financial"
    })
])

# Query with filter
results = index.query(
    vector=[...],
    top_k=5,
    filter={"date": {"$gte": 2024}}  # Only 2024+ documents
)
```

**Use case:** Time-sensitive queries, multi-tenant isolation, compliance (filter by department).

---

### Namespace Partitioning (Multi-Tenancy)

Each tenant gets their own namespace (isolated vector space).

```python
# Tenant A
index.upsert(vectors=[...], namespace="tenant-a")

# Tenant B
index.upsert(vectors=[...], namespace="tenant-b")

# Query only Tenant A
results = index.query(vector=[...], namespace="tenant-a", top_k=5)
```

**Use case:** SaaS where data must never leak between customers.

---

## Deployment Patterns

### Pattern 1: Batch Ingestion

New documents arrive weekly. Embed and insert them.

```python
# Weekly job
def ingest_documents():
    documents = fetch_new_documents_from_s3()
    chunks = chunk_documents(documents)

    embeddings = embed_batch(chunks)

    # Batch insert (cheaper than 1-by-1)
    pinecone.Index("rag").upsert(
        vectors=embeddings,
        batch_size=100
    )

    log_ingestion_complete()
```

**Cost optimization:** Batch inserts are cheaper than streaming.

---

### Pattern 2: Real-Time Indexing

Documents updated constantly. Index immediately.

```python
@app.post("/upload")
def upload_document(file):
    content = extract_text(file)
    chunks = chunk(content)
    embeddings = embed(chunks)

    vectordb.upsert(embeddings)  # Immediate

    return {"status": "indexed"}
```

**Latency tradeoff:** Real-time costs more but guarantees freshness.

---

## Production Checklist

- [ ] **Embedding model chosen** and cost understood
- [ ] **Vector DB selected** (Pinecone, Weaviate, pgvector?)
- [ ] **Similarity metric** understood (cosine default)
- [ ] **Metadata schema** designed (source, date, category)
- [ ] **Namespace/multi-tenancy** plan (if SaaS)
- [ ] **Ingestion pipeline** automated (batch or real-time)
- [ ] **Search latency** target set (<500ms for user-facing)
- [ ] **Replication/HA** configured
- [ ] **Cost model** calculated (vectors * storage + operations)
- [ ] **Monitoring** in place (query latency, insert latency, dead connections)

---

## System Design Interview Talking Points

When asked about vector DBs:

1. **Why vector DB:** "Semantic search beats keyword matching. Traditional DBs don't support nearest-neighbor efficiently."

2. **Pinecone vs Weaviate:** "Pinecone if SaaS with scale/compliance needs. Weaviate if on-prem or need hybrid search."

3. **Cost at scale:** "1B vectors in Pinecone = ~$100K/month. Use caching and hybrid search to reduce queries."

4. **Latency:** "Pinecone: ~50-200ms. pgvector: ~100-500ms. FAISS: <10ms but no metadata/CRUD."

5. **Metadata filtering:** "Always add metadata. Filters reduce search space, improve precision, enable multi-tenancy."

6. **Namespaces for isolation:** "SaaS needs per-tenant namespaces to prevent data leaks."
