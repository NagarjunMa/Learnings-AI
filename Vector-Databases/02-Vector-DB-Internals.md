# Vector Database Internals: Semantic Storage, Indexing, and Quantization

## How Vector Databases Store Semantic Data

### The End-to-End Pipeline

1. **Embedding Phase:** Raw text → embedding model → dense vector (768D, 1536D, 3072D)
   - Embedding model captures semantic meaning: "dog" and "puppy" → similar vectors (high cosine similarity)
   - Semantic similarity ≈ geometric proximity: objects meaning the same thing cluster together in vector space
   
2. **Storage Phase:** Vector stored in memory/disk with metadata
   - Native vector column: PostgreSQL `pgvector(768)`, Pinecone native, Weaviate native
   - Index structure organizes vectors for fast search (not linear scan)
   
3. **Query Phase:** Query text → same embedding model → search index for nearest neighbors

### Why Semantic > Keyword Search
- Keyword: exact term match only. "car insurance" misses "vehicle coverage"
- Semantic: captures intent. Both return when user intent is about insurance for vehicles
- Cost: embeddings are expensive (5x cheaper with batch APIs) but filter results before hitting LLM

---

## ANN: Approximate Nearest Neighbor Search

### The Problem It Solves
Exact KNN search is O(n*d): compare query vector to all n vectors in d dimensions.
- 1M vectors, 768 dims: 768M distance calculations per query — too slow for production (<100ms latency)
- ANN sacrifices exactness for speed: return "good enough" neighbors, not guaranteed best ones

### Recall vs Latency Tradeoff
- **Exact KNN:** 100% recall, O(n*d) latency — unusable at scale
- **ANN:** 90–99% recall, O(log n) or O(√n) latency — practical for production
- Industry standard: 99% recall at <100ms for 1M vector datasets

---

## HNSW: Hierarchical Navigable Small World

### Why It's Default (Pinecone, Weaviate, Qdrant)
Best balance: fast construction, high recall, low query latency, simple parameters.

### How It Works

**Layer-based structure:**
- Build multi-layer graph (L layers, L ~ log(n))
- Top layers sparse (contain ~10% of vectors), bottom layers dense (all vectors)
- Query starts at top, navigates down through layers

**Insert algorithm:**
```
1. Find insertion level l_new (typically random, ~ln(2) * level)
2. At each layer from top down:
   - Find M nearest neighbors (Candidate List)
   - Insert new vector, connect to these M neighbors
   - Layer's graph is updated
3. Connections are bidirectional (undirected graph)
```

**Search algorithm:**
```
1. Start at top layer, random entry point
2. Greedy search: find nearest neighbor to query in current layer
3. If neighbor is closer than entry point, move there (repeat until converged)
4. Move to layer below, repeat until layer 0
5. Return top k nearest neighbors from layer 0
```

**Key Parameters:**
- `M`: max number of connections per node (higher M = more neighbors, slower insert, better recall) — default 16
- `efConstruction`: size of candidate list during insert (higher = better recall but slower insert) — default 200
- `efSearch`: size of candidate list during query (higher = better recall but slower query) — default ef (often 200)

**Interview insight:** "HNSW navigation is greedy search through a graph. At each layer, find the locally closest neighbor, move there, keep going until no improvement. Drop to next layer, repeat. This logarithmic traversal avoids scanning all n vectors."

---

## IVF: Inverted File Index

### When to Use Over HNSW
- **Cheaper construction:** O(n) instead of O(n log n), no random access penalty
- **Easier approximate:** combine with PQ for aggressive compression
- **Fewer parameters:** fewer tuning knobs, simpler training

### How It Works

**Training phase (offline):**
```
1. Sample K vectors from dataset
2. Run k-means clustering on these K centroids
3. Store centroids, discard sampled vectors
```

**Insert phase:**
```
For each vector:
1. Find nearest centroid (Voronoi cell assignment)
2. Append vector to that centroid's "inverted file" (list)
```

**Search phase:**
```
1. Find nearest centroids to query (top 1, top 5, configurable)
2. Search within each centroid's inverted file (linear scan within cluster)
3. Return top k overall
```

**Key Parameters:**
- `nlist`: number of clusters (higher = finer partitioning, slower insert, faster query) — default ~√n
- `nprobe`: how many centroids to search (1 = only closest cluster, 5 = closest 5) — default 1
- Higher nprobe = higher recall, slower query

**Interview insight:** "IVF pre-partitions space with k-means, then search only probes a subset of clusters. If you want high recall, increase nprobe. If you want speed, decrease nprobe and accept lower recall."

---

## Product Quantization (PQ)

### Why: Memory is the Bottleneck

Storage cost per vector:
```
768-dim float32 vector:  768 * 4 bytes = 3,072 bytes
1M vectors × 3KB = 3GB RAM required
```

Quantization goal: compress to ~200 bytes (6% of original) while keeping search quality.

### How Product Quantization Works

**Core idea:** Split vector into M chunks, quantize each chunk independently.

**Training phase (offline):**
```
1. Sample random vectors from dataset
2. Split each into M subvectors of size d/M
3. For each subvector position:
   - Run k-means with k=256 clusters on that position
   - Store the 256 centroids (codebook) for that position
4. Result: M codebooks, each with 256 centroids
```

**Encode phase:**
```
For each vector:
1. Split into M subvectors
2. For each subvector:
   - Find nearest centroid in that position's codebook
   - Store centroid ID (1 byte, 0-255) instead of full subvector
3. Result: M-byte code instead of 768-float vector
```

Example: 768-dim with M=96 → 96-byte code (32x compression).

**Search phase — two options:**

**(1) ADC (Asymmetric Distance Computation):**
```
For each database vector (encoded as PQ code):
1. Decode PQ code back to approximate full vector
2. Compute distance to query (full vector)
3. Keep top k
```
Fast query (~1ms for 1M vectors), acceptable recall loss.

**(2) OPQ (Optimized PQ):**
```
1. Pre-rotate data + query to maximize variance captured in early chunks
2. Use truncated PQ: only compute distance using first L/M chunks
3. Recall loss is lower than standard PQ
```

### IVFPQ: Combining IVF + PQ

**Best of both:**
```
Training:
1. Run k-means to get nlist clusters (IVF)
2. Within each cluster, train PQ codebooks (PQ)

Insertion:
1. Find nearest cluster centroid
2. Encode vector with PQ, append to cluster

Search:
1. Find nearest nprobe clusters
2. Within each cluster, use ADC (PQ codes + codebooks)
3. Return top k overall
```

**Recall vs latency vs memory:**
- Pure IVF: high recall, medium memory, slow search
- IVF + PQ: medium recall (95%+), low memory (32x compression), fast search
- PQ alone: medium recall, lowest memory, fastest search

**Interview talking points:**
- "PQ trades recall for memory: 768-dim → 96-byte code is 32x compression with ~95% recall"
- "IVFPQ is practical: partition space (IVF), compress within each partition (PQ)"
- "ADC keeps full query vector, only DB vectors are PQ-encoded; asymmetric distance computation"

---

## Quantization Techniques Comparison

| Technique | Compression | Recall | Use Case |
|-----------|------------|--------|----------|
| No quantization | 1x | 100% | Small dataset, high recall required |
| Scalar (INT8) | 4x | 98%+ | Moderate data, simple inference |
| Binary quantization | 32x | 92–95% | Large scale, extreme compression |
| PQ (M=96) | 32x | 95%+ | Large scale, balanced |
| IVFPQ (nlist=1024, M=96) | 32x | 96%+ | Very large (10M+), practical production |
| OPQ | 32x | 96%+ | High-recall requirement, offline training |

### When to Use Each

**PQ vs Binary:**
- PQ: if you need >94% recall
- Binary: if you need extreme compression and can sacrifice recall to ~90%

**HNSW vs IVFPQ:**
- HNSW: smaller datasets (<10M), need high recall with fewer parameters
- IVFPQ: massive datasets (10M+), can afford offline training phase

---

## Production Checklist: Vector DB Internals

- [ ] Know your embedding model dimensionality (768, 1536, 3072) and cost per token
- [ ] Understand indexing trade-off: HNSW for simplicity, IVF/IVFPQ for scale and compression
- [ ] If using PQ: test recall with your actual query distribution (synthetic tests ≠ real queries)
- [ ] Set efSearch/nprobe dynamically: high recall for critical queries (e.g., retrieval), low for pre-filtering
- [ ] Monitor index size vs memory budget; quantization adds encoding/decoding overhead (~5% latency)
- [ ] Batch embeddings API to amortize cost (batches of 100 vectors = 50% cheaper)
- [ ] Test re-indexing: is it in-place (HNSW) or full rebuild (IVF)? Plan accordingly for large updates
- [ ] Measure actual recall, not theoretical — run offline evaluation with real query logs

---

## Interview Talking Points

**"Explain how a vector database stores semantic information"**
- Embedding model captures meaning as dense vectors: similar concepts cluster in vector space
- Index structures (HNSW, IVF) enable fast nearest-neighbor search without scanning all vectors
- Trade semantic precision for recall/latency based on use case

**"How does HNSW work?"**
- Multi-layer graph: top layers sparse, lower layers dense
- Insert: find nearest neighbors at each layer, build bidirectional connections
- Search: greedy graph traversal layer-by-layer, mimics small-world graph navigation
- Parameters: M (degree), efConstruction (candidate list size), efSearch (query candidate list)

**"Why use Product Quantization?"**
- Memory bottleneck: 768-dim float32 = 3KB per vector × 1M vectors = 3GB
- PQ solution: split into M chunks, each chunk quantized to k-means centroid ID (1 byte)
- Result: 768-dim → 96-byte code, 32x compression, ~95% recall
- Search with ADC: keep query full precision, compute distances using PQ-approximated vectors

**"When would you use IVFPQ over HNSW?"**
- Dataset scale: IVFPQ for 10M+ vectors, HNSW practical up to 100M
- Memory constraint: IVFPQ enables compression, HNSW doesn't
- Construction time: HNSW faster online insertion, IVFPQ cheaper if batch offline training is ok
- Recall needs: both can hit 95%+, HNSW simpler tuning

**"How do you optimize vector DB for production?"**
- Measure actual recall with real queries, not synthetic tests
- Batch embed: 100 vectors per request is 50% cheaper than 1-at-a-time
- Adjust search parameters (efSearch, nprobe) per query type: strict retrieval vs pre-filter
- Monitor index size + memory; quantization adds encode/decode overhead
