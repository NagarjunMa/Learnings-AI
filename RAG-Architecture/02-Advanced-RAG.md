# Advanced RAG — Agentic RAG, Corrective RAG, GraphRAG, vs Fine-Tuning

Vanilla RAG (retrieve then generate) has a critical flaw: it retrieves before reasoning about what to retrieve. Advanced patterns flip this.

## Problem with Vanilla RAG

Vanilla RAG flow:
1. User query
2. Retrieve top-k docs from vector DB (static strategy)
3. Concatenate into prompt
4. LLM generates answer

**Problems:**
- Query ambiguity: "What is the best ML model?" doesn't specify domain (NLP, vision, RL?)
- Over-retrieval: retrieve 10 docs, but only 2 are relevant
- Under-retrieval: static top-k misses relevant docs
- No introspection: LLM doesn't know if it has enough context

## Pattern 1: Agentic RAG

Agent decides **when** and **what** to retrieve.

```python
from langchain.agents import create_tool_calling_agent
from langchain_community.tools.retriever import create_retriever_tool
from langchain.chat_models import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet")

# Define retrieval as a tool (not automatic)
retriever_tool = create_retriever_tool(
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    name="search_documents",
    description="Search our knowledge base. Use when you need factual information."
)

tools = [retriever_tool]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a document expert. Use search_documents to answer questions."),
        ("human", "{input}")
    ])
)

# Agent decides when to search
from langchain.agents import AgentExecutor
executor = AgentExecutor(agent=agent, tools=tools)

# Query: agent reasons → decides if search needed → retrieves if needed → answers
result = executor.invoke({
    "input": "What is the best ML model?"
})
```

Agent trace:

```
Agent: "I need to search for information about ML models to answer this."
  → Tool call: search_documents(query="best ML model for NLP")
  → Results: [doc1, doc2, doc3]
Agent: "Based on the documents, GPT-4 and Claude are best for NLP."
  → Return answer
```

**Benefits:**
- Multi-step reasoning: agent can search → read → search again
- Query expansion: agent reformulates ambiguous query ("best ML model?" → "best model for NLP classification")
- Conditional retrieval: search only when needed

**Implementation with LangGraph:**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class RAGState(TypedDict):
    question: str
    context: list[str]
    answer: str
    retrieval_count: int

def node_route(state: RAGState):
    """Decide: should we retrieve?"""
    if state["retrieval_count"] >= 3:
        return "answer"  # max 3 retrievals
    
    # Ask LLM: do you need more info?
    decision = llm.invoke(
        f"Question: {state['question']}\n"
        f"Current context: {state['context']}\n"
        f"Do you need more information? (yes/no)"
    )
    
    return "retrieve" if "yes" in decision.lower() else "answer"

def node_retrieve(state: RAGState):
    """Retrieve based on question + current context."""
    # Query expansion: ask LLM what to search for
    search_query = llm.invoke(
        f"What should I search for to answer: {state['question']}?\n"
        f"Already know: {state['context']}"
    )
    
    docs = vector_db.similarity_search(search_query, k=5)
    state["context"].extend([d.page_content for d in docs])
    state["retrieval_count"] += 1
    
    return state

def node_answer(state: RAGState):
    """Generate final answer with all context."""
    prompt = f"Context:\n{'\n'.join(state['context'])}\n\nQuestion: {state['question']}"
    state["answer"] = llm.invoke(prompt)
    return state

graph = StateGraph(RAGState)
graph.add_node("retrieve", node_retrieve)
graph.add_node("route", node_route)
graph.add_node("answer", node_answer)

graph.add_edge("START", "route")
graph.add_conditional_edges("route", 
    lambda x: "retrieve" if x.get("needs_retrieval") else "answer"
)
graph.add_edge("retrieve", "route")
graph.add_edge("answer", END)

agent_rag = graph.compile()
result = agent_rag.invoke({
    "question": "Best ML model?",
    "context": [],
    "retrieval_count": 0
})
```

## Pattern 2: Corrective RAG (CRAG)

Detect retrieval quality and adjust.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def corrective_rag(question: str):
    """Retrieve, evaluate, and correct if needed."""
    
    # Step 1: Retrieve
    docs = vector_db.similarity_search(question, k=5)
    context = "\n".join([d.page_content for d in docs])
    
    # Step 2: Evaluate retrieval quality
    # Use LLM to score: is retrieved context relevant?
    evaluation = llm.invoke(
        f"Rate relevance of this context to the question (0-1):\n"
        f"Question: {question}\n"
        f"Context: {context}"
    )
    
    score = float(evaluation.split(":")[-1].strip())
    
    # Step 3: If score low, correct
    if score < 0.5:
        # Strategy 1: Expand query (add keywords)
        expanded_query = llm.invoke(
            f"Expand this question with relevant keywords:\n{question}"
        )
        docs = vector_db.similarity_search(expanded_query, k=10)
        context = "\n".join([d.page_content for d in docs])
        
        # Strategy 2: If still bad, use web search
        if score < 0.3:
            web_results = search_web(question)
            context += "\n" + "\n".join(web_results)
    
    # Step 4: Generate with corrected context
    answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {question}")
    return answer
```

**Flow:**
- Retrieve → Score → (If low) Expand query → Retrieve again → (If still low) Web search → Answer

**Implementation note:** CRAG requires grading model (smaller/faster LLM to evaluate).

## Pattern 3: GraphRAG

Enhance retrieval with knowledge graph structure.

Standard RAG: documents → chunks → embeddings → vector search.
GraphRAG: documents → extract entities + relationships → knowledge graph → graph traversal + embeddings.

```python
# Example: Financial documents → entity extraction
from langchain.chains.question_answering import load_qa_chain
from langchain.graphs import Neo4jGraph

# Step 1: Extract entities from docs
def extract_entities(text: str):
    """Use LLM to extract entities + relationships."""
    prompt = """Extract entities and relationships from this text.
    Format as JSON: {
        "entities": [{"name": "...", "type": "company|person|regulation"}],
        "relationships": [{"from": "...", "to": "...", "relation": "..."}]
    }
    
    Text: """ + text
    
    result = llm.invoke(prompt)
    return json.loads(result)

# Step 2: Build knowledge graph
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="...")

for doc in documents:
    entities = extract_entities(doc.text)
    
    for entity in entities["entities"]:
        graph.query(f"CREATE (n:{entity['type']} {{name: '{entity['name']}'}})")
    
    for rel in entities["relationships"]:
        graph.query(
            f"MATCH (a), (b) WHERE a.name = '{rel['from']}' AND b.name = '{rel['to']}' "
            f"CREATE (a)-[:{rel['relation']}]->(b)"
        )

# Step 3: Hybrid retrieval (embedding + graph)
def graphrag_search(question: str):
    # Semantic search
    semantic_docs = vector_db.similarity_search(question, k=5)
    
    # Graph search: extract entities from question, traverse graph
    entities = extract_entities(question)
    for entity in entities["entities"]:
        related = graph.query(
            f"MATCH (n:{entity['type']} {{name: '{entity['name']}'}}) "
            f"-[*1..2]-(m) RETURN m"
        )
        # Convert graph nodes to text
    
    # Combine results
    combined_context = "\n".join(
        [d.page_content for d in semantic_docs] + related_texts
    )
    
    answer = llm.invoke(f"Context:\n{combined_context}\n\nQuestion: {question}")
    return answer
```

**Benefits:**
- Captures relationships (semantic search only captures similarity)
- Example: "Tell me about companies with exports to China" → graph finds export relationships directly
- Reduces hallucination: facts are extracted from documents, not LLM-generated

**Trade-off:** Expensive (requires entity extraction + graph construction). Use for domain-specific apps (finance, legal, biotech).

## Pattern 4: Self-RAG

Detect if LLM's answer is backed by retrieved context.

```python
def selfrag(question: str):
    """Retrieve, generate, then reflect on quality."""
    
    # Step 1: Retrieve
    docs = vector_db.similarity_search(question, k=5)
    context = "\n".join([d.page_content for d in docs])
    
    # Step 2: Generate
    answer = llm.invoke(
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        f"Also provide citations: [source1] ... [source2]"
    )
    
    # Step 3: Reflect - is answer supported by context?
    reflection = llm.invoke(
        f"Is this answer supported by the context?\n"
        f"Context: {context}\n"
        f"Answer: {answer}\n"
        f"Score (0-1) and explain why."
    )
    
    score = float(reflection.split(":")[0])
    
    # Step 4: If low confidence, regenerate with retrieved context emphasized
    if score < 0.6:
        answer = llm.invoke(
            f"Critically important - ONLY answer based on this context:\n{context}\n\n"
            f"Question: {question}\n"
            f"If you cannot answer from context, say 'I don't have enough information'."
        )
    
    return answer
```

**Use case:** High-stakes applications (legal, medical) where hallucinations are unacceptable.

## RAG vs Fine-Tuning — Decision Matrix

When to use which?

| Factor | RAG | Fine-Tuning |
|---|---|---|
| **Knowledge updates** | Real-time (just re-index docs) | Slow (retrain model) |
| **Knowledge depth** | Broad (entire corpus searchable) | Deep (absorbed into params) |
| **Cost** | Lower (vector DB + retrieval) | Higher (training infra) |
| **Latency** | Higher (retrieval step) | Lower (direct prediction) |
| **Use case** | Open-domain QA, document search | Task-specific (classification, summarization style) |
| **Hallucination** | Lower if grounded in retrieval | Higher (model may fabricate) |
| **Compliance** | Better (citations trace to source) | Worse (can't explain origin of facts) |

**Combination:** RAG + fine-tuning.
- Fine-tune on domain-specific instruction following (e.g., financial domain language style)
- RAG for factual grounding (retrieve latest regulations, recent news)

Example: Financial advisor bot
- Fine-tune on financial advisor persona + task (loan approval reasoning)
- RAG retrieves: compliance docs, customer history, market data
- Result: fine-tuned style + current facts

## Interview Talking Points

- "Agentic RAG: agent decides when/what to retrieve. Solves vanilla RAG's static retrieval problem. Use LangGraph with conditional edges."
- "Corrective RAG: evaluate retrieval quality, expand query if needed. Trade-off: extra LLM calls for better quality."
- "GraphRAG: extract entities + relationships, build knowledge graph. Better for relationship-heavy domains (finance, legal). Expensive."
- "Self-RAG: reflect on answer quality, regenerate if low confidence. Critical for compliance/legal applications."
- "RAG for knowledge updates (real-time), fine-tuning for task-specific behavior. Often combine both."
- "If compliance matters, RAG > fine-tuning (citations provide audit trail)."
