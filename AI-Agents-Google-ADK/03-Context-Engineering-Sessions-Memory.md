# Context Engineering: Sessions & Memory

> Google Whitepaper — Authors: Kimberly Milam, Antonio Gulli (Nov 2025)

> **Stateful and personal AI begins with Context Engineering.**

## Introduction

Whitepaper explores Sessions and Memory in stateful, intelligent LLM agents.

LLMs are inherently stateless. To enable them to **remember, learn, and personalize**, developers must dynamically assemble and manage information within the context window — process known as **Context Engineering**.

Core concepts:
- **Context Engineering**: dynamic assembly/management of info within LLM's context window for stateful agents
- **Sessions**: container for an entire conversation with an agent, holding chronological history + agent's working memory
- **Memory**: long-term persistence mechanism, capturing/consolidating key info across multiple sessions for continuous personalized experience

## Context Engineering

LLMs are stateless. Outside training data, reasoning/awareness confined to "context window" of single API call. AI agents must be equipped with operating instructions identifying what actions can be taken, evidential and factual data to reason over, and immediate conversational information defining current task.

To build stateful agents, developers construct context for **every turn of conversation**. Dynamic assembly = **Context Engineering**.

Evolution from traditional **Prompt Engineering**:
- Prompt engineering = crafting optimal, often static system instructions
- Context Engineering = entire payload, dynamically constructing state-aware prompt based on user, conversation history, external data
- Strategically selects/summarizes/injects different info types to maximize relevance while minimizing noise
- External systems (RAG dbs, session stores, memory managers) manage much of this context. Agent framework orchestrates these to retrieve and assemble context into final prompt.

**Mise en place** analogy — chef gathering and preparing ingredients before cooking. If you only give the chef the recipe (the prompt), they might produce an okay meal with random ingredients. If you first ensure they have right, high-quality ingredients, specialized tools, clear understanding of presentation style, they reliably produce excellent customized result. Goal: model has **no more and no less** than the most relevant info to complete its task.

### Components of Context Payload

**Context to guide reasoning** — defines fundamental reasoning patterns and available actions:
- **System Instructions**: high-level directives defining persona, capabilities, constraints
- **Tool Definitions**: schemas for APIs/functions agent can use to interact externally
- **Few-Shot Examples**: curated examples guiding model's reasoning via in-context learning

**Evidential & Factual Data** — substantive data agent reasons over (the "evidence"):
- **Long-Term Memory**: persisted knowledge about user/topic, gathered across multiple sessions
- **External Knowledge**: info retrieved from databases/documents, often using **RAG**
- **Tool Outputs**: data/results returned by a tool
- **Sub-Agent Outputs**: conclusions/results from delegated sub-agents
- **Artifacts**: non-textual data (files, images) associated with user/session

**Immediate conversational information** — grounds agent in current interaction:
- **Conversation History**: turn-by-turn record of current interaction
- **State / Scratchpad**: temporary, in-progress info/calculations agent uses for immediate reasoning
- **User's Prompt**: immediate query

Dynamic construction critical. Memories not static; selectively retrieved/updated as user interacts or new data ingested. Effective reasoning relies on **in-context learning** (LLM learning from demonstrations in prompt). More effective with relevant few-shot examples than hardcoded ones. External knowledge retrieved by RAG tools based on immediate query.

**Context rot**: phenomenon where model's ability to pay attention to critical info diminishes as context grows. Context Engineering addresses via summarization, selective pruning, compaction techniques to preserve vital info while managing token count.

### Continuous Cycle in Agent's Operational Loop

> **Figure 1**: Flow of context management for agents
> - User Query → Fetch Context ↔ Context Storage → Prepare Context → Invoke LLMs + Tools → Upload Context (events, async)
> - Agent Response (streaming)
> - Loop: agent "decides" to fetch context

For each turn:
1. **Fetch Context**: retrieve user memories, RAG documents, recent events. Use user query + metadata to identify what to retrieve.
2. **Prepare Context**: framework dynamically constructs full prompt for LLM call. Individual API calls may be async, but preparing context is a blocking, "hot-path" process — agent cannot proceed until context ready.
3. **Invoke LLM and Tools**: iteratively calls LLM and necessary tools until final response generated. Tool/model output appended to context.
4. **Upload Context**: new info gathered uploaded to persistent storage. Often background process, allowing agent to complete execution while memory consolidation/post-processing occurs asynchronously.

At heart: **sessions** + **memory**. Session = turn-by-turn state of single conversation. Memory = mechanism for long-term persistence, capturing/consolidating key info across multiple sessions.

**Workbench/Filing-cabinet analogy**: Session = workbench/desk you're using for a specific project. Covered in necessary tools/notes/reference materials. Immediately accessible but temporary and specific. When project finished, you don't shove messy desk into storage. Instead: **create memory** = organized filing cabinet. Review materials, discard rough drafts, file only critical finalized documents into labeled folders. Filing cabinet = clean, reliable, efficient source of truth for future projects.

## Sessions

A foundational element of Context Engineering. **Encapsulates immediate dialogue history and working memory for a single, continuous conversation.** Each session = self-contained record tied to specific user. Allows agent to maintain context within a single conversation. User can have multiple sessions; each = distinct, disconnected log.

Two key components:
- **Events**: chronological history (the building blocks of conversation)
  - **user input** (text, audio, image)
  - **agent response** (reply to user)
  - **tool call** (decision to use external tool/API)
  - **tool output** (data returned from tool call)
- **State**: structured "working memory" / scratchpad. Temporary structured data relevant to current conversation (e.g. shopping cart contents).

As conversation progresses, agent appends events to session and may mutate state.

Events structure analogous to list of `Content` objects passed to Gemini API — each item with `role` and `parts` represents one turn/Event.

```python
contents = [
    {
        "role": "user",
        "parts": [ {"text": "What is the capital of France?"} ]
    }, {
        "role": "model",
        "parts": [ {"text": "The capital of France is Paris."} ]
    }
]
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents
)
```

Production agent execution typically stateless — retains no info after request. Conversation history must be saved to persistent storage. In-memory storage suitable for dev; production should leverage robust databases. Example: store conversation history in managed solutions like **Agent Engine Sessions**.

### Variance across frameworks and models

Agent frameworks responsible for: maintaining conversation history/state, building LLM requests using context, parsing/storing LLM response.

Frameworks act as **universal translator** between code and LLM. Developer works with framework's consistent internal data structures. Framework converts to precise format LLM requires. Decouples agent's logic from specific LLM, prevents vendor lock-in.

> **Figure 2**: Flow of context management for agents
> - Model (Gemini) ↔ Model Serving (Model Garden, Endpoints) ↔ Agent Framework (ADK, LangGraph) [translate/translate]
> - Agent Framework ↔ Other data sources (GCS, RAG Engine), Session Store (Agent Engine Storage) [no translation, strong contract]
> - Memory Manager (Agent Engine Memory Bank) [lightweight translation, generation only]
> - User Query → Agent Framework

Goal: produce "request" LLM understands. For Gemini = `List[Content]`. Each `Content` = dict-like with two keys:
- `role`: "user" or "model"
- `parts`: actual content (text, images, tool calls)

Framework automatically maps internal object (e.g. ADK `Event`) to corresponding role/parts in `Content` before API call. Stable internal API for developer; framework manages varied external APIs.

**ADK** uses explicit `Session` object containing list of `Event` objects + separate state object. Session = filing cabinet, one folder for events, another for working memory state.

**LangGraph** has no formal "session" object. **State is the session** — all-encompassing state object holds conversation history (list of `Message` objects) + all working data. Unlike append-only log of traditional session, LangGraph state is **mutable** — can be transformed; strategies like history compaction can alter the record. Useful for managing long conversations and token limits.

### Sessions for multi-agent systems

Multiple specialized agents collaborate. System architecture defines communication patterns. Central to architecture: how system handles **session history** — persistent log of all interactions.

> **Figure 3**: Different multi-agent architectural patterns
> - Single Agent (LLM with tools)
> - Network (agents in mesh)
> - Supervisor (one orchestrator, several workers)
> - Supervisor (as tools) — orchestrator with tools as agents
> - Hierarchical
> - Custom

Distinguish session history from context sent to LLM:
- Session history = **permanent, unabridged transcript** of entire conversation
- Context = **carefully crafted info payload** sent to LLM for a single turn

Agent might construct context by selecting only relevant excerpt from history or adding special formatting (guiding preamble). **What's passed across agents** ≠ **what context is sent to LLM**.

Two primary approaches to handling session history for multi-agent systems:

#### Shared, unified history

All agents read from + write to same single conversation history. Every message, tool call, observation appended to one central log in chronological order. Best for **tightly coupled, collaborative tasks** requiring single source of truth (e.g. multi-step problem-solving where one agent's output is direct input for next). Sub-agent might process log before passing to LLM (filter for relevant events, add labels identifying which agent generated each event).

ADK's LLM-driven delegation to handoff to sub-agents → all intermediary events of sub-agent written to same session as root agent:

```python
from google.adk.agents import LlmAgent

# The sub-agent has access to Session and writes events to it.
sub_agent_1 = LlmAgent(...)

# Optionally, the sub-agent can save the final response text (or structured
# output) to the specified state key.
sub_agent_2 = LlmAgent(
    ...,
    output_key="..."
)

# Parent agent.
root_agent = LlmAgent(
    ...,
    sub_agents=[sub_agent_1, sub_agent_2]
)
```

#### Separate, individual histories

Each agent maintains its own private conversation history → functions like a black box to other agents. Internal processes (intermediary thoughts, tool use, reasoning) kept private. Communication occurs only through explicit messages where agent shares final output, not its process.

Implemented by **Agent-as-a-tool** or **Agent-to-Agent (A2A) Protocol**:
- **Agent-as-a-Tool**: one agent invokes another like a tool, passing inputs and receiving final self-contained output
- **A2A Protocol**: agents use structured protocol for direct messaging

### Interoperability across multiple agent frameworks

> **Figure 4**: A2A communication across multiple agents using different frameworks
> - Two Agents communicating via A2A protocol across organizational/technological boundaries
> - Agent 1: Local Agents → Vertex AI (Gemini API, 3P) ← Session storage; Agent Development Kit (ADK) ↔ MCP ↔ APIs & Enterprise Applications
> - Agent 2: Local Agents → LLM ← Checkpoint storage; LangGraph ↔ MCP ↔ APIs & Enterprise Applications

Framework's internal data representation = critical architectural trade-off. Decoupling from LLM also isolates from agents using other frameworks. Solidified at persistence layer. Storage model for `Session` typically couples DB schema directly to framework's internal objects → rigid, non-portable conversation record. **LangGraph cannot natively interpret distinct `Session` and `Event` objects persisted by ADK-based agent → seamless task handoffs impossible**.

**Agent-to-Agent (A2A)** is one emerging architectural pattern for coordinating isolated agents. Enables agents to exchange messages but **fails to address core problem of sharing rich, contextual state**. Each agent's history encoded in framework's internal schema. A2A message containing session events requires translation layer.

More robust pattern: **abstracting shared knowledge into framework-agnostic data layer such as Memory**. Unlike Session store (preserves raw, framework-specific objects like `Events` and `Messages`), memory layer holds **processed, canonical info** — summaries, extracted entities, facts. Stored as strings/dictionaries. Memory layer's data structures not coupled to any single framework's internal representation → universal, common data layer. Heterogeneous agents achieve true collaborative intelligence by sharing common cognitive resource without custom translators.

### Production Considerations for Sessions

When moving to production, session management must evolve from simple log to robust enterprise-grade service. Three critical areas: **security and privacy, data integrity, performance**. Managed session store like Agent Engine Sessions specifically designed.

#### Security and Privacy

Protecting sensitive info in session = non-negotiable. **Strict Isolation** = most critical principle. Session owned by single user; system must enforce strict isolation so one user can never access another's session data (via ACLs). Every request to session store must be authenticated and authorized against session's owner.

Best practice: **redact PII before session data is written to storage**. Drastically reduces risk and "blast radius" of potential breach. Sensitive data never persisted using tools like **Model Armor** → simplifies GDPR/CCPA compliance and builds trust.

#### Data Integrity and Lifecycle Management

Production system requires clear rules for data storage/maintenance over time. Sessions should not live forever. Implement **Time-to-Live (TTL)** policy to automatically delete inactive sessions → manage storage costs, reduce data management overhead. Clear data retention policy defining how long sessions kept before archived/permanently deleted.

System must guarantee operations are appended in **deterministic order**. Maintaining correct chronological sequence of events fundamental to integrity of conversation log.

#### Performance and Scalability

Session data on "hot path" of every user interaction → performance is primary concern. Reading/writing must be extremely fast. Agent runtimes typically stateless → entire session history retrieved from central database at start of every turn → network transfer latency.

To mitigate latency, reduce data transferred. Filter or compact session history before sending to agent. Remove old, irrelevant function call outputs no longer needed for current state.

### Managing long context conversation: tradeoffs and optimizations

Simplistic architecture: session = immutable log of conversation. As conversation scales, token usage increases. Modern LLMs handle long contexts but limitations exist, especially for **latency-sensitive applications**:

1. **Context Window Limits**: every LLM has max text amount. If history exceeds, API call fails.
2. **API Costs ($)**: providers charge based on tokens sent/received. Shorter histories = lower costs per turn.
3. **Latency (Speed)**: more text = longer to process.
4. **Quality**: as token count increases, performance can worsen due to noise + autoregressive errors.

**Suitcase analogy**: agent's context window = limited suitcase. Stuff everything in → too heavy and disorganized, hard to find what you need quickly. Pack too little → leave behind essential items like passport. Both traveler and agent operate under same constraint: success hinges not on how much you can carry, but on **carrying only what you need**.

**Compaction strategies** shrink long histories, condensing dialogue to fit context window, reducing API costs and latency. Strategies range from simple truncation to sophisticated compaction:

- **Keep the last N turns**: simplest. Only keep most recent N turns ("sliding window") and discard everything older.
- **Token-Based Truncation**: count tokens starting with most recent and working backward. Include as many messages as possible without exceeding limit (e.g. 4000 tokens). Everything older cut off.
- **Recursive Summarization**: older parts replaced by AI-generated summary. As conversation grows, agent periodically uses another LLM call to summarize oldest messages. Summary used as condensed form of history, often prefixed to more recent verbatim messages.

**Keep last N turns with ADK** using built-in plug-in (does not modify historical events stored):

```python
from google.adk.apps import App
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin

app = App(
    name='hello_world_app',
    root_agent=agent,
    plugins=[
        # Keep the last 10 turns and the most recent user query.
        ContextFilterPlugin(num_invocations_to_keep=10),
    ],
)
```

Sophisticated compaction strategies aim to reduce cost/latency → critical to perform expensive operations (recursive summarization) **asynchronously in background and persist results**. Agent's memory manager often responsible for both generating and persisting these summaries. Agent must keep record of which events included in compacted summary → prevents original verbose events from being needlessly re-sent.

Agent must decide **when** compaction necessary. Triggers fall into categories:
- **Count-Based Triggers** (token size or turn count threshold): compacted once conversation exceeds predefined threshold. Often "good enough" for managing context length.
- **Time-Based Triggers**: not by size, but by lack of activity. User stops interacting for set period (e.g. 15-30 min) → run compaction job in background.
- **Event-Based Triggers** (Semantic/Task Completion): agent decides when specific task/sub-goal/topic concludes.

ADK's `EventsCompactionConfig` triggers LLM-based summarization after configured number of turns:

```python
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig

app = App(
    name='hello_world_app',
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,
        overlap_size=1,
    ),
)
```

Memory generation = broad capability of extracting persistent knowledge from verbose/noisy data. Section covered primary example: session compaction. Compaction distills verbatim transcript, extracting key facts/summaries while discarding conversational filler.

## Memory

Memory and Sessions share **deeply symbiotic** relationship: sessions = primary data source for generating memories; memories = key strategy for managing session size. Memory = **snapshot of extracted, meaningful info** from conversation/data source. Condensed representation preserving important context. Generally persisted across sessions for continuous personalized experience.

**Memory manager** = specialized, decoupled service providing foundation for multi-agent interoperability. Frequently use framework-agnostic data structures (simple strings, dictionaries) → agents on different frameworks connect to single memory store, shared knowledge base.

> Some frameworks may also refer to Sessions or verbatim conversation as "short-term memory." For this whitepaper, memories = extracted info, **not raw dialogue**.

### Capabilities a robust memory system unlocks

- **Personalization**: most common — remember user preferences, facts, past interactions to tailor future responses (favorite sports team, preferred airplane seat)
- **Context Window Management**: compact full history by summaries/key facts → preserve context without sending thousands of tokens
- **Data Mining and Insight**: aggregate stored memories across users (privacy-preserving) → extract insights from noise. E.g. retail chatbot identifies many users asking about return policy for specific product → flag potential issue.
- **Agent Self-Improvement and Adaptation**: agent learns from previous runs by creating procedural memories about own performance — which strategies/tools/reasoning paths led to successful outcomes. Builds playbook of effective solutions.

### Collaborative process of creating, storing, utilizing memory

Each component has distinct role:
1. **The User**: provides raw source data. Some systems users provide memories directly (form).
2. **The Agent (Developer Logic)**: configures how to decide what and when to remember. Orchestrates calls to memory manager. Simple architectures: developer implements logic such that memory is *always* retrieved and *always* triggered-to-be-generated. More advanced: developer implements **memory-as-a-tool** where agent (via LLM) decides when memory should be retrieved or generated.
3. **The Agent Framework (e.g. ADK, LangGraph)**: structure and tools for memory interaction. Plumbing. Defines how developer's logic accesses conversation history and interacts with memory manager. Doesn't manage long-term storage itself. Defines how to stuff retrieved memories into context window.
4. **The Session Storage** (e.g. Agent Engine Sessions, Spanner, Redis): stores turn-by-turn conversation. Raw dialogue ingested into memory manager to generate memories.
5. **The Memory Manager** (e.g. Agent Engine Memory Bank, Mem0, Zep): handles storage, retrieval, compaction. Mechanisms depend on provider. Specialized service/component taking potential memory and handling entire lifecycle:
   - **Extraction**: distills key info from source data
   - **Consolidation**: curates memories to merge duplicative entities
   - **Storage**: persists memory to durable databases
   - **Retrieval**: fetches relevant memories for context

> **Figure 5**: The flow of information between sessions, memory, and external knowledge
> - External Knowledge Bases (RAG databases) ← Write external to agent
> - Agent contains: LLM ↔ Tools, LLM Response, Tool Response
> - Agent ← Read Before agent execution (External, RAG dynamic as-a-tool)
> - Memory Manager → Read Memories (Static at start of each turn; Dynamic memory-as-a-tool)
> - Memory Manager → Create Memories (External, Internal "memory-as-a-tool")
> - Session Store ← Write Events; Source data for Memory Generation
> - User ↔ Agent

Division of responsibilities ensures developer focuses on agent's unique logic. Memory manager = **active system, not just a passive vector database**. Uses similarity search for retrieval but core value lies in intelligently extracting, consolidating, curating memories over time. Managed services like Agent Engine Memory Bank handle entire lifecycle.

### Memory vs RAG

Frequently compared but built on different principles. RAG handles static, external data; Memory curates dynamic, user-specific context.

**RAG = expert on facts; memory = expert on the user.**

| | RAG Engines | Memory Managers |
|---|---|---|
| Primary Goal | Inject **external, factual** knowledge into context | Personalized, stateful experience. Agent remembers facts, adapts to user over time, maintains long-running context. |
| Data source | Static, pre-indexed external knowledge base (PDFs, wikis, documents, APIs) | Dialogue between user and agent |
| Isolation Level | **Generally Shared**. Knowledge base = global, read-only resource accessible by all users for consistent factual answers | **Highly Isolated**: Almost always scoped per-user to prevent data leaks |
| Information type | Static, factual, authoritative. Domain-specific data, product details, technical docs | Dynamic, user-specific. Memories derived from conversation → inherent uncertainty |
| Write patterns | Batch processing. Triggered via offline administrative action. | Event-based processing. Triggered at some cadence (every turn, end of session) or memory-as-a-tool. |
| Read patterns | RAG data almost always retrieved "as-a-tool". Retrieved when agent decides query requires external info. | Two common patterns: **Memory-as-a-tool** (retrieved when query requires user info), **Static retrieval** (always retrieved at start of each turn) |
| Data Format | Natural-language "chunk" | Natural language snippet or structured profile |
| Data preparation | **Chunking and Indexing**: source documents broken into smaller chunks, converted to embeddings, stored for fast lookup | **Extraction and consolidation**: extract key details from conversation, ensuring not duplicative or contradictory |

**Research librarian (RAG) vs personal assistant (memory) analogy**:
- Research librarian works in vast public library of encyclopedias, textbooks, official documents. When agent needs established fact (product specs, historical date), consults librarian. Static, shared, authoritative knowledge base. Expert on world's facts, doesn't know anything personal about the user.
- Personal assistant follows agent, carries private notebook recording details of every interaction with specific user. Notebook = dynamic, highly isolated; personal preferences, past conversations, evolving goals. Expertise: not in global facts, but in the user themselves.

Truly intelligent agent needs **both**.

### Types of memory

Categorized by how info is stored and how captured. Across all types: **memories are descriptive, not predictive**.

A "memory" = atomic piece of context returned by memory manager. Single memory generally consists of:
- **Content**: substance extracted from source data (raw dialogue). Designed framework-agnostic, simple data structures. **Structured memories** = info typically stored in universal formats like dictionary/JSON. Schema typically defined by developer, not framework. E.g. `{"seat_preference": "Window"}`. **Unstructured memories** = natural language descriptions capturing essence of longer interaction/event. E.g. "The user prefers a window seat."
- **Metadata**: context about the memory, typically simple string. Includes unique identifier, "owner" identifiers, labels describing content/data source.

#### Types of information

Memories classified by fundamental type of knowledge they represent. From cognitive science:
- **Declarative memory** ("knowing what"): agent's knowledge of facts, figures, events. Info agent can explicitly state or "declare." Answers "what" question. Encompasses general world knowledge (Semantic) + specific user facts (Entity/Episodic).
- **Procedural memory** ("knowing how"): agent's knowledge of skills/workflows. Demonstrates implicitly how to perform task correctly. Answers "how" question (correct sequence of tool calls to book a trip).

#### Organization patterns

How individual memories relate to each other and to user. Memory managers employ one or more of:

- **Collections**: organize content into multiple self-contained, natural language memories for a single user. Each memory = distinct event, summary, observation. Multiple memories for single high-level topic. Allows storing/searching through larger less structured pool.
- **Structured User Profile**: memories as set of core facts about user, like contact card continuously updated with stable info. Designed for quick lookups of essential factual info (names, preferences, account details).
- **"Rolling" Summary**: consolidates all info into single evolving memory representing natural-language summary of entire user-agent relationship. Instead of new individual memories, manager continuously updates this one master document. Frequently used to compact long Sessions, preserving vital info while managing token count.

#### Storage architectures

Critical decision determining how quickly and intelligently agent retrieves memories. Choice defines whether agent excels at finding conceptually similar ideas, understanding structured relationships, or both.

Generally stored in **vector databases** and/or **knowledge graphs**.

- **Vector databases**: most common. Retrieval based on semantic similarity rather than exact keywords. Memories converted to embedding vectors, database finds closest conceptual matches. Excels at retrieving unstructured natural language memories where context/meaning are key (atomic facts).
- **Knowledge graphs**: store memories as network of entities (nodes) and relationships (edges). Retrieval involves traversing graph for direct/indirect connections, allowing reasoning about how facts are linked. Ideal for structured, relational queries (knowledge triples).
- **Hybrid approach**: combine. Enrich knowledge graph's structured entities with vector embeddings. Both relational and semantic searches simultaneously. Structured reasoning of graph + nuanced conceptual search of vectors.

#### Creation mechanisms

Classify memories by how created:
- **Explicit memories**: created when user gives direct command to remember (e.g. "Remember my anniversary is October 26th")
- **Implicit memories**: created when agent infers and extracts info from conversation without direct command (e.g. "My anniversary is next week. Can you help me find a gift for my partner?")

By location:
- **Internal memory**: memory management built directly into agent framework. Convenient for getting started but often lacks advanced features. Can use external storage but generation mechanism is internal.
- **External Memory**: separate specialized service dedicated to memory management (e.g. **Agent Engine Memory Bank, Mem0, Zep**). Framework makes API calls. More sophisticated features: semantic search, entity extraction, automatic summarization. Offloads complex task to purpose-built tool.

#### Memory scope

Who or what a memory describes. Implications on entity (user, session, application) used to aggregate and retrieve memories.

- **User-Level scope**: most common. Continuous personalized experience for individual ("the User prefers the middle seat."). Tied to specific user ID, persist across all sessions.
- **Session-Level scope**: designed for compaction of long conversations ("the User is shopping for tickets between New York and Paris between November 7, 2025 and November 14, 2025. They prefer direct flights and the middle seat"). Persistent record of insights extracted from single session, allowing replacement of verbose token-heavy transcript with concise key facts. **Distinct from raw session log**; only processed insights from dialogue, isolated to specific session.
- **Application-level scope** (or global context): accessible by all users of an application ("The codename XYZ refers to the project..."). Provides shared context, broadcast system-wide info, baseline common knowledge. Common use case: **procedural memories** providing "how-to" instructions for agent's reasoning across all users. **Critical to sanitize all sensitive content** to prevent data leaks.

#### Multimodal memory

Crucial concept describing how agent handles non-textual info (images, videos, audio). Distinguish between data the memory is **derived from** (its source) and the data the memory is **stored as** (its content).

- **Memory from a multimodal source**: most common. Agent processes various data types — text, images, audio — but memory it creates is a **textual insight** derived from source. Doesn't store audio file; transcribes audio and creates textual memory like "User expressed frustration about the recent shipping delay."
- **Memory with Multimodal Content**: more advanced. Memory itself contains non-textual media. Agent doesn't just describe content; stores content directly. E.g. user uploads image and says "Remember this design for our logo." Agent creates memory directly containing image file linked to user's request.

Most contemporary memory managers focus on multimodal sources while producing textual content. Generating/retrieving unstructured binary data (images, audio) for specific memory requires specialized models, algorithms, infrastructure. Far simpler to convert all inputs into common, searchable format: text.

Generate memories from multimodal input using Agent Engine Memory Bank — output memories will be textual insights extracted from content:

```python
from google.genai import types

client = vertexai.Client(project=..., location=...)
response = client.agent_engines.memories.generate(
        name=agent_engine_name,
        direct_contents_source={
            "events": [
              {
                "content": types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            "This is context about the multimodal input."
                        ),
                        types.Part.from_bytes(
                            data=CONTENT_AS_BYTES,
                            mime_type=MIME_TYPE
                        ),
                        types.Part.from_uri(
                            file_uri="file/path/to/content",
                            mime_type=MIME_TYPE
                        )
                    ])}]},
        scope={"user_id": user_id}
)
```

## Memory Generation: Extraction and Consolidation

Memory generation autonomously transforms raw conversational data into structured, meaningful insights. **LLM-driven ETL (Extract, Transform, Load) pipeline** designed to extract and condense memories. Distinguishes memory managers from RAG engines and traditional databases.

Rather than requiring developers to manually specify DB operations, memory manager uses LLM to intelligently decide when to add, update, or merge memories. Automation = core strength.

> **Figure 6**: High-level algorithm of memory generation
> - Data Source (Session Events) → Memory Extraction (LLM) → Raw Memory ×3
> - Raw Memory → Retrieve similar existing memories ↔ Memory Storage
> - Similar Memory ×2 → Memory Consolidation (LLM)
> - Consolidated Memory with Operation (e.g. ADD), Consolidated Memory with Operation (e.g. UPDATE) → Memory Storage

While specifics vary by platform (Agent Engine Memory Bank, Mem0, Zep), high-level process generally follows four stages:

1. **Ingestion**: client provides source of raw data (typically conversation history) to memory manager.
2. **Extraction & Filtering**: memory manager uses LLM to extract meaningful content matching predefined **topic definition**. Doesn't extract everything; only what fits topics. If no matching info → no memory created.
3. **Consolidation**: most sophisticated stage. Conflict resolution and deduplication. "Self-editing" process using LLM to compare newly extracted info with existing memories. Decides:
   - **Merge** new insight into existing memory
   - **Delete** existing memory if invalidated
   - **Create** entirely new memory if topic novel
4. **Storage**: new/updated memory persisted to durable storage (vector DB or knowledge graph).

Managed memory manager (Agent Engine Memory Bank) fully automates pipeline:

```python
from google.cloud import vertexai

client = vertexai.Client(project=..., location=...)

client.agent_engines.memories.generate(
    name="projects/.../locations/...reasoningEngines/...",
    scope={"user_id": "123"},
    direct_contents_source={
        "events": [...]
    },
    config={
        # Run memory generation in the background.
        "wait_for_completion": False
    }
)
```

**Gardener analogy**: Extraction = receiving new seeds and saplings. Gardener doesn't throw them randomly. Performs Consolidation by pulling weeds (deleting redundant/conflicting data), pruning back overgrown branches (refining/summarizing existing memories), then carefully planting new saplings in optimal location. Constant thoughtful curation ensures garden remains healthy, organized, continues to flourish. Asynchronous process happens in background, ensuring garden always ready for next visit.

### Deep-dive: Memory Extraction

Goal: answer fundamental question — **"What information in this conversation is meaningful enough to become a memory?"** Not simple summarization. Targeted, intelligent filtering: separate signal (facts, preferences, goals) from noise (pleasantries, filler).

"Meaningful" = defined entirely by agent's purpose/use case. Customer support agent (order numbers, technical issues) ≠ personal wellness coach (long-term goals, emotional states). Customizing what info preserved = key.

Memory manager's LLM decides what to extract via carefully constructed programmatic guardrails/instructions in complex system prompt. Defines "meaningful" by providing LLM with set of **topic definitions**.

- **Schema and template-based extraction**: LLM given predefined JSON schema or template using **structured output** features. LLM instructed to construct JSON using corresponding info in conversation.
- **Natural language topic definitions**: LLM guided by simple natural language description of topic.
- **Few-shot prompting**: LLM "shown" what info to extract using examples. Prompt includes input text + ideal high-fidelity memory. LLM learns desired extraction pattern from examples. Highly effective for custom/nuanced topics difficult to describe with schema/definition.

Memory managers work out-of-box looking for common topics (user preferences, key facts, goals). Many platforms allow custom topics for specific domain.

Example — customize Agent Engine Memory Bank topic + few-shot:

```python
from google.genai.types import Content, Part

memory_bank_config = {
  "customization_configs": [{
    "memory_topics": [
      { "managed_memory_topic": { "managed_topic_enum": "USER_PERSONAL_INFO" }},
      {
        "custom_memory_topic": {
          "label": "business_feedback",
          "description": """Specific user feedback about their experience at the coffee
shop. This includes opinions on drinks, food, pastries, ambiance, staff friendliness,
service speed, cleanliness, and any suggestions for improvement."""
        }
      }
    ],
    "generate_memories_examples": {
      "conversationSource": {
        "events": [
          {
            "content": Content(
                  role="model",
                  parts=[Part(text="Welcome back to The Daily Grind! We'd love to hear
your feedback on your visit.")])
          }, {
            "content": Content(
                  role="user",
                  parts=[Part(text= "Hey. The drip coffee was a bit lukewarm today, which
was a bummer. Also, the music was way too loud, I could barely hear my friend.")])
          }]
      },
      "generatedMemories": [
        {"fact": "The user reported that the drip coffee was lukewarm."},
        {"fact": "The user felt the music in the shop was too loud."}
      ]
    }
  }]
}

agent_engine = client.agent_engines.create(
      config={
            "context_spec": {"memory_bank_config": memory_bank_config }
      }
)
```

Although memory extraction itself isn't "summarization," algorithm may incorporate summarization to distill info. Many memory managers incorporate **rolling summary** of conversation directly into memory extraction prompt. Condensed history provides necessary context to extract key info from most recent interactions. Eliminates repeatedly processing full verbose dialogue.

### Deep-dive: Memory Consolidation

After extraction, **consolidation** integrates new info into coherent, accurate, evolving knowledge base. Most sophisticated stage. Transforms simple collection of facts into curated understanding. Without consolidation, memory becomes noisy, contradictory, unreliable log. "Self-curation" typically managed by LLM — elevates memory manager beyond simple database.

Addresses fundamental problems from conversational data:
- **Information Duplication**: user mentions same fact in multiple ways (e.g. "I need a flight to NYC" and later "I'm planning a trip to New York"). Simple extraction = two redundant memories.
- **Conflicting Information**: user's state changes over time. Without consolidation, contradictory facts.
- **Information Evolution**: simple fact becomes more nuanced. Initial memory "user is interested in marketing" might evolve into "the user is leading a marketing project focused on Q4 customer acquisition."
- **Memory Relevance Decay**: not all memories useful forever. Agent must engage in **forgetting** — proactively pruning old, stale, low-confidence memories. Forgetting via instructing LLM to defer to newer info during consolidation, or automatic deletion via TTL.

Consolidation = LLM-driven workflow comparing newly extracted insights against existing memories. Workflow:
1. Try to retrieve existing memories similar to newly extracted.
2. LLM presented with both *existing* and *new info*. Analyzes together, identifies what operations should be performed:
   - **UPDATE**: modify existing memory with new/corrected info
   - **CREATE**: if entirely novel/unrelated, create new
   - **DELETE / INVALIDATE**: if new info makes old memory irrelevant/incorrect
3. Memory manager translates LLM's decision into transaction updating memory store.

### Memory Provenance

Classic ML axiom "garbage in, garbage out" even more critical for LLMs — outcome often "garbage in, **confident** garbage out." Agent must critically evaluate quality of own memories. Trustworthiness derived directly from memory's **provenance** — detailed record of origin and history.

> **Figure 7**: Flow between data sources and memories. Single memory derived from multiple data sources, single source contributes to multiple memories.
> - Request 1 (data source: A) → Memory 1 (Memory Revision 1, 2, 3)
> - Request 2 (data source: B) → Memory 1 (Revision 2)
> - Request 3 (data source: C) → Memory 1 (Revision 3)
> - Request 4 (data source: D) → Memory 2 (Revisions 1, 2)

Process of memory consolidation — merging info from multiple sources into single evolving memory — creates need to track lineage. To assess trustworthiness, agent must track key details for each source: **origin (source type) and age ("freshness")**. Dictate weight each source has during consolidation, inform how much agent should rely on memory during inference.

Source type categories:
- **Bootstrapped Data**: pre-loaded from internal systems (e.g. CRM). High-trust data. Initialize user's memories to address **cold-start problem** (challenge of providing personalized experience to user agent never interacted with).
- **User Input**: explicitly via form (high-trust) or implicitly extracted from conversation (less trustworthy).
- **Tool Output**: data returned from external tool call. Generating memories from Tool Output generally **discouraged** — tend to be brittle and stale, better suited for short-term caching.

#### Accounting for memory lineage during memory management

Two primary operational challenges: **conflict resolution** + **deleting derived data**.

Memory consolidation inevitably leads to conflicts where one source contradicts another. Provenance allows hierarchy of trust. Conflict strategies:
- Prioritize most trusted source
- Favor most recent info
- Look for corroboration across multiple data points

Deleting memories: memory derived from multiple sources. When user revokes access to one source, data from that source should be removed. Deleting every memory "touched" can be overly aggressive. More precise (but computationally expensive): regenerate affected memories from scratch using only remaining valid sources.

Beyond static provenance, confidence in memory must evolve. Increases through corroboration (multiple trusted sources provide consistent info). Efficient memory system actively curates through **memory pruning** — identifies and "forgets" no-longer-useful memories. Triggers:
- **Time-based Decay**: importance decreases over time. Memory about meeting two years ago likely less relevant than one from last week.
- **Low Confidence**: created from weak inference and never corroborated.
- **Irrelevance**: as agent gains sophisticated understanding of user, some older trivial memories no longer relevant to current goals.

Combining reactive consolidation pipeline + proactive pruning ensures knowledge base = curated understanding, not growing log.

#### Accounting for memory lineage during inference

Memory's trustworthiness should also be considered at inference time. Agent's confidence in memory should not be static; must evolve based on new info and passage of time. Confidence increases through corroboration. Decreases (decays) over time as older memories become stale, drops when contradictory info introduced. Eventually system can "forget" by archiving/deleting low-confidence memories. **Dynamic confidence score** critical during inference. Rather than shown to user, memories and confidence scores **injected into the prompt**, enabling LLM to assess reliability and make nuanced decisions.

Trust framework serves agent's internal reasoning process. Memories + confidence scores typically not shown to user directly. Injected into system prompt → LLM weighs evidence, considers reliability, makes more nuanced trustworthy decisions.

### Triggering memory generation

Although memory managers automate extraction/consolidation once generation triggered, agent must decide **when** to attempt. Critical architectural choice balancing data freshness against computational cost/latency. Typically managed by agent's logic:

- **Session Completion**: triggering at end of multi-turn session
- **Turn Cadence**: after specific number of turns (every 5)
- **Real-Time**: after every single turn
- **Explicit Command**: direct user command ("Remember this")

**Cost vs fidelity tradeoff**:
- **Frequent generation** (real-time): highly detailed and fresh, captures every nuance. Highest LLM/database costs, latency if not handled properly.
- **Infrequent generation** (session completion): far more cost-effective. Risks lower-fidelity memories — LLM must summarize larger block at once. Avoid processing same events multiple times → unnecessary cost.

#### Memory-as-a-Tool

More sophisticated approach — agent decides for itself when to create memory. Memory generation exposed as tool (e.g. `create_memory`). Tool definition defines what types meaningful. Agent analyzes conversation and autonomously calls tool when identifying meaningful info. Shifts responsibility for identifying "meaningful info" from external memory manager to agent (and developer) itself.

Using ADK by packaging memory generation code into a `Tool`. Send Session to Memory Bank → extract/consolidate from history:

```python
from google.adk.agents import LlmAgent
from google.adk.memory import VertexAiMemoryBankService
from google.adk.runners import Runner
from google.adk.tools import ToolContext

def generate_memories(tool_context: ToolContext):
    """Triggers memory generation to remember the session."""
    # Option 1: Extract memories from the complete conversation history using the
    # ADK memory service.
    tool_context._invocation_context.memory_service.add_session_to_memory(
        session)

    # Option 2: Extract memories from the last conversation turn.
    client.agent_engines.memories.generate(
        name="projects/.../locations/...reasoningEngines/...",
        direct_contents_source={
            "events": [
                {"content": tool_context._invocation_context.user_content}
            ]
        },
        scope={
            "user_id": tool_context._invocation_context.user_id,
            "app_name": tool_context._invocation_context.app_name
        },
        # Generate memories in the background
        config={"wait_for_completion": False}
    )
    return {"status": "success"}

agent = LlmAgent(
    ...,
    tools=[generate_memories]
)

runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=VertexAiMemoryBankService(
      agent_engine_id=AGENT_ENGINE_ID,
      project=PROJECT,
      location=LOCATION
    )
)
```

Alternative — internal memory where agent actively decides what to remember from conversation. Agent responsible for extracting key info. Optionally these extracted memories sent to Agent Engine Memory Bank for consolidation:

```python
def extract_memories(query: str, tool_context: ToolContext):
    """Triggers memory generation to remember information.

    Args:
      query: Meaningful information that should be persisted about the user.
    """
    client.agent_engines.memories.generate(
        name="projects/.../locations/...reasoningEngines/...",
        # The meaningful information is already extracted from the conversation, so we
        # just want to consolidate it with existing memories for the same user.
        direct_memories_source={
            "direct_memories": [{"fact": query}]
        },
        scope={
            "user_id": tool_context._invocation_context.user_id,
            "app_name": tool_context._invocation_context.app_name
        },
        config={"wait_for_completion": False}
    )
    return {"status": "success"}

agent = LlmAgent(
    ...,
    tools=[extract_memories]
)
```

#### Background vs. Blocking Operations

Memory generation = expensive (LLM calls + DB writes). Production agents: should almost always be handled **asynchronously as background process**.

After agent sends response to user, memory generation pipeline can run in parallel without blocking UX. Decoupling essential for keeping agent fast/responsive. Blocking (synchronous) approach where user waits for memory write before receiving response = unacceptably slow/frustrating. Memory generation occurs in service architecturally separate from agent's core runtime.

## Memory Retrieval

With generation in place, focus shifts to **retrieval**. Intelligent retrieval strategy essential for performance: which memories should be retrieved and when.

Strategy depends heavily on how memories organized. **Structured user profile** = straightforward lookup for full profile or specific attribute. **Collection of memories** = far more complex search problem. Goal: discover most pertinent, conceptually related info from large pool of unstructured/semi-structured data.

Memory retrieval searches for most pertinent memories for current conversation. Critical: providing irrelevant memories confuses model and degrades response. Finding perfect piece = remarkably intelligent interaction. Core challenge: balancing memory 'usefulness' within strict latency budget.

Advanced systems go beyond simple search and score across multiple dimensions:
- **Relevance (Semantic Similarity)**: how conceptually related to current conversation?
- **Recency (Time-based)**: how recently created?
- **Importance (Significance)**: how critical overall? Defined at generation-time.

Relying solely on vector-based relevance = common pitfall. Similarity scores can surface conceptually similar but old/trivial memories. Most effective: **blended approach** combining scores from all three dimensions.

For accuracy-critical applications:
- **Query rewriting**: LLM improves search query itself. Rewriting ambiguous input into more precise query, or **expanding** single query into multiple related ones to capture facets. Significantly improves quality but adds latency of extra LLM call.
- **Reranking**: initial retrieval fetches broad set of candidate memories (e.g. top 50) using similarity search. LLM re-evaluates and re-ranks smaller set to produce more accurate final list.
- **Specialized retriever**: trained using fine-tuning. Requires labeled data; significantly increases costs.

Best approach to retrieval starts with **better memory generation**. High-quality corpus free of irrelevant info = most effective way to guarantee retrieved memories will be helpful.

### Timing for retrieval

Final architectural decision: **when** to retrieve memories.

- **Proactive retrieval**: memories automatically loaded at start of every turn. Ensures context always available but introduces unnecessary latency for turns not requiring memory access. Memories static throughout turn → can be cached to mitigate performance cost.

Implement proactive retrieval in ADK using built-in `PreloadMemoryTool` or custom callback:

```python
# Option 1: Use the built-in PreloadMemoryTool which retrieves memories with
# similarity search every turn.
agent = LlmAgent(
    ...,
    tools=[adk.tools.preload_memory_tool.PreloadMemoryTool()]
)

# Option 2: Use a custom callback to have more control over how memories
# are retrieved.
def retrieve_memories_callback(callback_context, llm_request):
    user_id = callback_context._invocation_context.user_id
    app_name = callback_context._invocation_context.app_name

    response = client.agent_engines.memories.retrieve(
        name="projects/.../locations/...reasoningEngines/...",
        scope={
            "user_id": user_id,
            "app_name": app_name
        }
    )
    memories = [f"* {memory.memory.fact}" for memory in list(response)]
    if not memories:
        # No memories to add to System Instructions.
        return
    # Append formatted memories to the System Instructions
    llm_request.config.system_instruction += "\nHere is information that you have
about the user:\n"
    llm_request.config.system_instruction += "\n".join(memories)

agent = LlmAgent(
    ...,
    before_model_callback=retrieve_memories_callback,
)
```

- **Reactive retrieval ("Memory-as-a-Tool")**: agent given tool to query memory, deciding for itself when to retrieve context. More efficient and robust but requires additional LLM call → increases latency/cost. However memory retrieved only when necessary → latency cost incurred less frequently. Agent may not know if relevant info exists; mitigate by making agent aware of memory types available (in tool description).

```python
# Option 1: Use the built-in LoadMemory.
agent = LlmAgent(
    ...,
    tools=[adk.tools.load_memory_tool.LoadMemoryTool()],
)

# Option 2: Use a Custom tool where you can describe what type of information
# might be available.
def load_memory(query: str, tool_context: ToolContext):
    """Retrieves memories for the user.

    The following types of information may be stored for the user:
    * User preferences, like the user's favorite foods.
    ...
    """
    # Retrieve memories using similarity search.
    response = tool_context.search_memory(query)
    return response.memories

agent = LlmAgent(
    ...,
    tools=[load_memory],
)
```

## Inference with Memories

Once retrieved, final step: strategically place memories into model's context window. Critical — placement significantly influences reasoning, costs, final answer quality.

Primarily presented by **appending to system instructions or injecting into conversation history**. Hybrid strategy often most effective:
- Use **system prompt** for stable, global memories (user profile) that should always be present
- Use **dialogue injection** or **memory-as-a-tool** for transient, episodic memories only relevant to immediate context

### Memories in the System Instructions

Append memories to system instructions. Keeps conversation history clean by appending retrieved memories directly to system prompt alongside preamble, framing as foundational context for entire interaction.

Use Jinja to dynamically add memories to system instructions:

```python
from jinja2 import Template

template = Template("""
{{ system_instructions }}}

<MEMORIES>
Here is some information about the user:
{% for retrieved_memory in data %}* {{ retrieved_memory.memory.fact }}
{% endfor %}</MEMORIES>
""")

prompt = template.render(
    system_instructions=system_instructions,
    data=retrieved_memories
)
```

Including in system instructions = high authority, cleanly separates context from dialogue, ideal for stable "global" info like user profile. **Risk of over-influence** — agent might try to relate every topic back to memories in core instructions even when inappropriate.

Constraints: requires framework to support dynamic construction of system prompt before each LLM call (not always supported). Pattern incompatible with **Memory-as-a-Tool** (system prompt must be finalized before LLM can decide to call memory retrieval tool). Poorly handles non-textual memories — most LLMs only accept text for system instructions.

### Memories in the Conversation History

Retrieved memories injected directly into turn-by-turn dialogue. Placed before full conversation history or right before latest user query.

Method can be **noisy**, increasing token costs and potentially confusing model if retrieved memories irrelevant. Primary risk: **dialogue injection** — model might mistakenly treat memory as something actually said. Be careful about perspective of memories injected; if using "user" role and user-level memories, written in first-person.

Special case of injecting memories into conversation history: retrieving via **tool calls**. Memories included directly as part of tool output:

```python
def load_memory(query: str, tool_context: ToolContext):
    """Loads memories into the conversation history..."""
    response = tool_context.search_memory(query)
    return response.memories

agent = LlmAgent(
    ...,
    tools=[load_memory],
)
```

## Procedural memories

Whitepaper focused primarily on **declarative memories** (mirrors current commercial landscape). Most platforms architected for declarative — extracting/storing/retrieving "what" (facts, history, user data).

Not designed to manage **procedural memories** — mechanism for improving agent's workflows and reasoning. Storing the "how" = not info retrieval problem; **reasoning augmentation** problem. Managing "knowing how" requires completely separate specialized algorithmic lifecycle, similar high-level structure:

1. **Extraction**: requires specialized prompts to distill reusable strategy or "playbook" from successful interaction, rather than capturing fact.
2. **Consolidation**: while declarative consolidation merges related facts (the "what"), procedural consolidation curates the workflow itself (the "how"). Active logic management — integrating new successful methods with existing "best practices," patching flawed steps in known plan, pruning outdated/ineffective procedures.
3. **Retrieval**: not retrieving data to answer question; retrieving plan that guides agent on how to execute complex task. May have different data schema than declarative.

Capacity for agent to **'self-evolve' its logic** invites comparison to fine-tuning (often via RLHF). While both improve agent behavior, fundamentally different:
- Fine-tuning = relatively slow, offline training that alters model weights
- Procedural memory = fast, online adaptation by dynamically injecting "playbook" into prompt, guiding agent via **in-context learning** without fine-tuning

## Testing and Evaluation

Validate memory-enabled agent via comprehensive quality and evaluation tests. Multi-layered:
- **Quality**: agent remembering right things
- **Retrieval**: can find memories when needed
- **Task success**: using memories actually helps accomplish goals

While academia focuses on reproducible benchmarks, industry evaluation centered on impact to performance/usability of production agent.

**Memory generation quality** metrics evaluate content of memories. Question: **"Is the agent remembering the right things?"** Compare agent's generated memories against manually created "golden set" of ideal memories.
- **Precision**: of memories agent created, what % accurate and relevant? High precision guards against "over-eager" memory system polluting knowledge base with irrelevant noise.
- **Recall**: of relevant facts it should have remembered from source, what % captured? High recall ensures agent doesn't miss critical info.
- **F1-Score**: harmonic mean of precision and recall.

**Memory retrieval performance** metrics evaluate ability to find right memory at right time.
- **Recall@K**: when memory needed, is correct one found within top 'K' retrieved results? Primary measure of retrieval system's accuracy.
- **Latency**: retrieval on "hot path" of agent's response. Entire process must execute within strict latency budget (e.g. under 200ms) to avoid degrading UX.

**End-to-End task success** = ultimate test. Does memory actually help agent perform job better? Measured by evaluating performance on downstream tasks using memory, often with LLM "judge" comparing final output to golden answer.

Evaluation = engine for continuous improvement. Iterative process: establishing baseline, analyzing failures, tuning system (refining prompts, adjusting retrieval algorithms), re-evaluating impact.

While metrics focus on quality, production-readiness depends on performance. Critical to measure latency of underlying algorithms and ability to scale under load. Retrieval "on the hot-path" = strict, sub-second budget. Generation/consolidation, while async, must have throughput keeping up with user demand.

## Production considerations for Memory

Beyond performance, transitioning memory-enabled agent from prototype to production demands focus on enterprise-grade architecture: scalability, resilience, security.

To ensure UX never blocked by computationally expensive memory generation, decouple memory processing from main app logic. Event-driven pattern, typically implemented via direct, **non-blocking API calls** to dedicated memory service rather than self-managed message queue. Flow:

1. **Agent pushes data**: after relevant event (session ends), agent makes non-blocking API call to memory manager, "pushing" raw source data (conversation transcript) to be processed.
2. **Memory manager processes in background**: service immediately acknowledges request, places generation task into internal managed queue. Solely responsible for async heavy lifting: LLM calls to extract, consolidate, format memories. May delay processing events until certain period of inactivity elapses.
3. **Memories are persisted**: service writes final memories — new entries or updates — to dedicated durable database. Managed memory managers: storage built-in.
4. **Agent retrieves memories**: main agent application queries memory store directly when needs context for new user interaction.

Service-based, non-blocking approach ensures failures/latency in memory pipeline don't impact user-facing application → far more resilient. Informs choice between:
- **Online (real-time) generation**: ideal for conversational freshness
- **Offline (batch) processing**: useful for populating system from historical data

As application grows, memory system must handle high-frequency events without failure. Given **concurrent** requests, prevent deadlocks/race conditions when multiple events try to modify same memory. Mitigate via transactional database operations or optimistic locking; can introduce **queuing** or **throttling**. Robust message queue essential to buffer high volumes.

Resilient to transient errors (**failure handling**). If LLM call fails: retry mechanism with exponential backoff, route persistent failures to dead-letter queue.

For global applications, memory manager must use database with built-in **multi-region replication** for low latency and high availability. Client-side replication not feasible — consolidation requires single transactionally consistent view to prevent conflicts. System must handle replication internally, presenting single logical datastore while ensuring underlying knowledge base globally consistent.

Managed memory systems (Agent Engine Memory Bank) help address these production considerations.

### Privacy and security risks

Memories derived from + include user data → require stringent privacy and security controls. Useful analogy: system's memory = secure corporate archive managed by professional archivist, whose job is to preserve valuable knowledge while protecting the company.

Cardinal rule: **data isolation**. Just as archivist would never mix confidential files from different departments, memory must be strictly isolated at user/tenant level. Agent serving one user must never have access to another's memories, enforced using restrictive **Access Control Lists (ACLs)**. Users must have programmatic control over data, with clear options to opt-out of memory generation or request deletion.

Before filing, archivist performs critical security steps:
- Meticulously goes through each page to **redact sensitive PII**, ensuring knowledge saved without creating liability
- Trained to spot and discard forgeries or intentionally misleading documents — safeguard against **memory poisoning**

Same way: system must validate/sanitize info before committing to long-term memory to prevent malicious user from corrupting agent's persistent knowledge through prompt injection. Include safeguards like **Model Armor** to validate/sanitize before committing.

**Exfiltration risk** if multiple users share memory set (procedural memories teaching agent how to do something). E.g. procedural memory from one user used as example for another (sharing a memo company-wide) → archivist must perform rigorous **anonymization** to prevent sensitive info leaking across user boundaries.

## Conclusion

Whitepaper explored discipline of **Context Engineering**, focusing on two central components: **Sessions** and **Memory**. Journey from simple conversational turn to piece of persistent actionable intelligence governed by this practice — dynamically assembling all necessary info (conversation history, memories, external knowledge) into LLM's context window. Process relies on interplay between two distinct but interconnected systems: immediate Session and long-term Memory.

**Session** governs the "now" — low-latency chronological container for single conversation. Primary challenge: performance and security, requiring **low-latency access** and **strict isolation**. To prevent context window overflow and latency, use **extraction techniques** like token-based truncation or recursive summarization to **compact** content within Session's history or single request payload. Security paramount, mandating **PII redaction** before session data persisted.

**Memory** = engine of long-term personalization, core mechanism for persistence across multiple sessions. Moves beyond RAG (makes agent expert on facts) to make agent **expert on the user**. Active LLM-driven ETL pipeline — responsible for **extraction, consolidation, retrieval** — distilling most important info from conversation. With **extraction**, system distills most critical info into key memory points. **Consolidation** curates and integrates new info with existing corpus, resolving conflicts, deleting redundant data → coherent knowledge base. To maintain snappy UX, memory generation must run as **asynchronous background process** after agent has responded. Tracking **provenance** and employing safeguards against risks like memory poisoning → developers can build trusted, adaptive assistants that truly learn and grow with the user.

## Key Takeaways

- **Context Engineering** = dynamic assembly of system prompts, memories, RAG data, conversation history, tool outputs into LLM context window per turn
- Evolution from static prompt engineering to dynamic state-aware context construction
- **Session** = single conversation container (events + state); ADK uses explicit `Session`, LangGraph uses mutable state
- Multi-agent session patterns: **shared unified history** (collaborative) vs **separate individual histories** (Agent-as-a-Tool, A2A protocol)
- Framework lock-in at persistence layer is real; **memory layer** (framework-agnostic) is more portable interop pattern than A2A messaging
- Long-context strategies: keep last N turns, token-based truncation, recursive summarization. Triggers: count-based, time-based, event-based
- **Memory** = extracted persistent insight (NOT raw dialogue). Components: content (structured/unstructured) + metadata
- Memory types: **declarative** ("what") vs **procedural** ("how"). Most platforms only handle declarative
- Memory organization: collections, structured user profile, rolling summary
- Storage: vector DB (semantic similarity), knowledge graph (relational), hybrid
- Memory ETL pipeline: ingestion → extraction (filter via topic definitions) → consolidation (UPDATE/CREATE/DELETE) → storage
- **Provenance** matters: bootstrapped > explicit user input > implicit conversation > tool output
- **Memory ≠ RAG**: RAG = expert on world's facts (static, shared); Memory = expert on the user (dynamic, isolated per-user)
- Triggering memory: session completion, turn cadence, real-time, explicit command, **memory-as-a-tool**
- Memory generation should always be **async background** to not block user experience
- Retrieval scoring: relevance + recency + importance (blended). Advanced: query rewriting, reranking
- Inference placement: **system instructions** (stable global) + **conversation history injection** (transient episodic)
- Production: PII redaction, ACL isolation, TTL, multi-region replication, dead-letter queues, Model Armor for memory poisoning protection
- Evaluation: precision/recall/F1 (generation quality), Recall@K + latency (retrieval), end-to-end task success
