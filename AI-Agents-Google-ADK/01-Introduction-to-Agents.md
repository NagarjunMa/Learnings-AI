# Introduction to Agents — Google White Paper Notes

Source: "Introduction to Agents" by Google (November 2025). First in five-part series. Authors: Alan Blount, Antonio Gulli, Shubham Saboo, Michael Zimmermann, Vladimir Vuskovic.

> Agents are the natural evolution of Language Models, made useful in software.

---

## From Predictive AI to Autonomous Agents

AI shifting from passive, discrete tasks (answer question, translate text, generate image) to autonomous problem-solving. Old paradigm requires constant human direction for every step.

### What Changed

- Old: Model predicts or creates content → human directs next step → repeat
- New: Agent combines LM's ability to **reason** with practical ability to **act**
- Critical capability: agents figure out next steps on their own, no human guiding every turn

### Developer Paradigm Shift

Traditional developer = "bricklayer" — defines every logical step in code. Agent developer = "director" — sets the scene (instructions/prompts), selects the cast (tools/APIs), provides context (data). Primary task becomes guiding autonomous "actor" to deliver intended performance.

### Context Engineering (formerly Prompt Engineering)

LM's greatest strength (flexibility) is also biggest headache. Hard to compel it to do *one specific thing* reliably. For each LM call, you fill context window with: instructions, facts, available tools, examples, session history, user profile. **Agents are software that manage inputs of LMs to get work done.**

### Agent Ops

Debugging agents requires monitoring "thought process" via traces and logs. Agent Ops = familiar cycle of measurement, analysis, system optimization — applied to non-deterministic systems. Comprehensive evaluations and assessments often outweigh initial prompt's influence.

### When an Agent Transcends "Workflow Automation"

When precisely configured with clear instructions, reliable tools, integrated context as memory, great UI, ability to plan and problem solve, and general world knowledge — agent becomes collaborative entity: efficient, uniquely adaptable, remarkably capable team member.

> *An agent is a system dedicated to context window curation. It is a relentless loop of assembling context, prompting the model, observing the result, and re-assembling context for the next step.*

---

## What Is an AI Agent — Core Definition

**AI Agent = Models + Tools + Orchestration Layer + Deployment**, using LM in a loop to accomplish a goal. Four elements form essential architecture of any autonomous system.

### The Four Components

| Component | Analogy | Role |
|-----------|---------|------|
| **Model** | Brain | Central reasoning engine. Processes info, evaluates options, makes decisions. Type (general-purpose, fine-tuned, multimodal) dictates cognitive capabilities. Agentic system is ultimate curator of input context window. |
| **Tools** | Hands | Connect reasoning to outside world. API extensions, code functions, data stores (databases, vector stores). Agent plans which tools to use, executes them, puts results into next LM call's context window. |
| **Orchestration Layer** | Nervous System | Governs operational loop. Handles planning, memory (state), reasoning strategy execution. Uses frameworks like Chain-of-Thought or ReAct to break goals into steps. Decides when to think vs use tool. Also responsible for agent memory. |
| **Deployment** | Body and Legs | Production hosting on secure, scalable server. Integration with monitoring, logging, management services. Accessed by users via GUI or programmatically by other agents via Agent-to-Agent (A2A) API. |

---

## The Agentic Problem-Solving Process

Agent operates on continuous, cyclical process. Five fundamental steps (from "Agentic System Design" book):

### The 5-Step Loop

**1. Get the Mission**
- Process initiated by specific, high-level goal
- Source: user request ("Organize my team's travel") OR automated trigger ("New high-priority ticket arrived")

**2. Scan the Scene**
- Agent perceives environment to gather context
- Checks: user request, term memory, past attempts, user guidance history, available tools (calendars, databases, APIs)

**3. Think It Through**
- Core "think" loop driven by reasoning model
- Analyzes Mission (Step 1) against Scene (Step 2)
- Devises plan — not single thought but chain of reasoning
- Example: "To book travel → first `get_team_roster` → then check availability via `calendar_api`"

**4. Take Action**
- Orchestration layer executes first concrete step of plan
- Selects and invokes appropriate **tool** — calling API, running code function, querying database
- This is agent *acting* on world beyond its internal reasoning

**5. Observe and Iterate**
- Agent observes *outcome* of its action
- New information added to agent's context/"memory"
- Loop repeats back to Step 3 with updated context
- Continues until initial Mission achieved

### How the Loop Maps to Architecture

"Think, Act, Observe" cycle managed by **Orchestration Layer**, reasoned by **Model**, executed by **Tools** — until plan complete and Mission achieved.

### Real-World Example: Customer Support Agent

User asks: "Where is my order #12345?"

**Think It Through phase** — devises multi-step plan:
1. **Identify:** Find order in internal database, confirm existence, get details
2. **Track:** Extract shipping carrier's tracking number, query carrier's API for live status
3. **Report:** Synthesize gathered info into clear response

**Execution:**
- Act: calls `find_order("12345")` → Observe: gets full record including tracking "ZYX987"
- Act: calls `get_shipping_status("ZYX987")` → Observe: "Out for Delivery"
- Act: generates response: "Your order #12345 is 'Out for Delivery'!"

Each step feeds next — orchestration recognizes when each sub-plan completes.

---

## A Taxonomy of Agentic Systems

5-step loop can be scaled in complexity to create different classes of agents. Each level builds on previous. Key architectural decision: scoping *what kind* of agent to build.

### Level 0: The Core Reasoning System

- LM in most basic form — reasoning engine itself
- Operates in isolation: no tools, no memory, no live environment interaction
- Responds solely on vast pre-trained knowledge
- **Strength:** Explains established concepts with great depth
- **Trade-off:** Complete lack of real-time awareness — functionally "blind" to events outside training data
- Example: Can explain Yankees history but can't tell last night's score

### Level 1: The Connected Problem-Solver

- Reasoning engine + external tools ("Hands" component)
- No longer confined to static, pre-trained knowledge
- Uses 5-step loop: recognizes real-time data need → invokes tool (Google Search API) → observes result → synthesizes answer
- **Core capability:** Ability to interact with world — search tools, financial APIs, databases via RAG
- Example: Now CAN answer last night's Yankees score by searching

### Level 2: The Strategic Problem-Solver

- Moves from simple tasks to strategically planning complex, multi-part goals
- **Key emerging skill: Context Engineering** — actively select, package, manage most relevant information for each step of plan
- Agent accuracy depends on focused, high-quality context. Context engineering curates model's limited attention to prevent overload.

**Example:** "Find good coffee shop halfway between Mountain View and San Francisco offices"
1. Think: "Find halfway point" → Act: call `Maps` → Observe: "Millbrae, CA"
2. Think: "Find coffee shops in Millbrae, 4-star+" → Act: call `google_places(query="coffee shop in Millbrae, CA", min_rating=4.0)` → Observe: results
3. Think: "Synthesize and present"

Each step's output becomes context-engineered input for next step. Also enables proactive assistance (e.g., reading flight confirmation email → extracting key context → adding to calendar).

### Level 3: The Collaborative Multi-Agent System

- Paradigm shift: from single "super-agent" to "team of specialists" working in concert
- Mirrors human organization — division of labor
- Agents treat other agents as tools
- Each agent simpler, more focused, easier to build/test/maintain
- Ideal for dynamic or long-running business processes

**Example:** Project Manager agent receives "Launch new 'Solaris' headphones"
1. Delegates to **MarketResearchAgent**: "Analyze competitor pricing for noise-canceling headphones"
2. Delegates to **MarketingAgent**: "Draft three press release versions using Solaris spec sheet"
3. Delegates to **WebDevAgent**: "Generate product page HTML from design mockups"

Currently constrained by LM reasoning limitations but represents frontier of automating entire complex business workflows.

### Level 4: The Self-Evolving System

- Profound leap: from delegation to autonomous creation and adaptation
- Can identify gaps in own capabilities and dynamically create new tools or even new agents
- Moves from fixed set of resources to actively expanding them

**Example:** Project Manager agent needs to monitor social media sentiment for "Solaris" launch, but no such tool/agent exists:
1. **Think (Meta-Reasoning):** "I must track social media buzz but lack this capability"
2. **Act (Autonomous Creation):** Invokes AgentCreator tool: "Build new agent that monitors social media for 'Solaris', performs sentiment analysis, reports daily summary"
3. **Observe:** New SentimentAnalysisAgent created, tested, added to team on the fly

System that dynamically expands own capabilities = truly learning and evolving organization.

---

## Core Agent Architecture: Model, Tools, and Orchestration

Three core components — how to actually *build* it.

### Model: The "Brain"

LM is reasoning core. Selection is critical architectural decision dictating cognitive capabilities, operational cost, and speed.

**Common mistake:** Picking model with highest benchmark score. Agent success in production rarely determined by generic academic benchmarks.

**What matters — Agentic Fundamentals:**
- Superior **reasoning** to navigate complex, multi-step problems
- Reliable **tool use** to interact with world
- Test models against metrics that directly map to YOUR business problem
- "Best" model = optimal intersection of quality, speed, and price for *your specific task*

**Model Routing — Team of Specialists:**
- Don't use sledgehammer to crack a nut
- Frontier model (e.g., Gemini 2.5 Pro) for heavy lifting: initial planning, complex reasoning
- Cost-effective model (e.g., Gemini 2.5 Flash) for simpler high-volume tasks: classifying intent, summarizing
- Routing can be automatic or hard-coded — key strategy for optimizing performance and cost

**Multimodal Handling:**
- Natively multimodal model (Gemini live mode) for images/audio — streamlined
- Alternative: specialized tools (Cloud Vision API, Speech-to-Text API) convert to text first — flexible but complex
- Trade-off: simplicity vs best-of-breed components

**Model Obsolescence:**
- Model chosen today superseded in six months
- "Set it and forget it" mindset unsustainable
- Invest in Agent Ops CI/CD pipeline that continuously evaluates new models against key business metrics
- De-risk upgrades, ensure always powered by best available brain

### Tools: The "Hands"

Connect reasoning to reality. Move beyond static training data to retrieve real-time info and take action. Three-part loop: define what tool does → invoke it → observe result.

#### Retrieving Information: Grounding in Reality

Most foundational tool type — accessing up-to-date information:
- **RAG (Retrieval-Augmented Generation):** "Library card" to query external knowledge. Stored in Vector Databases or Knowledge Graphs. Sources: internal company documents → web knowledge via Google Search
- **NL2SQL (Natural Language to SQL):** Query databases for structured data. "What were our top-selling products last quarter?"
- Looking things up before speaking = grounding in fact = dramatically reduces hallucinations

#### Executing Actions: Changing the World

True power unleashed when moving from reading to *doing*:
- Wrap existing **APIs** and code functions as tools → send email, schedule meeting, update CRM record
- **Write and execute code on the fly** — in secure sandbox, generate SQL/Python to solve complex problems
- Transforms from knowledgeable assistant into **autonomous actor**
- **Human in the Loop (HITL):** Tool to pause workflow and ask for confirmation (`ask_for_confirmation()`) or request specific input (`ask_for_date_input()`). Can be implemented via SMS + database task. Ensures human involvement in critical decisions.

#### Function Calling: Connecting Tools to Agent

For reliable function calling, agent needs clear instructions, secure connections, orchestration:
- **OpenAPI specification:** Structured contract describing tool's purpose, required parameters, expected response. Lets model generate correct function call every time.
- **Model Context Protocol (MCP):** Open standard for simpler discovery and connection to tools. More convenient alternative to OpenAPI.
- **Native tools:** Some models have built-in tools (e.g., Gemini with native Google Search) — function invocation happens as part of LM call itself.

### The Orchestration Layer

Central nervous system connecting brain and hands. Engine running "Think, Act, Observe" loop. State machine governing agent behavior. Where developer's carefully crafted logic comes to life.

Not just plumbing — **conductor of entire agentic symphony**: deciding when model should reason, which tool should act, how action results inform next movement.

#### Core Design Choices

**Autonomy Spectrum:**
- One end: deterministic, predictable workflows calling LM as tool for specific task (sprinkle of AI)
- Other end: LM in driver's seat, dynamically adapting, planning, executing (full autonomy)

**Implementation Method:**
- No-code builders: speed, accessibility, simple agents rapidly
- Code-first frameworks (Google ADK): deep control, customizability, integration for mission-critical systems

**Production-Grade Framework Requirements:**
1. **Open:** Plug in any model or tool, prevent vendor lock-in
2. **Precise control:** Hybrid approach — non-deterministic LM reasoning governed by hard-coded business rules
3. **Observability:** Generate detailed traces and logs exposing entire reasoning trajectory: model's internal monologue, tool chosen, parameters generated, result observed. Can't put breakpoint in model's "thought."

#### Instruct with Domain Knowledge and Persona

Developer's most powerful lever. System prompt = agent's constitution.
- Domain knowledge, distinct persona
- Constraints, desired output schema, rules of engagement, tone of voice
- Explicit guidance on when/why to use tools
- Few example scenarios in instructions = usually very effective

```
You are a helpful customer support agent for Acme Corp, ...
```

#### Augment with Context

Agent's "memory" orchestrated into LM context window at runtime.

**Short-term memory:**
- Active "scratchpad" — running history of current conversation
- Tracks sequence of (Action, Observation) pairs from ongoing loop
- Provides immediate context for next decision
- Implemented as: state, artifacts, sessions, threads

**Long-term memory:**
- Persistence across sessions
- Architecturally: specialized tool — RAG system connected to vector database or search engine
- Orchestrator pre-fetches and queries agent's own history
- Enables "remembering" user preferences, outcomes of similar past tasks
- Truly personalized, continuous experience

#### Multi-Agent Systems and Design Patterns

Building single "super-agent" becomes inefficient as complexity grows. "Team of specialists" approach mirrors human organization.

**Multi-agent system:** Complex process segmented into discrete sub-tasks, each assigned to dedicated, specialized AI agent. Division of labor → each agent simpler, more focused, easier to build/test/maintain.

**Key Design Patterns:**

| Pattern | Description | Use When |
|---------|-------------|----------|
| **Coordinator** | Manager agent analyzes request, segments task, routes sub-tasks to specialist agents, aggregates responses | Dynamic or non-linear tasks |
| **Sequential** | Digital assembly line — output of one agent = direct input for next | Linear workflows |
| **Iterative Refinement** | "Generator" agent creates content, "Critic" agent evaluates against quality standards, loop until quality met | Quality-critical output |
| **Human-in-the-Loop (HITL)** | Deliberate pause in workflow to get human approval before significant action | High-stakes tasks |

---

## Agent Deployment and Services

After building local agent → deploy to server where it runs 24/7 and others can use it. "Body and legs" of our analogy.

**Required Services:**
- Session history and memory persistence
- Monitoring, logging, management
- Security measures, data privacy, regulation compliance
- All in scope when deploying to production

**Deployment Options:**
- **Vertex AI Agent Engine:** Purpose-built, agent-specific. Support runtime + everything else in one platform
- **Docker + Cloud Run/GKE:** For developers wanting control over application stack. Deploy within existing DevOps infrastructure.
- Many frameworks offer simple `deploy` command for early exploration. Production readiness requires more investment: CI/CD, automated testing.

---

## Agent Ops: A Structured Approach to the Unpredictable

Transition from deterministic software to stochastic, agentic systems requires new operational philosophy. Traditional unit tests (`output == expected`) don't work — agent response is probabilistic by design. Language is complicated — evaluating "quality" usually requires LM itself.

**Agent Ops** = natural evolution of DevOps and MLOps, tailored for building, deploying, governing AI agents. Turns unpredictability from liability into managed, measurable, reliable feature.

**Ops Hierarchy:** DevOps → MLOps → FMOps → GenAIOps → AgentOps (+ RAGOps, PromptOps as subcategories)

### Measure What Matters: Instrumenting Success Like A/B Experiment

Define "better" in context of your business. KPIs that prove agent delivers value:
- Goal completion rates
- User satisfaction scores
- Task latency
- Operational cost per interaction
- **Impact on business goals:** revenue, conversion, customer retention

Frame observability as A/B test. Top-down view → metrics-driven development → ROI calculation.

### Quality Instead of Pass/Fail: Using LM Judge

Since simple pass/fail impossible, evaluate quality using "LM as Judge":
- Powerful model assesses agent output against predefined rubric
- Did it give right answer? Was response factually grounded? Did it follow instructions?
- Run against **golden dataset** of prompts → consistent quality measure

**Building evaluation datasets:**
- Sample scenarios from existing production/development interactions
- Must cover full breadth of expected use cases + few unexpected ones
- Domain expert review before accepting as valid
- Increasingly, PM + Domain expert responsibility to curate and maintain

### Metrics-Driven Development: Go/No-Go for Deployment

Once you have automated evaluation scenarios with trusted quality scores:
- Run new version against entire evaluation dataset
- Compare scores directly to existing production version
- Eliminates guesswork → confident in every deployment
- Don't forget: latency, cost, task success rates alongside quality scores
- Use A/B deployments to slowly roll out, compare real-world production metrics alongside simulation scores

### Debug with OpenTelemetry Traces: Answering "Why?"

When metrics dip or user reports bug → need to understand "why."

**OpenTelemetry trace** = high-fidelity, step-by-step recording of agent's entire execution path (trajectory):
- Exact prompt sent to model
- Model's internal reasoning (if available)
- Specific tool chosen and precise parameters generated
- Raw data that came back as observation

Traces complicated at first but provide root cause details. Collected in platforms like **Google Cloud Trace** for visualization and search.

Important: traces primarily for debugging, not performance overviews. Important trace details may be turned into metrics.

### Cherish Human Feedback: Guiding Your Automation

Human feedback = most valuable, data-rich resource for improvement. Bug reports, thumbs-down = gift — real-world edge cases your automated evals missed.

**Closing the loop:**
1. Collect and aggregate feedback
2. Statistically significant reports → tie back to analytics → trigger alerts
3. Capture feedback → replicate issue → convert to new permanent test case in evaluation dataset
4. Fix the bug AND vaccinate against entire class of error

---

## Agent Interoperability

Once you have high-quality agents → interconnect them with users and other agents. "Face" of the agent. **Agents are not tools** — connecting to agents differs from connecting agents to data/APIs.

### Agents and Humans

**User interfaces — from simple to advanced:**
- Chatbot (simplest): user types request → agent processes → returns text
- Structured data (JSON) to power rich, dynamic front-end experiences
- HITL interaction patterns: intent refinement, goal expansion, confirmation, clarification requests

**Computer Use:** LM takes control of UI — navigate pages, click buttons, pre-fill forms. Implemented via:
- MCP UI tools (controlling UI via MCP)
- AG UI (protocol for controlling UI via event passing + shared state)
- A2UI (generating bespoke interfaces via structured output + A2A message passing)

**Multimodal — Beyond Text:**
- Breaking text barrier with "live mode" — real-time, multimodal communication
- **Gemini Live API:** Bidirectional streaming — speak to agent, interrupt naturally
- With camera + microphone access: agent sees what user sees, hears what they say
- Responds with generated speech at human conversation latency
- Use cases: hands-free technician guidance, real-time style advice for shoppers

### Agents and Agents

Enterprise scales → different teams build different specialized agents. Without common standard → tangled web of brittle custom API integrations.

**Core challenge is twofold:**
1. **Discovery:** How does my agent find other agents and know their capabilities?
2. **Communication:** How do we ensure they speak same language?

**Agent2Agent (A2A) Protocol** — open standard solving this:
- Universal handshake for agentic economy
- Agent publishes digital "business card" = **Agent Card** (JSON file)
- Advertises: capabilities, network endpoint, security credentials
- Makes discovery simple and standardized

**A2A vs MCP:** MCP focuses on transactional requests (tool calls). A2A is for additional problem-solving between agents.

**Communication model:**
- Task-oriented architecture (not simple request-response)
- Client agent sends task request → server agent provides streaming updates over long-running connection
- Enables collaborative Level 3 multi-agent systems
- Transforms isolated agents into true interoperable ecosystem

### Agents and Money

Current web built for humans clicking "buy." Autonomous agent clicking "buy" creates crisis of trust — who is at fault?

**Two emerging protocols building trust layer:**

| Protocol | Purpose | Mechanism |
|----------|---------|-----------|
| **Agent Payments Protocol (AP2)** | Definitive language for agentic commerce | Extends A2A with cryptographically-signed "digital mandates" = verifiable proof of user intent. Non-repudiable audit trail per transaction. |
| **x402** | Frictionless machine-to-machine micropayments | Uses HTTP 402 "Payment Required" status code. Pay-per-use for API access or digital content. No complex accounts/subscriptions needed. |

Together: foundational trust layer for agentic web.

---

## Securing a Single Agent: The Trust Trade-Off

Fundamental tension: **utility vs security**. More power = more risk. Primary concerns: **rogue actions** (unintended/harmful behaviors) and **sensitive data disclosure**.

Can't rely solely on AI model's judgment — manipulable via prompt injection.

### Defense-in-Depth Approach (Two Layers)

**Layer 1: Traditional, Deterministic Guardrails**
- Hardcoded rules as security chokepoint OUTSIDE model's reasoning
- Policy engine blocking purchases over $100, requiring user confirmation before external API calls
- Predictable, auditable hard limits on agent's power

**Layer 2: Reasoning-Based Defenses**
- AI securing AI
- Adversarial training for resilience
- Smaller, specialized "guard models" acting as security analysts
- Examine agent's proposed plan BEFORE execution
- Flag potentially risky or policy-violating steps

Hybrid model = rigid certainty of code + contextual awareness of AI = robust security posture.

### Agent Identity: A New Class of Principal

Traditional security model has two principal types: human users (OAuth/SSO) and services (IAM/service accounts). **Agents add a 3rd category.**

Agent is not merely code — it's autonomous actor, new kind of *principal* requiring own verifiable identity. Like employees issued ID badges, each agent needs secure, verifiable "digital passport."

**Agent Identity is DISTINCT from:**
- Identity of user who invoked it
- Identity of developer who built it

Fundamental shift in how we approach IAM in enterprise.

| Principal Entity | Authentication | Notes |
|-----------------|---------------|-------|
| **Users** | OAuth / SSO | Full autonomy and responsibility |
| **Agents** (new) | SPIFFE verification | Delegated authority, acting on behalf of users |
| **Service accounts** | IAM integration | Fully deterministic, no responsibility for actions |

**Granular control via verified identity:**
- `SalesAgent` granted read/write CRM access
- `HRonboardingAgent` explicitly denied CRM access
- Even if single agent compromised → blast radius contained

Without agent identity construct, agents cannot work on behalf of humans with limited delegated authority.

### Policies to Constrain Access

Policy = authorization (AuthZ), distinct from authentication (AuthN). Policies limit capabilities of principal.

Apply permissions to: agents, their tools, other internal agents, context they share, remote agents. **Think of it this way:** if you add ALL APIs, data, tools, agents → you must constrain access to subset required. **Principle of least privilege** while remaining contextually relevant.

### Securing an ADK Agent

Practical exercise applying identity + policy concepts through code and configuration:

**Authentication layers:**
1. User account (e.g., OAuth)
2. Service account (to run code)
3. Agent identity (to use delegated authority)

**Policy enforcement layers:**
1. API governance layer — constrain access to services (supporting MCP + A2A)
2. In-tool guardrails — tool's own logic refuses unsafe/out-of-policy actions regardless of LM reasoning or malicious prompt
3. Provides predictable, auditable security baseline

**Dynamic Security — Callbacks and Plugins:**
- `before_tool_callback` — inspect parameters of tool call BEFORE execution, validate against agent's current state
- Reusable plugins pattern: "Gemini as Judge" — fast, inexpensive model (Gemini Flash-Lite or fine-tuned Gemma) screens inputs/outputs for prompt injections or harmful content in real time

**Model Armor** (managed service):
- Specialized security layer screening prompts and responses
- Covers: prompt injection, jailbreak attempts, PII leakage, malicious URLs
- Offloads complex security tasks → consistent, robust protection without building guardrails yourself

**Hybrid approach in ADK:** Strong identity + deterministic in-tool logic + dynamic AI-powered guardrails + optional managed services like Model Armor = powerful AND trustworthy single agent.

---

## Scaling Up: Single Agent to Enterprise Fleet

Single agent success = triumph. Fleet of hundreds = architecture challenge. One or two agents → focus on security. Many agents → design systems for governance, discovery, cost, reliability.

Like API sprawl — agents and tools proliferating across org create complex network of interactions, data flows, security vulnerabilities. Requires higher-order governance layer integrating identities, policies, reporting into central control plane.

### Security and Privacy: Hardening the Agentic Frontier

Enterprise-grade platform must address unique security/privacy challenges even with single agent running. Agent itself = new attack vector:
- **Prompt injection:** Hijack agent's instructions
- **Data poisoning:** Corrupt information used for training or RAG
- **Data leakage:** Poorly constrained agent leaks sensitive customer data or proprietary info

**Platform defense-in-depth:**
1. Enterprise proprietary info never used to train base models — protected by VPC Service Controls
2. Input/output filtering — firewall for prompts and responses
3. Contractual protections — IP indemnity for training data AND generated output

### Agent Governance: A Control Plane Instead of Sprawl

Managing agent sprawl requires moving beyond securing individual agents → implementing **central gateway** as control plane for all agentic activity.

**Gateway analogy:** Metropolis with thousands of autonomous vehicles. Without traffic lights, license plates, central control → chaos. Gateway = mandatory entry point for ALL agentic traffic:
- User-to-agent prompts / UI interactions
- Agent-to-tool calls (via MCP)
- Agent-to-agent collaborations (via A2A)
- Direct inference requests to LMs

**Two interconnected functions:**

**1. Runtime Policy Enforcement:**
- Architectural chokepoint for security
- Handles AuthN ("Who is this actor?") and AuthZ ("Can they do this?")
- Centralizing → "single pane of glass" for observability
- Common logs, metrics, traces for every transaction
- Transforms spaghetti of disparate agents into transparent, auditable system

**2. Centralized Governance:**
- Central registry — enterprise app store for agents and tools
- Developers discover and reuse existing assets (prevents redundant work)
- Admins get complete inventory
- Enables formal lifecycle: security reviews before publication, versioning, fine-grained policies for which business units access which agents

Gateway + registry → transforms chaotic sprawl into managed, secure, efficient ecosystem.

### Cost and Reliability: The Infrastructure Foundation

Enterprise agents must be reliable AND cost-effective.
- Agent that frequently fails or is slow = negative ROI
- Prohibitively expensive agent = can't scale to meet business demands
- Infrastructure must manage trade-off securely with regulatory + data sovereignty compliance

**Scaling options:**
- Scale-to-zero for irregular traffic to specific agent/sub-function
- **Provisioned Throughput** for mission-critical, latency-sensitive workloads (dedicated, guaranteed capacity)
- 99.9% SLAs for runtimes (Cloud Run)

Spectrum of infrastructure options + comprehensive cost/performance monitoring = foundation for scaling AI agents from promising innovation to core, reliable enterprise component.

---

## How Agents Evolve and Learn

Agents in real world operate where policies, technologies, data formats constantly change. Without ability to adapt → performance degrades over time ("aging") → loss of utility and trust. Manually updating large fleet = uneconomical and slow.

**Solution:** Design agents that learn and evolve autonomously, improving quality on the job with minimal engineering effort.

### How Agents Learn and Self-Evolve

Agents learn from experience and external signals:

**Learning Sources:**
- **Runtime Experience:** Session logs, traces, memory capturing successes, failures, tool interactions, decision trajectories. Crucially includes HITL feedback — authoritative corrections and guidance.
- **External Signals:** Updated enterprise policies, public regulatory guidelines, critiques from other agents.

**Two Most Successful Adaptation Techniques:**

| Technique | How It Works |
|-----------|-------------|
| **Enhanced Context Engineering** | System continuously refines prompts, few-shot examples, information retrieved from memory. Optimizes context provided to LM per task → increases success likelihood. |
| **Tool Optimization and Creation** | Agent's reasoning identifies capability gaps and fills them. Gain access to new tool, create one on the fly (Python script), or modify existing tool (update API schema). |

Additional research areas: dynamically reconfiguring multi-agent design patterns, RLHF.

### Example: Learning New Compliance Guidelines

Enterprise agent generating reports in regulated industry (finance, life sciences). Must comply with privacy/regulatory rules (GDPR).

**Multi-agent workflow:**
1. **Querying Agent** — retrieves raw data for user request
2. **Reporting Agent** — synthesizes into draft report
3. **Critiquing Agent** — reviews against known compliance guidelines. Escalates to human if ambiguous.
4. **Learning Agent** — observes entire interaction, especially corrective feedback from human expert. Generalizes feedback into new reusable guideline (updated rule for critiquing agent or refined context for reporting agent).

**Self-improvement loop:** Human expert flags "household statistics must be anonymized" → Learning Agent records → next time Critiquing Agent automatically applies new rule → reduced human intervention. Loop of critique → human feedback → generalization = autonomous adaptation to evolving compliance.

### Simulation and Agent Gym — The Next Frontier

Design pattern above = "in-line learning" (learns with resources of its own design pattern). More advanced: **Agent Gym** — dedicated off-production platform for optimizing multi-agent systems offline.

**Key Attributes of Agent Gym:**
1. **Not in execution path** — standalone, off-production. Can leverage any LM, offline tools, cloud applications
2. **Simulation environment** — agent "exercises" on new data and learns. Trial-and-error with many optimization pathways
3. **Synthetic data generators** — guide simulation to be realistic. Pressure test agent via red-teaming, dynamic evaluation, family of critiquing agents
4. **Extensible optimization tools** — not fixed. Adopts new tools via MCP/A2A. Advanced: learns new concepts and crafts tools around them
5. **Human fabric connection** — for edge cases involving "tribal knowledge," Agent Gym connects to domain experts for guidance on next optimizations

---

## Examples of Advanced Agents

### Google Co-Scientist

Advanced AI agent functioning as virtual research collaborator. Accelerates scientific discovery by systematically exploring complex problem spaces.

**How it works:**
- Researcher defines goal, grounds in public/proprietary knowledge sources
- System generates and evaluates landscape of novel hypotheses
- Spawns entire ecosystem of collaborating agents

**Architecture:**
- Supervisor agent acts as project manager
- Delegates to specialized agents, distributes computing resources
- Ensures project scales up and methods improve toward final goal

**Specialized Agents in Co-Scientist:**
- **Generation Agent:** Literature exploration, simulated scientific debate
- **Reflection Agent:** Full review with web search, simulation review, tournament review, deep verification
- **Evolution Agent:** Inspiration from other ideas, simplification, research extension
- **Ranking Agent:** Research hypothesis comparison/ranking via tournaments. Win/loss patterns → feedback → self-improving loop
- **Proximity Check Agent:** Research overview formulation
- **Meta-review Agent:** Research overview formulation
- **Knowledge Agent:** Builds ranked list of ideas, research plan, knowledge base

Agents work hours or days, running loops and meta-loops improving not only generated ideas but also the way they judge and create new ideas.

### AlphaEvolve Agent

AI agent that discovers and optimizes algorithms for complex problems in mathematics and computer science.

**Mechanism:** Combines creative code generation of Gemini LMs with automated evaluation system. **Evolutionary process:** AI generates potential solutions → evaluator scores them → most promising ideas = inspiration for next generation of code.

**Breakthroughs:**
- Improved efficiency of Google data centers, chip design, AI training
- Discovered faster matrix multiplication algorithms
- Found new solutions to open mathematical problems

**Key insight:** Excels where verifying quality of solution is far easier than finding it.

**Human-AI Collaboration (Two Dimensions):**
- **Transparent Solutions:** AI generates human-readable code. Users understand logic, gain insights, trust results, modify directly.
- **Expert Guidance:** Humans essential for defining problem. Refine evaluation metrics, steer exploration, prevent system from exploiting unintended loopholes. Interactive loop ensures solutions are powerful AND practical.

**Result:** Continuous improvement of code that keeps improving metrics specified by human.

---

## Conclusion — Key Takeaways

1. Agents = pivotal evolution from passive tool to active, autonomous problem-solver
2. Three essential components: **Model** (Brain) + **Tools** (Hands) + **Orchestration Layer** (Nervous System)
3. Continuous "Think, Act, Observe" loop unlocks true potential
4. Taxonomy: Level 0 (reasoning only) → Level 4 (self-evolving) — scope ambitions to match task complexity
5. Developer paradigm shift: no longer "bricklayers" defining explicit logic → "architects" and "directors" guiding autonomous entities
6. LM flexibility = source of power AND unreliability. Success NOT in initial prompt alone — in engineering rigor: robust tool contracts, resilient error handling, sophisticated context management, comprehensive evaluation
7. Principles here serve as foundational blueprint for navigating this new frontier of software
