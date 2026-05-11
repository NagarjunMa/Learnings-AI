# Prototype to Production: AgentOps Lifecycle

> Google Whitepaper — Authors: Sokratis Kartakis, Gabriela Hernandez Larios, Ran Li, Elia Secchi, Huang Xia (Nov 2025)

> **Building an agent is easy. Trusting it is hard.**

## Abstract

Comprehensive technical guide to operational lifecycle of AI agents — deployment, scaling, productionizing. Building on Day 4 (evaluation/observability), emphasizes how to build necessary trust to move agents into production through robust CI/CD pipelines and scalable infrastructure. Special attention to **Agent2Agent (A2A) interoperability**.

## Introduction: From Prototype to Production

You can spin up AI agent prototype in minutes, maybe seconds. Turning clever demo into trusted production-grade system that business can depend on = where real work begins. **"Last mile" production gap**: roughly **80% of effort** spent not on agent's core intelligence, but on infrastructure, security, validation needed to make it reliable and safe.

Skipping final steps causes problems:
- **Customer service agent tricked into giving products away for free** (forgot to set up right guardrails)
- **User discovers they can access confidential internal database** through agent (improper auth)
- **Agent generates large consumption bill over weekend** (no monitoring)
- **Critical agent that worked perfectly yesterday suddenly stops** (no continuous evaluation)

These = major **business failures**, not just technical problems. Principles from DevOps and MLOps provide critical foundation but aren't enough. Deploying agentic systems introduces new class of challenges requiring **evolution in operational discipline**. Unlike traditional ML models, agents are **autonomously interactive, stateful, follow dynamic execution paths**.

Unique operational headaches:
- **Dynamic Tool Orchestration**: agent's "trajectory" assembled on the fly. Robust versioning, access control, observability for system that behaves differently every time.
- **Scalable State Management**: agents remember things across interactions. Managing session/memory securely and consistently at scale = complex systems design problem.
- **Unpredictable Cost & Latency**: agent can take many paths to find answer → cost/response time hard to predict and control without smart budgeting and caching.

To navigate, foundation built on **three key pillars: Automated Evaluation, Automated Deployment (CI/CD), Comprehensive Observability**.

Step-by-step playbook: pre-production essentials (CI/CD + rigorous evaluation as quality check) → in-production challenges (scaling, performance tuning, real-time monitoring) → multi-agent systems with Agent-to-Agent protocol.

> **Practical Implementation Guide**: Examples reference **Google Cloud Platform Agent Starter Pack** — Python package with production-ready GenAI agent templates. Includes pre-built agents, automated CI/CD setup, Terraform deployment, Vertex AI evaluation integration, built-in Cloud observability.

## People and Process

Best technology is ineffective without right team. Customer service agent not magically prevented from giving away products — **AI Engineer** and **Prompt Engineer** design and implement guardrails. Confidential database not secured by abstract concept — **Cloud Platform team** configures authentication.

> **Figure 1**: "Ops" = intersection of People, Processes, Technology
> - MLOps = Machine Learning & Operations: combination of people, processes, and technology to productionize ML solutions efficiently

Traditional MLOps key teams:
- **Cloud Platform Team**: cloud architects, administrators, security specialists. Manages foundational cloud infrastructure, security, access control. Grants engineers/service accounts least-privilege roles.
- **Data Engineering Team**: data engineers + data owners. Builds/maintains data pipelines (ingestion, preparation, quality standards).
- **Data Science and MLOps Team**: data scientists experiment/train models, ML engineers automate end-to-end pipeline (preprocessing, training, post-processing) at scale using CI/CD. MLOps Engineers build standardized pipeline infrastructure.
- **Machine Learning Governance**: centralized function — product owners, auditors. Oversees ML lifecycle, repository for artifacts/metrics, ensures compliance, transparency, accountability.

Generative AI introduces new specialized roles:
- **Prompt Engineers**: blend technical skill in crafting prompts with deep domain expertise. Define right questions and expected answers from a model. In practice may be done by AI Engineers, domain experts, dedicated specialists depending on org maturity.
- **AI Engineers**: scale GenAI to production. Build robust backend systems incorporating evaluation at scale, guardrails, RAG/tool integration.
- **DevOps/App Developers**: build front-end + user-friendly interfaces integrating with GenAI backend.

Smaller orgs = multiple hats; mature orgs = specialized teams. Coordinating diverse roles essential for robust operational foundation.

> **Figure 2**: How multiple teams collaborate to operationalize models and GenAI applications
> - Cloud Platform layer (Cloud Architect/Administrator, Security SME, GenAI/MLOps Engineer): Provision infrastructure (IAM, networking, security); Standardize requirement and path to production
> - Data Science & MLOps: Experiment with algorithms/models → Automate data pre-processing/training/post-processing → Evaluate at scale → Continuously monitor/improve. (Data Scientist/Fine-tuner, ML Engineer)
> - Data Engineering: Ingest/Clean/Catalog Data, Maintain governance/access, Visualize Data. (Data Engineer, Data Owner, Business Stakeholder)
> - Generative AI Application: Select/Evaluate Foundation Model → Build robust backend with guardrails/evaluation/monitoring/context retrieval → Build front-end → Test at scale, promote to production via CI/CD pipelines and multiple stages (dev, staging, prod). (Prompt Engineer/Tester, AI Engineer, DevOps/AppDev)
> - AI Governance layer: Establish success metrics, Approve model/app to move to production, Audit data/artifacts, Receive alerts/customer feedback. (Product Owner/Approver, Auditor)

## The Journey to Production

Translate work of all specialists into trustworthy reliable system ready for users. Disciplined pre-production process built on single core principle: **Evaluation-Gated Deployment**. No agent version reaches users without first passing comprehensive evaluation proving quality and safety. Three pillars: rigorous evaluation as quality gate, automated CI/CD pipeline enforcing it, safe rollout strategies de-risking final step.

### Evaluation as a Quality Gate

Why special quality gate for agents? Traditional software tests insufficient for systems that reason and adapt. Evaluating an agent is distinct from evaluating an LLM — assesses not just final answer, but **entire trajectory of reasoning and actions** taken to complete task. Agent can pass 100 unit tests but still fail spectacularly by choosing wrong tool or hallucinating. Evaluate **behavioral quality**, not just functional correctness.

Two primary implementations:

1. **Manual "Pre-PR" Evaluation**: for teams seeking flexibility or beginning evaluation journey. Quality gate enforced through team process. Before submitting PR, AI Engineer or Prompt Engineer (whoever responsible) runs evaluation suite locally. Resulting performance report — comparing new agent against production baseline — linked in PR description. Evaluation results = mandatory artifact for human review. Reviewer (typically another AI Engineer or **Machine Learning Governor**) responsible for assessing not just code, but agent's behavioral changes against guardrail violations and prompt injection vulnerabilities.

2. **Automated In-Pipeline Gate**: for mature teams. Evaluation harness — built/maintained by Data Science and MLOps Team — integrated directly into CI/CD pipeline. Failing evaluation automatically blocks deployment, providing rigid programmatic enforcement of quality standards Machine Learning Governance team has defined. Trades flexibility of manual review for consistency of automation. CI/CD configured to automatically trigger evaluation job comparing new agent's responses against **golden dataset**. Programmatically blocked if key metrics ("tool call success rate" or "helpfulness") fall below threshold.

Regardless of method: principle is same — no agent proceeds to production without quality check. Specifics covered in **Day 4: Agent Quality** (golden dataset, LLM-as-a-judge, **Vertex AI Evaluation** service).

### The Automated CI/CD Pipeline

AI agent = composite system: source code + prompts + tool definitions + configuration files. How ensure prompt change doesn't degrade tool performance? How test interplay between artifacts before reaching users?

**CI/CD pipeline** = solution. More than automation script — structured process helping different team people collaborate to manage complexity and ensure quality. Tests changes in stages, incrementally building confidence before agent released.

Robust pipeline = funnel. Catches errors as early and cheaply as possible — practice often called **"shifting left"**. Separates fast pre-merge checks from comprehensive resource-intensive post-merge deployments. Three distinct phases:

1. **Phase 1: Pre-Merge Integration (CI)**: pipeline's first responsibility = rapid feedback to AI Engineer or Prompt Engineer who opened PR. Triggered automatically. CI phase = gatekeeper for main branch. Fast checks: unit tests, code linting, dependency scanning. **Crucially, this is the ideal stage to run the agent quality evaluation suite** designed by Prompt Engineers. Immediate feedback on whether change improves or degrades agent's performance against key scenarios before merged. Catches issues here → prevents polluting main branch. **PR checks configuration template** generated with **Agent Starter Pack (ASP)** = practical example using Cloud Build.

2. **Phase 2: Post-Merge Validation in Staging (CD)**: after change passes CI checks (including performance evaluation) and merged → focus shifts from code/performance correctness to **operational readiness of integrated system**. CD process, often managed by **MLOps Team**, packages agent and deploys to staging environment — high-fidelity replica of production. More comprehensive resource-intensive tests run: **load testing**, **integration tests** against remote services. Critical phase for internal user testing (often called **"dogfooding"**) — humans within company interact with agent and provide qualitative feedback before reaching end user. Ensures agent as integrated system performs reliably/efficiently under production-like conditions before release. Staging deployment template from ASP.

3. **Phase 3: Gated Deployment to Production**: after thorough validation in staging, final step = production. Almost never fully automatic — typically requires **Product Owner** sign-off, ensuring human-in-the-loop. Upon approval, exact deployment artifact tested and validated in staging promoted to production. Production deployment template from ASP shows how this final phase retrieves validated artifact and deploys to production with appropriate safeguards.

> **Figure 3**: Different stages of the CI/CD process
> - On Pull Request → Triggers → CI Pipeline (Unit Tests → Integration Tests)
> - Merge to main branch → Triggers → CD Pipeline #1 (Container build → Deployment to Staging → Load tests) → CD Pipeline #2 (with manual approval) (Deploy to Prod)

Three-phase workflow requires robust automation infrastructure + proper secrets management. Powered by:
- **Infrastructure as Code (IaC)**: Terraform programmatically defines environments — identical, repeatable, version-controlled. Template generated with Agent Starter Pack provides Terraform configurations for complete agent infrastructure including Vertex AI, Cloud Run, BigQuery resources.
- **Automated Testing Frameworks**: Pytest executes tests/evaluations at each stage, handling agent-specific artifacts (conversation histories, tool invocation logs, dynamic reasoning traces).

Sensitive info like API keys for tools should be managed securely using **Secret Manager** and injected into agent's environment at runtime, not hardcoded.

### Safe Rollout Strategies

While comprehensive pre-production checks essential, real-world application inevitably reveals unforeseen issues. Rather than switching 100% of users at once, minimize risk through gradual rollouts with careful monitoring.

Four proven patterns:
- **Canary**: start with 1% of users, monitoring for prompt injections and unexpected tool usage. Scale up gradually or roll back instantly.
- **Blue-Green**: run two identical production environments. Route traffic to "blue" while deploying to "green," then switch instantly. If issues emerge, switch back — zero downtime, instant recovery.
- **A/B Testing**: compare agent versions on real business metrics for data-driven decisions. Internal or external users.
- **Feature Flags**: deploy code but control release dynamically, testing new capabilities with select users first.

Foundation: **rigorous versioning**. Every component — code, prompts, model endpoints, tool schemas, memory structures, even evaluation datasets — must be versioned. When issues arise despite safeguards, enables instant rollback to known-good state. **Production "undo" button.**

Deploy agents using **Agent Engine** or **Cloud Run**, leverage **Cloud Load Balancing** for traffic management across versions or to connect to other microservices. Agent Starter Pack provides ready-to-use templates with **GitOps workflows** — every deployment is git commit, every rollback is git revert, repository = single source of truth for current state and complete deployment history.

### Building Security from the Start

Safe deployment strategies protect from bugs/outages. Agents face unique challenge: **they can reason and act autonomously**. Perfectly deployed agent can still cause harm if not built with proper security/responsibility measures. Comprehensive governance strategy embedded from day one, not added as afterthought.

Unlike traditional software following predetermined paths, agents make decisions. Interpret ambiguous requests, access multiple tools, maintain memory across sessions. Autonomy creates distinct risks:
- **Prompt Injection & Rogue Actions**: malicious users trick agents into unintended actions or bypassing restrictions
- **Data Leakage**: agents may inadvertently expose sensitive info through responses or tool usage
- **Memory Poisoning**: false info stored in agent's memory can corrupt all future interactions

Frameworks like **Google's Secure AI Agents approach** and **Google Secure AI Framework (SAIF)** address through three layers of defense:

1. **Policy Definition and System Instructions (The Agent's Constitution)**: process begins by defining policies for desired and undesired behavior. Engineered into **System Instructions (SIs)** acting as agent's core constitution.

2. **Guardrails, Safeguards, and Filtering (The Enforcement Layer)**: hard-stop enforcement.
   - **Input Filtering**: classifiers and services like Perspective API to analyze prompts and block malicious inputs before reaching agent.
   - **Output Filtering**: after agent generates response, **Vertex AI's built-in safety filters** provide final check for harmful content, PII, policy violations. Configurable to block PII, toxic language, other harmful content.
   - **Human-in-the-Loop (HITL) Escalation**: high-risk or ambiguous actions → system pauses and escalates to human for review/approval.

3. **Continuous Assurance and Testing**: safety isn't one-time setup. Constant evaluation and adaptation.
   - **Rigorous Evaluation**: any change to model or its safety systems must trigger full re-run of comprehensive evaluation pipeline using Vertex AI Evaluation.
   - **Dedicated RAI Testing**: rigorously test for specific risks via dedicated datasets or simulation agents, including **Neutral Point of View (NPOV) evaluations** and **Parity evaluations**.
   - **Proactive Red Teaming**: actively try to break safety systems through creative manual testing and AI-driven **persona-based simulation**.

## Operations in-Production

Agent live. Focus shifts from development to fundamentally different challenge: **keeping system reliable, cost-effective, safe** as it interacts with thousands of users. Traditional service operates on predictable logic. Agent = autonomous actor. Ability to follow unexpected reasoning paths means it can exhibit emergent behaviors and accumulate costs without direct oversight.

Managing autonomy requires different operational model. Instead of static monitoring, effective teams adopt continuous loop: **Observe → Act → Evolve**.
- Observe system's behavior in real-time
- Act to maintain performance and safety
- Evolve agent based on production learnings

Integrated cycle = core discipline for operating agents successfully in production.

### Observe: Your Agent's Sensory System

To trust and manage autonomous agent, must first understand its process. Observability provides this insight = **sensory system** for subsequent "Act" and "Evolve" phases. Three pillars:

- **Logs**: granular factual diary of what happened — every tool call, error, decision
- **Traces**: narrative connecting individual logs, revealing causal path of why agent took action
- **Metrics**: aggregated report card summarizing performance, cost, operational health at scale

In Google Cloud, achieved through operations suite: user's request generates unique ID in **Cloud Trace** linking **Vertex AI Agent Engine** invocation, model calls, tool executions with visible durations. Detailed logs flow to **Cloud Logging**. **Cloud Monitoring** dashboards alert when latency thresholds exceeded. **Agent Development Kit (ADK)** provides built-in Cloud Trace integration for automatic instrumentation.

Move from operating in dark to clear data-driven view of behavior, providing foundation needed to manage effectively. (Full discussion: **Agent Quality: Observability, Logging, Tracing, Evaluation, Metrics**.)

### Act: The Levers of Operational Control

Observations without action = expensive dashboards. **Act** phase = real-time intervention. Levers to manage performance, cost, safety based on what you observe.

Think of **Act** as system's **automated reflexes** designed to maintain stability in real-time. **Evolve** = strategic process of learning from behavior to create fundamentally better system.

Agent autonomous → cannot pre-program every possible outcome. Must build robust mechanisms to influence behavior in production. Two primary categories: managing system's health and managing its risk.

#### Managing System Health: Performance, Cost, and Scale

Unlike traditional microservices, agent's workload = dynamic and stateful. Strategy for handling unpredictability.

**Designing for Scale**: foundation = decoupling agent's logic from its state.
- **Horizontal Scaling**: design agent as stateless, containerized service. With external state, any instance can handle any request → enables serverless platforms like **Cloud Run** or managed **Vertex AI Agent Engine Runtime** to scale automatically.
- **Asynchronous Processing**: for long-running tasks, offload work using event-driven patterns. Keeps agent responsive while complex jobs process in background. On Google Cloud: web service publishes tasks to **Pub/Sub**, which triggers Cloud Run service for async processing.
- **Externalized State Management**: since LLMs stateless, persisting memory externally non-negotiable. Highlights architectural choice — **Vertex AI Agent Engine** provides built-in durable Session and memory service. **Cloud Run** offers flexibility to integrate directly with **AlloyDB** or **Cloud SQL**.

**Balancing Competing Goals**: scaling involves balancing three competing goals: **speed, reliability, cost**.
- **Speed (Latency)**: design to work in parallel, aggressively cache results, use smaller efficient models for routine tasks.
- **Reliability (Handling Glitches)**: agents must handle temporary failures. Failed call → automatically retry, ideally with **exponential backoff**. Requires designing **"safe-to-retry"** (idempotent) tools to prevent bugs like duplicate charges.
- **Cost**: shorten prompts, use cheaper models for easier tasks, send requests in groups (**batching**).

#### Managing Risk: The Security Response Playbook

Because agent can act on its own, need playbook for rapid containment. When threat detected, response should follow clear sequence: **contain, triage, resolve**.

- **Immediate containment**: priority = stop the harm. Typically with **"circuit breaker"** — feature flag to instantly disable affected tool.
- **Triage**: with threat contained, suspicious requests routed to human-in-the-loop (HITL) review queue to investigate exploit's scope and impact.
- **Permanent resolution**: team develops a patch — like updated input filter or system prompt — and deploys through automated CI/CD pipeline, ensuring fix fully tested before blocking exploit for good.

### Evolve: Learning from Production

While "Act" phase = system's immediate tactical reflexes, **Evolve** phase = long-term strategic improvement. Begins by looking at patterns and trends collected in observability data and asking crucial question: **"How do we fix the root cause so this problem never happens again?"**

Move from reacting to production incidents to proactively making agent smarter, more efficient, safer. Turn raw data from "Observe" phase into durable improvements in agent's architecture, logic, behavior.

#### The Engine of Evolution: An Automated Path to Production

Insight from production only valuable if you can act quickly. Observing 30% of users fail at specific task = useless if takes team six months to deploy fix.

**Automated CI/CD pipeline** built in pre-production = most critical component of operational loop. Engine that powers rapid evolution. Fast reliable path to production allows closing loop between observation and improvement in **hours or days**, not weeks/months.

When you identify potential improvement — refined prompt, new tool, updated safety guardrail — process should be:
1. **Commit the Change**: proposed improvement committed to version-controlled repository
2. **Trigger Automation**: commit automatically triggers CI/CD pipeline
3. **Validate Rigorously**: pipeline runs full suite of unit tests, security scans, agent quality evaluation suite against updated datasets
4. **Deploy Safely**: once validated, change deployed to production using safe rollout strategy

Automated workflow transforms evolution from slow high-risk manual project into fast, repeatable, data-driven process.

#### The Evolution Workflow: From Insight to Deployed Improvement

1. **Analyze Production Data**: identify trends in user behavior, task success rates, security incidents from production logs
2. **Update Evaluation Datasets**: transform production failures into tomorrow's test cases, augmenting golden dataset
3. **Refine and Deploy**: commit improvements to trigger automated pipeline — refining prompts, adding tools, updating guardrails

Creates virtuous cycle where agent continuously improves with every user interaction.

> **An Evolve Loop in Action**: Retail agent's logs (**Observe**) show 15% of users receive error when asking for 'similar products.' Product team **Acts** by creating high-priority ticket. **Evolve** phase begins: production logs used to create new failing test case for evaluation dataset. AI Engineer refines agent's prompt and adds new more robust tool for similarity search. Change committed, passes now-updated evaluation suite in CI/CD pipeline, safely rolled out via canary deployment, resolving user issue in **under 48 hours**.

### Evolving Security: The Production Feedback Loop

While foundational security/responsibility framework established in pre-production, work never truly finished. Security is not static checklist; dynamic continuous process of adaptation. Production environment = ultimate testing ground.

**Observe → Act → Evolve** loop critical for security. Direct extension of evolution workflow:

1. **Observe**: monitoring/logging systems detect new threat vector. Could be novel prompt injection technique bypassing filters, or unexpected interaction leading to minor data leak.
2. **Act**: immediate security response team contains threat (Section 4.2).
3. **Evolve**: crucial step for long-term resilience. Security insight fed back into development lifecycle:
   - **Update Evaluation Datasets**: new prompt injection attack added as permanent test case
   - **Refine Guardrails**: Prompt Engineer or AI Engineer refines system prompt, input filters, tool-use policies to block new attack vector
   - **Automate and Deploy**: engineer commits change, triggering full CI/CD pipeline. Updated agent rigorously validated against newly expanded evaluation set and deployed, closing vulnerability

Powerful feedback loop where every production incident makes agent stronger and more resilient. Transforms security posture from defensive stance to **continuous proactive improvement**.

(More on Responsible AI and securing AI Agentic Systems: **Google's Approach for Secure AI Agents** and **Google Secure AI Framework (SAIF)**.)

### Beyond Single-Agent Operations

Mastered operating individual agents in production at high velocity. But as orgs scale to dozens of specialized agents — each built by different teams with different frameworks — new challenge: **these agents can't collaborate**. Standardized protocols can transform isolated agents into interoperable ecosystem, unlocking exponential value through agent collaboration.

## A2A — Reusability and Standardization

Built dozens of specialized agents across organization. Customer service team has support agent. Analytics built forecasting system. Risk management created fraud detection. Problem: **these agents can't talk to each other** — different frameworks, projects, or different clouds altogether.

Isolation creates massive inefficiency. Every team rebuilds same capabilities. Critical insights stay trapped in silos. Need **interoperability** — ability for any agent to leverage any other agent's capabilities, regardless of who built it or what framework.

Principled approach to standardization built on two distinct but complementary protocols:
- **Model Context Protocol (MCP)** (covered in **Agent Tools and Interoperability with MCP**) provides universal standard for **tool integration** but not sufficient for complex stateful collaboration required between intelligent agents
- **Agent2Agent (A2A)** protocol, now governed by **Linux Foundation**, designed to solve this

Distinction critical: simple stateless function (fetching weather data, querying database) → tool that speaks **MCP**. Delegating complex goal ("analyze last quarter's customer churn and recommend three intervention strategies") → intelligent partner that can reason, plan, act autonomously via **A2A**.

> **In short**: MCP lets you say *"Do this specific thing"*; A2A lets you say *"Achieve this complex goal."*

### A2A Protocol: From Concept to Implementation

A2A protocol designed to break down organizational silos and enable seamless collaboration. Scenario: fraud detection agent spots suspicious activity. To understand full context, needs data from separate transaction analysis agent. Without A2A, human analyst must manually bridge gap (could take hours). With A2A, agents collaborate automatically, resolving issue in minutes.

First step of collaboration = discovering right agent to delegate to → made possible through **Agent Cards**: standardized JSON specifications acting as "business card" for each agent. Agent Card describes what agent can do, security requirements, skills, how to reach it (URL), allowing any other agent in ecosystem to dynamically discover peers.

Example Agent Card:

```python
{
    "name": "check_prime_agent",
    "version": "1.0.0",
    "description": "An agent specialized in checking whether numbers are prime",
    "capabilities": {},
    "securitySchemes": {
        "agent_oauth_2_0": {
            "type": "oauth2",
        }
    },
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["application/json"],
    "skills": [
        {
            "id": "prime_checking",
            "name": "Prime Number Checking",
            "description": "Check if numbers are prime using efficient algorithms",
            "tags": ["mathematical", "computation", "prime"]
        }
    ],
    "url": "http://localhost:8001/a2a/check_prime_agent"
}
```

Adopting protocol doesn't require architectural overhaul. Frameworks like ADK simplify significantly. Make existing agent A2A-compatible with single function call, automatically generating AgentCard and making available on network:

```python
# Example using ADK: Exposing an agent via A2A
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Your existing agent
root_agent = Agent(
    name='hello_world_agent',
    # ... your agent code ...
)

# Make it A2A-compatible
a2a_app = to_a2a(root_agent, port=8001)

# Serve with uvicorn
# uvicorn agent:a2a_app --host localhost --port 8001
# Or serve with Agent Engine
# from vertexai.preview.reasoning_engines import A2aAgent
# from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
# a2a_agent = A2aAgent(
#     agent_executor_builder=lambda: A2aAgentExecutor(agent=root_agent)
# )
```

Once exposed, any other agent can consume it by referencing AgentCard. E.g. customer service agent can query remote product catalog agent without needing to know internal workings:

```python
# Example using ADK: Consuming a remote agent via A2A
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

prime_agent = RemoteA2aAgent(
    name="prime_agent",
    description="Agent that handles checking if numbers are prime.",
    agent_card="http://localhost:8001/a2a/check_prime_agent/.well-known/agent-card.json"
)
```

Unlocks powerful **hierarchical compositions**. Root agent configured to orchestrate both local sub-agent for simple task + remote specialized agent via A2A → more capable system:

```python
# Example using ADK: Hierarchical agent composition
# ADK Local sub-agent for dice rolling
roll_agent = Agent(
    name="roll_agent",
    instruction="You are an expert at rolling dice."
)

# ADK Remote A2A agent for prime checking
prime_agent = RemoteA2aAgent(
    name="prime_agent",
    agent_card="http://localhost:8001/.well-known/agent-card.json"
)

# ADK Root orchestrator combining both
root_agent = Agent(
    name="root_agent",
    instruction="""Delegate rolling dice to roll_agent, prime checking
to prime_agent.""",
    sub_agents=[roll_agent, prime_agent]
)
```

Enabling autonomous collaboration introduces two non-negotiable technical requirements:
1. **Distributed Tracing**: every request carries unique trace ID, essential for debugging and maintaining coherent audit trail across multiple agents
2. **Robust State Management**: A2A interactions inherently stateful, requiring sophisticated persistence layer for tracking progress and ensuring transactional integrity

**A2A is best suited for formal, cross-team integrations** that require durable service contract. For tightly coupled tasks within single application, **lightweight local sub-agents often remain a more efficient choice**. As ecosystem matures, new agents should be built with native support for both protocols, ensuring every new component immediately discoverable, interoperable, reusable.

### How A2A and MCP Work Together

> **Figure 4**: A2A and MCP collaboration with a single glance
> - User interacts with router Agent (Client/Router Agent)
> - Specialized Agents in Agent Registry perform task (Server/Specialized Agent A, B, C) connected via A2A
> - Specialized Agents user tools leveraging Tool Registry: API HUB Z (Tool X, Tool Y), MCP Server X (Tool X, Tool Y), MCP Server Y (Tool 1, Tool 2)

A2A and MCP are not competing standards; **complementary protocols** designed to operate at different levels of abstraction. Distinction depends on what an agent is interacting with.
- **MCP** = domain of **tools and resources** — primitives with well-defined structured inputs/outputs (calculator, database API)
- **A2A** = domain of other **agents** — autonomous systems that can reason, plan, use multiple tools, maintain state to achieve complex goals

Most powerful agentic systems use both protocols in **layered architecture**. Application might primarily use A2A to orchestrate high-level collaboration between intelligent agents, while each of those agents internally uses MCP to interact with own specific set of tools/resources.

Practical analogy = auto repair shop staffed by autonomous AI agents:
1. **User-to-Agent (A2A)**: customer uses A2A to communicate with "Shop Manager" agent: "My car is making a rattling noise."
2. **Agent-to-Agent (A2A)**: Shop Manager engages in multi-turn diagnostic conversation, then delegates task to specialized "Mechanic" agent via A2A.
3. **Agent-to-Tool (MCP)**: Mechanic agent needs specific actions. Uses MCP to call specialized tools: runs `scan_vehicle_for_error_codes()` on diagnostic scanner, queries repair manual database with `get_repair_procedure()`, operates platform lift with `raise_platform()`.
4. **Agent-to-Agent (A2A)**: After diagnosing, Mechanic determines part needed. Uses A2A to communicate with external "Parts Supplier" agent to inquire about availability and place order.

A2A facilitates higher-level conversational task-oriented interactions between customer, shop's agents, external suppliers. MCP provides standardized plumbing enabling mechanic agent to reliably use specific structured tools to do its job.

### Registry Architectures: When and How to Build Them

Why some orgs build registries while others don't? Scale and complexity. Fifty tools = manual configuration works. **Five thousand tools across teams/environments** → discovery problem demanding systematic solution.

**Tool Registry** uses protocol like MCP to catalog all assets, from functions to APIs. Instead of giving agents access to thousands, create curated lists. Three common patterns:
- **Generalist agents**: full catalog access, trading speed/accuracy for scope
- **Specialist agents**: predefined subsets for higher performance
- **Dynamic agents**: query registry at runtime to adapt to new tools

Primary benefit: **human discovery** — developers search for existing tools before building duplicates, security teams audit tool access, product owners understand agents' capabilities.

**Agent Registry** applies same concept to agents using formats like A2A's AgentCards. Helps teams discover and reuse existing agents, reducing redundant work. Lays groundwork for automated agent-to-agent delegation (still emerging pattern).

Registries offer discovery and governance at cost of maintenance. Consider starting without one and only build when ecosystem's scale demands centralized management.

| Decision Framework for Registries | |
|---|---|
| **Tool Registry** | Build when tool discovery becomes bottleneck or security requires centralized auditing |
| **Agent Registry** | Build when multiple teams need to discover and reuse specialized agents without tight coupling |

## Putting It All Together: The AgentOps Lifecycle

Assemble pillars into single cohesive reference architecture. Lifecycle begins in **developer's inner loop** — phase of rapid local testing/prototyping to shape agent's core logic. Once change ready, enters formal pre-production engine where automated evaluation gates validate quality/safety against golden dataset. From there, safe rollouts release to production where comprehensive observability captures real-world data needed to fuel continuous evolution loop, turning every insight into next improvement.

For comprehensive walkthrough of operationalizing AI agents (evaluation, tool management, CI/CD standardization, effective architecture designs), watch **AgentOps: Operationalize AI Agents** video on official Google Cloud YouTube channel.

> **Figure 5**: AgentOps core capabilities, environments, and processes
> - **Cloud Infrastructure Environment**: Infrastructure as Code, Central cloud security, Central cloud observability, Central cloud billing, Central cloud Env/User governance
> - **Central Agent Code Templates**
> - **Data Lake / Mesh Environment**: Data Governance — Data storage/lineage/cataloging, Data Preparation, Agent Evaluation Data, Data Tools, RAG, Long-term Memory
> - **Development Environment**:
>   - Experimentation: Agent experimentations
>   - Development: AI Agent, AI Application (Backend/Frontend), Central cloud security, Context Management, Monitoring Mechanism
>   - AI Security: AI Model Gateway, Guardrails
> - **Staging Environment**: AI Agent / App Deployment, Automatic Test at Scale, Automatic Evaluation (Scale), Agent Simulation
> - **Production Environment**: AI Agent / App Deployment (A/B, etc.), Short-term Memory, Security/RAI/Agent Alerts & Vulnerabilities, AI Agent / App Serving, Observability/Logs, Monitoring
> - **AI Governance Environment**: Artifacts (Repositories, CI/CD Pipelines), Agents (Agent Registry, Agent Governance), Tools (Tool Registry, Tool Governance)

## Conclusion: Bridging the Last Mile with AgentOps

Moving AI prototype to production = organizational transformation requiring new operational discipline: **AgentOps**.

Most agent projects fail in "last mile" not due to technology, but because operational complexity of autonomous systems is underestimated. Guide maps path to bridge gap. Begins with establishing **People and Process** as foundation for governance. Next, **Pre-Production** strategy built on evaluation-gated deployment automates high-stakes releases. Once live, continuous **Observe → Act → Evolve** loop turns every user interaction into potential insight. Finally, **Interoperability** protocols scale system by transforming isolated agents into collaborative intelligent ecosystem.

Immediate benefits — preventing security breach, enabling rapid rollback — justify investment. But real value = **velocity**. Mature AgentOps practices allow teams to deploy improvements in hours, not weeks, turning static deployments into continuously evolving products.

### Your Path Forward

- **If you're starting out**: focus on fundamentals. Build first evaluation dataset, implement CI/CD pipeline, establish comprehensive monitoring. **Agent Starter Pack** = great place to start — creates production-ready agent project in minutes with foundations built-in.
- **If you're scaling**: elevate practice. Automate feedback loop from production insight to deployed improvement, standardize on interoperable protocols to build cohesive ecosystem, not just point solutions.

Next frontier = not just better individual agents, but orchestrating sophisticated multi-agent systems that learn and collaborate. Operational discipline of **AgentOps** = foundation that makes this possible.

Bridging the last mile = not the final step in a project, but the **first step in creating value**.

## Key Takeaways

- "Last mile" gap: 80% of effort is infrastructure/security/validation, not core agent intelligence
- Three foundational pillars: **Automated Evaluation, Automated Deployment (CI/CD), Comprehensive Observability**
- Specialized teams: Cloud Platform, Data Engineering, Data Science/MLOps, ML Governance + GenAI-specific (Prompt Engineers, AI Engineers, DevOps/App Devs)
- **Evaluation-Gated Deployment** = no agent reaches users without comprehensive evaluation. Two patterns: Manual "Pre-PR" + Automated In-Pipeline Gate
- CI/CD three phases: **Pre-Merge CI** (unit tests, linting, agent quality eval) → **Post-Merge Staging CD** (load/integration tests, dogfooding) → **Gated Production Deploy** (Product Owner sign-off)
- Powered by **IaC (Terraform)** + **Pytest** + **Secret Manager**
- **Safe rollout**: Canary, Blue-Green, A/B Testing, Feature Flags. Foundation = rigorous versioning of code, prompts, models, tools, memory, eval datasets
- Security = three layers: **Policy/SI (constitution)** + **Guardrails (input/output filtering, HITL)** + **Continuous Assurance (rigorous eval, RAI testing, red teaming)**
- Production loop: **Observe → Act → Evolve**
- **Observe** = three pillars (logs, traces, metrics) via Cloud Trace, Cloud Logging, Cloud Monitoring, ADK Cloud Trace integration
- **Act** = automated reflexes for system health (horizontal scaling, async via Pub/Sub, externalized state via AlloyDB/Cloud SQL, exponential backoff, idempotent tools, batching) + risk (contain via circuit breaker → triage HITL → resolve via CI/CD)
- **Evolve** = Analyze production data → augment golden dataset with failures → refine and deploy. Closes loop in **hours/days**, not weeks
- **MCP vs A2A**: MCP = "do this specific thing" (tools/resources); A2A = "achieve this complex goal" (autonomous agents)
- **Agent Cards** = JSON business card for agents (capabilities, security, skills, URL)
- A2A in ADK: `to_a2a()` exposes agent; `RemoteA2aAgent` consumes remote
- A2A best for **formal cross-team integrations**; lightweight local sub-agents for tightly coupled tasks. Build with native support for both
- Registries: build when scale demands. Tool Registry (MCP) catalogs assets; Agent Registry (A2A AgentCards) catalogs agents. Patterns: generalist, specialist, dynamic
- AgentOps lifecycle: developer inner loop → pre-production engine (eval gates) → safe rollout → production with observability fueling evolution loop
