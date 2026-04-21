# AI Learning Repository

> Auto-generated index. Last updated: 2026-04-21 15:46 UTC
> 9 topic(s) · 49 subtopic file(s)

---

## Topics

### AI Infrastructure

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [The GPU: The "Factory Floor" of AI](AI-Infrastructure/01-GPU-Basics.md) | _Core Components_ |
| 2 | [Model Architecture & Loading Flow](AI-Infrastructure/02-Model-Architecture.md) | _The Loading Sequence_ |
| 3 | [RunPod Infrastructure: Pods vs. Serverless](AI-Infrastructure/03-RunPod-Infrastructure.md) | _Comparison Table_ |
| 4 | [Resource Optimization](AI-Infrastructure/04-Resource-Optimization.md) | _VRAM Calculation Rule of Thumb, Advanced Tactics_ |
| 5 | [Hybrid Training Workflow: Mac Mini + RunPod](AI-Infrastructure/05-Hybrid-Training-Workflow.md) | _The Pipeline, When to Use Each_ |
| 6 | [The Framework Handshake](AI-Infrastructure/06-Framework-Handshake.md) | _Core Tools, How They Connect_ |

### Agentic AI

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [What Are AI Agents](Agentic-AI/01-What-Are-AI-Agents.md) | _Definition, The Four Core Properties, Major Agent Architectures +3 more_ |
| 2 | [Agent Architecture and Mental Model](Agentic-AI/02-Agent-Architecture-Mental-Model.md) | _The Six-Layer Agent Stack, Single Query Flow (Step by Step), The ReAct Loop (Explicit Reasoning Pattern) +5 more_ |
| 3 | [LLM vs Agentic AI](Agentic-AI/03-LLM-vs-Agentic-AI.md) | _The Core Difference, What Plain LLMs Cannot Do, LLM vs Agent: Capabilities Matrix +5 more_ |
| 4 | [Tools and Tool-Calling](Agentic-AI/04-Tools-and-Tool-Calling.md) | _What Is a Tool?, Tool-Calling Protocol: How It Works, Tool Categories +6 more_ |
| 5 | [Memory and State in Agents](Agentic-AI/05-Memory-and-State.md) | _Four Types of Agent Memory, In-Context Memory, Sliding Window Strategy +8 more_ |
| 6 | [Planning and Reasoning](Agentic-AI/06-Planning-and-Reasoning.md) | _ReAct Pattern (Reasoning + Acting), Chain-of-Thought vs ReAct vs Plan-and-Execute, Tree-of-Thought: Branching Reasoning +7 more_ |
| 7 | [LangChain Core](Agentic-AI/07-LangChain-Core.md) | _Core Abstractions, LCEL: LangChain Expression Language, Building a ReAct Agent +9 more_ |
| 8 | [LangGraph Core](Agentic-AI/08-LangGraph-Core.md) | _What LangGraph Adds Over LangChain, Core Concepts, Full Working LangGraph Agent +9 more_ |
| 9 | [Multi-Agent Systems](Agentic-AI/09-Multi-Agent-Systems.md) | _Core Patterns, Supervisor Pattern (Most Common), Swarm Pattern +7 more_ |
| 10 | [Production and Observability](Agentic-AI/10-Production-and-Observability.md) | _LangSmith Tracing, Key Metrics to Monitor, Cost Management +5 more_ |
| 11 | [Interview Prep](Agentic-AI/11-Interview-Prep.md) | _Concept Questions, System Design Questions, Coding Challenges +4 more_ |

### Bedrock

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [AWS Bedrock — Foundation Models as a Service](Bedrock/01-AWS-Bedrock-Guide.md) | _Mental Model, Bedrock Core APIs, Bedrock Guardrails +4 more_ |
| 2 | [AWS Bedrock AgentCore — Production Agent Infrastructure](Bedrock/02-AWS-Bedrock-AgentCore.md) | _Mental Model, Architecture, Session Model +16 more_ |
| 3 | [Strands Agents SDK — Model-Driven AI Agents Framework](Bedrock/03-Strands-Agents-SDK.md) | _Core Architecture, Memory System (Auto-Managed), Tool System (20+ Built-in) +11 more_ |
| 4 | [AI Agent Frameworks Comparison — Interview Deep Dive](Bedrock/04-Agent-Frameworks-Comparison.md) | _The Three Production Frameworks You Need to Know, Side-by-Side Comparison, When to Use Each (Decision Tree) +6 more_ |
| 5 | [Bedrock Invoke Functions & Streaming — Complete Reference](Bedrock/05-Invoke-Functions-and-Streaming.md) | _Decision Tree: Which Invoke Function?, 1. invoke_model — Synchronous One-Shot, 2. invoke_model_with_response_stream — Streaming One-Shot +11 more_ |

### FastAPI

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [FastAPI: Overview and Core Concepts](FastAPI/01-Overview-And-Core-Concepts.md) | _What FastAPI is, ASGI vs WSGI, Path Operations (Route Decorators) +4 more_ |
| 2 | [FastAPI: Request and Response Handling](FastAPI/02-Request-And-Response.md) | _Path Parameters, Query Parameters, Request Body +6 more_ |
| 3 | [FastAPI: Authentication Patterns](FastAPI/03-Authentication.md) | _API Key Auth, HTTP Basic Auth, Bearer Token / JWT +4 more_ |
| 4 | [FastAPI: Connecting to a Frontend](FastAPI/04-Frontend-Integration.md) | _CORS, How a JS/React Frontend Calls FastAPI, Gradio / Streamlit Frontends +3 more_ |
| 5 | [FastAPI: Backend Management and Serverless Deployment](FastAPI/05-Serverless-And-RunPod.md) | _FastAPI on RunPod Pods (Always-On), Model Loading at Startup — Lifespan Pattern, Passing State to Routes +4 more_ |
| 6 | [FastAPI vs Flask: Detailed Comparison](FastAPI/06-FastAPI-vs-Flask.md) | _Architecture, Performance, Type System +6 more_ |

### Financial AI Compliance

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [AI in Financial Services — Compliance, Risk, and Patterns](Financial-AI-Compliance/01-AI-in-Financial-Services.md) | _The Compliance Landscape, PII in LLM Pipelines — Detection & Redaction, Audit Logging — The Compliance Trail +5 more_ |

### Prompt Engineering

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [Prompt Engineering — From Basics to Production](Prompt-Engineering/01-Prompt-Engineering.md) | _Core Mental Model, Prompt Anatomy, SYSTEM PROMPT (who are you + guardrails) +8 more_ |
| 2 | [Reasoning Techniques — CoT, Self-Consistency, Generate Knowledge, Least-to-Most](Prompt-Engineering/02-Reasoning-Techniques.md) | _Zero-Shot Chain-of-Thought (CoT), Few-Shot Chain-of-Thought, Auto-CoT (Automatic Chain-of-Thought) +6 more_ |
| 3 | [Agent Frameworks — ReAct, Tree of Thoughts, Reflexion, PAL](Prompt-Engineering/03-Agent-Frameworks.md) | _ReAct (Reason + Act), Tree of Thoughts (ToT), Reflexion +3 more_ |
| 4 | [Advanced Structural Techniques — Chaining, SoT, XML, Context Management](Prompt-Engineering/04-Advanced-Structural.md) | _Prompt Chaining, Skeleton-of-Thought (SoT), XML / Structured Prompt Architecture +3 more_ |
| 5 | [Meta-Prompting and Automated Prompt Engineering](Prompt-Engineering/05-Meta-and-Automation.md) | _Meta-Prompting, Automatic Prompt Engineer (APE), DSPy — Stanford's Automated Prompt Framework +4 more_ |
| 6 | [Provider Playbook — Claude, OpenAI, Gemini](Prompt-Engineering/06-Provider-Playbook.md) | _Claude (Anthropic), OpenAI (GPT series), Google Gemini +6 more_ |
| 7 | [Security and Adversarial Prompting — Injection, Jailbreaking, Defenses](Prompt-Engineering/07-Security-and-Adversarial.md) | _Prompt Injection, Jailbreaking, Defenses +5 more_ |

### RAG Architecture

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [RAG (Retrieval-Augmented Generation) — From Basics to Production](RAG-Architecture/01-RAG-Core-Patterns.md) | _The Core Formula, RAG Workflow — 5 Stages, Advanced RAG Patterns +2 more_ |

### RunPod

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [RunPod Platform Overview](RunPod/01-Overview-And-Platform.md) | _What Is RunPod, Core Value Proposition vs Hyperscalers, Two Compute Paradigms +3 more_ |
| 2 | [RunPod Product Components](RunPod/02-Product-Components.md) | _Secure Cloud vs Community Cloud, Pods, Serverless +3 more_ |
| 3 | [RunPod Python SDK](RunPod/03-Python-SDK.md) | _Installation and Setup, Handler Pattern — `runpod.serverless.start()`, Async Handler +7 more_ |
| 4 | [GPU Selection Guide](RunPod/04-GPU-Selection-Guide.md) | _Full GPU Catalog, VRAM Requirements by Model Size and Precision, GPU-to-Model Matching +5 more_ |
| 5 | [Serverless Deep Dive](RunPod/05-Serverless-Deep-Dive.md) | _Worker Lifecycle States, Worker Configuration, FlashBoot +8 more_ |
| 6 | [Model-GPU Interaction](RunPod/06-Model-GPU-Interaction.md) | _VRAM Formula, Two Phases of Inference, Memory Bandwidth Table and Impact on Decode +5 more_ |
| 7 | [Python Code Patterns](RunPod/07-Python-Code-Patterns.md) | _Complete Serverless Handler — Model at Module Level, OOM-Safe Handler with Recovery, Streaming Handler — TextIteratorStreamer + Background Thread +9 more_ |
| 8 | [Engineering Decisions](RunPod/08-Engineering-Decisions.md) | _Pods vs Serverless vs Bare GPU, When to Use Pods, When to Use Serverless +5 more_ |
| 9 | [Debugging and Pitfalls](RunPod/09-Debugging-And-Pitfalls.md) | _OOM Crashes, Cold Start Too Slow, Spot Pod Interruption +8 more_ |
| 10 | [RunPod Serverless Codebase Reference](RunPod/10-Serverless-Codebase-Reference.md) | _`FluxPipeline` and `pipe`, `torch` role in the handler, `torch.Generator` and deterministic generation +7 more_ |
| 11 | [ML Infrastructure Fundamentals](RunPod/11-ML-Infrastructure-Fundamentals.md) | _Why GPU over CPU, GPU hardware components, SIMD architecture +11 more_ |

### Vector Databases

| # | Subtopic | Covers |
|---|----------|--------|
| 1 | [Vector Databases — Choosing and Using at Scale](Vector-Databases/01-Vector-DB-Guide.md) | _Why Vector DB, Not Regular DB?, Core Concepts, Vector Database Comparison +5 more_ |

---

## Learning Log

| Date (UTC) | Topic | Subtopic | Action |
|---|---|---|---|
| 2026-04-21 15:46 UTC | Prompt-Engineering | Reasoning Techniques — CoT, Self-Consistency, Generate Knowledge, Least-to-Most | Added/Updated |
| 2026-04-21 15:46 UTC | Prompt-Engineering | Agent Frameworks — ReAct, Tree of Thoughts, Reflexion, PAL | Added/Updated |
| 2026-04-21 15:46 UTC | Prompt-Engineering | Advanced Structural Techniques — Chaining, SoT, XML, Context Management | Added/Updated |
| 2026-04-21 15:46 UTC | Prompt-Engineering | Meta-Prompting and Automated Prompt Engineering | Added/Updated |
| 2026-04-21 15:46 UTC | Prompt-Engineering | Provider Playbook — Claude, OpenAI, Gemini | Added/Updated |
| 2026-04-21 15:46 UTC | Prompt-Engineering | Security and Adversarial Prompting — Injection, Jailbreaking, Defenses | Added/Updated |
| 2026-04-17 20:01 UTC | root | AI Learning Repository | Added/Updated |
| 2026-04-17 20:01 UTC | Agentic-AI | What Are AI Agents | Added/Updated |
| 2026-04-17 20:01 UTC | Agentic-AI | Agent Architecture and Mental Model | Added/Updated |
| 2026-04-17 20:01 UTC | Agentic-AI | LLM vs Agentic AI | Added/Updated |

---

_Auto-generated by `scripts/update_readme.py`. Do not edit by hand._
_To add a topic: create a new folder. To add a subtopic: add a numbered `.md` file inside it._
