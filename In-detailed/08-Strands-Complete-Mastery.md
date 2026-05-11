# Strands Agents SDK — Complete Mastery Deep-Dive

**Objective:** Every minute detail needed to master Strands framework for production systems and interview scenarios.

---

## Part 1: Foundation & Philosophy

### What Strands Is (and Isn't)

Strands ≠ LangChain. It's an **open-source Python framework** for multi-agent systems with model-driven orchestration.

```
LangChain:   Dev chains steps → Agent follows fixed pipeline
Strands:     LLM sees tools → Agent dynamically decides tool calls
```

**Core philosophy:**
- **Model-driven:** LLM is the orchestrator, not the developer
- **Autonomous:** Agent calls tools without explicit step-by-step coding
- **Memory-first:** Conversation history auto-managed
- **Multi-agent-native:** Built for agent-to-agent coordination
- **Framework-agnostic to frameworks:** Works with Bedrock, OpenAI, Anthropic, Ollama

### Strands vs Competitors

| Dimension | Strands | LangGraph | Bedrock Agents | LangChain |
|---|---|---|---|---|
| **Model decides flow** | Yes | No | No | No |
| **Memory auto-managed** | Yes | No | Yes | No |
| **Multi-agent patterns** | 4 (hierarchical, swarm, graph, A2A) | DAG only | Fixed workflow | Limited |
| **Tool definition** | Python docstring → auto-spec | Manual JSON | AWS OpenAPI | Manual JSON |
| **Deployment** | Docker, Lambda, K8s, local | DIY | AWS Lambda | DIY |
| **Real-time streaming** | Native (SSE, WebSocket) | Manual | Native | Limited |
| **Observability** | Built-in OTEL | LangSmith (paid) | CloudWatch | LangSmith (paid) |
| **Built-in tools** | 20+ (batteries-included) | 100+ (scattered) | Limited | 1000+ (scattered) |

**When to use Strands:**
- Multi-agent systems where LLM orchestrates dynamically
- Rapid prototyping (hour to agent)
- Real-time streaming needed
- Autonomous behavior (less developer code)

**When NOT to use Strands:**
- Fixed, deterministic workflow (use LangGraph)
- Compliance requires explicit DAG (use LangGraph)
- AWS-native architecture preferred (use Bedrock Agents)

---

## Part 2: Core Architecture

### Agent Model (Declarative)

**Strands agents are Python dataclasses:**

```python
from strands.agents import Agent
from strands.memory import ShortTermMemory
from dataclasses import dataclass

@dataclass
class SearchAgent(Agent):
    """Search specialist agent"""
    name: str = "SearchAgent"
    model: str = "claude-3-5-sonnet-20241022"
    description: str = "Searches documents and web"
    system_prompt: str = "You are a research expert..."
    memory: ShortTermMemory = ShortTermMemory()
    tools: list = None  # Auto-populated from methods
```

**Key invariants:**
1. Agents are instances, not classes
2. Tools are Python methods (docstring parsed for LLM spec)
3. Memory is auto-managed (no manual save/load)
4. LLM decides what tools to call (no dev orchestration)

### Execution Loop (How Tool Calls Work)

```
┌─────────────────────────────────────────────────────────┐
│ 1. invoke_agent("What's on my calendar?")              │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│ 2. Agent memory → get_context()                        │
│    (last 10 turns, embedded in prompt)                 │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│ 3. LLM call: system_prompt + context + tool specs      │
│    LLM decides: "I should call fetch_calendar tool"    │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│ 4. Tool execution: fetch_calendar(date='today')        │
│    (Strands matches LLM choice to actual Python method)│
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│ 5. Tool result → add to context                        │
│    "Returned: [9am Meeting, 3pm Standup]"             │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│ 6. Continue LLM reasoning (1-3 tool calls per turn)    │
│    Or return final response                             │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│ 7. Save to memory: (turn, response)                    │
│    Ready for next invoke (same session)                 │
└─────────────────────────────────────────────────────────┘
```

**Tool call detection:** Strands matches LLM's tool_name + arguments to Python function signature.

**Failure mode:** Tool raises exception → LLM sees error → attempts retry or fallback.

---

## Part 3: Memory System (Deep-Dive)

### Short-Term Memory (Session-Scoped)

Auto-managed conversation history:

```python
from strands.memory import ShortTermMemory

memory = ShortTermMemory(
    max_turns=10,           # Keep last 10 conversation turns
    max_tokens=4096,        # OR token limit (whichever hits first)
    prune_strategy="sliding_window"  # Remove oldest when limit exceeded
)

# Automatic updates on every agent call
agent.memory.add_message("user", "Find Python tutorials")
agent.memory.add_message("assistant", "Searching...")

# Retrieve context for LLM
messages = agent.memory.get_context()
# [
#   {"role": "user", "content": "Find Python tutorials"},
#   {"role": "assistant", "content": "Searching..."},
# ]
```

**Automatic features:**
- Turn tracking (preserves conversation order)
- Token counting (knows when to prune)
- Role inference (user vs assistant)
- No manual state management

**Prune strategy (when max_turns exceeded):**
```
max_turns = 10
Turns: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 arrives]
→ Drop turn 1
→ Keep: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Downside: oldest context lost
Mitigation: store important facts in long-term memory before pruning
```

### Long-Term Memory (Multi-Session)

Persist knowledge across separate conversations:

```python
from strands.memory import LongTermMemory

ltm = LongTermMemory(
    storage="redis",  # Or "postgres", "dynamodb"
    namespace="user_123",  # Isolation key
    ttl=2592000  # 30 days (seconds)
)

# Query past conversations
past_context = ltm.retrieve(
    query="What did user prefer last time?",
    top_k=3,
    relevance_threshold=0.7
)
# Returns: [
#   {"turn_id": 456, "content": "User preferred async patterns", "score": 0.92},
#   ...
# ]

# Store facts learned in current session
ltm.store({
    "namespace": "/user_123/preferences",
    "facts": ["Prefers async patterns", "Works in fintech", "Hates SQL"],
    "metadata": {"source": "session_789"}
})
```

**Storage backends:**
- `redis`: Fast, in-memory (good for sessions, TTL auto-cleanup)
- `postgres`: Durable, queryable (good for long-term)
- `dynamodb`: AWS-native, scale to any size
- Custom: Implement MemoryBackend interface

**Retrieval strategies:**
1. **Vector similarity** (embedding-based): "What did we discuss about X?"
2. **Keyword match**: Exact string matching
3. **Metadata filter**: by timestamp, source, category
4. **Hybrid**: Vector + keyword

---

## Part 4: Tool System (Complete)

### Tool Definition (Docstring → LLM Spec)

**Strands auto-generates LLM tool specs from Python docstrings:**

```python
from strands.tools import Tool
from typing import Optional

class InvestmentCalculator(Tool):
    """Calculate ROI for investment scenarios"""
    
    name: str = "calculate_roi"
    description: str = "Calculate return on investment"
    
    def execute(
        self,
        investment: float,
        returns: float,
        years: Optional[int] = None
    ) -> dict:
        """
        Calculate return on investment.
        
        Args:
            investment: Initial investment amount (USD)
            returns: Final returns amount (USD)
            years: Number of years (optional, for annualized ROI)
        
        Returns:
            dict with keys:
            - roi_percent: ROI as percentage
            - roi_absolute: Absolute return amount
            - annualized_roi: Annualized return (if years provided)
        
        Raises:
            ValueError: If investment is 0 or negative
        """
        if investment <= 0:
            raise ValueError("Investment must be positive")
        
        roi = ((returns - investment) / investment) * 100
        result = {
            "roi_percent": roi,
            "roi_absolute": returns - investment
        }
        
        if years and years > 0:
            annualized = ((returns / investment) ** (1 / years) - 1) * 100
            result["annualized_roi"] = annualized
        
        return result
```

**What LLM sees:**
```
Tool: calculate_roi
Description: Calculate return on investment
Parameters:
  - investment (required, float): Initial investment amount (USD)
  - returns (required, float): Final returns amount (USD)
  - years (optional, int): Number of years (optional, for annualized ROI)
Returns: dict with keys: roi_percent, roi_absolute, annualized_roi
```

**Key mechanism:** Strands parses Python docstring → generates OpenAI tool spec format → sends to LLM.

### Built-in Tools (20+)

```python
from strands.tools import (
    # Web & Search
    WebSearch,              # Google, Bing, DuckDuckGo
    NewsSearch,             # News articles
    ImageSearch,            # Image retrieval
    
    # Documents & Files
    DocumentSearch,         # Vector search in PDFs
    FileSystem,             # Read/write files (sandboxed)
    PDFExtractor,           # Extract tables/text from PDFs
    
    # Computing
    Calculator,             # Math expressions
    CodeExecutor,           # Python/JS execution
    
    # Data
    DatabaseQuery,          # SQL execution (parameterized)
    ExcelAnalyzer,          # Read/analyze Excel
    CSVAnalyzer,            # CSV processing
    
    # Communication
    EmailClient,            # Send emails (SMTP)
    SlackClient,            # Post messages, read channels
    
    # Integration
    GitHubClient,           # List repos, create issues
    JiraClient,             # Query/create tickets
    
    # Utilities
    DateTimeUtils,          # Date/time operations
    Summarizer,             # Summarize text
    Translator,             # Translate text
    WebScraper,             # Parse HTML
)

agent = Agent(
    model="claude-3-5-sonnet",
    tools=[
        WebSearch(),
        Calculator(),
        DatabaseQuery(connection_string="postgresql://..."),
        EmailClient(smtp_server="...", password="..."),
    ]
)
```

### Custom Tool Implementation

**Three ways to add custom tools:**

#### Method 1: Tool Subclass

```python
from strands.tools import Tool

class CustomerLookup(Tool):
    name = "lookup_customer"
    description = "Find customer by ID or email"
    
    def __init__(self, db_connection):
        super().__init__()
        self.db = db_connection
    
    def execute(self, customer_id: int = None, email: str = None) -> dict:
        """
        Look up customer information.
        
        Args:
            customer_id: Customer ID number (optional)
            email: Customer email address (optional)
        
        Returns:
            dict with customer data or empty dict if not found
        """
        if customer_id:
            return self.db.query("SELECT * FROM customers WHERE id = %s", customer_id)
        elif email:
            return self.db.query("SELECT * FROM customers WHERE email = %s", email)
        else:
            raise ValueError("Must provide customer_id or email")

# Usage
db = psycopg2.connect("postgresql://...")
agent.tools = [CustomerLookup(db)]
```

#### Method 2: Function Decorator

```python
from strands.tools import tool

@tool(name="weather")
def get_weather(city: str, units: str = "celsius") -> dict:
    """
    Get current weather for a city.
    
    Args:
        city: City name (e.g., 'San Francisco')
        units: Temperature units ('celsius' or 'fahrenheit')
    
    Returns:
        dict with temp, humidity, condition
    """
    # Implementation
    response = requests.get(f"https://api.weather.com/current?city={city}")
    return response.json()

agent.tools = [get_weather]
```

#### Method 3: Agent Method

```python
class ResearchAgent(Agent):
    name = "researcher"
    
    def search_arxiv(self, query: str, limit: int = 5) -> list:
        """
        Search arXiv for papers.
        
        Args:
            query: Search term (e.g., 'transformer attention')
            limit: Max results (default 5)
        
        Returns:
            list of papers with title, authors, date
        """
        # Implementation
        response = requests.get(f"https://arxiv.org/...", params={"q": query})
        return response.json()[:limit]
```

**Strands auto-discovers methods with docstrings as tools.**

### Tool Error Handling

**What happens when tool fails:**

```python
class Risky Tool(Tool):
    def execute(self, user_id: int):
        # Might fail: database down, invalid user_id, etc.
        response = self.api.fetch_user(user_id)
        return response

# LLM sees error in tool result:
# "Tool returned error: Connection timeout after 5 seconds"
# → LLM can:
#   1. Retry tool with different params
#   2. Use fallback tool
#   3. Ask user for clarification
#   4. Return error message to user
```

**Best practice:**
```python
def execute(self, user_id: int) -> dict:
    try:
        response = self.api.fetch_user(user_id)
        return {"success": True, "data": response}
    except ConnectionError as e:
        return {"success": False, "error": "API unavailable", "retry": True}
    except ValueError as e:
        return {"success": False, "error": "Invalid user ID", "retry": False}
```

---

## Part 5: Multi-Agent Patterns (4 Patterns)

### Pattern 1: Agents as Tools (Hierarchical)

Agents can call other agents like tools:

```python
from strands.agents import Agent, AgentPool

class ResearchAgent(Agent):
    name = "researcher"
    description = "Gathers background information"
    
    def __init__(self):
        super().__init__(
            model="claude-3-5-sonnet",
            system_prompt="Find accurate information from reliable sources."
        )

class AnalysisAgent(Agent):
    name = "analyst"
    description = "Analyzes research findings"
    
    def __init__(self):
        super().__init__(
            model="claude-3-5-sonnet",
            system_prompt="Analyze findings critically. Identify patterns and risks."
        )

class ReportAgent(Agent):
    name = "writer"
    description = "Writes executive reports"
    
    def __init__(self):
        super().__init__(
            model="claude-3-5-sonnet",
            system_prompt="Write clear, concise executive summaries."
        )
    
    def __post_init__(self):
        # Other agents become tools
        self.tools = [
            ResearchAgent(),
            AnalysisAgent(),
        ]

# Execution flow:
coordinator = ReportAgent()
result = coordinator.run("Write a report on AI market trends in 2025")

# What happens:
# 1. ReportAgent → "I need research. Call researcher."
# 2. ResearchAgent executes → returns findings
# 3. ReportAgent → "Analyze these findings."
# 4. AnalysisAgent executes → returns analysis
# 5. ReportAgent → writes final report
```

**Hierarchy levels:**
- Top: Coordinator agent (decides flow)
- Middle: Specialist agents (execute work)
- Leaf: Built-in tools (calculator, web search)

**Gotcha:** Each agent = separate memory instance. No shared context. Pass data via tool results.

### Pattern 2: Swarm (Peer-to-Peer)

All agents at same level; router coordinates:

```python
from strands.swarm import Swarm

class DataCollector(Agent):
    name = "collector"
    tools = [WebSearch(), DatabaseQuery()]

class Processor(Agent):
    name = "processor"
    tools = [Calculator(), DataTransformer()]

class Reporter(Agent):
    name = "reporter"
    tools = [FileWriter(), EmailClient()]

swarm = Swarm(
    agents=[DataCollector(), Processor(), Reporter()],
    model="claude-3-5-sonnet-20241022",
    routing_strategy="semantic"  # Route based on agent description match
)

result = swarm.run("Analyze Q3 sales trends and email report to execs")

# What happens:
# 1. Router reads: "Analyze Q3 sales"
# 2. Routes to DataCollector → fetches data
# 3. Router sees data ready
# 4. Routes to Processor → processes
# 5. Router sees report ready
# 6. Routes to Reporter → sends email

# Each agent runs in parallel (async)
```

**When to use swarm:**
- Tasks naturally decompose into parallel steps
- No ordering required
- Each agent is specialized

**Gotcha:** Swarm doesn't guarantee order. Use pattern 3 (graph) if order critical.

### Pattern 3: Graph (Deterministic DAG)

Explicit workflow for structured problems (like LangGraph):

```python
from strands.graph import AgentGraph, Node, Edge, Condition

graph = AgentGraph()

# Define nodes
validate_node = Node(
    agent=ValidatorAgent(),
    name="validate_input",
    description="Validate input data"
)

process_node = Node(
    agent=ProcessorAgent(),
    name="process",
    description="Process validated data"
)

review_node = Node(
    agent=ReviewerAgent(),
    name="review",
    description="Review processed data"
)

publish_node = Node(
    agent=PublisherAgent(),
    name="publish",
    description="Publish to database"
)

# Add nodes
graph.add_node(validate_node)
graph.add_node(process_node)
graph.add_node(review_node)
graph.add_node(publish_node)

# Define edges with conditions
graph.add_edge(
    validate_node,
    process_node,
    condition=Condition(
        fn=lambda state: state.get("is_valid") == True,
        description="If input is valid"
    )
)

graph.add_edge(
    validate_node,
    Node(agent=ErrorAgent(), name="error_handler"),
    condition=Condition(
        fn=lambda state: state.get("is_valid") == False,
        description="If input invalid"
    )
)

graph.add_edge(process_node, review_node)

# Allow loops (e.g., revision)
graph.add_edge(
    review_node,
    process_node,
    condition=Condition(
        fn=lambda state: state.get("needs_revision") == True,
        description="If reviewer requests revision"
    )
)

graph.add_edge(
    review_node,
    publish_node,
    condition=Condition(
        fn=lambda state: state.get("approved") == True,
        description="If reviewer approves"
    )
)

# Execute
result = graph.run(input_data={"document": "..."})
```

**Graph benefits:**
- Explicit flow (easy to debug)
- Deterministic routing
- Clear error handling
- Supports loops and branching

**Trade-off:** More verbose than hierarchical, but guaranteed order.

### Pattern 4: A2A Protocol (Agent-to-Agent Across Frameworks)

Strands agents call Bedrock/LangGraph agents seamlessly:

```python
from strands.bedrock import BedrockBridge
from strands.agents import Agent

# Strands agent
strands_agent = Agent(
    name="data_analyst",
    model="claude-3-5-sonnet"
)

# Bedrock agent as tool
bedrock_bridge = BedrockBridge(
    agent_id="arn:aws:bedrock:us-east-1:123:agent/compliance-checker"
)

# Strands agent calls Bedrock agent
strands_agent.tools = [bedrock_bridge]

result = strands_agent.run("Analyze data and check compliance")

# What happens:
# 1. Strands decides: "Need compliance check"
# 2. Calls BedrockBridge → invokes Bedrock agent (via API)
# 3. Bedrock agent returns result
# 4. Strands continues reasoning
# 5. Returns final response
```

**A2A enables:**
- Cross-framework orchestration (Strands + Bedrock + LangGraph)
- Hybrid LLM selection (Sonnet for reasoning, Haiku for cost)
- Distributed agent networks
- Service-level separation (scale agents independently)

---

## Part 6: Model Support & Selection

### Supported Models

Strands works with **any LLM:**

```python
# Anthropic Claude (recommended for reasoning)
agent = Agent(model="claude-3-5-sonnet-20241022")

# OpenAI
agent = Agent(
    model="gpt-4-turbo",
    provider="openai",
    api_key="sk-..."
)

# AWS Bedrock
agent = Agent(
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    provider="bedrock",
    region="us-west-2"
)

# Open-source (Ollama/RunPod)
agent = Agent(
    model="llama-2-70b",
    provider="ollama",
    base_url="http://localhost:11434"
)

# Fallback chain
agent = Agent(
    model="claude-3-5-sonnet",  # Primary
    fallback_models=["gpt-4-turbo", "claude-3-5-haiku"],
    provider_fallback=["anthropic", "openai", "openai"]
)
```

### Model Switching (Mid-Conversation)

```python
agent = Agent(model="claude-3-5-haiku")  # Cheap by default

# Complex task detected
if task_complexity > 0.7:
    agent.model = "claude-3-5-sonnet"  # Switch to better model
    result = agent.run(query)
    agent.model = "claude-3-5-haiku"   # Switch back
else:
    result = agent.run(query)  # Use cheap model
```

**Cost optimization:**
- Haiku for routing, classification, summaries
- Sonnet for reasoning, creative tasks
- Opus for complex multi-step problems

**Strands auto-handles:**
- Prompt formatting per model (e.g., Claude system_prompt vs OpenAI system)
- Token counting (knows when to truncate)
- Rate limiting (respects per-model limits)
- Retries with backoff

---

## Part 7: Configuration & Customization

### Agent Configuration

```python
from strands.config import AgentConfig

config = AgentConfig(
    temperature=0.3,        # Lower = deterministic
    max_tokens=2048,        # Output limit
    top_p=0.9,              # Nucleus sampling
    frequency_penalty=0.1,  # Reduce repetition
    presence_penalty=0.0,   # Encourage new topics
    
    # Execution
    timeout=30,             # seconds per tool call
    max_retries=3,          # Retry failed tools
    retry_delay=1,          # exponential backoff: 1s, 2s, 4s
    
    # Tool behavior
    max_tool_calls_per_turn=5,  # Prevent infinite loops
    tool_call_budget=20,    # Max total tool calls
    
    # Memory
    max_turns=10,
    context_window=4096,
)

agent = Agent(
    name="analyst",
    config=config,
    system_prompt="..."
)
```

### Observability & Logging

```python
from strands.observability import Tracer, Logger

# OpenTelemetry export
tracer = Tracer(
    backend="otel",
    exporter="otlp",
    endpoint="http://localhost:4317"
)

agent = Agent(
    name="search_agent",
    tracer=tracer,
    log_level="DEBUG"
)

# Auto-captures:
# - Tool invocations (name, arguments, result, latency)
# - Model calls (prompt, completion, tokens)
# - Agent-to-agent handoffs (which agent called which)
# - Memory access (search queries, retrieval results)
# - Errors and retries (exception type, retry count)
# - Performance metrics (latency per turn)

# Access logs
logs = agent.get_logs()
# [
#   {"timestamp": "...", "level": "INFO", "event": "tool_call", "tool": "search", ...},
#   ...
# ]
```

---

## Part 8: Streaming & Real-Time Response

### SSE (Server-Sent Events)

Stream agent responses token-by-token:

```python
from strands.streaming import StreamAgent

agent = StreamAgent(name="streamer")

# Streaming callbacks
def on_token(token: str):
    print(token, end="", flush=True)  # Print immediately

def on_tool_call(tool_name: str, args: dict):
    print(f"\n[Tool: {tool_name}({args})]")

def on_tool_result(tool_name: str, result):
    print(f"[Result: {result}]")

response = agent.stream_run(
    query="Write a detailed market analysis",
    on_token=on_token,
    on_tool_call=on_tool_call,
    on_tool_result=on_tool_result,
)

# Output appears in real-time (like ChatGPT)
```

**Backend:** SSE endpoint returns event stream. Client reads line-by-line.

### WebSocket (Bi-Directional Real-Time)

```python
from strands.websocket import WebSocketAgent
import asyncio

agent = WebSocketAgent(name="chat")

async def handle_websocket(websocket):
    # Client connects
    async for user_message in websocket:
        # Stream response back
        async for chunk in agent.astream(user_message):
            await websocket.send(chunk)
    # Client disconnects

# Deploy as async web server
```

**Use case:** Real-time chat UI (like Claude.ai, ChatGPT).

### Controlling Stream Rate

```python
async def stream_with_delay(agent, query):
    """Stream with human-readable pacing"""
    async for token in agent.astream(query):
        print(token, end="", flush=True)
        await asyncio.sleep(0.01)  # 10ms per token
```

---

## Part 9: Integration Patterns

### With Bedrock Knowledge Bases

```python
from strands.agents import Agent
from strands.bedrock import BedrockKnowledgeBase

agent = Agent(name="doc_agent")

kb = BedrockKnowledgeBase(
    knowledge_base_id="arn:aws:bedrock:...",
    region="us-east-1"
)

# KB becomes a tool
agent.tools = [kb.retrieval_tool()]

result = agent.run("What does section 3.2 of our policy say?")
# → Agent calls KB.retrieve() → searches documents → returns relevant passages
```

### With Vector Databases (Pinecone, Weaviate, etc.)

```python
from strands.agents import Agent
from strands.integrations import PineconeDB

agent = Agent(name="kb_agent")

vector_db = PineconeDB(
    api_key="...",
    index_name="documents",
    namespace="legal"
)

agent.tools = [
    vector_db.similarity_search_tool(
        top_k=5,
        metric="cosine"
    )
]

result = agent.run("Find cases similar to X")
# → Agent calls vector_db.search() → returns similar documents
```

### With Databases (SQL)

```python
from strands.agents import Agent
from strands.integrations import PostgreSQL

agent = Agent(name="analyst")

db = PostgreSQL(
    connection_string="postgresql://user:pass@localhost/mydb"
)

agent.tools = [
    db.query_tool(
        schema_hint="users, transactions, accounts, products",
        max_rows=1000
    )
]

result = agent.run("How many fraudulent transactions last month?")
# → Agent generates SQL
# → Executes via query_tool()
# → Interprets results
```

**Strands benefits:**
- Agent generates SQL (not you)
- Schema provided as hint (agent knows what tables exist)
- Results automatically parsed and formatted

---

## Part 10: Deployment

### Local Development

```bash
pip install strands-agents

# Create agent file
cat > my_agent.py << 'EOF'
from strands import Agent
from strands.tools import WebSearch

agent = Agent(
    name="researcher",
    tools=[WebSearch()]
)

if __name__ == "__main__":
    response = agent.run("What's the latest in AI?")
    print(response)
EOF

# Run locally
python my_agent.py

# Or dev server
python -m strands dev --agent my_agent.py --port 8000
# POST http://localhost:8000/invoke {"query": "..."}
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agents/ agents/
COPY models/ models/

# Expose HTTP server
EXPOSE 8000

CMD ["python", "-m", "strands", "serve", "--agent", "agents/main.py", "--port", "8000"]
```

**Build & run:**
```bash
docker build -t my-agent:latest .
docker run -p 8000:8000 -e MODEL_NAME=claude-3-5-sonnet my-agent:latest

curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?"}'
```

### AWS Lambda Deployment

```python
from strands.lambda_handler import StrandsLambdaHandler

agent = Agent(
    name="lambda_agent",
    tools=[...]
)

handler = StrandsLambdaHandler(agent)

# AWS Lambda invokes: handler(event, context)
# event['query'] → agent → returns result
```

**Deploy:**
```bash
# Package agent + dependencies
pip install -r requirements.txt -t package/
cp my_agent.py package/
cd package && zip -r ../lambda.zip . && cd ..

# Upload to Lambda
aws lambda create-function \
  --function-name my-agent \
  --runtime python3.11 \
  --handler my_agent.handler \
  --zip-file fileb://lambda.zip \
  --timeout 60 \
  --memory-size 1024
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strands-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
    spec:
      containers:
      - name: agent
        image: my-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL
          value: "claude-3-5-sonnet-20241022"
        - name: MEMORY_BACKEND
          value: "redis"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: MAX_RETRIES
          value: "3"
        - name: TIMEOUT
          value: "30"
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Part 11: Limitations & Constraints

### Hard Constraints

```python
# 1. Context Window (per agent, not shared across agents)
short_term_max = 10  # Default max turns
context_window_max = 128_000  # Claude model limit

# Mitigation: summarize old turns before context limit

# 2. Model Cost (no native caching/batching)
# Each agent call = fresh LLM call (no prompt caching)
# Each tool call = separate inference

# Mitigation: batch requests, reuse sessions

# 3. Tool availability (all agents see all tools)
# Can't restrict tools per agent (design limitation)

# Mitigation: create separate agent instances per role

# 4. Determinism (temperature, randomness)
# LLM behavior varies run-to-run unless temperature=0

# Mitigation: use temperature=0 if reproducibility critical

# 5. Debugging (less visibility than LangGraph DAG)
# Can't step through tool calls interactively
# Rely on logs + OpenTelemetry traces

# Mitigation: enable debug logging, use tracer
```

### Soft Limits (Configurable)

| Limit | Default | Impact |
|---|---|---|
| max_turns | 10 | Older context pruned |
| max_tool_calls_per_turn | 5 | Prevents infinite loops |
| timeout | 30s | Long tool calls fail |
| max_retries | 3 | Tool failures escalate |

---

## Part 12: Production Patterns & Best Practices

### Pattern 1: Error Resilience

```python
from strands.agents import Agent
from strands.config import AgentConfig

config = AgentConfig(
    max_retries=3,
    retry_delay=1,  # Exponential: 1s, 2s, 4s
    timeout=30,
)

agent = Agent(config=config, tools=[...])

# Tool fail → retry → fail → retry → fail → raise error
# Agent catches error → asks LLM what to do
# LLM can: retry with different params, use fallback tool, ask user
```

### Pattern 2: Cost Optimization

```python
config = AgentConfig(
    temperature=0,  # Deterministic (avoid re-generation)
    max_tokens=1024,  # Output limit
    max_tool_calls_per_turn=3,  # Prevent runaway
)

# Use cheaper model by default
agent = Agent(model="claude-3-5-haiku", config=config)

# Upgrade for complex queries
def run_with_auto_upgrade(query):
    result = agent.run(query)
    if result.confidence < 0.5:  # Low confidence
        agent.model = "claude-3-5-sonnet"  # Upgrade
        result = agent.run(query)  # Retry
        agent.model = "claude-3-5-haiku"  # Downgrade
    return result
```

### Pattern 3: Memory Persistence

```python
from strands.memory import LongTermMemory

# Persistent backend (Redis for fast, Postgres for durable)
ltm = LongTermMemory(storage="postgres", namespace="user_123")

agent = Agent(
    model="claude-3-5-sonnet",
    tools=[...]
)

# Every session, load user context
context = ltm.retrieve("What is the user's background?", top_k=3)

# Run agent with context
response = agent.run(query)

# After session, save important facts
ltm.store({
    "preferences": extract_preferences(response),
    "learned_facts": extract_facts(response),
})
```

### Pattern 4: Multi-Agent Orchestration

```python
from strands.swarm import Swarm

class DocumentAgent(Agent):
    name = "document_specialist"
    tools = [DocumentSearch()]

class CodeAgent(Agent):
    name = "code_specialist"
    tools = [CodeExecutor()]

class SummaryAgent(Agent):
    name = "summary_specialist"
    tools = [Summarizer()]

swarm = Swarm(
    agents=[DocumentAgent(), CodeAgent(), SummaryAgent()],
    routing_strategy="semantic"  # Route by agent description
)

# Route complex query to right agents
result = swarm.run("Explain this code and document it")
# → CodeAgent explains code
# → DocumentAgent prepares docs
# → SummaryAgent creates summary
```

---

## Part 13: Interview Questions & Answers

### Q1: Why Strands over LangChain for multi-agent?

**Answer:**
```
LangChain: Sequential chains (A → B → C)
Strands: Model-driven (LLM decides A, then B, then C)

Strands advantages:
1. Autonomous: LLM decides what to do (more flexible)
2. Memory auto-managed (no manual state dict)
3. 20+ built-in tools (batteries included)
4. Tool definition via docstring (simpler than manual JSON)
5. Multi-agent patterns native (4 patterns built-in)
6. Real-time streaming (SSE, WebSocket)
7. Less boilerplate (LangChain is verbose)

Trade-off: Less control. If you need rigid DAG → LangGraph.
```

### Q2: Design a research agent that cites sources. How?

**Answer:**
```
Architecture:
1. Define tools:
   - WebSearch (returns URL + snippet)
   - DocumentSearch (returns document path + excerpt)
   - Summarizer (summarizes + returns source)

2. System prompt:
   "Always cite your sources. Include URL or document path."

3. Tool result handling:
   Tool returns: {"content": "...", "source": "https://..."}
   Agent formats: "According to [source], ..."

4. Long-term memory:
   Store used sources to avoid redundant searches

Example:
  Query: "What's new in transformers?"
  → Agent calls WebSearch → gets URLs
  → Agent reasons → selects 3 best sources
  → Response: "According to [source1], ... [source2], ... [source3]"
```

### Q3: Agent calls tool → tool fails. What happens?

**Answer:**
```
Sequence:
1. Agent decides: "Call fetch_user(id=999)"
2. Tool throws exception: "User 999 not found"
3. Strands catches error → formats as tool result:
   {"error": "User not found", "retry": False}
4. LLM sees error → decides:
   Option A: Retry with different ID
   Option B: Use fallback tool
   Option C: Tell user "User not found"
5. Agent continues

Config:
max_retries=3 → LLM can retry up to 3 times
timeout=30 → if tool takes >30s → timeout error

Best practice:
Tool should return dict with success flag:
  {"success": True, "data": ...}
  or
  {"success": False, "error": "...", "retry": True/False}
```

### Q4: Session ends after 8 hours in AgentCore. How to handle?

**Answer:**
```
Problem: Agent running in AgentCore (8h max), conversation needs to continue next day.

Solution: Long-term memory bridge
1. Before session ends (at 7:59h):
   - Summarize current turn
   - Extract facts/preferences
   - Store to LongTermMemory with namespace="/user_123/session_history"

2. Next session (new runtimeSessionId):
   - Agent starts fresh short-term memory
   - Agent retrieves summary from long-term: 
     "Previous session summary: We discussed X, resolved Y, next is Z"
   - Agent continues from turn 201 (looks seamless to user)

3. Memory retrieval:
   session_manager.retrieve(
       query="What did we discuss yesterday?",
       top_k=3
   )
   → Returns top 3 relevant facts from yesterday's session

Key: Long-term memory survives session end. Short-term memory doesn't.
```

### Q5: Strands agent calling Bedrock agent (A2A). Architecture?

**Answer:**
```
Setup:
1. Bedrock agent deployed (in AgentCore)
   ARN: arn:aws:bedrock:us-east-1:123:agent/compliance-checker

2. Strands agent defines bridge:
   bedrock_bridge = BedrockBridge(
       agent_id="arn:aws:bedrock:...",
       invoke_params={"model": "claude-sonnet"}
   )
   
   strands_agent.tools = [bedrock_bridge]

Execution:
1. Strands agent: "I need to check compliance. Call compliance-checker."
2. BedrockBridge: Invokes Bedrock agent via API
3. Bedrock agent: Runs in its own microVM, returns result
4. Strands agent: Sees result, continues reasoning
5. Response: Includes both Strands + Bedrock reasoning

Benefit:
- Isolation: Bedrock agent scales independently
- Specialization: Bedrock handles compliance (domain expert)
- Cost: Only pay for actual execution (not idle time)
- Framework: Strands orchestrates, Bedrock executes
```

---

## Part 14: Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| "Tool not found" | Tool method name doesn't match LLM call | Ensure Python method name matches tool name in docstring |
| "Tool timeout (30s)" | Tool call takes > 30s | Increase timeout in AgentConfig, or make tool faster |
| "Memory limit exceeded" | Context window full | Prune old turns, or increase max_tokens |
| "Rate limited" | Too many API calls | Add retry backoff, or batch requests |
| "No matching model" | Model string invalid | Check model name (e.g., claude-3-5-sonnet-20241022) |
| "OAuth token expired" | Credential refresh failed | Re-authenticate, or set longer TTL |

---

## Part 15: Production Checklist

```
[ ] Select appropriate model (Haiku/Sonnet by task)
[ ] Set temperature=0 if determinism required
[ ] Enable OpenTelemetry tracing (observability)
[ ] Test all tool failure modes (network down, timeout, invalid input)
[ ] Set max_retries and timeout values
[ ] Monitor token usage (catch runaway agents)
[ ] Implement input validation (prevent injection)
[ ] Use separate agent instances per user (memory isolation)
[ ] Store long-term memory in persistent backend (Redis/Postgres)
[ ] Test multi-agent coordination (hand-offs work)
[ ] Document system prompt (what behaviors are enabled)
[ ] Review tool definitions (no unintended side effects)
[ ] Set up alerting (high error rate, cost spikes)
[ ] Load test (concurrent users)
[ ] Plan session handoff for long-running conversations
```

---

**Created for:** Complete mastery before interview. Every detail needed to design, build, and operate production Strands agents.
