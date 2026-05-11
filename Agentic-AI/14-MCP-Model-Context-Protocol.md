# Model Context Protocol (MCP) — USB-C for AI

**MCP is the universal connector between LLM applications and external data/tools. One protocol. Any integration.**

---

## Part 1: Foundation

### What Is MCP?

MCP is an **open standard by Anthropic** defining how LLM applications (Claude Desktop, Claude Code, agents, chatbots) connect to external servers that expose tools and data.

**Analogy:** USB-C. Before USB-C, every device had proprietary connectors. USB-C: one port, any device. MCP: one protocol, any LLM app connects to any server.

### Why MCP Exists

**Without MCP:**
```
You build an agent.
Agent needs file access → write custom file handler
Agent needs Slack integration → write custom Slack handler
Agent needs database query → write custom SQL handler
Agent needs GitHub info → write custom GitHub handler
Each integration = custom code, custom bugs, custom maintenance.
```

**With MCP:**
```
You build an agent.
Agent needs file access → use MCP filesystem server
Agent needs Slack integration → use MCP Slack server
Agent needs database query → use MCP Postgres server
Agent needs GitHub info → use MCP GitHub server
All servers follow MCP protocol. Agent code = generic MCP client.
```

### MCP Architecture

```
┌─────────────────────────┐
│ LLM Application         │ (Host)
│ (agent, Claude Desktop) │
│                         │
│ ┌─────────────────────┐ │
│ │ MCP Client          │ │
│ │ (protocol layer)    │ │
│ └──────────┬──────────┘ │
└────────────┼────────────┘
             │ stdio/http
             │
┌────────────▼──────────────┐
│ MCP Server                │
│ (lightweight process)     │
│                           │
│ ┌─────────────────────┐   │
│ │ Tools               │   │ (what it exposes)
│ │ - query_database    │   │
│ │ - read_file         │   │
│ │ - call_slack_api    │   │
│ └─────────────────────┘   │
│                           │
│ ┌─────────────────────┐   │
│ │ Resources           │   │
│ │ - /docs/README.md   │   │
│ │ - /data/sales.csv   │   │
│ └─────────────────────┘   │
│                           │
│ ┌─────────────────────┐   │
│ │ Prompts             │   │
│ │ - write_sql_query   │   │
│ │ - analyze_data      │   │
│ └─────────────────────┘   │
└───────────────────────────┘
```

**Host:** LLM app (Claude Desktop, agent in AgentCore, Claude Code).
**Client:** Protocol layer inside the host.
**Server:** Lightweight process exposing tools, resources, prompts. Can run anywhere (local, container, cloud function).

---

## Part 2: Protocol Primitives

### Core Operations

MCP defines 6 core operations:

#### 1. `tools/list` — Discover Available Tools

```bash
# Request
curl -X POST http://localhost:3000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/list"
  }'

# Response
{
  "tools": [
    {
      "name": "query_database",
      "description": "Execute SQL query",
      "inputSchema": {
        "type": "object",
        "properties": {
          "sql": {"type": "string", "description": "SQL query"},
          "limit": {"type": "number", "description": "Max rows"}
        },
        "required": ["sql"]
      }
    },
    {
      "name": "read_file",
      "description": "Read file from disk",
      "inputSchema": {
        "type": "object",
        "properties": {
          "path": {"type": "string"}
        },
        "required": ["path"]
      }
    }
  ]
}
```

**Use case:** Agent discovers what tools are available at server startup.

#### 2. `tools/call` — Invoke a Tool

```bash
# Request
curl -X POST http://localhost:3000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "query_database",
      "arguments": {
        "sql": "SELECT * FROM users WHERE age > 30",
        "limit": 100
      }
    }
  }'

# Response
{
  "result": [
    {"id": 1, "name": "Alice", "age": 32},
    {"id": 2, "name": "Bob", "age": 35}
  ],
  "executionTime": 145
}
```

**Use case:** Agent executes tool and gets result.

#### 3. `resources/list` — List Data Resources

```bash
# Request
{
  "method": "resources/list"
}

# Response
{
  "resources": [
    {
      "uri": "file:///docs/api-guide.md",
      "name": "API Documentation",
      "mimeType": "text/markdown",
      "description": "Complete API reference"
    },
    {
      "uri": "database://sales/2024",
      "name": "Sales Data 2024",
      "mimeType": "application/json",
      "description": "Annual sales by region"
    }
  ]
}
```

**Use case:** Agent discovers what data resources are available.

#### 4. `resources/read` — Read a Resource

```bash
# Request
{
  "method": "resources/read",
  "params": {
    "uri": "file:///docs/api-guide.md"
  }
}

# Response
{
  "content": "# API Guide\n\nThe API supports...",
  "mimeType": "text/markdown"
}
```

**Use case:** Agent retrieves content of a specific resource.

#### 5. `prompts/list` — List Prompt Templates

```bash
# Response
{
  "prompts": [
    {
      "name": "sql_expert",
      "description": "System prompt for SQL query generation",
      "arguments": [
        {"name": "dialect", "description": "PostgreSQL, MySQL, etc."}
      ]
    },
    {
      "name": "data_analyst",
      "description": "System prompt for data analysis"
    }
  ]
}
```

**Use case:** Agent discovers available prompt templates.

#### 6. `prompts/get` — Retrieve a Prompt

```bash
# Request
{
  "method": "prompts/get",
  "params": {
    "name": "sql_expert",
    "arguments": {"dialect": "PostgreSQL"}
  }
}

# Response
{
  "messages": [
    {
      "role": "user",
      "content": "You are a PostgreSQL expert. Generate efficient, indexed SQL queries."
    }
  ]
}
```

**Use case:** Agent uses prompt template for context.

---

## Part 3: Building an MCP Server (Python)

### Setup

```bash
pip install mcp  # MCP SDK for Python
```

### Basic Server (Tools Only)

```python
from mcp.server import Server
from mcp.types import Tool
import json

server = Server(name="database-server")

# Define a tool
@server.tool()
def query_database(sql: str, limit: int = 100) -> dict:
    """
    Execute SQL query against PostgreSQL.
    
    Args:
        sql: SQL query string
        limit: Maximum rows to return
    
    Returns:
        Query results as JSON
    """
    # Implementation
    results = execute_sql(sql, limit)
    return {"rows": results, "count": len(results)}

@server.tool()
def read_file(path: str) -> str:
    """Read file from disk."""
    with open(path) as f:
        return f.read()

# Run server
if __name__ == "__main__":
    # Option 1: stdio transport (pipe to LLM app)
    server.run_async_stdio()
    
    # Option 2: HTTP transport (run as service)
    # server.run_http(host="localhost", port=3000)
```

### Server with Resources

```python
from mcp.server import Server
from mcp.types import Resource

server = Server(name="file-server")

@server.tool()
def search_files(query: str) -> list[str]:
    """Search for files matching query."""
    import glob
    return glob.glob(f"/data/**/*{query}*", recursive=True)

# Expose files as resources
@server.resource()
def get_resource(uri: str) -> str:
    """Read file by URI."""
    # uri = "file:///path/to/file.txt"
    path = uri.replace("file://", "")
    with open(path) as f:
        return f.read()

# List available resources
@server.resources()
def list_resources() -> list[Resource]:
    """List all exposed resources."""
    resources = []
    for path in glob.glob("/data/**/*.txt", recursive=True):
        resources.append(Resource(
            uri=f"file://{path}",
            name=path.split("/")[-1],
            mimeType="text/plain"
        ))
    return resources
```

### Server with Prompts

```python
from mcp.server import Server

server = Server(name="prompt-server")

@server.prompt()
def sql_expert(dialect: str = "PostgreSQL") -> str:
    """Prompt template for SQL generation."""
    return f"""
You are a {dialect} expert. Generate efficient, indexed SQL queries.
- Always use proper JOINs
- Index column names
- Add LIMIT clauses
- Explain query plan if complex
"""

@server.prompt()
def data_analyst() -> str:
    """Prompt template for data analysis."""
    return """
You are a data analyst. Analyze trends and patterns.
- Identify outliers
- Suggest optimizations
- Provide confidence intervals
"""
```

### Running the Server

```bash
# As stdio (piped to parent process)
python server.py  # Parent redirects stdin/stdout

# As HTTP service
python server.py --transport http --port 3000
# Now accessible at http://localhost:3000/mcp
```

---

## Part 4: Connecting Agents to MCP (Client Side)

### Via MCP Python SDK

```python
from mcp.client import Client

# Connect to server
client = Client(transport="stdio", command="python server.py")

# Discover tools
tools = client.call("tools/list")
print(tools)
# {
#   "tools": [{"name": "query_database", ...}, ...]
# }

# Call tool
result = client.call("tools/call", {
    "name": "query_database",
    "arguments": {"sql": "SELECT * FROM users", "limit": 10}
})
print(result)
# {"rows": [...], "count": 10}

# Read resource
content = client.call("resources/read", {
    "uri": "file:///docs/api-guide.md"
})
print(content)
```

### Via Claude Desktop (Config)

MCP servers can be exposed to Claude Desktop via `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "database": {
      "command": "python",
      "args": ["/path/to/database-server.py"]
    },
    "filesystem": {
      "command": "python",
      "args": ["/path/to/filesystem-server.py"]
    }
  }
}
```

Claude Desktop auto-discovers tools from all configured servers.

### Via AgentCore (Bedrock)

In AgentCore, Gateway uses MCP protocol internally:

```python
# Agent code (Strands, LangGraph, etc.)
agent = Agent(
    tools=[
        # Gateway handles MCP internally
        DatabaseTool(),
        FileSystemTool(),
    ]
)

# AgentCore Gateway:
# - Discovers tools from MCP servers
# - Routes agent tool calls to correct server
# - Returns results to agent
```

---

## Part 5: Available Community MCP Servers

| Server | Tools | Resources | Use Case |
|---|---|---|---|
| **filesystem** | read_file, list_files, search | `/path/to/file.txt` | File access |
| **github** | list_repos, create_issue, list_prs | repos, issues, PRs | GitHub integration |
| **slack** | post_message, read_channel | channels, messages | Slack bot |
| **postgres** | query, list_tables, execute_sql | tables, schemas | Database access |
| **google-drive** | list_files, read_document | documents, sheets | Google Drive |
| **sqlite** | query_database | tables, views | SQLite DB |
| **stripe** | list_charges, create_payment | charges, subscriptions | Stripe API |
| **openai** | list_models, call_gpt | models, conversations | OpenAI API |

**Install from npm:**
```bash
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-postgres
# ... etc
```

---

## Part 6: MCP vs Function Calling vs OpenAPI

| Dimension | MCP | Function Calling | OpenAPI |
|---|---|---|---|
| **Purpose** | Protocol for LLM ↔ external server | Direct tool invocation | API spec for REST services |
| **Scope** | Tools + Resources + Prompts | Tools only | API endpoints |
| **Transport** | stdio / HTTP | Any (SDK-dependent) | HTTP REST |
| **Discovery** | `tools/list`, `resources/list` | Baked into schema | OpenAPI schema |
| **Server complexity** | Lightweight (can be function) | Varies | Can be complex service |
| **Standardization** | Open standard (Anthropic) | Per-model (OpenAI, Anthropic) | ISO standard |
| **Use case** | Universal LLM integrations | Quick prototypes, simple tools | Public APIs |

**Decision tree:**
- **Simple agent + few tools:** Function calling (built into SDK)
- **Complex integrations, resource access:** MCP (cleaner separation)
- **Integrating public APIs:** OpenAPI (already defined)
- **New integration:** MCP (future-proof, reusable)

---

## Part 7: Design Patterns

### Pattern 1: Federated Servers (Multiple Servers)

```
Agent connected to 4 servers:
- MCP Server 1: Database tools
- MCP Server 2: Filesystem tools
- MCP Server 3: Slack tools
- MCP Server 4: GitHub tools

Agent sends tool call → Server decides which one handles it
Each server independent (can scale, restart, update separately)
```

### Pattern 2: Resource-First Design

```
Agent doesn't know all tools upfront.
Discovers resources via resources/list.
Reads resource URI → automatically selects right tool.

Example:
- User: "Summarize /docs/api-guide.md"
- Agent: "What resources exist?"
- Server: "uri=file:///docs/api-guide.md"
- Agent: "Read that resource"
- Server: "Returns file content"
- Agent: Summarizes
```

### Pattern 3: Prompt-Driven Tool Selection

```
Server provides domain-specific prompts via prompts/list.
Agent retrieves prompt → uses it as system context.
LLM can then select right tool with domain knowledge.

Example:
- Server: prompt="sql_expert" for database server
- Agent: retrieves SQL expert prompt
- LLM (with prompt): "I should write efficient SQL"
- Calls database tools with domain context
```

---

## Part 8: Interview Questions & Answers

### Q1: What is MCP and why should I care?

**Answer:**
```
MCP is a universal protocol for LLM apps to connect to external services.
Think USB-C: one port, any device.

Without MCP:
- Every integration = custom code (file access, database, Slack)
- No reusability across tools
- No standard for discovery (what tools exist?)

With MCP:
- One protocol for all integrations
- Servers are interchangeable
- Standard discovery (tools/list, resources/list)
- Any agent, any server, works together

Key insight: MCP decouples agent from integration details.
```

### Q2: Design an MCP server for user preferences. What tools?

**Answer:**
```
Tools:
1. save_preference(key: str, value: any) → save user pref
2. get_preference(key: str) → retrieve pref
3. list_preferences() → list all prefs
4. delete_preference(key: str) → remove pref

Resources:
1. user:///prefs/all.json → JSON export of all prefs
2. user:///prefs/{key} → individual preference

Prompts:
1. preference_expert → "Manage user preferences carefully..."

Implementation:
- Backend: DynamoDB or Redis
- Server: Python MCP server, 50 lines
- Reusable: any agent can use it
```

### Q3: Agent connected to 3 MCP servers. How does it choose which tool to call?

**Answer:**
```
Agent connected to: Database MCP, Filesystem MCP, Slack MCP

When agent needs to "save file":
1. Agent calls tools/list on all servers
2. Gets union of tools from all servers
3. LLM sees: read_file, query_db, post_message, ...
4. LLM: "I need to save file → read_file is available"
5. Calls Filesystem MCP server's read_file tool
6. Filesystem server executes, returns result

Key: Each server has unique tool names or descriptions
LLM naturally routes to right server based on tool name + description.
```

### Q4: Traditional REST API vs MCP server. When each?

**Answer:**
```
Use REST API (OpenAPI) when:
- Public facing (users call it directly)
- High throughput (millions of requests)
- Multiple clients (web, mobile, desktop)
- Standard HTTP tools exist

Use MCP when:
- Private integration (LLM agent only)
- Tool discovery needed (agent doesn't know what exists)
- Resources + tools + prompts (not just endpoints)
- Lightweight, easy to build (decorate Python functions)

Hybrid: Build MCP server, expose via REST wrapper if needed.
```

### Q5: Build an MCP server that lets agent query GitHub, update Slack, and write files. Architecture?

**Answer:**
```
One MCP server OR three separate servers?

Option A: One monolithic server
  Tools: query_github, post_slack, write_file
  Pros: Single process to manage
  Cons: Mixed concerns (GitHub + Slack + files)

Option B: Three federated servers
  - GitHub MCP: list_repos, get_issues, create_pr
  - Slack MCP: post_message, read_channel
  - Filesystem MCP: read_file, write_file
  Pros: Separation of concerns, reusable, scalable
  Cons: Three processes

Recommendation: Option B (three servers)
Agent connected to all three.
Each server independent (can be updated, scaled, swapped).

Implementation:
- Each server: <100 lines Python
- Discovery: Agent calls tools/list on each
- Execution: Agent routes tool calls correctly
```

---

## Part 9: Common Patterns & Best Practices

### Pattern: Error Handling in Tools

```python
@server.tool()
def query_database(sql: str) -> dict:
    """Execute SQL query."""
    try:
        result = db.execute(sql)
        return {"success": True, "data": result}
    except SyntaxError:
        return {"success": False, "error": "Invalid SQL", "retry": False}
    except ConnectionError:
        return {"success": False, "error": "DB unavailable", "retry": True}
```

**Agent sees result → decides to retry (if retry=True) or escalate.**

### Anti-Pattern: Exposing All Database Rows

```python
# BAD:
@server.tool()
def get_all_data(table: str) -> list:
    return db.execute(f"SELECT * FROM {table}")
    # If table has 10M rows → huge response, slow

# GOOD:
@server.tool()
def query_data(table: str, limit: int = 100) -> list:
    return db.execute(f"SELECT * FROM {table} LIMIT {limit}")
    # Default limit, agent can request more if needed
```

### Best Practice: Resource Caching

```python
from functools import lru_cache

@server.resource()
@lru_cache(maxsize=100)  # Cache recent reads
def get_resource(uri: str) -> str:
    # Read file, but cache results
    # Avoids re-reading same file repeatedly
    pass
```

---

## Part 10: Summary

| Concept | Key Takeaway |
|---|---|
| **MCP** | Universal protocol for LLM ↔ external services |
| **USB-C analogy** | One port, any device |
| **Client/Server** | Host (LLM app) → Client (protocol) → Server (tools + resources) |
| **Primitives** | tools/list, tools/call, resources/list, resources/read, prompts/list, prompts/get |
| **Building** | Python SDK: `@server.tool()`, `@server.resource()`, `@server.prompt()` |
| **Connecting** | Via SDK, Claude Desktop config, or AgentCore Gateway |
| **Community servers** | filesystem, github, slack, postgres, google-drive, stripe, etc. |
| **vs Function Calling** | MCP = protocol for all integrations; function calling = quick prototypes |
| **vs OpenAPI** | MCP = LLM-specific; OpenAPI = general REST spec |
| **Interview ready** | One protocol to rule them all: discover, call, stream resources |

**You now understand MCP at production depth.**
