# Tools and Tool-Calling

## What Is a Tool?

A tool is a function the agent can call. **Tool = Name + Description + JSON Schema + Callable Implementation**

```python
{
    "name": "web_search",
    "description": "Search the web for current information. Use this when you need real-time data or facts not in your training data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, e.g. 'weather in NYC today'"
            }
        },
        "required": ["query"]
    }
}
```

The LLM reads the description and schema, then decides: "Should I call this tool?" The quality of the description determines if the LLM uses it correctly.

## Tool-Calling Protocol: How It Works

### Step 1: Define Tools (Anthropic / OpenAI Format)

```python
tools = [
    {
        "name": "calculator",
        "description": "Perform arithmetic: add, subtract, multiply, divide",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Always use this for recent events, prices, weather.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]
```

### Step 2: LLM Decides to Call a Tool

```python
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's 47 * 23? And what's the weather in NYC?"}
    ]
)

# Response includes:
# - Text content (if any reasoning before tool calls)
# - Tool use blocks:
# {
#     "type": "tool_use",
#     "id": "toolu_01a0c4...",
#     "name": "calculator",
#     "input": {"operation": "multiply", "a": 47, "b": 23}
# }
# {
#     "type": "tool_use",
#     "id": "toolu_01b3d5...",
#     "name": "web_search",
#     "input": {"query": "weather in NYC today"}
# }
```

### Step 3: Execute Tools

```python
# Map tool names to implementations
def calculator(operation: str, a: float, b: float) -> float:
    if operation == "add": return a + b
    if operation == "subtract": return a - b
    if operation == "multiply": return a * b
    if operation == "divide": return a / b if b != 0 else None

def web_search(query: str) -> str:
    # In reality, call an API like SerpAPI or DuckDuckGo
    return f"Search results for: {query}"

# Execute each tool call
results = []
for block in response.content:
    if block.type == "tool_use":
        tool_name = block.name
        tool_input = block.input

        if tool_name == "calculator":
            result = calculator(**tool_input)
        elif tool_name == "web_search":
            result = web_search(**tool_input)

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(result)
        })
```

### Step 4: Feed Results Back to LLM

```python
# Continue the conversation with tool results
response_2 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's 47 * 23? And what's the weather in NYC?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": results}  # Tool results
    ]
)

# LLM now synthesizes: "47 * 23 = 1081. The weather in NYC is..."
print(response_2.content[0].text)
```

## Tool Categories

| Category | Purpose | Examples | When to Use |
|----------|---------|----------|------------|
| **Search** | Fetch real-time data | web_search, news_search, code_search | Recent events, current prices, breaking news |
| **Code Executor** | Run code and get output | python_executor, bash_executor | Calculations, data processing, testing |
| **API Integration** | Call external services | stripe_charge, slack_post, github_create_issue | Business integrations, automations |
| **Memory** | Store and retrieve facts | memory_store, retriever | Long-context tasks, learning from interactions |
| **File Operations** | Read/write files | file_read, file_write | Document processing, code generation |
| **SQL/DB** | Query databases | sql_executor, vector_db_query | Data lookups, complex filtering |
| **Specialized** | Domain-specific tasks | pdf_parser, image_generator, video_transcriber | Multimodal processing |

## GOOD vs BAD Tool Descriptions

### BAD: Vague Description

```python
{
    "name": "search",
    "description": "Search for things",  # Too vague!
    "input_schema": {
        "properties": {
            "q": {"type": "string"}  # What does 'q' mean?
        }
    }
}
```

**Problem**: LLM doesn't know when to call it. Is it for web search? Product search? Internal search?

### GOOD: Specific, Use-Case Driven

```python
{
    "name": "web_search",
    "description": "Search the public web for current information. Use this ONLY for recent events (last 30 days), real-time prices, weather, or breaking news. Do NOT use for general knowledge questions you can answer from your training data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific: 'Apple stock price today' not 'stock'. Include keywords."
            }
        },
        "required": ["query"]
    }
}
```

**Better**: Clear guidance on when and how to use the tool. LLM is much more likely to use it correctly.

## Tool Argument Validation

Always validate tool inputs before execution. Bad data can cause failures:

```python
def web_search(query: str) -> str:
    # Validation
    if not query or len(query) == 0:
        return "ERROR: Query cannot be empty"
    if len(query) > 500:
        return "ERROR: Query too long (max 500 chars)"
    if any(c in query for c in ['<', '>', '{', '}']):
        return "ERROR: Query contains invalid characters"

    # Safe to execute
    return perform_search(query)
```

## Tool Execution Loop (Full Example)

```python
def run_agent_loop(user_query: str, max_iterations: int = 10):
    messages = [{"role": "user", "content": user_query}]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # LLM decides what to do
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        # Did LLM decide to use tools?
        tool_calls = [block for block in response.content if block.type == "tool_use"]

        if not tool_calls:
            # No more tool calls → agent is done
            text_response = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "No response"
            )
            return text_response

        # Execute tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for tool_call in tool_calls:
            if tool_call.name == "calculator":
                result = calculator(**tool_call.input)
            elif tool_call.name == "web_search":
                result = web_search(**tool_call.input)
            else:
                result = f"ERROR: Unknown tool {tool_call.name}"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": str(result)
            })

        messages.append({"role": "user", "content": tool_results})

    return f"Max iterations ({max_iterations}) reached"

# Usage
response = run_agent_loop("What's 1000 * 2000? Then search for the latest AI news.")
print(response)
```

## Security: Sandboxing Tools

Not all tool calls are safe. Implement guards:

```python
ALLOWED_OPERATIONS = ["add", "subtract", "multiply", "divide"]
BLOCKED_PATHS = ["/etc", "/sys", "/proc", "~/.ssh"]

def calculator(operation: str, a: float, b: float) -> float:
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Operation {operation} not allowed")
    # ... rest of implementation

def file_read(path: str) -> str:
    # Security check
    if any(blocked in path for blocked in BLOCKED_PATHS):
        return f"ERROR: Access denied to {path}"
    with open(path) as f:
        return f.read()

def code_executor(code: str) -> str:
    # CRITICAL: Never execute untrusted code
    # Use sandboxing (Docker, subprocess timeout, or specialized services like Replit/E2B)
    # Example: timeout after 5 seconds, limit memory to 512MB
    import subprocess
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            timeout=5,
            text=True
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out"
```

## Rate Limiting and Cost Control

```python
from collections import defaultdict
import time

class ToolRateLimiter:
    def __init__(self):
        self.call_counts = defaultdict(list)
        self.limits = {
            "web_search": 10,      # Max 10 calls per minute
            "calculator": 100,
            "code_executor": 5     # Expensive, limit to 5/minute
        }

    def can_call(self, tool_name: str) -> bool:
        now = time.time()
        minute_ago = now - 60

        # Remove calls older than 1 minute
        self.call_counts[tool_name] = [
            t for t in self.call_counts[tool_name] if t > minute_ago
        ]

        # Check limit
        if len(self.call_counts[tool_name]) >= self.limits.get(tool_name, 100):
            return False

        self.call_counts[tool_name].append(now)
        return True
```

## Why Tool Descriptions Matter Most

The LLM's decision to call a tool depends 90% on the description. Poor descriptions = poor tool usage.

```
Query: "What time is it?"

With BAD description ("tool for general info"):
→ LLM might call web_search, wasting tokens

With GOOD description ("Only use this for web search, not for time"):
→ LLM knows this is for web data, avoids calling it
```

Invest time in writing clear, specific tool descriptions. It directly impacts agent reliability.
