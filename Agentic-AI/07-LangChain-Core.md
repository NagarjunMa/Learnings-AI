# LangChain Core

## Core Abstractions

LangChain is a framework for building AI applications. Its main abstractions are:

| Abstraction | Purpose | Example |
|-------------|---------|---------|
| **LLM** | Language model interface (OpenAI, Anthropic, etc.) | `ChatOpenAI(model="gpt-4")` |
| **PromptTemplate** | Format input into a prompt | `PromptTemplate(template="Q: {question}\nA:")` |
| **Chain** | Sequence of operations (deterministic) | `prompt \| llm \| output_parser` |
| **Agent** | Autonomous system with tools and memory | `ReActAgent(tools=[...], llm=...)` |
| **Tool** | Function the agent can call | `web_search_tool`, `calculator_tool` |
| **Memory** | Conversation history and context | `ConversationBufferMemory()` |
| **Retriever** | Fetch relevant documents | `vectorstore.as_retriever()` |

## LCEL: LangChain Expression Language

LCEL is the pipe-based syntax for composing chains. It uses the `|` operator:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define components
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer: {question}"
)
model = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

# Chain them together with pipes
chain = prompt | model | output_parser

# Use the chain
result = chain.invoke({"question": "What is machine learning?"})
print(result)  # Output: "Machine learning is..."
```

### How Pipe Works

```
Input: {"question": "..."}
  ↓
prompt (PromptTemplate)
  → Formatted string: "You are a helpful assistant. Answer: ..."
  ↓
model (ChatOpenAI)
  → AIMessage(content="Machine learning is...")
  ↓
output_parser (StrOutputParser)
  → "Machine learning is..."
  ↓
Output: "Machine learning is..."
```

Each `|` passes the output of the left to the input of the right.

## Building a ReAct Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# Define tools
def calculator(a: float, b: float, operation: str) -> float:
    if operation == "add": return a + b
    if operation == "multiply": return a * b
    return 0

tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(calculator(float(x.split()[0]), float(x.split()[2]), x.split()[1])),
        description="Useful for math. Input: '5 add 3' or '10 multiply 2'"
    )
]

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Use pre-built ReAct prompt template
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:""")

# Create agent
agent = create_react_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10
)

# Run agent
result = agent_executor.invoke({"input": "What's 47 * 23?"})
print(result["output"])  # Output: "47 * 23 = 1081"
```

## RAG Pipeline with Retriever

RAG (Retrieval-Augmented Generation) fetches relevant documents, then uses them to answer:

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create vector store from documents
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "NLP processes text data"
]

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(documents, embeddings)

# Retrieve similar documents
retriever = vectorstore.as_retriever()

# Build RAG chain
template = """
Use the following context to answer the question:

{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
    | output_parser
)

# Use RAG chain
result = rag_chain.invoke({"question": "What is machine learning?"})
print(result)
# Output: "Machine learning is a subset of AI..."
```

## Callbacks and Tracing

LangChain integrates with LangSmith for tracing and debugging:

```python
from langsmith import traceable
from langchain_core.callbacks import LangChainTracer

# Option 1: Use decorator
@traceable
def my_agent(query: str):
    # Your agent code
    return agent_executor.invoke({"input": query})

# Option 2: Use callback in chain
from langsmith.client import Client

client = Client()
tracer = LangChainTracer(project_name="my_agent", client=client)

result = agent.invoke(
    {"input": "What's 5+5?"},
    {"callbacks": [tracer]}
)

# View traces at: https://smith.langchain.com
```

Traces show:
- Each step of the agent
- Tool calls and results
- Token usage
- Latency
- Errors

## Agent Types in LangChain

### ReAct Agent (Most Common)

```python
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt_template)
executor = AgentExecutor.from_agent_and_tools(agent, tools)
```

### OpenAI Functions Agent

Uses OpenAI's native function calling:

```python
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt_template)
```

Simpler than ReAct, leverages model’s native tool support.

### Structured Tool Chat Agent

For Anthropic models with tool use:

```python
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt_template)
```

## Modern Tool Creation (Preferred Pattern)

Modern LangChain uses decorators and Pydantic for cleaner tool definition.

### 1. @tool Decorator (Simplest)

```python
from langchain_core.tools import tool

@tool
def search_stations(location: str, radius_km: int = 10) -> str:
    """Search for EV charging stations near a location.
    
    Args:
        location: City or address to search near
        radius_km: Search radius in kilometers
    """
    return f"Found 5 stations within {radius_km}km of {location}"

# Tool automatically extracts:
# - name: "search_stations" (function name)
# - description: docstring
# - args_schema: Pydantic from type hints

print(search_stations.name)           # "search_stations"
print(search_stations.description)    # Full docstring
print(search_stations.args_schema)    # Pydantic schema

# Use in agent
agent = create_react_agent(llm, [search_stations], prompt)
```

### 2. StructuredTool (Complex Pydantic Schema)

For agents with complex input validation:

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class StationSearchInput(BaseModel):
    location: str = Field(description="City or coordinates")
    radius_km: int = Field(default=10, description="Search radius in km")
    connector_type: str = Field(default="any", description="Charger type (Tesla, CCS, etc)")
    max_price_kwh: float = Field(default=999.0, description="Max price per kWh")

def search_stations_impl(location: str, radius_km: int, connector_type: str, max_price_kwh: float) -> str:
    # Implementation queries DB with all constraints
    return f"Found stations near {location} with {connector_type} under ${max_price_kwh}/kWh"

search_tool = StructuredTool.from_function(
    func=search_stations_impl,
    name="search_stations",
    description="Search EV charging stations with filters",
    args_schema=StationSearchInput  # Enforce input validation
)

# LLM will only call this with valid inputs
agent = create_react_agent(llm, [search_tool], prompt)
```

**Why StructuredTool matters:** Agent receives structured params (location, connector_type, price), not a string to parse. Prevents hallucination of invalid inputs.

### 3. PydanticOutputParser (Structured Responses)

Force LLM to return structured JSON matching a schema:

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

class StationRecommendation(BaseModel):
    station_name: str
    distance_km: float
    price_per_kwh: float
    reasoning: str

parser = PydanticOutputParser(pydantic_object=StationRecommendation)

# Parser auto-generates format instructions
prompt = PromptTemplate(
    template="Recommend best charging station.\n{format_instructions}\nQuery: {user_query}",
    input_variables=["user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

result = chain.invoke({"user_query": "Fast charger, under $2, 10km away"})
print(result.station_name)      # "Station A" (validated string)
print(result.price_per_kwh)     # 1.80 (validated float)
print(type(result))              # <class 'StationRecommendation'> (Pydantic model)
```

**Why this matters in production:** Parser validates JSON before returning. No parsing errors. Type-safe.

---

## Common Patterns

### Pattern 1: Simple Question → Answer

```python
prompt | llm | StrOutputParser()
```

### Pattern 2: Question → Search → Answer

```python
prompt | llm | search_tool | answer_prompt | llm | StrOutputParser()
```

### Pattern 3: Agent with Tools

```python
agent_executor (handles tool routing internally)
```

## Deprecation and Version Changes (v0.1 → v0.2+)

LangChain v0.2 simplified the API significantly:

| Old (v0.1) | New (v0.2+) | Why |
|-----------|-----------|-----|
| `BaseLanguageModel` | `LLM` | Simpler naming |
| `Document` | `Document` (same) | No change |
| `VectorStore.as_retriever()` | Same | Still works |
| `Agent` + `AgentExecutor` | `create_react_agent()` | Cleaner API |
| `load_tools()` | Define tools directly | More explicit |

**Migration tip**: If you see old code using `from langchain.agents import load_tools`, that's deprecated. Define tools explicitly instead:

```python
# Old
from langchain.agents import load_tools
tools = load_tools(["web-search"], llm=llm)

# New
from langchain_community.tools import DuckDuckGoSearchRun
tools = [DuckDuckGoSearchRun()]
```

## Memory in LangChain

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Multi-turn conversation
conversation.invoke({"input": "Hi, my name is Alice"})
conversation.invoke({"input": "What's my name?"})
# Output: "Your name is Alice"
```

Memory is automatically retrieved and injected into prompts.

## Key Gotchas

1. **Tool descriptions matter**: Vague descriptions = poor tool usage
2. **Context window limits**: Long conversations need memory management
3. **Cost**: Each loop iteration costs money; plan for multiple passes
4. **Model matters**: GPT-3.5 agents fail more often than GPT-4 or Claude
5. **Callbacks aren't automatic**: You must pass them to `invoke()` to trace

## When to Use LangChain vs LangGraph

| Task | LangChain | LangGraph |
|------|-----------|-----------|
| Simple chain | ✓ Recommended | Overkill |
| RAG pipeline | ✓ Recommended | Can use, more control |
| ReAct agent | ✓ Works well | Better for complex agents |
| Multi-agent system | Harder to manage | ✓ Recommended |
| Stateful, multi-turn | ✓ With memory | ✓ Native support |
| Human-in-the-loop | Possible but manual | ✓ Built-in |

For simple agents, use LangChain. For complex, stateful agents, use LangGraph (see `08-LangGraph-Core.md`).

## Getting Started

```bash
pip install langchain langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

template = "You are helpful. {input}"
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4")

chain = prompt | model
result = chain.invoke({"input": "What is AI?"})
print(result.content)
```

That's it. You're now using LangChain LCEL.
