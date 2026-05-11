# Geographic-Aware Vector Search: ReAct Loop + Distance Filtering

Problem: Vector search finds *semantically* relevant stations ("fast charger"), but geography matters ("near me"). How to combine both?

Answer: **ReAct loop (LLM reasons → tools execute geographic filter + distance calc → LLM evaluates results)**

---

## The Problem with Pure Vector Search

### Scenario 1: Naive Vector Search (WRONG)

```
User: "I need a fast charger nearby"
Location: San Francisco (37.7749, -122.4194)

Vector embedding: "fast charger"
↓
Vector DB search (top 5 semantic matches)
Results:
1. FastCharge Premium, downtown SF, 0.5 km away ✅
2. Tesla Supercharger, Oakland, 15 km away ⚠️
3. ChargePoint Fast, Berkeley, 25 km away ❌
4. Electrify America, San Jose, 60 km away ❌
5. SuperCharger, LA, 400 km away ❌

Problem: Vector search returns semantic matches globally.
User wants: closest fast charger (geographic + semantic)
```

### Why Pure Vector Search Fails

Vector DB knows: "fast charger" ≈ "rapid charging" ≈ "DC fast"  
Vector DB does NOT know: "nearby" = "within 5 km"

```
Embedding space is semantic:
- "fast charger" and "DC Fast" are close (similar meaning)
- "San Francisco" and "Oakland" are ALSO close (both California)
- But geographically: 15 km apart!

Vector distance ≠ Geographic distance
```

---

## Solution: ReAct Loop with Spatial Filtering

```
User Query
    ↓
LLM (Reasoning): "User wants fast charger nearby. 
                  Must search by:
                  1. Semantic relevance (charger type)
                  2. Geographic proximity (< 10km)"
    ↓
Tool 1: Vector Search
        Input: "fast charger" embedding
        Output: [top 10 stations by semantic match]
    ↓
Tool 2: Geographic Filter + Distance Calc
        Input: [10 stations], user_location, max_distance=10km
        Calculation: haversine(user_lat, user_lon, station_lat, station_lon)
        Output: [stations within 10km, sorted by distance]
    ↓
LLM (Evaluation): "Best option is downtown SF (0.5km, 4.9★, $0.45/kWh)"
    ↓
User Response
```

---

## Architecture: Three Layers

```
┌─────────────────────────────────────────┐
│ LLM Reasoning & Orchestration (ReAct)   │
├─────────────────────────────────────────┤
│ "I need to search semantic + geographic"│
│ Call: vector_search + geographic_filter │
└─────────────────────────────────────────┘
         ↓                          ↓
┌──────────────────┐      ┌─────────────────┐
│ Semantic Layer   │      │ Geographic Layer│
│ (Vector DB)      │      │ (PostGIS)       │
├──────────────────┤      ├─────────────────┤
│ Pinecone Index:  │      │ PostgreSQL:     │
│ "fast DC charger"│      │ ST_Distance     │
│ 10 results       │      │ ST_DWithin      │
│ (global, ranked │      │ haversine()     │
│ by embedding     │      │ Calculate km    │
│ similarity)      │      │ Filter by radius│
└──────────────────┘      └─────────────────┘
```

---

## Implementation: ReAct Pattern

### Step 1: LLM Tool Definition

```python
from langchain_core.tools import StructuredTool
from typing import Optional

# Tool 1: Vector search (semantic)
vector_search_tool = StructuredTool(
    name="vector_search_stations",
    description="""Search EV charging stations by semantic relevance.
    Use for: "fast charger", "cheap option", "Tesla compatible"
    Returns: top 10 semantically relevant stations (UNFILTERED by location)""",
    func=vector_search_stations,
    args_schema={
        "query": "Natural language query (e.g., 'fast DC charger')",
        "top_k": "Number of results (default: 10)",
    }
)

# Tool 2: Geographic filter (spatial)
geographic_filter_tool = StructuredTool(
    name="filter_by_distance",
    description="""Filter stations by geographic distance from user.
    Calculates distance using haversine formula.
    Use AFTER vector_search to get nearby results.""",
    func=filter_by_distance,
    args_schema={
        "station_ids": "List of station UUIDs to filter",
        "user_latitude": "User's lat",
        "user_longitude": "User's lon",
        "max_distance_km": "Max distance (default: 10)",
    }
)

# Tool 3: Get detailed info (station details + pricing)
station_details_tool = StructuredTool(
    name="get_station_details",
    description="""Get detailed info about a specific station:
    - Current availability (ports available)
    - Pricing ($/kWh)
    - Charger types
    - Reviews/ratings
    - Estimated charge time""",
    func=get_station_details,
    args_schema={
        "station_id": "Station UUID",
    }
)
```

### Step 2: Define ReAct Agent with Tools

```python
from langgraph.graph import StateGraph
from langgraph.agents import AgentExecutor
from typing_extensions import TypedDict

class DriverSearchState(TypedDict):
    user_id: str
    user_latitude: float
    user_longitude: float
    query: str
    messages: list  # Conversation history
    search_results: list  # From vector search
    filtered_results: list  # After geographic filter
    final_recommendation: str

# System prompt with explicit reasoning
SEARCH_SYSTEM_PROMPT = """You are a helpful EV charging assistant.

When user asks for nearby charging:
1. REASON: Identify what matters - charger type? Speed? Price? Distance?
2. SEARCH: Call vector_search_stations with query
3. FILTER: Call filter_by_distance with results + user location + max_distance
4. EVALUATE: Call get_station_details for top 3 results
5. RECOMMEND: Present best 2-3 options with reasoning

Example:
User: "I need to charge ASAP near me"
Your reasoning: "User is urgent (ASAP) and location-aware (near me). 
                 Must search for fast chargers, then filter by distance."
Actions:
1. vector_search_stations("fast DC charger")
2. filter_by_distance(results, user_lat, user_lon, max_distance_km=5)
3. get_station_details for top 3
Present results sorted by distance, not just semantic similarity.
"""

# Build agent
driver_search_agent = create_react_agent(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    tools=[
        vector_search_tool,
        geographic_filter_tool,
        station_details_tool
    ],
    system_prompt=SEARCH_SYSTEM_PROMPT,
    messages_modifier=None  # Use full conversation history
)
```

### Step 3: Tool Implementations

```python
from math import radians, sin, cos, sqrt, atan2
import asyncio

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two geographic points (km)
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

async def vector_search_stations(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic search in Pinecone
    """
    # Embed the query
    embedding = await embed_model.aembed_query(query)
    
    # Search Pinecone (returns stations ranked by embedding similarity)
    results = pinecone_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract station metadata
    stations = []
    for match in results.matches:
        stations.append({
            "id": match.metadata["station_id"],
            "name": match.metadata["name"],
            "latitude": match.metadata["lat"],
            "longitude": match.metadata["lon"],
            "charger_types": match.metadata["charger_types"],
            "avg_rating": match.metadata["avg_rating"],
            "semantic_score": match.score  # Embedding similarity (0-1, higher=more relevant)
        })
    
    return stations

async def filter_by_distance(
    station_ids: list[str],
    user_latitude: float,
    user_longitude: float,
    max_distance_km: float = 10
) -> list[dict]:
    """
    Filter stations by distance using PostGIS
    Returns: [{"id": "...", "distance_km": 2.3, ...}, ...]
    """
    
    # Query PostgreSQL with PostGIS
    query = """
    SELECT 
        s.id,
        s.name,
        s.latitude,
        s.longitude,
        s.charger_types,
        s.avg_rating,
        ST_Distance(
            ST_Point(%s, %s)::geography,
            ST_Point(s.longitude, s.latitude)::geography
        ) / 1000 as distance_km
    FROM stations s
    WHERE s.id = ANY(%s)
    AND ST_DWithin(
        ST_Point(%s, %s)::geography,
        ST_Point(s.longitude, s.latitude)::geography,
        %s * 1000  -- Convert km to meters
    )
    ORDER BY distance_km ASC
    """
    
    results = db.execute(
        query,
        user_longitude, user_latitude,  # User location
        station_ids,  # Filter to these stations
        user_longitude, user_latitude,  # For ST_DWithin
        max_distance_km
    )
    
    filtered = []
    for row in results:
        filtered.append({
            "id": row.id,
            "name": row.name,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "charger_types": row.charger_types,
            "avg_rating": row.avg_rating,
            "distance_km": round(row.distance_km, 1),
            "travel_time_min": int(row.distance_km * 2.5)  # Rough estimate: 2.5 min per km
        })
    
    return filtered

async def get_station_details(station_id: str) -> dict:
    """
    Get full details: availability, pricing, reviews
    """
    
    # Station base info
    station = db.query(
        "SELECT * FROM stations WHERE id = %s",
        station_id
    )[0]
    
    # Current availability (real-time from IoT)
    ports = db.query(
        "SELECT status FROM port_status WHERE station_id = %s",
        station_id
    )
    available_count = sum(1 for p in ports if p.status in ['available', 'reserved'])
    
    # Pricing
    pricing = db.query(
        "SELECT * FROM pricing WHERE station_id = %s AND effective_from <= now() ORDER BY effective_from DESC LIMIT 1",
        station_id
    )[0]
    
    # Estimate charge time (0-80% in 30 min for DC Fast)
    charge_time_minutes = 30 if 'DC_Fast' in station.charger_types else 120
    
    # Recent reviews
    reviews = db.query(
        "SELECT rating, body FROM reviews WHERE station_id = %s ORDER BY created_at DESC LIMIT 3",
        station_id
    )
    
    return {
        "id": station.id,
        "name": station.name,
        "available_ports": available_count,
        "total_ports": station.total_ports,
        "price_per_kwh": float(pricing.base_price_per_kwh),
        "price_multiplier": pricing.current_multiplier,  # Peak pricing?
        "estimated_charge_time_min": charge_time_minutes,
        "avg_rating": station.avg_rating,
        "charger_types": station.charger_types,
        "recent_reviews": [{"rating": r.rating, "text": r.body} for r in reviews]
    }
```

---

## ReAct Execution Flow Example

### User Query: "I need a fast charger ASAP, I'm at 37.7749, -122.4194"

#### Step 1: LLM Reasoning

```
LLM thinks:
"User is urgent (ASAP = no time to travel far).
 Fast = DC Fast Charger preferred.
 Location: San Francisco (37.7749, -122.4194).
 
 Action plan:
 1. Search for 'fast DC charger' semantically
 2. Filter results to within 5km (urgent = close)
 3. Get details for top 2 options
 4. Recommend by: distance first (urgency), then rating/price
"

Output: "I'll find you the closest fast charger. Let me search..."
```

#### Step 2: Tool Call 1 - Vector Search

```
LLM calls: vector_search_stations(query="fast DC charger", top_k=10)

Tool returns (semantic ranking, GLOBAL):
[
  {
    "id": "sf-downtown-1",
    "name": "FastCharge Downtown SF",
    "latitude": 37.7833,
    "longitude": -122.4089,
    "charger_types": ["DC_Fast", "Level2"],
    "avg_rating": 4.8,
    "semantic_score": 0.98  ← Highly semantically relevant
  },
  {
    "id": "oakland-1",
    "name": "ElectrifyAmerica Oakland",
    "latitude": 37.8316,
    "longitude": -122.2723,
    "charger_types": ["DC_Fast"],
    "avg_rating": 4.2,
    "semantic_score": 0.96  ← Also semantically relevant
  },
  ... 8 more results globally (some in LA, San Diego, etc.)
]
```

#### Step 3: Tool Call 2 - Geographic Filter

```
LLM calls: filter_by_distance(
  station_ids=[list of 10 IDs],
  user_latitude=37.7749,
  user_longitude=-122.4194,
  max_distance_km=5
)

Tool returns (geographic ranking, LOCAL):
[
  {
    "id": "sf-downtown-1",
    "name": "FastCharge Downtown SF",
    "distance_km": 0.8,
    "travel_time_min": 2,
    "charger_types": ["DC_Fast"],
    "avg_rating": 4.8
  },
  {
    "id": "sf-soma-1",
    "name": "ChargePoint SoMa",
    "distance_km": 2.1,
    "travel_time_min": 5,
    "charger_types": ["DC_Fast", "Level2"],
    "avg_rating": 4.3
  }
  
  # Oakland (15 km away) is FILTERED OUT (> 5 km)
  # LA results are FILTERED OUT (> 5 km)
]

Result: Now only NEARBY stations remain
```

#### Step 4: Tool Call 3 - Station Details

```
LLM calls: get_station_details(station_id="sf-downtown-1")

Tool returns:
{
  "id": "sf-downtown-1",
  "name": "FastCharge Downtown SF",
  "available_ports": 3,
  "total_ports": 8,
  "price_per_kwh": 0.45,
  "price_multiplier": 1.2,  # Peak pricing (1.2x = 12% surge)
  "estimated_charge_time_min": 30,
  "avg_rating": 4.8,
  "charger_types": ["DC_Fast", "Level2"],
  "recent_reviews": [
    {"rating": 5, "text": "Super fast, convenient location!"},
    {"rating": 5, "text": "Great experience"},
    {"rating": 4, "text": "Clean, well maintained"}
  ]
}

(Also gets details for 2nd option: sf-soma-1)
```

#### Step 5: LLM Evaluation & Recommendation

```
LLM synthesizes and responds:

"✅ Found 2 fast chargers nearby!

Best option (CLOSEST):
🔌 FastCharge Downtown SF (0.8 km, 2 min away)
   • 3 ports available
   • 4.8★ rating (customers love it)
   • ⚡ 30 min to 80%
   • 💰 $0.45/kWh (12% surge pricing now)
   
Backup option:
🔌 ChargePoint SoMa (2.1 km, 5 min away)
   • 2 ports available
   • 4.3★ rating
   • ⚡ 45 min to 80%
   • 💰 $0.38/kWh (cheaper, but farther)

Which would you like to book?"

→ User picks option 1
→ System proceeds to reservation
```

---

## Why ReAct Beats Pure Vector Search

### Comparison

| Aspect | Vector Search Only | ReAct Loop |
|--------|-------------------|-----------|
| Query | "fast charger" | "fast charger ASAP near me" |
| Result ranking | Semantic score | Distance + semantic |
| Geographic awareness | ❌ Returns global results | ✅ Filters locally |
| Distance calculation | ❌ None | ✅ Haversine + PostGIS |
| LLM reasoning | ❌ Implicit | ✅ Explicit ("I'll filter to 5km") |
| User experience | "Oakland charger?" (15km away) ❌ | "Downtown charger" (0.8km away) ✅ |
| Observability | Black box | Clear tool calls logged |

---

## Implementation Checklist

- [ ] **Vector DB:** Pinecone index with station metadata (name, location, types)
- [ ] **PostGIS:** PostgreSQL spatial index on station locations
- [ ] **Tools:** 3 tools (vector_search, filter_by_distance, get_station_details)
- [ ] **LLM:** ReAct agent with system prompt guiding geographic reasoning
- [ ] **Distance calc:** Haversine function for km estimation
- [ ] **Real-time availability:** IoT → port_status table updates
- [ ] **Logging:** Log which tools called, results, final recommendation
- [ ] **Testing:** Test queries like "fast near me", "cheapest option", "longest charger hours"

---

## Interview Answer

**"How do you ensure vector search finds the nearest charging location?"**

**Script (3 min):**

"Vector search alone finds semantically relevant stations globally — not good if user is in SF but it returns LA results. Solution: ReAct loop.

LLM reasons about the query: 'User wants fast AND nearby.' Then:
1. **Vector search:** Call tool with 'fast DC charger', get top 10 semantic matches (global)
2. **Geographic filter:** PostGIS ST_DWithin filters results to 5km radius using haversine distance
3. **Details lookup:** Call tool to get real-time availability + pricing for top 2-3 filtered results
4. **Evaluation:** LLM synthesizes and recommends by distance first, then rating

Concrete example: User in SF (37.77, -122.42) asks 'fast charger ASAP':
- Vector search returns: Downtown SF (0.8km, 4.8★), Oakland (15km, 4.2★), LA (400km, 4.5★)
- Filter by 5km: Only Downtown & SoMa remain
- User gets closest option, not best semantic match globally

Logging every tool call + final recommendation ensures we can debug if nearby stations are missed."

---

## Code Template: Minimal ReAct Driver Search

```python
from langgraph.agents import create_react_agent
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class DriverSearchState(TypedDict):
    user_location: tuple[float, float]
    query: str
    messages: list

# Initialize agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
tools = [vector_search_tool, geographic_filter_tool, station_details_tool]

search_agent = create_react_agent(llm, tools, system_prompt=SEARCH_SYSTEM_PROMPT)

# Invoke
state = {
    "user_location": (37.7749, -122.4194),
    "query": "fast charger ASAP",
    "messages": [{"role": "user", "content": "fast charger ASAP"}]
}

# Stream response (real-time tool calls visible)
for event in search_agent.stream(state):
    if "tool_calls" in event:
        print(f"Tool: {event['tool_calls'][0]['name']}")
    elif "text" in event:
        print(f"LLM: {event['text']}")

# Final recommendation
print(state["messages"][-1]["content"])
```

