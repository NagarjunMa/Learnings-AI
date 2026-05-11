# Production Deep-Dive: Dual-Sided EV Charging Marketplace Chatbot

Interview scenario: "Tell me about your EV charging marketplace chatbot project."  
If they dig: "What was the architecture? How did you handle multi-user state? Payment integration?"

This guide covers production implementation for both answering and building.

---

## System Overview

### Two User Types, Two Agents

```
┌─────────────────────────────────────────────────────────────┐
│ Shared Backend (PostgreSQL, Redis, Vector DB)               │
└─────────────────────────────────────────────────────────────┘
         ↑                           ↑
    ┌────────────────────┐   ┌──────────────────────┐
    │ DRIVER Agent       │   │ STATION OWNER Agent  │
    ├────────────────────┤   ├──────────────────────┤
    │ Find stations      │   │ Manage listings      │
    │ Check availability │   │ Set pricing          │
    │ Book & pay         │   │ View bookings        │
    │ Rate experience    │   │ Respond to inquiries │
    └────────────────────┘   └──────────────────────┘
         ↑                           ↑
    Driver App                Station App
    (Mobile/Web)             (Mobile/Web)
```

---

## Agent Design

### Agent 1: Driver Agent

**Persona:** Helpful assistant finding charging solutions

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic

class DriverState(TypedDict):
    user_id: str
    location: tuple[float, float]  # (lat, lon)
    query: str
    retrieved_stations: list[dict]
    selected_station: dict
    booking_details: dict
    messages: list

driver_agent_prompt = """
You are a helpful EV charging assistant for drivers.
Your role:
1. Understand driver needs (charge urgently? cheapest? fastest?)
2. Search nearby stations with real-time availability
3. Provide booking guidance
4. Handle payment questions
5. Suggest alternatives if chosen station is full

Always:
- Confirm driver location (for safety)
- Show prices and charging speed
- Ask about arrival time (5 min vs 1 hour)
- Offer alternatives proactively
- Suggest scheduling off-peak hours for savings
"""

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Tools available to driver agent
tools = [
    "search_nearby_stations",      # Query vector DB for stations within 5km
    "check_availability",           # Real-time occupancy from IoT sensors
    "get_pricing",                  # Dynamic pricing (time-of-day, demand)
    "place_reservation",            # Book slot
    "estimate_charge_time",         # Based on battery %, station speed
    "check_payment_methods",        # Saved cards, wallets
    "initiate_payment",             # Process payment
]
```

### Agent 2: Station Owner Agent

**Persona:** Efficient business tool for station operators

```python
station_owner_prompt = """
You are an intelligent assistant for EV charging station operators.
Your role:
1. Help owners monitor station performance
2. Optimize pricing based on demand
3. Manage reservations and no-shows
4. Answer inquiries from drivers
5. Monitor equipment health

Always:
- Report utilization rate (current, hourly, daily trends)
- Suggest optimal pricing (high demand = higher price)
- Alert on no-shows (charge cancellation fee?)
- Handle customer complaints (slow charging, broken port)
- Predict peak hours (AI forecasting)
"""

# Tools available to station owner agent
tools = [
    "list_my_stations",             # Get all owned stations
    "view_reservations",            # Today's + upcoming bookings
    "update_pricing",               # Dynamic pricing rules
    "mark_port_unavailable",        # Broken charger? Mark OOS
    "view_revenue",                 # Earnings by day/week/month
    "respond_to_inquiry",           # Auto-respond or escalate to human
    "view_analytics",               # Utilization, churn, customer feedback
    "refund_failed_charge",         # Customer support
]
```

---

## Database Schema

### Core Tables

```sql
-- Users (drivers and owners)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    user_type VARCHAR(50) CHECK (user_type IN ('driver', 'owner')),
    created_at TIMESTAMP DEFAULT now(),
    kyc_verified BOOLEAN DEFAULT FALSE,  -- Payment compliance
    stripe_customer_id VARCHAR(255)       -- Payment processor
);

-- Charging Stations (owner-created)
CREATE TABLE stations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    location POINT NOT NULL,  -- PostGIS: latitude, longitude
    total_ports INT NOT NULL,
    charger_types VARCHAR(50)[],  -- ['Level2', 'DC_Fast', 'Tesla']
    connector_types VARCHAR(50)[],  -- ['CCS', 'NACS', 'Tesla']
    avg_rating FLOAT DEFAULT 0,
    total_reviews INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

-- Real-time Availability (updated by IoT sensors)
CREATE TABLE port_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id UUID NOT NULL REFERENCES stations(id),
    port_number INT NOT NULL,
    status VARCHAR(50) CHECK (status IN ('available', 'charging', 'reserved', 'broken')),
    current_occupant_id UUID REFERENCES users(id),
    estimated_free_time TIMESTAMP,  -- When will port be available?
    updated_at TIMESTAMP DEFAULT now()
);

-- Dynamic Pricing
CREATE TABLE pricing (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id UUID NOT NULL REFERENCES stations(id),
    charger_type VARCHAR(50) NOT NULL,
    base_price_per_kwh DECIMAL(5, 2),
    session_fee DECIMAL(5, 2),
    current_multiplier FLOAT DEFAULT 1.0,  -- 1.0 = base, 1.5 = 50% markup
    effective_from TIMESTAMP,
    reason VARCHAR(100)  -- 'peak_demand', 'scheduled_maintenance', 'promotion'
);

-- Reservations & Bookings
CREATE TABLE reservations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    driver_id UUID NOT NULL REFERENCES users(id),
    station_id UUID NOT NULL REFERENCES stations(id),
    port_number INT NOT NULL,
    status VARCHAR(50) CHECK (status IN ('reserved', 'active', 'completed', 'cancelled')),
    reserved_at TIMESTAMP DEFAULT now(),
    start_time TIMESTAMP,
    estimated_end_time TIMESTAMP,
    actual_end_time TIMESTAMP,
    kwh_delivered DECIMAL(10, 2),
    amount_paid DECIMAL(10, 2),
    payment_status VARCHAR(50) CHECK (payment_status IN ('pending', 'paid', 'refunded')),
    driver_rating INT CHECK (driver_rating BETWEEN 1 AND 5),
    driver_review TEXT,
    owner_response TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- Conversation History (multi-turn context)
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    agent_type VARCHAR(50) CHECK (agent_type IN ('driver', 'owner')),
    started_at TIMESTAMP DEFAULT now(),
    last_message_at TIMESTAMP,
    session_state JSONB,  -- Current state (location, selected_station, etc.)
    messages JSONB[]  -- Array of [{"role": "user", "content": "..."}, ...]
);

-- Message Log (for observability)
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    role VARCHAR(50) CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    tokens_used INT,
    latency_ms INT,
    cost DECIMAL(10, 4),  -- Track per-message cost
    created_at TIMESTAMP DEFAULT now()
);

-- Reviews & Ratings
CREATE TABLE reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id UUID NOT NULL REFERENCES stations(id),
    reviewer_id UUID NOT NULL REFERENCES users(id),
    rating INT CHECK (rating BETWEEN 1 AND 5),
    title VARCHAR(255),
    body TEXT,
    aspects JSONB,  -- {"cleanliness": 5, "speed": 3, "price": 2}
    created_at TIMESTAMP DEFAULT now(),
    helpful_count INT DEFAULT 0  -- Upvotes
);

-- Indexes for fast queries
CREATE INDEX idx_stations_location ON stations USING GIST(location);
CREATE INDEX idx_port_status_station ON port_status(station_id);
CREATE INDEX idx_reservations_driver ON reservations(driver_id);
CREATE INDEX idx_conversations_user ON conversations(user_id);
CREATE INDEX idx_reviews_station ON reviews(station_id);
```

### Vector Database (Embeddings)

```python
# Store station descriptions + reviews as embeddings
# For semantic search: "I want a fast charger near downtown"

from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("ev-stations")

# Embed station profile
station_embedding = embed_model.embed(
    f"EV Charging Station: {station.name}. Location: downtown. "
    f"Charger types: {', '.join(station.charger_types)}. "
    f"Speed: Fast DC. Rating: {station.avg_rating}/5. "
    f"Price: ${pricing.base_price_per_kwh}/kWh. "
    f"Reviews: {top_reviews_summary}"
)

index.upsert([{
    "id": station.id,
    "values": station_embedding,
    "metadata": {
        "name": station.name,
        "lat": station.location.lat,
        "lon": station.location.lon,
        "price": pricing.base_price_per_kwh,
        "rating": station.avg_rating,
        "availability": current_available_ports
    }
}])

# Driver query: "Fast charger near me"
driver_query_embedding = embed_model.embed("Fast charger nearby")
results = index.query(vector=driver_query_embedding, top_k=5)
# Returns 5 most semantically relevant stations
```

---

## Conversational Flow: Driver Example

### Multi-Turn State Machine

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

class DriverConversationFlow:
    """
    Driver journey: Need → Find → Evaluate → Book → Pay → Rate
    """
    
    def __init__(self):
        self.graph = StateGraph(DriverState)
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    async def understand_need(self, state: DriverState) -> DriverState:
        """
        LLM determines: urgent charge? cheapest? fastest?
        """
        # System prompt
        system = """You are a helpful EV charging assistant.
        Understand the driver's needs from their query.
        
        Possible intents:
        1. URGENT: "I need to charge NOW" → Find closest available
        2. ECONOMY: "Cheapest option" → Find lowest price
        3. FAST: "Fastest charging" → Find highest kW/h
        4. SCHEDULED: "I'll arrive in 1 hour" → Check future availability
        """
        
        messages = state["messages"] + [
            {"role": "user", "content": state["query"]}
        ]
        
        response = await self.llm.ainvoke({
            "system": system,
            "messages": messages
        })
        
        # Parse intent
        intent = self._parse_intent(response.content)
        state["intent"] = intent
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        return state
    
    async def search_stations(self, state: DriverState) -> DriverState:
        """
        Query vector DB + real-time availability
        """
        intent = state.get("intent", "default")
        lat, lon = state["location"]
        
        if intent == "URGENT":
            # Find closest available (within 2km)
            query = f"Closest EV charger. Urgent need. {state['query']}"
        elif intent == "ECONOMY":
            query = f"Cheapest EV charging. Low cost. {state['query']}"
        else:
            query = state["query"]
        
        # Vector search
        embedding = embed_model.embed(query)
        vector_results = vector_db.query(embedding, top_k=10)
        
        # Filter by location (within 5km)
        nearby = [r for r in vector_results 
                  if haversine(lat, lon, r.lat, r.lon) < 5]
        
        # Check real-time availability
        available = []
        for station in nearby:
            ports = db.query(
                "SELECT * FROM port_status WHERE station_id = %s AND status IN ('available', 'reserved')",
                station.id
            )
            if ports:
                available.append({
                    "station": station,
                    "available_ports": len(ports),
                    "next_free": ports[0].estimated_free_time
                })
        
        # LLM presents options
        options_text = "\n".join([
            f"- {s['station'].name}: {s['available_ports']} ports, "
            f"${pricing_map[s['station'].id]}/kWh, "
            f"Rating: {s['station'].avg_rating}/5"
            for s in available[:3]
        ])
        
        assistant_message = f"""I found these options for you:
        {options_text}
        
        Which interests you? Or should I search with different criteria?"""
        
        state["retrieved_stations"] = available
        state["messages"].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return state
    
    async def book_and_pay(self, state: DriverState) -> DriverState:
        """
        User selected a station. Confirm details and process payment.
        """
        if "selected_station" not in state:
            return state  # User hasn't selected yet
        
        station = state["selected_station"]
        pricing_info = db.query(
            "SELECT * FROM pricing WHERE station_id = %s",
            station.id
        )
        
        # Ask for confirmation
        system = """You are assisting the driver to complete booking.
        Confirm details:
        - Station
        - Estimated charge time
        - Cost
        - Payment method
        
        Get explicit consent before charging payment method.
        """
        
        messages = state["messages"] + [
            {"role": "user", "content": 
            f"I want to book {station.name}"}
        ]
        
        response = await self.llm.ainvoke({
            "system": system,
            "messages": messages
        })
        
        # If user confirms, process payment
        if "confirm" in response.content.lower():
            reservation = await self._create_reservation(
                driver_id=state["user_id"],
                station_id=station.id,
                estimated_duration=state.get("estimated_duration", 60)
            )
            
            # Payment
            payment = await self._process_payment(
                user_id=state["user_id"],
                amount=reservation.estimated_cost,
                reservation_id=reservation.id
            )
            
            state["booking_details"] = {
                "reservation_id": reservation.id,
                "status": "confirmed",
                "payment_status": "paid"
            }
            
            state["messages"].append({
                "role": "assistant",
                "content": f"✅ Booked! Your reservation ID: {reservation.id}"
            })
        
        return state
    
    async def handle_rate_and_feedback(self, state: DriverState) -> DriverState:
        """
        After charging completes, ask for rating + review
        """
        reservation_id = state.get("booking_details", {}).get("reservation_id")
        if not reservation_id:
            return state
        
        reservation = db.query(
            "SELECT * FROM reservations WHERE id = %s",
            reservation_id
        )
        
        if reservation.status != "completed":
            return state  # Not done yet
        
        # Ask for feedback
        system = """User just finished charging. 
        Ask for:
        1. Rating (1-5 stars)
        2. Brief review
        3. Feedback on specific aspects (speed, cleanliness, price, etc.)
        """
        
        response = await self.llm.ainvoke({
            "system": system,
            "messages": state["messages"] + [
                {"role": "user", "content": "I finished charging"}
            ]
        })
        
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })
        
        return state

# Build the graph
graph = StateGraph(DriverState)
graph.add_node("understand_need", driver_flow.understand_need)
graph.add_node("search_stations", driver_flow.search_stations)
graph.add_node("book_and_pay", driver_flow.book_and_pay)
graph.add_node("rate_feedback", driver_flow.handle_rate_and_feedback)

graph.add_edge("start", "understand_need")
graph.add_edge("understand_need", "search_stations")
graph.add_edge("search_stations", "book_and_pay")
graph.add_edge("book_and_pay", "rate_feedback")
graph.add_edge("rate_feedback", END)

driver_chatbot = graph.compile()
```

---

## Payment Integration

### Stripe for Transactions

```python
import stripe

stripe.api_key = "sk_live_..."

async def process_payment(driver_id: str, amount: float, reservation_id: str):
    """
    Process payment for reservation
    """
    try:
        # Get driver's saved payment method
        driver = db.query("SELECT * FROM users WHERE id = %s", driver_id)
        stripe_customer = stripe.Customer.retrieve(driver.stripe_customer_id)
        
        # Create charge
        charge = stripe.Charge.create(
            amount=int(amount * 100),  # Convert to cents
            currency="usd",
            customer=stripe_customer.id,
            description=f"EV Charging at {reservation.station.name}",
            idempotency_key=f"{reservation_id}-{int(time.time())}"
        )
        
        # Update reservation
        db.query(
            """UPDATE reservations SET 
               payment_status = 'paid', 
               stripe_charge_id = %s 
               WHERE id = %s""",
            charge.id, reservation_id
        )
        
        return {"status": "success", "charge_id": charge.id}
    
    except stripe.error.CardError as e:
        # Declined card
        db.query(
            "UPDATE reservations SET payment_status = 'failed' WHERE id = %s",
            reservation_id
        )
        return {"status": "failed", "reason": "card_declined"}
    
    except Exception as e:
        logger.error(f"Payment error: {e}")
        return {"status": "error", "reason": str(e)}
```

---

## Production Observability

### Monitoring & Cost Tracking

```python
import structlog
from opentelemetry import trace

log = structlog.get_logger()
tracer = trace.get_tracer(__name__)

async def chat_endpoint(user_id: str, message: str):
    """
    Track every conversation turn
    """
    start_time = time.time()
    
    with tracer.start_as_current_span("driver_chat") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("message_length", len(message))
        
        # Call agent
        response = await driver_agent.ainvoke({
            "user_id": user_id,
            "query": message
        })
        
        # Log structured data
        latency_ms = (time.time() - start_time) * 1000
        tokens_used = estimate_tokens(message + response)
        cost = tokens_used * 0.001 / 1000  # $0.001 per 1M tokens (example)
        
        log.info(
            "chat_turn",
            user_id=user_id,
            agent_type="driver",
            query_length=len(message),
            response_length=len(response),
            latency_ms=latency_ms,
            tokens=tokens_used,
            cost=cost,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Save to database
        db.execute(
            """INSERT INTO chat_messages 
               (conversation_id, role, content, tokens_used, latency_ms, cost) 
               VALUES (%s, %s, %s, %s, %s, %s)""",
            conversation_id, "assistant", response, tokens_used, latency_ms, cost
        )
        
        return response
```

### Dashboards

```
Cost per user:      $0.12/month (avg)
Latency p95:        1200ms
Booking conversion: 73% (search → book)
Payment success:    97.2%
Avg rating:         4.6/5 stars
Churn (drivers):    8% monthly

Alerts:
- Latency p95 > 2s
- Payment success < 95%
- Cost per booking > $0.50
```

---

## Interview Answers

### "Walk me through the architecture"

**Script (3 min):**

"Two separate agents built on LangGraph, sharing PostgreSQL backend. Driver agent: understands intent (urgent/cheap/fast), vector-searches 10K+ stations, checks real-time port availability via IoT, facilitates booking and payment via Stripe. Station owner agent: monitors utilization, suggests dynamic pricing based on demand, handles inquiries.

Conversation state stored in PostgreSQL with JSONB. Every turn logged: tokens, latency, cost. Multi-turn context passed to Claude via messages array, not full history (prevents context rot). Payment integration via Stripe idempotency keys (no double charges).

Observability: OpenTelemetry traces to CloudWatch, cost tracking per message, Grafana dashboards for conversion funnel and P95 latency."

### "How did you prevent duplicate charges?"

**Script (2 min):**

"Stripe idempotency keys. Before processing payment, generate key from `reservation_id + timestamp`. If network fails mid-charge, retry with same key — Stripe returns same charge, not duplicate. Also: database transaction for reservation → payment → confirmation (atomic all-or-nothing).

Redis distributed lock on reservation slot: when driver selects port, acquire lock. If payment fails, release lock. Prevents two drivers booking same port simultaneously."

### "How did you optimize costs?"

**Script (2 min):**

"Prompt caching for system prompts (driver agent uses same 500-token system prompt every turn). Semantic caching: if driver query is similar to recent query, return cached station list instead of re-querying vector DB. Model routing: simple queries use Haiku (cheap), complex negotiations with owners use Sonnet.

Token budgeting: cap conversation history to last 10 turns + 5 most relevant past stations. This prevents context bloat and saves ~40% tokens. Measured ROI: -$0.08/conversation after optimizations."

### "How would you handle a surge (10x traffic)?"

**Script (2 min):**

"Infrastructure: PostgreSQL scales to 10K ops/sec with connection pooling (pgBouncer). Redis for session cache, vector DB (Pinecone) is serverless (auto-scales). LLM calls are stateless, can parallelize across regions.

Graceful degradation:
- If LLM is slow (rate limited), queue requests, show 'estimating options...'
- If payment processor is down, ask driver to retry later (reservation holds port for 10 min)
- If vector DB is slow, fall back to geographic radius search (PostgreSQL PostGIS)

Circuit breakers on Stripe (if failures > 10%, stop processing temporarily, queue for retry).

Actual metrics: currently ~50 concurrent drivers, each message takes 1-2s, 98% availability."

---

## Code Scaffolding

### FastAPI Backend

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.websocket("/ws/driver/{user_id}")
async def websocket_driver(websocket: WebSocket, user_id: str):
    """
    WebSocket for real-time driver chat
    """
    await websocket.accept()
    
    conversation_id = str(uuid.uuid4())
    db.execute(
        """INSERT INTO conversations 
           (id, user_id, agent_type) 
           VALUES (%s, %s, %s)""",
        conversation_id, user_id, "driver"
    )
    
    try:
        while True:
            # Receive message from driver
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Get current conversation state
            state = db.query(
                "SELECT session_state FROM conversations WHERE id = %s",
                conversation_id
            )[0].session_state
            
            # Add to messages array
            state["messages"].append({
                "role": "user",
                "content": message["text"]
            })
            
            # Stream response
            async for chunk in driver_agent.astream(state):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })
            
            # Save updated state
            db.execute(
                """UPDATE conversations 
                   SET session_state = %s, last_message_at = now() 
                   WHERE id = %s""",
                json.dumps(state), conversation_id
            )
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/api/driver/reserve")
async def reserve_station(user_id: str, station_id: str, port_number: int):
    """
    Direct API for booking (bypasses chat if needed)
    """
    # Acquire lock
    lock_key = f"station:{station_id}:port:{port_number}"
    if not redis.set(lock_key, user_id, nx=True, ex=30):
        return {"error": "Port just booked, sorry!"}
    
    try:
        # Create reservation
        reservation = Reservation(
            driver_id=user_id,
            station_id=station_id,
            port_number=port_number,
            status="reserved"
        )
        db.add(reservation)
        db.commit()
        
        return {
            "status": "success",
            "reservation_id": reservation.id
        }
    finally:
        redis.delete(lock_key)
```

### Deployment (Docker + Kubernetes)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

# Run with Gunicorn + 4 workers
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
```

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ev-chatbot
spec:
  replicas: 3  # Auto-scale to 10 under load
  selector:
    matchLabels:
      app: ev-chatbot
  template:
    metadata:
      labels:
        app: ev-chatbot
    spec:
      containers:
      - name: ev-chatbot
        image: gcr.io/project/ev-chatbot:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: db-url
        - name: STRIPE_API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: stripe-key
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
```

---

## What Interviewers Will Probe

1. **Scaling:** "What happens at 100K concurrent drivers?" → Kubernetes, multi-region, circuit breakers
2. **Payments:** "How do you handle failed charges?" → Idempotency, retries, refund logic
3. **State:** "How do you prevent lost context?" → JSONB storage, message indexing, recovery
4. **Latency:** "Why is chat slow?" → Profile with OpenTelemetry, optimize vector search, reduce token count
5. **Cost:** "How much does this cost?" → Estimate tokens/user, optimize prompts, use cheaper models for simple queries
6. **Compliance:** "What about data privacy?" → PII redaction, encryption at rest, audit logs

---

## Red Flag Avoidance

❌ "We just stored everything in memory" → Scale = crash  
✅ "PostgreSQL with JSONB for conversation state, recoverable across reboots"

❌ "We called Stripe synchronously on every message" → Latency  
✅ "Async payment processing, queue if needed, acknowledge to user immediately"

❌ "Same prompt for all users" → Inflexible  
✅ "Different system prompts for driver vs owner agent, personalized based on user history"

❌ "We didn't track costs" → Surprised by bill  
✅ "Every message logged: tokens, latency, cost per user monthly"

