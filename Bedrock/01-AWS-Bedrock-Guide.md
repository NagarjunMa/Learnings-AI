# AWS Bedrock — Foundation Models as a Service

Bedrock is AWS's managed API for foundation models (Claude, Llama, Cohere, Stability, etc.). Instead of managing model infrastructure, you call an API. Critical for regulated industries (financial services, healthcare) because data stays on AWS.

---

## Mental Model

Bedrock = managed hosted models without OpenAI's API calls leaving AWS.

```
Traditional LLM:
User → Internet → OpenAI API → OpenAI servers → Response
                  (data leaves your VPC)

Bedrock:
User → AWS VPC → Bedrock API (AWS-managed) → Response
                  (data stays in AWS)
```

**Why this matters for financial services:**
- Data residency compliance (data never touches 3rd-party servers)
- HIPAA/FedRAMP certified (regulated industries)
- Cost (no egress fees; pay per-token to AWS)
- Latency (models hosted in same region)

---

## Bedrock Core APIs

### 1. Basic Model Invocation

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Invoke Claude
response = client.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-06-01",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ]
    })
)

# Parse response
result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

**Available models:**
- Claude 3.5 Sonnet (best quality, balanced)
- Claude 3 Opus (most capable, higher latency)
- Llama 2/3 (open-source, lower cost)
- Cohere Command (good for text generation, cheaper)
- Mistral (open-source, very fast)
- Stability AI (image generation)

**Pricing:** Per-token (input + output). Claude 3.5 Sonnet: $3/1M input, $15/1M output.

---

### 2. Bedrock Knowledge Bases (RAG)

**Problem:** You want RAG but don't want to manage vector DBs or embeddings.

**Solution:** Bedrock Knowledge Bases. Upload documents → automatic chunking, embedding, retrieval.

```python
# Upload documents to S3
s3.upload_file("earnings_report.pdf", bucket="my-docs", "earnings_report.pdf")

# Create knowledge base
kb_response = client.create_knowledge_base(
    name="financial-kb",
    roleArn="arn:aws:iam::ACCOUNT:role/BedrockKBRole",
    knowledgeBaseConfiguration={
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": "arn:aws:bedrock:REGION::foundation-model/amazon.titan-embed-text-v2:0"
        }
    }
)

kb_id = kb_response["knowledgeBaseId"]

# Create data source
data_source = client.create_data_source(
    knowledgeBaseId=kb_id,
    name="financial-docs",
    dataSourceConfiguration={
        "s3Configuration": {
            "bucketArn": "arn:aws:s3:::my-docs"
        }
    }
)

# Sync documents (automatic chunking and embedding)
client.start_ingestion_job(
    knowledgeBaseId=kb_id,
    dataSourceId=data_source["dataSourceId"]
)
```

**Bedrock does:**
- Document chunking (automatic, configurable)
- Embedding (using Amazon Titan or custom)
- Vector storage (managed by AWS)
- Retrieval (similarity search)

**You get:**
- Automatic embedding updates when documents change
- Built-in metadata filtering
- No vector DB management
- Integrated with Bedrock Agents

**Use case:** Quick RAG without infrastructure. Automatic everything.

---

### 3. Bedrock Agents (Agentic AI)

**Problem:** You want an AI agent that uses tools and retrieves documents.

**Solution:** Bedrock Agents. Define tools → agent handles orchestration.

```python
# Define tools (actions agent can take)
tools = [
    {
        "toolName": "get_revenue",
        "description": "Get company revenue for a given year",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Year (e.g., 2025)"
                    }
                },
                "required": ["year"]
            }
        }
    },
    {
        "toolName": "search_knowledge_base",
        "description": "Search company documents for information",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Create agent
agent = client.create_agent(
    agentName="financial-analyst",
    agentRoleArn="arn:aws:iam::ACCOUNT:role/BedrockAgentRole",
    foundationModel="anthropic.claude-3-5-sonnet-20241022-v2:0",
    description="Financial Q&A agent with document search",
    tools=tools,
    knowledgeBaseConfigurations=[
        {
            "knowledgeBaseId": kb_id,
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "maxResults": 5,
                    "overrideSearchType": "HYBRID"  # Keyword + semantic
                }
            }
        }
    ]
)

agent_id = agent["agent"]["agentId"]
agent_alias_id = agent["agentAliasId"]
```

**Agent workflow:**
```
1. User: "What was our Q3 2025 revenue?"
2. Agent decides: Need search_knowledge_base tool
3. Calls tool: search_knowledge_base(query="Q3 2025 revenue")
4. Knowledge base returns: "Q3 2025 revenue: $500M"
5. Agent reads tool result, generates answer
6. Response: "Our Q3 2025 revenue was $500M"
```

**Invoke the agent:**

```python
# Invoke agent
response = client.invoke_agent(
    agentId=agent_id,
    agentAliasId=agent_alias_id,
    sessionId="session-123",
    inputText="What was our revenue growth?"
)

# Stream response
for event in response["output"]:
    if "text" in event:
        print(event["text"])
```

**Agent features:**
- **Tool calling:** Define actions; agent calls them as needed
- **Memory:** Session ID preserves conversation history
- **Knowledge base integration:** Automatic RAG
- **Streaming:** Stream responses as they generate

**Bedrock Agents vs. LangGraph:**
- **Bedrock:** Simpler, AWS-native, less flexible
- **LangGraph:** More control, local-first, multi-model support

Use Bedrock Agents if you're already on AWS and want plug-and-play. Use LangGraph for complex multi-step orchestration.

---

## Bedrock Guardrails

**Problem:** Financial agents might say inappropriate things or leak PII.

**Solution:** Bedrock Guardrails. Define content policies.

```python
# Create guardrail
guardrail = client.create_guardrail(
    name="financial-guardrail",
    description="Prevents PII leakage and regulatory violations",
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "illegal_advice",
                "definition": "Advice that would violate securities law",
                "examples": [
                    "I can help you hide assets from the IRS",
                    "This insider trading opportunity..."
                ],
                "type": "DENY"
            }
        ]
    },
    contentPolicyConfig={
        "filtersConfig": [
            {
                "type": "SEXUAL",
                "strength": "HIGH"
            },
            {
                "type": "VIOLENCE",
                "strength": "MEDIUM"
            }
        ]
    },
    sensitiveInformationPolicyConfig={
        "piiEntitiesConfig": [
            {
                "type": "CREDIT_CARD_NUMBER",
                "action": "ANONYMIZE"  # Or "BLOCK"
            },
            {
                "type": "EMAIL_ADDRESS",
                "action": "BLOCK"
            }
        ],
        "regexesConfig": [
            {
                "name": "ssn_pattern",
                "pattern": r"\d{3}-\d{2}-\d{4}",
                "action": "BLOCK"
            }
        ]
    }
)

guardrail_id = guardrail["guardrailId"]

# Use guardrail with model
response = client.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    guardrailIdentifier=guardrail_id,
    guardrailVersion="DRAFT",
    body=json.dumps({...})
)
```

**Guardrail actions:**
- **BLOCK:** Reject the input/output
- **ANONYMIZE:** Replace PII with placeholders
- **FLAG:** Allow but log for review

**Use case:** Compliance. Required in financial services.

---

## Bedrock Data Automation

**Problem:** You have thousands of PDFs. Need to extract structured data.

**Solution:** Bedrock Data Automation. Upload documents → get structured output (JSON).

```python
import json

# Create classification job
job = client.create_classification_job(
    jobName="invoice-classification",
    jobConfig={
        "inputDocuments": {
            "s3Uri": "s3://my-docs/invoices/"
        },
        "outputConfiguration": {
            "s3Uri": "s3://my-outputs/"
        },
        "documentClassificationConfig": {
            "classificationLabels": [
                "invoice",
                "receipt",
                "purchase_order",
                "credit_memo"
            ]
        }
    }
)

# Wait for job completion
# Bedrock returns classified documents in S3
```

**What it does:**
- Classify documents (what type is it?)
- Extract entities (invoice number, date, total)
- Table extraction (convert PDF tables to structured format)
- Multi-page handling

**Output:**
```json
{
  "classification": "invoice",
  "confidence": 0.98,
  "extracted_data": {
    "invoice_number": "INV-2025-001",
    "date": "2025-04-11",
    "total": 5000.00,
    "items": [
      {"description": "Service", "quantity": 1, "price": 5000}
    ]
  }
}
```

**Use case:** Document processing at scale. No custom vision models needed.

---

## Production Pattern — Bedrock in Financial Services

```python
# 1. Setup: Create KB + Agent + Guardrail

kb_id = create_knowledge_base("financial-kb", embedding_model="amazon.titan-embed-text-v2:0")
agent_id = create_agent("financial-analyst", kb_id=kb_id)
guardrail_id = create_guardrail("compliance-guardrail", pii_action="BLOCK", topics=["illegal_advice"])

# 2. Ingest: Upload documents
upload_documents_to_s3(bucket="financial-docs", documents=quarterly_reports)
start_kb_sync(kb_id)  # Automatic chunking + embedding

# 3. Query: User asks question
user_query = "What was our revenue growth?"

response = client.invoke_agent(
    agentId=agent_id,
    agentAliasId=agent_alias_id,
    sessionId=user_id,
    inputText=user_query,
    guardrailConfig={
        "guardrailIdentifier": guardrail_id,
        "guardrailVersion": "DRAFT"
    }
)

# 4. Log: Audit trail
log_query(user_id, user_query, response["output"], timestamp=now)

# 5. Monitor: Track cost, accuracy, compliance
monitor_agent_performance(agent_id)
```

---

## Bedrock vs. OpenAI in Financial Services

| Aspect | Bedrock | OpenAI |
|---|---|---|
| **Data residency** | Stays in AWS | Goes to OpenAI servers |
| **Compliance** | FedRAMP, HIPAA | Not FedRAMP |
| **Compliance cost** | Included | Not suitable |
| **Pricing** | Per token | Per token |
| **Inference latency** | 50-200ms | 100-300ms |
| **Agent framework** | Bedrock Agents (simple) | OpenAI + LangGraph (flexible) |
| **Knowledge base** | KB service (built-in) | You build RAG |
| **Model selection** | 10+ models | GPT only |

**For financial services:** Bedrock is often required by compliance. Use it.

---

## System Design Interview Pattern

**Question:** "Design an AI system for a bank's document processing."

**Your answer:**

```
1. Problem: 100K documents/month (PDFs, images). Need extraction, classification, compliance.

2. Architecture:
   - Bedrock Knowledge Base: Auto-chunks, embeds, stores
   - Bedrock Agent: Routes queries, uses tools
   - Bedrock Guardrails: Blocks PII, flags illegal advice
   - Bedrock Data Automation: Extracts structured data from PDFs

3. Workflow:
   - Ingest: S3 → KB sync (automatic chunking)
   - Query: Agent searches KB, uses extraction tools
   - Compliance: Guardrails block PII, log all decisions
   - Audit: Every interaction logged (who asked, what returned)

4. Cost: ~$0.003 per query (token-based)

5. Compliance:
   - Data never leaves AWS
   - PII redaction built-in
   - FedRAMP certified
   - Audit logs for regulators

6. Scaling:
   - 1M documents in KB
   - <500ms query latency
   - Auto-scaling handled by AWS
```

This shows: **end-to-end thinking + compliance-first + Bedrock expertise**.
