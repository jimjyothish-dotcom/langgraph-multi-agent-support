# LangGraph Multi-Agent Support System
### Galaxy Telecom — AI-Powered Customer Support

A production-grade multi-agent AI system built with LangGraph and LangChain,
demonstrating coordinator-worker orchestration, RAG-based knowledge retrieval,
LLMOps observability, and intelligent human escalation.

📄 **[Portfolio Document (PDF)](langgraph-multi-agent-support.pdf)** — full write-up with architecture, sample interactions, and observability screenshots

---

## Overview

Galaxy Telecom receives hundreds of customer support requests daily covering
billing queries, technical faults, contract renewals, and product information.
This system replaces a linear, rule-based routing approach with a dynamic
multi-agent architecture where an AI coordinator classifies customer intent
at runtime and delegates to the appropriate specialist worker agent.

---

## Architecture

```
Customer Message
        │
        ▼
┌─────────────────┐
│   Coordinator   │  ← Intent classification (Claude Haiku, temp=0)
│     Agent       │
└────────┬────────┘
         │ Conditional Edge — routes based on intent at runtime
         │
    ┌────┴─────────────────────────────────┐
    │         │           │        │       │
    ▼         ▼           ▼        ▼       ▼
Billing   Technical   Retention   KB   Escalation
 Agent      Agent       Agent    Agent   Handler
(Haiku)   (Haiku)    (Sonnet)  (Haiku)  (No LLM)
    │         │           │        │
    └────┬────┴───────────┘        │
         │                         │
         ▼                         ▼
   ┌───────────┐             Slack Alert +
   │ Responder │             Audit Log
   │   Agent   │
   └─────┬─────┘
         │
         ▼
  Final Response +
   Audit Log
```

---

## Agent Roles

| Agent | Responsibility | Model | Temperature |
|---|---|---|---|
| Coordinator | Intent classification and routing | Claude Haiku | 0 |
| Billing Agent | Billing and payment queries | Claude Haiku | 0.3 |
| Technical Agent | Faults and connectivity issues | Claude Haiku | 0.3 |
| Retention Agent | Cancellations and renewals | Claude Sonnet | 0.4 |
| KB Agent | RAG-based product knowledge retrieval | Claude Haiku | 0.3 |
| Escalation Handler | Human escalation — Slack alert + audit log | None | — |
| Responder | Final response assembly and brand tone | Claude Haiku | 0.3 |

---

## Model Selection Strategy

A tiered model selection strategy is applied based on task complexity
and business impact — rather than a one-size-fits-all approach.

**Claude Haiku** is used for structured, constrained tasks where
determinism and cost efficiency are the priority.

**Claude Sonnet** is reserved for the retention agent, where response
quality and empathetic tone directly correlate with customer lifetime value.
A warmer, more naturalistic response meaningfully improves retention outcomes
compared to a structured, bullet-pointed reply.

This pattern reduces token cost by approximately 80% compared to running
all agents on Sonnet, while maintaining quality exactly where it matters.

---

## Key Design Decisions

**Dynamic control flow**
Unlike traditional workflow tools (n8n, Make.com) where routing is
determined at design time, LangGraph uses conditional edges with router
functions evaluated at runtime. The coordinator classifies intent from
the live customer message and the graph routes accordingly.

**Graceful degradation**
When the coordinator cannot classify intent with sufficient confidence,
the system escalates to a human agent rather than guessing. A Slack alert
is sent to the human agent team with full context, and the interaction
is logged to the audit database.

**RAG-grounded KB responses**
The KB agent performs semantic search against a Qdrant vector store
containing Galaxy Telecom product and policy documents. Claude generates
responses strictly grounded in retrieved context — it will not fabricate
product details, prices, or policies.

**Dual-layer observability**
LangFuse traces every LLM call with token counts, latency, and full
execution traces. A separate SQLite audit log captures business-level
outcomes — intent, worker used, resolution, and escalation status —
for every interaction regardless of outcome.

**Externalised configuration**
All model names, temperatures, and API credentials are externalised
to environment variables. Model selection and tuning requires no
code changes or redeployment.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph 1.x |
| LLM framework | LangChain |
| LLM provider | Anthropic Claude API (Haiku + Sonnet) |
| Vector database | Qdrant Cloud |
| Embedding model | sentence-transformers (all-MiniLM-L6-v2) |
| LLM observability | LangFuse |
| Audit logging | SQLite |
| Escalation alerts | Slack Incoming Webhooks |
| Language | Python 3.11+ |

---

## Project Structure

```
langgraph-multi-agent-support/
├── agents/
│   ├── coordinator.py      # Intent classification and routing
│   ├── billing.py          # Billing and payment queries
│   ├── technical.py        # Faults and connectivity issues
│   ├── retention.py        # Cancellations and renewals
│   ├── kb.py               # RAG-based knowledge retrieval
│   ├── escalation.py       # Human escalation handler
│   └── responder.py        # Final response assembler
├── graph/
│   ├── state.py            # Shared GalaxyState TypedDict schema
│   └── workflow.py         # LangGraph StateGraph definition
├── observability/
│   └── langfuse_config.py  # LangFuse callback handler setup
├── audit/
│   └── logger.py           # SQLite audit log — write and read
├── data/
│   └── sample_requests.json
├── screenshots/            # Project screenshots for portfolio
├── main.py                 # Entry point and CLI test harness
├── requirements.txt
├── langgraph-multi-agent-support.pdf  # Portfolio document
└── .env.example            # Environment variable template
```

---

## Setup and Installation

### Prerequisites
- Python 3.11+
- Anthropic API key
- Qdrant Cloud account (free tier)
- LangFuse Cloud account (free tier)
- Slack workspace with Incoming Webhooks enabled

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd langgraph-multi-agent-support

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Variables

```env
# Anthropic
ANTHROPIC_API_KEY=your_key_here

# LangFuse
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key_here
LANGFUSE_HOST=https://cloud.langfuse.com

# Qdrant
QDRANT_URL=your_cluster_url_here
QDRANT_API_KEY=your_key_here
QDRANT_COLLECTION_NAME=galaxy-telecom-products

# Slack
SLACK_WEBHOOK_URL=your_webhook_url_here

# LLM Model Configuration — tiered by task complexity
CLAUDE_MODEL_STANDARD=claude-haiku-4-5-20251001
CLAUDE_MODEL_ADVANCED=claude-sonnet-4-6

# Temperature Configuration — tuned per agent role
CLAUDE_TEMPERATURE_PRECISE=0
CLAUDE_TEMPERATURE_BALANCED=0.3
CLAUDE_TEMPERATURE_CREATIVE=0.4
```

### Run

```bash
python main.py
```

---

## Sample Interactions

**Billing query**
```
Customer : My bill this month is much higher than usual, can you help?
Intent   : billing → billing_node
Response : Structured billing guidance with contact details
```

**Technical fault**
```
Customer : My broadband has been down since yesterday morning.
Intent   : technical → technical_node
Response : Step-by-step troubleshooting with escalation path
```

**Retention**
```
Customer : I am thinking about cancelling my contract with Galaxy Telecom.
Intent   : retention → retention_node
Response : Empathetic response exploring concerns, loyalty offers presented
```

**Knowledge base**
```
Customer : What 5G plans do you offer and what is the coverage like?
Intent   : kb → kb_node
Response : RAG-grounded response from Qdrant vector store
```

**Human escalation**
```
Customer : Purple elephant dancing in the moonlight.
Intent   : unknown → escalation_node
Response : Polite holding message + Slack alert to human agent team
```

---

## Observability

**LangFuse** — traces every LLM call across all agents with token counts,
latency per node, and full execution traces. Access via cloud.langfuse.com.

**SQLite audit log** — records every interaction with intent classification,
worker used, final response, escalation flag, and timestamp.

Query the audit log:
```bash
python -c "from audit.logger import read_logs; import json; [print(json.dumps(log, indent=2)) for log in read_logs()]"
```

---

## Skills Demonstrated

- LangGraph multi-agent orchestration
- Coordinator-worker agent pattern
- Stateful graph execution with TypedDict schema
- Conditional edge routing with runtime decision making
- RAG pipeline with Qdrant vector store
- Tiered model selection strategy
- LLMOps and AI observability with LangFuse
- Human-in-the-loop escalation with Slack integration
- Structured audit logging
- Externalised configuration management

---

## Credits

Galaxy Telecom CRM AI Support System — multi-agent orchestration with
LangGraph and LangChain for intelligent customer support automation.
---

## Docker

The system is published as a Docker image on Docker Hub.
All secrets are loaded at runtime via `.env` — nothing is baked into the image.

### Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- A `.env` file — copy `.env.example` and fill in your API keys:
  ```
  cp .env.example .env
  # Edit .env — add Anthropic, Qdrant, LangFuse, and Slack keys
  ```

---

### Option A — Pull from Docker Registry and run with Compose

This is the fastest route. No local build required.

```bash
# Pull the image
docker pull galaxy-telecom/langgraph-multi-agent-support:latest

# Run with Docker Compose (reads .env automatically)
docker compose up
```

The SQLite audit log is persisted in a named Docker volume (`langgraph-audit-data`)
and survives container restarts.

---

### Option B — Build locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd langgraph-multi-agent-support

cp .env.example .env
# Edit .env with your keys

# Build and run
docker compose up --build
```

Or build the image directly:

```bash
docker build -t galaxy-telecom/langgraph-multi-agent-support:latest .
docker run --env-file .env -v langgraph-audit-data:/app/data \
  galaxy-telecom/langgraph-multi-agent-support:latest
```

---

### Read the SQLite audit log from a running container

```bash
docker exec langgraph-support \
  python -c "from audit.logger import read_logs; import json; [print(json.dumps(log, indent=2)) for log in read_logs()]"
```

---

### Image details

| Property     | Value                                                         |
|--------------|---------------------------------------------------------------|
| Base image   | `python:3.11-slim`                                            |
| Build        | Multi-stage (builder + runtime — no compilers in final image) |
| Run as       | Non-root user (`appuser`, UID 1001)                           |
| Secrets      | Loaded via `.env` at runtime — never baked into image         |
| Persistence  | SQLite audit log → named volume `langgraph-audit-data`        |
| Docker Hub   | `my-org/langgraph-multi-agent-support:latest`            |

