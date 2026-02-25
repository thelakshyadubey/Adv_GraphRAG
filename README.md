# Hybrid RAG — GraphRAG + KAG + CAG

A unified document intelligence system that combines three retrieval paradigms — **Cache-Augmented Generation (CAG)**, **Knowledge-Augmented Generation (KAG)**, and **GraphRAG community search** — under a single smart router.

---

## What It Does

Upload any document (PDF, DOCX, TXT, CSV). Ask questions in natural language. The system automatically picks the fastest and most accurate retrieval path based on your question type.

```
Simple / repeated question   →  CAG   (~2ms, from cache)
Complex / multi-hop question →  KAG   (~3500ms, graph reasoning)
Normal question              →  KAG Simple (~2000ms, hybrid search)
Thematic / overview question →  Global (~2500ms, community summaries)
```

---

## System Architecture

```
┌─────────────────────────────────┐
│         LAYER 1: USER           │  Browser — Upload · Ask · Cache tabs
├─────────────────────────────────┤
│         LAYER 2: API            │  FastAPI Server — routes all requests
├─────────────────────────────────┤
│       LAYER 3: PROCESSING       │  Ingestion pipeline + Retrieval logic
├─────────────────────────────────┤
│        LAYER 4: STORAGE         │  Qdrant · Neo4j · Cache (RAM + Redis)
├─────────────────────────────────┤
│      LAYER 5: EXTERNAL AI       │  Groq LLM · MiniLM Embedding Model
└─────────────────────────────────┘
```

---

## How It Works

### On Upload

```
Parse → Chunk → Embed → Extract Entities → Store → Community Detection → Summarize
```

1. **Parse** — reads PDF/DOCX/TXT, extracts plain text (PyMuPDF, ~5ms/page)
2. **Chunk** — splits text at topic boundaries using sentence-level semantic similarity
3. **Embed** — converts each chunk to a 384-dim vector (all-MiniLM-L6-v2, local CPU)
4. **Extract** — sends each chunk to Groq LLM → gets entities + relations as JSON
5. **Store** — chunks + vectors → Qdrant, entities + relations → Neo4j, cross-links → both (Mutual Index)
6. **Community Detection** — clusters related entities with Union-Find, generates LLM summary per cluster → stored in Qdrant
7. **Summarize** — doc-level and section-level summaries → stored in Qdrant
8. **Auto-preload** — if document is small enough, full text is loaded into RAM for instant future answers

---

### On Question

```
Question → Smart Router → Pick Path → Retrieve → Context Builder → Groq LLM → Answer
```

**Router decision (first match wins):**

```
1. Exact question seen before?        → CAG  (cache hit)
2. Question < 8 words, no "?"?        → CAG  (short factual)
3. All selected docs preloaded?       → CAG  (doc in memory)
4. Contains thematic keywords?        → GLOBAL
   (overview, main theme, summarize, across all...)
5. Contains complex keywords?         → KAG
   (compare, why, relationship, explain, difference...)
6. 2+ "and" clauses or > 25 words?   → KAG
7. Default                            → KAG_SIMPLE
```

---

## The Four Retrieval Paths

### CAG — Cache-Augmented Generation
Never touches Qdrant or Neo4j.
- **Exact cache hit** → return answer in 0.001ms
- **Preloaded doc** → full doc text in RAM → one LLM call → answer in ~1500ms
- Every answer is automatically cached in L1 (RAM) + L2 (Redis, TTL 1hr)

### KAG_SIMPLE — Single Hybrid Search
One round of retrieval using two methods in parallel:
- **Vector search** — embed query → find closest chunks in Qdrant
- **Entity search** — query Neo4j fulltext index → get entities → fetch their chunks
- Results merged with RRF → top chunks → LLM → answer

### KAG — Multi-Step Knowledge-Augmented Generation
LLM generates a retrieval plan first, then executes it:
- Breaks complex query into sub-queries with operator assignments
- Independent steps run in parallel (asyncio.gather)
- Dependent steps wait for their dependencies
- Six operators available: VECTOR_SEARCH, ENTITY_SEARCH, HYBRID, GRAPH_EXACT, MULTI_HOP, COMMUNITY_SEARCH
- All step results merged via RRF → LLM → answer

### GLOBAL — Community Search (GraphRAG)
For broad thematic questions, uses pre-built community summaries:
- Searches Qdrant summaries collection per selected doc in parallel
- Each doc contributes its top-3 community summaries
- RRF merge → LLM sees bird's eye view of all themes → thematic answer

---

## The Three Databases

| Database | What it stores | Why needed |
|---|---|---|
| **Qdrant** | Chunk vectors + summaries (384-dim) | Semantic similarity search |
| **Neo4j** | Entity nodes + relation edges | Graph traversal, multi-hop reasoning |
| **Cache** (RAM + Redis) | Cached answers + preloaded doc text | Skip all databases on repeat questions |

### The Mutual Index (Bridge Between Qdrant and Neo4j)

Every chunk in Qdrant stores the IDs of its Neo4j entities.
Every entity in Neo4j stores the IDs of its Qdrant chunks.

```
Find entity in Neo4j → jump to source chunks in Qdrant
Find chunk in Qdrant → jump to its entities in Neo4j
```

This bidirectional link is what enables the KAG path to combine graph reasoning with semantic retrieval.

---

## RRF — How Results Are Merged

**Reciprocal Rank Fusion** merges multiple ranked lists without needing manual weights:

```
score(chunk) = Σ 1 / (60 + rank_in_each_list)

Chunk appearing in rank 1 of vector AND rank 1 of entity:
  score = 1/61 + 1/61 = 0.0328  ← ranked highest

Chunk appearing only in rank 1 of vector:
  score = 1/61 = 0.0164  ← ranked lower
```

Chunks found by **multiple retrieval methods** rank higher — they are more trustworthy.

---

## Technology Choices

| Technology | Purpose | Why chosen |
|---|---|---|
| **FastAPI** | API server | Async-native, auto Swagger docs, Pydantic validation |
| **Qdrant** | Vector database | MatchAny filter, async client, free cloud tier |
| **Neo4j AuraDB** | Graph database | Native graph traversal, Cypher, fulltext index |
| **Redis** | L2 answer cache | Sub-ms lookup, TTL per key, survives restarts |
| **Groq (Llama 3.1 8B)** | LLM inference | 800 tok/s (vs OpenAI ~50), free tier, 128K context |
| **all-MiniLM-L6-v2** | Embedding model | 22M params, 384-dim, runs on CPU, Apache 2.0 |
| **PyMuPDF** | PDF parsing | C bindings, 50x faster than pdfplumber |
| **Union-Find** | Community clustering | O(α(n)) ≈ O(1), no GPU, works on edge list |
| **RRF** | Result merging | Parameter-free, rewards multi-method agreement |
| **structlog** | Logging | Structured JSON, silenceable per library |

---

## Project Structure

```
hybrid_rag/
├── api/
│   ├── main.py              ← entry point, starts server
│   └── routes.py            ← all HTTP endpoints
├── ingestion/
│   ├── pipeline.py          ← orchestrates full ingestion
│   ├── parser.py            ← file → plain text
│   ├── chunker.py           ← text → semantic chunks
│   ├── embedder.py          ← chunks → 384-dim vectors
│   ├── extractor.py         ← chunks → entities + relations (Groq)
│   ├── summarizer.py        ← chunks → doc/section summaries
│   ├── community_builder.py ← entities → community summaries (GraphRAG)
│   └── mutual_index.py      ← links Neo4j nodes ↔ Qdrant chunks
├── retrieval/
│   ├── router.py            ← decides CAG / KAG / KAG_SIMPLE / GLOBAL
│   ├── operators.py         ← 6 retrieval operators
│   ├── kag_planner.py       ← LLM-based query decomposition
│   ├── cag_engine.py        ← cache-based answering
│   ├── context_builder.py   ← formats context for LLM
│   └── rrf_merger.py        ← Reciprocal Rank Fusion
├── storage/
│   ├── schema.py            ← all Pydantic data models
│   ├── qdrant_client.py     ← vector DB operations
│   ├── neo4j_client.py      ← graph DB operations
│   └── cache_manager.py     ← L1 RAM + L2 Redis cache
├── config.py                ← all settings from .env
├── logging_config.py        ← structured log setup
├── frontend/
│   └── index.html           ← full UI (single file)
└── .env                     ← secrets and config (not committed)
```

---

## Setup

### Prerequisites
- Python 3.11+
- Qdrant Cloud account (free tier)
- Neo4j AuraDB account (free tier)
- Groq API key (free tier)
- Redis (portable — no install needed)

### Install

```bash
git clone https://github.com/thelakshyadubey/Adv_GraphRAG.git
cd GraphRAG_KAG_CAG
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r hybrid_rag/requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your credentials:

```ini
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

QDRANT_URL=https://your-instance.qdrant.io
QDRANT_API_KEY=your-key

GROQ_API_KEY=your-key

REDIS_URL=redis://localhost:6379

HF_TOKEN=your-token           # optional, for private HF models

CAG_CONTEXT_LIMIT=20000       # max words for CAG auto-preload
```

### Start Redis (portable, no install)

```powershell
Start-Process "C:\path\to\redis-server.exe" -WindowStyle Minimized
```

### Run

```bash
python -m hybrid_rag.api.main
```

Server starts at `http://localhost:8000`
- UI: `http://localhost:8000/ui/index.html`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/upload` | Upload file bytes + doc_id |
| `POST` | `/ingest` | Ingest by server-side file path |
| `POST` | `/query` | Ask a question |
| `GET` | `/health` | Check all service connections |
| `POST` | `/cache/preload` | Force-preload a doc into CAG memory |
| `POST` | `/cache/invalidate` | Clear cache for a doc |
| `GET` | `/cache/inspect` | See current cache state |

### Query Request

```json
{
  "query": "How does multi-head attention work?",
  "doc_ids": ["transformer_paper", "bert_paper"]
}
```

### Query Response

```json
{
  "answer": "Multi-head attention computes...",
  "sources": [{"doc_id": "transformer_paper", "page": 3}],
  "path_used": "KAG_SIMPLE",
  "latency_ms": 1842.5
}
```

---

## Key Design Principles

**1. Pre-compute everything expensive at ingestion time**
Entity extraction, community summaries, and doc summaries all require LLM calls. They happen once at upload. Query time is fast because the heavy work is already done.

**2. Cache aggressively, check cheapest option first**
L1 RAM → L2 Redis → Qdrant → Neo4j. A repeated question never hits a database.

**3. Right database for each job**
```
"Find similar text"       → Qdrant (vector similarity)
"Find connected concepts" → Neo4j  (graph traversal)
"Answer this again"       → Cache  (exact match)
```

**4. Fail fast at startup, graceful fallback at runtime**
Missing indexes, broken connections — detected at server startup. Individual chunk extraction failures — logged and skipped, ingestion continues.

---

## Concepts

### KAG vs GraphRAG

| | GraphRAG (Microsoft) | KAG (Ant Group) | This Project |
|---|---|---|---|
| **Graph use** | Summarization tool | Reasoning engine | Both |
| **Best for** | "What are themes?" | "Why does X imply Y?" | All query types |
| **Communities** | Leiden algorithm | Not used | Union-Find |
| **Multi-hop** | Not core | Core feature | Implemented |

### CAG vs RAG

| | Traditional RAG | CAG |
|---|---|---|
| **Every query** | Hits vector DB + LLM | Hits cache first |
| **Repeated questions** | ~2000ms every time | ~0.001ms after first |
| **Preloaded doc** | Searches chunks | Reads full doc from RAM |
| **LLM calls** | Always 1+ | 0 on exact hit |
