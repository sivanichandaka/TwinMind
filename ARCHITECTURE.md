# TwinMind: Second Brain Architecture Documentation

## 1. Executive Summary

**Second Brain** is a **privacy-first, AI-powered personal assistant** that ingests personal knowledge (documents, audio, images) and provides intelligent conversational retrieval. It runs **locally** with optional cloud LLM support, using a containerized architecture that combines:

- **Hybrid search** (semantic vectors + BM25 keywords)
- **Multimodal ingestion** (PDF, audio, images, text)
- **Temporal query support** ("what from last week?")
- **Flexible LLM backends** (Ollama local or Gemini cloud)

---

## 2. System Architecture

### 2.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│   │  Chat   │     │ Upload  │     │Settings │                  │
│   │   Tab   │     │ (Cortex)│     │ (Model) │                  │
│   └────┬────┘     └────┬────┘     └────┬────┘                  │
└────────┼───────────────┼───────────────┼───────────────────────┘
         │               │               │
         ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8000)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ /chat    │  │ /upload  │  │/documents│  │ /health  │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘        │
│       │             │             │                             │
│       ▼             ▼             ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    RAG Engine                               ││
│  │  • Temporal Query Parsing (dateparser)                      ││
│  │  • Hybrid Search (Vector + BM25 with RRF fusion)            ││
│  │  • Context Assembly & Deduplication                         ││
│  └─────────────────────────────────────────────────────────────┘│
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │             Multimodal Ingestion Pipeline                   ││
│  │  • PDF (pypdf)      • Audio (Whisper + ffmpeg)              ││
│  │  • Images (Tesseract OCR)  • Text/Markdown                  ││
│  └─────────────────────────────────────────────────────────────┘│
└────────┬──────────────────────────────────────────┬────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────┐              ┌─────────────────────────────┐
│   PostgreSQL    │              │       LLM Providers         │
│   + pgvector    │              │  ┌───────────────────────┐  │
│ ┌─────────────┐ │              │  │ Ollama (Local GPU)    │  │
│ │ documents   │ │              │  │ • llama3 model        │  │
│ │ chunks      │ │              │  │ • Auto-pull on start  │  │
│ │ embeddings  │ │              │  └───────────────────────┘  │
│ │ content_tsv │ │              │  ┌───────────────────────┐  │
│ └─────────────┘ │              │  │ Google Gemini (Cloud) │  │
│   :5432         │              │  │ • API key required    │  │
└─────────────────┘              │  └───────────────────────┘  │
                                 └─────────────────────────────┘
```

### 2.2 Component Breakdown

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | HTML/JS + sidebar UI | Chat interface, file upload, model selection |
| **Backend API** | FastAPI (Python 3.9) | REST endpoints, streaming responses, static files |
| **Vector DB** | PostgreSQL + pgvector | Document storage, 384-dim embeddings, HNSW index |
| **Full-Text Search** | PostgreSQL ts_vector | BM25 keyword search, GIN index |
| **Embedding Model** | all-MiniLM-L6-v2 | Local sentence embeddings (22M params) |
| **LLM (Local)** | Ollama + llama3 | Privacy-first inference (8B params) |
| **LLM (Cloud)** | Google Gemini | Optional cloud API for lower latency |
| **Audio** | OpenAI Whisper + ffmpeg | Speech-to-text transcription |
| **OCR** | Tesseract | Image text extraction |

---

## 3. Data Flow

### 3.1 Document Ingestion Pipeline

```
User uploads file
        │
        ▼
┌───────────────────┐
│ File Type Router  │
└───────┬───────────┘
        │
   ┌────┴────┬────────┬─────────┐
   ▼         ▼        ▼         ▼
┌─────┐  ┌─────┐  ┌──────┐  ┌──────┐
│ PDF │  │Audio│  │Image │  │ Text │
│pypdf│  │Whis.│  │Tess. │  │direct│
└──┬──┘  └──┬──┘  └──┬───┘  └──┬───┘
   └────────┴────────┴─────────┘
                │
                ▼
        ┌───────────────┐
        │ Text Chunking │ (500 chars, 50 overlap)
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ Generate      │ (sentence-transformers)
        │ Embeddings    │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ Store in DB   │ (documents + chunks tables)
        └───────────────┘
```

### 3.2 Query Processing Pipeline

```
User query: "What did I add last week about databases?"
                    │
                    ▼
            ┌───────────────┐
            │ Temporal Parse│ → Extracts "last week" → date filter
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ Generate Query│ → [0.02, -0.15, 0.08, ...]
            │ Embedding     │
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Vector Search │       │Keyword Search │
│ (cosine sim)  │       │ (ts_vector)   │
│ Top 5 results │       │ Top 5 results │
└───────┬───────┘       └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    ▼
            ┌───────────────┐
            │ RRF Fusion    │ → score = 1/(k + rank)
            │ Deduplicate   │
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ Build Prompt  │ → System prompt + context + query
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ LLM Generate  │ → Streaming response
            │ (Ollama/Gemini)
            └───────────────┘
```

---

## 4. Database Schema

```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    source TEXT,                    -- 'upload', 'web', etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table with hybrid search indexes
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT,
    chunk_index INTEGER,
    embedding vector(384),          -- HNSW indexed
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Indexes
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_content_tsv_idx ON chunks USING gin (content_tsv);
```

---

## 5. Hybrid Retrieval Strategy

### Why Hybrid?

| Search Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Vector** | Semantic understanding ("database issues" → "SQL problems") | Misses exact terms |
| **Keyword** | Exact matches ("error 0x8004") | No semantic understanding |
| **Hybrid** | Best of both worlds | Slightly more complex |

### Reciprocal Rank Fusion (RRF)

Combines ranked lists from both search methods:

```python
def rrf_score(rank, k=60):
    return 1 / (k + rank)

# Example: Document appears #2 in vector, #5 in keyword
# RRF score = 1/(60+2) + 1/(60+5) = 0.016 + 0.015 = 0.031
```

---

## 6. Docker Services

| Service | Image | Port | GPU | Auto-Actions |
|---------|-------|------|-----|--------------|
| `backend` | twinmind-backend | 8000 | No | - |
| `db` | pgvector/pgvector:pg16 | 5432 | No | - |
| `ollama` | ollama/ollama | 11434 | Yes | Auto-pulls llama3 |

### Volumes
- `postgres_data` - Persistent database
- `ollama_data` - Cached LLM models (~5GB)

---

## 7. Known Limitations & Future Improvements

### Current Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **LLM Hallucinations** | Fabricated answers | Add source citations, "I don't know" prompting |
| **Whisper on CPU** | Slow transcription | Enable GPU passthrough |
| **No authentication** | Single-user only | Add JWT auth for multi-user |
| **Fixed chunking** | Breaks mid-sentence | Implement semantic chunking |
| **No caching** | Repeated DB hits | Add Redis caching layer |

### Planned Improvements

1. **GraphRAG** - Entity extraction + knowledge graph
2. **Cross-encoder reranking** - Better relevance scoring
3. **Conversation memory** - Multi-turn context
4. **Document versioning** - Track changes over time
5. **Batch upload progress** - Real-time ingestion status

---

## 8. Scalability Path

### Horizontal Scaling
```
                    Load Balancer
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │Backend 1│    │Backend 2│    │Backend 3│
    └────┬────┘    └────┬────┘    └────┬────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    ┌──────────────────┐  ┌─────────────────┐
    │ PostgreSQL       │  │ Redis Cache     │
    │ (Read Replicas)  │  │                 │
    └──────────────────┘  └─────────────────┘
```

### Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Vector search | ~100ms | <50ms |
| LLM response (first token) | ~2s | <500ms |
| Concurrent users | ~10 | 1000+ |
| Documents indexed | ~100 | 1M+ |

---

## 9. Security Considerations

| Concern | Current State | Recommendation |
|---------|---------------|----------------|
| Authentication | None | Add JWT + OAuth2 |
| API keys | Browser localStorage | Server-side encrypted storage |
| Data at rest | Unencrypted | Enable PostgreSQL TDE |
| Network | Docker internal | Add TLS/SSL |
| PII handling | Not detected | Add PII detection + redaction |

---

## 10. File Structure

```
TwinMind/
├── backend/
│   ├── __init__.py
│   ├── main.py                       # FastAPI endpoints
│   ├── database.py                   # PostgreSQL connection
│   ├── ingestion.py                  # Multimodal document processing
│   ├── rag.py                        # Retrieval + generation logic
│   ├── requirements.txt
│   └── static/
│       └── index.html                # Frontend SPA
├── data/
│   ├── DESIGN.md                     # Detailed system design
│   └── initial_knowledge.txt
├── docker-compose.yml                # Service orchestration
├── Dockerfile                        # Backend container
├── ARCHITECTURE.md                   # This document
├── run_ingestion.py                  # Batch ingest local files
└── README.md                         # Quick start
```

---

## 11. Quick Start

```bash
# Start all services (first run pulls llama3 ~5 min)
docker-compose up -d

# Check Ollama model status
docker logs -f twinmind-ollama

# Access UI
open http://localhost:8000

# Upload a document via UI or API
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
```

---

*Version: 1.0 | Last Updated: December 2024*
