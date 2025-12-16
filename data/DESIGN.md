# Second Brain - System Design Document

## Overview

Second Brain is a **privacy-first, AI-powered knowledge management system** that allows you to ingest, search, and query your personal documents using natural language. It combines local LLM inference with hybrid retrieval (semantic + keyword search) for accurate, context-aware responses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│   │  Chat   │     │ Cortex  │     │ Settings│                  │
│   │   Tab   │     │  Upload │     │  Model  │                  │
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
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RAG Engine                           │   │
│  │  • Hybrid Search (Vector + BM25)                        │   │
│  │  • Temporal Query Parsing                               │   │
│  │  • Context Assembly                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │             Ingestion Pipeline                          │   │
│  │  • PDF (pypdf)                                          │   │
│  │  • Audio (Whisper)                                      │   │
│  │  • Images (Tesseract OCR)                               │   │
│  │  • Text/Markdown                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬──────────────────────────────────┬────────────────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────────────┐
│   PostgreSQL    │              │     LLM Provider        │
│   + pgvector    │              │  ┌─────────────────┐    │
│ ┌─────────────┐ │              │  │ Ollama (Local)  │    │
│ │ documents   │ │              │  │ llama3 model    │    │
│ │ chunks      │ │              │  └─────────────────┘    │
│ │ embeddings  │ │              │  ┌─────────────────┐    │
│ └─────────────┘ │              │  │ Google Gemini   │    │
│   :5432         │              │  │ (Cloud API)     │    │
└─────────────────┘              │  └─────────────────┘    │
                                 │       :11434            │
                                 └─────────────────────────┘
```

---

## Components

### 1. Frontend (index.html)
- **Single-page application** using vanilla JS + Tailwind CSS
- **Three tabs**: Chat, Cortex (Upload), Settings
- **Features**:
  - Real-time streaming responses
  - Drag-and-drop file upload
  - Model selection (Ollama/Gemini)
  - Markdown rendering with syntax highlighting

### 2. Backend (FastAPI)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Query the knowledge base with natural language |
| `/upload` | POST | Upload and ingest documents |
| `/documents` | GET | List all ingested documents |
| `/health` | GET | Health check endpoint |

### 3. Database (PostgreSQL + pgvector)
**Tables**:
```sql
documents (id, title, content, source, created_at)
chunks (id, document_id, content, chunk_index, embedding, content_tsv)
```

**Indexes**:
- `HNSW` index on `embedding` for fast vector similarity search
- `GIN` index on `content_tsv` for full-text keyword search

### 4. LLM Providers
| Provider | Type | Use Case |
|----------|------|----------|
| **Ollama** | Local | Privacy-first, offline, GPU-accelerated |
| **Gemini** | Cloud | Lower latency, no cold start, requires API key |

---

## Data Flow

### Document Ingestion
```
1. User uploads file → /upload endpoint
2. File type detection (PDF, audio, image, text)
3. Content extraction:
   - PDF → pypdf
   - Audio → Whisper transcription
   - Image → Tesseract OCR
   - Text → Direct read
4. Text chunking (500 chars, 50 overlap)
5. Embedding generation (sentence-transformers all-MiniLM-L6-v2)
6. Store in PostgreSQL (document + chunks + embeddings)
```

### Query Processing
```
1. User sends question → /chat endpoint
2. Temporal parsing ("last week", "yesterday", etc.)
3. Hybrid search:
   a. Vector similarity (cosine distance, top 5)
   b. Keyword search (BM25/ts_vector, top 5)
   c. Reciprocal Rank Fusion to combine results
4. Context assembly (deduplicated, sorted by relevance)
5. LLM prompt construction with context
6. Streaming response back to user
```

---

## Hybrid Retrieval Strategy

### Why Hybrid?
- **Vector search**: Understands semantics ("database issues" → "SQL performance problems")
- **Keyword search**: Exact matches ("error 0x8004" → finds that specific code)
- **Combined**: Best of both worlds

### Reciprocal Rank Fusion (RRF)
```python
score = 1 / (k + rank)  # k=60 typically
```
Merges ranked lists from both search methods, prioritizing items that appear high in both.

---

## Temporal Query Support

Parses natural language time expressions:
- "last week" → filters to past 7 days
- "yesterday" → filters to past 24 hours  
- "this month" → filters to current month
- "in January" → filters to January dates

Uses `dateparser` library with relative date support.

---

## Multimodal Ingestion

| Format | Library | Output |
|--------|---------|--------|
| PDF | `pypdf` | Extracted text |
| Audio (.mp3, .m4a, .wav) | `openai-whisper` | Transcription |
| Images (.png, .jpg) | `pytesseract` | OCR text |
| Text (.txt, .md) | Built-in | Raw content |
| Web URLs | `beautifulsoup4` | Scraped content |

---

## Docker Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `backend` | Custom (Python 3.9) | 8000 | FastAPI application |
| `db` | pgvector/pgvector:pg16 | 5432 | Vector database |
| `ollama` | ollama/ollama | 11434 | Local LLM inference |

### Volumes
- `postgres_data`: Persistent database storage
- `ollama_data`: Cached LLM models

### Startup
Ollama automatically pulls `llama3` model on first start via entrypoint script.

---

## Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | db | Database hostname |
| `POSTGRES_USER` | user | Database username |
| `POSTGRES_PASSWORD` | password | Database password |
| `POSTGRES_DB` | secondbrain | Database name |
| `OLLAMA_HOST` | ollama | Ollama API hostname |

### Model Configuration
Users can switch between Ollama and Gemini via the Settings tab. Gemini requires an API key from Google AI Studio.

---

## File Structure

```
TwinMind/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints
│   ├── database.py      # PostgreSQL connection
│   ├── ingestion.py     # Document processing
│   ├── rag.py           # Retrieval & generation
│   ├── requirements.txt # Python dependencies
│   └── static/
│       └── index.html   # Frontend SPA
├── data/
│   └── DESIGN.md        # This file
├── docker-compose.yml   # Service orchestration
├── Dockerfile           # Backend container
└── README.md            # Quick start guide
```

---

## Security Considerations

1. **Local-first**: Default mode uses Ollama (no data leaves your machine)
2. **No authentication**: Designed for single-user local deployment
3. **API key storage**: Gemini keys stored in browser localStorage (user responsibility)
4. **Network isolation**: Docker internal network for service communication

---

## Performance

| Operation | Expected Latency |
|-----------|------------------|
| Document ingestion | 2-10s depending on size |
| Vector search | <100ms |
| Keyword search | <50ms |
| LLM response (Ollama) | 5-30s (first token ~2s) |
| LLM response (Gemini) | 1-10s |

### Optimization Notes
- HNSW index provides O(log N) vector search
- Whisper runs on CPU in container (GPU requires additional setup)
- First Ollama query has cold start (~30s to load model)

---

## Future Enhancements

- [ ] GraphRAG with entity extraction
- [ ] Cross-encoder reranking
- [ ] User authentication
- [ ] Document deletion endpoint
- [ ] Batch upload progress
- [ ] GPU passthrough for Whisper

---

## Quick Start

```bash
# Start all services
docker-compose up -d

# Wait for Ollama to pull model (~5 min first time)
docker logs -f twinmind-ollama

# Access UI
open http://localhost:8000
```

---

*Document Version: 1.0*  
*Last Updated: December 2024*
