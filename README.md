# Second Brain PoC Walkthrough

## Overview
We built a Local-First "Second Brain" Proof of Concept using:
- **FastAPI**: Backend API and Static File Server.
- **PostgreSQL + pgvector**: Vector database (via Docker).
- **SentenceTransformers**: Local text embeddings (`all-MiniLM-L6-v2`).
- **Ollama**: Local LLM for generation.
- **Vanilla JS/HTML**: Clean, dark-mode frontend.

## How to Run automatically
1. Ensure Docker is running.
2. Run `docker-compose up -d --build`.
3. Open `http://localhost:8000/` in your browser.

*Note: The entire stack (Backend, DB, Ollama) runs in Docker.*
*To reset data: `docker exec twinmind-backend python run_ingestion.py`*

## Features
- **Chat**: Streamed responses from your local LLM.
- **RAG**: Retrives relevant chunks from the "Second Brain Architecture" document.
- **Local-First**: No data leaves your machine (except purely local Docker/Ollama calls).
