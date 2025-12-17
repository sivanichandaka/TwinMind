"""
Main Module - FastAPI Application Entry Point

This module defines the FastAPI application and all HTTP endpoints:
- POST /chat: Query the knowledge base with natural language
- POST /upload: Upload and ingest documents (PDF, audio, images, text)
- GET /documents: List all ingested documents
- GET /health: Health check endpoint

The application serves a static frontend and provides a REST API
for the Second Brain knowledge management system.

Startup:
    The application initializes the database schema on startup
    using the lifespan context manager.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from .database import init_db
from .ingestion import ingest_document
from .rag import generate_response
import json


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.
    
    On startup:
        - Initializes the database schema (creates tables, indexes)
        
    On shutdown:
        - Currently no cleanup needed
    """
    # Startup: Initialize the database
    await init_db()
    yield
    # Shutdown: Add any cleanup here if needed


# =============================================================================
# FastAPI Application
# =============================================================================

from fastapi.staticfiles import StaticFiles

# Create the FastAPI application with lifespan handler
app = FastAPI(
    title="Second Brain API",
    description="A privacy-first, AI-powered knowledge management system",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """
    Request model for the /chat endpoint.
    
    Attributes:
        message: The user's question or message
        model: Which LLM to use - "ollama" (local) or "gemini" (cloud)
        api_key: API key for Gemini (required if model="gemini")
    """
    message: str
    model: str = "ollama"  # "ollama" or "gemini"
    api_key: Optional[str] = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Query the knowledge base using natural language.
    
    This endpoint:
    1. Parses the user's question
    2. Retrieves relevant documents using hybrid search
    3. Generates a response using the selected LLM
    4. Streams the response back to the client
    
    Args:
        request: ChatRequest with message, model choice, and optional API key
        
    Returns:
        StreamingResponse: Real-time streamed LLM response
        
    Example:
        POST /chat
        {
            "message": "What are the main features?",
            "model": "ollama"
        }
    """
    return StreamingResponse(
        generate_response(request.message, model=request.model, api_key=request.api_key), 
        media_type="text/plain"
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the knowledge base.
    
    Supported file types:
    - Text: .txt, .md
    - PDF: .pdf
    - Images: .png, .jpg, .jpeg (OCR extraction)
    - Audio: .mp3, .m4a, .wav (Whisper transcription)
    
    Args:
        file: The uploaded file
        
    Returns:
        dict: Status, document ID, and number of chunks created
        
    Raises:
        HTTPException 400: If file type is not supported
        HTTPException 500: If content extraction fails
        
    Example:
        POST /upload
        Content-Type: multipart/form-data
        file: <binary data>
    """
    import tempfile
    import os
    from .ingestion import read_pdf, read_image, read_audio
    
    # Save uploaded file to a temporary location for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name
    
    try:
        # Determine file type and extract content accordingly
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith(('.txt', '.md')):
            # Plain text files - decode directly
            content = content_bytes.decode('utf-8')
            
        elif filename_lower.endswith('.pdf'):
            # PDF files - extract text using pypdf
            content = read_pdf(tmp_path)
            
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
            # Image files - extract text using OCR
            content = read_image(tmp_path)
            
        elif filename_lower.endswith(('.mp3', '.m4a', '.wav')):
            # Audio files - transcribe using Whisper
            content = read_audio(tmp_path)
            if content:
                # Prefix with audio source tag for identification
                content = f"[AUDIO TRANSCRIPT: {file.filename}]\n\n{content}"
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Verify we got content
        if not content:
            raise HTTPException(status_code=500, detail="Failed to extract content")
        
        # Log what we're ingesting
        print(f"[UPLOAD] Ingesting '{file.filename}' with {len(content)} characters")
        
        # Ingest the document into the knowledge base
        doc_id, chunks_count = await ingest_document(file.filename, content, source="upload")
        
        print(f"[UPLOAD] Successfully ingested doc_id={doc_id} with {chunks_count} chunks")
        
        return {
            "status": "success", 
            "document_id": doc_id, 
            "chunks_processed": chunks_count
        }
        
    finally:
        # Always clean up the temporary file
        os.unlink(tmp_path)


@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Simple status indicator
        
    Example:
        GET /health
        Response: {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/documents")
async def list_documents():
    """
    List all documents that have been ingested into the knowledge base.
    
    Returns document metadata including:
    - ID and title
    - Source (how it was ingested)
    - Creation timestamp
    - Number of chunks
    
    Returns:
        list: Array of document metadata objects
        
    Example:
        GET /documents
        Response: [
            {
                "id": 1,
                "title": "meeting_notes.pdf",
                "source": "upload",
                "created_at": "2024-12-15T10:30:00Z",
                "chunks": 5
            }
        ]
    """
    from .database import get_db_connection
    conn = await get_db_connection()
    
    try:
        # Query documents with chunk count
        rows = await conn.fetch("""
            SELECT d.id, d.title, d.source, d.created_at, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """)
        
        # Format response
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "chunks": row["chunk_count"]
            }
            for row in rows
        ]
        
    finally:
        await conn.close()


# =============================================================================
# Static Files
# =============================================================================

# Mount the static files directory at root
# This serves the frontend (index.html) and must be last to avoid catching API routes
app.mount("/", StaticFiles(directory="backend/static", html=True), name="static")
