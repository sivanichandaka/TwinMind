from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from .database import init_db
from .ingestion import ingest_document
from .rag import generate_response
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Init DB
    await init_db()
    yield
    # Shutdown

from fastapi.staticfiles import StaticFiles

app = FastAPI(lifespan=lifespan)



class ChatRequest(BaseModel):
    message: str
    model: str = "ollama"  # "ollama" or "gemini"
    api_key: Optional[str] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_response(request.message, model=request.model, api_key=request.api_key), 
        media_type="text/plain"
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    import tempfile
    import os
    from .ingestion import read_pdf, read_image, read_audio
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name
    
    try:
        # Determine file type and extract content
        filename_lower = file.filename.lower()
        if filename_lower.endswith(('.txt', '.md')):
            content = content_bytes.decode('utf-8')
        elif filename_lower.endswith('.pdf'):
            content = read_pdf(tmp_path)
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
            content = read_image(tmp_path)
        elif filename_lower.endswith(('.mp3', '.m4a', '.wav')):
            content = read_audio(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        if not content:
            raise HTTPException(status_code=500, detail="Failed to extract content")
        
        # Ingest the document
        doc_id, chunks_count = await ingest_document(file.filename, content, source="upload")
        
        return {"status": "success", "document_id": doc_id, "chunks_processed": chunks_count}
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/documents")
async def list_documents():
    from .database import get_db_connection
    conn = await get_db_connection()
    try:
        rows = await conn.fetch("""
            SELECT d.id, d.title, d.source, d.created_at, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """)
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

# Mount static files at the root (catch-all), so it must be last
app.mount("/", StaticFiles(directory="backend/static", html=True), name="static")
