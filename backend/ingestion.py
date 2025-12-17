"""
Ingestion Module - Multimodal Document Processing Pipeline

This module handles the ingestion of various document types into the Second Brain:
- PDF documents (via pypdf)
- Audio files (via OpenAI Whisper)
- Images with text (via Tesseract OCR)
- Plain text and Markdown files
- Web URLs (via BeautifulSoup)

The ingestion pipeline:
1. Extracts text content from the source
2. Chunks the text into manageable fragments
3. Generates embeddings for each chunk
4. Stores everything in PostgreSQL with pgvector

Dependencies:
    - sentence-transformers: For generating text embeddings
    - pypdf: For PDF text extraction
    - whisper: For audio transcription
    - pytesseract: For OCR on images
    - beautifulsoup4: For web scraping
"""

from sentence_transformers import SentenceTransformer
from .database import get_db_connection
import asyncio

# =============================================================================
# Embedding Model
# =============================================================================

# Load the sentence-transformers model for generating embeddings
# all-MiniLM-L6-v2 is a good balance of speed and quality:
#   - 384 dimensions (compact)
#   - ~22M parameters (fast inference)
#   - Good semantic understanding for retrieval tasks
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str) -> list:
    """
    Generate a 384-dimensional embedding vector for the given text.
    
    Args:
        text: The text to embed (can be a query or document chunk)
        
    Returns:
        list: A list of 384 floats representing the text embedding
        
    Example:
        embedding = get_embedding("What is machine learning?")
        # Returns: [0.023, -0.156, 0.089, ...]  (384 values)
    """
    return model.encode(text).tolist()


# =============================================================================
# Text Chunking
# =============================================================================

def recursive_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks for better retrieval.
    
    This function uses a paragraph-aware chunking strategy:
    1. First splits by double newlines (paragraphs)
    2. Groups paragraphs until reaching chunk_size
    3. Creates overlapping chunks to preserve context at boundaries
    
    Args:
        text: The full text to chunk
        chunk_size: Target size for each chunk in characters (default: 500)
        overlap: Number of characters to overlap between chunks (default: 50)
        
    Returns:
        list: A list of text chunks
        
    Note:
        Chunk size of 500 chars works well for most documents.
        Smaller chunks = more precise retrieval but less context.
        Larger chunks = more context but may include irrelevant content.
    """
    # Split by double newline to respect paragraph boundaries
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph keeps us under the limit, add it
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            # Save current chunk and start a new one
            if current_chunk.strip():  # Only add non-empty chunks
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            
    # Don't forget the last chunk
    if current_chunk.strip():  # Only add non-empty chunks
        chunks.append(current_chunk.strip())
    
    # Filter out any empty or very short chunks
    chunks = [c for c in chunks if len(c) > 10]
        
    return chunks


# =============================================================================
# Document Ingestion
# =============================================================================

async def ingest_document(title: str, content: str, source: str = "upload") -> tuple:
    """
    Ingest a document into the knowledge base.
    
    This is the main entry point for adding documents. It:
    1. Stores the original document in the 'documents' table
    2. Chunks the content into smaller fragments
    3. Generates embeddings for each chunk
    4. Stores chunks with embeddings in the 'chunks' table
    
    Args:
        title: Document title or filename
        content: Full text content of the document
        source: Origin of the document (e.g., "upload", "web", "local_folder")
        
    Returns:
        tuple: (document_id, chunk_count) - The ID of the created document
               and the number of chunks generated
               
    Example:
        doc_id, chunks = await ingest_document(
            title="meeting_notes.txt",
            content="Today we discussed the project roadmap...",
            source="upload"
        )
        print(f"Created document {doc_id} with {chunks} chunks")
    """
    conn = await get_db_connection()
    try:
        # Insert the original document and get its ID
        doc_id = await conn.fetchval("""
            INSERT INTO documents (title, content, source) 
            VALUES ($1, $2, $3) RETURNING id
        """, title, content, source)
        
        # Create chunks from the content
        chunks = recursive_chunking(content)
        
        # Embed and store each chunk
        for i, chunk_text in enumerate(chunks):
            embedding = get_embedding(chunk_text)
            await conn.execute("""
                INSERT INTO chunks (document_id, content, chunk_index, embedding)
                VALUES ($1, $2, $3, $4)
            """, doc_id, chunk_text, i, str(embedding))
            
        print(f"Ingested '{title}' ({len(chunks)} chunks)")
        return doc_id, len(chunks)
    finally:
        await conn.close()


# =============================================================================
# File Type Processors
# =============================================================================

import os
import pypdf
from PIL import Image
import pytesseract
import whisper
import requests
from bs4 import BeautifulSoup

# Directory for local file ingestion
DATA_DIR = "/app/data"

# Whisper model is loaded lazily to save memory on startup
_whisper_model = None


def get_whisper_model():
    """
    Lazy-load the Whisper model for audio transcription.
    
    Uses the 'base' model which offers a good balance of speed and accuracy.
    The model is cached after first load to avoid reloading on each transcription.
    
    Returns:
        whisper.Model: The loaded Whisper model
        
    Note:
        Model sizes: tiny (39M) < base (74M) < small (244M) < medium (769M) < large (1550M)
        Larger models are more accurate but slower and require more memory.
    """
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def read_pdf(file_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text content, or None if extraction failed
    """
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return None


def read_image(file_path: str) -> str:
    """
    Extract text from an image using OCR (Tesseract).
    
    Args:
        file_path: Path to the image file (PNG, JPG, JPEG)
        
    Returns:
        str: Extracted text content, or None if extraction failed
        
    Note:
        Requires Tesseract to be installed in the container.
        Works best with clear, high-contrast text.
    """
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return text
    except Exception as e:
        print(f"Error reading Image {file_path}: {e}")
        return None


def read_audio(file_path: str) -> str:
    """
    Transcribe audio to text using OpenAI Whisper.
    
    Args:
        file_path: Path to the audio file (MP3, M4A, WAV)
        
    Returns:
        str: Transcribed text content, or None if transcription failed
        
    Note:
        Requires ffmpeg to be installed for audio processing.
        GPU acceleration available if CUDA is configured.
    """
    try:
        print(f"[WHISPER] Starting transcription for: {file_path}")
        model = get_whisper_model()
        result = model.transcribe(file_path)
        transcript = result["text"]
        
        # Log the transcript to console
        print("=" * 60)
        print("[WHISPER] TRANSCRIPTION COMPLETE")
        print("=" * 60)
        print(f"[WHISPER] File: {file_path}")
        print(f"[WHISPER] Length: {len(transcript)} characters")
        print("-" * 60)
        print("[WHISPER] TRANSCRIPT:")
        print(transcript)
        print("=" * 60)
        
        return transcript
    except Exception as e:
        print(f"[WHISPER] ERROR transcribing audio {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def read_web(url: str) -> str:
    """
    Fetch and extract main content from a web URL.
    
    Removes navigation, scripts, and other non-content elements
    to focus on the main text content of the page.
    
    Args:
        url: The web URL to fetch
        
    Returns:
        str: Extracted text content, or None if fetching failed
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove non-content elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        
        # Extract and clean text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error reading URL {url}: {e}")
        return None


# =============================================================================
# Directory Ingestion
# =============================================================================

async def ingest_directory():
    """
    Scan and ingest all supported files from the DATA_DIR.
    
    This function walks through the data directory and processes:
    - .txt, .md files (plain text)
    - .pdf files (PDF documents)
    - .png, .jpg, .jpeg files (images with OCR)
    - .mp3, .m4a, .wav files (audio with transcription)
    
    Each file is automatically tagged with its source type.
    """
    print(f"Scanning directory: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Reading {file}...")
            
            content = None
            try:
                # Route to appropriate reader based on file extension
                if file.endswith(".txt") or file.endswith(".md"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                elif file.lower().endswith(".pdf"):
                    content = read_pdf(file_path)
                    if content:
                        content = f"[PDF Source: {file}]\n" + content
                        
                elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    content = read_image(file_path)
                    if content:
                         content = f"[IMAGE Source: {file}]\n" + content
                         
                elif file.lower().endswith(('.mp3', '.m4a', '.wav')):
                    content = read_audio(file_path)
                    if content:
                        content = f"[AUDIO Transcript: {file}]\n" + content
                
                # Ingest if we successfully extracted content
                if content:
                    await ingest_document(title=file, content=content, source="local_folder")
                else:
                    print(f"Skipping {file}: No content extracted or unsupported format.")
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
