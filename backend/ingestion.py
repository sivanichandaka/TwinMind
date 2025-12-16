from sentence_transformers import SentenceTransformer
from .database import get_db_connection
import asyncio

# Load local embedding model
# all-MiniLM-L6-v2 is fast and small, outputting 384-dim vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    return model.encode(text).tolist()

def recursive_chunking(text: str, chunk_size: int = 500, overlap: int = 50):
    # Simple semantic-ish chunking (splitting by paragraphs/sentences roughly)
    # For PoC we keep it simple: split by double newline, then re-group
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

async def ingest_document(title: str, content: str, source: str = "upload"):
    conn = await get_db_connection()
    try:
        # Insert Document
        doc_id = await conn.fetchval("""
            INSERT INTO documents (title, content, source) 
            VALUES ($1, $2, $3) RETURNING id
        """, title, content, source)
        
        # Create Chunks
        chunks = recursive_chunking(content)
        
        # Embed and Insert Chunks
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
        
import os
import pypdf
from PIL import Image
import pytesseract
import whisper
import requests
from bs4 import BeautifulSoup

DATA_DIR = "/app/data"

# Load Whisper model once (lazy load)
_whisper_model = None
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def read_pdf(file_path):
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return None

def read_image(file_path):
    try:
        # Tesseract must be installed in container
        text = pytesseract.image_to_string(Image.open(file_path))
        return text
    except Exception as e:
        print(f"Error reading Image {file_path}: {e}")
        return None

def read_audio(file_path):
    """Transcribe audio using Whisper (local GPU-accelerated if available)"""
    try:
        model = get_whisper_model()
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio {file_path}: {e}")
        return None

def read_web(url):
    """Fetch and extract main content from a URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error reading URL {url}: {e}")
        return None

async def ingest_directory():
    """Scans DATA_DIR and ingests all supported files."""
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
                elif file.lower().endsWith(('.mp3', '.m4a', '.wav')):
                    content = read_audio(file_path)
                    if content:
                        content = f"[AUDIO Transcript: {file}]\n" + content
                
                if content:
                    await ingest_document(title=file, content=content, source="local_folder")
                else:
                    print(f"Skipping {file}: No content extracted or unsupported format.")
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
