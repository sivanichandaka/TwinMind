import httpx
import logging
import json
from .database import get_db_connection
from .ingestion import get_embedding

import os
import dateparser
from datetime import datetime, timedelta
import re
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
MODEL_NAME = "llama3" # Or 'mistral', ensure user has it pulled

def parse_temporal_intent(query: str):
    """
    Detect temporal keywords and extract date range.
    Returns (start_date, end_date) tuple or (None, None) if no time intent.
    """
    query_lower = query.lower()
    
    # Common temporal patterns
    temporal_keywords = [
        'yesterday', 'today', 'last week', 'this week', 'last month', 
        'this month', 'last year', 'recent', 'ago', 'since', 'before', 'after'
    ]
    
    has_temporal = any(keyword in query_lower for keyword in temporal_keywords)
    if not has_temporal:
        return None, None
    
    # Try to extract dates using dateparser
    # Look for phrases like "last week", "3 days ago", etc.
    try:
        # Parse relative dates
        if 'last week' in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
        elif 'yesterday' in query_lower:
            start_date = datetime.now() - timedelta(days=1)
            end_date = start_date + timedelta(days=1)
        elif 'last month' in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        elif 'this month' in query_lower:
            now = datetime.now()
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()
        else:
            # Fallback: try dateparser on the entire query
            parsed = dateparser.parse(query_lower, settings={'PREFER_DATES_FROM': 'past'})
            if parsed:
                start_date = parsed - timedelta(days=7)  # Default 7-day window
                end_date = datetime.now()
            else:
                return None, None
                
        return start_date, end_date
    except Exception as e:
        logging.warning(f"Failed to parse temporal intent: {e}")
        return None, None

async def search_similar_chunks(query: str, limit: int = 5, start_date=None, end_date=None):
    query_embedding = get_embedding(query)
    conn = await get_db_connection()
    try:
        # Build query with optional temporal filter
        if start_date and end_date:
            logging.info(f"Applying temporal filter: {start_date} to {end_date}")
            # Join with documents table to access created_at
            rows = await conn.fetch("""
                SELECT c.content, (c.embedding <=> $1) as distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.created_at BETWEEN $3 AND $4
                ORDER BY c.embedding <=> $1
                LIMIT $2
            """, str(query_embedding), limit, start_date, end_date)
        else:
            # Original query without filter
            rows = await conn.fetch("""
                SELECT content, (embedding <=> $1) as distance
                FROM chunks
                ORDER BY embedding <=> $1
                LIMIT $2
            """, str(query_embedding), limit)
        return rows
    finally:
        await conn.close()

async def keyword_search(query: str, limit: int = 10):
    """
    Perform full-text keyword search using PostgreSQL ts_vector.
    Returns chunks ranked by BM25-style relevance.
    """
    conn = await get_db_connection()
    try:
        rows = await conn.fetch("""
            SELECT content, ts_rank(content_tsv, query) as rank
            FROM chunks, plainto_tsquery('english', $1) query
            WHERE content_tsv @@ query
            ORDER BY rank DESC
            LIMIT $2
        """, query, limit)
        return rows
    finally:
        await conn.close()

def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
    """
    Combine results from vector and keyword search using RRF.
    k is a constant (typically 60) that smooths the ranking.
    """
    scores = {}
    
    # Score vector results
    for rank, row in enumerate(vector_results, start=1):
        content = row['content']
        scores[content] = scores.get(content, 0) + 1 / (k + rank)
    
    # Score keyword results
    for rank, row in enumerate(keyword_results, start=1):
        content = row['content']
        scores[content] = scores.get(content, 0) + 1 / (k + rank)
    
    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"content": content, "score": score} for content, score in ranked]

async def generate_response(query: str, model: str = "ollama", api_key: Optional[str] = None):
    # Parse temporal intent
    start_date, end_date = parse_temporal_intent(query)
    if start_date and end_date:
        logging.info(f"Detected temporal query: {start_date} to {end_date}")
    
    # Hybrid Retrieval: Vector + Keyword search
    vector_results = await search_similar_chunks(query, limit=10, start_date=start_date, end_date=end_date)
    keyword_results = await keyword_search(query, limit=10)
    
    # Combine results using RRF
    hybrid_results = reciprocal_rank_fusion(vector_results, keyword_results)
    
    # Take top 5 for context
    context_text = "\n\n".join([result['content'] for result in hybrid_results[:5]])
    
    system_prompt = f"""You are a helpful Second Brain assistant. 
    Use the following context to answer the user's question.
    
    Context:
    {context_text}
    
    Instructions:
    - If the user greets you (e.g., "Hi", "Hello"), respond politely and introduce yourself as their Second Brain.
    - If the answer is found in the context, use it.
    - If the answer is NOT in the context but is a general question, you may answer generally but mention you are using general knowledge.
    - Be concise.
    """
    
    logging.info(f"Using model: {model}")
    
    # Route to appropriate model
    if model == "gemini":
        async for chunk in generate_gemini_response(system_prompt, query, api_key):
            yield chunk
    else:  # Default to Ollama
        async for chunk in generate_ollama_response(system_prompt, query):
            yield chunk

async def generate_gemini_response(system_prompt: str, query: str, api_key: Optional[str]):
    """Generate response using Google Gemini API"""
    try:
        import google.generativeai as genai
        
        if not api_key:
            yield "Error: Gemini API key not provided"
            return
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        full_prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
        response = model.generate_content(full_prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        import traceback
        logging.error(f"Gemini error: {traceback.format_exc()}")
        yield f"Error: {str(e)}"

async def generate_ollama_response(system_prompt: str, query: str):
    """Generate response using Ollama (local LLM)"""
    logging.info(f"Sending request to Ollama ({OLLAMA_API_URL})...")
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", OLLAMA_API_URL, json={
                "model": MODEL_NAME,
                "prompt": f"{system_prompt}\n\nUser: {query}\nAssistant:",
                "stream": True
            }, timeout=300.0) as response:
                logging.info(f"Ollama connected. Status Code: {response.status_code}")
                async for line in response.aiter_lines():
                    if line:
                        try:
                            json_obj = json.loads(line)
                            text_chunk = json_obj.get("response", "")
                            if text_chunk:
                                yield text_chunk
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Error connecting to Ollama: {error_details}")
        yield f"Error: {str(e)}"

