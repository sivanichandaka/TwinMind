"""
RAG Module - Retrieval-Augmented Generation Engine

This module implements the core RAG (Retrieval-Augmented Generation) logic:
- Temporal query parsing (understanding "last week", "yesterday", etc.)
- Hybrid retrieval (combining vector similarity + keyword search)
- Reciprocal Rank Fusion (RRF) for result merging
- LLM response generation (Ollama local or Gemini cloud)

The hybrid approach significantly improves retrieval accuracy over
pure vector search, especially for queries with specific terms or codes.

Environment Variables:
    OLLAMA_HOST: Hostname for Ollama server (default: "localhost")
"""

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

# =============================================================================
# Configuration
# =============================================================================

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ollama configuration - uses environment variable for Docker networking
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
MODEL_NAME = "llama3"  # The model to use in Ollama


# =============================================================================
# Temporal Query Parsing
# =============================================================================

def parse_temporal_intent(query: str) -> tuple:
    """
    Detect and extract temporal (time-based) intent from a query.
    
    This enables queries like:
    - "What did I add last week?"
    - "Show me yesterday's notes"
    - "Find documents from this month"
    
    Args:
        query: The user's natural language query
        
    Returns:
        tuple: (start_date, end_date) datetime objects, or (None, None)
               if no temporal intent is detected
               
    Examples:
        >>> parse_temporal_intent("what from last week?")
        (datetime(2024, 12, 8), datetime(2024, 12, 15))
        
        >>> parse_temporal_intent("explain machine learning")
        (None, None)
    """
    query_lower = query.lower()
    
    # List of keywords that indicate temporal intent
    temporal_keywords = [
        'yesterday', 'today', 'last week', 'this week', 'last month', 
        'this month', 'last year', 'recent', 'ago', 'since', 'before', 'after'
    ]
    
    # Check if any temporal keywords are present
    has_temporal = any(keyword in query_lower for keyword in temporal_keywords)
    if not has_temporal:
        return None, None
    
    # Parse specific temporal expressions
    try:
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
            # Fallback: use dateparser library for complex expressions
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


# =============================================================================
# Search Functions
# =============================================================================

async def search_similar_chunks(query: str, limit: int = 5, start_date=None, end_date=None) -> list:
    """
    Perform vector similarity search on document chunks.
    
    Uses cosine similarity to find chunks whose embeddings are
    closest to the query embedding. Optionally filters by date range.
    
    Args:
        query: The search query text
        limit: Maximum number of results to return (default: 5)
        start_date: Optional start of date range filter
        end_date: Optional end of date range filter
        
    Returns:
        list: List of matching chunks with content and distance scores
    """
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    conn = await get_db_connection()
    
    try:
        if start_date and end_date:
            # Query with temporal filter - join with documents table
            logging.info(f"Applying temporal filter: {start_date} to {end_date}")
            rows = await conn.fetch("""
                SELECT c.content, (c.embedding <=> $1) as distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.created_at BETWEEN $3 AND $4
                ORDER BY c.embedding <=> $1
                LIMIT $2
            """, str(query_embedding), limit, start_date, end_date)
        else:
            # Standard vector similarity search
            rows = await conn.fetch("""
                SELECT content, (embedding <=> $1) as distance
                FROM chunks
                ORDER BY embedding <=> $1
                LIMIT $2
            """, str(query_embedding), limit)
            
        return rows
        
    finally:
        await conn.close()


async def keyword_search(query: str, limit: int = 10) -> list:
    """
    Perform full-text keyword search using PostgreSQL.
    
    Uses PostgreSQL's built-in text search with ts_vector and ts_rank
    for BM25-style relevance scoring. This complements vector search
    by catching exact term matches.
    
    Args:
        query: The search query text
        limit: Maximum number of results to return (default: 10)
        
    Returns:
        list: List of matching chunks with content and rank scores
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


# =============================================================================
# Result Fusion
# =============================================================================

def reciprocal_rank_fusion(vector_results: list, keyword_results: list, k: int = 60) -> list:
    """
    Combine results from vector and keyword search using Reciprocal Rank Fusion.
    
    RRF is a simple but effective method for merging ranked lists.
    For each result, the score is: sum(1 / (k + rank)) across all lists.
    
    Why k=60? It's a smoothing constant that prevents high-ranked items
    from dominating too heavily. 60 is the standard value from the original paper.
    
    Args:
        vector_results: Results from vector similarity search
        keyword_results: Results from keyword search
        k: Smoothing constant (default: 60, from original RRF paper)
        
    Returns:
        list: Combined results sorted by RRF score (highest first)
        
    Example:
        If a chunk ranks #1 in vector search and #3 in keyword search:
        score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
    """
    scores = {}
    
    # Score vector search results
    for rank, row in enumerate(vector_results, start=1):
        content = row['content']
        scores[content] = scores.get(content, 0) + 1 / (k + rank)
    
    # Score keyword search results
    for rank, row in enumerate(keyword_results, start=1):
        content = row['content']
        scores[content] = scores.get(content, 0) + 1 / (k + rank)
    
    # Sort by combined score (highest first)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"content": content, "score": score} for content, score in ranked]


# =============================================================================
# Response Generation
# =============================================================================

async def generate_response(query: str, model: str = "ollama", api_key: Optional[str] = None):
    """
    Generate a response to the user's query using RAG.
    
    This is the main entry point for the chat endpoint. It:
    1. Parses temporal intent from the query
    2. Performs hybrid retrieval (vector + keyword search)
    3. Combines results using RRF
    4. Builds a context-aware prompt
    5. Streams the LLM response
    
    Args:
        query: The user's question
        model: Which LLM to use ("ollama" or "gemini")
        api_key: API key for Gemini (required if model="gemini")
        
    Yields:
        str: Chunks of the generated response (for streaming)
    """
    # Parse any temporal intent (e.g., "last week", "yesterday")
    start_date, end_date = parse_temporal_intent(query)
    if start_date and end_date:
        logging.info(f"Detected temporal query: {start_date} to {end_date}")
    
    # Hybrid Retrieval: run both vector and keyword search
    vector_results = await search_similar_chunks(query, limit=10, start_date=start_date, end_date=end_date)
    keyword_results = await keyword_search(query, limit=10)
    
    # Combine results using Reciprocal Rank Fusion
    hybrid_results = reciprocal_rank_fusion(vector_results, keyword_results)
    
    # Build context from top 5 results
    context_text = "\n\n".join([result['content'] for result in hybrid_results[:5]])
    
    # Construct the system prompt with retrieved context
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
    
    # Route to the appropriate LLM provider
    if model == "gemini":
        async for chunk in generate_gemini_response(system_prompt, query, api_key):
            yield chunk
    else:
        # Default to Ollama (local LLM)
        async for chunk in generate_ollama_response(system_prompt, query):
            yield chunk


async def generate_gemini_response(system_prompt: str, query: str, api_key: Optional[str]):
    """
    Generate a response using Google Gemini API.
    
    Gemini offers lower latency than local LLMs but requires an API key
    and sends data to Google's servers.
    
    Args:
        system_prompt: The system prompt with context
        query: The user's question
        api_key: Google Gemini API key (from AI Studio)
        
    Yields:
        str: Chunks of the generated response
    """
    try:
        import google.generativeai as genai
        
        if not api_key:
            yield "Error: Gemini API key not provided"
            return
        
        # Configure the Gemini client
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Build the full prompt
        full_prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
        response = model.generate_content(full_prompt, stream=True)
        
        # Stream the response chunks
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        import traceback
        logging.error(f"Gemini error: {traceback.format_exc()}")
        yield f"Error: {str(e)}"


async def generate_ollama_response(system_prompt: str, query: str):
    """
    Generate a response using Ollama (local LLM).
    
    Ollama runs the LLM locally, ensuring privacy and offline capability.
    Requires a GPU for reasonable performance with larger models.
    
    Args:
        system_prompt: The system prompt with context
        query: The user's question
        
    Yields:
        str: Chunks of the generated response (streamed)
    """
    logging.info(f"Sending request to Ollama ({OLLAMA_API_URL})...")
    
    try:
        async with httpx.AsyncClient() as client:
            # Stream the response from Ollama
            async with client.stream("POST", OLLAMA_API_URL, json={
                "model": MODEL_NAME,
                "prompt": f"{system_prompt}\n\nUser: {query}\nAssistant:",
                "stream": True
            }, timeout=300.0) as response:
                
                logging.info(f"Ollama connected. Status Code: {response.status_code}")
                
                # Parse JSONL response and yield text chunks
                async for line in response.aiter_lines():
                    if line:
                        try:
                            json_obj = json.loads(line)
                            text_chunk = json_obj.get("response", "")
                            if text_chunk:
                                yield text_chunk
                        except json.JSONDecodeError:
                            pass  # Skip malformed lines
                            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Error connecting to Ollama: {error_details}")
        yield f"Error: {str(e)}"
