"""
Database Module - PostgreSQL + pgvector Connection and Schema

This module handles all database operations for the Second Brain application:
- Connection management to PostgreSQL
- Schema initialization with pgvector extension
- Table creation for documents and chunks
- Index setup for vector and full-text search

Environment Variables:
    POSTGRES_USER: Database username (default: "user")
    POSTGRES_PASSWORD: Database password (default: "password")
    POSTGRES_DB: Database name (default: "secondbrain")
    POSTGRES_HOST: Database host (default: "localhost")
    POSTGRES_PORT: Database port (default: "5432")
"""

import os
import asyncpg
from typing import Optional

# =============================================================================
# Database Configuration
# =============================================================================

# Load database credentials from environment variables with sensible defaults
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "secondbrain")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# Construct the full database connection URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# =============================================================================
# Connection Functions
# =============================================================================

async def get_db_connection():
    """
    Create and return a new database connection.
    
    Returns:
        asyncpg.Connection: An active database connection.
        
    Note:
        The caller is responsible for closing the connection when done.
        Consider using connection pooling for production workloads.
        
    Example:
        conn = await get_db_connection()
        try:
            result = await conn.fetch("SELECT * FROM documents")
        finally:
            await conn.close()
    """
    return await asyncpg.connect(DATABASE_URL)


# =============================================================================
# Schema Initialization
# =============================================================================

async def init_db():
    """
    Initialize the database schema.
    
    This function:
    1. Enables the pgvector extension for vector similarity search
    2. Creates the 'documents' table for storing original documents
    3. Creates the 'chunks' table for storing document fragments with embeddings
    4. Creates HNSW index for fast approximate nearest neighbor search
    5. Creates GIN index for full-text keyword search
    
    The schema supports hybrid retrieval (vector + keyword search) which is
    crucial for accurate RAG (Retrieval-Augmented Generation) applications.
    
    Tables:
        documents: Stores original documents with metadata
            - id: Auto-incrementing primary key
            - title: Document filename or title
            - content: Full document content
            - source: Origin of document (e.g., "upload", "web")
            - created_at: Timestamp for temporal queries
            
        chunks: Stores document fragments for retrieval
            - id: Auto-incrementing primary key
            - document_id: Foreign key to parent document
            - content: Chunk text content
            - chunk_index: Position within the original document
            - embedding: 384-dimensional vector (all-MiniLM-L6-v2)
            - content_tsv: Auto-generated tsvector for full-text search
    
    Indexes:
        chunks_embedding_idx: HNSW index for vector similarity (cosine)
        chunks_content_tsv_idx: GIN index for full-text search
    """
    conn = await get_db_connection()
    try:
        # Enable pgvector extension for vector operations
        # This must be done before creating tables with vector columns
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create documents table to store original documents
        # The created_at timestamp enables temporal queries like "what from last week?"
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title TEXT,
                content TEXT,
                source TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create chunks table for document fragments
        # We use 384 dimensions to match the all-MiniLM-L6-v2 embedding model
        # The content_tsv column is auto-generated for full-text search
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT,
                chunk_index INTEGER,
                embedding vector(384),
                content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            );
        """)
        
        # Create HNSW index for fast approximate nearest neighbor search
        # HNSW (Hierarchical Navigable Small World) provides O(log N) search time
        # vector_cosine_ops uses cosine similarity, which works well for text embeddings
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
        """)
        
        # Create GIN index for full-text search
        # GIN (Generalized Inverted Index) is optimized for text search
        # This enables fast keyword-based retrieval
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv);
        """)
        
    finally:
        await conn.close()
