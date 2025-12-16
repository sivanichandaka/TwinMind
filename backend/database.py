import os
import asyncpg
from typing import Optional

DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "secondbrain")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

async def get_db_connection():
    return await asyncpg.connect(DATABASE_URL)

async def init_db():
    conn = await get_db_connection()
    try:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title TEXT,
                content TEXT,
                source TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create chunks table
        # We use 384 dimensions for all-MiniLM-L6-v2
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
        
        # Create HNSW index for faster vector search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
        """)
        
        # Create GIN index for full-text search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv);
        """)
        
    finally:
        await conn.close()
