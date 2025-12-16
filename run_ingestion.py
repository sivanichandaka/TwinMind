import asyncio
import os
import sys

# Add current directory to path so we can import backend modules
sys.path.append(os.getcwd())

from backend.ingestion import ingest_document
from backend.database import init_db

async def main():
    # Initialize Database Schema
    await init_db()

    file_path = "second_brain_doc.txt"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        # return - don't return, continue to folder ingestion

    print(f"Reading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("Ingesting initial document...")
    try:
         await ingest_document("Architecture Doc", content, source="seed_script")
    except Exception as e:
        print(f"Error seeding doc: {e}")

    # NEW: Run Folder Ingestion
    from backend.ingestion import ingest_directory
    print("Scanning data/ folder for new files (PDF/Image/Text)...")
    await ingest_directory()
    print("Ingestion flow complete.")

    print("Ingesting document (this triggers embedding generation)...")
    try:
        doc_id, chunks = await ingest_document("Architecture Doc", content, source="seed_script")
        print(f"Success! Document ID: {doc_id}, Chunks detected: {chunks}")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    asyncio.run(main())
