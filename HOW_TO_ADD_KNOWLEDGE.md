# How to Add Knowledge

## The Data Folder
The system now watches the `data/` folder in your project root.

## Supported Formats
Currently, the system ingests:
*   **`.txt`** (Text Files)
*   **`.md`** (Markdown Files)

*(Structure is in place for Images/PDFs/Videos, but requires adding extra Python libraries)*.

## Step-by-Step Guide
1.  **Drop Files**: Place your files into the `data/` folder in your project directory.
    *   Examples: `meeting_notes.txt`, `project_roadmap.md`.
    *   You can organize them into subfolders (e.g., `data/finance/`, `data/tech/`).
2.  **Re-Run Ingestion**:
    Run this command in your terminal to process all new files:
    ```bash
    docker exec twinmind-backend python run_ingestion.py
    ```
3.  **Chat**: The new information is instantly available to the bot.

## Adding Support for More Types
To add Image/PDF support in the future:
1.  Add `pytesseract` (for OCR) or `pypdf` to `backend/requirements.txt`.
2.  Uncomment the logic in `backend/ingestion.py`.
3.  Rebuild the container.
