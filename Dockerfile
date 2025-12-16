FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (ffmpeg for audio, tesseract for OCR)
RUN apt-get update && apt-get install -y gcc libpq-dev tesseract-ocr ffmpeg && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Reinstall torch CPU version to keep image smaller/compatible
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY backend ./backend
COPY second_brain_doc.txt .
COPY run_ingestion.py .

# Expose port
EXPOSE 8000

# Command to run (we'll override this in compose maybe, or just default)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
