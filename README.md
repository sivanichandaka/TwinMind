# 🧠 Second Brain

```
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║   ┌─────────────────────────────────┐     ║
    ║   │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │     ║
    ║   │  ░  YOUR KNOWLEDGE  ░░░░░░░░░░  │     ║
    ║   │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │     ║
    ║   │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │     ║
    ║   │  ░░░░ ALWAYS REMEMBERED ░░░░░   │     ║
    ║   │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │     ║
    ║   └─────────────────────────────────┘     ║
    ║              🤖 AI Assistant             ║
    ╚═══════════════════════════════════════════╝
```

> *Your personal AI that remembers everything you teach it.*

---

## ✨ What is this?

A **privacy-first** knowledge assistant that:

- 📄 Ingests your **PDFs, audio, images, and text**
- 🔍 Finds answers using **hybrid search** (meaning + keywords)
- 🤖 Answers questions with a **local LLM** (your data stays on your machine!)

---

## 🚀 Quick Start

```bash
# 1. Start everything
docker-compose up -d

# 2. Wait for the AI model to download (~5 min first time)
docker logs -f twinmind-ollama

# 3. Open your browser
open http://localhost:8000
```

That's it! 🎉

---

## 📁 What can I upload?

| Format | How it works |
|--------|--------------|
| 📄 PDF | Text extraction |
| 🎤 Audio | Whisper transcription |
| 🖼️ Images | OCR (Tesseract) |
| 📝 Text/Markdown | Direct ingestion |

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────┐
│  Frontend    │  HTML + Vanilla JS       │
├──────────────┼──────────────────────────┤
│  Backend     │  FastAPI (Python)        │
├──────────────┼──────────────────────────┤
│  Database    │  PostgreSQL + pgvector   │
├──────────────┼──────────────────────────┤
│  LLM         │  Ollama (llama3)         │
├──────────────┼──────────────────────────┤
│  Audio       │  OpenAI Whisper          │
└─────────────────────────────────────────┘
```

---

## 🔒 Privacy

Your data **never leaves your computer**. Everything runs locally in Docker containers.

---

## 📖 More Docs

- [Architecture](ARCHITECTURE.md) — System design details
- [Design Doc](data/DESIGN.md) — Deep dive into components

---

## 💡 Example Queries

```
"What did I save about project management?"
"Find my notes from last week"
"Summarize the PDF I uploaded"
```

---

Made with ❤️ and ☕

```
   ( (
    ) )
  ........
  |      |]
  \      /
   `----'
```
