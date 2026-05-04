# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Install dependencies (Python 3.11 required)
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Start the Flask server
python server.py  # Runs on port 5000

# Run system verification tests (from repo root)
python tests/test_system.py
```

## Required Environment Variables

```
PINECONE_API_KEY=<from https://app.pinecone.io>
GROQ_API_KEY=<from https://console.groq.com>
PINECONE_INDEX=sermon-index
PORT=5000  # optional
```

Pinecone index must be configured with 384 dimensions and cosine similarity.

## Architecture

**Stack:** Flask + Groq (Llama 3.3 70B) + Pinecone vector DB + SQLite conversation memory + sentence-transformers embeddings (all-MiniLM-L6-v2, 384-dim)

**Request flow:**
1. User query → embedded via `app/embeddings.py` → Pinecone vector search (top_k=10)
2. Hybrid ranking in `server.py`: 60% semantic + 40% keyword score, threshold hybrid>0.5 + semantic>0.45
3. Context + conversation history (SQLite via `app/memory.py`) → Groq LLM via `app/llm.py`
4. Response with sermon citations and relevant Bible verses; falls back to general Biblical knowledge when no sermons match

**Chat endpoint:** `POST /chat` — structured response (Quick Answer / Your Path Forward / Theological Foundation), Bible verse integration, sermon citations

**Health check:** `GET /health`

## Structure

```
app/               # Core application package
  llm.py           # Groq LLM integration (generate_answer)
  embeddings.py    # sentence-transformers wrapper (embed)
  memory.py        # SQLite conversation history
  retrieval.py     # Pinecone retrieval wrapper (used by tests)
ingestion/         # All data pipeline scripts
  scrape_and_embed.py   # Daily Selenium scraper (run by GitHub Actions)
  upload_data.py        # One-time sermon JSON uploader
  bible_parser.py       # PDF→JSON parser (run from repo root)
  upload_bible.py       # Bible verse uploader to Pinecone
  fix.py                # Re-upload with trimmed metadata (Pinecone size fix)
data/
  sermon_data.json      # Cached sermon metadata
  NLT_Bible/            # Processed Bible JSON files (PDFs are gitignored)
tests/
  test_system.py        # Component verification (env, Pinecone, LLM, etc.)
static/            # Web assets
templates/
  index.html       # Chat widget UI
```

## Key Relationships

- `server.py` imports from `app.*` and owns the hybrid ranking logic and response formatting
- `app/llm.py` has two answer paths: sermon-grounded (with context) and general Biblical knowledge (no sermon match)
- Conversation history is keyed by session ID (passed in request body); anonymous sessions are supported
- Ingestion scripts all use `sys.path.append` to find the repo root — run them from the repo root
- Bible pipeline order: `bible_parser.py` → `upload_bible.py` (or `fix.py` if metadata is too large)
