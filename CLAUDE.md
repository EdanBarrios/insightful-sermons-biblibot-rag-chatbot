# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Install dependencies (Python 3.11 required)
pip install -r requirements.txt

# Start the Flask server
python server.py  # Runs on port 5000

# Run system verification tests
python test_system.py
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
1. User query → embedded via `embeddings.py` → Pinecone vector search (top_k=10) in `retrieval.py`
2. Hybrid ranking in `server.py`: 60% semantic + 40% keyword score
3. Context + conversation history (SQLite via `memory.py`) → Groq LLM via `llm.py`
4. Response with sermon citations and relevant Bible verses

**Two chat endpoints:**
- `POST /chat` — New style: structured sections (Quick Answer / Path Forward / Theological Foundation), Bible verse integration, sermon citations
- `POST /chat-legacy` — Old style: ≤8 sentence answers, TF-IDF source matching via `legacy_source_match.py`, sermon-only context

**Supporting infrastructure:**
- `ingestion/scrape_and_embed.py` — Selenium scraper for insightfulsermons.com, runs daily via GitHub Actions (`.github/workflows/daily_scraping.yml`)
- `data/sermon_data.json` — Cached sermon metadata used for TF-IDF matching in the legacy endpoint
- `data/NLT_Bible/` — Bible verse JSON files loaded into Pinecone as a separate namespace

**Health check:** `GET /health`

## Key Relationships

- `server.py` imports from all other modules and owns both endpoints and hybrid ranking logic
- `llm.py` has separate functions for new-style vs legacy-style answer generation
- `legacy_source_match.py` provides TF-IDF matching used exclusively by `/chat-legacy` — if missing, the legacy endpoint will fail at import time
- Conversation history is keyed by session ID (passed in request body); anonymous sessions are supported
