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

# Tests (from repo root)
python tests/test_search.py   # Search routing — no network needed
python tests/test_system.py   # Environment, Pinecone, imports
```

## Required Environment Variables

```
PINECONE_API_KEY=<from https://app.pinecone.io>
GROQ_API_KEY=<from https://console.groq.com>
PINECONE_INDEX=sermon-index
PORT=5000  # optional
```

Pinecone index must be configured with 384 dimensions and cosine similarity.

## What this bot is

A **sermon discovery tool**, not a Q&A bot. `/chat` returns title, speaker, and
link — no generated prose. There is no LLM on the request path, so there is
nothing to hallucinate. Do not reintroduce generated answers without an explicit
client request.

**Stack:** Flask + Pinecone (sermon-index, 384-dim cosine) + SQLite conversation
memory + sentence-transformers (all-MiniLM-L6-v2). `app/llm.py` still exists and
is still covered by `tests/test_system.py`, but nothing in `/chat` calls it.

## Architecture

A question is routed to whichever index can actually answer it:

1. **Greeting** → canned reply.
2. **Author** ("Jonathan Edwards") → exact lookup in the in-memory catalog.
3. **Category** ("what is stewardship") → the site's own category, then a
   category-filtered Pinecone query to order within it.
4. **Topic** → Pinecone vector search, hybrid-ranked; if nothing clears the
   threshold, a local keyword scan over the catalog.

Author and category searches are deliberately **not** vector searches: a
speaker's name is a few words buried in a 500-word chunk and a category is not in
the sermon text at all, which is exactly why those queries used to fail.

Results are paged. The response says how many exist in total, and a follow-up
("more", "any more?") re-runs the previous subject while excluding links already
sent — recovered by parsing the assistant's own earlier messages, so it works
across gunicorn workers with no extra state.

## Structure

```
server.py          Flask routes, follow-up detection, response formatting
app/
  catalog.py       In-memory catalog from data/sermon_data.json;
                   author extraction, author/category/keyword indexes
  search.py        Author / category / topic lookup, ranking, thresholds
  embeddings.py    sentence-transformers wrapper (embed)
  memory.py        SQLite conversation history
  retrieval.py     Pinecone retrieval wrapper (used by tests)
  llm.py           Groq integration — NOT on the request path
ingestion/         All data pipeline scripts (run from repo root)
data/
  sermon_data.json Corpus: content, url, category per sermon
  NLT_Bible/       Processed Bible JSON (PDFs are gitignored)
deploy/
  weebly_embed.html  Production widget source; pasted into Weebly by hand
templates/
  index.html       Standalone chat UI for local development
tests/
```

## Key Relationships

- `server.py` owns routing and formatting; `app/search.py` owns ranking. Scoring
  constants (thresholds, title bonus) live at the top of `app/search.py`.
- The catalog is the source of truth for **speaker names and categories**;
  Pinecone is the source of truth for **semantic similarity**. Pinecone's sermon
  vectors carry `category` metadata but no `author`.
- Speaker names are extracted from sermon text by regex, never by an LLM.
  Coverage is 224/238 sermons across 39 speakers; the rest are genuinely
  unattributed ("A Sermon Summary"), and showing no name is correct for those.
- Possessive names ("Tim Kellers") are folded into the canonical form only when
  the shortened name is credited elsewhere in the corpus — which is why it does
  not break Francis Collins.
- Ingestion scripts all use `sys.path.append` to find the repo root.
- Bible pipeline order: `bible_parser.py` → `upload_bible.py` (or `fix.py` if
  metadata is too large).

## Gotchas

- `chat_memory.db` is untracked and holds live user conversations. Never
  `git add .` in this repo.
- The Weebly widget is production config that is applied by hand. Editing
  `deploy/weebly_embed.html` changes nothing until it is pasted into the Weebly
  page editor. Weebly silently drops characters on large pastes, so keep lines
  short with one HTML attribute per line, and verify by diffing the live page.
