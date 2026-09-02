# BibliBot

A sermon search bot for [insightfulsermons.com](https://www.insightfulsermons.com).
Someone types what they are looking for and BibliBot points them at sermon
summaries on the site: title, speaker, and a link.

It deliberately does **not** write answers about the Bible. Every line it returns
about a sermon is either the sermon's own title or a link to it, so there is
nothing for a language model to get wrong.

---

## What it can search by

| Ask for | Example | How it is answered |
| --- | --- | --- |
| A subject | "how do I forgive someone" | Vector search over the sermon text |
| A category | "what is stewardship" | The site's own category, matched exactly |
| An author | "Jonathan Edwards" | The speaker credited in the sermon summary |

Routing matters here. An author's name is a few words buried inside a 500-word
chunk and a category is not in the sermon text at all, so a pure vector search
answers both badly. Each kind of question goes to the index that can actually
answer it, and only genuine subject questions reach Pinecone.

Results are paged: the bot says how many it found ("3 of the 14 sermons by
Jonathan Edwards") and "more" walks through the rest without repeating itself.

---

## Running it

Python 3.11.

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in the keys
python server.py          # http://localhost:5000
```

`.env`:

```
PINECONE_API_KEY=...      # https://app.pinecone.io
GROQ_API_KEY=...          # https://console.groq.com  (not used at request time)
PINECONE_INDEX=sermon-index
PORT=5000
```

The Pinecone index must be 384 dimensions, cosine similarity, to match
`all-MiniLM-L6-v2`.

### Tests

```bash
python tests/test_search.py    # search routing; no network needed
python tests/test_system.py    # environment, Pinecone, imports
```

---

## How a request is served

```
question
   |
   +-- greeting?  ------------------------------> canned reply
   |
   +-- names an author?  -----------------------> catalog lookup
   |
   +-- names a site category?  -----------------> catalog + category-filtered vector search
   |
   +-- otherwise ------------------------------> vector search over sermon chunks
                                                    |
                                                    +-- nothing above threshold?
                                                          -> local keyword scan
```

The catalog (`app/catalog.py`) is built in memory at startup from
`data/sermon_data.json`: every sermon's title, URL, category, and the speakers
credited in its text. It is what makes author and category search exact.

Topic search blends Pinecone's similarity with keyword overlap, plus a bonus
when the query's words appear in the **title** — "children" in "DISCIPLINE Your
Children" is what the sermon is about, while "children" in "The Holy Spirit's
Intercession" is one passing mention.

---

## Layout

```
server.py            Flask app: routing, paging, response formatting
app/
  catalog.py         In-memory sermon catalog; author and category indexes
  search.py          Author / category / topic lookup and ranking
  embeddings.py      sentence-transformers wrapper (all-MiniLM-L6-v2, 384-dim)
  memory.py          SQLite conversation history, keyed by session id
  retrieval.py       Thin Pinecone wrapper (used by tests)
  llm.py             Groq integration; kept, but not on the request path
ingestion/
  scrape_and_embed.py  Daily Selenium scraper, run by GitHub Actions
  upload_data.py       One-time uploader for data/sermon_data.json
  bible_parser.py      Bible PDF -> JSON
  upload_bible.py      Bible verses -> Pinecone
  fix.py               Re-upload with trimmed metadata
data/
  sermon_data.json   The corpus: content, url, category per sermon
  NLT_Bible/         Parsed Bible JSON (source PDFs are gitignored)
deploy/
  weebly_embed.html  The production widget, pasted into the Weebly page editor
templates/
  index.html         Standalone chat UI, for local development
tests/
```

### Ingestion

`ingestion/scrape_and_embed.py` runs daily from GitHub Actions, adds sermons the
index has not seen, and commits the refreshed `data/sermon_data.json` back to the
repo. Run ingestion scripts from the repo root. For the Bible pipeline the order
is `bible_parser.py` then `upload_bible.py`.

---

## Endpoints

`POST /chat`

```json
{ "message": "Jonathan Edwards", "session_id": "abc-123" }
```

```json
{ "answer": "Here are 3 of the 14 sermons by Jonathan Edwards: ..." }
```

The answer is markdown; links render as `[Read the sermon](url)`. `session_id`
scopes conversation history so "more" means more for that person only. It is
optional, and anonymous sessions are supported.

`GET /health` returns status and the number of sermons in the catalog.

---

## Deployment

The API runs on Render and redeploys on push to `main`. The chat widget lives in
the Weebly page editor, not on Render: `deploy/weebly_embed.html` is the source
of truth for it and has to be pasted in by hand after a change.

Weebly drops characters on large pastes, so that file keeps every line short
with one HTML attribute per line. Copy it from the file, never from terminal
output, and check the paste by diffing the live page against it.
