# 🕊️ BibliBot - Biblical Sermon RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions about Biblical sermons using AI.

---

## ✨ Features

- 🤖 **AI-Powered Answers** - Uses Groq's Llama 3.3 70B for natural, conversational responses
- 📚 **Sermon Knowledge Base** - 20+ sermons across topics like Faith, Grace, Prayer, Love, and Hope
- 🔍 **Smart Search** - Vector similarity search via Pinecone for relevant context retrieval
- 💬 **Conversational UI** - Clean, responsive chat widget that works on any device
- 🔄 **Auto-Updates** - Daily automated scraping of new sermons at midnight
- ⚡ **Fast & Reliable** - Direct SDK integration (no LangChain) for stability

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Pinecone account (free tier)
- Groq API key (free tier)

### Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd [your-directory-name]

# Create virtual environment
conda create -n rag_env python=3.11 -y
conda activate rag_env

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration
Create a `.env` file with:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Getting API Keys:**
- Groq: https://console.groq.com (free, fast, 30 req/min)
- Pinecone: https://app.pinecone.io (free tier: 100k vectors)

### Pinecone Setup

1. Create index named `sermon-index`
2. Dimensions: **384** (for all-MiniLM-L6-v2)
3. Metric: **cosine**

### Initial Data Load

```bash
# Option 1: Scrape fresh data
python ingestion/scrape_and_embed_fixed.py

# Option 2: Upload existing data
python ingestion/upload_data.py
```

### Run the Server

```bash
python server.py
```

Open http://localhost:5001 in your browser.

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   User UI   │─────▶│ Flask Server │─────▶│  Pinecone   │
│ (index.html)│      │  (server.py) │      │  (vectors)  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Groq LLM    │
                     │ (Llama 3.3)  │
                     └──────────────┘
```

**Flow:**
1. User asks question → Flask endpoint
2. Question embedded → Search Pinecone for relevant sermons
3. Context + question → Groq LLM
4. Generated answer → User

---

## 📁 Project Structure

```
RAG_LLM_2024/
├── .github/workflows/
│   └── daily_scraping.yml      # Automated daily scraping
├── data/                       # Scraped sermon backups
├── ingestion/
│   ├── scrape_and_embed_fixed.py  # Main scraper
│   ├── upload_existing_data.py    # Manual data upload
│   └── debug_scraper.py           # Debugging tool
├── static/
│   ├── chatbot_logo.png
│   └── send_logo.png
├── templates/
│   └── index.html              # Frontend UI
├── embeddings.py               # Embedding generation
├── llm.py                      # Groq LLM integration
├── retrieval.py                # Pinecone queries
├── server.py                   # Flask API server
├── test_system.py              # System verification
├── requirements.txt            # Dependencies
├── .env                        # API keys (gitignored)
└── README.md
```

---

## 🔧 Key Components

### `server.py`
Flask server with `/chat` endpoint. Handles retrieval + generation pipeline.

### `llm.py`
Groq API integration with smart routing:
- Greetings → Friendly responses
- Questions → RAG-based answers from sermons

### `retrieval.py`
Pinecone vector search for relevant sermon chunks.

### `embeddings.py`
Sentence-transformers (all-MiniLM-L6-v2) for query/document embeddings.

### `ingestion/scrape_and_embed_fixed.py`
Scrapes insightfulsermons.com, chunks content, embeds, and uploads to Pinecone.

---

## 🤖 Automated Daily Scraping

GitHub Actions runs scraping daily at midnight UTC.

**Setup:**
1. Add secrets to GitHub repo:
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
2. Push `.github/workflows/daily_scraping.yml`
3. Manual trigger: Actions tab → Run workflow

**Logs:** Check GitHub Actions tab for run history.

---

## 🧪 Testing

```bash
# Verify entire system
python test_system.py

# Test specific components
python -c "from embeddings import embed; print(len(embed('test')))"
python -c "from retrieval import retrieve; print(len(retrieve('faith')))"
```

---

## 📊 API Endpoints

### `GET /`
Serves the main chat UI (index.html).

### `POST /chat`
Main chatbot endpoint.

**Request:**
```json
{
  "message": "What is faith?"
}
```

**Response:**
```json
{
  "answer": "Faith is trust in God. It means believing in what you cannot see..."
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "BibliBot RAG API",
  "version": "2.0"
}
```

---

## 🚀 Deployment

### Option 1: Render (Recommended)

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect repository
4. Add environment variables
5. Deploy

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
python server.py
```

### Option 2: Railway

Similar setup to Render. Add env vars in Railway dashboard.

### Option 3: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "server.py"]
```

---

## 🔒 Security Notes

- Never commit `.env` file (gitignored)
- Rotate API keys regularly
- Use environment variables for production
- Keep dependencies updated for security patches

---

## 🐛 Troubleshooting

### "No documents retrieved"
- Check Pinecone has data: Run `python test_system.py`
- Verify index dimension is 384
- Re-run ingestion script

### "Groq API error"
- Check API key in `.env`
- Verify free tier quota at console.groq.com
- Check rate limits (30 req/min)

### Scraping fails
- Website structure may have changed
- Check `ingestion/ingestion.log` for details
- Use `debug_scraper.py` to investigate

### Frontend doesn't connect
- Ensure server is running on port 5001
- Check CORS settings in `server.py`
- Verify browser console for errors

---

## 📈 Performance

- **Query latency:** 1-3 seconds
- **Embedding:** ~100ms
- **Retrieval:** ~200ms
- **LLM generation:** 500ms-2s
- **Throughput:** ~20 queries/minute (Groq free tier)

---

## 🛠️ Tech Stack

- **Backend:** Flask 3.1.0
- **LLM:** Groq (Llama 3.3 70B)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** Pinecone
- **Scraping:** Selenium + Chrome
- **Frontend:** Vanilla HTML/CSS/JavaScript

---

## 📝 Maintenance

### Weekly Tasks
- Check GitHub Actions logs
- Verify scraping succeeded
- Monitor Pinecone vector count

### Monthly Tasks
- Review Groq usage
- Update dependencies if needed (run `pip-audit`)
- Check for sermon website changes

### Quarterly Tasks
- Rotate API keys
- Review and optimize prompts
- Update documentation

---

## 🎯 Roadmap

- [ ] Add conversation memory (multi-turn chat)
- [ ] Include sermon citations in responses
- [ ] Support more sermon sources
- [ ] Add analytics dashboard
- [ ] Multi-language support
- [ ] Voice input/output

---

## 👥 Contributors

- Edan Barrios - Developer
- Luke Kottom - Developer (2024)
