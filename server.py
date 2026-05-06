import json
import os
import logging
import re
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone

from app.memory import init_db, save_turn, get_recent_messages
from app.embeddings import embed
from app.llm import generate_answer

load_dotenv()

# -------------------- Setup --------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("sermon-index")

init_db()

# -------------------- Constants --------------------

_GREETINGS = frozenset([
    "hi", "hello", "hey", "yo", "sup",
    "greetings", "good morning", "good afternoon", "good evening",
])

# Pre-compiled regex patterns for author extraction
_NAME = r'([A-Z][a-zA-Z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zA-Z]+)?)'
_START_PATTERNS = [re.compile(p) for p in [
    rf'^A\s+{_NAME}\s+Sermon\s+Summary',
    rf'^A\s+Lesson\s+from\s+{_NAME}',
    rf'^of\s+a\s+[Ll]esson\s+from\s+{_NAME}',
]]
_END_PATTERNS = [re.compile(p) for p in [
    rf'from\s+a\s+{_NAME}\s+[Ss]ermon',
    rf'[Ss]ermon\s+by\s+{_NAME}',
    rf'\sby\s+{_NAME}\s*(?:https?://)',
]]

_VERSE_REF_RE = re.compile(r"^([1-3]?\s?[A-Za-z]+\s+\d+:\d+)\s+(.*)$")
_NEXT_VERSE_RE = re.compile(r"\b[1-3]?\s?[A-Za-z]+\s+\d+:\d+\b")

# -------------------- Helpers --------------------

def extract_author_from_text(text: str) -> str:
    text = text.strip()

    for pattern in _START_PATTERNS:
        m = pattern.match(text)
        if m:
            author = m.group(1).strip()
            if author:
                return author

    tail = text[-300:] if len(text) > 300 else text
    for pattern in _END_PATTERNS:
        m = pattern.search(tail)
        if m:
            author = m.group(1).strip()
            if author:
                return author

    return ""


def extract_keywords(text: str) -> set:
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'be', 'do',
        'does', 'did', 'have', 'has', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'how', 'why', 'when', 'where', 'does',
    }
    return {
        w for w in re.findall(r'\b\w+\b', text.lower())
        if len(w) > 2 and w not in stop_words
    }


def calculate_keyword_score(text: str, keywords: set) -> float:
    """Word-boundary keyword match — avoids substring false positives."""
    if not keywords:
        return 0
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    return len(keywords & text_words) / len(keywords)


def hybrid_search(semantic_results, question: str) -> list:
    question_keywords = extract_keywords(question)
    logger.info(f"Question keywords: {sorted(question_keywords)}")

    scored_matches = []
    for match in semantic_results.get("matches", []):
        semantic_score = match.get("score", 0)
        metadata = match.get("metadata", {})
        text = (metadata.get("text", "") + " " + metadata.get("title", "")).lower()
        keyword_score = calculate_keyword_score(text, question_keywords)
        match["hybrid_score"] = (semantic_score * 0.6) + (keyword_score * 0.4)
        match["keyword_score"] = keyword_score
        scored_matches.append(match)

    scored_matches.sort(key=lambda m: m.get("hybrid_score", 0), reverse=True)

    if scored_matches:
        top = scored_matches[0]
        logger.info(f"Top match hybrid score: {top.get('hybrid_score', 'N/A')}")
        logger.info(f"Top match keyword score: {top.get('keyword_score', 'N/A')}")

    return scored_matches


def extract_single_verse(reference: str, verse_text: str) -> tuple:
    cleaned = " ".join((verse_text or "").split()).strip()
    ref = (reference or "").strip()

    if not cleaned:
        return ref, ""

    m = _VERSE_REF_RE.match(cleaned)
    if m:
        ref = m.group(1).strip()
        remainder = m.group(2).strip()
    else:
        remainder = cleaned

    next_marker = _NEXT_VERSE_RE.search(remainder)
    if next_marker:
        remainder = remainder[: next_marker.start()].strip()

    return ref, remainder


def build_formatted_response(answer: str, sources=None, bible_verses=None) -> str:
    sections = [answer.strip()]

    if bible_verses:
        ref, text = extract_single_verse(
            bible_verses[0].get("reference", ""),
            bible_verses[0].get("text", "")
        )
        if text:
            sections.append(f'Bible Verse:\n"{text}"\n— {ref}')

    if sources:
        lines = ["Related sermons:"]
        seen = set()
        for source in sources[:2]:
            title = source.get("title", "Sermon").replace('"', "").strip()
            url = source.get("url", "").strip()
            if url and url not in seen:
                lines.append(f"- [{title}]({url})")
                seen.add(url)
        sections.append("\n".join(lines))

    return "\n\n".join(sections).strip()


# -------------------- Routes --------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("message", "").strip()
        session_id = data.get("session_id", "").strip() or "anonymous"

        if not question:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Question: {question}")

        if question.lower() in _GREETINGS or len(question.split()) == 1:
            greeting = (
                "Hello! I'm BibliBot, here to help you explore sermons and the Bible. "
                "Ask me anything about faith, relationships, or spiritual growth."
            )
            save_turn(session_id, question, greeting)
            return jsonify({"answer": greeting})

        history = get_recent_messages(session_id, limit=6)
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        )

        logger.info("Starting embed")
        vector = embed(question)
        logger.info("Finished embed")

        logger.info("Starting Pinecone query")
        res = index.query(vector=vector, top_k=10, include_metadata=True)
        logger.info("Finished Pinecone query")

        logger.info("Starting hybrid search ranking")
        hybrid_results = hybrid_search(res, question)
        logger.info("Finished hybrid search ranking")

        sources = []
        bible_verses = []
        context_chunks = []
        seen_urls = set()

        relevant = [
            m for m in hybrid_results
            if m.get("hybrid_score", 0) > 0.5 and m.get("score", 0) > 0.45
        ]

        for match in relevant:
            md = match.get("metadata", {})
            doc_type = md.get("type", "sermon")

            if "text" in md:
                if doc_type != "bible":
                    title = md.get("title", "").strip()
                    category = md.get("category", "").strip()
                    author = extract_author_from_text(md["text"])
                    header_parts = [f'Sermon: "{title}"']
                    if author:
                        header_parts.append(f"by {author}")
                    if category:
                        header_parts.append(f"[{category}]")
                    context_chunks.append(f"[{', '.join(header_parts)}]\n{md['text']}")
                else:
                    context_chunks.append(md["text"])

            if doc_type == "bible" and not bible_verses:
                if match.get("keyword_score", 0) > 0.6:
                    bible_verses.append({
                        "reference": md.get("reference", ""),
                        "text": md.get("text", "")
                    })

            if doc_type != "bible":
                url = md.get("url", "")
                if url and url not in seen_urls:
                    sources.append({"title": md.get("title", "Sermon"), "url": url})
                    seen_urls.add(url)

        context = "\n\n---\n\n".join(context_chunks)
        combined_context = (
            f"Conversation history:\n{history_text}\n\nRelevant context:\n{context}"
            if history_text else context
        )

        bible_verse_context = ""
        if bible_verses:
            ref, text = extract_single_verse(
                bible_verses[0].get("reference", ""),
                bible_verses[0].get("text", "")
            )
            if text:
                bible_verse_context = f'{ref}: "{text}"'

        logger.info("Starting LLM generation")
        answer = generate_answer(
            combined_context, question,
            has_sermon_content=bool(context_chunks),
            bible_verse_context=bible_verse_context
        )
        logger.info("Finished LLM generation")

        if not answer:
            answer = "I'm sorry, I couldn't generate a response. Please try again."

        final_answer = build_formatted_response(
            answer=answer, sources=sources, bible_verses=bible_verses
        )
        save_turn(session_id, question, final_answer)

        return jsonify({"answer": final_answer})

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"answer": "Something went wrong. Please try again."}), 500


# -------------------- Errors --------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500


# -------------------- Run --------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
