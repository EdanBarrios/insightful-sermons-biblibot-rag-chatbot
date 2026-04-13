import os
import logging
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone
from llm import generate_answer
from memory import init_db, save_message, get_recent_messages
from embeddings import embed

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("sermon-index")

# -------------------- Setup --------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

init_db()

# -------------------- Helpers --------------------

def extract_keywords(text):
    """Extract meaningful keywords from text"""
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out common stop words and short words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'be', 'do', 'does', 'did', 'have', 'has', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'how', 'why', 'when', 'where', 'does'}
    keywords = {w for w in words if len(w) > 2 and w not in stop_words}
    return keywords


def calculate_keyword_score(text, keywords):
    """Calculate how many keywords appear in text"""
    if not keywords:
        return 0
    text_lower = text.lower()
    matches = sum(1 for keyword in keywords if keyword in text_lower)
    return matches / len(keywords)


def hybrid_search(semantic_results, question):
    """
    Combine semantic search results with keyword matching.
    Returns re-ranked results based on hybrid score.
    """
    question_keywords = extract_keywords(question)
    logger.info(f"Question keywords: {question_keywords}")
    
    # Calculate hybrid scores
    scored_matches = []
    
    for match in semantic_results.get("matches", []):
        semantic_score = match.get("score", 0)
        
        # Get text from metadata
        metadata = match.get("metadata", {})
        text = (metadata.get("text", "") + " " + metadata.get("title", "")).lower()
        
        # Calculate keyword score
        keyword_score = calculate_keyword_score(text, question_keywords)
        
        # Hybrid score: 60% semantic, 40% keyword
        hybrid_score = (semantic_score * 0.6) + (keyword_score * 0.4)
        
        # Manually add scores to match without unpacking
        match["hybrid_score"] = hybrid_score
        match["keyword_score"] = keyword_score
        scored_matches.append(match)
    
    # Re-rank by hybrid score
    scored_matches.sort(key=lambda m: m.get("hybrid_score", 0), reverse=True)
    
    logger.info(f"Top match hybrid score: {scored_matches[0].get('hybrid_score', 'N/A') if scored_matches else 'N/A'}")
    logger.info(f"Top match keyword score: {scored_matches[0].get('keyword_score', 'N/A') if scored_matches else 'N/A'}")
    
    return scored_matches


def extract_single_verse(reference, verse_text):
    cleaned = " ".join((verse_text or "").split()).strip()
    ref = (reference or "").strip()

    if not cleaned:
        return ref, ""

    m = re.match(r"^([1-3]?\s?[A-Za-z]+\s+\d+:\d+)\s+(.*)$", cleaned)
    if m:
        ref = m.group(1).strip()
        remainder = m.group(2).strip()
    else:
        remainder = cleaned

    next_marker = re.search(r"\b[1-3]?\s?[A-Za-z]+\s+\d+:\d+\b", remainder)
    if next_marker:
        remainder = remainder[: next_marker.start()].strip()

    return ref, remainder


def build_formatted_response(answer, sources=None, bible_verses=None):
    sections = []

    # -------- Summary --------
    sections.append(answer.strip())

    # -------- Bible Verse --------
    if bible_verses:
        verse = bible_verses[0]
        ref, text = extract_single_verse(
            verse.get("reference", ""), verse.get("text", "")
        )

        if text:
            sections.append(f'Bible Verse:\n"{text}"\n— {ref}')

    # -------- Sermons --------
    if sources:
        lines = ["Related sermons:"]
        seen = set()

        for source in sources[:2]:  # limit to 2
            title = source.get("title", "Sermon").replace('"', "").strip()
            url = source.get("url", "").strip()

            if url and url not in seen:
                lines.append(f"- [{title}]({url})")
                seen.add(url)

        sections.append("\n".join(lines))

    return "\n\n".join(sections).strip()


def save_turn(session_id, user_msg, assistant_msg):
    save_message(session_id, "user", user_msg)
    save_message(session_id, "assistant", assistant_msg)


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

        # -------- Greeting --------
        greetings = [
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
        ]

        if question.lower() in greetings or len(question.split()) == 1:
            greeting = (
                "Hello! I'm BibliBot, here to help you explore sermons and the Bible. "
                "Ask me anything about faith, relationships, or spiritual growth."
            )
            formatted = greeting

            save_turn(session_id, question, formatted)
            return jsonify({"answer": formatted})

        # -------- Memory --------
        history = get_recent_messages(session_id, limit=6)

        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        )

        # -------- Retrieval with Hybrid Search --------
        logger.info("Starting embed")
        vector = embed(question)
        logger.info("Finished embed")

        logger.info("Starting Pinecone query")
        res = index.query(vector=vector, top_k=10, include_metadata=True)
        logger.info("Finished Pinecone query")

        # -------- Hybrid Search Ranking --------
        logger.info("Starting hybrid search ranking")
        hybrid_results = hybrid_search(res, question)
        logger.info("Finished hybrid search ranking")

        sources = []
        bible_verses = []
        context_chunks = []
        seen_urls = set()

        if hybrid_results:
            # Use higher threshold with hybrid scoring
            relevant = [m for m in hybrid_results if m.get("hybrid_score", 0) > 0.35]

            for match in relevant:
                md = match.get("metadata", {})
                doc_type = md.get("type", "sermon")

                if "text" in md:
                    context_chunks.append(md["text"])

                # Only add Bible verse if it has HIGH keyword match
                if doc_type == "bible" and not bible_verses:
                    keyword_score = m.get("keyword_score", 0)
                    if keyword_score > 0.5:  
                        bible_verses.append({
                            "reference": md.get("reference", ""),
                            "text": md.get("text", "")
                        })

                if doc_type != "bible":
                    url = md.get("url", "")
                    if url and url not in seen_urls:
                        sources.append(
                            {
                                "title": md.get("title", "Sermon"),
                                "url": url,
                                "content": md.get("text", ""),
                            }
                        )
                        seen_urls.add(url)

        context = "\n\n---\n\n".join(context_chunks)

        # -------- Combine Context --------
        combined_context = context

        if history_text:
            combined_context = (
                f"Conversation history:\n{history_text}\n\n"
                f"Relevant context:\n{context}"
            )

        # -------- LLM --------
        logger.info("Starting LLM generation")
        answer = generate_answer(combined_context, question, has_sermon_content=True)
        logger.info("Finished LLM generation")

        if not answer:
            answer = "I'm sorry, I couldn't generate a response. Please try again."

        # -------- Format --------
        final_answer = build_formatted_response(
            answer=answer, sources=sources, bible_verses=bible_verses
        )

        # -------- Save --------
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