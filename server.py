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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

init_db()

# -------------------- Helpers --------------------

def extract_single_verse(reference, verse_text):
    cleaned = " ".join((verse_text or "").split()).strip()
    ref = (reference or "").strip()

    if not cleaned:
        return ref, ""

    m = re.match(r'^([1-3]?\s?[A-Za-z]+\s+\d+:\d+)\s+(.*)$', cleaned)
    if m:
        ref = m.group(1).strip()
        remainder = m.group(2).strip()
    else:
        remainder = cleaned

    next_marker = re.search(r'\b[1-3]?\s?[A-Za-z]+\s+\d+:\d+\b', remainder)
    if next_marker:
        remainder = remainder[:next_marker.start()].strip()

    return ref, remainder


def build_formatted_response(answer, sources=None, bible_verses=None):
    sections = []

    # -------- Summary --------
    sections.append(answer.strip())

    # -------- Bible Verse --------
    if bible_verses:
        verse = bible_verses[0]
        ref, text = extract_single_verse(
            verse.get("reference", ""),
            verse.get("text", "")
        )

        if text:
            sections.append(f'Bible Verse:\n"{text}"\n— {ref}')

    # -------- Sermons --------
    if sources:
        lines = ["Related sermons:"]
        seen = set()

        for source in sources[:2]:  # limit to 2
            title = source.get("title", "Sermon").replace('"', '').strip()
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
            'hi', 'hello', 'hey', 'yo', 'sup',
            'greetings', 'good morning', 'good afternoon', 'good evening'
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
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        )

        # -------- Retrieval --------
        logger.info("Starting embed")
        vector = embed(question)
        logger.info("Finished embed")

        logger.info("Starting Pinecone query")
        res = index.query(vector=vector, top_k=5, include_metadata=True)
        logger.info("Finished Pinecone query")

        sources = []
        bible_verses = []
        context_chunks = []
        seen_urls = set()

        if res and "matches" in res:
            relevant = [m for m in res["matches"] if m.get("score", 0) > 0.2]

            for match in relevant:
                md = match.get("metadata", {})
                doc_type = md.get("type", "sermon")

                if "text" in md:
                    context_chunks.append(md["text"])

                if doc_type == "bible" and not bible_verses:
                    bible_verses.append({
                        "reference": md.get("reference", ""),
                        "text": md.get("text", "")
                    })

                if doc_type != "bible":
                    url = md.get("url", "")
                    if url and url not in seen_urls:
                        sources.append({
                            "title": md.get("title", "Sermon"),
                            "url": url,
                            "content": md.get("text", "")
                        })
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
            answer=answer,
            sources=sources,
            bible_verses=bible_verses
        )

        # -------- Save --------
        save_turn(session_id, question, final_answer)

        return jsonify({"answer": final_answer})

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({
            "answer": "Something went wrong. Please try again."
        }), 500


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
