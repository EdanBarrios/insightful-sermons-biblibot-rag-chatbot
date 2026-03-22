import os
import logging
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from retrieval import retrieve
from llm import generate_answer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("✅ Environment variables loaded")

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


def build_formatted_response(answer, sources=None, bible_verses=None):
    sections = []

    # Main answer only once
    if answer:
        sections.append(answer.strip())

    # One single Bible verse only
    if bible_verses:
        verse = bible_verses[0]
        reference = verse.get("reference", "").strip()
        verse_text = verse.get("text", "").strip()

        one_verse_ref = reference
        one_verse_text = verse_text

        if verse_text:
            cleaned = " ".join(verse_text.split())

            # Case 1: text starts with something like "Gal 5:17 ..."
            start_match = re.match(r'^([1-3]?\s?[A-Za-z]+\s+\d+:\d+)\s+(.*)$', cleaned)
            if start_match:
                one_verse_ref = start_match.group(1).strip()
                remainder = start_match.group(2).strip()

                # Stop when the next verse marker appears, like "Gal 5:18"
                next_verse = re.search(r'\b[1-3]?\s?[A-Za-z]+\s+\d+:\d+\b', remainder)
                if next_verse:
                    one_verse_text = remainder[:next_verse.start()].strip()
                else:
                    one_verse_text = remainder.strip()
            else:
                # Fallback: just take first sentence
                sentence_parts = re.split(r'(?<=[.!?])\s+', cleaned)
                one_verse_text = sentence_parts[0].strip()

        verse_lines = []
        if one_verse_text:
            verse_lines.append(f'Bible Verse: "{one_verse_text}"')
        if one_verse_ref:
            verse_lines.append(f'— {one_verse_ref}')

        if verse_lines:
            sections.append("\n".join(verse_lines))

    # Related sermons with markdown links
    if sources:
        link_lines = ["Related sermons:"]
        seen_links = set()

        for source in sources:
            title = source.get("title", "Sermon").strip()
            url = source.get("url", "").strip()

            if url and url not in seen_links:
                link_lines.append(f"- [{title}]({url})")
                seen_links.add(url)

        if len(link_lines) > 1:
            sections.append("\n".join(link_lines))

    return "\n\n".join(sections).strip()


@app.route("/", methods=["GET"])
def home():
    """Serve the main chat UI"""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "BibliBot RAG API",
        "version": "2.0"
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    Expects: {"message": "user question"}
    Returns: {"answer": "formatted answer"}
    """
    try:
        data = request.get_json()
        if not data:
            logger.warning("Request missing JSON body")
            return jsonify({"error": "No JSON data provided"}), 400

        question = data.get("message", "").strip()
        if not question:
            logger.warning("Empty message received")
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"📩 Received question: {question[:100]}...")

        greetings = [
            'hi', 'hello', 'hey', 'yo', 'sup', 'howdy',
            'greetings', 'good morning', 'good afternoon', 'good evening'
        ]

        if question.lower().strip() in greetings or len(question.split()) == 1:
            greeting_answer = (
                "Hello! I'm BibliBot, here to help you explore our sermons and the Bible. "
                "Ask me about faith, grace, prayer, love, hope, or any Biblical topic!"
            )
            return jsonify({"answer": f"Answer:\n{greeting_answer}"})

        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("sermon-index")

        from embeddings import embed
        vector = embed(question)

        res = index.query(
            vector=vector,
            top_k=5,
            include_metadata=True
        )

        if not res or "matches" not in res or not res["matches"]:
            logger.warning("No documents retrieved")
            answer = generate_answer("", question, has_sermon_content=False)
            return jsonify({"answer": f"Answer:\n{answer}"})

        relevant_matches = [m for m in res["matches"] if m.get("score", 0) > 0.2]

        if not relevant_matches:
            logger.warning("No sufficiently relevant matches found")
            answer = generate_answer("", question, has_sermon_content=False)
            return jsonify({"answer": f"Answer:\n{answer}"})

        context_chunks = []
        sources = []
        bible_verses = []
        seen_urls = set()

        for match in relevant_matches:
            metadata = match.get("metadata", {})
            doc_type = metadata.get("type", "sermon")

            if "text" in metadata:
                context_chunks.append(metadata["text"])

            if doc_type != "bible":
                url = metadata.get("url", "")
                if url and url not in seen_urls:
                    sources.append({
                        "title": metadata.get("title", "Sermon"),
                        "url": url,
                        "category": metadata.get("category", "General"),
                        "content": metadata.get("text", "")
                    })
                    seen_urls.add(url)

            if doc_type == "bible" and not bible_verses:
                bible_verses.append({
                    "reference": metadata.get("reference", ""),
                    "text": metadata.get("text", ""),
                    "book": metadata.get("book", "")
                })

        context = "\n\n---\n\n".join(context_chunks)
        logger.info(f"📚 Context built: {len(context)} chars from {len(context_chunks)} chunks")
        logger.info(f"📖 Found {len(sources)} sermon(s) and {len(bible_verses)} Bible verse(s)")

        answer = generate_answer(context, question, has_sermon_content=True)

        if not answer:
            logger.error("Empty answer generated")
            return jsonify({
                "answer": "Answer:\nI'm sorry, I couldn't generate a proper response. Please try again."
            })

        if "don't have any sermons" in answer.lower() or "don't have sermons" in answer.lower():
            return jsonify({"answer": f"Answer:\n{answer}"})

        final_answer = build_formatted_response(
            answer=answer,
            sources=sources,
            bible_verses=bible_verses
        )

        return jsonify({
            "answer": final_answer
        })

    except Exception as e:
        logger.error(f"❌ Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({
            "answer": "Answer:\nI apologize, but something went wrong on my end. Please try again in a moment."
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("🚀 Starting BibliBot server...")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)