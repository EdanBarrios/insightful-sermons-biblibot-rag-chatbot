import os
import logging

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone

from app.memory import init_db, save_turn, get_recent_messages
from app.embeddings import embed
from app.catalog import load_catalog, content_words, tokenize
from app.search import find_sermons, previously_shown

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

catalog = load_catalog()

init_db()

# -------------------- Constants --------------------

_GREETINGS = frozenset([
    "hi", "hello", "hey", "yo", "sup",
    "greetings", "good morning", "good afternoon", "good evening",
])

# "any more?" carries no subject of its own — it means more of whatever was
# asked for last.
_FOLLOWUP_WORDS = frozenset([
    "anymore", "more", "others", "other", "else", "another", "next", "again",
])

_MAX_SERMONS = 3

# Enough turns to page through a prolific author a few results at a time.
_HISTORY_LIMIT = 12

_CLOSING_LINE = "If you'd like to search for other sermons, feel free to ask."

_NO_MATCH_MESSAGE = (
    "I don't have sermons matching that search. "
    "If you'd like to search for something else, feel free to ask."
)

_GREETING_MESSAGE = (
    "Hello! I'm BibliBot. I can find sermon summaries by title, subject, or "
    "author. What are you looking for?"
)

# -------------------- Helpers --------------------

def _is_followup(question: str) -> bool:
    # Tokenize rather than split: "any more?" has to count, punctuation and all.
    words = tokenize(question)
    return bool(set(words) & _FOLLOWUP_WORDS) and len(words) <= 5


def build_search_query(question: str, history: list) -> str:
    """
    Resolve a bare follow-up against what was asked before.

    "any more?" after "Jonathan Edwards" has to search for Jonathan Edwards, and
    the turn before that may itself have been "more", so walk back to the last
    message that actually named a subject.
    """
    if not history or not _is_followup(question):
        return question

    for message in reversed(history):
        if message["role"] != "user":
            continue
        previous = message["content"]
        if not _is_followup(previous) and content_words(previous):
            logger.info(f"Follow-up resolved against earlier question: {previous!r}")
            return previous

    return question


def _sermon_block(sermon: dict) -> str:
    lines = [sermon["title"]]
    authors = sermon.get("authors") or []
    if authors:
        lines.append(", ".join(authors))
    lines.append(f'[Read the sermon]({sermon["url"]})')
    return "\n".join(lines)


def _subject(found: dict) -> str:
    if found["kind"] == "author":
        return f"by {found['label']}"
    if found["kind"] == "category":
        return f"in the {found['label']} category"
    return "on that topic"


def _intro(found: dict, is_followup: bool) -> str:
    """
    Say how much more there is.

    Peter had to coax the bot into showing a second Jonathan Edwards sermon
    because nothing ever hinted there were fourteen of them.
    """
    count = len(found["results"])
    total = found["total"]
    subject = _subject(found)

    lead = "Here is" if count == 1 else "Here are"
    more = "more " if is_followup else ""

    if total > count:
        return f"{lead} {count} {more}of the {total} sermons {subject}:"
    if count == 1:
        return f"Here is {'one more ' if is_followup else 'a '}sermon {subject}:"
    return f"{lead} {count} {more}sermons {subject}:"


def build_formatted_response(found: dict, is_followup: bool) -> str:
    """Sermon discovery output: title, speaker, link. No generated prose."""
    if found["exhausted"]:
        return (
            f"That's all {found['total']} sermons I have {_subject(found)}. "
            "If you'd like to search for something else, feel free to ask."
        )

    if not found["results"]:
        return _NO_MATCH_MESSAGE

    closing = (
        "Ask for more to see the rest."
        if found["remaining"] > len(found["results"])
        else _CLOSING_LINE
    )
    blocks = [_sermon_block(s) for s in found["results"]]
    return "\n\n".join([_intro(found, is_followup), *blocks, closing])


# -------------------- Routes --------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "sermons": len(catalog.sermons)})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("message", "").strip()
        session_id = data.get("session_id", "").strip() or "anonymous"

        if not question:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Question: {question}")

        if question.lower() in _GREETINGS:
            save_turn(session_id, question, _GREETING_MESSAGE)
            return jsonify({"answer": _GREETING_MESSAGE})

        history = get_recent_messages(session_id, limit=_HISTORY_LIMIT)
        is_followup = _is_followup(question)
        search_query = build_search_query(question, history)

        # Only skip past what has already been shown when the person is asking
        # for more of the same. A fresh search should start from the best match
        # even if it was mentioned earlier.
        exclude = previously_shown(history) if is_followup else set()

        found = find_sermons(
            catalog, index, embed, search_query, exclude, _MAX_SERMONS
        )
        logger.info(f"Returning {len(found['results'])} sermon(s) of {found['total']}")

        answer = build_formatted_response(found, is_followup)
        save_turn(session_id, question, answer)

        return jsonify({"answer": answer})

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
