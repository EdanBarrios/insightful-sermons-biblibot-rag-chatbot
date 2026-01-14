import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from retrieval import retrieve
from llm import generate_answer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("✅ Environment variables loaded")

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

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
    Returns: {"answer": "bot response"}
    """
    try:
        # Parse request
        data = request.get_json()
        if not data:
            logger.warning("Request missing JSON body")
            return jsonify({"error": "No JSON data provided"}), 400
        
        question = data.get("message", "").strip()
        if not question:
            logger.warning("Empty message received")
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f"📩 Received question: {question[:100]}...")
        
        # Step 1: Retrieve relevant context
        docs = retrieve(question, top_k=5)
        
        if not docs:
            logger.warning("No documents retrieved")
            return jsonify({
                "answer": "I couldn't find any relevant sermon content to answer your question. Could you try rephrasing or asking about a different topic?"
            })
        
        # Step 2: Build context string
        context = "\n\n---\n\n".join(docs)
        logger.info(f"📚 Context built: {len(context)} chars from {len(docs)} chunks")
        
        # Step 3: Generate answer
        answer = generate_answer(context, question)
        
        if not answer:
            logger.error("Empty answer generated")
            return jsonify({
                "answer": "I'm sorry, I couldn't generate a proper response. Please try again."
            })
        
        logger.info(f"✅ Answer generated successfully")
        
        # Step 4: Return response (frontend only needs 'answer')
        return jsonify({
            "answer": answer
        })
    
    except Exception as e:
        logger.error(f"❌ Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({
            "answer": "I apologize, but something went wrong on my end. Please try again in a moment."
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("🚀 Starting BibliBot server...")
    port = int(os.environ.get("PORT", "5000"))  # Render sets PORT
    app.run(host="0.0.0.0", port=port, debug=False)