import os
import logging
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
    Returns: {"answer": "brief answer with sermon links"}
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
        
        # Handle greetings BEFORE retrieval
        greetings = ['hi', 'hello', 'hey', 'yo', 'sup', 'howdy', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if question.lower().strip() in greetings or len(question.split()) <= 2:
            return jsonify({
                "answer": "Hello! I'm BibliBot, here to help you explore our sermons. Ask me about faith, grace, prayer, love, hope, or any Biblical topic!"
            })
        
        # Step 1: Retrieve relevant context with metadata
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("sermon-index")
        
        from embeddings import embed
        vector = embed(question)
        
        res = index.query(
            vector=vector,
            top_k=3,  # Get top 3 most relevant
            include_metadata=True
        )
        
        # Check if results are actually relevant (score threshold)
        if not res or "matches" not in res or not res["matches"]:
            logger.warning("No documents retrieved")
            # Still answer, but flag that we have no sermon content
            answer = generate_answer("", question, has_sermon_content=False)
            return jsonify({"answer": answer})
        
        # Filter by relevance score (only keep high-confidence matches)
        relevant_matches = [m for m in res["matches"] if m.get("score", 0) > 0.25]
        
        if not relevant_matches:
            logger.warning("No sufficiently relevant matches found")
            # Still answer, but flag that we have no relevant sermon content
            answer = generate_answer("", question, has_sermon_content=False)
            return jsonify({"answer": answer})
        
        # Extract both text and metadata
        context_chunks = []
        sources = []
        seen_urls = set()
        
        for match in relevant_matches:  # Use filtered matches
            if "metadata" in match:
                metadata = match["metadata"]
                
                # Add text for context
                if "text" in metadata:
                    context_chunks.append(metadata["text"])
                
                # Add unique sources
                url = metadata.get("url", "")
                if url and url not in seen_urls:
                    sources.append({
                        "title": metadata.get("title", "Sermon"),
                        "url": url,
                        "category": metadata.get("category", "General")
                    })
                    seen_urls.add(url)
        
        context = "\n\n---\n\n".join(context_chunks)
        logger.info(f"📚 Context built: {len(context)} chars from {len(context_chunks)} chunks")
        
        # Step 2: Generate answer (with sermon content)
        answer = generate_answer(context, question, has_sermon_content=True)
        
        if not answer:
            logger.error("Empty answer generated")
            return jsonify({
                "answer": "I'm sorry, I couldn't generate a proper response. Please try again."
            })
        
        # Check if this was a "no sermon content" answer
        no_sermon_indicators = [
            "don't have specific sermons",
            "don't have sermons",
            "no sermons about",
            "sermons don't cover"
        ]
        
        has_sermons = not any(indicator in answer.lower() for indicator in no_sermon_indicators)
        
        # Step 3: Add sermon links ONLY if we have relevant sermons
        if has_sermons and sources:
            answer += "\n\n📖 **Learn more from these sermons:**"
            for source in sources[:3]:  # Max 3 links
                answer += f"\n• [{source['title']}]({source['url']})"
        
        logger.info(f"✅ Answer generated successfully with {len(sources)} sources")
        
        # Step 4: Return response
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