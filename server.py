import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from sympy import re
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
    Returns: {"answer": "answer with sermon link and Bible verse citation"}
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
        
        if question.lower().strip() in greetings or len(question.split()) == 1:
            return jsonify({
                "answer": "Hello! I'm BibliBot, here to help you explore our sermons and the Bible. Ask me about faith, grace, prayer, love, hope, or any Biblical topic!"
            })
        
        # Step 1: Retrieve relevant context with metadata
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("sermon-index")
        
        from embeddings import embed
        vector = embed(question)
        
        # Query for top matches (mix of sermons and Bible verses)
        res = index.query(
            vector=vector,
            top_k=5,  # Get top 5 to include both sermons and verses
            include_metadata=True
        )
        
        # Check if results are actually relevant (score threshold)
        if not res or "matches" not in res or not res["matches"]:
            logger.warning("No documents retrieved")
            answer = generate_answer("", question, has_sermon_content=False)
            return jsonify({"answer": answer})
        
        # Filter by relevance score
        relevant_matches = [m for m in res["matches"] if m.get("score", 0) > 0.2]
        
        if not relevant_matches:
            logger.warning("No sufficiently relevant matches found")
            answer = generate_answer("", question, has_sermon_content=False)
            return jsonify({"answer": answer})
        
        # Separate sermon and Bible verse matches
        context_chunks = []
        sources = []
        bible_verses = []
        seen_urls = set()
        
        for match in relevant_matches:
            if "metadata" in match:
                metadata = match["metadata"]
                doc_type = metadata.get("type", "sermon")
                
                # Add text for context
                if "text" in metadata:
                    context_chunks.append(metadata["text"])
                
                # Handle sermons
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
                
                # Handle Bible verses
                if doc_type == "bible" and not bible_verses:  # Only take first verse match
                    bible_verses.append({
                        "reference": metadata.get("reference", ""),
                        "text": metadata.get("text", ""),
                        "book": metadata.get("book", "")
                    })
        
        context = "\n\n---\n\n".join(context_chunks)
        logger.info(f"📚 Context built: {len(context)} chars from {len(context_chunks)} chunks")
        logger.info(f"📖 Found {len(sources)} sermon(s) and {len(bible_verses)} Bible verse(s)")
        
        # Step 2: Generate answer (with sermon content)
        answer = generate_answer(context, question, has_sermon_content=True)
        
        if not answer:
            logger.error("Empty answer generated")
            return jsonify({
                "answer": "I'm sorry, I couldn't generate a proper response. Please try again."
            })
        
        # Check if we refused to answer (no sermon content)
        if "don't have any sermons" in answer.lower() or "don't have sermons" in answer.lower():
            return jsonify({"answer": answer})
        
        # Step 3: Add sermon link
        if sources:
            primary_source = sources[0]
            answer += f"\n\nHere's a sermon related to your question!\n\n📖 [{primary_source['title']}]({primary_source['url']})"
            
            # Look for original sermon links in content
            import re
            content = primary_source.get('content', '')
            original_links = re.findall(r'https?://[^\s]+', content)

            if original_links:
                for link in original_links:
                    if 'insightfulsermons.com' not in link:
                        answer += f"\n🎧 Full sermon: [{link}]({link})"
                        break
        else:
            logger.warning("⚠️ No specific sources found, but sermon content was used")
            answer += "\n\nHere's a sermon related to your question!\n\n📖 [Browse all sermons](https://www.insightfulsermons.com/)"
        
        # Step 4: Add Bible verse citation
        if bible_verses:
            verse = bible_verses[0]
            reference = verse.get('reference', '')
            verse_text = verse.get('text', '')
            
            # Format verse reference as link to Bible.com or similar
            # Format: "Book Chapter:Verse"
            bible_link = f"https://www.bible.com/search?q={reference.replace(' ', '%20')}"
            
            answer += f"\n\nA Bible verse addressing your question:\n\n\"{verse_text}\"\n\n— {reference}"
            
            logger.info(f"✅ Added Bible verse citation: {reference}")
        
        logger.info(f"✅ Answer generated successfully with links and Bible verse")
        
        # Step 5: Return response
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