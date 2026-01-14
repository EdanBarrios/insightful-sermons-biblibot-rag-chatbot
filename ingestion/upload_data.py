"""
Upload existing sermon data to Pinecone (no scraping needed).
Uses your existing sermon_data.json or documents_formatted.json files.

Usage:
    python ingestion/upload_existing_data.py
"""

import json
import os
import sys
import logging
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Initialize services
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("sermon-index")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ Services initialized")
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")
    sys.exit(1)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def generate_doc_id(title, url):
    """Generate stable document ID"""
    combined = f"{title}|{url}"
    return hashlib.md5(combined.encode()).hexdigest()

def load_sermon_data():
    """Try to load from multiple possible file locations"""
    possible_files = [
        'sermon_data.json',
        'documents_formatted.json',
        'data/sermon_data.json',
        'data/documents_formatted.json'
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            logger.info(f"📂 Found data file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, filepath
    
    logger.error("❌ No sermon data files found!")
    logger.error(f"Looked for: {possible_files}")
    return None, None

def process_sermon_data(data, filepath):
    """Convert sermon data to articles format"""
    articles = {}
    
    # Handle documents_formatted.json format
    if isinstance(data, list) and len(data) > 0 and 'page_content' in data[0]:
        logger.info("📋 Detected documents_formatted.json format")
        for i, doc in enumerate(data):
            title = f"Sermon {i+1}"
            articles[title] = {
                'content': doc['page_content'],
                'url': doc.get('metadata', {}).get('source', f'sermon_{i+1}'),
                'category': 'General'
            }
    
    # Handle sermon_data.json format
    elif isinstance(data, dict):
        logger.info("📋 Detected sermon_data.json format")
        articles = data
    
    else:
        logger.error("❌ Unknown data format")
        return None
    
    logger.info(f"✅ Loaded {len(articles)} sermons")
    return articles

def embed_and_upsert(articles):
    """Embed sermon chunks and upsert to Pinecone"""
    logger.info("🔄 Embedding and upserting...")
    
    vectors = []
    
    for title, article in articles.items():
        content = article.get('content', '')
        if not content:
            logger.warning(f"⚠️ Empty content for: {title}")
            continue
            
        url = article.get('url', 'unknown')
        category = article.get('category', 'General')
        
        # Chunk the content
        chunks = chunk_text(content)
        
        if not chunks:
            logger.warning(f"⚠️ No chunks created for: {title}")
            continue
        
        for i, chunk in enumerate(chunks):
            # Generate unique ID
            doc_id = f"{generate_doc_id(title, url)}_chunk_{i}"
            
            # Embed
            embedding = embedder.encode(chunk).tolist()
            
            # Prepare vector
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "title": title,
                    "url": url,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
    
    logger.info(f"✅ Created {len(vectors)} vectors from {len(articles)} sermons")
    
    if not vectors:
        logger.error("❌ No vectors created!")
        return False
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            logger.info(f"✅ Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"❌ Error upserting batch: {e}")
            return False
    
    logger.info("✅ All vectors upserted")
    return True

def main():
    logger.info("=" * 50)
    logger.info("Upload Existing Sermon Data to Pinecone")
    logger.info("=" * 50)
    
    # Load data
    data, filepath = load_sermon_data()
    if not data:
        logger.error("❌ No data found. Please ensure you have:")
        logger.error("   - sermon_data.json OR")
        logger.error("   - documents_formatted.json")
        logger.error("   in the project root or data/ folder")
        sys.exit(1)
    
    # Process data
    articles = process_sermon_data(data, filepath)
    if not articles:
        sys.exit(1)
    
    # Upload to Pinecone
    success = embed_and_upsert(articles)
    
    if success:
        # Verify
        stats = index.describe_index_stats()
        count = stats.get('total_vector_count', 0)
        
        logger.info("=" * 50)
        logger.info(f"✅ Upload complete!")
        logger.info(f"📊 Total vectors in Pinecone: {count}")
        logger.info("=" * 50)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run: python server.py")
        logger.info("  2. Open: http://localhost:5001")
        logger.info("  3. Ask: 'What is faith?'")
    else:
        logger.error("❌ Upload failed")
        sys.exit(1)

if __name__ == "__main__":
    main()