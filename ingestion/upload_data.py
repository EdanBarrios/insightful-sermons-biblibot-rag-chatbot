"""
Upload existing sermon_data.json to Pinecone
No scraping needed - just embeds and uploads your existing data.

Usage:
    python ingestion/upload_existing_data.py
"""

import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import hashlib

sys.path.append(str(Path(__file__).parent.parent))

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

def generate_doc_id(title, url):
    """Generate stable document ID"""
    combined = f"{title}|{url}"
    return hashlib.md5(combined.encode()).hexdigest()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def upload_sermon_data(json_file):
    """Upload sermon data from JSON file to Pinecone"""
    logger.info(f"Loading sermon data from {json_file}...")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            sermon_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"❌ Invalid JSON in {json_file}")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(sermon_data)} sermons")
    
    vectors = []
    
    for title, sermon in sermon_data.items():
        try:
            content = sermon.get('content', '')
            url = sermon.get('url', '')
            category = sermon.get('category', 'General')
            
            if not content or len(content) < 50:
                logger.warning(f"⚠️ Skipping '{title}' - content too short")
                continue
            
            # Chunk the content
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{generate_doc_id(title, url)}_chunk_{i}"
                embedding = embedder.encode(chunk).tolist()
                
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
            
            logger.info(f"✅ Processed: {title[:50]}... ({len(chunks)} chunks)")
        
        except Exception as e:
            logger.warning(f"⚠️ Error processing '{title}': {e}")
            continue
    
    logger.info(f"📊 Created {len(vectors)} vectors from {len(sermon_data)} sermons")
    
    if not vectors:
        logger.error("❌ No vectors created!")
        sys.exit(1)
    
    # Upsert in batches
    logger.info("📤 Upserting to Pinecone...")
    batch_size = 100
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            progress = f"{min(i + batch_size, len(vectors))}/{len(vectors)}"
            logger.info(f"  Batch {i//batch_size + 1}: Upserted {progress} vectors")
        except Exception as e:
            logger.error(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
            continue
    
    # Final stats
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        logger.info(f"✅ Upsert complete! Total vectors in Pinecone: {total_vectors}")
    except:
        logger.info("✅ Upsert complete!")

def main():
    """Main upload pipeline"""
    logger.info("=" * 60)
    logger.info(f"Starting upload at {datetime.now()}")
    logger.info("=" * 60)
    
    # Try multiple possible locations for the file
    possible_paths = [
        'sermon_data.json',
        '../sermon_data.json',
        'data/sermon_data.json',
        '../data/sermon_data.json',
    ]
    
    json_file = None
    for path in possible_paths:
        if os.path.exists(path):
            json_file = path
            logger.info(f"Found sermon data at: {path}")
            break
    
    if not json_file:
        logger.error(f"❌ Could not find sermon_data.json in:")
        for path in possible_paths:
            logger.error(f"   - {path}")
        sys.exit(1)
    
    upload_sermon_data(json_file)
    
    logger.info("=" * 60)
    logger.info("✅ Upload pipeline complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()