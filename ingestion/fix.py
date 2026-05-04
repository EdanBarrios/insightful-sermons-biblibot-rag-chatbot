"""
Upload parsed Bible verses to Pinecone.
Embeds each verse group and stores with minimal metadata to stay under size limit.
"""

import json
import os
import sys
import logging
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

def upload_bible_to_pinecone(json_file):
    """Upload Bible verses from JSON file to Pinecone"""
    logger.info(f"Loading Bible data from {json_file}...")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            bible_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"❌ Invalid JSON in {json_file}")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(bible_data)} verse groups")
    
    vectors = []
    
    for i, verse_group in enumerate(bible_data):
        try:
            text = verse_group.get('text', '')
            reference = verse_group.get('reference', '')
            book = verse_group.get('book', '')
            
            if not text or len(text) < 20:
                logger.warning(f"⚠️ Skipping verse group {i} - text too short")
                continue
            
            # Embed the verse text
            embedding = embedder.encode(text).tolist()
            
            # Create unique ID
            doc_id = f"bible_{reference.replace(' ', '_').replace(':', '_').lower()}"
            
            # IMPORTANT: Keep metadata minimal to stay under 40KB Pinecone limit
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "text": text[:500],  # Truncate to 500 chars
                    "reference": reference,
                    "book": book,
                    "type": "bible"
                }
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed: {i + 1}/{len(bible_data)} verse groups")
        
        except Exception as e:
            logger.warning(f"⚠️ Error processing verse group {i}: {e}")
            continue
    
    logger.info(f"📊 Created {len(vectors)} vectors from {len(bible_data)} verse groups")
    
    if not vectors:
        logger.error("❌ No vectors created!")
        sys.exit(1)
    
    # Upsert in batches
    logger.info("📤 Upserting to Pinecone...")
    batch_size = 100
    successful = 0
    failed = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            progress = f"{min(i + batch_size, len(vectors))}/{len(vectors)}"
            logger.info(f"  Batch {i//batch_size + 1}: Upserted {progress} vectors")
            successful += len(batch)
        except Exception as e:
            logger.error(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
            failed += len(batch)
            continue
    
    # Final stats
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        logger.info(f"✅ Upsert complete!")
        logger.info(f"   Successful: {successful}, Failed: {failed}")
        logger.info(f"   Total vectors in Pinecone: {total_vectors}")
    except:
        logger.info(f"✅ Upsert complete! Successful: {successful}, Failed: {failed}")

def main():
    """Main upload pipeline"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Bible upload to Pinecone")
    logger.info("=" * 60)
    
    # Try multiple possible locations for the file
    possible_paths = [
        'data/NLT_Bible/bible_for_embedding.json',
        'bible_for_embedding.json',
        'data/NLT Bible/bible_for_embedding.json',
        '../NLT Bible/bible_for_embedding.json',
    ]
    
    json_file = None
    for path in possible_paths:
        if os.path.exists(path):
            json_file = path
            logger.info(f"Found Bible data at: {path}")
            break
    
    if not json_file:
        logger.error(f"❌ Could not find bible_for_embedding.json in:")
        for path in possible_paths:
            logger.error(f"   - {path}")
        sys.exit(1)
    
    upload_bible_to_pinecone(json_file)
    
    logger.info("=" * 60)
    logger.info("✅ Bible upload pipeline complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()