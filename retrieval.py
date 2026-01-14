from pinecone import Pinecone
from embeddings import embed
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("sermon-index")
    logger.info("✅ Pinecone initialized successfully")
except Exception as e:
    logger.error(f"❌ Pinecone initialization failed: {e}")
    raise

def retrieve(query: str, top_k: int = 5) -> list[str]:
    """
    Retrieve relevant sermon chunks from Pinecone.
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        
    Returns:
        List of text chunks (empty list on error)
    """
    try:
        # Embed the query
        vector = embed(query)
        logger.info(f"Query embedded: {query[:50]}...")
        
        # Query Pinecone
        res = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract text from matches
        if not res or "matches" not in res:
            logger.warning("No matches returned from Pinecone")
            return []
        
        matches = res["matches"]
        if not matches:
            logger.warning("Empty matches list")
            return []
        
        # Extract text from metadata
        chunks = []
        for match in matches:
            if "metadata" in match and "text" in match["metadata"]:
                chunks.append(match["metadata"]["text"])
            else:
                logger.warning(f"Match missing metadata/text: {match.get('id', 'unknown')}")
        
        logger.info(f"✅ Retrieved {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        return []