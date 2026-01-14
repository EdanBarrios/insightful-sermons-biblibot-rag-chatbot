from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(text: str) -> list[float]:
    return embedder.encode(text).tolist()