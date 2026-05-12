"""Dense vector retrieval via ChromaDB and Mistral embeddings."""

import logging
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "arxiv_papers"
_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

_collection: chromadb.Collection | None = None
_mistral_client: OpenAI | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_mistral() -> OpenAI:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = OpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url=_MISTRAL_BASE_URL,
        )
    return _mistral_client


def dense_retrieve(query: str, top_k: int = 20) -> list[dict]:
    """Embed the query and return top-k chunks from ChromaDB.

    Args:
        query: Search query string.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: chunk_id, text, metadata, score.
        Sorted by score descending (higher = more similar).
    """
    client = _get_mistral()
    embedding = client.embeddings.create(
        model="mistral-embed", input=[query]
    ).data[0].embedding

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)
    response = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    ids = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        score = max(0.0, min(1.0, 1.0 - distance))
        results.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
