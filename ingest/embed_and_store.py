"""Embed chunks with Mistral, upsert to ChromaDB, and serialize BM25 index."""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
CORPUS_METADATA_PATH = os.getenv("CORPUS_METADATA_PATH", "./data/corpus_metadata.json")
COLLECTION_NAME = "arxiv_papers"
EMBED_BATCH_SIZE = 100
_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with Mistral mistral-embed."""
    response = client.embeddings.create(model="mistral-embed", input=texts)
    return [item.embedding for item in response.data]


def get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the ChromaDB collection for arXiv papers."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(chunks: list[dict[str, Any]]) -> None:
    """Embed all chunks and upsert to ChromaDB; also build and save BM25 index.

    Args:
        chunks: List of chunk dicts from chunk_all_papers().
    """
    if not chunks:
        logger.info("No chunks to embed.")
        return

    openai_client = OpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url=_MISTRAL_BASE_URL,
    )
    collection = get_chroma_collection()

    # Embed and upsert in batches
    for start in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="Embedding"):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = _embed_batch(openai_client, texts)

        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "arxiv_id": c["arxiv_id"],
                "title": c["title"],
                "section": c["section"],
                "page_start": c["page_start"],
                "url": c["url"],
            }
            for c in batch
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # Build BM25 index
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = [c["chunk_id"] for c in chunks]

    Path(BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((bm25, chunk_ids), f)

    # Save corpus metadata
    corpus_meta = {c["chunk_id"]: c for c in chunks}
    Path(CORPUS_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_METADATA_PATH, "w") as f:
        json.dump(corpus_meta, f)

    logger.info(
        "Stored %d chunks in ChromaDB, BM25 index at %s, metadata at %s",
        len(chunks),
        BM25_INDEX_PATH,
        CORPUS_METADATA_PATH,
    )


def load_bm25() -> tuple[BM25Okapi, list[str]]:
    """Load the BM25 index from disk.

    Returns:
        Tuple of (BM25Okapi, list of chunk_ids).

    Raises:
        FileNotFoundError: If the BM25 index file does not exist.
    """
    if not Path(BM25_INDEX_PATH).exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Run embed_and_store first.")
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Embed chunks and populate ChromaDB + BM25")
    parser.add_argument("--chunks-path", default="data/chunks.json")
    args = parser.parse_args()

    with open(args.chunks_path) as f:
        chunks_data = json.load(f)

    embed_and_store(chunks_data)
    print(f"Embedded and stored {len(chunks_data)} chunks")
