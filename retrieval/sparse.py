"""BM25 sparse retrieval over the arXiv corpus."""

import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
logger = logging.getLogger(__name__)

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
CORPUS_METADATA_PATH = os.getenv("CORPUS_METADATA_PATH", "./data/corpus_metadata.json")

_bm25: BM25Okapi | None = None
_chunk_ids: list[str] | None = None
_corpus_meta: dict | None = None


def _load_resources() -> tuple[BM25Okapi, list[str], dict]:
    global _bm25, _chunk_ids, _corpus_meta
    if _bm25 is None:
        if not Path(BM25_INDEX_PATH).exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_INDEX_PATH}. Run embed_and_store first."
            )
        import pickle
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25, _chunk_ids = pickle.load(f)
    if _corpus_meta is None:
        with open(CORPUS_METADATA_PATH) as f:
            _corpus_meta = json.load(f)
    return _bm25, _chunk_ids, _corpus_meta


def sparse_retrieve(query: str, top_k: int = 20) -> list[dict]:
    """Retrieve top-k chunks using BM25 keyword scoring.

    Args:
        query: Search query string.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: chunk_id, text, metadata, score.
        Sorted by BM25 score descending.
    """
    bm25, chunk_ids, corpus_meta = _load_resources()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    actual_top_k = min(top_k, len(chunk_ids))
    top_indices = np.argsort(scores)[-actual_top_k:][::-1]

    results: list[dict] = []
    for idx in top_indices:
        cid = chunk_ids[idx]
        meta = corpus_meta.get(cid, {})
        results.append({
            "chunk_id": cid,
            "text": meta.get("text", ""),
            "metadata": {
                "arxiv_id": meta.get("arxiv_id", ""),
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "page_start": meta.get("page_start", 1),
                "url": meta.get("url", ""),
            },
            "score": float(scores[idx]),
        })

    return results
