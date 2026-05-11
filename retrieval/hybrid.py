"""Hybrid retrieval: RRF fusion of dense (ChromaDB) and sparse (BM25) results."""

import logging
from concurrent.futures import ThreadPoolExecutor

from retrieval.dense import dense_retrieve
from retrieval.sparse import sparse_retrieve

logger = logging.getLogger(__name__)


def hybrid_retrieve(query: str, top_k: int = 20, rrf_k: int = 60) -> list[dict]:
    """Fuse dense and sparse results using Reciprocal Rank Fusion.

    Args:
        query: Search query string.
        top_k: Number of results to return after fusion.
        rrf_k: RRF constant (default 60 per the standard formula).

    Returns:
        List of dicts with keys: chunk_id, text, metadata, score.
        Sorted by RRF score descending.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(dense_retrieve, query, top_k)
        sparse_future = executor.submit(sparse_retrieve, query, top_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # Build RRF score dict keyed by chunk_id
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunk_data[cid] = result

    for rank, result in enumerate(sparse_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        if cid not in chunk_data:
            chunk_data[cid] = result

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: list[dict] = []
    for cid, rrf_score in merged[:top_k]:
        entry = dict(chunk_data[cid])
        entry["score"] = rrf_score
        results.append(entry)

    return results
