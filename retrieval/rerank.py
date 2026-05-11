"""Cross-encoder reranking of hybrid retrieval candidates."""

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Rerank candidates using a cross-encoder model.

    Args:
        query: The original search query.
        candidates: List of candidate dicts (chunk_id, text, metadata, score).
        top_n: Number of top results to return.

    Returns:
        Top-n candidates sorted by cross-encoder score descending.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results: list[dict] = []
    for candidate, score in ranked[:top_n]:
        entry = dict(candidate)
        entry["score"] = float(score)
        results.append(entry)

    return results
