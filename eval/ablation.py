"""Ablation study: dense-only vs hybrid-no-rerank vs full pipeline."""

import json
from pathlib import Path
from typing import Callable

from agent.graph import run_agent
from retrieval.dense import dense_retrieve
from retrieval.hybrid import hybrid_retrieve
from retrieval.rerank import rerank

_QUESTIONS_PATH = Path(__file__).parent / "questions.json"

# First 5 RETRIEVE questions
_RETRIEVE_IDS = {"q01", "q02", "q06", "q07", "q08"}


def _get_retrieve_questions(questions_path: Path = _QUESTIONS_PATH) -> list[dict]:
    with open(questions_path, encoding="utf-8") as fh:
        all_q = json.load(fh)
    return [q for q in all_q if q["id"] in _RETRIEVE_IDS]


def run_ablation(
    config: dict,
    questions: list[dict],
    retrieve_fn: Callable = hybrid_retrieve,
    rerank_fn: Callable | None = rerank,
) -> list[dict]:
    """Run a single ablation config over the given questions.

    Args:
        config: Dict with 'name' key and optional retrieval overrides.
        questions: List of question dicts.
        retrieve_fn: Retrieval function to inject.
        rerank_fn: Optional reranking function (None to skip).

    Returns:
        List of result dicts with scores.
    """
    results = []
    for q in questions:
        state = run_agent(q["question"])
        topic_score = 0.0
        if q["expected_topics"]:
            answer = state.get("answer", "").lower()
            hits = sum(1 for t in q["expected_topics"] if t.lower() in answer)
            topic_score = hits / len(q["expected_topics"])
        results.append({
            "config": config["name"],
            "id": q["id"],
            "topic_coverage": round(topic_score, 3),
            "action_match": 1 if state.get("action") == q["expected_action"] else 0,
        })
    return results


def _run_all_configs(questions_path: Path = _QUESTIONS_PATH) -> None:
    """Run all 3 ablation configs and print a comparison table."""
    questions = _get_retrieve_questions(questions_path)

    configs = [
        {"name": "dense_only"},
        {"name": "hybrid_no_rerank"},
        {"name": "full_pipeline"},
    ]

    import unittest.mock as mock

    all_results: dict[str, list[dict]] = {}

    for cfg in configs:
        if cfg["name"] == "dense_only":
            with mock.patch("agent.graph.hybrid_retrieve", side_effect=lambda q, top_k=20: dense_retrieve(q, top_k=top_k)):
                with mock.patch("agent.graph.rerank", side_effect=lambda q, c, top_n=5: sorted(c, key=lambda x: -x.get("score", 0))[:top_n]):
                    rows = run_ablation(cfg, questions)
        elif cfg["name"] == "hybrid_no_rerank":
            with mock.patch("agent.graph.rerank", side_effect=lambda q, c, top_n=5: sorted(c, key=lambda x: -x.get("score", 0))[:top_n]):
                rows = run_ablation(cfg, questions)
        else:
            rows = run_ablation(cfg, questions)
        all_results[cfg["name"]] = rows

    # Print comparison table
    q_ids = [q["id"] for q in questions]
    header = f"{'Config':<22}" + "".join(f" {qid:>6}" for qid in q_ids) + f" {'Avg':>6}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        rows = all_results[cfg["name"]]
        scores = [r["topic_coverage"] for r in rows]
        avg = sum(scores) / len(scores) if scores else 0.0
        row_str = f"{cfg['name']:<22}" + "".join(f" {s:>6.3f}" for s in scores) + f" {avg:>6.3f}"
        print(row_str)


if __name__ == "__main__":
    _run_all_configs()
