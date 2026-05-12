"""Evaluation harness: action accuracy, topic coverage, refusal/clarification correctness."""

import json
import os
from pathlib import Path

from agent.graph import run_agent

_QUESTIONS_PATH = Path(__file__).parent / "questions.json"
_RESULTS_PATH = Path(__file__).parent / "results.json"


def _score_action(state: dict, expected_action: str) -> int:
    return 1 if state.get("action", "") == expected_action else 0


def _score_topic_coverage(state: dict, expected_topics: list[str]) -> float:
    if not expected_topics:
        return 1.0
    answer = (state.get("answer", "") + " " + state.get("refusal_message", "")).lower()
    hits = sum(1 for t in expected_topics if t.lower() in answer)
    return hits / len(expected_topics)


def _score_refusal(state: dict, should_refuse: bool) -> int:
    has_refusal = bool(state.get("refusal_message", "").strip())
    return 1 if has_refusal == should_refuse else 0


def _score_clarification(state: dict, should_clarify: bool) -> int:
    has_clarify = bool(state.get("clarification_question", "").strip())
    return 1 if has_clarify == should_clarify else 0


def run_evaluation(questions_path: str | Path = _QUESTIONS_PATH) -> dict:
    """Run all eval questions and compute scores.

    Args:
        questions_path: Path to questions.json.

    Returns:
        Dict mapping question id → score breakdown.
    """
    with open(questions_path, encoding="utf-8") as fh:
        questions = json.load(fh)

    results = {}
    rows = []

    for q in questions:
        state = run_agent(q["question"])
        action_score = _score_action(state, q["expected_action"])
        topic_score = _score_topic_coverage(state, q["expected_topics"])
        refusal_score = _score_refusal(state, q["should_refuse"])
        clarify_score = _score_clarification(state, q["should_clarify"])

        results[q["id"]] = {
            "question": q["question"][:60],
            "expected_action": q["expected_action"],
            "actual_action": state.get("action", "?"),
            "action_match": action_score,
            "topic_coverage": round(topic_score, 3),
            "refusal_correct": refusal_score,
            "clarify_correct": clarify_score,
        }
        rows.append((
            q["id"],
            q["expected_action"],
            state.get("action", "?"),
            action_score,
            round(topic_score, 3),
            refusal_score,
            clarify_score,
        ))

    # Print ASCII table
    header = f"{'ID':<6} {'Expected':<10} {'Actual':<10} {'Act':>3} {'Topic':>6} {'Ref':>4} {'Cla':>4}"
    print(header)
    print("-" * len(header))
    totals = [0, 0.0, 0, 0]
    for qid, exp, act, am, tc, rs, cs in rows:
        print(f"{qid:<6} {exp:<10} {act:<10} {am:>3} {tc:>6.3f} {rs:>4} {cs:>4}")
        totals[0] += am
        totals[1] += tc
        totals[2] += rs
        totals[3] += cs
    n = len(rows)
    print("-" * len(header))
    print(f"{'Average':<27} {totals[0]/n:>3.2f} {totals[1]/n:>6.3f} {totals[2]/n:>4.2f} {totals[3]/n:>4.2f}")

    with open(_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return results


if __name__ == "__main__":
    run_evaluation()
