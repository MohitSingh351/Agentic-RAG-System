"""Episodic memory: JSONL log with difflib similarity search."""

import difflib
import json
import logging
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EPISODIC_LOG_PATH = os.getenv("EPISODIC_LOG_PATH", "./data/episodic_memory.jsonl")

# Stable session ID for this process
SESSION_ID = str(uuid.uuid4())


def _log_path() -> str:
    return os.getenv("EPISODIC_LOG_PATH", EPISODIC_LOG_PATH)


def log_episode(query: str, answer: str, sources: list[str]) -> None:
    """Append one episode to the JSONL log.

    Args:
        query: The user query.
        answer: The agent's final answer.
        sources: List of chunk_ids used to produce the answer.
    """
    entry = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "session_id": SESSION_ID,
    }
    path = Path(_log_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_all() -> list[dict]:
    """Load all valid episodes from the JSONL file."""
    path = Path(_log_path())
    if not path.exists():
        return []
    episodes: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line: %s", line[:80])
    return episodes


def find_similar(query: str, threshold: float = 0.8) -> list[dict]:
    """Return past episodes with query similarity >= threshold.

    Args:
        query: The current user query.
        threshold: Minimum SequenceMatcher ratio to qualify.

    Returns:
        List of matching episode dicts, sorted by similarity descending.
    """
    episodes = _load_all()
    if not episodes:
        return []

    q_lower = query.lower()
    scored: list[tuple[float, dict]] = []
    for ep in episodes:
        ratio = difflib.SequenceMatcher(None, q_lower, ep["query"].lower()).ratio()
        if ratio >= threshold:
            scored.append((ratio, ep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored]


def get_recent(n: int = 5) -> list[dict]:
    """Return the last n episodes from the log.

    Args:
        n: Number of recent episodes to return.

    Returns:
        List of the most recent n episode dicts (oldest first).
    """
    episodes = _load_all()
    return list(deque(episodes, maxlen=n))
