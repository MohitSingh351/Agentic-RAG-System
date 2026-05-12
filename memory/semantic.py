"""Semantic memory: persist extracted user facts in SQLite."""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SEMANTIC_DB_PATH = os.getenv("SEMANTIC_DB_PATH", "./data/semantic_memory.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS semantic_memory (
    key       TEXT PRIMARY KEY,
    value     TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    timestamp TEXT NOT NULL
)
"""

_EXTRACT_PROMPT = """You are extracting persistent facts about a user from a conversation message.

Given the user message below, extract any facts about the user's research interests,
expertise level, preferred topics, or domain knowledge.

Return a JSON array of objects, each with "key", "value", and "confidence" (0.0–1.0).
Return an empty array [] if no facts are present.

Examples of good keys: "research_interest", "expertise_level", "preferred_language", "topic_focus"

User message: {message}

Return only valid JSON, nothing else."""


def _db_path() -> str:
    return os.getenv("SEMANTIC_DB_PATH", SEMANTIC_DB_PATH)


def _get_conn() -> sqlite3.Connection:
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def upsert_fact(key: str, value: str, confidence: float) -> None:
    """Insert or update a fact in semantic memory.

    Args:
        key: Normalized fact key (e.g., 'research_interest').
        value: The fact value string.
        confidence: Confidence score in [0.0, 1.0].
    """
    ts = datetime.now(tz=timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO semantic_memory (key, value, confidence, timestamp) VALUES (?, ?, ?, ?)",
            (key, value, confidence, ts),
        )


def get_fact(key: str) -> dict | None:
    """Retrieve a single fact by key.

    Args:
        key: The fact key to look up.

    Returns:
        Dict with key/value/confidence/timestamp, or None if not found.
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT key, value, confidence, timestamp FROM semantic_memory WHERE key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return None
    return {"key": row[0], "value": row[1], "confidence": row[2], "timestamp": row[3]}


def get_all_facts() -> list[dict]:
    """Return all stored facts as a list of dicts."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT key, value, confidence, timestamp FROM semantic_memory"
        ).fetchall()
    return [{"key": r[0], "value": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]


def clear() -> None:
    """Delete all facts from semantic memory."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM semantic_memory")


def extract_and_store(user_message: str, llm_client) -> None:
    """Use an LLM to extract facts from a user message and store them.

    Args:
        user_message: The latest user message.
        llm_client: An OpenAI-compatible client (Mistral) with a .chat.completions.create() method.
    """
    if not user_message.strip():
        return

    prompt = _EXTRACT_PROMPT.format(message=user_message)
    try:
        response = llm_client.chat.completions.create(
            model="mistral-small-latest",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        return

    try:
        facts = json.loads(raw)
        if not isinstance(facts, list):
            logger.warning("LLM returned non-list: %s", raw[:100])
            return
    except json.JSONDecodeError:
        logger.warning("Malformed JSON from LLM: %s", raw[:100])
        return

    for fact in facts:
        if isinstance(fact, dict) and "key" in fact and "value" in fact:
            upsert_fact(
                key=str(fact["key"]),
                value=str(fact["value"]),
                confidence=float(fact.get("confidence", 1.0)),
            )
