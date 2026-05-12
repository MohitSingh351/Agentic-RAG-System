"""Structured JSONL tracer with per-node timing and debug-mode output."""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent.state import AgentState

logger = logging.getLogger(__name__)

_TRACES_DIR = os.getenv("TRACES_DIR", "./traces")


class Tracer:
    """Append-only trace log for a single agent session."""

    def __init__(self, session_id: str, query: str, debug: bool = False) -> None:
        self.session_id = session_id
        self.query = query
        self.debug = debug
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def log(self, node: str, data: dict) -> None:
        """Append a trace entry with timestamp and session metadata."""
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "node": node,
            "session_id": self.session_id,
            "query": self.query,
            **data,
        }
        with self._lock:
            self._entries.append(entry)

    def finalize(self, final_state: AgentState) -> None:
        """Append a FINAL summary entry and flush all entries to disk."""
        summary = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "node": "FINAL",
            "session_id": self.session_id,
            "query": self.query,
            "action": final_state.get("action", ""),
            "confidence": final_state.get("confidence", 0.0),
            "answer_length": len(final_state.get("answer", "")),
        }
        with self._lock:
            self._entries.append(summary)
            entries_snapshot = list(self._entries)

        traces_dir = Path(_TRACES_DIR)
        traces_dir.mkdir(parents=True, exist_ok=True)
        out_path = traces_dir / f"{self.session_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(entries_snapshot) + "\n")

    def print_trace(self) -> None:
        """Print each entry as formatted JSON when debug mode is on."""
        if not self.debug:
            return
        with self._lock:
            entries_snapshot = list(self._entries)
        for entry in entries_snapshot:
            print(json.dumps(entry, indent=2))
            print("---")

    @staticmethod
    def load_trace(session_id: str) -> list[dict]:
        """Load a previously saved trace file.

        Args:
            session_id: The session UUID used when finalize() was called.

        Returns:
            List of trace entry dicts.

        Raises:
            FileNotFoundError: If no trace file exists for this session_id.
        """
        traces_dir = Path(_TRACES_DIR)
        path = traces_dir / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No trace file found for session: {session_id}")
        with open(path, encoding="utf-8") as fh:
            return json.loads(fh.readline())
