"""Sliding-window conversational memory for the last N turns."""

import logging
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ConversationalMemory:
    """Rolling window of the last `max_turns` conversation turns."""

    _VALID_ROLES = {"user", "assistant"}

    def __init__(self, max_turns: int = 6) -> None:
        self._turns: deque[dict] = deque(maxlen=max_turns)

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn to the conversation history.

        Args:
            role: Must be 'user' or 'assistant'.
            content: The message text.

        Raises:
            ValueError: If role is not 'user' or 'assistant'.
        """
        if role not in self._VALID_ROLES:
            raise ValueError(f"role must be one of {self._VALID_ROLES}, got {role!r}")
        self._turns.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    def get_context(self) -> list[dict]:
        """Return the current conversation window as a list of turn dicts."""
        return list(self._turns)

    def clear(self) -> None:
        """Clear all stored turns."""
        self._turns.clear()

    def get_formatted_context(self) -> str:
        """Return a plain-text representation of the conversation history.

        Returns:
            Multi-line string like 'User: ...\\nAssistant: ...\\n', or '' if empty.
        """
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines) + "\n"
