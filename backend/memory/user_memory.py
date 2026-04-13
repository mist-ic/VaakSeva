"""
Per-user conversation memory for VaakSeva.

Stores user profile and conversation history as JSON files on disk.
One file per user (phone number hashed for privacy).

Design choices:
  - JSON files (not a database) for simplicity and zero dependencies
  - Phone number is SHA-256 hashed with a salt before use as filename
  - TTL-based cleanup: users inactive for 30 days are purged
  - Thread-safe via file locking

Profile fields accumulated across conversation turns:
  age, gender, state, district, occupation, income, category, education,
  has_aadhaar, land_holding_acres, is_bpl
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

USER_TTL_DAYS = 30
MAX_HISTORY_TURNS = 20  # max turns stored per user


class UserMemory:
    """Per-user state management backed by JSON files."""

    def __init__(self, storage_dir: Path | None = None):
        self._dir = Path(storage_dir or settings.user_memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_phone(self, phone_number: str) -> str:
        """SHA-256 hash the phone number for privacy-preserving filenames."""
        raw = f"{settings.phone_hash_salt}:{phone_number}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _file_path(self, phone_number: str) -> Path:
        return self._dir / f"{self._hash_phone(phone_number)}.json"

    def _load(self, phone_number: str) -> dict:
        fp = self._file_path(phone_number)
        if not fp.exists():
            return {
                "phone_hash": self._hash_phone(phone_number),
                "profile": {},
                "conversation_history": [],
                "last_active": datetime.now(timezone.utc).isoformat(),
            }
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt user memory file %s: %s", fp.name, exc)
            return {"profile": {}, "conversation_history": [], "last_active": datetime.now(timezone.utc).isoformat()}

    def _save(self, phone_number: str, state: dict) -> None:
        fp = self._file_path(phone_number)
        fp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, phone_number: str) -> dict:
        """Return the accumulated user profile dict."""
        return self._load(phone_number).get("profile", {})

    def update_profile(self, phone_number: str, extracted: dict) -> None:
        """
        Merge new profile fields into the stored profile.

        Only non-None values in extracted are written.
        Existing values are NOT overwritten unless the new value differs.
        """
        if not extracted:
            return

        state = self._load(phone_number)
        profile = state.get("profile", {})

        for key, value in extracted.items():
            if value is not None:
                profile[key] = value

        state["profile"] = profile
        state["last_active"] = datetime.now(timezone.utc).isoformat()
        self._save(phone_number, state)
        logger.debug("Updated profile for %s: %s", self._hash_phone(phone_number)[:8], list(extracted.keys()))

    def get_conversation_history(self, phone_number: str, last_n: int = 6) -> list[dict]:
        """Return the last N conversation turns as list of {role, content, timestamp}."""
        history = self._load(phone_number).get("conversation_history", [])
        return history[-last_n:]

    def add_turn(self, phone_number: str, user_message: str, bot_response: str) -> None:
        """Append a user/bot exchange to the conversation history."""
        state = self._load(phone_number)
        history = state.get("conversation_history", [])

        now = datetime.now(timezone.utc).isoformat()
        history.append({"role": "user", "content": user_message, "timestamp": now})
        history.append({"role": "assistant", "content": bot_response, "timestamp": now})

        # Trim to max
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]

        state["conversation_history"] = history
        state["last_active"] = now
        self._save(phone_number, state)

    def clear_history(self, phone_number: str) -> None:
        """Clear conversation history (but keep profile)."""
        state = self._load(phone_number)
        state["conversation_history"] = []
        self._save(phone_number, state)

    def purge_inactive_users(self) -> int:
        """Delete user files not accessed in USER_TTL_DAYS. Returns count purged."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=USER_TTL_DAYS)
        purged = 0
        for fp in self._dir.glob("*.json"):
            try:
                state = json.loads(fp.read_text(encoding="utf-8"))
                last_active_str = state.get("last_active", "")
                if last_active_str:
                    last_active = datetime.fromisoformat(last_active_str)
                    if last_active.tzinfo is None:
                        last_active = last_active.replace(tzinfo=timezone.utc)
                    if last_active < cutoff:
                        fp.unlink()
                        purged += 1
            except Exception:
                pass
        logger.info("Purged %d inactive user memory files", purged)
        return purged
