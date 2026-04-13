"""Tests for user memory module."""

from __future__ import annotations

import json
import time
import pytest
from pathlib import Path

from backend.memory.user_memory import UserMemory


PHONE = "+919876543210"


class TestUserMemory:
    def setup_method(self, temp_dir=None):
        import tempfile
        self.storage_dir = Path(tempfile.mkdtemp())
        self.memory = UserMemory(self.storage_dir)

    def test_empty_profile_for_new_user(self):
        profile = self.memory.get_profile(PHONE)
        assert profile == {}

    def test_update_profile(self):
        self.memory.update_profile(PHONE, {"age": 25, "state": "Karnataka"})
        profile = self.memory.get_profile(PHONE)
        assert profile["age"] == 25
        assert profile["state"] == "Karnataka"

    def test_profile_accumulates(self):
        self.memory.update_profile(PHONE, {"age": 25})
        self.memory.update_profile(PHONE, {"state": "Karnataka"})
        profile = self.memory.get_profile(PHONE)
        assert "age" in profile
        assert "state" in profile

    def test_profile_persists_across_instances(self):
        self.memory.update_profile(PHONE, {"occupation": "farmer"})
        new_memory = UserMemory(self.storage_dir)
        profile = new_memory.get_profile(PHONE)
        assert profile.get("occupation") == "farmer"

    def test_add_conversation_turn(self):
        self.memory.add_turn(PHONE, "PM Kisan क्या है?", "PM-KISAN किसानों को ₹6000 देता है।")
        history = self.memory.get_conversation_history(PHONE)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_limit(self):
        for i in range(15):
            self.memory.add_turn(PHONE, f"सवाल {i}", f"जवाब {i}")
        history = self.memory.get_conversation_history(PHONE, last_n=6)
        assert len(history) <= 6

    def test_clear_history(self):
        self.memory.add_turn(PHONE, "question", "answer")
        self.memory.clear_history(PHONE)
        history = self.memory.get_conversation_history(PHONE)
        assert history == []

    def test_clear_history_keeps_profile(self):
        self.memory.update_profile(PHONE, {"state": "Delhi"})
        self.memory.add_turn(PHONE, "q", "a")
        self.memory.clear_history(PHONE)
        profile = self.memory.get_profile(PHONE)
        assert profile.get("state") == "Delhi"

    def test_different_users_isolated(self):
        phone2 = "+918765432109"
        self.memory.update_profile(PHONE, {"state": "UP"})
        self.memory.update_profile(phone2, {"state": "Bihar"})
        assert self.memory.get_profile(PHONE)["state"] == "UP"
        assert self.memory.get_profile(phone2)["state"] == "Bihar"

    def test_phone_hash_privacy(self):
        self.memory.update_profile(PHONE, {"age": 30})
        # Check that no plaintext phone number is stored in the file
        files = list(self.storage_dir.glob("*.json"))
        assert len(files) == 1
        filename = files[0].stem
        assert PHONE not in filename
        assert "9876543210" not in filename

    def test_none_values_not_stored(self):
        self.memory.update_profile(PHONE, {"age": 25, "income": None})
        profile = self.memory.get_profile(PHONE)
        assert profile["age"] == 25
        assert "income" not in profile
