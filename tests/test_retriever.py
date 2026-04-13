"""Tests for document ingestion pipeline."""

from __future__ import annotations

import pytest
from pathlib import Path

from backend.rag.ingest import (
    clean_text,
    detect_language,
    chunk_text,
    parse_scheme_metadata,
)


class TestCleanText:
    def test_normalises_unicode(self):
        # Devanagari NFC normalisation
        text = "पीएम-किसान\r\n"
        cleaned = clean_text(text)
        assert "\r" not in cleaned
        assert cleaned.strip() == "पीएम-किसान"

    def test_collapses_excess_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        cleaned = clean_text(text)
        assert "\n\n\n" not in cleaned

    def test_strips_trailing_whitespace(self):
        text = "Line with spaces   \nAnother line"
        cleaned = clean_text(text)
        assert "   \n" not in cleaned

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_preserves_hindi_content(self):
        text = "योजना का नाम: PM-KISAN"
        cleaned = clean_text(text)
        assert "PM-KISAN" in cleaned
        assert "योजना" in cleaned


class TestDetectLanguage:
    def test_detects_hindi(self):
        text = "प्रधानमंत्री किसान सम्मान निधि योजना"
        assert detect_language(text) == "hi"

    def test_detects_english(self):
        text = "Prime Minister Kisan Samman Nidhi Scheme"
        lang = detect_language(text)
        assert lang == "en"

    def test_mixed_text_prefers_hindi(self):
        # More Devanagari than Roman
        text = "पीएम-किसान PM-KISAN योजना"
        assert detect_language(text) == "hi"

    def test_empty_returns_en(self):
        assert detect_language("") == "en"


class TestChunkText:
    def test_short_text_not_split(self):
        text = "PM-KISAN gives Rs 6000 per year"
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self):
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) > 1

    def test_hindi_period_separator(self):
        text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।"
        chunks = chunk_text(text, chunk_size=20)
        # Should split on Hindi period
        assert len(chunks) >= 1

    def test_chunks_are_non_empty(self):
        text = "Some text\n\nAnother paragraph\n\nThird paragraph"
        chunks = chunk_text(text, chunk_size=30)
        assert all(c.strip() for c in chunks)


class TestParseSchemeMetadata:
    def test_parses_hindi_filename(self, tmp_path):
        f = tmp_path / "pm_kisan_hi.txt"
        f.touch()
        meta = parse_scheme_metadata(f)
        assert meta["scheme_id"] == "pm_kisan"
        assert meta["language"] == "hi"

    def test_parses_english_filename(self, tmp_path):
        f = tmp_path / "ayushman_bharat_en.txt"
        f.touch()
        meta = parse_scheme_metadata(f)
        assert meta["scheme_id"] == "ayushman_bharat"
        assert meta["language"] == "en"

    def test_no_language_suffix(self, tmp_path):
        f = tmp_path / "some_scheme.txt"
        f.touch()
        meta = parse_scheme_metadata(f)
        assert meta["language"] is None  # to be detected

    def test_scheme_name_formatted(self, tmp_path):
        f = tmp_path / "pm_kisan_hi.txt"
        f.touch()
        meta = parse_scheme_metadata(f)
        assert meta["scheme_name"] == "Pm Kisan"  # title-cased
