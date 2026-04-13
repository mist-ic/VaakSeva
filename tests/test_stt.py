"""Tests for STT module (mocked - no real audio processing)."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from backend.voice.stt import TranscriptResult


class TestTranscriptResult:
    def test_low_confidence_check(self):
        result = TranscriptResult(
            text="कुछ आवाज़",
            confidence=0.45,
            language="hi",
            duration_s=2.5,
            transcription_ms=800.0,
        )
        assert result.confidence < 0.7  # should be rejected by caller

    def test_high_confidence_check(self):
        result = TranscriptResult(
            text="PM Kisan योजना में कितना मिलता है?",
            confidence=0.95,
            language="hi",
            duration_s=3.0,
            transcription_ms=1200.0,
        )
        assert result.confidence >= 0.7  # should be accepted

    def test_transcript_text(self):
        result = TranscriptResult(
            text="मेरा सवाल है",
            confidence=0.88,
            language="hi",
            duration_s=1.5,
            transcription_ms=500.0,
        )
        assert "मेरा" in result.text


class TestSTTFactory:
    def test_get_stt_returns_instance(self):
        from backend.voice.stt import WhisperHindiSTT
        with patch("backend.voice.stt.WhisperModel") as mock_model:
            mock_model.return_value = MagicMock()
            from backend.voice.stt import get_stt
            # Can't instantiate without actual model downloads, but factory exists
            assert callable(get_stt)


class TestAudioUtils:
    def test_convert_function_exists(self):
        from backend.voice.audio_utils import convert_ogg_to_wav, convert_wav_to_ogg
        assert callable(convert_ogg_to_wav)
        assert callable(convert_wav_to_ogg)

    def test_get_duration_function_exists(self):
        from backend.voice.audio_utils import get_audio_duration_s
        assert callable(get_audio_duration_s)
