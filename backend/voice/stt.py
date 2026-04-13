"""
Speech-to-Text module for VaakSeva.

Primary: collabora/faster-whisper-large-v2-hindi
  - 5.33% WER on Hindi FLEURS (best published open model as of 2025)
  - CTranslate2 format for 4x faster inference
  - CC-BY-4.0 license

Silero VAD preprocessing:
  - Detects speech segments in the audio
  - Trims silence at start/end
  - Handles noisy WhatsApp recordings

Confidence gating:
  - Returns language_probability from Whisper
  - Caller should reject if < settings.stt_language_threshold (default 0.7)
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    confidence: float       # language_probability from Whisper
    language: str           # detected language code
    duration_s: float       # audio duration in seconds
    transcription_ms: float


class WhisperHindiSTT:
    """
    Wraps faster-whisper for Hindi transcription.

    CTranslate2 backend gives ~4x speedup over PyTorch Whisper.
    INT8 quantization reduces VRAM from 2.9GB to 1.0GB with minimal WER impact.

    Model: collabora/faster-whisper-large-v2-hindi
    WER on Hindi FLEURS: 5.33% (trained on Shrutilipi + IndicVoices-R + Lahaja)
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        from faster_whisper import WhisperModel

        model_size = model_size or settings.stt_model
        device = device or settings.stt_device
        compute_type = compute_type or settings.stt_compute_type

        logger.info(
            "Loading Whisper model: %s on %s (%s)",
            model_size, device, compute_type,
        )

        # Use the collabora Hindi-tuned model
        # In production: download from HuggingFace and reference local path
        model_id = "collabora/faster-whisper-large-v2-hindi"

        self._model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper STT ready")

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        Transcribe a WAV audio file to Hindi text.

        Always transcribes as Hindi (language="hi") — the model is specifically
        trained on Hindi data. language_probability is still returned as a
        confidence signal for audio quality gating.
        """
        import time as _time

        t0 = _time.perf_counter()

        segments, info = self._model.transcribe(
            str(audio_path),
            language="hi",           # Force Hindi (model is Hindi-tuned)
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,        # Built-in VAD (Silero VAD)
            vad_parameters={
                "min_speech_duration_ms": 500,
                "max_speech_duration_s": 120,
                "min_silence_duration_ms": 1000,
            },
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts).strip()
        transcription_ms = (_time.perf_counter() - t0) * 1000

        logger.info(
            "Transcribed %.1fs audio in %.0fms: %r (lang_prob=%.2f)",
            info.duration,
            transcription_ms,
            full_text[:80],
            info.language_probability,
        )

        return TranscriptResult(
            text=full_text,
            confidence=info.language_probability,
            language=info.language,
            duration_s=info.duration,
            transcription_ms=transcription_ms,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_stt() -> WhisperHindiSTT:
    """Return the configured STT model."""
    return WhisperHindiSTT()
