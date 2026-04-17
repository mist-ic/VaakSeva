"""
Speech-to-Text module for VaakSeva.

Primary: Sarvam Saaras V3 (hosted API)
  - Ranked #1 on IndicVoices Hindi benchmark
  - Beats GPT-4o Transcribe, Deepgram Nova-3, ElevenLabs Scribe v2
  - Trained on 1M+ hours; optimized for Indian accents, code-mixed, noisy speech
  - Supports streaming for low-latency decoding
  - Pricing: Rs 30/hour (~Rs 0.08-0.25 per voice note)

Self-hosted fallback: faster-whisper large-v3
  - Upgraded from large-v2 (4+ WER points better on Hindi)
  - For best Hindi accuracy: use IndicWhisper weights from AI4Bharat
    model_size = "ai4bharat/IndicWhisper" (lowest WER on 39/59 Vistaar benchmarks)
  - CTranslate2 backend: ~4x speedup over PyTorch Whisper
  - INT8 quantization reduces memory from 2.9GB to ~1.0GB

Confidence gating:
  - Returns language_probability / confidence from the backend
  - Caller should reject if < settings.stt_language_threshold (default 0.7)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    confidence: float       # language_probability or API confidence score
    language: str           # detected/forced language code
    duration_s: float       # audio duration in seconds
    transcription_ms: float


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseStt(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Transcribe an audio file to text."""


# ---------------------------------------------------------------------------
# Sarvam Saaras V3 (primary — hosted API)
# ---------------------------------------------------------------------------


class SarvamSTT(BaseStt):
    """
    Sarvam Saaras V3 Hindi STT via hosted API.

    Endpoint: POST https://api.sarvam.ai/speech-to-text
    Auth: api-subscription-key header
    Input: multipart/form-data (file, language_code, model)
    Output: {"transcript": "...", "language_code": "hi-IN", "request_id": "..."}

    Accepts OGG/Opus (WhatsApp native), WAV, MP3, FLAC directly.
    No client-side audio conversion needed.
    """

    _STT_PATH = "/speech-to-text"
    _MIME_MAP = {
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ):
        self._api_key = api_key or settings.sarvam_api_key
        self._model = model or settings.sarvam_stt_model
        self._language = language or settings.sarvam_stt_language
        self._base_url = settings.sarvam_base_url.rstrip("/")

        if not self._api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. Sign up at https://docs.sarvam.ai "
                "to get a free API key."
            )

        logger.info(
            "SarvamSTT configured: model=%s language=%s",
            self._model, self._language,
        )

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        Transcribe audio using Sarvam Saaras V3.

        Sends audio as multipart/form-data. Returns TranscriptResult.
        Confidence is set to 1.0 since Saaras V3 does not expose
        per-request confidence scores in the current API.
        """
        import httpx

        t0 = time.perf_counter()

        audio_bytes = audio_path.read_bytes()
        content_type = self._MIME_MAP.get(audio_path.suffix.lower(), "audio/ogg")

        # Rough duration estimate (Saaras does not return duration in response)
        estimated_duration_s = max(1.0, len(audio_bytes) / 16000.0)

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{self._base_url}{self._STT_PATH}",
                    headers={"api-subscription-key": self._api_key},
                    files={
                        "file": (audio_path.name, audio_bytes, content_type),
                    },
                    data={
                        "language_code": self._language,
                        "model": self._model,
                        "with_disfluencies": "false",
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Sarvam STT HTTP error: %s — %s", exc.response.status_code, exc.response.text)
            raise
        except httpx.TimeoutException:
            logger.error("Sarvam STT request timed out after 30s")
            raise

        data = response.json()
        transcript = data.get("transcript", "").strip()
        transcription_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "SarvamSTT: %.0fms | %r",
            transcription_ms,
            transcript[:80],
        )

        return TranscriptResult(
            text=transcript,
            confidence=1.0,
            language=data.get("language_code", self._language),
            duration_s=estimated_duration_s,
            transcription_ms=transcription_ms,
        )


# ---------------------------------------------------------------------------
# Whisper STT (self-hosted fallback)
# ---------------------------------------------------------------------------


class WhisperHindiSTT(BaseStt):
    """
    faster-whisper Hindi transcription (self-hosted fallback).

    Default model: large-v3 (upgraded from large-v2).
    large-v3 improves Hindi WER by 4+ points over large-v2.

    For best open-source Hindi accuracy, use IndicWhisper from AI4Bharat:
      stt_model = "ai4bharat/IndicWhisper"  (in .env)
      Achieves lowest WER on 39/59 Vistaar Hindi benchmarks.

    CTranslate2 backend gives ~4x speedup over PyTorch Whisper.
    INT8 quantization reduces VRAM from 2.9GB to ~1.0GB.
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        from faster_whisper import WhisperModel

        model_size = model_size or settings.stt_model      # default: large-v3
        device = device or settings.stt_device
        compute_type = compute_type or settings.stt_compute_type

        logger.info(
            "Loading Whisper model: %s on %s (%s)",
            model_size, device, compute_type,
        )

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper STT ready (model=%s)", model_size)

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        Transcribe audio to Hindi text using faster-whisper.

        Forces language=hi. Silero VAD filters silence.
        language_probability is used as confidence for quality gating.
        """
        t0 = time.perf_counter()

        segments, info = self._model.transcribe(
            str(audio_path),
            language="hi",
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": 500,
                "max_speech_duration_s": 120,
                "min_silence_duration_ms": 1000,
            },
        )

        text_parts = [segment.text.strip() for segment in segments]
        full_text = " ".join(text_parts).strip()
        transcription_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Whisper: %.1fs audio | %.0fms | lang_prob=%.2f | %r",
            info.duration,
            transcription_ms,
            info.language_probability,
            full_text[:80],
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


def get_stt() -> BaseStt:
    """Return the configured STT backend."""
    backend = settings.stt_backend
    if backend == "sarvam":
        return SarvamSTT()
    elif backend == "whisper":
        return WhisperHindiSTT()
    else:
        raise ValueError(f"Unknown STT backend: {backend!r}")
