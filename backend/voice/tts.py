"""
Text-to-Speech module for VaakSeva.

Primary: Sarvam Bulbul v3 (hosted API)
  - Released February 2026
  - CER: 0.0173 on Sarvam TTS benchmark (very high intelligibility)
  - Latency: ~600ms for typical response lengths
  - Pricing: Rs 30/10,000 characters (~Rs 0.18 per 100-word Hindi response)
  - Hindi-native, multiple speakers (meera, arvind, amol, etc.)

Self-hosted fallback: Kokoro v1.0 Hindi (hexgrad/Kokoro-82M)
  - 82M parameters, Apache 2.0 license
  - Ranked #1 on TTS Arena leaderboard
  - Runs in seconds on CPU; correct pronunciation of Hindi vocabulary
  - Voices: hf_alpha (female), hf_omega (male)

Emergency fallback: Microsoft Edge TTS
  - Free, no API key, uses unofficial browser endpoint
  - Risk of rate-limiting or deprecation; do NOT use as primary

All backends return a WAV file path. Caller converts to OGG/Opus for WhatsApp.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseTTS(ABC):
    @abstractmethod
    async def synthesise(self, text: str) -> Path:
        """Synthesise speech from text. Returns path to WAV (or MP3) file."""


# ---------------------------------------------------------------------------
# Sarvam Bulbul v3 (primary — hosted API)
# ---------------------------------------------------------------------------


class SarvamTTS(BaseTTS):
    """
    Sarvam Bulbul v3 Hindi TTS via hosted API.

    Endpoint: POST https://api.sarvam.ai/text-to-speech
    Auth: api-subscription-key header
    Input: JSON body with target_language_code, text, speaker, model
    Output: {"audios": ["<base64-encoded-wav>"], "request_id": "..."}

    The base64-decoded bytes are a raw WAV file ready for ffmpeg conversion.
    """

    _TTS_PATH = "/text-to-speech"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        speaker: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.sarvam_api_key
        self._model = model or settings.sarvam_tts_model
        self._speaker = speaker or settings.sarvam_tts_speaker
        self._base_url = settings.sarvam_base_url.rstrip("/")

        if not self._api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. "
                "Sign up at https://docs.sarvam.ai for free API access."
            )

        logger.info(
            "SarvamTTS configured: model=%s speaker=%s",
            self._model, self._speaker,
        )

    async def synthesise(self, text: str) -> Path:
        """Synthesise Hindi text via Sarvam Bulbul v3. Returns WAV path."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesise_sync, text)

    def _synthesise_sync(self, text: str) -> Path:
        import httpx

        payload = {
            "target_language_code": "hi-IN",
            "text": text,
            "speaker": self._speaker,
            "model": self._model,
            "enable_preprocessing": True,
            "speech_sample_rate": 22050,
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{self._base_url}{self._TTS_PATH}",
                    headers={
                        "api-subscription-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Sarvam TTS HTTP error: %s — %s",
                exc.response.status_code, exc.response.text,
            )
            raise

        data = response.json()
        audios = data.get("audios", [])
        if not audios:
            raise RuntimeError("Sarvam TTS returned no audio data in response")

        audio_bytes = base64.b64decode(audios[0])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            wav_path = Path(tmp.name)

        logger.debug("SarvamTTS: %d chars -> %s", len(text), wav_path.name)
        return wav_path


# ---------------------------------------------------------------------------
# Kokoro TTS (self-hosted fallback)
# ---------------------------------------------------------------------------


class KokoroTTS(BaseTTS):
    """
    Kokoro v1.0 Hindi TTS (self-hosted, CPU-fast, Apache 2.0).

    82M parameter model, ranked #1 on TTS Arena leaderboard.
    Generates 5 minutes of audio in seconds on CPU.

    Hindi voices:
      hf_alpha - clear, natural female voice
      hf_omega - male voice

    Install: pip install kokoro soundfile
    """

    def __init__(self, voice: str | None = None) -> None:
        self._voice = voice or settings.kokoro_voice
        self._pipeline = None
        logger.info("KokoroTTS configured: voice=%s", self._voice)

    def _load_pipeline(self) -> None:
        """Lazy-load Kokoro pipeline on first use."""
        if self._pipeline is not None:
            return
        from kokoro import KPipeline
        # lang_code='h' for Hindi
        self._pipeline = KPipeline(lang_code="h")
        logger.info("KokoroTTS pipeline loaded (lang=hi)")

    async def synthesise(self, text: str) -> Path:
        """Synthesise Hindi text using Kokoro. Returns WAV path."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesise_sync, text)

    def _synthesise_sync(self, text: str) -> Path:
        import numpy as np
        import soundfile as sf

        self._load_pipeline()

        audio_chunks = []
        # Speed=0.9 gives slightly slower speech, better for comprehension
        for _, _, audio in self._pipeline(text, voice=self._voice, speed=0.9):
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("KokoroTTS produced no audio output")

        full_audio = np.concatenate(audio_chunks)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        sf.write(str(wav_path), full_audio, samplerate=24000)
        logger.debug("KokoroTTS: %d chars -> %s", len(text), wav_path.name)
        return wav_path


# ---------------------------------------------------------------------------
# Edge TTS (emergency fallback only)
# ---------------------------------------------------------------------------


class EdgeTTS(BaseTTS):
    """
    Microsoft Edge TTS (emergency fallback).

    Uses unofficial browser endpoint via edge-tts library.
    Free, no API key, no GPU required. ~200-500ms network latency.

    DO NOT use as primary in production — endpoint is unofficial and may
    be rate-limited or deprecated without notice. Kokoro is the preferred
    self-hosted fallback (faster, no network dependency, Apache 2.0).
    """

    def __init__(self, voice: str | None = None) -> None:
        self._voice = voice or settings.edge_tts_voice
        logger.info("EdgeTTS configured: voice=%s [emergency fallback]", self._voice)

    async def synthesise(self, text: str) -> Path:
        """Synthesise text using Edge TTS. Returns path to MP3 file."""
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            out_path = Path(tmp.name)

        communicate = edge_tts.Communicate(text, self._voice)
        await communicate.save(str(out_path))

        logger.debug("EdgeTTS: %d chars -> %s", len(text), out_path.name)
        return out_path


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_tts() -> BaseTTS:
    """Return the configured TTS backend."""
    backend = settings.tts_backend
    if backend == "sarvam":
        return SarvamTTS()
    elif backend == "kokoro":
        return KokoroTTS()
    elif backend == "edge":
        return EdgeTTS()
    else:
        raise ValueError(f"Unknown TTS backend: {backend!r}")
