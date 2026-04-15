"""
Text-to-Speech module for VaakSeva.

Production backend: maya-research/Veena
  - 3B Llama-based autoregressive transformer + SNAC 24kHz codec
  - Apache 2.0 license
  - 4 named Hindi/English voices
  - MOS ~4.2/5 on Hindi
  - <80ms latency on H100, ~200ms on RTX 4090
  - Explicit Hindi + English + Hinglish code-mix support
  - 4-bit NF4 quantization via bitsandbytes

Development fallback: edge-tts
  - Microsoft Edge TTS API (free, no API key)
  - Natural Hindi voice (hi-IN-MadhurNeural)
  - No GPU required
  - ~200-500ms latency (network dependent)
  - NOT self-hosted — for dev iteration only

Both return a WAV file path. The caller converts to OGG for WhatsApp.
"""

from __future__ import annotations

import asyncio
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
        """Synthesise speech from text. Returns path to WAV file."""


# ---------------------------------------------------------------------------
# Veena TTS (production, GPU recommended)
# ---------------------------------------------------------------------------


class VeenaTTS(BaseTTS):
    """
    Wraps maya-research/Veena for Hindi TTS.

    Model: 3B parameters (Llama backbone + SNAC codec)
    Voices: Ira, Kiran, Meera, Priya (Hindi female voices)
    Quantization: NF4 4-bit for reduced VRAM usage
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        voice: str = "Meera",
    ):
        self._model_name = model_name or settings.veena_model
        self._device = device or settings.tts_device
        self._voice = voice
        self._model = None
        self._tokenizer = None
        logger.info("VeenaTTS configured: %s on %s (voice=%s)", self._model_name, self._device, voice)

    def _load_model(self):
        """Lazy model load."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ) if self._device == "cuda" else None

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            quantization_config=quant_config,
            device_map="auto" if self._device == "cuda" else "cpu",
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        logger.info("Veena model loaded")

    async def synthesise(self, text: str) -> Path:
        """Synthesise text to speech using Veena."""
        self._load_model()

        # Run inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        wav_path = await loop.run_in_executor(None, self._synthesise_sync, text)
        return wav_path

    def _synthesise_sync(self, text: str) -> Path:
        import soundfile as sf
        import torch

        prompt = f"<|im_start|>{self._voice}\n{text}<|im_end|>"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
            )

        # Decode audio tokens to waveform via SNAC codec
        # (Veena-specific post-processing)
        audio_tokens = output[0][inputs["input_ids"].shape[1]:]

        # Placeholder: actual Veena audio decoding depends on model release API
        # Reference: https://huggingface.co/maya-research/Veena
        raise NotImplementedError(
            "Veena inference decoding: check maya-research/Veena HuggingFace page for latest API. "
            "Use EDGE_TTS for development."
        )


# ---------------------------------------------------------------------------
# Edge TTS fallback (development)
# ---------------------------------------------------------------------------


class EdgeTTS(BaseTTS):
    """
    Microsoft Edge TTS — free, no API key, no GPU.

    Uses hi-IN-MadhurNeural (male) or hi-IN-SwaraNeural (female) by default.
    Network latency: 200-500ms typical.
    NOT self-hosted. For development only.
    """

    def __init__(self, voice: str | None = None):
        self._voice = voice or settings.edge_tts_voice
        logger.info("EdgeTTS configured (voice=%s) [dev fallback]", self._voice)

    async def synthesise(self, text: str) -> Path:
        """Synthesise text using Edge TTS. Returns path to MP3 file."""
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            out_path = Path(tmp.name)

        communicate = edge_tts.Communicate(text, self._voice)
        await communicate.save(str(out_path))

        logger.debug("EdgeTTS synthesised %d chars to %s", len(text), out_path.name)
        return out_path


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_tts() -> BaseTTS:
    """Return the configured TTS backend."""
    backend = settings.tts_backend
    if backend == "veena":
        return VeenaTTS()
    elif backend == "edge":
        return EdgeTTS()
    else:
        raise ValueError(f"Unknown TTS backend: {backend}")

# Edge TTS voice hi-IN-MadhurNeural is male, SwaraNeural is female
