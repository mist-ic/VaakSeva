"""
Audio utilities for VaakSeva.

Handles:
  - OGG/Opus to WAV conversion (for Whisper input)
  - WAV/MP3 to OGG/Opus conversion (for WhatsApp voice note output)
  - Audio normalisation
  - Silence trimming
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_ogg_to_wav(ogg_path: Path) -> Path:
    """
    Convert OGG/Opus audio (WhatsApp voice note format) to WAV for Whisper.

    Uses ffmpeg for conversion. ffmpeg must be installed on the system.
    Returns path to temporary WAV file.
    """
    wav_path = Path(tempfile.mktemp(suffix=".wav"))

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(ogg_path),
                "-ar", "16000",        # 16kHz sample rate (Whisper optimal)
                "-ac", "1",            # mono
                "-c:a", "pcm_s16le",   # PCM 16-bit little-endian
                "-y",                  # overwrite without prompt
                str(wav_path),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg conversion failed: %s", exc.stderr.decode(errors="replace"))
        raise RuntimeError(f"Audio conversion failed: {exc}") from exc
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: sudo apt install ffmpeg (Linux) or "
            "choco install ffmpeg (Windows) or brew install ffmpeg (macOS)"
        )

    logger.debug("Converted %s -> %s (%.1f KB)", ogg_path.name, wav_path.name, wav_path.stat().st_size / 1024)
    return wav_path


def convert_wav_to_ogg(wav_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert WAV audio to OGG/Opus format for WhatsApp voice notes.

    WhatsApp expects OGG container with Opus codec:
      - 16kHz sample rate, mono
      - Opus codec (libopus)

    Returns the output OGG path.
    """
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".ogg"))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(wav_path),
                "-c:a", "libopus",
                "-b:a", "32k",         # 32kbps — good quality, small size
                "-ar", "16000",
                "-ac", "1",
                "-y",
                str(output_path),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg OGG conversion failed: %s", exc.stderr.decode(errors="replace"))
        raise RuntimeError(f"OGG conversion failed: {exc}") from exc

    return output_path


def convert_mp3_to_ogg(mp3_path: Path, output_path: Path | None = None) -> Path:
    """Convert Edge TTS MP3 output to OGG/Opus for WhatsApp."""
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".ogg"))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(mp3_path),
                "-c:a", "libopus",
                "-b:a", "32k",
                "-y",
                str(output_path),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"MP3 to OGG conversion failed: {exc}") from exc

    return output_path


def get_audio_duration_s(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            check=True,
            timeout=10,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def normalise_audio_volume(wav_path: Path) -> Path:
    """
    Normalise audio volume using ffmpeg loudnorm filter.
    Helps with noisy WhatsApp recordings.
    Returns path to normalised WAV file.
    """
    out_path = Path(tempfile.mktemp(suffix=".wav"))

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(wav_path),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                str(out_path),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError:
        logger.warning("Volume normalisation failed, using original")
        return wav_path

    return out_path

# ffmpeg loudnorm filter: I=-16 LUFS, TP=-1.5 dBTP, LRA=11 LU
