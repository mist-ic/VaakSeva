"""
Structured JSON logging with request tracing for VaakSeva.

Every request produces a structured log entry with:
  - Unique request ID (for tracing across logs)
  - User phone hash (privacy: SHA-256 of phone number)
  - Input type (text or voice)
  - Per-stage latency breakdown
  - Retrieval metadata (top scheme, score)
  - Response length and language
  - Error info if pipeline failed

Logs are written as JSONL (one JSON object per line) for easy parsing.
Compatible with GCP Cloud Logging structured log format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings

_CONFIGURED = False


def _configure_root_logger():
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    # JSON formatter for structured logging
    from pythonjsonlogger import jsonlogger

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_dir / "vaakseva.log",
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Stream handler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with structured JSON formatting configured."""
    _configure_root_logger()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Request-level logger
# ---------------------------------------------------------------------------


class RequestLogger:
    """
    Writes structured per-request log entries to a JSONL file.

    Format matches GCP Cloud Logging structured log expectations.
    """

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "requests.jsonl"

    def log_request(self, result, original_query: str) -> None:
        """Write a structured log entry for a completed pipeline request."""
        from backend.models.schemas import QueryResponse

        try:
            timings = result.timings.model_dump() if result.timings else {}
            top_chunk = result.retrieved_chunks[0] if result.retrieved_chunks else None

            entry = {
                "request_id": result.request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_type": "text",
                "pipeline_timings": timings,
                "retrieval": {
                    "num_results": len(result.retrieved_chunks),
                    "top_scheme": top_chunk.scheme_id if top_chunk else None,
                    "top_score": top_chunk.score if top_chunk else None,
                },
                "response_length": len(result.response_text),
                "language_detected": result.language_detected,
                "output_valid": result.output_validation.is_valid if result.output_validation else None,
                "safety_flagged": not result.safety.is_safe if result.safety else False,
            }

            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as exc:
            logging.getLogger(__name__).error("Failed to write request log: %s", exc)

    def log_error(self, request_id: str, error: str, phone_number: str | None = None) -> None:
        """Write an error log entry."""
        phone_hash = None
        if phone_number:
            phone_hash = hashlib.sha256(
                f"{settings.phone_hash_salt}:{phone_number}".encode()
            ).hexdigest()[:24]

        entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "user_phone_hash": phone_hash,
        }

        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

# RotatingFileHandler: 50MB max per file, 5 backup files
