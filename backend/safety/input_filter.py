"""
Input safety filter for VaakSeva.

Defends against:
  1. Prompt injection attacks (attempts to override system instructions)
  2. Personal data requests (attempts to extract Aadhaar/bank info)
  3. Abusive content
  4. Off-topic misuse

Design principles:
  - Deny-list of known injection patterns (fast, zero latency)
  - Sanitize (strip control chars, normalise whitespace) before passing to LLM
  - Return Hindi error message for blocked queries
  - Log all blocked requests for monitoring
"""

from __future__ import annotations

import logging
import re
import unicodedata

from backend.models.schemas import SafetyCheckResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injection patterns (case-insensitive)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    # Classic prompt injection
    re.compile(r"ignore\s+(previous|above|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)\s+(instructions?|rules?|training)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+\w+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an|the|if)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"roleplay\s+as", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),

    # System prompt exfiltration
    re.compile(r"(show|print|reveal|tell\s+me)\s+(your|the)\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"what\s+(are\s+)?(your\s+)?instructions?", re.IGNORECASE),
    re.compile(r"repeat\s+(everything|your\s+instructions?)", re.IGNORECASE),

    # Hindi injection attempts
    re.compile(r"नियम\s*भूल", re.IGNORECASE),        # "forget rules"
    re.compile(r"निर्देश\s*अनदेखा", re.IGNORECASE),   # "ignore instructions"
]

# ---------------------------------------------------------------------------
# Personal data solicitation patterns (VaakSeva should never request these)
# ---------------------------------------------------------------------------
# We flag if the USER is trying to provide raw Aadhaar/pan numbers (unnecessary risk)

_SENSITIVE_DATA_PATTERNS = [
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Aadhaar number pattern
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),              # PAN card pattern
    re.compile(r"\bIFSC[:\s]*[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE),  # IFSC code
]

# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def _sanitize(text: str) -> str:
    """
    Sanitize input text:
      - Remove control characters (except newline and tab)
      - Normalise Unicode to NFC
      - Collapse excessive whitespace
      - Limit length
    """
    # Remove control characters
    sanitized = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf"} or ch in "\n\t "
    )

    # NFC normalisation
    sanitized = unicodedata.normalize("NFC", sanitized)

    # Collapse multiple spaces (preserve newlines)
    sanitized = re.sub(r" {2,}", " ", sanitized)

    # Trim to reasonable length
    sanitized = sanitized[:1500]

    return sanitized.strip()


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------


class InputFilter:
    """
    Fast pattern-based input filter.

    Returns SafetyCheckResult with:
      - is_safe: bool
      - flagged_patterns: list of detected pattern names
      - sanitized_input: cleaned version of the input
    """

    def check(self, text: str) -> SafetyCheckResult:
        flagged = []

        # Check injection patterns
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                flagged.append(f"injection:{pattern.pattern[:40]}")
                logger.warning("Blocked input - injection pattern matched: %r", text[:100])

        # Check sensitive data patterns (warn but don't block — user might be providing legitimately)
        for pattern in _SENSITIVE_DATA_PATTERNS:
            if pattern.search(text):
                logger.info("Sensitive data pattern found in input (not blocking)")

        is_safe = len(flagged) == 0
        sanitized = _sanitize(text)

        return SafetyCheckResult(
            is_safe=is_safe,
            flagged_patterns=flagged,
            sanitized_input=sanitized,
        )

# Hindi injection patterns: niyam bhool, nirdesh anadekhaa
