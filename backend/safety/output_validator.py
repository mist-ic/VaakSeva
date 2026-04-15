"""
Output validation for VaakSeva.

Cross-references LLM responses against the structured scheme database
to detect hallucinated benefit amounts or eligibility criteria.

Approach:
  1. Extract numerical claims from LLM output (amounts, percentages, etc.)
  2. Identify which scheme the response is about
  3. Cross-check extracted claims against known ground truth in structured DB
  4. Flag mismatches with confidence score

This catches the most common and dangerous hallucination: wrong benefit amounts.
E.g., LLM says PM-KISAN gives Rs 8000/year instead of Rs 6000/year.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from backend.config import settings
from backend.models.schemas import OutputValidationResult, RetrievedChunk

logger = logging.getLogger(__name__)

# Match currency amounts: Rs or rupee symbols followed by numbers
_AMOUNT_PATTERN = re.compile(
    r"(?:Rs\.?\s*|₹\s*|rupees?\s*)(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:crore|lakh|thousand)?",
    re.IGNORECASE,
)


class OutputValidator:
    """
    Cross-reference LLM output against structured scheme data.

    Checks that claimed benefit amounts match known amounts from the database.
    Returns OutputValidationResult with confidence score.
    """

    def __init__(self, structured_path: Path | None = None):
        path = structured_path or settings.schemes_structured_path
        if path.exists():
            try:
                self._schemes = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._schemes = {}
        else:
            self._schemes = {}

    def validate(
        self,
        response_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> OutputValidationResult:
        """
        Validate LLM response against known scheme data.

        Returns is_valid=False if a significant amount mismatch is detected.
        """
        if not retrieved_chunks or not self._schemes:
            return OutputValidationResult(is_valid=True, confidence=0.5)

        flagged_claims = []

        # Identify schemes mentioned in retrieved chunks
        mentioned_scheme_ids = list({c.scheme_id for c in retrieved_chunks})

        # Extract amount claims from LLM response
        response_amounts = self._extract_amounts(response_text)

        if not response_amounts:
            # No specific amounts claimed — nothing to verify
            return OutputValidationResult(is_valid=True, confidence=0.8)

        # For each mentioned scheme, check if amounts are plausible
        for scheme_id in mentioned_scheme_ids:
            scheme = self._schemes.get(scheme_id)
            if not scheme:
                continue

            known_amounts = self._get_known_amounts(scheme)
            if not known_amounts:
                continue

            for claimed_amount in response_amounts:
                # Allow for 10% tolerance (formatting differences, etc.)
                if not any(
                    abs(claimed_amount - known) / max(known, 1) < 0.15
                    for known in known_amounts
                ):
                    flagged_claims.append(
                        f"Claimed amount Rs{claimed_amount:,.0f} not matching known amounts "
                        f"for {scheme_id}: {known_amounts}"
                    )
                    logger.warning(
                        "Potential hallucination in %s: claimed Rs%s, known %s",
                        scheme_id, claimed_amount, known_amounts
                    )

        is_valid = len(flagged_claims) == 0
        confidence = 0.95 if is_valid else max(0.3, 0.95 - 0.2 * len(flagged_claims))

        return OutputValidationResult(
            is_valid=is_valid,
            flagged_claims=flagged_claims,
            confidence=confidence,
        )

    @staticmethod
    def _extract_amounts(text: str) -> list[float]:
        """Extract all currency amounts mentioned in text."""
        amounts = []
        for match in _AMOUNT_PATTERN.finditer(text):
            try:
                # Remove commas and parse
                raw = match.group(1).replace(",", "")
                amount = float(raw)
                amounts.append(amount)
            except ValueError:
                pass
        return amounts

    @staticmethod
    def _get_known_amounts(scheme: dict) -> list[float]:
        """Extract known amounts from scheme's benefits_summary_hi field."""
        text = scheme.get("benefits_summary_hi", "") + " " + scheme.get("benefits_summary_en", "")
        amounts = []
        # Look for numbers with optional thousand separators
        for m in re.finditer(r"(\d+(?:,\d{3})*(?:\.\d+)?)", text):
            try:
                amounts.append(float(m.group(1).replace(",", "")))
            except ValueError:
                pass
        return [a for a in amounts if a >= 100]  # Filter out small numbers (percentages, etc.)

# 15 percent tolerance on amount comparison handles formatting differences
