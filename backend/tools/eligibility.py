"""
Structured eligibility checker for VaakSeva.

Maintains a JSON database of scheme eligibility rules and performs
rule-based matching against the user profile.

This is separate from the RAG pipeline:
  - RAG retrieves general scheme information (descriptions, how-to-apply, etc.)
  - This tool does precise eligibility checking using structured rules

The structured approach handles cases where RAG might hallucinate about
eligibility criteria, and enables showing match scores and missing criteria.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.models.schemas import EligibilityResult

logger = logging.getLogger(__name__)


@dataclass
class Criterion:
    field: str
    op: str          # eq, in, gt, lt, gte, lte, any
    value: Any
    mandatory: bool = True

    def evaluate(self, profile: dict) -> bool:
        """Return True if this criterion is satisfied by the user profile."""
        user_val = profile.get(self.field)

        if user_val is None:
            return False  # Unknown fields fail the check

        if self.op == "eq":
            return user_val == self.value
        elif self.op == "in":
            return str(user_val).lower() in [str(v).lower() for v in self.value]
        elif self.op == "gt":
            return float(user_val) > float(self.value)
        elif self.op == "lt":
            return float(user_val) < float(self.value)
        elif self.op == "gte":
            return float(user_val) >= float(self.value)
        elif self.op == "lte":
            return float(user_val) <= float(self.value)
        elif self.op == "any":
            return True  # Always passes (field can be any value)
        else:
            logger.warning("Unknown criterion op: %s", self.op)
            return False


class EligibilityChecker:
    """
    Rule-based eligibility checker.

    Loads scheme rules from schemes_structured.json.
    Matches user profile fields against criteria.
    Returns scored results with matched/missing criteria.
    """

    def __init__(self, structured_path: Path | None = None):
        path = structured_path or settings.schemes_structured_path
        if not path.exists():
            logger.warning("schemes_structured.json not found at %s", path)
            self._schemes: dict = {}
        else:
            try:
                self._schemes = json.loads(path.read_text(encoding="utf-8"))
                logger.info("Loaded %d schemes for eligibility checking", len(self._schemes))
            except Exception as exc:
                logger.error("Failed to load schemes_structured.json: %s", exc)
                self._schemes = {}

    def check_all(self, profile: dict) -> list[EligibilityResult]:
        """
        Check eligibility for all schemes in the database.

        Returns list of EligibilityResult sorted by score descending.
        Only returns schemes with score > 0.
        """
        if not profile:
            return []

        results = []
        for scheme_id, scheme_data in self._schemes.items():
            result = self._check_scheme(scheme_id, scheme_data, profile)
            if result is not None and result.score > 0:
                results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def check_scheme(self, scheme_id: str, profile: dict) -> EligibilityResult | None:
        """Check eligibility for a specific scheme."""
        scheme_data = self._schemes.get(scheme_id)
        if not scheme_data:
            return None
        return self._check_scheme(scheme_id, scheme_data, profile)

    def _check_scheme(self, scheme_id: str, scheme_data: dict, profile: dict) -> EligibilityResult | None:
        """Internal: evaluate one scheme against the profile."""
        criteria_raw = scheme_data.get("criteria", [])
        if not criteria_raw:
            return None

        criteria = [Criterion(**c) for c in criteria_raw]

        matched = []
        missing = []

        # Check mandatory criteria first
        all_mandatory_met = True
        for criterion in criteria:
            satisfied = criterion.evaluate(profile)
            if satisfied:
                matched.append(criterion.field)
            else:
                missing.append(criterion.field)
                if criterion.mandatory:
                    all_mandatory_met = False

        if not all_mandatory_met:
            # Scheme is not eligible — but still return with low score for partial info
            # Only discard if user has confirmed info that violates mandatory criteria
            profile_has_relevant_fields = any(
                c.field in profile for c in criteria if c.mandatory
            )
            if profile_has_relevant_fields:
                return EligibilityResult(
                    scheme_id=scheme_id,
                    scheme_name=scheme_data.get("name_en", scheme_id),
                    scheme_name_hi=scheme_data.get("name_hi", scheme_id),
                    score=0.0,
                    matched_criteria=matched,
                    missing_criteria=missing,
                    benefits_summary_hi=scheme_data.get("benefits_summary_hi", ""),
                    apply_url=scheme_data.get("apply_url"),
                )
            return None  # Not enough profile info to determine

        # Score: fraction of all criteria met, weighted by mandatory
        total_weight = sum(2 if c.mandatory else 1 for c in criteria)
        matched_weight = sum(
            (2 if c.mandatory else 1)
            for c in criteria
            if c.field in matched
        )
        score = matched_weight / total_weight if total_weight > 0 else 0.0

        return EligibilityResult(
            scheme_id=scheme_id,
            scheme_name=scheme_data.get("name_en", scheme_id),
            scheme_name_hi=scheme_data.get("name_hi", scheme_id),
            score=round(score, 3),
            matched_criteria=matched,
            missing_criteria=missing,
            benefits_summary_hi=scheme_data.get("benefits_summary_hi", ""),
            apply_url=scheme_data.get("apply_url"),
        )
