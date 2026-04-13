"""
Profile extractor for VaakSeva.

Sends user Hindi messages through the LLM with an extraction prompt
to parse structured profile information (age, state, occupation, etc.).

This extracted info is stored in UserMemory and used for:
  - RAG context enrichment
  - Eligibility checking
"""

from __future__ import annotations

import json
import logging
import re

from backend.llm.prompts import build_profile_extraction_prompt

logger = logging.getLogger(__name__)


class ProfileExtractor:
    """
    Extracts structured user profile fields from free-form Hindi text.
    """

    def __init__(self, llm_client):
        self._llm = llm_client

    async def extract(self, message: str) -> dict:
        """
        Extract profile fields from a Hindi message.

        Returns a dict with any of:
          age, gender, state, district, occupation, income,
          category, education, has_aadhaar, is_bpl
        Only fields explicitly mentioned are returned.
        Returns {} on failure.
        """
        prompt = build_profile_extraction_prompt(message)

        try:
            response = await self._llm.agenerate(
                prompt,
                system="You are a JSON extraction assistant. Return only valid JSON, nothing else.",
            )
            return self._parse_json(response)
        except Exception as exc:
            logger.warning("Profile extraction failed: %s", exc)
            return {}

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Strip markdown code block if present
        text = text.strip()
        # Remove ```json ... ``` wrapping
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # Find first { ... } block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return {}
            return {k: v for k, v in data.items() if v is not None}
        except json.JSONDecodeError:
            logger.debug("Could not parse JSON from: %r", text[:200])
            return {}
