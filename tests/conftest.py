"""Shared pytest fixtures for VaakSeva tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_scheme_structured():
    return {
        "pm_kisan": {
            "name_en": "PM-KISAN",
            "name_hi": "पीएम-किसान",
            "category": "agriculture",
            "apply_url": "https://pmkisan.gov.in",
            "benefits_summary_hi": "प्रति वर्ष ₹6,000 सीधे बैंक खाते में",
            "benefits_summary_en": "Rs 6000 per year",
            "criteria": [
                {"field": "occupation", "op": "in", "values": ["farmer"], "mandatory": True},
                {"field": "has_aadhaar", "op": "eq", "value": True, "mandatory": True},
            ],
        },
        "ayushman_bharat": {
            "name_en": "Ayushman Bharat",
            "name_hi": "आयुष्मान भारत",
            "category": "health",
            "apply_url": "https://pmjay.gov.in",
            "benefits_summary_hi": "₹5 लाख तक का बीमा",
            "benefits_summary_en": "Rs 5 lakh health insurance",
            "criteria": [
                {"field": "is_bpl", "op": "eq", "value": True, "mandatory": True},
            ],
        },
    }


@pytest.fixture
def structured_db_file(temp_dir, sample_scheme_structured):
    path = temp_dir / "schemes_structured.json"
    path.write_text(json.dumps(sample_scheme_structured), encoding="utf-8")
    return path


@pytest.fixture
def sample_chunks():
    from backend.models.schemas import RetrievedChunk
    return [
        RetrievedChunk(
            chunk_id="abc123",
            scheme_name="PM-KISAN",
            scheme_id="pm_kisan",
            content="PM-KISAN योजना में किसानों को प्रति वर्ष ₹6,000 मिलते हैं।",
            language="hi",
            chunk_index=0,
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id="def456",
            scheme_name="Ayushman Bharat",
            scheme_id="ayushman_bharat",
            content="आयुष्मान भारत में ₹5 लाख का बीमा मिलता है।",
            language="hi",
            chunk_index=0,
            score=0.85,
        ),
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns a static Hindi response."""
    mock = AsyncMock()
    mock.agenerate = AsyncMock(
        return_value="पीएम-किसान योजना में किसानों को प्रति वर्ष ₹6,000 की सहायता मिलती है।"
    )
    return mock

# Shared fixtures: temp_dir, structured_db_file, sample_chunks, mock_llm
