"""Tests for eligibility checker."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from backend.tools.eligibility import EligibilityChecker, Criterion


class TestCriterion:
    def test_eq_match(self):
        c = Criterion(field="has_aadhaar", op="eq", value=True, mandatory=True)
        assert c.evaluate({"has_aadhaar": True}) is True
        assert c.evaluate({"has_aadhaar": False}) is False

    def test_in_match(self):
        c = Criterion(field="occupation", op="in", values=["farmer", "laborer"], mandatory=True)
        assert c.evaluate({"occupation": "farmer"}) is True
        assert c.evaluate({"occupation": "FARMER"}) is True  # case-insensitive
        assert c.evaluate({"occupation": "doctor"}) is False

    def test_gt_match(self):
        c = Criterion(field="income", op="gt", value=50000, mandatory=False)
        assert c.evaluate({"income": 100000}) is True
        assert c.evaluate({"income": 30000}) is False

    def test_lte_match(self):
        c = Criterion(field="age", op="lte", value=40, mandatory=True)
        assert c.evaluate({"age": 35}) is True
        assert c.evaluate({"age": 40}) is True
        assert c.evaluate({"age": 50}) is False

    def test_missing_field(self):
        c = Criterion(field="income", op="gt", value=0, mandatory=True)
        assert c.evaluate({}) is False  # Missing field = fail

    def test_any_op(self):
        c = Criterion(field="state", op="any", value=None, mandatory=False)
        assert c.evaluate({"state": "Karnataka"}) is True


class TestEligibilityChecker:
    def test_farmer_pm_kisan(self, structured_db_file):
        checker = EligibilityChecker(structured_db_file)
        profile = {"occupation": "farmer", "has_aadhaar": True}
        results = checker.check_all(profile)
        scheme_ids = [r.scheme_id for r in results]
        assert "pm_kisan" in scheme_ids

    def test_non_farmer_not_pm_kisan(self, structured_db_file):
        checker = EligibilityChecker(structured_db_file)
        profile = {"occupation": "doctor", "has_aadhaar": True}
        results = checker.check_all(profile)
        # PM-KISAN should have score 0 for non-farmers
        pm_kisan = next((r for r in results if r.scheme_id == "pm_kisan"), None)
        if pm_kisan:
            assert pm_kisan.score == 0.0

    def test_bpl_ayushman(self, structured_db_file):
        checker = EligibilityChecker(structured_db_file)
        profile = {"is_bpl": True}
        results = checker.check_all(profile)
        scheme_ids = [r.scheme_id for r in results]
        assert "ayushman_bharat" in scheme_ids

    def test_score_between_0_and_1(self, structured_db_file):
        checker = EligibilityChecker(structured_db_file)
        profile = {"occupation": "farmer", "has_aadhaar": True, "is_bpl": True}
        results = checker.check_all(profile)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_empty_profile_returns_empty(self, structured_db_file):
        checker = EligibilityChecker(structured_db_file)
        results = checker.check_all({})
        assert results == []

    def test_missing_db_returns_empty(self, temp_dir):
        checker = EligibilityChecker(temp_dir / "nonexistent.json")
        results = checker.check_all({"occupation": "farmer"})
        assert results == []

    def test_results_sorted_by_score(self, structured_db_file):
        checker = EligibilityChecker(structured_db_file)
        profile = {"occupation": "farmer", "has_aadhaar": True, "is_bpl": True}
        results = checker.check_all(profile)
        if len(results) > 1:
            assert results[0].score >= results[1].score
