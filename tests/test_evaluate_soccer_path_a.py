"""Tests for Soccer Path A grading script."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_soccer_path_a import grade_entry  # noqa: E402


def test_grade_entry_a_plus_when_calibrated_beats_market() -> None:
    assert grade_entry({"log_loss_calibrated": 0.95, "log_loss_market_closing": 0.96}) == "A+"


def test_grade_entry_a_when_within_tolerance() -> None:
    assert grade_entry({"log_loss_calibrated": 0.961, "log_loss_market_closing": 0.96}) == "A"


def test_grade_entry_b_when_no_market() -> None:
    assert grade_entry({"log_loss_calibrated": 0.98}) == "B"
