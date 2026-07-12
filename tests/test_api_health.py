"""Health endpoint reports v2 artifact availability."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.app import health  # noqa: E402

EXPECTED_V2_KEYS = ("nba", "wnba", "nhl", "mlb", "soccer", "nfl", "cfb", "cbb")


def test_health_reports_v2_artifacts() -> None:
    payload = health()
    assert payload["status"] == "ok"
    artifacts = payload["v2_artifacts"]
    assert set(artifacts) == set(EXPECTED_V2_KEYS)
    assert all(isinstance(v, bool) for v in artifacts.values())
    assert payload["v2_artifacts_total"] == len(EXPECTED_V2_KEYS)
    assert 0 <= payload["v2_artifacts_ready"] <= len(EXPECTED_V2_KEYS)
