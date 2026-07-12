"""Health endpoint reports v2 artifact availability."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.app import _http_error, health  # noqa: E402

EXPECTED_V2_KEYS = ("nba", "wnba", "nhl", "mlb", "soccer", "nfl", "cfb", "cbb")


def test_health_reports_v2_artifacts() -> None:
    payload = health()
    assert payload["status"] == "ok"
    artifacts = payload["v2_artifacts"]
    assert set(artifacts) == set(EXPECTED_V2_KEYS)
    assert all(isinstance(v, bool) for v in artifacts.values())
    assert payload["v2_artifacts_total"] == len(EXPECTED_V2_KEYS)
    assert 0 <= payload["v2_artifacts_ready"] <= len(EXPECTED_V2_KEYS)


def test_http_error_is_structured() -> None:
    exc = _http_error(
        404,
        "Database snapshot not found.",
        code="db_snapshot_missing",
        hint="Snapshots are rebuilt on the next Pages deploy.",
        path="nba/league.json",
    )
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 404
    detail = exc.detail
    assert isinstance(detail, dict)
    assert detail["error"] is True
    assert detail["code"] == "db_snapshot_missing"
    assert detail["message"] == "Database snapshot not found."
    assert "hint" in detail
    assert detail["path"] == "nba/league.json"


def test_v2_live_ttl_is_three_hours() -> None:
    from web.cbb_v2 import live as cbb_live
    from web.nba_v2 import live as nba_live

    assert nba_live.EVENTS_TTL_SECONDS == 3 * 3600
    assert cbb_live.EVENTS_TTL_SECONDS == 3 * 3600
