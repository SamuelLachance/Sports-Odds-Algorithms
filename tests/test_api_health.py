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
    assert "cause" not in detail


def test_cors_origins_default_to_pages_and_localhost() -> None:
    from web.app import cors_allow_origins

    origins = cors_allow_origins()
    assert "https://samuellachance.github.io" in origins
    assert "http://127.0.0.1:8000" in origins
    assert "*" not in origins


def test_cors_origins_env_star_override(monkeypatch) -> None:
    from web.app import cors_allow_origins

    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    assert cors_allow_origins() == ["*"]


def test_v2_live_ttl_is_three_hours() -> None:
    from web.cbb_v2 import live as cbb_live
    from web.nba_v2 import live as nba_live

    assert nba_live.EVENTS_TTL_SECONDS == 3 * 3600
    assert cbb_live.EVENTS_TTL_SECONDS == 3 * 3600


def test_get_seasons_rejects_unknown_league() -> None:
    from web.predict_service import get_seasons

    try:
        get_seasons("not-a-league")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported league" in str(exc)


def test_seasons_route_returns_structured_400() -> None:
    from web.app import seasons

    try:
        seasons("zzz")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert isinstance(exc.detail, dict)
        assert exc.detail["code"] == "invalid_league"


def test_read_db_json_corrupt_returns_structured_500(tmp_path, monkeypatch) -> None:
    import web.app as app_mod

    monkeypatch.setattr(app_mod, "DB_DIR", tmp_path)
    bad = tmp_path / "manifest.json"
    bad.write_text("{not-json", encoding="utf-8")
    try:
        app_mod._read_db_json("manifest.json")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 500
        assert isinstance(exc.detail, dict)
        assert exc.detail["code"] == "db_snapshot_corrupt"
        assert "traceback" not in str(exc.detail).lower()


def test_read_db_json_non_object_returns_structured_500(tmp_path, monkeypatch) -> None:
    import web.app as app_mod

    monkeypatch.setattr(app_mod, "DB_DIR", tmp_path)
    (tmp_path / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    try:
        app_mod._read_db_json("manifest.json")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 500
        assert exc.detail["code"] == "db_snapshot_invalid"
