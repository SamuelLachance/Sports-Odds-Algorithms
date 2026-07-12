"""MLB Stats API helper tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID, fetch_mlb_season_standings  # noqa: E402


def test_espn_mlb_team_map_has_all_clubs() -> None:
    assert len(ESPN_TO_MLB_TEAM_ID) >= 30
    assert ESPN_TO_MLB_TEAM_ID["bos"] == 111
    assert ESPN_TO_MLB_TEAM_ID["nyy"] == 147


def test_standings_team_key_aliases_white_sox_to_espn() -> None:
    """MLB fileCode is cws; live slate / ESPN use chw."""
    from web.mlb_stats_api import _standings_team_key

    assert _standings_team_key({"fileCode": "cws", "abbreviation": "CWS"}) == "chw"
    assert _standings_team_key({"fileCode": "bos"}) == "bos"


def test_mlb_api_matchup_edge_resolves_chw_from_cws_standings(monkeypatch) -> None:
    from web import mlb_stats_api as api

    api.fetch_mlb_season_standings.cache_clear()

    def _fake_standings(season: int) -> dict:
        return {
            "chw": {
                "win_pct": 0.55,
                "runs_per_game": 4.5,
                "runs_allowed_per_game": 4.0,
                "run_diff_per_game": 0.5,
                "games": 100.0,
            },
            "bos": {
                "win_pct": 0.45,
                "runs_per_game": 4.0,
                "runs_allowed_per_game": 4.5,
                "run_diff_per_game": -0.5,
                "games": 100.0,
            },
        }

    monkeypatch.setattr(api, "fetch_mlb_season_standings", _fake_standings)
    edge = api.mlb_api_matchup_edge("chw", "bos", 2025)
    assert edge is not None
    assert edge > 0


def test_fetch_mlb_season_standings_shape() -> None:
    standings = fetch_mlb_season_standings(2024)
    if not standings:
        return
    assert "nyy" in standings or "bos" in standings
    # White Sox must be keyed as ESPN chw, not MLB fileCode cws.
    assert "cws" not in standings or "chw" in standings
    if standings.get("chw") or standings.get("cws"):
        assert "chw" in standings
    sample = next(iter(standings.values()))
    assert "win_pct" in sample
    assert "runs_per_game" in sample
