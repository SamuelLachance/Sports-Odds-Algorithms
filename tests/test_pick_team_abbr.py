"""Pick payloads include team_abbr for frontend team links."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import BetPick, pick_to_dict  # noqa: E402
from web.daily_service import _enrich_pick_with_team_abbr  # noqa: E402


def test_enrich_pick_with_team_abbr() -> None:
    pick = pick_to_dict(
        BetPick(
            side="home",
            team_name="Diamondbacks",
            team_slug="arizona-diamondbacks",
            strategy="value",
            confidence="medium",
            edge=45.0,
            model_projection=120,
            market_odds=141,
            win_probability=55.0,
            reason="Edge",
        )
    )
    matchup = {
        "away": {"abbr": "min", "name": "Twins"},
        "home": {"abbr": "ari", "name": "Diamondbacks"},
    }
    enriched = _enrich_pick_with_team_abbr(pick, matchup)
    assert enriched["team_abbr"] == "ari"
    assert enriched["team_slug"] == "arizona-diamondbacks"


def test_pick_to_dict_includes_base_win_probability_for_away() -> None:
    """Honest EV track needs pick-scoped base prob, not home-only model fields."""
    pick = pick_to_dict(
        BetPick(
            side="away",
            team_name="Twins",
            team_slug="minnesota-twins",
            strategy="value",
            confidence="medium",
            edge=12.0,
            model_projection=130,
            market_odds=145,
            win_probability=52.0,
            reason="Edge",
            extra={"base_win_probability": 54.5, "games_played_proxy": 40, "kelly_pct": 1.2},
        )
    )
    assert pick["side"] == "away"
    assert pick["base_win_probability"] == 54.5
    assert pick["games_played_proxy"] == 40
    assert pick["kelly_pct"] == 1.2


def test_draw_pick_has_no_team_abbr() -> None:
    pick = {"side": "draw", "team_name": "Draw"}
    matchup = {
        "away": {"abbr": "ars", "name": "Arsenal"},
        "home": {"abbr": "che", "name": "Chelsea"},
    }
    assert _enrich_pick_with_team_abbr(pick, matchup) == pick
