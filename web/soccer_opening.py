"""Soccer 1X2 opening-line steam detection (ESPN public odds)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from web.bet_advisor import american_implied_prob
from web.espn_client import _parse_american_odds, get_league_profile
from web.soccer_decorrelation import devig_threeway_from_odds

STEAM_IMPLIED_SHIFT_PP = 2.5
STEAM_MODEL_EDGE_PP = 4.0


def _fetch_event_odds_block(league: str, event_id: str) -> dict[str, Any] | None:
    if not event_id:
        return None
    profile = get_league_profile(league)
    sport_path = profile.get("sport_path")
    if not sport_path:
        return None
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/{sport_path}/"
        f"summary?event={event_id}"
    )
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None
    competition = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    if not competition:
        competition = ((payload.get("gameInfo") or {}).get("competitions") or [{}])[0]
    odds_blocks = competition.get("odds") or payload.get("pickcenter") or []
    if isinstance(odds_blocks, list) and odds_blocks:
        block = odds_blocks[0]
        return block if isinstance(block, dict) else None
    return None


def _extract_open_close_moneyline(
    odds_block: dict[str, Any] | None,
    side: str,
) -> tuple[int | None, int | None]:
    if not odds_block:
        return None, None
    moneyline = odds_block.get("moneyline") or {}
    side_block = (moneyline.get(side) or {})
    open_odds = _parse_american_odds((side_block.get("open") or {}).get("odds"))
    close_odds = _parse_american_odds((side_block.get("close") or {}).get("odds"))
    if close_odds is None and side in {"home", "away"}:
        team_key = "homeTeamOdds" if side == "home" else "awayTeamOdds"
        close_odds = _parse_american_odds((odds_block.get(team_key) or {}).get("moneyLine"))
    return open_odds, close_odds


def fetch_soccer_opening_moneylines(
    league: str,
    event_id: str | None,
) -> tuple[int | None, int | None, int | None]:
    """Return opening home/draw/away moneylines when ESPN exposes them."""
    odds_block = _fetch_event_odds_block(league, event_id or "")
    if not odds_block:
        return None, None, None
    # Missing open must stay None — copying close fabricates 0pp "steam".
    open_home, _ = _extract_open_close_moneyline(odds_block, "home")
    open_draw, _ = _extract_open_close_moneyline(odds_block, "draw")
    open_away, _ = _extract_open_close_moneyline(odds_block, "away")
    return open_home, open_draw, open_away


def _implied_shift_pp(open_ml: int | None, close_ml: int | None) -> float | None:
    """Implied-prob move (pp) from open → close; None on missing/invalid juice."""
    from web.bet_advisor import normalize_american_odds

    if open_ml is None or close_ml is None:
        return None
    open_n = normalize_american_odds(open_ml)
    close_n = normalize_american_odds(close_ml)
    if open_n is None or close_n is None:
        return None
    try:
        open_prob = american_implied_prob(open_n)
        close_prob = american_implied_prob(close_n)
    except ValueError:
        return None
    return (close_prob - open_prob) * 100.0


def soccer_opening_steam_meta(
    *,
    home_ml: int | None,
    draw_ml: int | None,
    away_ml: int | None,
    open_home_ml: int | None,
    open_draw_ml: int | None,
    open_away_ml: int | None,
    model_home: float,
    model_draw: float,
    model_away: float,
) -> dict[str, Any]:
    """Detect 1X2 steam when closing line moves materially from open."""
    meta: dict[str, Any] = {
        "steam_signal": False,
        "steam_direction": None,
        "opening_home_ml": open_home_ml,
        "opening_draw_ml": open_draw_ml,
        "opening_away_ml": open_away_ml,
    }
    shifts = {
        "home": _implied_shift_pp(open_home_ml, home_ml),
        "draw": _implied_shift_pp(open_draw_ml, draw_ml),
        "away": _implied_shift_pp(open_away_ml, away_ml),
    }
    meta["implied_shifts_pp"] = {
        key: round(value, 2) if value is not None else None for key, value in shifts.items()
    }

    market_probs = devig_threeway_from_odds(home_ml, draw_ml, away_ml)
    if market_probs is None:
        return meta

    model_probs = (model_home, model_draw, model_away)
    best_outcome = ("home", "draw", "away")[
        model_probs.index(max(model_probs))
    ]
    model_prob = model_probs[["home", "draw", "away"].index(best_outcome)]
    market_prob = market_probs[["home", "draw", "away"].index(best_outcome)]
    meta["model_best_outcome"] = best_outcome
    meta["model_edge_pp"] = round(model_prob - market_prob, 2)

    steam_outcome = None
    steam_mag = 0.0
    for outcome, shift in shifts.items():
        if shift is None:
            continue
        if abs(shift) >= STEAM_IMPLIED_SHIFT_PP and abs(shift) > abs(steam_mag):
            steam_outcome = outcome
            steam_mag = shift
    if steam_outcome is None:
        return meta

    meta["steam_outcome"] = steam_outcome
    meta["steam_move_pp"] = round(steam_mag, 2)
    if meta["model_edge_pp"] >= STEAM_MODEL_EDGE_PP and steam_outcome == best_outcome:
        meta["steam_signal"] = True
        meta["steam_direction"] = steam_outcome
    return meta
