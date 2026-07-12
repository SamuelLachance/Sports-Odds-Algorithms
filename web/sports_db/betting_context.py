"""Extract betting-relevant context from the daily slate for database sheets."""

from __future__ import annotations

from typing import Any


def _model_summary(model: dict[str, Any] | None) -> dict[str, Any]:
    if not model:
        return {}
    agreement = model.get("model_agreement") or {}
    return {
        "total_score": model.get("total_score"),
        "win_probability": model.get("win_probability"),
        "favorite_side": model.get("favorite_side"),
        "blended_home_win_probability": model.get("blended_home_win_probability"),
        "blend_mode": model.get("blend_mode"),
        "blend_layers": model.get("blend_layers"),
        "agreement": {
            "required": agreement.get("required"),
            "agreed": agreement.get("agreed"),
            "value_sides": agreement.get("value_sides") or agreement.get("value_outcomes"),
        },
        "away_projection": model.get("away_projection"),
        "home_projection": model.get("home_projection"),
    }


def slate_games_for_league(slate: dict[str, Any] | None, league: str) -> list[dict[str, Any]]:
    if not slate:
        return []
    return [g for g in slate.get("games") or [] if g.get("league") == league]


def game_betting_sheet(game: dict[str, Any]) -> dict[str, Any]:
    matchup = game.get("matchup") or {}
    return {
        "event_id": game.get("event_id"),
        "name": game.get("name"),
        "start_time": game.get("start_time"),
        "status": game.get("status"),
        "league": game.get("league"),
        "league_name": game.get("league_name"),
        "matchup": matchup,
        "market": game.get("market") or {},
        "model": _model_summary(game.get("model")),
        "recommendations": game.get("recommendations") or [],
        "top_pick": game.get("top_pick"),
        "eligible_for_official_picks": game.get("eligible_for_official_picks", False),
        "official_bet_type": game.get("official_bet_type"),
        "factors": (game.get("model") or {}).get("factors") or game.get("factors"),
    }


def league_betting_context(slate: dict[str, Any] | None, league: str) -> dict[str, Any]:
    games = [game_betting_sheet(g) for g in slate_games_for_league(slate, league)]
    official_picks = sum(
        1
        for g in games
        if g.get("eligible_for_official_picks", False) and (g.get("recommendations") or g.get("top_pick"))
    )
    return {
        "games_today": games,
        "game_count": len(games),
        "official_pick_count": official_picks,
        "has_soccer_predictions_only": any(
            not g.get("eligible_for_official_picks", False) for g in games
        ),
    }


def team_betting_context(
    slate: dict[str, Any] | None,
    league: str,
    abbr: str,
) -> dict[str, Any]:
    abbr = abbr.lower()
    sheets: list[dict[str, Any]] = []
    for game in slate_games_for_league(slate, league):
        matchup = game.get("matchup") or {}
        for side in ("home", "away"):
            side_abbr = ((matchup.get(side) or {}).get("abbr") or "").lower()
            if side_abbr != abbr:
                continue
            sheet = game_betting_sheet(game)
            sheet["team_side"] = side
            sheet["opponent"] = matchup.get("away" if side == "home" else "home")
            sheets.append(sheet)
    return {
        "upcoming_games": sheets,
        "game_count": len(sheets),
        "has_official_pick": any(
            (s.get("recommendations") or [])
            and s.get("eligible_for_official_picks", False)
            for s in sheets
        ),
    }
