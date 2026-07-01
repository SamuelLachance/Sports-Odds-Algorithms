"""Merge ESPN history with free supplemental sources for tuning/backtests."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from web.archive_csv_games import load_archive_csv_games
from web.football_data_uk import load_football_data_uk_games
from web.league_profiles import is_soccer_league
from web.live_data import _parse_cutoff
from web.season_games import GameTuple, load_league_completed_games

# Optional closing odds keyed by (date, home, away) for soccer backtests
_soccer_odds_cache: dict[str, dict[tuple[str, str, str], dict[str, int | None]]] = {}


def _game_dedupe_key(
    game_date: str,
    home_key: str,
    away_key: str,
) -> str:
    return f"{game_date}:{home_key}:{away_key}"


def _espn_games_with_dates(
    league: str,
    cutoff_date: str,
    *,
    for_backtest: bool,
) -> list[tuple[str, GameTuple]]:
    """Attach approximate dates to ESPN-only tuples (date unknown -> empty)."""
    games = load_league_completed_games(league, cutoff_date, for_backtest=for_backtest)
    return [("", game) for game in games]


def _supplemental_only_games(league: str, cutoff_date: str) -> list[tuple[str, GameTuple]]:
    cutoff = _parse_cutoff(cutoff_date)
    rows: list[tuple[str, GameTuple]] = []

    if is_soccer_league(league) and league in {
        "epl",
        "bundesliga",
        "laliga",
        "seriea",
        "ligue1",
    }:
        odds_map: dict[tuple[str, str, str], dict[str, int | None]] = {}
        for record in load_football_data_uk_games(league):
            game_dt = datetime.strptime(record["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            if game_dt >= cutoff:
                continue
            home_key = record["home_key"]
            away_key = record["away_key"]
            game_date = record["date"]
            rows.append(
                (
                    game_date,
                    (
                        home_key,
                        away_key,
                        record["home_name"],
                        record["away_name"],
                        int(record["home_goals"]),
                        int(record["away_goals"]),
                    ),
                )
            )
            odds_map[(game_date, home_key, away_key)] = {
                "home_odds": record.get("home_odds"),
                "draw_odds": record.get("draw_odds"),
                "away_odds": record.get("away_odds"),
            }
        _soccer_odds_cache[league] = odds_map
    elif league in ("nba", "nhl", "mlb"):
        espn_games = load_league_completed_games(league, cutoff_date, for_backtest=True)
        if len(espn_games) >= 2000:
            return rows
        for game in load_archive_csv_games(league, cutoff_date):
            rows.append(("", game))

    return rows


def soccer_backtest_odds(
    league: str,
    game_date: str,
    home_key: str,
    away_key: str,
) -> dict[str, int | None] | None:
    """Closing 1X2 odds from football-data when available."""
    league_map = _soccer_odds_cache.get(league.lower(), {})
    return league_map.get((game_date, home_key, away_key))


def load_supplemental_completed_games(
    league: str,
    cutoff_date: str,
    *,
    for_backtest: bool = True,
) -> list[GameTuple]:
    """
    ESPN completed games merged with supplemental history, deduped by teams+score.

    Live slate still uses ESPN directly; this is for calibration and walk-forward tuning.
    """
    league = league.lower()
    merged: list[GameTuple] = []
    seen: set[tuple[str, str, int, int]] = set()

    def _append(game: GameTuple) -> None:
        signature = (game[0], game[1], game[4], game[5])
        if signature in seen:
            return
        seen.add(signature)
        merged.append(game)

    for _game_date, game_tuple in _espn_games_with_dates(
        league, cutoff_date, for_backtest=for_backtest
    ):
        _append(game_tuple)

    for _game_date, game_tuple in _supplemental_only_games(league, cutoff_date):
        _append(game_tuple)

    return merged


def supplemental_metadata(league: str) -> dict[str, Any]:
    league = league.lower()
    sources: list[str] = ["espn"]
    if is_soccer_league(league) and league in {
        "epl",
        "bundesliga",
        "laliga",
        "seriea",
        "ligue1",
    }:
        sources.append("football-data.co.uk")
    if league in ("nba", "nhl", "mlb"):
        sources.append(f"{league}/team_data CSV archives")
    return {"league": league, "sources": sources}
