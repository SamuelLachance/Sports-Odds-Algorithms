"""Merge ESPN history with free supplemental sources for tuning/backtests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from web.archive_csv_games import load_archive_csv_games
from web.football_data_uk import load_football_data_uk_games
from web.international_soccer_odds import (
    international_odds_lookup,
    load_international_fd_games,
    load_international_odds_index,
)
from web.league_profiles import INTERNATIONAL_SOCCER_LEAGUES, is_soccer_league
from web.live_data import _parse_cutoff
from web.season_games import GameTuple, load_league_dated_games_for_backtest

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
    if for_backtest:
        return load_league_dated_games_for_backtest(league, cutoff_date)
    from web.season_games import load_league_completed_games

    return [("", game) for game in load_league_completed_games(league, cutoff_date)]


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
    elif league in INTERNATIONAL_SOCCER_LEAGUES or league == "fifa_friendlies":
        odds_map: dict[tuple[str, str, str], dict[str, int | None]] = dict(
            load_international_odds_index()
        )
        for record in load_international_fd_games():
            game_date = record["date"]
            game_dt = datetime.strptime(game_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if game_dt >= cutoff:
                continue
            home_key = record["home_key"]
            away_key = record["away_key"]
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
        from web.season_games import load_league_completed_games

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
    league = league.lower()
    league_map = _soccer_odds_cache.get(league, {})
    hit = league_map.get((game_date[:10], home_key.lower(), away_key.lower()))
    if hit:
        return hit
    if league in INTERNATIONAL_SOCCER_LEAGUES or league == "fifa_friendlies":
        return international_odds_lookup(game_date, home_key, away_key)
    return None


def load_supplemental_dated_games_for_backtest(
    league: str,
    cutoff_date: str,
) -> list[tuple[str, GameTuple]]:
    """ESPN + supplemental history with ISO dates when known."""
    league = league.lower()
    merged: list[tuple[str, GameTuple]] = []
    seen: set[tuple[str, str, int, int]] = set()

    def _append(game_date: str, game: GameTuple) -> None:
        signature = (game[0], game[1], game[4], game[5])
        if signature in seen:
            return
        seen.add(signature)
        merged.append((game_date, game))

    for game_date, game_tuple in _espn_games_with_dates(
        league, cutoff_date, for_backtest=True
    ):
        _append(game_date, game_tuple)

    for game_date, game_tuple in _supplemental_only_games(league, cutoff_date):
        _append(game_date, game_tuple)

    return merged


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
    dated = load_supplemental_dated_games_for_backtest(league, cutoff_date)
    return [game for _game_date, game in dated]


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
    if league in INTERNATIONAL_SOCCER_LEAGUES or league == "fifa_friendlies":
        sources.append("football-data.co.uk/international")
    if league in ("nba", "nhl", "mlb", "nfl"):
        sources.append("data/supplemental/closing-odds/")
    if league in ("nba", "nhl", "mlb"):
        sources.append(f"{league}/team_data CSV archives")
    return {"league": league, "sources": sources}
