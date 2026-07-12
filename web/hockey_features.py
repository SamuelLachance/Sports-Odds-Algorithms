"""PuckCast-inspired walk-forward features for hockey win probability."""

from __future__ import annotations

from datetime import date
from typing import Any

from web.live_data import _parse_cutoff

GameTuple = tuple[str, str, str, str, int, int]
DatedGame = tuple[str, GameTuple]

LEAGUE_PARAMS: dict[str, dict[str, int | float]] = {
    "nhl": {"short": 5, "mid": 10, "min_team_games": 3},
    "ncaah": {"short": 5, "mid": 10, "min_team_games": 3},
    "ncaawh": {"short": 5, "mid": 8, "min_team_games": 3},
}

DEFAULT_PARAMS = LEAGUE_PARAMS["nhl"]

FEATURE_NAMES: tuple[str, ...] = (
    "home_gf_pg_5",
    "home_gf_pg_10",
    "home_gf_pg_season",
    "home_ga_pg_5",
    "home_ga_pg_10",
    "home_ga_pg_season",
    "home_gd_5",
    "home_gd_10",
    "home_gd_season",
    "home_gf_pg_home",
    "home_ga_pg_home",
    "home_gf_pg_away",
    "home_ga_pg_away",
    "home_rest_days",
    "home_b2b",
    "home_win_rate_5",
    "home_win_rate_10",
    "home_form_5",
    "away_gf_pg_5",
    "away_gf_pg_10",
    "away_gf_pg_season",
    "away_ga_pg_5",
    "away_ga_pg_10",
    "away_ga_pg_season",
    "away_gd_5",
    "away_gd_10",
    "away_gd_season",
    "away_gf_pg_home",
    "away_ga_pg_home",
    "away_gf_pg_away",
    "away_ga_pg_away",
    "away_rest_days",
    "away_b2b",
    "away_win_rate_5",
    "away_win_rate_10",
    "away_form_5",
    "matchup_gf_edge",
    "matchup_ga_edge",
    "rest_diff",
)


def _league_params(league: str) -> dict[str, int | float]:
    return LEAGUE_PARAMS.get(league.lower(), DEFAULT_PARAMS)


def _parse_game_date(raw: str) -> date:
    text = str(raw or "").strip()[:10]
    if not text:
        return date.today()
    return date.fromisoformat(text)


def _cutoff_as_date(cutoff_date: str) -> date:
    try:
        return _parse_cutoff(cutoff_date).date()
    except (ValueError, AttributeError, TypeError):
        return _parse_game_date(cutoff_date)


def _filter_before_cutoff(
    dated_games: list[DatedGame],
    cutoff_date: str,
) -> list[DatedGame]:
    cutoff = _cutoff_as_date(cutoff_date)
    return [
        (game_date, game)
        for game_date, game in dated_games
        if _parse_game_date(game_date) < cutoff
    ]


def _team_records(
    dated_games: list[DatedGame],
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for game_date, (home, away, _hn, _an, home_score, away_score) in dated_games:
        day = _parse_game_date(game_date)
        home_key = home.lower()
        away_key = away.lower()
        records.setdefault(home_key, []).append(
            {
                "date": day,
                "is_home": True,
                "gf": home_score,
                "ga": away_score,
                "won": home_score > away_score,
            }
        )
        records.setdefault(away_key, []).append(
            {
                "date": day,
                "is_home": False,
                "gf": away_score,
                "ga": home_score,
                "won": away_score > home_score,
            }
        )
    for games in records.values():
        games.sort(key=lambda row: row["date"])
    return records


def _rolling_rates(
    games: list[dict[str, Any]],
    window: int | None,
) -> tuple[float, float, float]:
    subset = games[-window:] if window else games
    if not subset:
        return 0.0, 0.0, 0.0
    gp = len(subset)
    gf = sum(row["gf"] for row in subset) / gp
    ga = sum(row["ga"] for row in subset) / gp
    return gf, ga, gf - ga


def _split_rates(
    games: list[dict[str, Any]],
    *,
    is_home: bool,
) -> tuple[float, float]:
    subset = [row for row in games if row["is_home"] == is_home]
    if not subset:
        return 0.0, 0.0
    gp = len(subset)
    return sum(row["gf"] for row in subset) / gp, sum(row["ga"] for row in subset) / gp


def _win_rate(games: list[dict[str, Any]], window: int) -> float:
    subset = games[-window:]
    if not subset:
        return 0.5
    return sum(1 for row in subset if row["won"]) / len(subset)


def _form_score(games: list[dict[str, Any]], window: int) -> float:
    subset = games[-window:]
    if not subset:
        return 0.0
    return sum(2 if row["won"] else 0 for row in subset) / len(subset)


def _rest_days(games: list[dict[str, Any]], as_of: date) -> tuple[float, float]:
    if not games:
        return 3.0, 0.0
    last = games[-1]["date"]
    days = max((as_of - last).days, 0)
    b2b = 1.0 if days <= 1 else 0.0
    return float(days), b2b


def _team_feature_block(
    prefix: str,
    games: list[dict[str, Any]],
    *,
    as_of: date,
    short: int,
    mid: int,
) -> dict[str, float]:
    gf5, ga5, gd5 = _rolling_rates(games, short)
    gf10, ga10, gd10 = _rolling_rates(games, mid)
    gf_s, ga_s, gd_s = _rolling_rates(games, None)
    gf_home, ga_home = _split_rates(games, is_home=True)
    gf_away, ga_away = _split_rates(games, is_home=False)
    rest, b2b = _rest_days(games, as_of)
    return {
        f"{prefix}_gf_pg_5": gf5,
        f"{prefix}_gf_pg_10": gf10,
        f"{prefix}_gf_pg_season": gf_s,
        f"{prefix}_ga_pg_5": ga5,
        f"{prefix}_ga_pg_10": ga10,
        f"{prefix}_ga_pg_season": ga_s,
        f"{prefix}_gd_5": gd5,
        f"{prefix}_gd_10": gd10,
        f"{prefix}_gd_season": gd_s,
        f"{prefix}_gf_pg_home": gf_home,
        f"{prefix}_ga_pg_home": ga_home,
        f"{prefix}_gf_pg_away": gf_away,
        f"{prefix}_ga_pg_away": ga_away,
        f"{prefix}_rest_days": rest,
        f"{prefix}_b2b": b2b,
        f"{prefix}_win_rate_5": _win_rate(games, short),
        f"{prefix}_win_rate_10": _win_rate(games, mid),
        f"{prefix}_form_5": _form_score(games, short),
    }


def build_matchup_features(
    dated_games: list[DatedGame],
    cutoff_date: str,
    home: str,
    away: str,
    league: str,
    *,
    history_games: list[DatedGame] | None = None,
) -> dict[str, float]:
    """Leak-free features for a single matchup as of cutoff_date."""
    params = _league_params(league)
    short = int(params["short"])
    mid = int(params["mid"])
    as_of = _cutoff_as_date(cutoff_date)
    prior = history_games if history_games is not None else _filter_before_cutoff(dated_games, cutoff_date)
    records = _team_records(prior)
    home_games = records.get(home.lower(), [])
    away_games = records.get(away.lower(), [])

    features: dict[str, float] = {}
    features.update(_team_feature_block("home", home_games, as_of=as_of, short=short, mid=mid))
    features.update(_team_feature_block("away", away_games, as_of=as_of, short=short, mid=mid))

    home_gf_s = features.get("home_gf_pg_season", 0.0)
    home_ga_s = features.get("home_ga_pg_season", 0.0)
    away_gf_s = features.get("away_gf_pg_season", 0.0)
    away_ga_s = features.get("away_ga_pg_season", 0.0)
    features["matchup_gf_edge"] = home_gf_s - away_gf_s
    features["matchup_ga_edge"] = away_ga_s - home_ga_s
    features["rest_diff"] = features.get("home_rest_days", 0.0) - features.get("away_rest_days", 0.0)
    return features


def _features_to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def build_training_matrix(
    dated_games: list[DatedGame],
    cutoff_date: str,
    league: str,
) -> tuple[list[list[float]], list[int], tuple[str, ...]]:
    """Walk-forward training rows using only games strictly before each sample."""
    params = _league_params(league)
    min_team = int(params["min_team_games"])
    prior = _filter_before_cutoff(dated_games, cutoff_date)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []

    for index in range(len(prior)):
        game_date, (home, away, _hn, _an, home_score, away_score) = prior[index]
        train_slice = prior[:index]
        if len(train_slice) < min_team:
            continue
        records = _team_records(train_slice)
        home_hist = records.get(home.lower(), [])
        away_hist = records.get(away.lower(), [])
        if len(home_hist) < min_team or len(away_hist) < min_team:
            continue
        feats = build_matchup_features(
            prior,
            game_date,
            home,
            away,
            league,
            history_games=train_slice,
        )
        if home_score == away_score:
            # College hockey can tie; do not train ties as away wins.
            continue
        x_rows.append(_features_to_vector(feats))
        y_rows.append(1 if home_score > away_score else 0)

    return x_rows, y_rows, FEATURE_NAMES
