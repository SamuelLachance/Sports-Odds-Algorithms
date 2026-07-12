"""Hockey win probability: PuckCast features + xG + goalie blend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

from sklearn.linear_model import LogisticRegression

from web.hockey_features import (
    FEATURE_NAMES,
    build_matchup_features,
    build_training_matrix,
)
from web.hockey_goalie import goalie_xg_multipliers
from web.hockey_xg import matchup_expected_goals, team_xg_rates_from_metrics
from web.bet_advisor import devig_two_way_probs
from web.league_profiles import LEAGUE_PROFILES
from web.market_decorrelation import decorrelate_binary
from web.season_games import (
    GameTuple,
    _event_date_iso,
    _load_league_game_map,
    load_league_completed_games,
)

HOCKEY_LEAGUES: tuple[str, ...] = tuple(
    key for key, profile in LEAGUE_PROFILES.items() if profile.get("category") == "hockey"
)

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3
MIN_LOGISTIC_SAMPLES = 30
HOME_ADVANTAGE = 0.15
HOME_OT_ADVANTAGE = 0.52
MAX_GOALS = 10
RECENT_WINDOW = 5
RECENT_BLEND = 0.42
LOGISTIC_BLEND = 0.65
POISSON_BLEND = 0.35


class GameProbabilities(NamedTuple):
    home_win: float
    away_win: float
    home_regulation: float
    away_regulation: float
    overtime: float


@dataclass
class TeamMetrics:
    team: str
    goals_for_pg: float
    goals_against_pg: float
    games_played: int = 0


def is_hockey_league(league: str) -> bool:
    league = league.lower()
    profile = LEAGUE_PROFILES.get(league)
    return profile is not None and profile["category"] == "hockey"


def load_league_dated_games(
    league: str,
    cutoff_date: str,
) -> list[tuple[str, GameTuple]]:
    """Completed games with ISO dates for walk-forward feature building."""
    merged = _load_league_game_map(league, cutoff_date, for_backtest=False)
    return [
        (_event_date_iso(event_date), game_tuple)
        for event_date, game_tuple in sorted(merged.values(), key=lambda item: item[0])
    ]


def _poisson_prob(expected: float, actual: int) -> float:
    if actual < 0:
        return 0.0
    return (math.exp(-expected) * (expected**actual)) / math.factorial(actual)


def calculate_win_probability(
    home_xg: float,
    away_xg: float,
    *,
    home_ot_advantage: float = HOME_OT_ADVANTAGE,
    max_goals: int = MAX_GOALS,
) -> GameProbabilities:
    """Poisson regulation + OT split."""
    home_reg_win = 0.0
    away_reg_win = 0.0
    tie_prob = 0.0

    for home_goals in range(max_goals + 1):
        home_prob = _poisson_prob(home_xg, home_goals)
        for away_goals in range(max_goals + 1):
            away_prob = _poisson_prob(away_xg, away_goals)
            combined = home_prob * away_prob
            if home_goals > away_goals:
                home_reg_win += combined
            elif away_goals > home_goals:
                away_reg_win += combined
            else:
                tie_prob += combined

    ot_home_win = tie_prob * home_ot_advantage
    ot_away_win = tie_prob * (1 - home_ot_advantage)

    grid_total = home_reg_win + away_reg_win + tie_prob
    if grid_total > 0.0 and abs(grid_total - 1.0) > 1e-9:
        scale = 1.0 / grid_total
        home_reg_win *= scale
        away_reg_win *= scale
        tie_prob *= scale
        ot_home_win = tie_prob * home_ot_advantage
        ot_away_win = tie_prob * (1 - home_ot_advantage)

    return GameProbabilities(
        home_win=round(home_reg_win + ot_home_win, 4),
        away_win=round(away_reg_win + ot_away_win, 4),
        home_regulation=round(home_reg_win, 4),
        away_regulation=round(away_reg_win, 4),
        overtime=round(tie_prob, 4),
    )


def calculate_expected_goals(
    home_team: TeamMetrics,
    away_team: TeamMetrics,
    *,
    home_advantage: float = HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Blend offense vs opponent defense with home ice (gmalbert Poisson layer)."""
    home_xg = (home_team.goals_for_pg + away_team.goals_against_pg) / 2
    home_xg *= 1 + home_advantage

    away_xg = (away_team.goals_for_pg + home_team.goals_against_pg) / 2
    away_xg *= 1 - home_advantage / 2

    return round(max(0.5, home_xg), 2), round(max(0.5, away_xg), 2)


def _build_team_metrics(games: list[tuple]) -> dict[str, TeamMetrics]:
    stats: dict[str, dict[str, float]] = {}
    recent: dict[str, list[tuple[int, int]]] = {}

    for home_abbr, away_abbr, _home_name, _away_name, home_score, away_score in games:
        for abbr, scored, allowed in (
            (home_abbr.lower(), home_score, away_score),
            (away_abbr.lower(), away_score, home_score),
        ):
            bucket = stats.setdefault(abbr, {"gf": 0.0, "ga": 0.0, "gp": 0.0})
            bucket["gf"] += scored
            bucket["ga"] += allowed
            bucket["gp"] += 1
            recent.setdefault(abbr, []).append((scored, allowed))

    metrics: dict[str, TeamMetrics] = {}
    for abbr, bucket in stats.items():
        gp = int(bucket["gp"])
        if gp <= 0:
            continue
        season_gf = bucket["gf"] / gp
        season_ga = bucket["ga"] / gp
        recent_games = recent.get(abbr, [])[-RECENT_WINDOW:]
        if recent_games:
            recent_gf = sum(s for s, _ in recent_games) / len(recent_games)
            recent_ga = sum(a for _, a in recent_games) / len(recent_games)
            goals_for_pg = (1 - RECENT_BLEND) * season_gf + RECENT_BLEND * recent_gf
            goals_against_pg = (1 - RECENT_BLEND) * season_ga + RECENT_BLEND * recent_ga
        else:
            goals_for_pg = season_gf
            goals_against_pg = season_ga
        metrics[abbr] = TeamMetrics(
            team=abbr,
            goals_for_pg=round(goals_for_pg, 3),
            goals_against_pg=round(goals_against_pg, 3),
            games_played=gp,
        )
    return metrics


def _build_xg_rate_map(
    league: str,
    team_metrics: dict[str, TeamMetrics],
    dated_games: list[tuple[str, GameTuple]],
) -> dict[str, tuple[float, float]]:
    rates: dict[str, tuple[float, float]] = {}
    for abbr, metrics in team_metrics.items():
        rates[abbr] = team_xg_rates_from_metrics(
            league,
            abbr,
            metrics,
            team_metrics=team_metrics,
            dated_games=dated_games,
        )
    return rates


def build_hockey_model(
    dated_games: list[tuple[str, GameTuple]],
    league: str,
    cutoff_date: str,
) -> dict[str, Any] | None:
    games = [game for _date, game in dated_games]
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    team_metrics = _build_team_metrics(games)
    if not team_metrics:
        return None

    x_rows, y_rows, feature_names = build_training_matrix(dated_games, cutoff_date, league)
    classifier: LogisticRegression | None = None
    if len(x_rows) >= MIN_LOGISTIC_SAMPLES and len(set(y_rows)) > 1:
        classifier = LogisticRegression(C=0.01, max_iter=2000)
        classifier.fit(x_rows, y_rows)

    xg_rates = _build_xg_rate_map(league, team_metrics, dated_games)
    team_game_counts = {abbr: m.games_played for abbr, m in team_metrics.items()}
    return {
        "league": league.lower(),
        "team_metrics": team_metrics,
        "team_game_counts": team_game_counts,
        "xg_rates": xg_rates,
        "classifier": classifier,
        "feature_names": feature_names,
        "cutoff_date": cutoff_date,
        "dated_games": dated_games,
        "games_sampled": len(games),
    }


def extract_team_xg_strengths(model: dict[str, Any]) -> dict[str, float]:
    """Net xG strength (xGF/GP − xGA/GP) per team."""
    xg_rates: dict[str, tuple[float, float]] = model.get("xg_rates") or {}
    if xg_rates:
        return {abbr: round(xgf - xga, 4) for abbr, (xgf, xga) in xg_rates.items()}
    metrics: dict[str, TeamMetrics] = model["team_metrics"]
    return {
        abbr: round(m.goals_for_pg - m.goals_against_pg, 4)
        for abbr, m in metrics.items()
    }


def _blend_home_win_probability(
    logistic_prob: float | None,
    poisson_prob: float,
) -> float:
    if logistic_prob is None:
        return poisson_prob
    return LOGISTIC_BLEND * logistic_prob + POISSON_BLEND * poisson_prob


def predict_matchup_from_model(
    model: dict[str, Any],
    home_abbr: str,
    away_abbr: str,
    *,
    use_goalie_edge: bool = True,
    game_date: str | None = None,
) -> dict[str, float | str] | None:
    home = home_abbr.lower()
    away = away_abbr.lower()
    metrics: dict[str, TeamMetrics] = model["team_metrics"]
    if home not in metrics or away not in metrics:
        return None

    league = str(model.get("league") or "nhl").lower()
    dated_games: list[tuple[str, GameTuple]] = model.get("dated_games") or []
    cutoff_date = str(model.get("cutoff_date") or "")
    xg_rates: dict[str, tuple[float, float]] = model.get("xg_rates") or {}

    home_xg, away_xg = matchup_expected_goals(
        home,
        away,
        league,
        metrics,
        xg_rates,
        dated_games=dated_games,
    )

    team_ga_rates = {abbr: m.goals_against_pg for abbr, m in metrics.items()}
    goalie_meta: dict[str, Any] = {}
    if use_goalie_edge:
        home_mult, away_mult, goalie_meta = goalie_xg_multipliers(
            league,
            home,
            away,
            game_date or cutoff_date,
            team_ga_rates=team_ga_rates,
        )
        home_xg = round(home_xg * home_mult, 2)
        away_xg = round(away_xg * away_mult, 2)

    poisson_probs = calculate_win_probability(home_xg, away_xg)
    poisson_home = poisson_probs.home_win

    logistic_home: float | None = None
    classifier: LogisticRegression | None = model.get("classifier")
    if classifier is not None and cutoff_date:
        features = build_matchup_features(dated_games, cutoff_date, home, away, league)
        vector = [[float(features.get(name, 0.0)) for name in FEATURE_NAMES]]
        logistic_home = float(classifier.predict_proba(vector)[0][1])

    blended_home = _blend_home_win_probability(logistic_home, poisson_home)
    home_win_prob = blended_home * 100.0
    away_win_prob = (1.0 - blended_home) * 100.0

    return {
        "home_key": home,
        "away_key": away,
        "expected_home_goals": home_xg,
        "expected_away_goals": away_xg,
        "expected_total_goals": round(home_xg + away_xg, 2),
        "home_win_probability": round(home_win_prob, 2),
        "away_win_probability": round(away_win_prob, 2),
        "overtime_probability": round(poisson_probs.overtime * 100.0, 2),
        "logistic_home_probability": round(logistic_home * 100.0, 2) if logistic_home is not None else None,
        "poisson_home_probability": round(poisson_home * 100.0, 2),
        "goalie_metadata": goalie_meta,
    }


def hockey_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    league = league.lower()
    dated_games = load_league_dated_games(league, cutoff_date)
    if len(dated_games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(dated_games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season or sparse schedule."
        )

    model = build_hockey_model(dated_games, league, cutoff_date)
    if not model:
        return "Could not build hockey PuckCast model on available games."

    counts = model["team_game_counts"]
    home = home_abbr.lower()
    away = away_abbr.lower()
    if home not in counts or away not in counts:
        missing = [k for k in (home, away) if k not in counts]
        return f"Teams not found in hockey model: {', '.join(missing)}."
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the hockey model sample."
    return "Hockey PuckCast model unavailable."


@lru_cache(maxsize=32)
def get_hockey_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_hockey_league(league):
        return None
    dated_games = load_league_dated_games(league, cutoff_date)
    return build_hockey_model(dated_games, league, cutoff_date)


def _cutoff_to_iso(cutoff_date: str) -> str | None:
    """Accepts MM-DD-YYYY (legacy cutoff) or YYYY-MM-DD; returns ISO date."""
    parts = str(cutoff_date).split("-")
    if len(parts) != 3:
        return None
    try:
        a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if a >= 1900:  # already ISO
        return f"{a:04d}-{b:02d}-{c:02d}"
    return f"{c:04d}-{a:02d}-{b:02d}"


def _run_nhl_v2(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
) -> dict[str, Any] | None:
    """Persisted NHL v2 (MoneyPuck xG feature engine + gradient boost ensemble)."""
    if league.lower() != "nhl":
        return None
    game_date = _cutoff_to_iso(cutoff_date)
    if not game_date:
        return None
    try:
        from web.nhl_v2.live import predict_matchup_v2

        v2 = predict_matchup_v2(
            game_date,
            home_abbr,
            away_abbr,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
        )
    except Exception:  # noqa: BLE001 - any v2 failure falls back to PuckCast
        return None
    if not v2:
        return None

    calibrated_prob = float(v2["home_win_probability"])  # 0-100, calibrated
    home_prob = calibrated_prob
    market_decorrelated = False
    # Skip post-hoc decorrelation when the market-aware head already used odds.
    if (
        v2.get("model_variant") != "market_aware"
        and home_moneyline is not None
        and away_moneyline is not None
    ):
        _away_mkt, market_home = devig_two_way_probs(away_moneyline, home_moneyline)
        if market_home is not None:
            home_prob = decorrelate_binary(calibrated_prob, market_home)
            market_decorrelated = True

    payload: dict[str, Any] = {
        "algorithm": "NHLGradientBoost",
        "model_version": "v2",
        "model_variant": v2.get("model_variant") or "pure",
        "source": "moneypuck-xgb-lr-isotonic",
        "market_decorrelated": market_decorrelated,
        "pre_decorrelation_home_win_probability": round(calibrated_prob, 2),
        "raw_home_win_probability": round(calibrated_prob, 2),
        "home_win_probability": round(home_prob, 2),
        "away_win_probability": round(100.0 - home_prob, 2),
        "expected_home_goals": v2.get("predicted_home_goals"),
        "expected_away_goals": v2.get("predicted_away_goals"),
        "predicted_margin": v2.get("predicted_margin"),
        "home_games": v2.get("home_games"),
        "away_games": v2.get("away_games"),
        "home_elo": v2.get("home_elo"),
        "away_elo": v2.get("away_elo"),
        "home_xgf_pg": v2.get("home_xgf_pg"),
        "home_xga_pg": v2.get("home_xga_pg"),
        "away_xgf_pg": v2.get("away_xgf_pg"),
        "away_xga_pg": v2.get("away_xga_pg"),
        "home_goalie": v2.get("home_goalie"),
        "away_goalie": v2.get("away_goalie"),
        "home_goalie_confirmed": v2.get("home_goalie_confirmed"),
        "away_goalie_confirmed": v2.get("away_goalie_confirmed"),
        "home_goalie_gsax100": v2.get("home_goalie_gsax100"),
        "away_goalie_gsax100": v2.get("away_goalie_gsax100"),
        "home_rest_days": v2.get("home_rest_days"),
        "away_rest_days": v2.get("away_rest_days"),
        "home_b2b": v2.get("home_b2b"),
        "away_b2b": v2.get("away_b2b"),
        "features_used": v2.get("features_used"),
        "has_market": v2.get("has_market"),
    }
    total_goals = None
    if v2.get("predicted_home_goals") is not None and v2.get("predicted_away_goals") is not None:
        total_goals = round(
            float(v2["predicted_home_goals"]) + float(v2["predicted_away_goals"]), 2
        )
    payload["expected_total_goals"] = total_goals
    return payload


def run_hockey_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
) -> dict[str, Any] | None:
    """Run NHL v2 when artifacts exist; PuckCast otherwise (college, fallback)."""
    v2_payload = _run_nhl_v2(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
    )
    if v2_payload is not None:
        return v2_payload

    context = get_hockey_pred_context(league, cutoff_date)
    if not context:
        return None

    prediction = predict_matchup_from_model(
        context,
        home_abbr,
        away_abbr,
        use_goalie_edge=True,
        game_date=cutoff_date,
    )
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    home_prob = float(prediction["home_win_probability"])
    market_decorrelated = False
    if home_moneyline is not None and away_moneyline is not None:
        _away, market_home = devig_two_way_probs(away_moneyline, home_moneyline)
        if market_home is not None:
            home_prob = decorrelate_binary(home_prob, market_home)
            market_decorrelated = True

    away_prob = round(100.0 - home_prob, 2)

    return {
        "algorithm": "HockeyPuckCast",
        "source": "puckcast-xg-goalie",
        "home_win_probability": round(home_prob, 2),
        "away_win_probability": away_prob,
        "expected_home_goals": prediction["expected_home_goals"],
        "expected_away_goals": prediction["expected_away_goals"],
        "expected_total_goals": prediction["expected_total_goals"],
        "overtime_probability": prediction["overtime_probability"],
        "home_games": home_games,
        "away_games": away_games,
        "market_decorrelated": market_decorrelated,
    }
