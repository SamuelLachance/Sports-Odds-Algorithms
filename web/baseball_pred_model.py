"""SharpBaseball: multi-source Elo + Pythagorean + form + MLB Stats API enrichment."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.espn_client import current_season_year
from web.league_profiles import LEAGUE_PROFILES
from web.season_games import load_league_completed_games

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARP_CONFIG_PATH = PROJECT_ROOT / "data" / "baseball_sharp_config.json"

# Dual-track Elo (MLB-Model heritage).
ELO_K_FAST = 32.0
ELO_K_SLOW = 16.0
ELO_START = 1500.0
PYTH_EXPONENT = 1.83
RECENT_WINDOW = 10

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3

# Legacy blend weights for composite margin (used when sharp config missing).
ELO_MARGIN_WEIGHT = 10.0
PYTH_MARGIN_WEIGHT = 6.0
FORM_MARGIN_WEIGHT = 4.0
HOME_MARGIN_ADV = 0.35
BASEBALL_PROB_SHRINK = 0.86

WINTER_LEAGUES = frozenset({"dwl", "pwl", "vwl", "lmp", "wbc"})


def is_baseball_league(league: str) -> bool:
    league = league.lower()
    profile = LEAGUE_PROFILES.get(league)
    return profile is not None and profile["category"] == "baseball"


@lru_cache(maxsize=1)
def _load_sharp_config() -> dict[str, Any]:
    if not SHARP_CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(SHARP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _league_sharp_config(league: str) -> dict[str, Any]:
    config = _load_sharp_config()
    league = league.lower()
    if league in config:
        return config[league]
    if league in WINTER_LEAGUES:
        return config.get("default_winter", {})
    return config.get("mlb", {})


@dataclass
class EloTeam:
    rating_fast: float = ELO_START
    rating_slow: float = ELO_START

    def expected_fast(self, opponent: EloTeam, *, home_adv: float = 0.0) -> float:
        diff = (self.rating_fast + home_adv) - opponent.rating_fast
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def expected_slow(self, opponent: EloTeam, *, home_adv: float = 0.0) -> float:
        diff = (self.rating_slow + home_adv) - opponent.rating_slow
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, opponent: EloTeam, won: bool, *, home_adv: float = 0.0) -> None:
        for rating_attr, k in (("rating_fast", ELO_K_FAST), ("rating_slow", ELO_K_SLOW)):
            self_rating = getattr(self, rating_attr)
            opp_rating = getattr(opponent, rating_attr)
            diff = self_rating + home_adv - opp_rating
            expected = 1.0 / (1.0 + 10 ** (-diff / 400.0))
            score = 1.0 if won else 0.0
            delta = k * (score - expected)
            setattr(self, rating_attr, self_rating + delta)
            setattr(opponent, rating_attr, opp_rating - delta)


@dataclass
class TeamState:
    runs_scored: int = 0
    runs_allowed: int = 0
    recent: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    last_game_index: int = -1

    def pythagorean_win_pct(self) -> float:
        if self.runs_scored <= 0 and self.runs_allowed <= 0:
            return 0.5
        rs = max(float(self.runs_scored), 0.5)
        ra = max(float(self.runs_allowed), 0.5)
        rs_exp = rs**PYTH_EXPONENT
        ra_exp = ra**PYTH_EXPONENT
        return rs_exp / (rs_exp + ra_exp)

    def recent_run_diff_avg(self) -> float:
        if not self.recent:
            return 0.0
        return sum(h - a for h, a in self.recent) / len(self.recent)

    def record_game(self, scored: int, allowed: int, *, game_index: int) -> None:
        self.runs_scored += scored
        self.runs_allowed += allowed
        self.recent.append((scored, allowed))
        self.last_game_index = game_index


def _fit_margin_param(margins: list[float], outcomes: list[float]) -> float:
    if len(margins) < 5:
        return 8.0

    best_param = 8.0
    best_loss = float("inf")
    for trial in range(20, 200):
        param = trial / 10.0
        loss = 0.0
        count = 0
        for margin, outcome in zip(margins, outcomes):
            prob = 1.0 / (1.0 + math.exp(-margin / param))
            prob = min(max(prob, 1e-9), 1.0 - 1e-9)
            loss += -(outcome * math.log(prob) + (1.0 - outcome) * math.log(1.0 - prob))
            count += 1
        loss /= count
        if loss < best_loss:
            best_loss = loss
            best_param = param
    return best_param


def _rest_days_advantage(
    home_state: TeamState,
    away_state: TeamState,
    current_index: int,
) -> float:
    """Favor rested home team when away is on short rest."""
    if home_state.last_game_index < 0 or away_state.last_game_index < 0:
        return 0.0
    home_rest = current_index - home_state.last_game_index
    away_rest = current_index - away_state.last_game_index
    if away_rest <= 1 and home_rest >= 2:
        return 0.25
    if home_rest <= 1 and away_rest >= 2:
        return -0.25
    return 0.0


def _matchup_signals(
    home: EloTeam,
    away: EloTeam,
    home_state: TeamState,
    away_state: TeamState,
    *,
    home_elo_adv: float,
    game_index: int,
) -> dict[str, float]:
    elo_fast = home.expected_fast(away, home_adv=home_elo_adv)
    elo_slow = home.expected_slow(away, home_adv=home_elo_adv)
    elo_exp = 0.6 * elo_fast + 0.4 * elo_slow
    elo_signal = elo_exp - 0.5

    home_pyth = home_state.pythagorean_win_pct()
    away_pyth = away_state.pythagorean_win_pct()
    pyth_signal = home_pyth - away_pyth

    form_signal = (home_state.recent_run_diff_avg() - away_state.recent_run_diff_avg()) / 3.0
    rest_signal = _rest_days_advantage(home_state, away_state, game_index)

    return {
        "elo": elo_signal,
        "pyth": pyth_signal,
        "form": form_signal,
        "rest": rest_signal,
        "elo_exp": elo_exp,
        "home_pyth": home_pyth,
        "away_pyth": away_pyth,
    }


def _weighted_margin(
    signals: dict[str, float],
    *,
    league: str,
    home_abbr: str,
    away_abbr: str,
    season: int,
    use_api: bool = True,
) -> tuple[float, dict[str, float]]:
    cfg = _league_sharp_config(league)
    weights = cfg.get("signal_weights") or {
        "elo": 0.35,
        "pyth": 0.25,
        "form": 0.15,
        "mlb_api": 0.2,
        "rest": 0.05,
    }
    home_adv = float(cfg.get("home_elo_adv", 24.0)) / 70.0

    mlb_api_signal = 0.0
    if use_api and cfg.get("use_mlb_stats_api") and league == "mlb":
        from web.mlb_stats_api import mlb_api_matchup_edge

        edge = mlb_api_matchup_edge(home_abbr, away_abbr, season)
        if edge is not None:
            mlb_api_signal = edge / 6.0

    components = {
        "elo": ELO_MARGIN_WEIGHT * signals["elo"],
        "pyth": PYTH_MARGIN_WEIGHT * signals["pyth"],
        "form": FORM_MARGIN_WEIGHT * signals["form"],
        "rest": FORM_MARGIN_WEIGHT * signals["rest"],
        "mlb_api": 5.0 * mlb_api_signal,
    }

    margin = (
        weights.get("elo", 0.35) * components["elo"]
        + weights.get("pyth", 0.25) * components["pyth"]
        + weights.get("form", 0.15) * components["form"]
        + weights.get("rest", 0.05) * components["rest"]
        + weights.get("mlb_api", 0.0) * components["mlb_api"]
        + home_adv
    )
    return margin, components


def build_baseball_model(
    games: list[tuple[str, str, str, str, int, int]],
    league: str,
) -> dict[str, Any] | None:
    """Build SharpBaseball state from completed ESPN games."""
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    team_keys = sorted(
        {home for home, away, *_ in games} | {away for home, away, *_ in games}
    )
    if len(team_keys) < 4:
        return None

    league = league.lower()
    cfg = _league_sharp_config(league)
    home_elo_adv = float(cfg.get("home_elo_adv", 24.0))

    elos: dict[str, EloTeam] = {key: EloTeam() for key in team_keys}
    states: dict[str, TeamState] = {key: TeamState() for key in team_keys}
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}

    margins: list[float] = []
    outcomes: list[float] = []

    for index, (home_key, away_key, _hn, _an, home_score, away_score) in enumerate(games):
        if home_key not in elos or away_key not in elos:
            continue

        home_elo = elos[home_key]
        away_elo = elos[away_key]
        home_state = states[home_key]
        away_state = states[away_key]

        signals = _matchup_signals(
            home_elo, away_elo, home_state, away_state,
            home_elo_adv=home_elo_adv, game_index=index,
        )
        margin, _components = _weighted_margin(
            signals,
            league=league,
            home_abbr=home_key,
            away_abbr=away_key,
            season=2025,
            use_api=False,
        )
        margins.append(margin)
        if home_score > away_score:
            outcomes.append(1.0)
        elif home_score < away_score:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)

        home_won = home_score > away_score
        home_elo.update(away_elo, home_won, home_adv=home_elo_adv)
        away_elo.update(home_elo, not home_won, home_adv=-home_elo_adv)
        home_state.record_game(home_score, away_score, game_index=index)
        away_state.record_game(away_score, home_score, game_index=index)
        team_game_counts[home_key] += 1
        team_game_counts[away_key] += 1

    param = _fit_margin_param(margins, outcomes)

    return {
        "elos": elos,
        "states": states,
        "param": param,
        "team_game_counts": team_game_counts,
        "league": league,
        "home_elo_adv": home_elo_adv,
    }


def extract_team_elo_strengths(model: dict[str, Any]) -> dict[str, float]:
    elos: dict[str, EloTeam] = model["elos"]
    states: dict[str, TeamState] = model["states"]
    strengths: dict[str, float] = {}
    for key, elo in elos.items():
        state = states.get(key)
        pyth = state.pythagorean_win_pct() if state else 0.5
        strengths[key] = 0.6 * elo.rating_fast + 0.4 * (pyth * 400.0 + 1100.0)
    return strengths


def predict_matchup_from_model(
    model: dict[str, Any],
    home_key: str,
    away_key: str,
    *,
    season: int | None = None,
) -> dict[str, float | str] | None:
    home = home_key.lower()
    away = away_key.lower()
    elos: dict[str, EloTeam] = model["elos"]
    states: dict[str, TeamState] = model["states"]
    if home not in elos or away not in elos:
        return None

    league = str(model.get("league") or "mlb")
    cfg = _league_sharp_config(league)
    shrink = float(cfg.get("prob_shrink", BASEBALL_PROB_SHRINK))
    home_elo_adv = float(model.get("home_elo_adv", cfg.get("home_elo_adv", 24.0)))

    game_index = max(
        states[home].last_game_index,
        states[away].last_game_index,
        0,
    ) + 1
    signals = _matchup_signals(
        elos[home], elos[away], states[home], states[away],
        home_elo_adv=home_elo_adv, game_index=game_index,
    )
    margin, components = _weighted_margin(
        signals,
        league=league,
        home_abbr=home,
        away_abbr=away,
        season=season or 2025,
    )
    param = float(model["param"])
    prob = 1.0 / (1.0 + math.exp(-margin / param))
    home_win_prob = 50.0 + (prob * 100.0 - 50.0) * shrink

    predicted_home_runs = 4.5 + margin * 0.15
    predicted_away_runs = 4.5 - margin * 0.15

    return {
        "home_key": home,
        "away_key": away,
        "home_win_probability": round(home_win_prob, 2),
        "elo_exp": round(signals["elo_exp"] * 100.0, 2),
        "home_pythagorean": round(signals["home_pyth"] * 100.0, 2),
        "away_pythagorean": round(signals["away_pyth"] * 100.0, 2),
        "form_diff": round(signals["form"] * 3.0, 2),
        "predicted_margin": round(margin, 2),
        "predicted_home_runs": round(predicted_home_runs, 1),
        "predicted_away_runs": round(predicted_away_runs, 1),
        "param": round(param, 3),
        "mlb_api_component": round(components.get("mlb_api", 0.0), 3),
        "rest_component": round(components.get("rest", 0.0), 3),
    }


def baseball_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    league = league.lower()
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season or sparse schedule."
        )

    model = build_baseball_model(games, league)
    if not model:
        return "Could not build baseball rating model on available games."

    home = home_abbr.lower()
    away = away_abbr.lower()
    counts = model["team_game_counts"]
    if home not in counts or away not in counts:
        missing = [k for k in (home, away) if k not in counts]
        return f"Teams not found in baseball model: {', '.join(missing)}."
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the baseball model sample."
    return "Baseball model unavailable."


@lru_cache(maxsize=32)
def get_baseball_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_baseball_league(league):
        return None
    games = load_league_completed_games(league, cutoff_date)
    return build_baseball_model(games, league)


def _season_from_cutoff(cutoff_date: str, league: str) -> int:
    from datetime import date

    parts = cutoff_date.split("-")
    if len(parts) == 3:
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
        return current_season_year(league, date(year, month, day))
    return date.today().year


def run_baseball_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """SharpBaseball: ESPN walk-forward + MLB Stats API season enrichment."""
    context = get_baseball_pred_context(league, cutoff_date)
    if not context:
        return None

    season = _season_from_cutoff(cutoff_date, league)
    prediction = predict_matchup_from_model(
        context, home_abbr, away_abbr, season=season
    )
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    sources = ["ESPN", "SharpBaseball-Elo"]
    if league == "mlb":
        sources.append("MLB-StatsAPI")

    return {
        "algorithm": "SharpBaseball",
        "source": "+".join(sources),
        "home_win_probability": prediction["home_win_probability"],
        "elo_exp": prediction["elo_exp"],
        "home_pythagorean": prediction["home_pythagorean"],
        "away_pythagorean": prediction["away_pythagorean"],
        "form_diff": prediction["form_diff"],
        "predicted_margin": prediction["predicted_margin"],
        "predicted_home_runs": prediction["predicted_home_runs"],
        "predicted_away_runs": prediction["predicted_away_runs"],
        "param": prediction["param"],
        "mlb_api_component": prediction.get("mlb_api_component", 0.0),
        "rest_component": prediction.get("rest_component", 0.0),
        "home_games": home_games,
        "away_games": away_games,
    }
