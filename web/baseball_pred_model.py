"""MLB-Model style Elo + Pythagorean + recent-form baseball win probability."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from web.league_profiles import LEAGUE_PROFILES
from web.season_games import load_league_completed_games

BASEBALL_LEAGUES: tuple[str, ...] = ("mlb",)

# MLB-Model powerrankings.py: Elo K=32 (fast) and K=16 (slow).
ELO_K_FAST = 32.0
ELO_K_SLOW = 16.0
ELO_START = 1500.0
HOME_ELO_ADV = 24.0  # ~54% home win at parity
PYTH_EXPONENT = 1.83
RECENT_WINDOW = 10

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3

# Blend weights for composite margin (Elo primary signal in MLB-Model).
ELO_MARGIN_WEIGHT = 10.0
PYTH_MARGIN_WEIGHT = 6.0
FORM_MARGIN_WEIGHT = 4.0
HOME_MARGIN_ADV = 0.35


def is_baseball_league(league: str) -> bool:
    league = league.lower()
    profile = LEAGUE_PROFILES.get(league)
    return profile is not None and profile["category"] == "baseball"


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

    def update(self, opponent: EloTeam, won: bool) -> None:
        for rating_attr, k in (("rating_fast", ELO_K_FAST), ("rating_slow", ELO_K_SLOW)):
            self_rating = getattr(self, rating_attr)
            opp_rating = getattr(opponent, rating_attr)
            diff = self_rating - opp_rating
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

    def record_game(self, scored: int, allowed: int) -> None:
        self.runs_scored += scored
        self.runs_allowed += allowed
        self.recent.append((scored, allowed))


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


def _matchup_margin(
    home: EloTeam,
    away: EloTeam,
    home_state: TeamState,
    away_state: TeamState,
    *,
    home_field: bool,
) -> tuple[float, float, float, float]:
    home_adv = HOME_ELO_ADV if home_field else 0.0
    elo_fast = home.expected_fast(away, home_adv=home_adv)
    elo_slow = home.expected_slow(away, home_adv=home_adv)
    elo_exp = 0.6 * elo_fast + 0.4 * elo_slow

    home_pyth = home_state.pythagorean_win_pct()
    away_pyth = away_state.pythagorean_win_pct()
    pyth_diff = home_pyth - away_pyth

    form_diff = home_state.recent_run_diff_avg() - away_state.recent_run_diff_avg()

    margin = (
        ELO_MARGIN_WEIGHT * (elo_exp - 0.5)
        + PYTH_MARGIN_WEIGHT * pyth_diff
        + FORM_MARGIN_WEIGHT * (form_diff / 3.0)
    )
    if home_field:
        margin += HOME_MARGIN_ADV

    return margin, elo_exp, home_pyth, form_diff


def build_baseball_model(
    games: list[tuple[str, str, str, str, int, int]],
    league: str,
) -> dict[str, Any] | None:
    """Build Elo/Pythagorean state and logistic margin param from completed games."""
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    team_keys = sorted(
        {home for home, away, *_ in games} | {away for home, away, *_ in games}
    )
    if len(team_keys) < 4:
        return None

    elos: dict[str, EloTeam] = {key: EloTeam() for key in team_keys}
    states: dict[str, TeamState] = {key: TeamState() for key in team_keys}
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}

    margins: list[float] = []
    outcomes: list[float] = []

    for home_key, away_key, _hn, _an, home_score, away_score in games:
        if home_key not in elos or away_key not in elos:
            continue

        home_elo = elos[home_key]
        away_elo = elos[away_key]
        home_state = states[home_key]
        away_state = states[away_key]

        margin, _elo, _pyth, _form = _matchup_margin(
            home_elo, away_elo, home_state, away_state, home_field=True
        )
        margins.append(margin)
        if home_score > away_score:
            outcomes.append(1.0)
        elif home_score < away_score:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)

        home_won = home_score > away_score
        home_elo.update(away_elo, home_won)
        home_state.record_game(home_score, away_score)
        away_state.record_game(away_score, home_score)
        team_game_counts[home_key] += 1
        team_game_counts[away_key] += 1

    param = _fit_margin_param(margins, outcomes)

    return {
        "elos": elos,
        "states": states,
        "param": param,
        "team_game_counts": team_game_counts,
        "league": league,
    }


def predict_matchup_from_model(
    model: dict[str, Any],
    home_key: str,
    away_key: str,
) -> dict[str, float | str] | None:
    home = home_key.lower()
    away = away_key.lower()
    elos: dict[str, EloTeam] = model["elos"]
    states: dict[str, TeamState] = model["states"]
    if home not in elos or away not in elos:
        return None

    margin, elo_exp, home_pyth, form_diff = _matchup_margin(
        elos[home], elos[away], states[home], states[away], home_field=True
    )
    param = float(model["param"])
    prob = 1.0 / (1.0 + math.exp(-margin / param))
    home_win_prob = prob * 100.0

    away_pyth = states[away].pythagorean_win_pct()
    predicted_home_runs = 4.5 + margin * 0.15
    predicted_away_runs = 4.5 - margin * 0.15

    return {
        "home_key": home,
        "away_key": away,
        "home_win_probability": round(home_win_prob, 2),
        "elo_exp": round(elo_exp * 100.0, 2),
        "home_pythagorean": round(home_pyth * 100.0, 2),
        "away_pythagorean": round(away_pyth * 100.0, 2),
        "form_diff": round(form_diff, 2),
        "predicted_margin": round(margin, 2),
        "predicted_home_runs": round(predicted_home_runs, 1),
        "predicted_away_runs": round(predicted_away_runs, 1),
        "param": round(param, 3),
    }


def baseball_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    """Human-readable reason when baseball_pred blend layer cannot run."""
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


def run_baseball_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Run MLB-Model style Elo/Pythagorean model for a baseball matchup."""
    context = get_baseball_pred_context(league, cutoff_date)
    if not context:
        return None

    prediction = predict_matchup_from_model(context, home_abbr, away_abbr)
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    return {
        "algorithm": "BaseballElo",
        "source": "MLB-Model",
        "home_win_probability": prediction["home_win_probability"],
        "elo_exp": prediction["elo_exp"],
        "home_pythagorean": prediction["home_pythagorean"],
        "away_pythagorean": prediction["away_pythagorean"],
        "form_diff": prediction["form_diff"],
        "predicted_margin": prediction["predicted_margin"],
        "predicted_home_runs": prediction["predicted_home_runs"],
        "predicted_away_runs": prediction["predicted_away_runs"],
        "param": prediction["param"],
        "home_games": home_games,
        "away_games": away_games,
    }
