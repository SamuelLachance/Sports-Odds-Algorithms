"""Football-predictor style Elo + Pi-ratings + Dixon-Coles 1X2 probabilities.

Lightweight port of https://github.com/jdgoated1/football-predictor rating core
(Elo, Pi-rating, Dixon-Coles). XGBoost/LightGBM meta-learner omitted for static
GH Pages build — equal blend of the three statistical layers only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from web.league_profiles import SOCCER_DRAW_BASE, is_soccer_league
from web.live_data import resolve_team
from web.season_games import load_league_completed_games

MIN_LEAGUE_GAMES = 15
MIN_TEAM_GAMES = 2
MAX_GOALS = 6

# football-predictor defaults
ELO_HOME_ADV = 65.0
ELO_INIT = 1500.0
ELO_K = 25.0
PI_LAM = 0.054
PI_GAMMA = 0.79
PI_B = 10.0
PI_C = 3.0
DC_HOME_ADV = 0.25
DC_RHO = -0.05
DC_XI = 0.0019  # time decay per day


@dataclass
class SoccerPredResult:
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    expected_home_goals: float
    expected_away_goals: float
    elo_home: float
    elo_away: float
    pi_expected_gd: float
    source: str = "football-predictor"
    algorithm: str = "SoccerRatings"


GameTuple = tuple[str, str, str, str, int, int]


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(lam, 1e-9)
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _gd_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def _pi_rating_to_goals(rating: float) -> float:
    return math.copysign(PI_B ** (abs(rating) / PI_C) - 1, rating)


def _normalize_threeway(
    home: float, draw: float, away: float
) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return 33.33, 33.33, 33.34
    scale = 100.0 / total
    return home * scale, draw * scale, away * scale


def _draw_from_closeness(home_win: float, league: str) -> tuple[float, float, float]:
    """Split binary home win % into 1X2 using league draw base (Algo V2 style)."""
    home_binary = min(max(home_win, 0.1), 99.9)
    away_binary = 100.0 - home_binary
    base_draw = SOCCER_DRAW_BASE.get(league.lower(), SOCCER_DRAW_BASE["default"])
    closeness = 1.0 - abs(home_binary - 50.0) / 50.0
    draw = min(35.0, max(18.0, base_draw + closeness * 8.0))
    scale = (100.0 - draw) / 100.0
    return _normalize_threeway(home_binary * scale, draw, away_binary * scale)


class _EloEngine:
    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, ELO_INIT)

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        ra = self.rating(home) + ELO_HOME_ADV
        rb = self.rating(away)
        expected = _elo_expected(ra, rb)
        if hg > ag:
            score = 1.0
        elif hg < ag:
            score = 0.0
        else:
            score = 0.5
        delta = ELO_K * _gd_multiplier(hg - ag) * (score - expected)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta


class _PiRating:
    def __init__(self) -> None:
        self.home_rating: dict[str, float] = {}
        self.away_rating: dict[str, float] = {}

    def get_h(self, team: str) -> float:
        return self.home_rating.get(team, 0.0)

    def get_a(self, team: str) -> float:
        return self.away_rating.get(team, 0.0)

    def expected_gd(self, home: str, away: str) -> float:
        lam_h = _pi_rating_to_goals(self.get_h(home))
        lam_a = _pi_rating_to_goals(self.get_a(away))
        return lam_h - lam_a

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        expected = self.expected_gd(home, away)
        actual = hg - ag
        sat = math.copysign(math.log1p(abs(actual)), actual)
        exp_sat = math.copysign(math.log1p(abs(expected)), expected)
        err = sat - exp_sat
        self.home_rating[home] = self.get_h(home) + PI_LAM * err
        self.away_rating[away] = self.get_a(away) - PI_LAM * err
        self.away_rating[home] = self.get_a(home) + PI_LAM * PI_GAMMA * err
        self.home_rating[away] = self.get_h(away) - PI_LAM * PI_GAMMA * err


def _dc_score_matrix(lam: float, mu: float, rho: float = DC_RHO) -> list[list[float]]:
    matrix = [[_poisson_pmf(i, lam) * _poisson_pmf(j, mu) for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    if matrix[0][0]:
        matrix[0][0] *= max(0.0, 1.0 - lam * mu * rho)
    if len(matrix[0]) > 1:
        matrix[0][1] *= max(0.0, 1.0 + lam * rho)
    if len(matrix) > 1:
        matrix[1][0] *= max(0.0, 1.0 + mu * rho)
        matrix[1][1] *= max(0.0, 1.0 - rho)
    total = sum(cell for row in matrix for cell in row)
    if total <= 0:
        return matrix
    return [[cell / total for cell in row] for row in matrix]


def _matrix_to_threeway(matrix: list[list[float]]) -> tuple[float, float, float]:
    home = sum(matrix[i][j] for i in range(len(matrix)) for j in range(i))
    draw = sum(matrix[i][i] for i in range(len(matrix)))
    away = sum(matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix[i])))
    return _normalize_threeway(home * 100.0, draw * 100.0, away * 100.0)


def _fit_dixon_coles_params(
    games: list[GameTuple],
    team_keys: list[str],
) -> tuple[dict[str, float], dict[str, float], float]:
    """Iterative attack/defence strengths (log-scale) without scipy."""
    index = {key: idx for idx, key in enumerate(team_keys)}
    n = len(team_keys)
    attack = [0.0] * n
    defence = [0.0] * n
    home_goals = [0.0] * n
    away_goals = [0.0] * n
    home_matches = [0] * n
    away_matches = [0] * n

    for home, away, _hn, _an, hg, ag in games:
        if home not in index or away not in index:
            continue
        hi, ai = index[home], index[away]
        home_goals[hi] += hg
        away_goals[ai] += ag
        home_matches[hi] += 1
        away_matches[ai] += 1

    league_home_avg = sum(hg for *_, hg, _ in games) / max(len(games), 1)
    league_away_avg = sum(ag for *_, _, ag in games) / max(len(games), 1)
    league_home_avg = max(league_home_avg, 0.5)
    league_away_avg = max(league_away_avg, 0.5)

    for i, key in enumerate(team_keys):
        if home_matches[i]:
            attack[i] = math.log(max(home_goals[i] / home_matches[i], 0.2) / league_home_avg)
        if away_matches[i]:
            defence[i] = math.log(max(away_goals[i] / away_matches[i], 0.2) / league_away_avg)

    for _ in range(8):
        for home, away, _hn, _an, hg, ag in games:
            if home not in index or away not in index:
                continue
            hi, ai = index[home], index[away]
            lam = math.exp(attack[hi] - defence[ai] + DC_HOME_ADV)
            mu = math.exp(attack[ai] - defence[hi])
            lam = max(min(lam, 5.0), 0.05)
            mu = max(min(mu, 5.0), 0.05)
            err_h = (hg - lam) * 0.05
            err_a = (ag - mu) * 0.05
            attack[hi] += err_h
            defence[ai] -= err_h
            attack[ai] += err_a
            defence[hi] -= err_a

    return (
        {team_keys[i]: attack[i] for i in range(n)},
        {team_keys[i]: defence[i] for i in range(n)},
        DC_HOME_ADV,
    )


def _resolve_model_key(
    team_keys: set[str],
    league: str,
    abbr: str,
    display_name: str | None = None,
) -> str | None:
    candidates: list[str] = []
    resolved = resolve_team(league, abbr, display_name)
    if resolved:
        candidates.append(resolved[0].lower())
    candidates.append(abbr.lower())

    for key in candidates:
        if key in team_keys:
            return key

    if display_name:
        target = display_name.lower()
        for key in team_keys:
            if key == target:
                return key

    return None


def build_soccer_model(
    games: list[GameTuple],
    league: str,
) -> dict[str, Any] | None:
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    team_keys = sorted({home for home, away, *_ in games} | {away for home, away, *_ in games})
    if len(team_keys) < 4:
        return None

    elo = _EloEngine()
    pi = _PiRating()
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}

    for home, away, _hn, _an, hg, ag in games:
        if home not in team_game_counts or away not in team_game_counts:
            continue
        elo.update(home, away, hg, ag)
        pi.update(home, away, hg, ag)
        team_game_counts[home] += 1
        team_game_counts[away] += 1

    attack, defence, home_adv = _fit_dixon_coles_params(games, team_keys)

    return {
        "league": league,
        "team_keys": team_keys,
        "elo": elo,
        "pi": pi,
        "attack": attack,
        "defence": defence,
        "home_adv": home_adv,
        "team_game_counts": team_game_counts,
    }


def _elo_threeway(elo: _EloEngine, home: str, away: str, league: str) -> tuple[float, float, float]:
    ra = elo.rating(home) + ELO_HOME_ADV
    rb = elo.rating(away)
    home_win = _elo_expected(ra, rb) * 100.0
    return _draw_from_closeness(home_win, league)


def _pi_threeway(pi: _PiRating, home: str, away: str) -> tuple[float, float, float]:
    expected_gd = pi.expected_gd(home, away)
    lam = max(0.3, 1.35 + expected_gd * 0.35)
    mu = max(0.3, 1.35 - expected_gd * 0.35)
    matrix = _dc_score_matrix(lam, mu, rho=0.0)
    return _matrix_to_threeway(matrix)


def _dc_threeway(
    attack: dict[str, float],
    defence: dict[str, float],
    home_adv: float,
    home: str,
    away: str,
) -> tuple[float, float, float, float, float]:
    a_h = attack.get(home, 0.0)
    d_h = defence.get(home, 0.0)
    a_a = attack.get(away, 0.0)
    d_a = defence.get(away, 0.0)
    lam = float(math.exp(a_h - d_a + home_adv))
    mu = float(math.exp(a_a - d_h))
    lam = max(min(lam, 5.0), 0.05)
    mu = max(min(mu, 5.0), 0.05)
    matrix = _dc_score_matrix(lam, mu)
    home_p, draw_p, away_p = _matrix_to_threeway(matrix)
    return home_p, draw_p, away_p, lam, mu


def predict_matchup_from_model(
    model: dict[str, Any],
    home_key: str,
    away_key: str,
) -> SoccerPredResult | None:
    home = home_key.lower()
    away = away_key.lower()
    team_keys = set(model["team_keys"])
    if home not in team_keys or away not in team_keys:
        return None

    league = model["league"]
    elo = model["elo"]
    pi = model["pi"]
    attack = model["attack"]
    defence = model["defence"]
    home_adv = float(model["home_adv"])

    elo_probs = _elo_threeway(elo, home, away, league)
    pi_probs = _pi_threeway(pi, home, away)
    dc_home, dc_draw, dc_away, lam, mu = _dc_threeway(attack, defence, home_adv, home, away)

    home_p = (elo_probs[0] + pi_probs[0] + dc_home) / 3.0
    draw_p = (elo_probs[1] + pi_probs[1] + dc_draw) / 3.0
    away_p = (elo_probs[2] + pi_probs[2] + dc_away) / 3.0
    home_p, draw_p, away_p = _normalize_threeway(home_p, draw_p, away_p)

    return SoccerPredResult(
        home_win_probability=round(home_p, 2),
        draw_probability=round(draw_p, 2),
        away_win_probability=round(away_p, 2),
        expected_home_goals=round(lam, 2),
        expected_away_goals=round(mu, 2),
        elo_home=round(elo.rating(home), 1),
        elo_away=round(elo.rating(away), 1),
        pi_expected_gd=round(pi.expected_gd(home, away), 2),
    )


def soccer_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season or sparse schedule."
        )

    model = build_soccer_model(games, league)
    if not model:
        return "Could not build soccer rating model on available games."

    team_keys = set(model["team_keys"])
    home = _resolve_model_key(team_keys, league, home_abbr)
    away = _resolve_model_key(team_keys, league, away_abbr)
    if not home or not away:
        missing = [k for k, v in ((home_abbr, home), (away_abbr, away)) if not v]
        return f"Teams not found in soccer model: {', '.join(missing)}."

    counts = model["team_game_counts"]
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the soccer model sample."
    return "Soccer rating model unavailable."


@lru_cache(maxsize=32)
def get_soccer_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_soccer_league(league):
        return None
    games = load_league_completed_games(league, cutoff_date)
    return build_soccer_model(games, league)


def run_soccer_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_name: str | None = None,
    away_name: str | None = None,
) -> dict[str, Any] | None:
    context = get_soccer_pred_context(league, cutoff_date)
    if not context:
        return None

    team_keys = set(context["team_keys"])
    home_key = _resolve_model_key(team_keys, league, home_abbr, home_name)
    away_key = _resolve_model_key(team_keys, league, away_abbr, away_name)
    if not home_key or not away_key:
        return None

    prediction = predict_matchup_from_model(context, home_key, away_key)
    if not prediction:
        return None

    counts = context["team_game_counts"]
    if counts.get(home_key, 0) < MIN_TEAM_GAMES or counts.get(away_key, 0) < MIN_TEAM_GAMES:
        return None

    return {
        "algorithm": prediction.algorithm,
        "source": prediction.source,
        "home_win_probability": prediction.home_win_probability,
        "draw_probability": prediction.draw_probability,
        "away_win_probability": prediction.away_win_probability,
        "expected_home_goals": prediction.expected_home_goals,
        "expected_away_goals": prediction.expected_away_goals,
        "elo_home": prediction.elo_home,
        "elo_away": prediction.elo_away,
        "pi_expected_gd": prediction.pi_expected_gd,
        "home_games": counts.get(home_key, 0),
        "away_games": counts.get(away_key, 0),
    }
