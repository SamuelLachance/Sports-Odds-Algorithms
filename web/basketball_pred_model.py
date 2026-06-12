"""NBA-prediction style matrix completion for basketball win probability."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from web.league_profiles import LEAGUE_PROFILES
from web.season_games import load_league_completed_games

BASKETBALL_LEAGUES: tuple[str, ...] = ("nba", "wnba", "cbb")

# Soft-impute lambda (pace / OR) from christopherjenness/NBA-prediction R script.
LEAGUE_SOFT_IMPUTE: dict[str, dict[str, float]] = {
    "nba": {"pace_lambda": 25.0, "or_lambda": 50.0, "home_adv": 2.5},
    "wnba": {"pace_lambda": 20.0, "or_lambda": 40.0, "home_adv": 2.0},
    "cbb": {"pace_lambda": 30.0, "or_lambda": 55.0, "home_adv": 3.0},
}

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3
SOFT_IMPUTE_RANK = 10
SOFT_IMPUTE_ITERS = 50
SVD_POWER_ITERS = 25


def is_basketball_league(league: str) -> bool:
    league = league.lower()
    profile = LEAGUE_PROFILES.get(league)
    return profile is not None and profile["category"] == "basketball"


def _update_matrix_cell(matrix: list[list[float]], i: int, j: int, value: float) -> None:
    old = matrix[i][j]
    if old == 0.0:
        matrix[i][j] = value
    else:
        matrix[i][j] = (old + value) / 2.0


def _build_raw_matrices(
    games: list[tuple[str, str, str, str, int, int]],
    team_keys: list[str],
) -> tuple[list[list[float]], list[list[float]], float]:
    """Build OR and pace matrices from completed game scores."""
    index = {key: idx for idx, key in enumerate(team_keys)}
    n = len(team_keys)
    or_matrix = [[0.0] * n for _ in range(n)]
    pace_matrix = [[0.0] * n for _ in range(n)]
    totals: list[float] = []

    for home_key, away_key, _hn, _an, home_score, away_score in games:
        if home_key not in index or away_key not in index:
            continue
        game_total = float(home_score + away_score)
        if game_total <= 0:
            continue
        totals.append(game_total)
        hi = index[home_key]
        ai = index[away_key]
        league_avg = sum(totals) / len(totals)
        pace_val = 100.0 * game_total / league_avg

        _update_matrix_cell(or_matrix, hi, ai, float(home_score))
        _update_matrix_cell(or_matrix, ai, hi, float(away_score))
        _update_matrix_cell(pace_matrix, hi, ai, pace_val)
        _update_matrix_cell(pace_matrix, ai, hi, pace_val)

    league_avg_total = sum(totals) / len(totals) if totals else 200.0
    return or_matrix, pace_matrix, league_avg_total


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            if aik == 0.0:
                continue
            bk_row = b[k]
            out_row = out[i]
            for j in range(cols):
                out_row[j] += aik * bk_row[j]
    return out


def _mat_transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _scale_vec(v: list[float], factor: float) -> list[float]:
    return [x * factor for x in v]


def _top_singular_triplet(
    matrix: list[list[float]],
) -> tuple[float, list[float], list[float]]:
    """Power iteration for dominant singular value / vectors."""
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    v = [1.0 / math.sqrt(n_cols) for _ in range(n_cols)]
    for _ in range(SVD_POWER_ITERS):
        mt = _mat_transpose(matrix)
        v_new = _mat_mul(mt, [v])[0]
        norm = _norm(v_new)
        if norm < 1e-12:
            break
        v = _scale_vec(v_new, 1.0 / norm)

    u = _mat_mul(matrix, [v])[0]
    sigma = _norm(u)
    if sigma < 1e-12:
        return 0.0, [0.0] * n_rows, v
    u = _scale_vec(u, 1.0 / sigma)
    return sigma, u, v


def _deflate(matrix: list[list[float]], sigma: float, u: list[float], v: list[float]) -> None:
    for i in range(len(matrix)):
        ui = u[i]
        if ui == 0.0:
            continue
        row = matrix[i]
        for j in range(len(row)):
            row[j] -= sigma * ui * v[j]


def _truncated_svd(
    matrix: list[list[float]],
    rank: int,
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Return U (n x rank), singular values, V (rank x n) for top rank components."""
    work = [row[:] for row in matrix]
    n = len(work)
    u_cols: list[list[float]] = []
    sigmas: list[float] = []
    v_cols: list[list[float]] = []

    for _ in range(min(rank, n)):
        sigma, u_vec, v_vec = _top_singular_triplet(work)
        if sigma < 1e-9:
            break
        u_cols.append(u_vec)
        v_cols.append(v_vec)
        sigmas.append(sigma)
        _deflate(work, sigma, u_vec, v_vec)

    return u_cols, sigmas, v_cols


def _reconstruct(
    u_cols: list[list[float]],
    sigmas: list[float],
    v_cols: list[list[float]],
) -> list[list[float]]:
    n = len(u_cols[0]) if u_cols else 0
    out = [[0.0] * n for _ in range(n)]
    for sigma, u_vec, v_vec in zip(sigmas, u_cols, v_cols):
        shrunk = max(0.0, sigma)
        if shrunk == 0.0:
            continue
        for i in range(n):
            ui = u_vec[i]
            if ui == 0.0:
                continue
            row = out[i]
            for j in range(n):
                row[j] += shrunk * ui * v_vec[j]
    return out


def soft_impute(
    raw: list[list[float]],
    lambda_: float,
    *,
    max_rank: int = SOFT_IMPUTE_RANK,
    max_iters: int = SOFT_IMPUTE_ITERS,
) -> list[list[float]]:
    """
    Soft-thresholded SVD matrix completion (Mazumder et al. softImpute).

    Missing entries are zeros in `raw`; observed entries are preserved each step.
    """
    n = len(raw)
    observed = [[raw[i][j] != 0.0 for j in range(n)] for i in range(n)]
    z = [row[:] for row in raw]

    for _ in range(max_iters):
        u_cols, sigmas, v_cols = _truncated_svd(z, max_rank)
        if not sigmas:
            break
        shrunk_sigmas = [max(0.0, s - lambda_) for s in sigmas]
        z_new = _reconstruct(u_cols, shrunk_sigmas, v_cols)
        for i in range(n):
            for j in range(n):
                if observed[i][j]:
                    z[i][j] = raw[i][j]
                else:
                    z[i][j] = z_new[i][j]

    return z


def _fit_margin_param(margins: list[float], outcomes: list[float]) -> float:
    """Fit logistic scale for predicted margin -> win probability."""
    if len(margins) < 5:
        return 10.0

    best_param = 10.0
    best_loss = float("inf")
    for trial in range(30, 300):
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


def _predict_points(
    or_matrix: list[list[float]],
    pace_matrix: list[list[float]],
    offense_idx: int,
    defense_idx: int,
) -> float:
    return or_matrix[offense_idx][defense_idx] * pace_matrix[offense_idx][defense_idx] / 100.0


def build_basketball_model(
    games: list[tuple[str, str, str, str, int, int]],
    league: str,
) -> dict[str, Any] | None:
    """Build imputed OR/pace matrices and logistic margin param."""
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    team_keys = sorted(
        {home for home, away, *_ in games} | {away for home, away, *_ in games}
    )
    if len(team_keys) < 4:
        return None

    or_raw, pace_raw, league_avg_total = _build_raw_matrices(games, team_keys)
    config = LEAGUE_SOFT_IMPUTE.get(league, LEAGUE_SOFT_IMPUTE["nba"])

    or_completed = soft_impute(or_raw, config["or_lambda"])
    pace_completed = soft_impute(pace_raw, config["pace_lambda"])

    index = {key: idx for idx, key in enumerate(team_keys)}
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}
    margins: list[float] = []
    outcomes: list[float] = []
    home_adv = config["home_adv"]

    for home_key, away_key, _hn, _an, home_score, away_score in games:
        if home_key in team_game_counts:
            team_game_counts[home_key] += 1
        if away_key in team_game_counts:
            team_game_counts[away_key] += 1
        if home_key not in index or away_key not in index:
            continue
        hi = index[home_key]
        ai = index[away_key]
        pred_home = _predict_points(or_completed, pace_completed, hi, ai) + home_adv
        pred_away = _predict_points(or_completed, pace_completed, ai, hi)
        margin = pred_home - pred_away
        margins.append(margin)
        if home_score > away_score:
            outcomes.append(1.0)
        elif home_score < away_score:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)

    param = _fit_margin_param(margins, outcomes)

    return {
        "team_keys": team_keys,
        "index": index,
        "or_matrix": or_completed,
        "pace_matrix": pace_completed,
        "param": param,
        "home_adv": home_adv,
        "league_avg_total": league_avg_total,
        "team_game_counts": team_game_counts,
    }


def predict_matchup_from_model(
    model: dict[str, Any],
    home_key: str,
    away_key: str,
) -> dict[str, float | str] | None:
    index = model["index"]
    home = home_key.lower()
    away = away_key.lower()
    if home not in index or away not in index:
        return None

    hi = index[home]
    ai = index[away]
    or_matrix = model["or_matrix"]
    pace_matrix = model["pace_matrix"]
    home_adv = float(model["home_adv"])
    param = float(model["param"])

    home_points = _predict_points(or_matrix, pace_matrix, hi, ai) + home_adv
    away_points = _predict_points(or_matrix, pace_matrix, ai, hi)
    margin = home_points - away_points
    prob = 1.0 / (1.0 + math.exp(-margin / param))
    home_win_prob = prob * 100.0

    return {
        "home_key": home,
        "away_key": away,
        "predicted_home_score": round(home_points, 1),
        "predicted_away_score": round(away_points, 1),
        "predicted_margin": round(margin, 2),
        "home_win_probability": round(home_win_prob, 2),
        "param": round(param, 3),
    }


@lru_cache(maxsize=32)
def get_basketball_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_basketball_league(league):
        return None
    games = load_league_completed_games(league, cutoff_date)
    return build_basketball_model(games, league)


def run_basketball_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Run NBA-prediction matrix model for a basketball matchup."""
    context = get_basketball_pred_context(league, cutoff_date)
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
        "algorithm": "BasketballMatrix",
        "source": "NBA-prediction",
        "home_win_probability": prediction["home_win_probability"],
        "predicted_home_score": prediction["predicted_home_score"],
        "predicted_away_score": prediction["predicted_away_score"],
        "predicted_margin": prediction["predicted_margin"],
        "param": prediction["param"],
        "home_games": home_games,
        "away_games": away_games,
    }
