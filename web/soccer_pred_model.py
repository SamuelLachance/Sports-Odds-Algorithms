"""Soccer Path A: Hubáček-style 1X2 model (Elo + Pi + Dixon-Coles + form + XGB + decorrelation).

Stat layers are log-loss tuned via ``web.soccer_meta_model`` (``data/soccer_meta_weights.json``).
Walk-forward XGB 3-way classifier, temperature calibration, and capped market decorrelation.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from web.club_elo import elo_binary_home_win_pct, get_team_elo_rating, supports_club_elo
from web.league_profiles import SOCCER_DRAW_BASE, is_soccer_league
from web.live_data import resolve_team
from web.season_games import (
    GameTuple,
    _event_date_iso,
    _load_league_game_map,
    load_league_completed_games,
)
from web.soccer_decorrelation import blend_with_market_cap, devig_threeway_from_odds
from web.soccer_features import (
    blend_stat_and_xgb,
    build_matchup_features,
    features_to_vector,
)
from web.soccer_meta_model import (
    blend_weighted_threeway,
    fit_temperature_grid,
    get_league_meta_config,
    outcome_from_score,
    temperature_scale,
)
from web.soccer_pick_signals import build_soccer_pick_signals

MIN_LEAGUE_GAMES = 15
MIN_TEAM_GAMES = 2
MAX_GOALS = 6
WALK_FORWARD_MIN_TRAIN = 12
XGB_MIN_ROWS = 40
MIN_CALIBRATION_GAMES = 20

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
    source: str = "SoccerPathA"
    algorithm: str = "SoccerPathA"


def load_soccer_dated_games(league: str, cutoff_date: str) -> list[tuple[str, GameTuple]]:
    merged = _load_league_game_map(league, cutoff_date, for_backtest=False)
    return [
        (_event_date_iso(event_date), game_tuple)
        for event_date, game_tuple in sorted(merged.values(), key=lambda item: item[0])
    ]


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


class _FormTracker:
    def __init__(self) -> None:
        self.recent: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        if hg > ag:
            home_pts, away_pts = 3, 0
        elif hg < ag:
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1
        self.recent[home].append(home_pts)
        self.recent[away].append(away_pts)

    def form_score(self, team: str) -> float:
        points = self.recent.get(team)
        if not points:
            return 1.5
        return sum(points) / len(points)


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


def _train_xgb_threeway(
    x_rows: list[np.ndarray],
    y_rows: list[int],
) -> Any | None:
    if len(x_rows) < XGB_MIN_ROWS:
        return None
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    x_mat = np.vstack(x_rows)
    y = np.array(y_rows, dtype=int)
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    try:
        model.fit(x_mat, y)
        return model
    except ValueError:
        return None


def _snapshot_stat_model(
    games: list[GameTuple],
    league: str,
) -> dict[str, Any] | None:
    """Build stat-layer state without walk-forward artifacts."""
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    team_keys = sorted({home for home, away, *_ in games} | {away for home, away, *_ in games})
    if len(team_keys) < 4:
        return None

    elo = _EloEngine()
    pi = _PiRating()
    form = _FormTracker()
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}

    for home, away, _hn, _an, hg, ag in games:
        if home not in team_game_counts or away not in team_game_counts:
            continue
        elo.update(home, away, hg, ag)
        pi.update(home, away, hg, ag)
        form.update(home, away, hg, ag)
        team_game_counts[home] += 1
        team_game_counts[away] += 1

    attack, defence, home_adv = _fit_dixon_coles_params(games, team_keys)

    return {
        "league": league,
        "team_keys": team_keys,
        "elo": elo,
        "pi": pi,
        "form": form,
        "attack": attack,
        "defence": defence,
        "home_adv": home_adv,
        "team_game_counts": team_game_counts,
    }


def build_soccer_path_a_model(
    dated_games: list[tuple[str, GameTuple]],
    league: str,
) -> dict[str, Any] | None:
    """Walk-forward Path A model with XGB 3-way head and temperature calibration."""
    games_only = [game for _date, game in dated_games]
    base = _snapshot_stat_model(games_only, league)
    if not base:
        return None

    team_keys = list(base["team_keys"])
    elo = _EloEngine()
    pi = _PiRating()
    form = _FormTracker()
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}
    partial_games: list[GameTuple] = []

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    raw_samples: list[tuple[float, float, float, int]] = []

    for idx, (_game_date, game) in enumerate(dated_games):
        home, away, _hn, _an, hg, ag = game
        if home not in team_game_counts or away not in team_game_counts:
            continue

        if idx >= WALK_FORWARD_MIN_TRAIN and len(partial_games) >= MIN_LEAGUE_GAMES:
            attack, defence, home_adv = _fit_dixon_coles_params(partial_games, team_keys)
            snap = {
                **base,
                "elo": elo,
                "pi": pi,
                "form": form,
                "team_game_counts": dict(team_game_counts),
                "attack": attack,
                "defence": defence,
                "home_adv": home_adv,
            }
            stat = predict_matchup_from_model(snap, home, away, include_club_elo=True)
            if stat:
                stat_probs = (
                    stat.home_win_probability,
                    stat.draw_probability,
                    stat.away_win_probability,
                )
                form_diff = form.form_score(home) - form.form_score(away)
                feats = build_matchup_features(
                    elo_home=snap["elo"].rating(home),
                    elo_away=snap["elo"].rating(away),
                    pi_expected_gd=snap["pi"].expected_gd(home, away),
                    dc_lam=stat.expected_home_goals,
                    dc_mu=stat.expected_away_goals,
                    form_diff=form_diff,
                    stat_home=stat_probs[0],
                    stat_draw=stat_probs[1],
                    stat_away=stat_probs[2],
                    home_games=team_game_counts.get(home, 0),
                    away_games=team_game_counts.get(away, 0),
                )
                x_rows.append(features_to_vector(feats))
                y_rows.append(outcome_from_score(hg, ag))
                raw_samples.append((*stat_probs, outcome_from_score(hg, ag)))

        elo.update(home, away, hg, ag)
        pi.update(home, away, hg, ag)
        form.update(home, away, hg, ag)
        team_game_counts[home] += 1
        team_game_counts[away] += 1
        partial_games.append(game)

    xgb_model = _train_xgb_threeway(x_rows, y_rows)
    temperature = (
        fit_temperature_grid(raw_samples)
        if len(raw_samples) >= MIN_CALIBRATION_GAMES
        else get_league_meta_config(league)["temperature"]
    )

    final = _snapshot_stat_model(games_only, league)
    if not final:
        return None
    final["xgb_model"] = xgb_model
    final["temperature"] = temperature
    return final


def build_soccer_model(
    games: list[GameTuple],
    league: str,
) -> dict[str, Any] | None:
    return _snapshot_stat_model(games, league)


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


def extract_team_soccer_strengths(model: dict[str, Any]) -> dict[str, float]:
    """Dixon–Coles attack minus defence log-strength per team."""
    attack: dict[str, float] = model["attack"]
    defence: dict[str, float] = model["defence"]
    keys = set(attack) | set(defence)
    return {key: attack.get(key, 0.0) - defence.get(key, 0.0) for key in keys}


def _form_threeway(
    form: _FormTracker,
    home: str,
    away: str,
    league: str,
) -> tuple[float, float, float]:
    diff = form.form_score(home) - form.form_score(away)
    home_win = min(max(50.0 + diff * 8.0, 5.0), 95.0)
    return _draw_from_closeness(home_win, league)


def _club_elo_threeway(
    league: str,
    home: str,
    away: str,
) -> tuple[float, float, float] | None:
    if not supports_club_elo(league):
        return None
    home_elo = get_team_elo_rating(league, home)
    away_elo = get_team_elo_rating(league, away)
    if home_elo is None or away_elo is None:
        return None
    home_win = elo_binary_home_win_pct(home_elo, away_elo)
    return _draw_from_closeness(home_win, league)


def compute_sharp_stat_layers(
    model: dict[str, Any],
    home: str,
    away: str,
    *,
    include_club_elo: bool = True,
) -> tuple[dict[str, tuple[float, float, float] | None], float, float]:
    league = model["league"]
    dc_home, dc_draw, dc_away, lam, mu = _dc_threeway(
        model["attack"],
        model["defence"],
        float(model["home_adv"]),
        home,
        away,
    )
    club_elo = _club_elo_threeway(league, home, away) if include_club_elo else None
    layers = {
        "elo": _elo_threeway(model["elo"], home, away, league),
        "pi": _pi_threeway(model["pi"], home, away),
        "dc": (dc_home, dc_draw, dc_away),
        "form": _form_threeway(model["form"], home, away, league),
        "club_elo": club_elo,
    }
    return layers, lam, mu


def predict_matchup_from_model(
    model: dict[str, Any],
    home_key: str,
    away_key: str,
    *,
    include_club_elo: bool = True,
) -> SoccerPredResult | None:
    home = home_key.lower()
    away = away_key.lower()
    team_keys = set(model["team_keys"])
    if home not in team_keys or away not in team_keys:
        return None

    league = model["league"]
    layers, lam, mu = compute_sharp_stat_layers(
        model, home, away, include_club_elo=include_club_elo
    )
    stat_weights = get_league_meta_config(league)["stat_weights"]

    active_layers: list[tuple[float, float, float]] = []
    active_weights: list[float] = []
    for key in ("elo", "pi", "dc", "form", "club_elo"):
        triple = layers.get(key)
        weight = float(stat_weights.get(key, 0.0))
        if triple is None or weight <= 0:
            continue
        active_layers.append(triple)
        active_weights.append(weight)

    if not active_layers:
        return None

    home_p, draw_p, away_p = blend_weighted_threeway(active_layers, active_weights)
    stat_probs = (home_p, draw_p, away_p)

    xgb_model = model.get("xgb_model")
    if xgb_model is not None:
        try:
            form_diff = model["form"].form_score(home) - model["form"].form_score(away)
            feats = build_matchup_features(
                elo_home=model["elo"].rating(home),
                elo_away=model["elo"].rating(away),
                pi_expected_gd=model["pi"].expected_gd(home, away),
                dc_lam=lam,
                dc_mu=mu,
                form_diff=form_diff,
                stat_home=stat_probs[0],
                stat_draw=stat_probs[1],
                stat_away=stat_probs[2],
                home_games=model["team_game_counts"].get(home, 0),
                away_games=model["team_game_counts"].get(away, 0),
            )
            xgb_row = xgb_model.predict_proba(features_to_vector(feats).reshape(1, -1))[0]
            xgb_probs = (float(xgb_row[0]) * 100.0, float(xgb_row[1]) * 100.0, float(xgb_row[2]) * 100.0)
            home_p, draw_p, away_p = blend_stat_and_xgb(stat_probs, xgb_probs)
        except (ValueError, IndexError, AttributeError):
            pass

    return SoccerPredResult(
        home_win_probability=round(home_p, 2),
        draw_probability=round(draw_p, 2),
        away_win_probability=round(away_p, 2),
        expected_home_goals=round(lam, 2),
        expected_away_goals=round(mu, 2),
        elo_home=round(model["elo"].rating(home), 1),
        elo_away=round(model["elo"].rating(away), 1),
        pi_expected_gd=round(model["pi"].expected_gd(home, away), 2),
        source="path-a-xgb-decorrelated",
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
    dated = load_soccer_dated_games(league, cutoff_date)
    model = build_soccer_path_a_model(dated, league)
    if model:
        return model
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
    home_ml: int | None = None,
    draw_ml: int | None = None,
    away_ml: int | None = None,
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

    raw_probs = (
        prediction.home_win_probability,
        prediction.draw_probability,
        prediction.away_win_probability,
    )
    market_probs = devig_threeway_from_odds(home_ml, draw_ml, away_ml)
    decorrelated = blend_with_market_cap(raw_probs, market_probs)
    temperature = float(
        context.get("temperature") or get_league_meta_config(league)["temperature"]
    )
    calibrated = temperature_scale(*decorrelated, temperature)
    calibrated = tuple(round(v, 2) for v in calibrated)

    pick_signals = build_soccer_pick_signals(
        home_prob=calibrated[0],
        draw_prob=calibrated[1],
        away_prob=calibrated[2],
        home_ml=home_ml,
        draw_ml=draw_ml,
        away_ml=away_ml,
        home_games=counts.get(home_key, 0),
        away_games=counts.get(away_key, 0),
    )

    return {
        "algorithm": "SoccerPathA",
        "source": prediction.source,
        "home_win_probability": calibrated[0],
        "draw_probability": calibrated[1],
        "away_win_probability": calibrated[2],
        "raw_home_win_probability": raw_probs[0],
        "raw_draw_probability": raw_probs[1],
        "raw_away_win_probability": raw_probs[2],
        "expected_home_goals": prediction.expected_home_goals,
        "expected_away_goals": prediction.expected_away_goals,
        "elo_home": prediction.elo_home,
        "elo_away": prediction.elo_away,
        "pi_expected_gd": prediction.pi_expected_gd,
        "home_games": counts.get(home_key, 0),
        "away_games": counts.get(away_key, 0),
        "market_decorrelated": market_probs is not None,
        "soccer_pick_signals": pick_signals,
    }
