"""Iterative power ratings + logistic win probability (Sports-pred style)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PowerTeam:
    key: str
    name: str
    games: list["PowerGame"] = field(default_factory=list)
    apd: float = 0.0
    schedule: float = 0.0
    power: float = 0.0
    prev_power: float = 0.0
    wins: int = 0
    losses: int = 0
    ties: int = 0

    def calc_apd(self) -> float:
        if not self.games:
            return 0.0
        point_differential = 0.0
        for game in self.games:
            if self is game.home_team:
                point_differential += game.home_score - game.away_score
            else:
                point_differential += game.away_score - game.home_score
        return point_differential / len(self.games)

    def calc_sched(self) -> float:
        if not self.games:
            return 0.0
        opponent_powers = []
        for game in self.games:
            if self is game.home_team:
                opponent_powers.append(game.away_team.prev_power)
            else:
                opponent_powers.append(game.home_team.prev_power)
        return sum(opponent_powers) / len(opponent_powers)

    def calc_power(self) -> float:
        return self.calc_sched() + self.apd


@dataclass
class PowerGame:
    home_team: PowerTeam
    away_team: PowerTeam
    home_score: int
    away_score: int


def build_power_ratings(
    games: list[tuple[str, str, str, str, int, int]],
    *,
    iterations: int = 10,
) -> tuple[dict[str, PowerTeam], list[PowerGame], float | None]:
    """
    Build team power ratings from completed games.

    Each game tuple: (home_key, away_key, home_name, away_name, home_score, away_score)
    """
    teams: dict[str, PowerTeam] = {}
    total_games: list[PowerGame] = []

    def get_team(key: str, name: str) -> PowerTeam:
        if key not in teams:
            teams[key] = PowerTeam(key=key, name=name)
        return teams[key]

    for home_key, away_key, home_name, away_name, home_score, away_score in games:
        home_team = get_team(home_key, home_name)
        away_team = get_team(away_key, away_name)
        game = PowerGame(home_team, away_team, home_score, away_score)
        total_games.append(game)
        home_team.games.append(game)
        away_team.games.append(game)

        if home_score > away_score:
            home_team.wins += 1
            away_team.losses += 1
        elif home_score < away_score:
            home_team.losses += 1
            away_team.wins += 1
        else:
            home_team.ties += 1
            away_team.ties += 1

    if not total_games:
        return teams, total_games, None

    for team in teams.values():
        team.apd = team.calc_apd()

    for _ in range(iterations):
        for team in teams.values():
            team.schedule = team.calc_sched()
            team.power = team.calc_power()
        for team in teams.values():
            team.prev_power = team.power

    param = fit_logistic_param(total_games)
    return teams, total_games, param


def fit_logistic_param(games: list[PowerGame]) -> float | None:
    """Fit sigmoid scale param via log-loss minimization (no scipy)."""
    if len(games) < 3:
        return None

    xpoints: list[float] = []
    ypoints: list[float] = []
    for game in games:
        xpoints.append(game.home_team.power - game.away_team.power)
        if game.home_score > game.away_score:
            ypoints.append(1.0)
        elif game.home_score < game.away_score:
            ypoints.append(0.0)
        else:
            ypoints.append(0.5)

    best_param = 10.0
    best_loss = float("inf")
    for trial in range(50, 500):
        param = trial / 10.0
        loss = _log_loss(xpoints, ypoints, param)
        if loss < best_loss:
            best_loss = loss
            best_param = param

    # Refine around best coarse value
    coarse = best_param
    for trial in range(-20, 21):
        param = max(0.5, coarse + trial * 0.1)
        loss = _log_loss(xpoints, ypoints, param)
        if loss < best_loss:
            best_loss = loss
            best_param = param

    return best_param


def _sigmoid_prob(rating_diff: float, param: float) -> float:
    """Home win probability from home_power - away_power (matches Sports-pred fit)."""
    if param <= 0:
        return 0.5
    return 1.0 / (1.0 + math.exp(-rating_diff / param))


def _log_loss(xpoints: list[float], ypoints: list[float], param: float) -> float:
    total = 0.0
    count = 0
    for x, y in zip(xpoints, ypoints):
        if y == 0.5:
            continue
        prob = _sigmoid_prob(x, param)
        prob = min(max(prob, 1e-9), 1.0 - 1e-9)
        total += -(y * math.log(prob) + (1.0 - y) * math.log(1.0 - prob))
        count += 1
    return total / count if count else float("inf")


def calc_home_win_probability(
    home_power: float,
    away_power: float,
    param: float,
) -> float:
    """Return home win probability on 0–100 scale."""
    prob = _sigmoid_prob(home_power - away_power, param)
    return prob * 100.0


def predict_matchup(
    teams: dict[str, PowerTeam],
    param: float | None,
    home_key: str,
    away_key: str,
) -> dict[str, float | str | None] | None:
    home = teams.get(home_key)
    away = teams.get(away_key)
    if not home or not away or param is None:
        return None
    if not home.games or not away.games:
        return None

    home_win_prob = calc_home_win_probability(home.power, away.power, param)
    return {
        "home_key": home_key,
        "away_key": away_key,
        "home_power": round(home.power, 2),
        "away_power": round(away.power, 2),
        "power_diff": round(home.power - away.power, 2),
        "home_win_probability": round(home_win_prob, 2),
        "param": round(param, 3),
        "home_games": len(home.games),
        "away_games": len(away.games),
    }
