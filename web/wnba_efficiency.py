"""Her Hoop Stats–grade efficiency (ORtg/DRtg/pace per 100 poss) from game scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ORTG = 102.0
DEFAULT_DRTG = 102.0
DEFAULT_PACE = 95.0
DEFAULT_HCA = 2.5
DEFAULT_SIGMA = 10.0
EWMA_ALPHA = 0.22


@dataclass
class TeamEfficiency:
    ortg: float = DEFAULT_ORTG
    drtg: float = DEFAULT_DRTG
    pace: float = DEFAULT_PACE
    games: int = 0
    net_ewma: float = 0.0

    @property
    def net(self) -> float:
        return self.ortg - self.drtg


@dataclass
class EfficiencyState:
    teams: dict[str, TeamEfficiency] = field(default_factory=dict)

    def team(self, abbr: str) -> TeamEfficiency:
        key = abbr.lower()
        if key not in self.teams:
            self.teams[key] = TeamEfficiency()
        return self.teams[key]


def _estimate_possessions(home_score: int, away_score: int) -> float:
    total = float(home_score + away_score)
    return max(total * 0.48, 70.0)


def _game_efficiency(home_score: int, away_score: int) -> tuple[float, float, float]:
    poss = _estimate_possessions(home_score, away_score)
    home_ortg = 100.0 * home_score / poss
    away_ortg = 100.0 * away_score / poss
    pace = poss
    return home_ortg, away_ortg, pace


def update_efficiency_after_game(
    state: EfficiencyState,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> None:
    home_ortg, away_ortg, pace = _game_efficiency(home_score, away_score)
    for abbr, ortg, opp_ortg in (
        (home, home_ortg, away_ortg),
        (away, away_ortg, home_ortg),
    ):
        bucket = state.team(abbr)
        gp = bucket.games
        if gp <= 0:
            bucket.ortg = ortg
            bucket.drtg = opp_ortg
            bucket.pace = pace
            bucket.net_ewma = ortg - opp_ortg
        else:
            bucket.ortg = (bucket.ortg * gp + ortg) / (gp + 1)
            bucket.drtg = (bucket.drtg * gp + opp_ortg) / (gp + 1)
            bucket.pace = (bucket.pace * gp + pace) / (gp + 1)
            net = ortg - opp_ortg
            bucket.net_ewma = (1 - EWMA_ALPHA) * bucket.net_ewma + EWMA_ALPHA * net
        bucket.games = gp + 1


def expected_margin_from_efficiency(
    home: TeamEfficiency,
    away: TeamEfficiency,
    *,
    hca: float = DEFAULT_HCA,
) -> tuple[float, float, float]:
    """Return predicted home score, away score, home margin."""
    pace = (home.pace + away.pace) / 2.0
    home_pts = (home.ortg + away.drtg) / 2.0 * pace / 100.0 + hca / 2.0
    away_pts = (away.ortg + home.drtg) / 2.0 * pace / 100.0 - hca / 2.0
    margin = home_pts - away_pts
    return round(home_pts, 1), round(away_pts, 1), round(margin, 2)


def margin_to_home_win_prob(margin: float, *, sigma: float = DEFAULT_SIGMA) -> float:
    if sigma <= 0:
        sigma = DEFAULT_SIGMA
    prob = 1.0 / (1.0 + math.exp(-margin / sigma))
    return round(prob * 100.0, 2)


def build_efficiency_snapshot(games: list[tuple]) -> dict[str, TeamEfficiency]:
    state = EfficiencyState()
    for home, away, _hn, _an, home_score, away_score in games:
        update_efficiency_after_game(state, home, away, home_score, away_score)
    return state.teams
