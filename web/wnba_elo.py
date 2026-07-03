"""FiveThirtyEight-style Elo ratings for WNBA (K=32, HCA=80, 50% season reversion)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0
HOME_ELO_ADV = 80.0
SEASON_REGRESSION = 0.5
EXPANSION_ELO = 1300.0


@dataclass
class EloState:
    ratings: dict[str, float] = field(default_factory=dict)
    last_season: int | None = None

    def rating(self, team: str) -> float:
        return self.ratings.get(team.lower(), DEFAULT_ELO)

    def regress_season(self, season_year: int) -> None:
        if self.last_season is not None and season_year != self.last_season:
            for key in list(self.ratings):
                self.ratings[key] = (
                    SEASON_REGRESSION * DEFAULT_ELO
                    + (1.0 - SEASON_REGRESSION) * self.ratings[key]
                )
        self.last_season = season_year

    def ensure_team(self, team: str) -> None:
        key = team.lower()
        if key not in self.ratings:
            self.ratings[key] = EXPANSION_ELO


def elo_home_win_prob(
    home_elo: float,
    away_elo: float,
    *,
    neutral: bool = False,
) -> float:
    home_adj = home_elo if neutral else home_elo + HOME_ELO_ADV
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_adj) / 400.0))


def _mov_multiplier(margin: float) -> float:
    """538-style margin-of-victory multiplier with dampening."""
    mov = abs(float(margin))
    if mov <= 0:
        return 1.0
    return math.log(max(mov, 1.0) + 1.0) * (2.2 / (0.001 * mov + 2.2))


def update_elo_after_game(
    state: EloState,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> None:
    home = home.lower()
    away = away.lower()
    state.ensure_team(home)
    state.ensure_team(away)

    home_r = state.ratings[home]
    away_r = state.ratings[away]
    home_prob = elo_home_win_prob(home_r, away_r)
    home_won = home_score > away_score
    if home_score == away_score:
        outcome = 0.5
    else:
        outcome = 1.0 if home_won else 0.0

    margin = home_score - away_score
    mult = _mov_multiplier(margin)
    shift = K_FACTOR * mult * (outcome - home_prob)
    state.ratings[home] = home_r + shift
    state.ratings[away] = away_r - shift


def elo_predicted_margin(home_elo: float, away_elo: float) -> float:
    """Approximate point margin from Elo diff (WNBA ~25 pts per 400 Elo)."""
    home_adj = home_elo + HOME_ELO_ADV
    diff = home_adj - away_elo
    return diff / 25.0
