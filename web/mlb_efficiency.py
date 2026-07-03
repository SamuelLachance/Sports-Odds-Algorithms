"""Team run-scoring efficiency (EWMA) for MLB RunCast."""

from __future__ import annotations

from dataclasses import dataclass, field

LEAGUE_RUNS_PER_GAME = 4.5
EWMA_ALPHA = 0.12
DEFAULT_HCA_RUNS = 0.22
DEFAULT_SIGMA = 1.35


@dataclass
class TeamRunsProfile:
    rs_ewma: float = LEAGUE_RUNS_PER_GAME
    ra_ewma: float = LEAGUE_RUNS_PER_GAME
    games: int = 0

    @property
    def net_ewma(self) -> float:
        return self.rs_ewma - self.ra_ewma


@dataclass
class RunsEfficiencyState:
    teams: dict[str, TeamRunsProfile] = field(default_factory=dict)

    def team(self, key: str) -> TeamRunsProfile:
        key = key.lower()
        if key not in self.teams:
            self.teams[key] = TeamRunsProfile()
        return self.teams[key]


def update_runs_efficiency_after_game(
    state: RunsEfficiencyState,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> None:
    home = home.lower()
    away = away.lower()
    for team_key, scored, allowed in (
        (home, float(home_score), float(away_score)),
        (away, float(away_score), float(home_score)),
    ):
        profile = state.team(team_key)
        alpha = EWMA_ALPHA
        if profile.games == 0:
            profile.rs_ewma = scored
            profile.ra_ewma = allowed
        else:
            profile.rs_ewma = (1.0 - alpha) * profile.rs_ewma + alpha * scored
            profile.ra_ewma = (1.0 - alpha) * profile.ra_ewma + alpha * allowed
        profile.games += 1


def expected_runs_for_matchup(
    home: str,
    away: str,
    state: RunsEfficiencyState,
    *,
    hca: float = DEFAULT_HCA_RUNS,
    pitcher_home_adj: float = 0.0,
    pitcher_away_adj: float = 0.0,
) -> tuple[float, float]:
    """Return expected home and away runs before Monte Carlo."""
    home_prof = state.team(home)
    away_prof = state.team(away)
    home_runs = (
        (home_prof.rs_ewma + away_prof.ra_ewma) / 2.0
        + hca / 2.0
        + pitcher_home_adj
        - pitcher_away_adj / 2.0
    )
    away_runs = (
        (away_prof.rs_ewma + home_prof.ra_ewma) / 2.0
        - hca / 2.0
        + pitcher_away_adj
        - pitcher_home_adj / 2.0
    )
    return max(home_runs, 0.5), max(away_runs, 0.5)


def margin_to_home_win_prob(margin: float, *, sigma: float = DEFAULT_SIGMA) -> float:
    import math

    z = margin / max(sigma, 0.05)
    prob = 1.0 / (1.0 + math.exp(-z))
    return round(min(max(prob * 100.0, 1.0), 99.0), 2)
