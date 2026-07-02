"""Point-in-time Algo V2 home win probability from completed-game history."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from web.blend_service import total_score_to_home_win_prob
from web.espn_client import iso_to_project_date
from web.league_profiles import MIN_GAMES_FOR_MODEL, NUM_PERIODS, get_algo_league


@dataclass
class TeamHistoryBuilder:
    """Incrementally accumulate team game rows for Algo V2 backtests."""

    league: str
    _dates: dict[str, list[str]] = field(default_factory=dict)
    _opponents: dict[str, list[str]] = field(default_factory=dict)
    _home_away: dict[str, list[str]] = field(default_factory=dict)
    _scores: dict[str, list[list[int]]] = field(default_factory=dict)

    def add_game(
        self,
        game_date_iso: str,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
    ) -> None:
        home = home.lower()
        away = away.lower()
        project_date = iso_to_project_date(f"{game_date_iso}T12:00:00Z")
        periods = NUM_PERIODS[self.league]

        for team, opponent, role, team_score, opp_score in (
            (home, away, "home", home_score, away_score),
            (away, home, "away", away_score, home_score),
        ):
            self._dates.setdefault(team, []).append(project_date)
            self._opponents.setdefault(team, []).append(opponent)
            self._home_away.setdefault(team, []).append(role)
            self._scores.setdefault(team, []).append([team_score, opp_score])

    def team_data(self, team_abbr: str) -> list[dict[str, Any]]:
        team = team_abbr.lower()
        dates = self._dates.get(team) or []
        if len(dates) < MIN_GAMES_FOR_MODEL:
            return []
        periods = NUM_PERIODS[self.league]
        period_scores = [[[0] * periods, [0] * periods] for _ in dates]
        season_year = dates[-1].split("-")[-1] if dates else "2025"
        return [
            {
                "year": season_year,
                "dates": list(dates),
                "other_team": list(self._opponents[team]),
                "home_away": list(self._home_away[team]),
                "game_scores": list(self._scores[team]),
                "period_scores": period_scores,
                "seasons_used": [season_year],
                "used_prior_season": False,
            }
        ]


def legacy_home_win_probability(
    league: str,
    builder: TeamHistoryBuilder,
    *,
    home_abbr: str,
    away_abbr: str,
    home_team: list[str],
    away_team: list[str],
    game_date_iso: str,
) -> float | None:
    """Algo V2 home win % using only games already added to ``builder``."""
    home_data = builder.team_data(home_abbr)
    away_data = builder.team_data(away_abbr)
    if not home_data or not away_data:
        return None

    from algo import Algo
    from odds_calculator import Odds_Calculator

    algo_league = get_algo_league(league)
    cutoff = iso_to_project_date(f"{game_date_iso}T12:00:00Z")
    odds_calculator = Odds_Calculator(algo_league)
    algo = Algo(algo_league)

    with redirect_stdout(io.StringIO()):
        returned_away = odds_calculator.analyze2(away_team, home_team, away_data, "away")
        returned_home = odds_calculator.analyze2(home_team, away_team, home_data, "home")
        algo_data = algo.calculate_V2(cutoff, returned_away, returned_home)

    return total_score_to_home_win_prob(float(algo_data["total"]))
