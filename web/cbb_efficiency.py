"""KenPom-style efficiency margin and win probability for CBB."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from web.cbb_torvik import TorvikTeamRatings

DEFAULT_HCA = 3.0
DEFAULT_SIGMA = 12.0
LEAGUE_AVG_EFF = 100.0
POSSESSIONS_PER_GAME = 68.0


@dataclass
class TeamEfficiency:
    adj_o: float
    adj_d: float
    adj_t: float
    games: int = 0


def margin_to_home_win_prob(margin: float, sigma: float = DEFAULT_SIGMA) -> float:
    """Logistic mapping from expected home margin to home win probability (%)."""
    if sigma <= 0:
        sigma = DEFAULT_SIGMA
    prob = 1.0 / (1.0 + math.exp(-margin / sigma))
    return round(prob * 100.0, 2)


def expected_margin_from_ratings(
    home: TorvikTeamRatings | TeamEfficiency,
    away: TorvikTeamRatings | TeamEfficiency,
    *,
    hca: float = DEFAULT_HCA,
) -> tuple[float, float, float]:
    """Return (home_score, away_score, home_margin) from adj O/D/tempo."""
    avg_tempo = (home.adj_t + away.adj_t) / 2.0
    home_eff = home.adj_o - away.adj_d + LEAGUE_AVG_EFF
    away_eff = away.adj_o - home.adj_d + LEAGUE_AVG_EFF
    home_score = home_eff * avg_tempo / 100.0
    away_score = away_eff * avg_tempo / 100.0
    margin = home_score - away_score + hca
    return round(home_score, 1), round(away_score, 1), round(margin, 2)


def _iterative_team_efficiency(
    games: list[tuple[str, str, str, str, int, int]],
) -> dict[str, TeamEfficiency]:
    """Fallback adj O/D from completed ESPN games when Torvik is missing."""
    team_pts_for: dict[str, list[float]] = {}
    team_pts_against: dict[str, list[float]] = {}
    team_tempo: dict[str, list[float]] = {}
    game_counts: dict[str, int] = {}

    totals: list[float] = []
    for home, away, _hn, _an, home_score, away_score in games:
        game_total = float(home_score + away_score)
        if game_total <= 0:
            continue
        totals.append(game_total)
        for team, scored, allowed in (
            (home, home_score, away_score),
            (away, away_score, home_score),
        ):
            team_pts_for.setdefault(team, []).append(float(scored))
            team_pts_against.setdefault(team, []).append(float(allowed))
            team_tempo.setdefault(team, []).append(game_total)
            game_counts[team] = game_counts.get(team, 0) + 1

    league_avg_total = sum(totals) / len(totals) if totals else 140.0
    league_avg_ppp = league_avg_total / POSSESSIONS_PER_GAME

    ratings: dict[str, TeamEfficiency] = {}
    for team in set(game_counts):
        gp = game_counts[team]
        if gp == 0:
            continue
        off_raw = sum(team_pts_for.get(team, [])) / gp / POSSESSIONS_PER_GAME * 100.0
        def_raw = sum(team_pts_against.get(team, [])) / gp / POSSESSIONS_PER_GAME * 100.0
        tempo_raw = sum(team_tempo.get(team, [])) / gp
        # Shrink toward league average (KenPom-style prior).
        shrink = min(0.65, 8.0 / max(gp, 1))
        adj_o = off_raw * (1 - shrink) + LEAGUE_AVG_EFF * shrink
        adj_d = def_raw * (1 - shrink) + LEAGUE_AVG_EFF * shrink
        adj_t = tempo_raw * (1 - shrink) + POSSESSIONS_PER_GAME * shrink
        ratings[team] = TeamEfficiency(adj_o=adj_o, adj_d=adj_d, adj_t=adj_t, games=gp)
    return ratings


def resolve_team_ratings(
    home_abbr: str,
    away_abbr: str,
    torvik: dict[str, Any],
    fallback: dict[str, TeamEfficiency],
) -> tuple[Any, Any] | None:
    home = torvik.get(home_abbr.lower()) or fallback.get(home_abbr.lower())
    away = torvik.get(away_abbr.lower()) or fallback.get(away_abbr.lower())
    if home is None or away is None:
        return None
    return home, away


def build_fallback_efficiency(
    games: list[tuple[str, str, str, str, int, int]],
) -> dict[str, TeamEfficiency]:
    return _iterative_team_efficiency(games)
