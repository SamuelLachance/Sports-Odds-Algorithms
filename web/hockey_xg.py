"""Expected-goals layer: NHL API xG blend + college SOS-adjusted rates."""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from typing import Any, Protocol


class _TeamRates(Protocol):
    goals_for_pg: float
    goals_against_pg: float

USER_AGENT = "Sports-Odds-Algorithms/2.0"
NHL_STATS_API = "https://api.nhle.com/stats/rest/en/team/summary"
NHL_WEB_API = "https://api-web.nhle.com/v1"
API_XG_BLEND = 0.5
HOME_ADVANTAGE = 0.15


def _fetch_json(url: str, timeout: int = 20) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=8)
def _fetch_nhl_team_xg_summary() -> dict[str, tuple[float, float, float | None]]:
    """team_abbr -> (xgf_pg, xga_pg, shot_xg_rate or None)."""
    stats: dict[str, tuple[float, float, float | None]] = {}
    payload = _fetch_json(
        f"{NHL_STATS_API}?cayenneExp=gameTypeId=2&limit=50"
        '&sort=[{"property":"xGoalsFor","direction":"DESC"}]'
    )
    if not isinstance(payload, dict):
        return stats
    for row in payload.get("data") or []:
        tri = str(row.get("teamAbbrev") or row.get("teamTriCode") or "").lower()
        gp = _safe_float(row.get("gamesPlayed")) or 0.0
        if not tri or gp <= 0:
            continue
        xgf = _safe_float(row.get("xGoalsFor"))
        xga = _safe_float(row.get("xGoalsAgainst"))
        if xgf is None or xga is None:
            continue
        shots = _safe_float(row.get("shotsOnGoalFor"))
        goals = _safe_float(row.get("goalsFor"))
        shot_rate = None
        if shots and shots > 0 and goals is not None:
            shot_rate = goals / shots
        stats[tri] = (xgf / gp, xga / gp, shot_rate)
    return stats


def _poisson_rates(metrics: _TeamRates) -> tuple[float, float]:
    return metrics.goals_for_pg, metrics.goals_against_pg


def _college_sos_adjusted_rates(
    team_abbr: str,
    team_metrics: dict[str, _TeamRates],
    dated_games: list[tuple[str, tuple]] | None,
) -> tuple[float, float]:
    """Enhance goal rates with strength-of-schedule from opponent scoring."""
    team = team_abbr.lower()
    metrics = team_metrics.get(team)
    if not metrics:
        return 0.0, 0.0
    base_gf = metrics.goals_for_pg
    base_ga = metrics.goals_against_pg
    if not dated_games:
        return base_gf, base_ga

    opp_gf: list[float] = []
    for _date, (home, away, _hn, _an, home_score, away_score) in dated_games:
        if home.lower() == team:
            opp = away.lower()
            opp_gf.append(float(away_score))
        elif away.lower() == team:
            opp = home.lower()
            opp_gf.append(float(home_score))
        else:
            continue
        opp_metrics = team_metrics.get(opp)
        if opp_metrics:
            opp_gf[-1] = opp_metrics.goals_for_pg

    if not opp_gf:
        return base_gf, base_ga

    league_avg = sum(m.goals_for_pg for m in team_metrics.values()) / max(len(team_metrics), 1)
    sos = sum(opp_gf) / len(opp_gf)
    sos_factor = sos / league_avg if league_avg > 0 else 1.0
    sos_factor = max(0.85, min(1.15, sos_factor))
    adjusted_gf = base_gf / sos_factor
    adjusted_ga = base_ga * sos_factor
    return round(adjusted_gf, 3), round(adjusted_ga, 3)


@lru_cache(maxsize=256)
def team_xg_rates(
    league: str,
    team_abbr: str,
) -> tuple[float, float]:
    """Cached per-team xGF/GP and xGA/GP (goal-based fallback when API unavailable)."""
    league = league.lower()
    team = team_abbr.lower()
    if league == "nhl":
        api = _fetch_nhl_team_xg_summary()
        if team in api:
            xgf, xga, _shot = api[team]
            return xgf, xga
    return 0.0, 0.0


def team_xg_rates_from_metrics(
    league: str,
    team_abbr: str,
    metrics: _TeamRates,
    *,
    team_metrics: dict[str, _TeamRates] | None = None,
    dated_games: list[tuple[str, tuple]] | None = None,
) -> tuple[float, float]:
    """Blend API / SOS-adjusted rates with Poisson goal metrics."""
    league = league.lower()
    team = team_abbr.lower()
    poisson_gf, poisson_ga = _poisson_rates(metrics)

    if league == "nhl":
        api = _fetch_nhl_team_xg_summary()
        if team in api:
            api_gf, api_ga, shot_rate = api[team]
            xgf = API_XG_BLEND * api_gf + (1 - API_XG_BLEND) * poisson_gf
            xga = API_XG_BLEND * api_ga + (1 - API_XG_BLEND) * poisson_ga
            if shot_rate is not None and shot_rate > 0:
                xgf = 0.9 * xgf + 0.1 * (shot_rate * 30.0)
            return round(max(0.5, xgf), 3), round(max(0.5, xga), 3)
        return round(max(0.5, poisson_gf), 3), round(max(0.5, poisson_ga), 3)

    if league in ("ncaah", "ncaawh") and team_metrics is not None:
        gf, ga = _college_sos_adjusted_rates(team, team_metrics, dated_games)
        return round(max(0.5, gf), 3), round(max(0.5, ga), 3)

    return round(max(0.5, poisson_gf), 3), round(max(0.5, poisson_ga), 3)


def matchup_expected_goals(
    home: str,
    away: str,
    league: str,
    team_metrics: dict[str, _TeamRates],
    xg_rates: dict[str, tuple[float, float]] | None = None,
    *,
    dated_games: list[tuple[str, tuple]] | None = None,
    home_advantage: float = HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Expected goals for a matchup from blended xG rates."""
    home_key = home.lower()
    away_key = away.lower()
    home_metrics = team_metrics.get(home_key)
    away_metrics = team_metrics.get(away_key)
    if not home_metrics or not away_metrics:
        return 0.5, 0.5

    if xg_rates and home_key in xg_rates and away_key in xg_rates:
        home_gf, home_ga = xg_rates[home_key]
        away_gf, away_ga = xg_rates[away_key]
    else:
        home_gf, home_ga = team_xg_rates_from_metrics(
            league,
            home_key,
            home_metrics,
            team_metrics=team_metrics,
            dated_games=dated_games,
        )
        away_gf, away_ga = team_xg_rates_from_metrics(
            league,
            away_key,
            away_metrics,
            team_metrics=team_metrics,
            dated_games=dated_games,
        )

    home_xg = (home_gf + away_ga) / 2
    home_xg *= 1 + home_advantage
    away_xg = (away_gf + home_ga) / 2
    away_xg *= 1 - home_advantage / 2
    return round(max(0.5, home_xg), 2), round(max(0.5, away_xg), 2)
