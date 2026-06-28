"""Audit player rating coverage against ESPN rosters at build time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from web.league_profiles import SUPPORTED_LEAGUES
from web.sports_db.espn_fetch import fetch_team_roster
from web.sports_db.external_ratings import (
    RATING_DERIVED_SOURCE,
    RATING_PRIOR_SOURCE,
    lookup_player_rating,
    rating_source_label,
)
from web.sports_db.normalize import parse_roster
from web.team_service import fetch_espn_team_ids


@dataclass
class CoverageReport:
    league: str
    teams_checked: int = 0
    players_checked: int = 0
    direct_matches: int = 0
    derived: int = 0
    uncovered: int = 0
    uncovered_players: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.uncovered == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "league": self.league,
            "teams_checked": self.teams_checked,
            "players_checked": self.players_checked,
            "direct_matches": self.direct_matches,
            "derived": self.derived,
            "uncovered": self.uncovered,
            "uncovered_sample": self.uncovered_players[:25],
        }


def _is_covered(result: Any) -> tuple[bool, str]:
    source = (result.rating_source or "").lower()
    if source == RATING_PRIOR_SOURCE:
        return False, "prior"
    if result.rating is None:
        return False, "missing"
    if source == RATING_DERIVED_SOURCE:
        return False, "derived"
    if result.matched:
        return True, "direct"
    return False, source or "unknown"


def audit_league_ratings(
    league: str,
    *,
    team_abbrs: list[str] | None = None,
    max_teams: int | None = None,
) -> CoverageReport:
    """Check every ESPN roster player has external publisher OVR (no prior/derived)."""
    league = league.lower()
    report = CoverageReport(league=league)
    team_ids = fetch_espn_team_ids(league)
    abbrs = team_abbrs or sorted(team_ids.keys())
    if max_teams is not None:
        abbrs = abbrs[:max_teams]

    for abbr in abbrs:
        espn_id = team_ids.get(abbr)
        if not espn_id:
            continue
        payload = fetch_team_roster(league, espn_id)
        roster = parse_roster(payload)
        if not roster:
            continue
        report.teams_checked += 1
        for player in roster:
            name = (player.get("name") or "").strip()
            if not name or name == "?":
                continue
            report.players_checked += 1
            result = lookup_player_rating(
                league,
                name,
                team_abbr=abbr,
                espn_id=str(player.get("id") or ""),
                position=player.get("position"),
                roster_meta=player,
            )
            covered, kind = _is_covered(result)
            if kind == "direct":
                report.direct_matches += 1
            elif kind == "derived":
                report.derived += 1
            else:
                report.uncovered += 1
                if len(report.uncovered_players) < 50:
                    report.uncovered_players.append(
                        {
                            "team": abbr,
                            "name": player.get("name") or "?",
                            "reason": kind,
                            "source": result.rating_source or "",
                            "label": rating_source_label(result.rating_source, result.rating_year),
                        }
                    )
    return report


def audit_all_leagues(
    *,
    leagues: tuple[str, ...] | None = None,
    max_teams_per_league: int | None = None,
) -> dict[str, CoverageReport]:
    rows: dict[str, CoverageReport] = {}
    for league in leagues or SUPPORTED_LEAGUES:
        rows[league] = audit_league_ratings(league, max_teams=max_teams_per_league)
    return rows


def format_coverage_summary(reports: dict[str, CoverageReport]) -> str:
    lines = ["Player rating coverage audit:"]
    total_uncovered = 0
    for league in sorted(reports):
        report = reports[league]
        total_uncovered += report.uncovered
        status = "OK" if report.ok else f"FAIL ({report.uncovered} uncovered)"
        pct = (
            round(100.0 * report.direct_matches / report.players_checked, 1)
            if report.players_checked
            else 0.0
        )
        lines.append(
            f"  {league}: {status} — {pct}% ({report.direct_matches}/{report.players_checked} external)"
        )
    lines.append(f"Total uncovered: {total_uncovered}")
    return "\n".join(lines)


__all__ = [
    "CoverageReport",
    "audit_all_leagues",
    "audit_league_ratings",
    "format_coverage_summary",
]
