"""FIFA World Cup 2026 draw — 12 groups of four (48 teams)."""

from __future__ import annotations

# Official draw (Dec 2025). Keys are canonical names used for matching ESPN display names.
WORLD_CUP_2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "Czechia", "South Africa"],
    "B": ["Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ESPN / alternate spellings → canonical group member name
TEAM_ALIASES: dict[str, str] = {
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "bosnia and herzegovina": "Bosnia-Herzegovina",
    "bosnia-herzegovina": "Bosnia-Herzegovina",
    "usa": "United States",
    "united states": "United States",
    "turkey": "Türkiye",
    "türkiye": "Türkiye",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "iran": "Iran",
    "ir iran": "Iran",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "korea": "South Korea",
    "mex": "Mexico",
    "rsa": "South Africa",
}

WORLD_CUP_FORMAT = {
    "teams": 48,
    "groups": 12,
    "group_size": 4,
    "group_matches_per_team": 3,
    "total_matches": 104,
    "group_stage_matches": 72,
    "knockout_start": "Round of 32",
    "knockout_teams": 32,
    "qualification": (
        "Top two in each group plus the eight best third-placed teams "
        "advance to the Round of 32 (new for 2026)."
    ),
    "dates": {"start": "2026-06-11", "end": "2026-07-19", "final": "2026-07-19"},
    "hosts": ["Canada", "Mexico", "United States"],
}


def normalize_team_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    for alias, canonical in TEAM_ALIASES.items():
        if alias in key or key in alias:
            return canonical
    return (name or "").strip()


def team_group(name: str) -> str | None:
    canonical = normalize_team_name(name)
    for group_id, teams in WORLD_CUP_2026_GROUPS.items():
        for team in teams:
            if team.lower() == canonical.lower():
                return group_id
            if canonical.lower() in team.lower() or team.lower() in canonical.lower():
                return group_id
    return None


def all_group_team_names() -> list[str]:
    names: list[str] = []
    for teams in WORLD_CUP_2026_GROUPS.values():
        names.extend(teams)
    return names


def is_placeholder_team(name: str) -> bool:
    lower = (name or "").lower()
    markers = (
        "winner",
        "runner-up",
        "2nd place",
        "3rd place",
        "third place",
        "round of",
        "group ",
    )
    return any(m in lower for m in markers)
