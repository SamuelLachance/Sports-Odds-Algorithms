"""NFL team timezone / travel helpers for circadian features."""

from __future__ import annotations

# Approximate UTC offsets in standard time (ignore DST — relative diffs matter).
TEAM_TZ_OFFSET: dict[str, float] = {
    "ari": -7.0,
    "atl": -5.0,
    "bal": -5.0,
    "buf": -5.0,
    "car": -5.0,
    "chi": -6.0,
    "cin": -5.0,
    "cle": -5.0,
    "dal": -6.0,
    "den": -7.0,
    "det": -5.0,
    "gb": -6.0,
    "hou": -6.0,
    "ind": -5.0,
    "jax": -5.0,
    "jac": -5.0,
    "kc": -6.0,
    "la": -8.0,
    "lac": -8.0,
    "lar": -8.0,
    "lv": -8.0,
    "mia": -5.0,
    "min": -6.0,
    "ne": -5.0,
    "no": -6.0,
    "nyg": -5.0,
    "nyj": -5.0,
    "phi": -5.0,
    "pit": -5.0,
    "sea": -8.0,
    "sf": -8.0,
    "tb": -5.0,
    "ten": -6.0,
    "was": -5.0,
    "wsh": -5.0,
}


def team_tz(team: str) -> float:
    key = str(team or "").lower().strip()
    return float(TEAM_TZ_OFFSET.get(key, -5.0))


def timezone_diff(home: str, away: str) -> float:
    """Away team hours west of home stadium (positive => westbound travel)."""
    return team_tz(home) - team_tz(away)
