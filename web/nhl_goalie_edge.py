"""NHL goalie / team save-rate enrichment from public NHL APIs."""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from typing import Any

USER_AGENT = "Sports-Odds-Algorithms/2.0"
NHL_WEB_API = "https://api-web.nhle.com/v1"

# ESPN abbr -> NHL tri-code
ESPN_TO_NHL_CODE: dict[str, str] = {
    "ana": "ANA",
    "ari": "ARI",
    "bos": "BOS",
    "buf": "BUF",
    "car": "CAR",
    "cbj": "CBJ",
    "cgy": "CGY",
    "chi": "CHI",
    "col": "COL",
    "dal": "DAL",
    "det": "DET",
    "edm": "EDM",
    "fla": "FLA",
    "la": "LAK",
    "min": "MIN",
    "mtl": "MTL",
    "nj": "NJD",
    "nsh": "NSH",
    "nyi": "NYI",
    "nyr": "NYR",
    "ott": "OTT",
    "phi": "PHI",
    "pit": "PIT",
    "sea": "SEA",
    "sj": "SJS",
    "stl": "STL",
    "tb": "TBL",
    "tor": "TOR",
    "uta": "UTA",
    "van": "VAN",
    "vgk": "VGK",
    "wpg": "WPG",
    "wsh": "WSH",
}

MAX_GOALIE_XG_SHIFT = 0.18


def _fetch_json(url: str, timeout: int = 20) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


@lru_cache(maxsize=4)
def fetch_team_goalie_save_pct() -> dict[str, float]:
    """Team-level save% proxy from public NHL APIs."""
    stats: dict[str, float] = {}
    payload = _fetch_json(f"{NHL_WEB_API}/club-stats/now")
    if isinstance(payload, dict):
        for row in payload.get("data") or []:
            tri = str(row.get("teamAbbrev") or "").upper()
            save_pct = row.get("savePct")
            if tri and save_pct is not None:
                try:
                    stats[tri.lower()] = float(save_pct)
                except (TypeError, ValueError):
                    continue
    if stats:
        return stats

    fallback = _fetch_json(
        "https://api.nhle.com/stats/rest/en/team/summary"
        "?cayenneExp=gameTypeId=2&limit=50&sort=[{\"property\":\"savePct\",\"direction\":\"DESC\"}]"
    )
    if isinstance(fallback, dict):
        for row in fallback.get("data") or []:
            tri = str(row.get("teamAbbrev") or row.get("teamTriCode") or "").lower()
            save_pct = row.get("savePct")
            if tri and save_pct is not None:
                try:
                    stats[tri] = float(save_pct)
                except (TypeError, ValueError):
                    continue
    return stats


def goalie_xg_adjustment(home_abbr: str, away_abbr: str) -> tuple[float, float] | None:
    """Adjust home/away xG based on team save-rate differential."""
    save_rates = fetch_team_goalie_save_pct()
    if not save_rates:
        return None
    home_code = ESPN_TO_NHL_CODE.get(home_abbr.lower(), home_abbr.upper())
    away_code = ESPN_TO_NHL_CODE.get(away_abbr.lower(), away_abbr.upper())
    home_sv = save_rates.get(home_code.lower()) or save_rates.get(home_abbr.lower())
    away_sv = save_rates.get(away_code.lower()) or save_rates.get(away_abbr.lower())
    if home_sv is None or away_sv is None:
        return None
    # Better save% suppresses goals against.
    home_shift = (home_sv - 0.905) * 0.8
    away_shift = (away_sv - 0.905) * 0.8
    home_xg_mult = max(0.92, min(1.08, 1.0 - away_shift))
    away_xg_mult = max(0.92, min(1.08, 1.0 - home_shift))
    return home_xg_mult, away_xg_mult
