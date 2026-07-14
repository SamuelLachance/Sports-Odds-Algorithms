"""PIT AP Top 25 lookup for CBB (ESPN weekly polls).

Unranked teams use rank ``UNRANKED_SENTINEL`` (40) — a domain convention for
"outside the ballot", not an imputed missing value.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RANK_DIR = PROJECT_ROOT / "data" / "supplemental" / "cbb-rankings"
UNRANKED_SENTINEL = 40.0

AP_FEATURE_COLUMNS: tuple[str, ...] = (
    "ap_rank_diff",
    "home_ap_rank",
    "away_ap_rank",
    "home_ap_ranked",
    "away_ap_ranked",
    "both_ap_ranked",
    "ap_top10_clash",
    "ap_known",
)


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


@lru_cache(maxsize=16)
def _load_season(season: int) -> pd.DataFrame | None:
    path = RANK_DIR / f"ap_{season}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "asof_date" not in frame.columns:
        return None
    frame["asof_date"] = pd.to_datetime(frame["asof_date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["asof_date", "rank"])
    frame["team_abbr"] = frame["team_abbr"].astype(str).str.lower()
    frame["team_id"] = frame["team_id"].astype(str)
    frame["team_norm"] = frame["team_name"].map(_norm)
    return frame


def lookup_rank(
    *,
    team_id: str,
    team_abbr: str,
    team_name: str,
    asof: date,
    season: int,
) -> float:
    """Return AP rank as-of morning of ``asof`` (most recent poll on/before)."""
    frame = _load_season(season)
    if frame is None or frame.empty:
        # No poll file yet for this season — treat all as unranked with ap_known=0 upstream.
        return UNRANKED_SENTINEL
    sub = frame[frame["asof_date"] <= asof]
    if sub.empty:
        return UNRANKED_SENTINEL
    poll_date = sub["asof_date"].max()
    day = frame[frame["asof_date"] == poll_date]
    tid = str(team_id or "")
    abbr = str(team_abbr or "").lower()
    tn = _norm(team_name)
    hit = day[day["team_id"] == tid] if tid else day.iloc[0:0]
    if hit.empty and abbr:
        hit = day[day["team_abbr"] == abbr]
    if hit.empty and tn:
        hit = day[day["team_norm"] == tn]
        if hit.empty:
            # prefix (mascot strip)
            for cand_norm, rank in zip(day["team_norm"], day["rank"], strict=False):
                cn = str(cand_norm)
                if tn.startswith(cn) or cn.startswith(tn):
                    return float(rank)
    if hit.empty:
        return UNRANKED_SENTINEL
    return float(hit.iloc[0]["rank"])


def ap_feature_dict(
    *,
    home_id: str,
    away_id: str,
    home_abbr: str,
    away_abbr: str,
    home_name: str,
    away_name: str,
    asof: date,
    season: int,
) -> dict[str, float]:
    frame = _load_season(season)
    known = 1.0 if frame is not None and not frame.empty else 0.0
    home_r = lookup_rank(
        team_id=home_id, team_abbr=home_abbr, team_name=home_name, asof=asof, season=season
    )
    away_r = lookup_rank(
        team_id=away_id, team_abbr=away_abbr, team_name=away_name, asof=asof, season=season
    )
    home_ranked = 1.0 if home_r < UNRANKED_SENTINEL else 0.0
    away_ranked = 1.0 if away_r < UNRANKED_SENTINEL else 0.0
    return {
        "ap_rank_diff": away_r - home_r,  # positive => home better ranked
        "home_ap_rank": home_r,
        "away_ap_rank": away_r,
        "home_ap_ranked": home_ranked,
        "away_ap_ranked": away_ranked,
        "both_ap_ranked": 1.0 if home_ranked and away_ranked else 0.0,
        "ap_top10_clash": 1.0 if home_r <= 10 and away_r <= 10 else 0.0,
        "ap_known": known,
    }
