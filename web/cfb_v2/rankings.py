"""PIT AP / CFP Top 25 lookup for CFB (ESPN weekly polls).

Unranked teams use rank ``UNRANKED_SENTINEL`` (40). Missing poll files →
``has_rank=0`` / ``ap_known=0`` with neutral diffs (no NaNs).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RANK_DIR = PROJECT_ROOT / "data" / "supplemental" / "cfb-rankings"
UNRANKED_SENTINEL = 40.0


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


@lru_cache(maxsize=32)
def _load_poll(season: int, poll: str) -> pd.DataFrame | None:
    path = RANK_DIR / f"{poll}_{season}.csv"
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
    team_abbr: str,
    team_name: str = "",
    team_id: str = "",
    asof: date,
    season: int,
    poll: str = "ap",
) -> float:
    frame = _load_poll(season, poll)
    if frame is None or frame.empty:
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
            for cand_norm, rank in zip(day["team_norm"], day["rank"], strict=False):
                cn = str(cand_norm)
                if tn and (tn.startswith(cn) or cn.startswith(tn)):
                    return float(rank)
    if hit.empty:
        return UNRANKED_SENTINEL
    return float(hit.iloc[0]["rank"])


def rank_feature_dict(
    *,
    home_abbr: str,
    away_abbr: str,
    asof: date,
    season: int,
    home_name: str = "",
    away_name: str = "",
    home_id: str = "",
    away_id: str = "",
) -> dict[str, float]:
    ap_frame = _load_poll(season, "ap")
    cfp_frame = _load_poll(season, "cfp")
    ap_known = 1.0 if ap_frame is not None and not ap_frame.empty else 0.0
    has_rank = 1.0 if ap_known or (cfp_frame is not None and not cfp_frame.empty) else 0.0
    home_ap = lookup_rank(
        team_abbr=home_abbr, team_name=home_name, team_id=home_id, asof=asof, season=season, poll="ap"
    )
    away_ap = lookup_rank(
        team_abbr=away_abbr, team_name=away_name, team_id=away_id, asof=asof, season=season, poll="ap"
    )
    home_cfp = lookup_rank(
        team_abbr=home_abbr, team_name=home_name, team_id=home_id, asof=asof, season=season, poll="cfp"
    )
    away_cfp = lookup_rank(
        team_abbr=away_abbr, team_name=away_name, team_id=away_id, asof=asof, season=season, poll="cfp"
    )
    return {
        "ap_rank_diff": away_ap - home_ap,
        "ap_known": ap_known,
        "cfp_rank_diff": away_cfp - home_cfp,
        "has_rank": has_rank,
    }


def attach_rankings_to_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mutate games with AP/CFP diffs (PIT as-of game date)."""
    for game in games:
        day_iso = str(game.get("date") or "")[:10]
        if not day_iso:
            continue
        try:
            asof = date.fromisoformat(day_iso)
        except ValueError:
            continue
        season = int(game.get("season") or asof.year)
        feats = rank_feature_dict(
            home_abbr=str(game.get("home") or ""),
            away_abbr=str(game.get("away") or ""),
            home_name=str(game.get("home_name") or ""),
            away_name=str(game.get("away_name") or ""),
            asof=asof,
            season=season,
        )
        game.update(feats)
    return games
