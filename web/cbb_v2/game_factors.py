"""Rolling Torvik game-factor / pregame features (PIT-safe).

Uses public toRvik-data CSVs under:
  data/supplemental/torvik-game-factors/
  data/supplemental/torvik-pregame/

When a season archive is missing (2024+), callers pass ESPN walk-forward
four-factor EWMAs as fallbacks — never leave NA cells.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FACTORS_DIR = PROJECT_ROOT / "data" / "supplemental" / "torvik-game-factors"
PREGAME_DIR = PROJECT_ROOT / "data" / "supplemental" / "torvik-pregame"

FACTOR_FEATURE_COLUMNS: tuple[str, ...] = (
    "tf_off_efg_diff",
    "tf_def_efg_diff",
    "tf_off_to_diff",
    "tf_off_or_diff",
    "tf_off_ftr_diff",
    "tf_tempo_diff",
    "tf_adj_o_diff",
    "tf_adj_d_diff",
    "tf_known",
)

PREGAME_FEATURE_COLUMNS: tuple[str, ...] = (
    "torvik_pregame_home_wp",
    "torvik_pregame_known",
)

_METRIC_COLS = ("off_efg", "def_efg", "off_to", "off_or", "off_ftr", "tempo", "adj_o", "adj_d")


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


@lru_cache(maxsize=16)
def _load_factors(season: int) -> pd.DataFrame | None:
    path = FACTORS_DIR / f"game_factors_{season}.csv"
    if not path.is_file():
        return None
    usecols = ["date", "team", *_METRIC_COLS]
    frame = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    if "date" not in frame.columns or "team" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["date", "team"])
    frame["team_norm"] = frame["team"].map(_norm)
    frame = frame.sort_values(["team_norm", "date"]).reset_index(drop=True)
    return frame


@lru_cache(maxsize=16)
def _factor_lookup(season: int) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """team_norm -> (dates ndarray, metric arrays) for O(log n) as-of slices."""
    frame = _load_factors(season)
    if frame is None or frame.empty:
        return {}
    out: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for tn, grp in frame.groupby("team_norm", sort=False):
        dates = np.array(grp["date"].tolist(), dtype=object)
        metrics = {c: grp[c].to_numpy(dtype=float) for c in _METRIC_COLS if c in grp.columns}
        if len(metrics) < len(_METRIC_COLS):
            continue
        out[str(tn)] = (dates, metrics)
    return out


@lru_cache(maxsize=16)
def _load_pregame(season: int) -> pd.DataFrame | None:
    path = PREGAME_DIR / f"pregame_{season}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(
        path,
        usecols=lambda c: c in {"date", "team1", "team2", "team1_wp", "team2_wp"},
        low_memory=False,
    )
    if "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["date"])
    frame["team1_norm"] = frame["team1"].map(_norm)
    frame["team2_norm"] = frame["team2"].map(_norm)
    return frame


@lru_cache(maxsize=16)
def _pregame_index(season: int) -> tuple[dict[tuple[date, str, str], float], dict[date, frozenset[str]]]:
    """Return (oriented wp index, norms-by-date)."""
    frame = _load_pregame(season)
    if frame is None or frame.empty:
        return {}, {}
    out: dict[tuple[date, str, str], float] = {}
    by_date: dict[date, set[str]] = {}
    for row in frame.itertuples(index=False):
        d = row.date
        t1 = str(row.team1_norm)
        t2 = str(row.team2_norm)
        out[(d, t1, t2)] = float(row.team1_wp)
        out[(d, t2, t1)] = float(row.team2_wp)
        bucket = by_date.setdefault(d, set())
        bucket.add(t1)
        bucket.add(t2)
    return out, {d: frozenset(v) for d, v in by_date.items()}


def _resolve_norm(name: str, abbr: str, candidates: set[str] | frozenset[str]) -> str | None:
    from web.cbb_v2.torvik_asof import _resolve_team_norm

    for year in (2023, 2022, 2021):
        tn = _resolve_team_norm(name, abbr, year)
        if tn and tn in candidates:
            return tn
    dn = _norm(name)
    if dn in candidates:
        return dn
    for c in sorted(candidates, key=len, reverse=True):
        if len(c) >= 4 and (dn.startswith(c) or c.startswith(dn)):
            return c
    return None


@lru_cache(maxsize=8192)
def _cached_team_norm(season: int, name: str, abbr: str) -> str | None:
    lookup = _factor_lookup(season)
    if not lookup:
        return None
    return _resolve_norm(name, abbr, frozenset(lookup.keys()))


def rolling_factor_row(
    *,
    team_name: str,
    team_abbr: str,
    asof: date,
    season: int,
    min_games: int = 3,
) -> dict[str, float] | None:
    """Mean of last 8 games' four factors for ``team`` strictly before ``asof``."""
    lookup = _factor_lookup(season)
    if not lookup:
        return None
    tn = _cached_team_norm(season, team_name, team_abbr)
    if not tn or tn not in lookup:
        return None
    dates, metrics = lookup[tn]
    lo, hi = 0, len(dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if dates[mid] < asof:
            lo = mid + 1
        else:
            hi = mid
    end = lo
    if end < min_games:
        return None
    start = max(0, end - 8)
    out: dict[str, float] = {}
    for col in _METRIC_COLS:
        chunk = metrics[col][start:end]
        if chunk.size == 0 or not np.isfinite(chunk).all():
            return None
        out[col] = float(chunk.mean())
    return out


def factor_feature_dict(
    *,
    home_name: str,
    away_name: str,
    home_abbr: str,
    away_abbr: str,
    asof: date,
    season: int,
    home_fallback: dict[str, float],
    away_fallback: dict[str, float],
) -> dict[str, float]:
    h = rolling_factor_row(team_name=home_name, team_abbr=home_abbr, asof=asof, season=season)
    a = rolling_factor_row(team_name=away_name, team_abbr=away_abbr, asof=asof, season=season)
    known = 1.0 if h is not None and a is not None else 0.0
    if h is None:
        h = home_fallback
    if a is None:
        a = away_fallback
    return {
        "tf_off_efg_diff": h["off_efg"] - a["off_efg"],
        "tf_def_efg_diff": h["def_efg"] - a["def_efg"],
        "tf_off_to_diff": h["off_to"] - a["off_to"],
        "tf_off_or_diff": h["off_or"] - a["off_or"],
        "tf_off_ftr_diff": h["off_ftr"] - a["off_ftr"],
        "tf_tempo_diff": h["tempo"] - a["tempo"],
        "tf_adj_o_diff": h["adj_o"] - a["adj_o"],
        "tf_adj_d_diff": h["adj_d"] - a["adj_d"],
        "tf_known": known,
    }


def pregame_home_wp(
    *,
    home_name: str,
    away_name: str,
    home_abbr: str,
    away_abbr: str,
    asof: date,
    season: int,
) -> tuple[float, float]:
    """Return (home_wp, known_flag). known=0 => caller should use Elo implied."""
    index, by_date = _pregame_index(season)
    if not index:
        return 0.5, 0.0
    day_norms = by_date.get(asof) or frozenset()
    if not day_norms:
        return 0.5, 0.0
    hn = _resolve_norm(home_name, home_abbr, day_norms)
    an = _resolve_norm(away_name, away_abbr, day_norms)
    if not hn or not an:
        return 0.5, 0.0
    wp = index.get((asof, hn, an))
    if wp is None:
        return 0.5, 0.0
    return float(wp), 1.0
