"""Attach cfbfastR EPA priors (PIT-safe): form EWMAs + opponent-adjusted ratings.

Same-game EPA is never used as a feature — only games strictly before the
current kickoff date feed the features:

- ``epa_off_diff`` / ``epa_def_diff`` / ``sr_off_diff`` (and the newer
  ``sr_def_diff`` / ``pace_plays_diff`` / ``epa_pass_off_diff`` /
  ``epa_rush_off_diff``): recency EWMAs (alpha 0.20) of garbage-time-filtered
  per-game values, decayed 0.6 toward the running league mean at season
  rollover (the old implementation used never-resetting career means).
- ``adj_epa_off_diff`` / ``adj_epa_def_diff``: latest as-of opponent-adjusted
  ridge ratings from ``team_adjusted_epa.csv`` (see
  ``scripts/build_cfb_pbp_epa.py``), the SP+/Torvik-style schedule-corrected
  strength signal.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EPA_PATHS = (
    PROJECT_ROOT / "data" / "supplemental" / "cfb-pbp" / "game_team_epa.csv",
    PROJECT_ROOT / ".build-cache" / "cfb-pbp" / "game_team_epa.csv",
)
ADJ_PATHS = (
    PROJECT_ROOT / "data" / "supplemental" / "cfb-pbp" / "team_adjusted_epa.csv",
    PROJECT_ROOT / ".build-cache" / "cfb-pbp" / "team_adjusted_epa.csv",
)

ALPHA_EPA = 0.20
SEASON_DECAY = 0.60

_EWMA_FIELDS = (
    "epa_off",
    "epa_def",
    "sr_off",
    "sr_def",
    "plays_off",
    "epa_pass_off",
    "epa_rush_off",
)

_ZERO_FEATURES = (
    "epa_off_diff",
    "epa_def_diff",
    "sr_off_diff",
    "sr_def_diff",
    "pace_plays_diff",
    "epa_pass_off_diff",
    "epa_rush_off_diff",
    "adj_epa_off_diff",
    "adj_epa_def_diff",
    "has_epa",
    "has_adj_epa",
)


def _load_first(paths: tuple[Path, ...]) -> pd.DataFrame:
    for path in paths:
        if path.is_file():
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["team"] = frame["team"].astype(str).str.lower().str.strip()
            if "date" in frame.columns:
                frame["date"] = frame["date"].astype(str).str[:10]
            return frame
    return pd.DataFrame()


@lru_cache(maxsize=1)
def load_game_team_epa() -> pd.DataFrame:
    return _load_first(EPA_PATHS)


@lru_cache(maxsize=1)
def load_team_adjusted_epa() -> pd.DataFrame:
    return _load_first(ADJ_PATHS)


class _TeamEwma:
    __slots__ = ("values", "n", "season")

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.n = 0
        self.season = -1

    def roll(self, season: int, league_mean: dict[str, float]) -> None:
        if self.season == season:
            return
        if self.season >= 0 and self.values:
            for key, val in self.values.items():
                mean = league_mean.get(key, 0.0)
                self.values[key] = mean + SEASON_DECAY * (val - mean)
        self.season = season

    def update(self, row: dict[str, Any]) -> None:
        for key in _EWMA_FIELDS:
            raw = row.get(key)
            if raw is None or pd.isna(raw):
                continue
            val = float(raw)
            prev = self.values.get(key)
            self.values[key] = val if prev is None else ALPHA_EPA * val + (1.0 - ALPHA_EPA) * prev
        self.n += 1


def attach_epa_priors_to_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mutate games with prior EPA/SR/adjusted diffs (has_epa=0 when lake missing)."""
    epa = load_game_team_epa()
    if epa.empty or "date" not in epa.columns:
        for game in games:
            for key in _ZERO_FEATURES:
                game.setdefault(key, 0.0)
        return games

    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in epa.to_dict(orient="records"):
        team = str(row.get("team") or "").lower()
        day = str(row.get("date") or "")[:10]
        if not team or not day or day == "nan":
            continue
        row["date"] = day
        by_team[team].append(row)
    for team in by_team:
        by_team[team].sort(key=lambda r: str(r["date"]))

    adj = load_team_adjusted_epa()
    adj_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not adj.empty and {"date", "adj_epa_off", "adj_epa_def"}.issubset(adj.columns):
        for row in adj.to_dict(orient="records"):
            team = str(row.get("team") or "").lower()
            day = str(row.get("date") or "")[:10]
            if not team or not day or day == "nan":
                continue
            adj_by_team[team].append(
                {
                    "date": day,
                    "adj_epa_off": float(row["adj_epa_off"]) if pd.notna(row.get("adj_epa_off")) else 0.0,
                    "adj_epa_def": float(row["adj_epa_def"]) if pd.notna(row.get("adj_epa_def")) else 0.0,
                }
            )
        for team in adj_by_team:
            adj_by_team[team].sort(key=lambda r: str(r["date"]))

    # Running league means for season-rollover decay targets.
    league_sum: dict[str, float] = {k: 0.0 for k in _EWMA_FIELDS}
    league_n = 0.0
    ewma: dict[str, _TeamEwma] = defaultdict(_TeamEwma)
    ptr: dict[str, int] = defaultdict(int)
    adj_state: dict[str, dict[str, float]] = {}
    adj_ptr: dict[str, int] = defaultdict(int)

    def _season_of(day: str) -> int:
        try:
            year = int(day[:4])
            month = int(day[5:7])
        except ValueError:
            return -1
        return year if month >= 7 else year - 1

    ordered = sorted(games, key=lambda g: (str(g.get("date") or ""), str(g.get("home") or "")))
    for game in ordered:
        day = str(game.get("date") or "")[:10]
        season = _season_of(day)
        home = str(game.get("home") or "").lower()
        away = str(game.get("away") or "").lower()

        def _advance(team: str) -> None:
            nonlocal league_n
            rows = by_team.get(team) or []
            i = ptr[team]
            state = ewma[team]
            while i < len(rows) and str(rows[i]["date"]) < day:
                row_season = _season_of(str(rows[i]["date"]))
                league_mean = (
                    {k: league_sum[k] / league_n for k in _EWMA_FIELDS} if league_n else {}
                )
                state.roll(row_season, league_mean)
                state.update(rows[i])
                for key in _EWMA_FIELDS:
                    raw = rows[i].get(key)
                    if raw is not None and not pd.isna(raw):
                        league_sum[key] += float(raw)
                league_n += 1.0
                i += 1
            ptr[team] = i
            arow = adj_by_team.get(team) or []
            j = adj_ptr[team]
            while j < len(arow) and str(arow[j]["date"]) < day:
                adj_state[team] = arow[j]
                j += 1
            adj_ptr[team] = j

        _advance(home)
        _advance(away)

        hs, as_ = ewma.get(home), ewma.get(away)
        h_ok = hs is not None and hs.n > 0
        a_ok = as_ is not None and as_.n > 0

        def _val(state: _TeamEwma | None, ok: bool, key: str) -> float:
            if not ok or state is None:
                return 0.0
            return float(state.values.get(key, 0.0))

        game["epa_off_diff"] = _val(hs, h_ok, "epa_off") - _val(as_, a_ok, "epa_off")
        game["epa_def_diff"] = _val(hs, h_ok, "epa_def") - _val(as_, a_ok, "epa_def")
        game["sr_off_diff"] = _val(hs, h_ok, "sr_off") - _val(as_, a_ok, "sr_off")
        game["sr_def_diff"] = _val(hs, h_ok, "sr_def") - _val(as_, a_ok, "sr_def")
        game["pace_plays_diff"] = _val(hs, h_ok, "plays_off") - _val(as_, a_ok, "plays_off")
        game["epa_pass_off_diff"] = _val(hs, h_ok, "epa_pass_off") - _val(as_, a_ok, "epa_pass_off")
        game["epa_rush_off_diff"] = _val(hs, h_ok, "epa_rush_off") - _val(as_, a_ok, "epa_rush_off")
        game["has_epa"] = 1.0 if (h_ok or a_ok) else 0.0

        ha, aa = adj_state.get(home), adj_state.get(away)
        if ha is not None and aa is not None:
            game["adj_epa_off_diff"] = float(ha["adj_epa_off"]) - float(aa["adj_epa_off"])
            game["adj_epa_def_diff"] = float(ha["adj_epa_def"]) - float(aa["adj_epa_def"])
            game["has_adj_epa"] = 1.0
        else:
            game["adj_epa_off_diff"] = 0.0
            game["adj_epa_def_diff"] = 0.0
            game["has_adj_epa"] = 0.0
    return games
