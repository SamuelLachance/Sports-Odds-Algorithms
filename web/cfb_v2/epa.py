"""Attach cfbfastR game-team EPA as season-to-date priors (PIT-safe).

Same-game EPA is never used as a feature — only games strictly before the
current kickoff date update the rolling mean used for ``epa_*_diff``.
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


@lru_cache(maxsize=1)
def load_game_team_epa() -> pd.DataFrame:
    for path in EPA_PATHS:
        if path.is_file():
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["team"] = frame["team"].astype(str).str.lower().str.strip()
            if "date" in frame.columns:
                frame["date"] = frame["date"].astype(str).str[:10]
            return frame
    return pd.DataFrame()


def attach_epa_priors_to_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mutate games with prior EPA/SR diffs (has_epa=0 when lake missing)."""
    epa = load_game_team_epa()
    if epa.empty or "date" not in epa.columns:
        for game in games:
            game.setdefault("epa_off_diff", 0.0)
            game.setdefault("epa_def_diff", 0.0)
            game.setdefault("sr_off_diff", 0.0)
            game.setdefault("has_epa", 0.0)
        return games

    # Index rows by team for chronological scan
    by_team: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in epa.to_dict(orient="records"):
        team = str(row.get("team") or "").lower()
        day = str(row.get("date") or "")[:10]
        if not team or not day:
            continue
        by_team[team].append(
            {
                "date": day,
                "epa_off": float(row["epa_off"]) if pd.notna(row.get("epa_off")) else 0.0,
                "epa_def": float(row["epa_def"]) if pd.notna(row.get("epa_def")) else 0.0,
                "sr_off": float(row["sr_off"]) if pd.notna(row.get("sr_off")) else 0.0,
            }
        )
    for team in by_team:
        by_team[team].sort(key=lambda r: str(r["date"]))

    # Running sums for O(1) season-to-date means
    sums: dict[str, dict[str, float]] = {
        t: {"n": 0.0, "epa_off": 0.0, "epa_def": 0.0, "sr_off": 0.0} for t in by_team
    }
    ptr: dict[str, int] = {t: 0 for t in by_team}

    ordered = sorted(games, key=lambda g: (str(g.get("date") or ""), str(g.get("home") or "")))
    for game in ordered:
        day = str(game.get("date") or "")[:10]
        home = str(game.get("home") or "").lower()
        away = str(game.get("away") or "").lower()

        def _advance(team: str) -> None:
            rows = by_team.get(team) or []
            i = ptr.get(team, 0)
            while i < len(rows) and str(rows[i]["date"]) < day:
                sums[team]["n"] += 1.0
                sums[team]["epa_off"] += float(rows[i]["epa_off"])
                sums[team]["epa_def"] += float(rows[i]["epa_def"])
                sums[team]["sr_off"] += float(rows[i]["sr_off"])
                i += 1
            ptr[team] = i

        _advance(home)
        _advance(away)

        def _means(team: str) -> tuple[float, float, float, float] | None:
            s = sums.get(team)
            if not s or s["n"] < 1:
                return None
            n = s["n"]
            return s["epa_off"] / n, s["epa_def"] / n, s["sr_off"] / n, 1.0

        hm = _means(home)
        am = _means(away)
        if hm is None and am is None:
            game["epa_off_diff"] = 0.0
            game["epa_def_diff"] = 0.0
            game["sr_off_diff"] = 0.0
            game["has_epa"] = 0.0
            continue
        h_off, h_def, h_sr, _ = hm or (0.0, 0.0, 0.0, 0.0)
        a_off, a_def, a_sr, _ = am or (0.0, 0.0, 0.0, 0.0)
        game["epa_off_diff"] = h_off - a_off
        game["epa_def_diff"] = h_def - a_def
        game["sr_off_diff"] = h_sr - a_sr
        game["has_epa"] = 1.0 if (hm is not None or am is not None) else 0.0
    return games
