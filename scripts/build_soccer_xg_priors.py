"""Build PIT-safe soccer xG priors from Understat getLeagueData API.

Writes data/supplemental/soccer-xg/team_match_xg_priors.csv with rolling
prior-match xG for/against (current match excluded).

Usage:
  python scripts/build_soccer_xg_priors.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "supplemental" / "soccer-xg"

LEAGUES = {
    "EPL": ("E0",),
    "La_liga": ("SP1",),
    "Bundesliga": ("D1",),
    "Serie_A": ("I1",),
    "Ligue_1": ("F1",),
}


def _fetch_league_json(league: str, season: int) -> dict:
    url = f"https://understat.com/getLeagueData/{league}/{season}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Referer": f"https://understat.com/league/{league}/{season}",
        "X-Requested-With": "XMLHttpRequest",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def fetch_league_season(league: str, season: int) -> pd.DataFrame:
    """Expand Understat dates payload into match rows."""
    payload = _fetch_league_json(league, season)
    teams = payload.get("teams") or {}
    titles = {str(t["id"]): t.get("title") or "" for t in teams.values()}
    by_side: dict[tuple[str, str, str], float] = {}
    for team in teams.values():
        title = team.get("title") or ""
        for game in team.get("history") or []:
            day = str(game.get("date") or "")[:10]
            ha = str(game.get("h_a") or "").lower()
            if day and ha in {"h", "a"}:
                by_side[(day, title, ha)] = float(game.get("xG") or 0.0)

    rows: list[dict] = []
    dates = payload.get("dates") or []
    for item in dates:
        if not isinstance(item, dict):
            continue
        if str(item.get("isResult") or "").lower() not in {"true", "1"}:
            continue
        day = str(item.get("datetime") or item.get("date") or "")[:10]
        home_obj = item.get("h") or {}
        away_obj = item.get("a") or {}
        home = home_obj.get("title") or titles.get(str(home_obj.get("id") or ""), "")
        away = away_obj.get("title") or titles.get(str(away_obj.get("id") or ""), "")
        xg = item.get("xG")
        if isinstance(xg, dict):
            hx, ax = float(xg.get("h") or 0), float(xg.get("a") or 0)
        else:
            hx = float(by_side.get((day, home, "h"), 0.0))
            ax = float(by_side.get((day, away, "a"), 0.0))
        if day and home and away:
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "date": day,
                    "home": home,
                    "away": away,
                    "home_xg": hx,
                    "away_xg": ax,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["date", "home", "away"])


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def build_rolling_priors(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    matches = matches.sort_values(["date", "home", "away"]).reset_index(drop=True)
    state: dict[str, dict[str, float]] = {}
    out_rows: list[dict] = []
    for row in matches.to_dict(orient="records"):
        home = _norm(row["home"])
        away = _norm(row["away"])
        hs = state.setdefault(home, {"n": 0.0, "xf": 0.0, "xa": 0.0})
        as_ = state.setdefault(away, {"n": 0.0, "xf": 0.0, "xa": 0.0})
        out_rows.append(
            {
                "league": row["league"],
                "season": row["season"],
                "date": row["date"],
                "home": row["home"],
                "away": row["away"],
                "home_key": home,
                "away_key": away,
                "home_xg_for": (hs["xf"] / hs["n"]) if hs["n"] else None,
                "home_xg_against": (hs["xa"] / hs["n"]) if hs["n"] else None,
                "away_xg_for": (as_["xf"] / as_["n"]) if as_["n"] else None,
                "away_xg_against": (as_["xa"] / as_["n"]) if as_["n"] else None,
                "has_xg_home": 1.0 if hs["n"] else 0.0,
                "has_xg_away": 1.0 if as_["n"] else 0.0,
            }
        )
        hx = float(row["home_xg"])
        ax = float(row["away_xg"])
        hs["n"] += 1.0
        hs["xf"] += hx
        hs["xa"] += ax
        as_["n"] += 1.0
        as_["xf"] += ax
        as_["xa"] += hx
    return pd.DataFrame(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--sleep", type=float, default=0.35)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_matches: list[pd.DataFrame] = []
    for league in LEAGUES:
        parts: list[pd.DataFrame] = []
        for season in range(args.start_season, args.end_season + 1):
            try:
                part = fetch_league_season(league, season)
                if part.empty:
                    print(f"  {league} {season}: empty", flush=True)
                else:
                    parts.append(part)
                    print(f"  {league} {season}: {len(part)} matches", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  {league} {season}: fail {exc}", flush=True)
            time.sleep(args.sleep)
        if not parts:
            continue
        league_df = pd.concat(parts, ignore_index=True)
        dest = OUT_DIR / f"matches_{league}.csv"
        league_df.to_csv(dest, index=False)
        all_matches.append(league_df)
        print(f"wrote {dest}", flush=True)

    if not all_matches:
        print("ERROR: no Understat matches fetched", file=sys.stderr)
        return 1
    combined = pd.concat(all_matches, ignore_index=True)
    rolling = build_rolling_priors(combined)
    roll_path = OUT_DIR / "team_match_xg_priors.csv"
    rolling.to_csv(roll_path, index=False)
    (OUT_DIR / "README.md").write_text(
        "# Soccer xG priors (Understat)\n\n"
        "Built by `scripts/build_soccer_xg_priors.py` via `getLeagueData`. "
        "Rolling priors exclude the current match (PIT-safe).\n",
        encoding="utf-8",
    )
    print(f"wrote {roll_path} rows={len(rolling)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
