"""Build prior-season Baseball Savant expected-stats priors (PIT-safe).

Public CSV (Baseball Savant expected statistics leaderboard).
Season-end numbers are only attached as **previous-season** priors.

Outputs under data/supplemental/mlb-statcast/:
  pitcher_prev_xera.csv
  team_prev_xwoba.csv

Usage:
  python scripts/build_mlb_statcast_priors.py
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_v2.statsapi_data import fetch_team_ids  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "supplemental" / "mlb-statcast"
UA = {"User-Agent": "Mozilla/5.0 (compatible; Sports-Odds-Algorithms/mlb-statcast)"}


def _fetch_csv(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as resp:
        text = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(text))


def fetch_pitcher_year(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=pitcher&year={year}&position=&team=&min=q&csv=true"
    )
    frame = _fetch_csv(url)
    cols = {c.lower().strip().strip('"'): c for c in frame.columns}
    pid_col = cols.get("player_id")
    xera_col = cols.get("xera")
    if pid_col is None or xera_col is None:
        raise RuntimeError(f"unexpected Savant pitcher columns: {list(frame.columns)}")
    out = pd.DataFrame(
        {
            "player_id": pd.to_numeric(frame[pid_col], errors="coerce"),
            "season": year,
            "xera": pd.to_numeric(frame[xera_col], errors="coerce"),
        }
    ).dropna()
    out["player_id"] = out["player_id"].astype(int)
    return out


def fetch_team_batter_year(year: int, team_id: int) -> dict | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=batter&year={year}&position=&team={team_id}&min=1&csv=true"
    )
    frame = _fetch_csv(url)
    if frame.empty:
        return None
    cols = {c.lower().strip().strip('"'): c for c in frame.columns}
    xwoba_col = cols.get("est_woba")
    pa_col = cols.get("pa")
    if xwoba_col is None:
        return None
    xw = pd.to_numeric(frame[xwoba_col], errors="coerce")
    pa = pd.to_numeric(frame[pa_col], errors="coerce") if pa_col else pd.Series([1.0] * len(frame))
    mask = xw.notna() & pa.notna() & (pa > 0)
    if not mask.any():
        return None
    w = pa[mask]
    return {
        "team_id": int(team_id),
        "season": year,
        "xwoba": float((xw[mask] * w).sum() / w.sum()),
        "pa": float(w.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2017)
    parser.add_argument("--end-season", type=int, default=date.today().year - 1)
    args = parser.parse_args()

    pitcher_frames = []
    team_rows = []
    for year in range(args.start_season, args.end_season + 1):
        try:
            p = fetch_pitcher_year(year)
            pitcher_frames.append(p)
            print(f"pitcher {year}: {len(p)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"pitcher {year} FAIL {exc}", flush=True)
        try:
            team_ids = fetch_team_ids(year)
        except OSError as exc:
            print(f"team_ids {year} FAIL {exc}", flush=True)
            continue
        for tid in team_ids:
            try:
                row = fetch_team_batter_year(year, tid)
                if row:
                    team_rows.append(row)
                time.sleep(0.15)
            except Exception as exc:  # noqa: BLE001
                print(f"team {tid} {year} FAIL {exc}", flush=True)
                time.sleep(0.5)
        print(f"team bat {year}: {sum(1 for r in team_rows if r['season']==year)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if pitcher_frames:
        pitchers = pd.concat(pitcher_frames, ignore_index=True)
        pitchers["for_season"] = pitchers["season"] + 1
        pitchers.to_csv(OUT_DIR / "pitcher_prev_xera.csv", index=False)
        print(f"wrote pitcher priors {len(pitchers)}", flush=True)
    if team_rows:
        teams = pd.DataFrame(team_rows)
        teams["for_season"] = teams["season"] + 1
        teams.to_csv(OUT_DIR / "team_prev_xwoba.csv", index=False)
        print(f"wrote team priors {len(teams)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
