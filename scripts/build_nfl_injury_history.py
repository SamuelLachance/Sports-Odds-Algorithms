"""Build weekly NFL injury burden from nflverse public injury releases.

Writes data/supplemental/nfl-injuries/weekly_team_burden.csv with columns:
  season, week, team, injuries, out, ol_out, skill_out

PIT-safe: join on season+week before tip (nflverse weekly reports). No NA fills —
missing team-weeks simply leave injury_known=0 at attach time.

Usage:
  python scripts/build_nfl_injury_history.py --start-season 2009 --end-season 2025
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "supplemental" / "nfl-injuries"
CACHE_DIR = PROJECT_ROOT / ".build-cache" / "nfl-injuries"
URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/"
    "injuries_{season}.csv"
)
USER_AGENT = "Sports-Odds-Algorithms/2.0"

OL_POS = frozenset({"T", "G", "C", "OT", "OG", "OL"})
SKILL_POS = frozenset({"QB", "RB", "WR", "TE", "FB", "HB"})
OUT_STATUSES = frozenset({"out", "doubtful"})


def _download(season: int, force: bool = False) -> Path | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"injuries_{season}.csv"
    if path.is_file() and not force and path.stat().st_size > 1000:
        return path
    url = URL.format(season=season)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    except OSError as exc:
        print(f"  season {season}: skip ({exc})", flush=True)
        return None
    return path


def _burden(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "week", "team", "injuries", "out", "ol_out", "skill_out"]
        )
    work = frame.copy()
    work["team"] = work["team"].astype(str).str.lower().str.strip()
    work["team"] = work["team"].replace({"wsh": "was", "lar": "la", "jac": "jax"})
    work["position"] = work.get("position", pd.Series("", index=work.index)).astype(str).str.upper()
    status = (
        work.get("report_status", pd.Series("", index=work.index))
        .astype(str)
        .str.lower()
        .str.strip()
    )
    work["is_out"] = status.isin(OUT_STATUSES).astype(float)
    # Questionable counts half toward burden only (still real observation).
    work.loc[status == "questionable", "is_out"] = 0.5
    work["is_ol"] = work["position"].isin(OL_POS).astype(float) * work["is_out"]
    work["is_skill"] = work["position"].isin(SKILL_POS).astype(float) * work["is_out"]
    agg = (
        work.groupby(["season", "week", "team"], as_index=False)
        .agg(
            injuries=("is_out", "size"),
            out=("is_out", "sum"),
            ol_out=("is_ol", "sum"),
            skill_out=("is_skill", "sum"),
        )
    )
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parts: list[pd.DataFrame] = []
    for season in range(args.start_season, args.end_season + 1):
        path = _download(season, force=args.force)
        if path is None:
            continue
        try:
            raw = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            print(f"  season {season}: read fail {exc}", flush=True)
            continue
        # Regular season + playoffs only
        if "game_type" in raw.columns:
            raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
        part = _burden(raw)
        parts.append(part)
        print(f"  season {season}: {len(part)} team-weeks", flush=True)

    if not parts:
        print("ERROR: no injury seasons downloaded", flush=True)
        return 1

    out = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "weekly_team_burden.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest} rows={len(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
