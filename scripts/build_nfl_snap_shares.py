"""Build per-game team offensive snap-share continuity from nflverse snap counts.

Writes data/supplemental/nfl-snaps/game_team_snaps.csv with leak-free join keys.
Features use prior-game EWMA in the feature engine after attach.

Usage:
  python scripts/build_nfl_snap_shares.py --copy-supplemental
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / ".build-cache" / "nfl-snaps"
SUPP_DIR = PROJECT_ROOT / "data" / "supplemental" / "nfl-snaps"
URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/"
    "snap_counts_{season}.parquet"
)
USER_AGENT = "Sports-Odds-Algorithms/2.0"
OL_POS = {"T", "G", "C", "OT", "OG", "OL"}
WR_POS = {"WR"}
SKILL_POS = {"WR", "TE", "RB", "FB"}


def _download(season: int, *, force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"snap_counts_{season}.parquet"
    if path.is_file() and not force and path.stat().st_size > 50_000:
        return path
    url = URL.format(season=season)
    print(f"downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())
    return path


def _aggregate_season(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    need = {"game_id", "team", "position", "offense_pct"}
    if not need.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["team"] = frame["team"].astype(str).str.lower().str.strip()
    frame["position"] = frame["position"].astype(str).str.upper().str.strip()
    frame["offense_pct"] = pd.to_numeric(frame["offense_pct"], errors="coerce").fillna(0.0)
    # Top WR snap share = max offense_pct among WRs
    wr = (
        frame.loc[frame["position"].isin(WR_POS)]
        .groupby(["game_id", "team"], as_index=False)["offense_pct"]
        .max()
        .rename(columns={"offense_pct": "wr1_snap_share"})
    )
    skill = (
        frame.loc[frame["position"].isin(SKILL_POS)]
        .groupby(["game_id", "team"], as_index=False)["offense_pct"]
        .mean()
        .rename(columns={"offense_pct": "skill_snap_share"})
    )
    ol = frame.loc[frame["position"].isin(OL_POS)].copy()
    if not ol.empty:
        # Stability: mean offense_pct among OL with >=50% snaps (starters)
        starters = ol.loc[ol["offense_pct"] >= 0.5]
        ol_stab = (
            starters.groupby(["game_id", "team"], as_index=False)["offense_pct"]
            .mean()
            .rename(columns={"offense_pct": "ol_starter_share"})
        )
        ol_count = (
            starters.groupby(["game_id", "team"], as_index=False)
            .size()
            .rename(columns={"size": "ol_starters"})
        )
        ol_out = ol_stab.merge(ol_count, on=["game_id", "team"], how="outer")
    else:
        ol_out = pd.DataFrame(columns=["game_id", "team", "ol_starter_share", "ol_starters"])
    season = int(path.stem.split("_")[-1]) if "_" in path.stem else 0
    out = wr.merge(skill, on=["game_id", "team"], how="outer").merge(
        ol_out, on=["game_id", "team"], how="outer"
    )
    out["season"] = season
    return out


def build(*, start: int, end: int, force: bool = False, copy_supplemental: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    for season in range(start, end + 1):
        try:
            path = _download(season, force=force)
            part = _aggregate_season(path)
            if part.empty:
                print(f"  season {season}: empty", flush=True)
                continue
            parts.append(part)
            print(f"  season {season}: {len(part)} team-games", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  season {season} failed: {exc}", flush=True)
    if not parts:
        raise SystemExit("no snap seasons aggregated")
    out = pd.concat(parts, ignore_index=True)
    out_path = CACHE_DIR / "game_team_snaps.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)}", flush=True)
    if copy_supplemental:
        SUPP_DIR.mkdir(parents=True, exist_ok=True)
        supp = SUPP_DIR / "game_team_snaps.csv"
        out.to_csv(supp, index=False)
        print(f"copied {supp}", flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2012)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-supplemental", action="store_true")
    args = parser.parse_args()
    build(
        start=args.start_season,
        end=args.end_season,
        force=args.force,
        copy_supplemental=args.copy_supplemental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
