"""Build leak-free NFL play-level EPA aggregates from nflverse PBP.

Downloads season parquet from nflverse-data releases, aggregates per game/team
EPA/play, success rate, explosive rate, and pass/rush EPA, then writes
data/supplemental/nfl-pbp/game_team_epa.csv (optional .build-cache copy).

Usage:
  python scripts/build_nfl_pbp_epa.py --copy-supplemental
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / ".build-cache" / "nfl-pbp"
SUPP_DIR = PROJECT_ROOT / "data" / "supplemental" / "nfl-pbp"
PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.parquet"
)
EXPLOSIVE_YARDS = 20.0
USER_AGENT = "Sports-Odds-Algorithms/2.0"


def _download_season(season: int, cache_dir: Path, *, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"play_by_play_{season}.parquet"
    if path.is_file() and not force and path.stat().st_size > 1_000_000:
        return path
    url = PBP_URL.format(season=season)
    print(f"downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        path.write_bytes(resp.read())
    return path


def _aggregate_season(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    wanted = [
        "game_id",
        "season",
        "posteam",
        "defteam",
        "epa",
        "success",
        "yards_gained",
        "play_type",
        "pass",
        "rush",
        "special_teams_play",
    ]
    frame = raw[[c for c in wanted if c in raw.columns]].copy()
    if "special_teams_play" not in frame.columns:
        frame["special_teams_play"] = 0
    if "success" not in frame.columns:
        frame["success"] = 0.0
    if "yards_gained" not in frame.columns:
        frame["yards_gained"] = 0.0
    if "season" not in frame.columns:
        try:
            frame["season"] = int(path.stem.split("_")[-1])
        except ValueError:
            frame["season"] = 0

    mask = (
        frame["posteam"].notna()
        & frame["epa"].notna()
        & (frame["special_teams_play"].fillna(0).astype(float) < 0.5)
    )
    if "play_type" in frame.columns:
        mask &= ~frame["play_type"].isin(["no_play", "qb_kneel", "qb_spike"])
    plays = frame.loc[mask].copy()
    plays["posteam"] = plays["posteam"].astype(str).str.lower().str.strip()
    plays["defteam"] = plays["defteam"].astype(str).str.lower().str.strip()
    plays["success"] = plays["success"].fillna(0).astype(float)
    plays["yards_gained"] = plays["yards_gained"].fillna(0).astype(float)
    plays["explosive"] = (plays["yards_gained"] >= EXPLOSIVE_YARDS).astype(float)

    off = (
        plays.groupby(["game_id", "season", "posteam"], as_index=False)
        .agg(
            epa_off=("epa", "mean"),
            sr_off=("success", "mean"),
            explosive_off=("explosive", "mean"),
            plays_off=("epa", "size"),
        )
    )
    if "pass" in plays.columns:
        pass_plays = plays.loc[plays["pass"].fillna(0).astype(float) > 0.5]
        pass_agg = (
            pass_plays.groupby(["game_id", "posteam"], as_index=False)["epa"]
            .mean()
            .rename(columns={"epa": "pass_epa_off"})
        )
        off = off.merge(pass_agg, on=["game_id", "posteam"], how="left")
    else:
        off["pass_epa_off"] = np.nan
    if "rush" in plays.columns:
        rush_plays = plays.loc[plays["rush"].fillna(0).astype(float) > 0.5]
        rush_agg = (
            rush_plays.groupby(["game_id", "posteam"], as_index=False)["epa"]
            .mean()
            .rename(columns={"epa": "rush_epa_off"})
        )
        off = off.merge(rush_agg, on=["game_id", "posteam"], how="left")
    else:
        off["rush_epa_off"] = np.nan

    defense = (
        plays.groupby(["game_id", "season", "defteam"], as_index=False)
        .agg(
            epa_def=("epa", "mean"),
            sr_def=("success", "mean"),
            plays_def=("epa", "size"),
        )
        .rename(columns={"defteam": "team"})
    )
    off = off.rename(columns={"posteam": "team"})
    merged = off.merge(defense, on=["game_id", "season", "team"], how="outer")
    for col in (
        "epa_off",
        "sr_off",
        "explosive_off",
        "pass_epa_off",
        "rush_epa_off",
        "epa_def",
        "sr_def",
    ):
        if col in merged.columns:
            merged[col] = merged[col].astype(float)
    merged["plays_off"] = merged.get("plays_off", pd.Series(0, index=merged.index)).fillna(0).astype(int)
    merged["plays_def"] = merged.get("plays_def", pd.Series(0, index=merged.index)).fillna(0).astype(int)
    return merged


def build(
    *,
    start_season: int,
    end_season: int,
    force: bool = False,
    copy_supplemental: bool = False,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        try:
            path = _download_season(season, CACHE_DIR, force=force)
            part = _aggregate_season(path)
            parts.append(part)
            print(f"  season {season}: {len(part)} team-games", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  season {season} failed: {exc}", flush=True)
    if not parts:
        raise SystemExit("no PBP seasons aggregated")
    out = pd.concat(parts, ignore_index=True)
    out_path = CACHE_DIR / "game_team_epa.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)}", flush=True)
    if copy_supplemental:
        SUPP_DIR.mkdir(parents=True, exist_ok=True)
        supp = SUPP_DIR / "game_team_epa.csv"
        out.to_csv(supp, index=False)
        print(f"copied {supp}", flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=1999)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-supplemental", action="store_true")
    args = parser.parse_args()
    build(
        start_season=args.start_season,
        end_season=args.end_season,
        force=args.force,
        copy_supplemental=args.copy_supplemental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
