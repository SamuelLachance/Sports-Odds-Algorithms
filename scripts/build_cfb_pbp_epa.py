"""Build leak-free CFB play-level EPA aggregates from cfbfastR PBP.

Downloads season parquet from sportsdataverse releases, aggregates per
game/team EPA + success rate, writes .build-cache/cfb-pbp/game_team_epa.csv
(optional supplemental copy). Includes ``date`` for PIT season-to-date joins.

Usage:
  python scripts/build_cfb_pbp_epa.py --copy-supplemental
  python scripts/build_cfb_pbp_epa.py --start-season 2022 --end-season 2025
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

CACHE_DIR = PROJECT_ROOT / ".build-cache" / "cfb-pbp"
SUPP_DIR = PROJECT_ROOT / "data" / "supplemental" / "cfb-pbp"
PBP_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "cfbfastR_cfb_pbp/play_by_play_{season}.parquet"
)
USER_AGENT = "Sports-Odds-Algorithms/2.0"


def _download_season(season: int, cache_dir: Path, *, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"play_by_play_{season}.parquet"
    if path.is_file() and not force and path.stat().st_size > 1_000_000:
        return path
    url = PBP_URL.format(season=season)
    print(f"downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        path.write_bytes(resp.read())
    return path


def _pick(frame: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _espn_name_to_abbr() -> dict[str, str]:
    try:
        from web.season_games import _load_espn_team_ids
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for _eid, abbr, display in _load_espn_team_ids("cfb"):
        key = _norm_name(display)
        if key:
            out[key] = str(abbr).lower()
        # also short common aliases
        short = _norm_name(str(abbr))
        if short:
            out.setdefault(short, str(abbr).lower())
    return out


def _to_abbr(raw: str, mapping: dict[str, str]) -> str:
    name = str(raw or "").lower().strip()
    if not name or name == "nan":
        return ""
    hit = mapping.get(_norm_name(name))
    if hit:
        return hit
    # prefix fallback (e.g. "Miami" vs "Miami (FL)")
    nn = _norm_name(name)
    for cand, abbr in mapping.items():
        if nn and cand and (nn.startswith(cand) or cand.startswith(nn)):
            return abbr
    return name  # last resort: keep full name (won't join, has_epa stays 0)


def _aggregate_season(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    game_col = _pick(raw, "game_id", "gameId", "id")
    season_col = _pick(raw, "season", "year")
    date_col = _pick(raw, "start_date", "game_date", "date", "startDate")
    pos_col = _pick(raw, "pos_team", "posteam", "offense", "offense_play", "pos_team_name")
    def_col = _pick(raw, "def_pos_team", "defteam", "defense", "defense_play", "def_pos_team_name")
    epa_col = _pick(raw, "EPA", "epa", "expected_points_added")
    sr_col = _pick(raw, "success", "Success")
    if not game_col or not pos_col or not epa_col:
        raise RuntimeError(f"missing required PBP columns in {path.name}: {list(raw.columns)[:40]}")

    name_map = _espn_name_to_abbr()
    frame = pd.DataFrame(
        {
            "game_id": raw[game_col].astype(str),
            "posteam": raw[pos_col].map(lambda v: _to_abbr(v, name_map)),
            "epa": pd.to_numeric(raw[epa_col], errors="coerce"),
        }
    )
    if def_col:
        frame["defteam"] = raw[def_col].map(lambda v: _to_abbr(v, name_map))
    else:
        frame["defteam"] = ""
    if season_col:
        frame["season"] = pd.to_numeric(raw[season_col], errors="coerce")
    else:
        try:
            frame["season"] = int(path.stem.split("_")[-1])
        except ValueError:
            frame["season"] = 0
    if date_col:
        frame["date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        frame["date"] = ""
    if sr_col:
        frame["success"] = pd.to_numeric(raw[sr_col], errors="coerce").fillna(0.0)
    else:
        frame["success"] = 0.0

    mask = frame["posteam"].notna() & frame["epa"].notna() & (frame["posteam"] != "") & (frame["posteam"] != "nan")
    plays = frame.loc[mask].copy()

    off = (
        plays.groupby(["game_id", "season", "date", "posteam"], as_index=False)
        .agg(epa_off=("epa", "mean"), sr_off=("success", "mean"), plays_off=("epa", "size"))
        .rename(columns={"posteam": "team"})
    )
    if plays["defteam"].astype(str).str.len().gt(0).any():
        defense = (
            plays.loc[plays["defteam"].astype(str).str.len() > 0]
            .groupby(["game_id", "season", "date", "defteam"], as_index=False)
            .agg(epa_def=("epa", "mean"), sr_def=("success", "mean"), plays_def=("epa", "size"))
            .rename(columns={"defteam": "team"})
        )
        merged = off.merge(defense, on=["game_id", "season", "date", "team"], how="outer")
    else:
        merged = off.copy()
        merged["epa_def"] = np.nan
        merged["sr_def"] = np.nan
        merged["plays_def"] = 0
    for col in ("epa_off", "sr_off", "epa_def", "sr_def"):
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
        raise SystemExit("no CFB PBP seasons aggregated")
    out = pd.concat(parts, ignore_index=True)
    out_path = CACHE_DIR / "game_team_epa.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)}", flush=True)
    if copy_supplemental:
        SUPP_DIR.mkdir(parents=True, exist_ok=True)
        supp = SUPP_DIR / "game_team_epa.csv"
        out.to_csv(supp, index=False)
        (SUPP_DIR / "README.md").write_text(
            "# CFB PBP EPA\n\nBuilt by `scripts/build_cfb_pbp_epa.py` from cfbfastR "
            "sportsdataverse parquet releases. Joined as season-to-date priors only "
            "(never same-game EPA).\n",
            encoding="utf-8",
        )
        print(f"copied {supp}", flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2022)
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
