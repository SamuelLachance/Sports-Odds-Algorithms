"""Fetch ESPN AP Top 25 weekly rankings (historical Core API).

Uses sports.core.api.espn.com week rankings (correct as-of dates). Team objects
are usually ``$ref``-only; team id is parsed from the URL and joined to the
local ESPN CBB team list for abbr/name — no per-team HTTP chase.

Writes data/supplemental/cbb-rankings/ap_{season}.csv

Unranked => feature join uses rank 40 (outside top-25 sentinel).

Usage:
  python scripts/fetch_cbb_ap_rankings.py
  python scripts/fetch_cbb_ap_rankings.py --start-season 2018 --end-season 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "supplemental" / "cbb-rankings"
USER_AGENT = "Sports-Odds-Algorithms/2.0 (research; +https://github.com/SamuelLachance/Sports-Odds-Algorithms)"
RANKING_ID = 1
SEASON_TYPE = 2
TEAM_ID_RE = re.compile(r"/teams/(\d+)")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _espn_teams() -> dict[str, tuple[str, str]]:
    """espn team_id -> (abbr, displayName)."""
    from web.season_games import _load_espn_team_ids

    out: dict[str, tuple[str, str]] = {}
    for eid, abbr, display in _load_espn_team_ids("cbb"):
        out[str(eid)] = (str(abbr).lower(), str(display))
    return out


def _week_count(season: int) -> int:
    url = (
        "https://sports.core.api.espn.com/v2/sports/basketball/"
        f"leagues/mens-college-basketball/seasons/{season}/types/{SEASON_TYPE}/weeks"
    )
    try:
        data = _get_json(url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return 20
    items = data.get("items") or []
    return max(len(items), 1)


def _team_id_from_entry(entry: dict) -> str:
    team = entry.get("team") or {}
    if team.get("id"):
        return str(team["id"])
    ref = str(team.get("$ref") or "")
    m = TEAM_ID_RE.search(ref)
    return m.group(1) if m else ""


def fetch_season(
    season: int,
    *,
    force: bool = False,
    sleep_s: float = 0.12,
    team_map: dict[str, tuple[str, str]] | None = None,
) -> Path | None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"ap_{season}.csv"
    if dest.is_file() and dest.stat().st_size > 5_000 and not force:
        print(f"skip existing {dest.name} ({dest.stat().st_size:,} bytes)", flush=True)
        return dest

    if team_map is None:
        team_map = _espn_teams()

    n_weeks = _week_count(season)
    rows: list[dict[str, object]] = []
    for week in range(1, n_weeks + 1):
        url = (
            "https://sports.core.api.espn.com/v2/sports/basketball/"
            f"leagues/mens-college-basketball/seasons/{season}/types/{SEASON_TYPE}/"
            f"weeks/{week}/rankings/{RANKING_ID}?lang=en&region=us"
        )
        try:
            data = _get_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  week {week}: skip ({exc})", flush=True)
            time.sleep(sleep_s)
            continue
        asof = str(data.get("date") or data.get("lastUpdated") or "")[:10]
        ranks = data.get("ranks") or []
        if not ranks:
            time.sleep(sleep_s)
            continue
        for entry in ranks:
            rank = entry.get("current")
            if rank is None:
                continue
            tid = _team_id_from_entry(entry)
            abbr, name = team_map.get(tid, ("", ""))
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "asof_date": asof,
                    "team_id": tid,
                    "team_abbr": abbr,
                    "team_name": name,
                    "rank": int(rank),
                    "previous": entry.get("previous") if entry.get("previous") is not None else "",
                    "points": entry.get("points") if entry.get("points") is not None else "",
                }
            )
        print(f"  {season} week {week}: {len(ranks)} ranked (asof={asof})", flush=True)
        time.sleep(sleep_s)

    if not rows:
        print(f"FAIL {season}: no ranking rows", flush=True)
        return None
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "season",
                "week",
                "asof_date",
                "team_id",
                "team_abbr",
                "team_name",
                "rank",
                "previous",
                "points",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {dest} ({len(rows)} rows)", flush=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    team_map = _espn_teams()
    print(f"ESPN team map: {len(team_map)} teams", flush=True)
    ok = 0
    for season in range(args.start_season, args.end_season + 1):
        print(f"season {season}…", flush=True)
        if fetch_season(season, force=args.force, team_map=team_map):
            ok += 1
    print(f"done: {ok} seasons in {OUT_DIR}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
