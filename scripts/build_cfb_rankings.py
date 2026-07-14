"""Fetch ESPN AP / CFP weekly CFB rankings (historical Core API).

Writes data/supplemental/cfb-rankings/{ap,cfp}_{season}.csv

Usage:
  python scripts/build_cfb_rankings.py
  python scripts/build_cfb_rankings.py --start-season 2019 --end-season 2025
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

OUT_DIR = PROJECT_ROOT / "data" / "supplemental" / "cfb-rankings"
USER_AGENT = "Sports-Odds-Algorithms/2.0 (research; +https://github.com/SamuelLachance/Sports-Odds-Algorithms)"
SEASON_TYPE = 2  # regular season
# ESPN ranking poll ids (college-football)
POLLS = (("ap", 1), ("cfp", 9))
TEAM_ID_RE = re.compile(r"/teams/(\d+)")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _espn_teams() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    try:
        from web.season_games import _load_espn_team_ids

        for eid, abbr, display in _load_espn_team_ids("cfb"):
            out[str(eid)] = (str(abbr).lower(), str(display))
    except Exception:  # noqa: BLE001
        pass
    return out


def _week_count(season: int) -> int:
    url = (
        "https://sports.core.api.espn.com/v2/sports/football/"
        f"leagues/college-football/seasons/{season}/types/{SEASON_TYPE}/weeks"
    )
    try:
        data = _get_json(url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return 16
    items = data.get("items") or []
    return max(len(items), 1)


def _team_id_from_entry(entry: dict) -> str:
    team = entry.get("team") or {}
    if team.get("id"):
        return str(team["id"])
    ref = str(team.get("$ref") or "")
    m = TEAM_ID_RE.search(ref)
    return m.group(1) if m else ""


def fetch_poll_season(
    season: int,
    poll: str,
    ranking_id: int,
    *,
    force: bool = False,
    sleep_s: float = 0.12,
    team_map: dict[str, tuple[str, str]] | None = None,
) -> Path | None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{poll}_{season}.csv"
    if dest.is_file() and dest.stat().st_size > 2_000 and not force:
        print(f"skip existing {dest.name} ({dest.stat().st_size:,} bytes)", flush=True)
        return dest

    if team_map is None:
        team_map = _espn_teams()

    n_weeks = _week_count(season)
    rows: list[dict[str, object]] = []
    for week in range(1, n_weeks + 1):
        url = (
            "https://sports.core.api.espn.com/v2/sports/football/"
            f"leagues/college-football/seasons/{season}/types/{SEASON_TYPE}/"
            f"weeks/{week}/rankings/{ranking_id}?lang=en&region=us"
        )
        try:
            data = _get_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  {poll} {season} week {week}: skip ({exc})", flush=True)
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
        print(f"  {poll} {season} week {week}: {len(ranks)} ranked (asof={asof})", flush=True)
        time.sleep(sleep_s)

    if not rows:
        print(f"FAIL {poll} {season}: no ranking rows", flush=True)
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
    parser.add_argument("--start-season", type=int, default=2019)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    team_map = _espn_teams()
    print(f"espn team map: {len(team_map)} teams", flush=True)
    for season in range(args.start_season, args.end_season + 1):
        for poll, rid in POLLS:
            # CFP only starts mid-season historically; still try all weeks (empty weeks skipped)
            fetch_poll_season(
                season, poll, rid, force=args.force, team_map=team_map
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
