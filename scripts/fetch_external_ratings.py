#!/usr/bin/env python3
"""Fetch/scrape external game ratings and write data/ratings/*.csv files.

Sources per sport:
  NBA/WNBA/CBB — 2kratings.com team rosters
  NFL/CFB      — maddenratings.com team rosters (+ fuzzy name search)
  NHL/NCAAH    — nhlratings.net team rosters
  MLB/college  — theshowratings.com team rosters
  Soccer       — ea.com EA FC 26 full database (~18k players, paginated)

Usage:
  python scripts/fetch_external_ratings.py
  python scripts/fetch_external_ratings.py --leagues epl,nhl,nba
  python scripts/fetch_external_ratings.py --prefetch-fc
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import LEAGUE_PROFILES, SUPPORTED_LEAGUES
from web.sports_db.espn_fetch import fetch_team_roster
from web.sports_db.external_ratings import (
    LEAGUE_RATING_FILES,
    RATINGS_DIR,
    clear_rating_cache,
    match_external_rating,
    normalize_player_name,
    normalize_team_abbr,
)
from web.sports_db.normalize import parse_roster
from web.sports_db.rating_scrape import (
    REQUEST_DELAY,
    fetch_ea_fc_all_players,
    scrape_league_team_players,
    scrape_source_for_league,
)
from web.team_service import fetch_espn_team_ids

CSV_FIELDS = ["player_name", "team_abbr", "position", "overall", "espn_id", "rating_source", "rating_year"]

# One CSV per league file — merge all leagues sharing a file.
FILE_LEAGUES: dict[str, list[str]] = {}
for _league, _fname in LEAGUE_RATING_FILES.items():
    FILE_LEAGUES.setdefault(_fname, []).append(_league)


def _load_existing(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            key = (
                normalize_player_name(record.get("player_name") or ""),
                normalize_team_abbr(record.get("team_abbr")),
            )
            if key[0]:
                rows[key] = record
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["team_abbr"], item["player_name"])):
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _prefetch_soccer() -> int:
    players = fetch_ea_fc_all_players()
    print(f"  EA FC database: {len(players)} players cached")
    return len(players)


def fetch_league_file(
    filename: str,
    leagues: list[str],
    *,
    max_teams: int | None = None,
    sleep: float = REQUEST_DELAY,
) -> dict[str, int]:
    out_path = RATINGS_DIR / filename
    merged: dict[tuple[str, str], dict[str, str]] = _load_existing(out_path)
    stats = {
        "teams": 0,
        "players": 0,
        "matched": 0,
        "unmatched": 0,
        "existing": len(merged),
    }

    for league in leagues:
        league = league.lower()
        if league not in LEAGUE_PROFILES:
            continue
        source = scrape_source_for_league(league) or "curated"
        team_ids = fetch_espn_team_ids(league)
        abbrs = sorted(team_ids.keys())
        if max_teams is not None:
            abbrs = abbrs[:max_teams]

        for abbr in abbrs:
            espn_id = team_ids.get(abbr)
            if not espn_id:
                continue
            payload = fetch_team_roster(league, espn_id)
            roster = parse_roster(payload)
            if not roster:
                continue
            stats["teams"] += 1
            scrape_league_team_players(league, abbr)

            for player in roster:
                name = (player.get("name") or "").strip()
                if not name:
                    continue
                pid = str(player.get("id") or "")
                position = (player.get("position") or "").strip().upper()
                key = (normalize_player_name(name), normalize_team_abbr(abbr, league))

                hit = match_external_rating(
                    league,
                    name,
                    team_abbr=abbr,
                    espn_id=pid,
                    nationality=str(player.get("nationality") or player.get("country") or ""),
                )
                if not hit or not hit.matched:
                    stats["unmatched"] += 1
                    continue

                merged[key] = {
                    "player_name": name,
                    "team_abbr": normalize_team_abbr(abbr, league),
                    "position": position,
                    "overall": str(hit.rating),
                    "espn_id": pid,
                    "rating_source": hit.rating_source,
                    "rating_year": str(hit.rating_year or 2026),
                }
                stats["matched"] += 1
                stats["players"] += 1
            if sleep:
                time.sleep(sleep)

    _write_csv(out_path, list(merged.values()))
    clear_rating_cache()
    stats["total_rows"] = len(merged)
    stats["source"] = source
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch external player ratings into data/ratings/")
    parser.add_argument("--leagues", default="", help="Comma-separated league ids (default: all supported)")
    parser.add_argument("--max-teams", type=int, default=None, help="Limit teams per league (debug)")
    parser.add_argument("--sleep", type=float, default=REQUEST_DELAY, help="Delay between team scrapes")
    parser.add_argument("--prefetch-fc", action="store_true", help="Only download EA FC full database")
    args = parser.parse_args()

    if args.prefetch_fc:
        count = _prefetch_soccer()
        print(f"Done. {count} EA FC players cached.")
        return 0

    if args.leagues.strip():
        selected = {part.strip().lower() for part in args.leagues.split(",") if part.strip()}
    else:
        selected = set(SUPPORTED_LEAGUES)

    soccer_selected = any(
        LEAGUE_PROFILES.get(league, {}).get("category") == "soccer" for league in selected
    )
    if soccer_selected:
        print("Prefetching EA FC 26 full ratings database...")
        _prefetch_soccer()

    files_to_build: dict[str, list[str]] = {}
    for league in selected:
        fname = LEAGUE_RATING_FILES.get(league)
        if fname:
            files_to_build.setdefault(fname, [])
            if league not in files_to_build[fname]:
                files_to_build[fname].append(league)

    print(f"Fetching ratings for {len(files_to_build)} CSV files -> {RATINGS_DIR}")
    total_matched = 0
    total_unmatched = 0
    for filename, leagues in sorted(files_to_build.items()):
        stats = fetch_league_file(
            filename,
            leagues,
            max_teams=args.max_teams,
            sleep=args.sleep,
        )
        total_matched += stats.get("matched", 0)
        total_unmatched += stats.get("unmatched", 0)
        pct = (
            round(100.0 * stats["matched"] / (stats["matched"] + stats["unmatched"]), 1)
            if stats["matched"] + stats["unmatched"]
            else 0.0
        )
        print(
            f"  {filename}: {stats['total_rows']} rows ({stats['teams']} teams) "
            f"— {pct}% matched ({stats['matched']} ok, {stats['unmatched']} miss)"
        )

    print(f"Done. {total_matched} matched, {total_unmatched} unmatched across selected leagues.")
    return 0 if total_unmatched == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
