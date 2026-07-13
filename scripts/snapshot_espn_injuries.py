"""Snapshot ESPN NFL injury lists (point-in-time for future training joins).

Stores daily JSON under data/supplemental/nfl-injuries/YYYY-MM-DD.json.
Historical backfill is best-effort — ESPN usually returns current slate only.

Usage:
  python scripts/snapshot_espn_injuries.py
  python scripts/snapshot_espn_injuries.py --league nfl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.availability_signals import (  # noqa: E402
    _fetch_json,
    _injury_status_text,
    _status_means_out,
)

OUT_ROOT = PROJECT_ROOT / "data" / "supplemental" / "nfl-injuries"
ESPN_NFL_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
OL_HINTS = ("tackle", "guard", "center", "offensive line", "ot", "og", "ol", "c")
SKILL_HINTS = ("wide receiver", "tight end", "running back", "wr", "te", "rb", "qb")


def _position_bucket(pos: str) -> str:
    p = (pos or "").strip().lower()
    if any(h in p for h in OL_HINTS):
        return "ol"
    if any(h in p for h in SKILL_HINTS):
        return "skill"
    return "other"


def _team_injuries(team_id: str) -> list[dict[str, Any]]:
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    payload = _fetch_json(url)
    if not isinstance(payload, dict):
        return []
    items = payload.get("items") or []
    out: list[dict[str, Any]] = []
    for stub in items[:40]:
        href = stub.get("$ref") if isinstance(stub, dict) else None
        detail = _fetch_json(href) if href else stub
        if not isinstance(detail, dict):
            continue
        athlete = detail.get("athlete") if isinstance(detail.get("athlete"), dict) else {}
        name = str(athlete.get("displayName") or athlete.get("fullName") or "")
        pos = ""
        pos_block = athlete.get("position")
        if isinstance(pos_block, dict):
            pos = str(pos_block.get("abbreviation") or pos_block.get("name") or "")
        status = _injury_status_text(detail)
        out.append(
            {
                "player": name,
                "position": pos,
                "bucket": _position_bucket(pos),
                "status": status,
                "out": bool(_status_means_out(status)),
            }
        )
    return out


def snapshot_nfl() -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    teams_payload = _fetch_json(ESPN_NFL_TEAMS)
    teams = []
    if isinstance(teams_payload, dict):
        sports = teams_payload.get("sports") or []
        if sports:
            leagues = sports[0].get("leagues") or []
            if leagues:
                teams = leagues[0].get("teams") or []
    rows: list[dict[str, Any]] = []
    for wrapper in teams:
        team = wrapper.get("team") if isinstance(wrapper, dict) else None
        if not isinstance(team, dict):
            continue
        tid = str(team.get("id") or "")
        abbr = str(team.get("abbreviation") or "").lower()
        if not tid or not abbr:
            continue
        injuries = _team_injuries(tid)
        out_n = sum(1 for i in injuries if i.get("out"))
        ol_out = sum(1 for i in injuries if i.get("out") and i.get("bucket") == "ol")
        skill_out = sum(1 for i in injuries if i.get("out") and i.get("bucket") == "skill")
        rows.append(
            {
                "team": abbr,
                "team_id": tid,
                "injuries": len(injuries),
                "out": out_n,
                "ol_out": ol_out,
                "skill_out": skill_out,
                "details": injuries,
            }
        )
        print(f"  {abbr}: out={out_n} ol={ol_out} skill={skill_out}", flush=True)

    day = date.today().isoformat()
    path = OUT_ROOT / f"{day}.json"
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "league": "nfl",
        "teams": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Also write a flat CSV for join convenience
    flat = []
    for row in rows:
        flat.append(
            {
                "date": day,
                "team": row["team"],
                "injuries": row["injuries"],
                "out": row["out"],
                "ol_out": row["ol_out"],
                "skill_out": row["skill_out"],
            }
        )
    import csv

    csv_path = OUT_ROOT / f"{day}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "team", "injuries", "out", "ol_out", "skill_out"])
        writer.writeheader()
        writer.writerows(flat)
    print(f"wrote {path} and {csv_path}", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="nfl", choices=["nfl"])
    args = parser.parse_args()
    if args.league != "nfl":
        return 1
    snapshot_nfl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
