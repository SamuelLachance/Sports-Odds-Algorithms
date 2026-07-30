"""Build a player-id -> name/pos/team map from current NHL rosters (32 requests)."""
from __future__ import annotations

import csv
import gzip
import json
import os
import time
import urllib.request

HEADERS = {"User-Agent": "glassbox-nhl/1.0", "Accept-Encoding": "gzip"}


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


OUT = "data/nhl_player_names.json"


def main() -> int:
    rows = list(csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")))
    # Last TWO seasons of home teams: early in a new season only a subset of
    # the league has hosted a completed game yet, so the newest season alone
    # would shrink the map to those teams; the prior (completed) season keeps
    # full 32-team coverage across the rollover. 8-digit strings sort correctly.
    seasons = sorted({r["season"] for r in rows})[-2:]
    teams = sorted({r["home"] for r in rows if r["season"] in seasons})
    print(f"[nhl_names] seasons {seasons}: {len(teams)} teams")
    # Merge onto the existing map (refuse-to-shrink): a failed roster request
    # or partial-league team set must never delete players we already know.
    try:
        names = json.load(open(OUT, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        names = {}
    n_prev = len(names)
    ok_teams = 0
    for t in teams:
        try:
            d = get(f"https://api-web.nhle.com/v1/roster/{t}/current")
            for grp in ("forwards", "defensemen", "goalies"):
                for p in d.get(grp, []):
                    names[str(p["id"])] = {
                        "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                        "pos": p.get("positionCode", ""), "team": t}
            ok_teams += 1
        except Exception as ex:  # noqa: BLE001
            print(f"  {t}: {ex}")
        time.sleep(0.2)
    if ok_teams == 0:
        print(f"[nhl_names] all {len(teams)} roster requests failed — "
              f"keeping existing map ({n_prev:,} players), nothing written")
        return 1
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(names, fh)
    os.replace(tmp, OUT)   # atomic: no truncated JSON on interruption
    print(f"wrote {OUT}: {len(names):,} players ({n_prev:,} carried in) "
          f"from {ok_teams}/{len(teams)} rosters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
