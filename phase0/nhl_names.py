"""Build a player-id -> name/pos/team map from current NHL rosters (32 requests)."""
from __future__ import annotations

import csv
import gzip
import json
import time
import urllib.request

HEADERS = {"User-Agent": "glassbox-nhl/1.0", "Accept-Encoding": "gzip"}


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def main():
    teams = sorted({r["home"] for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8"))
                    if r["season"] == "20252026"})
    names = {}
    for t in teams:
        try:
            d = get(f"https://api-web.nhle.com/v1/roster/{t}/current")
            for grp in ("forwards", "defensemen", "goalies"):
                for p in d.get(grp, []):
                    names[str(p["id"])] = {
                        "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                        "pos": p.get("positionCode", ""), "team": t}
        except Exception as ex:  # noqa: BLE001
            print(f"  {t}: {ex}")
        time.sleep(0.2)
    json.dump(names, open("data/nhl_player_names.json", "w"))
    print(f"wrote data/nhl_player_names.json: {len(names):,} players from {len(teams)} rosters")


if __name__ == "__main__":
    main()
