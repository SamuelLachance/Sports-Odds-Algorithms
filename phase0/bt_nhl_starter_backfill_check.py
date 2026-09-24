"""Backfill check: can past starting goalies be reconstructed as known PRE-GAME?

Serving-feasibility task of the breakthrough program. Every round-1 NHL screen
identified the starter as the goalie who faced the first shot
(bt_nhl_anatomy_build.load_starters, from data/nhl_shots.csv). The pre-game truth
for a CONFIRMED-tier serve is the OFFICIAL starting goalie on the game roster —
the api-web boxscore flag playerByGameStats.*.goalies[].starter (and the bold
name in the nhl.com Playing Roster report, "* Starting Lineup in Bold").

This script measures, on a seeded random sample of DEV games only
(2011-12..2017-18, regular season), how often the first-shot proxy equals the
official starter, and whether the official flag is present for DEV-era games.
It computes no model metric and touches no TEST season and no odds.

    python -X utf8 phase0/bt_nhl_starter_backfill_check.py [--n 150]
    -> data/bt_nhl_starter_backfill.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import random
import time
import urllib.request
from collections import Counter

import pandas as pd

DEV = (20112012, 20122013, 20132014, 20142015, 20152016, 20162017, 20172018)
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
BOX = "https://api-web.nhle.com/v1/gamecenter/{}/boxscore"
OUT = "data/bt_nhl_starter_backfill.json"
SEED = 7


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=20).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def first_shot_starters(gids):
    """Same rule as bt_nhl_anatomy_build.load_starters: goalie facing the first
    shot row per defending side. -> {gid: {1: home goalie, 0: away goalie}}"""
    sh = pd.read_csv("data/nhl_shots.csv",
                     usecols=["nhl_game_id", "period", "per_sec", "is_home", "goalie_id"])
    sh = sh[sh.nhl_game_id.isin(gids)].dropna(subset=["goalie_id"])
    sh = sh.sort_values(["nhl_game_id", "period", "per_sec"])
    sh["def_home"] = 1 - sh["is_home"]
    first = sh.groupby(["nhl_game_id", "def_home"], sort=False).first().reset_index()
    out = {}
    for r in first.itertuples(index=False):
        out.setdefault(int(r.nhl_game_id), {})[int(r.def_home)] = int(r.goalie_id)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args(argv)
    g = pd.read_csv("data/nhl_games.csv", usecols=["game_id", "season", "type"])
    g = g[g.season.isin(DEV) & (g.type == 2)]
    pool = sorted(int(x) for x in g.game_id)
    rng = random.Random(SEED)
    sample = sorted(rng.sample(pool, a.n))
    fs = first_shot_starters(set(sample))

    rows, fails = [], 0
    for gid in sample:
        try:
            b = get(BOX.format(gid))
        except Exception:  # noqa: BLE001
            fails += 1
            continue
        pg = b.get("playerByGameStats") or {}
        rec = {"gid": gid}
        for side, h in (("homeTeam", 1), ("awayTeam", 0)):
            gl = (pg.get(side) or {}).get("goalies") or []
            flagged = [x["playerId"] for x in gl if x.get("starter") is True]
            has_flag = any("starter" in x for x in gl)
            played = [x["playerId"] for x in gl if (x.get("toi") or "00:00") != "00:00"]
            rec[side] = {"n_goalies": len(gl), "has_flag": has_flag, "official": flagged,
                         "first_shot": fs.get(gid, {}).get(h), "n_played": len(played)}
        rows.append(rec)
        time.sleep(0.3)

    c = Counter()
    mism = []
    for r in rows:
        for side in ("homeTeam", "awayTeam"):
            s = r[side]
            c["sides"] += 1
            c["has_flag"] += s["has_flag"]
            c["exactly_one_official"] += len(s["official"]) == 1
            c["relief_used"] += s["n_played"] >= 2
            if len(s["official"]) == 1 and s["first_shot"] is not None:
                c["compared"] += 1
                if s["official"][0] == s["first_shot"]:
                    c["agree"] += 1
                else:
                    mism.append({"gid": r["gid"], "side": side, **s})
            elif s["first_shot"] is None:
                c["no_first_shot"] += 1
    res = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": {"seasons": list(DEV), "dev_only": True, "test_touched": False,
                     "market_use": "none", "sample": f"random.Random({SEED}).sample of "
                     f"{len(pool)} DEV regular-season games, n={a.n}",
                     "requests": len(sample)},
        "fetch_failures": fails,
        "counts": dict(c),
        "official_flag_coverage": round(c["has_flag"] / max(c["sides"], 1), 4),
        "first_shot_equals_official": round(c["agree"] / max(c["compared"], 1), 4),
        "mismatches": mism,
    }
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=1))
    print("mismatches:", len(mism))
    for m in mism[:10]:
        print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
