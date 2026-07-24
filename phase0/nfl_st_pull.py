"""Pull special-teams plays (punt/kickoff + FG/XP) from nflverse pbp parquets.

Aggregates PER GAME PER TEAM (as posteam = "for", as defteam = "against"),
in two play groups: pk = punt+kickoff, fx = field_goal+extra_point.
Writes data/nfl_st_games.csv with columns:
  game_id, team, st_epa_for, n_for, st_epa_against, n_against,
  fx_epa_for, fx_n_for, fx_epa_against, fx_n_against
(st_* columns are the punt+kickoff group).

Re-runnable/resumable: seasons already present in the output csv are skipped;
new season rows are appended. Each parquet download retried once on failure.
"""
from __future__ import annotations

import os
import time
import urllib.request
from collections import defaultdict

import pandas as pd

OUT = "data/nfl_st_games.csv"
TMP_DIR = os.path.join(
    r"C:\Users\Admin\AppData\Local\Temp\claude"
    r"\C--Users-Admin-Projects-Sports-Odds-Algorithms"
    r"\c82ef32b-d2a5-4470-9d81-a522f1e0284f\scratchpad", "st_pbp")
os.makedirs(TMP_DIR, exist_ok=True)
URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{yr}.parquet")
COLS = ["game_id", "season", "play_type", "epa", "posteam", "defteam"]
HDR = ("game_id,team,st_epa_for,n_for,st_epa_against,n_against,"
       "fx_epa_for,fx_n_for,fx_epa_against,fx_n_against\n")

done_seasons = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as fh:
        next(fh, None)
        for ln in fh:
            done_seasons.add(int(ln[:4]))
else:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(HDR)
print(f"seasons already done: {sorted(done_seasons)}", flush=True)


def fetch(yr: int) -> str:
    path = os.path.join(TMP_DIR, f"pbp_{yr}.parquet")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    url = URL.format(yr=yr)
    for attempt in (1, 2):
        try:
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:  # noqa: BLE001
            print(f"  {yr} download attempt {attempt} failed: {e}", flush=True)
            if attempt == 1:
                time.sleep(5)
    raise RuntimeError(f"download failed twice for {yr}")


for yr in range(1999, 2026):
    if yr in done_seasons:
        continue
    t0 = time.time()
    path = fetch(yr)
    df = pd.read_parquet(path, columns=COLS)
    df = df[df["play_type"].isin(["punt", "kickoff", "field_goal", "extra_point"])]
    df = df.dropna(subset=["epa", "posteam", "defteam", "game_id"])
    df = df[(df["posteam"] != "") & (df["defteam"] != "")]
    df["grp"] = (df["play_type"].isin(["punt", "kickoff"])).map({True: "pk", False: "fx"})

    # agg[(game_id, team)] = [pk_for_s, pk_for_n, pk_ag_s, pk_ag_n, fx_for_s, fx_for_n, fx_ag_s, fx_ag_n]
    agg = defaultdict(lambda: [0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0])
    for gid, grp, epa, pos, deft in zip(df["game_id"], df["grp"], df["epa"],
                                        df["posteam"], df["defteam"]):
        off = 0 if grp == "pk" else 4
        a = agg[(gid, pos)]
        a[off] += epa; a[off + 1] += 1
        b = agg[(gid, deft)]
        b[off + 2] += epa; b[off + 3] += 1

    with open(OUT, "a", encoding="utf-8") as fh:
        for (gid, tm) in sorted(agg):
            a = agg[(gid, tm)]
            fh.write(f"{gid},{tm},{a[0]:.6f},{a[1]},{a[2]:.6f},{a[3]},"
                     f"{a[4]:.6f},{a[5]},{a[6]:.6f},{a[7]}\n")
    os.remove(path)
    print(f"{yr}: {len(df)} ST plays, {len(agg)} game-team rows "
          f"[{time.time()-t0:.0f}s]", flush=True)

print("done -> " + OUT, flush=True)
