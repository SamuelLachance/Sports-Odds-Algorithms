"""Pull per-batted-ball Statcast (2015-2024) -> as-of xwOBA components.

Baseball Savant statcast_search CSV, chunked by short date windows (with adaptive
re-split if a chunk nears the row cap, so nothing is silently truncated). For each
plate-appearance-terminal row we keep the xwOBA numerator (estimated_woba_using_
speedangle for balls in play, else the actual woba_value for K/BB/HBP) and woba_denom,
aggregated to per (game_date, player). This is the raw material for a leak-safe,
within-season AS-OF xwOBA (for batters and, as xwOBA-against, for pitchers).

Outputs (append-safe, resumable): data/statcast_bbe_batter.csv, statcast_bbe_pitcher.csv
"""

from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

SEARCH = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
          "&game_date_gt={a}&game_date_lt={b}&min_pitches=0&min_results=0")
CAP = 24000        # Savant silently truncates a CSV query at 25k rows; re-split below that
BAT = "data/statcast_bbe_batter.csv"
PIT = "data/statcast_bbe_pitcher.csv"


def fetch(a, b):
    u = SEARCH.format(a=a, b=b)
    d = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}),
                               timeout=120).read()
    return pd.read_csv(io.BytesIO(d), low_memory=False)


def aggregate(df, bat_rows, pit_rows):
    df = df[df["woba_denom"].notna()].copy()
    est = df["estimated_woba_using_speedangle"]
    df["xnum"] = est.where(est.notna(), df["woba_value"])
    for key, sink in (("batter", bat_rows), ("pitcher", pit_rows)):
        g = df.groupby(["game_date", key]).agg(xnum=("xnum", "sum"),
                                               denom=("woba_denom", "sum")).reset_index()
        for _, r in g.iterrows():
            sink.append((r["game_date"], int(r[key]), round(float(r["xnum"]), 4), round(float(r["denom"]), 4)))


def pull_window(a, b, bat_rows, pit_rows):
    df = fetch(a, b)
    if len(df) >= CAP and a != b:                 # near the row cap -> split to single days
        d0 = date.fromisoformat(a); d1 = date.fromisoformat(b)
        d = d0
        while d <= d1:
            pull_window(d.isoformat(), d.isoformat(), bat_rows, pit_rows)
            d += timedelta(days=1)
        return
    aggregate(df, bat_rows, pit_rows)


def flush(bat_rows, pit_rows):
    for path, rows in ((BAT, bat_rows), (PIT, pit_rows)):
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            if new:
                fh.write("date,mlbam,xnum,denom\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    bat_rows.clear(); pit_rows.clear()


def done_windows():
    seen = set()
    if os.path.exists(BAT):
        for line in open(BAT):
            seen.add(line.split(",", 1)[0])
    return seen


def main():
    done = done_windows()
    for yr in [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]:
        start = date(yr, 3, 15); end = date(yr, 11, 5)
        d = start
        while d <= end:
            b = min(d + timedelta(days=3), end)
            if d.isoformat() in done:             # already pulled (resume)
                d = b + timedelta(days=1); continue
            bat_rows, pit_rows = [], []
            try:
                pull_window(d.isoformat(), b.isoformat(), bat_rows, pit_rows)
                flush(bat_rows, pit_rows)
            except Exception as e:  # noqa: BLE001
                print(f"  {d} {type(e).__name__} {str(e)[:40]}", flush=True)
            time.sleep(0.2)
            d = b + timedelta(days=1)
        nb = (sum(1 for _ in open(BAT)) - 1) if os.path.exists(BAT) else 0
        print(f"{yr} done  (batter rows so far: {nb:,})", flush=True)
    print("PULL COMPLETE")


if __name__ == "__main__":
    main()
