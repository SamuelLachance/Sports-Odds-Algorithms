"""Extended per-play context for the participation TrueSkill: everything the
market-free pbp offers about a snap's importance -> data/nfl_play_ctx2.csv
(2016-2025, pass/run plays, matching nfl_rapm_plays)."""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_play_ctx2.csv"
TMP = os.environ.get("TEMP", "/tmp")
BURL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["game_id", "play_id", "play_type", "epa", "wp", "qtr", "down",
        "ydstogo", "score_differential", "half_seconds_remaining", "yardline_100"]

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())
for y in range(2016, 2027):
    if str(y) in done:
        print(f"{y}: done, skip", flush=True)
        continue
    fp = os.path.join(TMP, f"pbp_ctx2_{y}.parquet")
    try:
        urllib.request.urlretrieve(BURL.format(y=y), fp)
    except OSError as e:   # 404 etc: season parquet not published yet (e.g. 2026 preseason)
        print(f"{y}: not available yet ({e}), skip", flush=True)
        if os.path.exists(fp):
            os.remove(fp)
        continue
    pbp = pd.read_parquet(fp, columns=COLS)
    pbp = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna()]
    pbp = pbp.drop(columns=["play_type", "epa"])
    pbp["play_id"] = pbp.play_id.astype(int)
    pbp.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(pbp)} plays", flush=True)
print("DONE", flush=True)
