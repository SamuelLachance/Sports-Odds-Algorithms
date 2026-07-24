"""Per-game penalty counts/yards per team, 1999-2025 -> data/nfl_penalties.csv"""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_penalties.csv"
TMP = os.environ.get("TEMP", "/tmp")
BURL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())
for y in range(1999, 2026):
    if str(y) in done:
        continue
    fp = os.path.join(TMP, f"pbp_pen_{y}.parquet")
    urllib.request.urlretrieve(BURL.format(y=y), fp)
    pbp = pd.read_parquet(fp, columns=["game_id", "penalty", "penalty_team", "penalty_yards"])
    pen = pbp[(pbp.penalty == 1) & pbp.penalty_team.notna()]
    agg = pen.groupby(["game_id", "penalty_team"]).agg(
        n=("penalty", "size"), yds=("penalty_yards", "sum")).reset_index()
    agg.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(agg)} team-games", flush=True)
print("DONE", flush=True)
