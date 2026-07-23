"""Per-play protagonists for micro-matches: passer/rusher/receiver gsis ids keyed
by (game_id, play_id) -> data/nfl_play_prot.csv (2016-2025, pass/run plays)."""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_play_prot.csv"
TMP = os.environ.get("TEMP", "/tmp")
BURL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["game_id", "play_id", "play_type", "epa",
        "passer_player_id", "rusher_player_id", "receiver_player_id"]

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())
for y in range(2016, 2026):
    if str(y) in done:
        print(f"{y}: done, skip", flush=True)
        continue
    fp = os.path.join(TMP, f"pbp_prot_{y}.parquet")
    urllib.request.urlretrieve(BURL.format(y=y), fp)
    pbp = pd.read_parquet(fp, columns=COLS)
    pbp = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna()]
    pbp = pbp.drop(columns=["play_type", "epa"])
    pbp["play_id"] = pbp.play_id.astype(int)
    pbp.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(pbp)} plays", flush=True)
print("DONE", flush=True)
