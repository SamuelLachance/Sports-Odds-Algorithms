"""Pull nflverse play-by-play 1999-2025, reduce to the per-play essentials for
offense-vs-defense TrueSkill: game_id, game_date, posteam, defteam, epa.

Pass/run plays only (no ST/kneels/spikes/no_play), valid epa+teams required.
Each season's parquet is deleted after reduction. Output: data/nfl_plays.csv
"""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_plays.csv"
TMP = os.environ.get("TEMP", "/tmp")
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["game_id", "game_date", "posteam", "defteam", "epa", "play_type", "play_id"]

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())

for y in range(1999, 2027):
    if str(y) in done:
        print(f"{y}: already reduced, skip", flush=True)
        continue
    fp = os.path.join(TMP, f"pbp_{y}.parquet")
    try:
        urllib.request.urlretrieve(URL.format(y=y), fp)
    except OSError as e:   # 404 etc: season parquet not published yet (e.g. 2026 preseason)
        print(f"{y}: not available yet ({e}), skip", flush=True)
        if os.path.exists(fp):
            os.remove(fp)
        continue
    df = pd.read_parquet(fp, columns=COLS)
    df = df[df.play_type.isin(["pass", "run"]) & df.epa.notna()
            & df.posteam.notna() & df.defteam.notna()]
    df = df.sort_values(["game_date", "game_id", "play_id"])
    df = df[["game_id", "game_date", "posteam", "defteam", "epa"]]
    df.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(df)} plays", flush=True)

print("DONE", flush=True)
