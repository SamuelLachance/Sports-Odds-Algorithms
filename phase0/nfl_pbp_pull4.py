"""Fourth pbp reduction: plays with non-vegas WP for garbage-time filtering.
Output: data/nfl_plays_wp.csv (game_id, posteam, defteam, epa, wp)."""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_plays_wp.csv"
TMP = os.environ.get("TEMP", "/tmp")
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())

for y in range(1999, 2027):
    if str(y) in done:
        print(f"{y}: done, skip", flush=True)
        continue
    fp = os.path.join(TMP, f"pbp5_{y}.parquet")
    try:
        urllib.request.urlretrieve(URL.format(y=y), fp)
    except OSError as e:   # 404 etc: season parquet not published yet (e.g. 2026 preseason)
        print(f"{y}: not available yet ({e}), skip", flush=True)
        if os.path.exists(fp):
            os.remove(fp)
        continue
    df = pd.read_parquet(fp, columns=["game_id", "game_date", "posteam", "defteam",
                                      "play_type", "play_id", "epa", "wp"])
    df = df[df.play_type.isin(["pass", "run"]) & df.epa.notna()
            & df.posteam.notna() & df.defteam.notna()]
    df = df.sort_values(["game_date", "game_id", "play_id"])
    df = df[["game_id", "posteam", "defteam", "epa", "wp"]]
    df.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(df)} plays", flush=True)

print("DONE", flush=True)
