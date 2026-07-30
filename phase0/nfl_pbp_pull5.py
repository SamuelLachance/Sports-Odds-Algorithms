"""Fifth pbp reduction: field goals for kicker-luck.
Output: data/nfl_fgs.csv (game_id, posteam, kick_distance, made)."""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_fgs.csv"
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
    fp = os.path.join(TMP, f"pbp6_{y}.parquet")
    try:
        urllib.request.urlretrieve(URL.format(y=y), fp)
    except OSError as e:   # 404 etc: season parquet not published yet (e.g. 2026 preseason)
        print(f"{y}: not available yet ({e}), skip", flush=True)
        if os.path.exists(fp):
            os.remove(fp)
        continue
    df = pd.read_parquet(fp, columns=["game_id", "posteam", "play_type",
                                      "kick_distance", "field_goal_result"])
    df = df[(df.play_type == "field_goal") & df.kick_distance.notna()
            & df.posteam.notna() & df.field_goal_result.notna()]
    df["made"] = (df.field_goal_result == "made").astype(int)
    df = df[["game_id", "posteam", "kick_distance", "made"]]
    df.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(df)} FGs", flush=True)

print("DONE", flush=True)
