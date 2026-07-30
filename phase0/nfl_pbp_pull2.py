"""Second pbp reduction: per-(game, offense) turnover events for the luck feature.

Pulls nflverse pbp 1999-2025 again, keeps interception / fumble / fumble_lost flags,
aggregates per game-team: int_thrown, fumbles, fumbles_lost.
Output: data/nfl_turnovers.csv. Parquet deleted after each season.
"""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_turnovers.csv"
TMP = os.environ.get("TEMP", "/tmp")
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["game_id", "game_date", "posteam", "interception", "fumble", "fumble_lost"]

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())

for y in range(1999, 2027):
    if str(y) in done:
        print(f"{y}: done, skip", flush=True)
        continue
    fp = os.path.join(TMP, f"pbp2_{y}.parquet")
    try:
        urllib.request.urlretrieve(URL.format(y=y), fp)
    except OSError as e:   # 404 etc: season parquet not published yet (e.g. 2026 preseason)
        print(f"{y}: not available yet ({e}), skip", flush=True)
        if os.path.exists(fp):
            os.remove(fp)
        continue
    df = pd.read_parquet(fp, columns=COLS)
    df = df[df.posteam.notna()]
    a = (df.groupby(["game_id", "game_date", "posteam"])
           .agg(int_thrown=("interception", "sum"), fumbles=("fumble", "sum"),
                fumbles_lost=("fumble_lost", "sum")).reset_index())
    a.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(a)} team-games", flush=True)

print("DONE", flush=True)
