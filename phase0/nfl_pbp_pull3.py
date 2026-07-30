"""Third pbp reduction: per-play protagonist ids for MLB-grade player TrueSkill.

Keeps: game_id, game_date, posteam, defteam, passer/rusher/receiver gsis ids, epa.
Output: data/nfl_duel_plays.csv (pass/run plays 1999-2025).
"""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_duel_plays.csv"
TMP = os.environ.get("TEMP", "/tmp")
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
COLS = ["game_id", "game_date", "posteam", "defteam", "play_type", "play_id", "epa",
        "passer_player_id", "rusher_player_id", "receiver_player_id"]

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())

for y in range(1999, 2027):
    if str(y) in done:
        print(f"{y}: done, skip", flush=True)
        continue
    fp = os.path.join(TMP, f"pbp4_{y}.parquet")
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
    df = df[["game_id", "game_date", "posteam", "defteam", "epa",
             "passer_player_id", "rusher_player_id", "receiver_player_id"]]
    df.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(df)} plays", flush=True)

print("DONE", flush=True)
