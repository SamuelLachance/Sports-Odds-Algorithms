"""Reduce participation (2016-2025) + pbp to RAPM input:
game_id, play_id, off_players, def_players, epa   -> data/nfl_rapm_plays.csv
"""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_rapm_plays.csv"
TMP = os.environ.get("TEMP", "/tmp")
PURL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{y}.parquet"
BURL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())

for y in range(2016, 2027):
    if str(y) in done:
        print(f"{y}: done, skip", flush=True)
        continue
    fp1 = os.path.join(TMP, f"part_{y}.parquet")
    fp2 = os.path.join(TMP, f"pbp3_{y}.parquet")
    try:
        urllib.request.urlretrieve(PURL.format(y=y), fp1)
        urllib.request.urlretrieve(BURL.format(y=y), fp2)
    except OSError as e:   # 404 etc: season parquet not published yet (e.g. 2026 preseason)
        print(f"{y}: not available yet ({e}), skip", flush=True)
        for fp_ in (fp1, fp2):
            if os.path.exists(fp_):
                os.remove(fp_)
        continue
    part = pd.read_parquet(fp1)
    gcol = "nflverse_game_id" if "nflverse_game_id" in part.columns else "game_id"
    part = part[[gcol, "play_id", "offense_players", "defense_players"]].rename(
        columns={gcol: "game_id"})
    part = part[(part.offense_players.str.len() > 0) & (part.defense_players.str.len() > 0)]
    pbp = pd.read_parquet(fp2, columns=["game_id", "play_id", "epa", "play_type"])
    pbp = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna()]
    df = part.merge(pbp[["game_id", "play_id", "epa"]], on=["game_id", "play_id"])
    df.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    os.remove(fp1); os.remove(fp2)
    print(f"{y}: {len(df)} plays", flush=True)

print("DONE", flush=True)
