"""Per-play context for the participation TrueSkill: game_id, play_id, wp
-> data/nfl_play_ctx.csv (2016-2025, pass/run plays only, matching nfl_rapm_plays)."""
import os
import urllib.request

import pandas as pd

OUT = "data/nfl_play_ctx.csv"
TMP = os.environ.get("TEMP", "/tmp")
BURL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"

first = not os.path.exists(OUT)
done = set()
if not first:
    done = set(pd.read_csv(OUT, usecols=["game_id"]).game_id.str[:4].unique())

for y in range(2016, 2027):
    if str(y) in done:
        print(f"{y}: done, skip", flush=True)
        continue
    fp = os.path.join(TMP, f"pbp_ctx_{y}.parquet")
    try:
        urllib.request.urlretrieve(BURL.format(y=y), fp)
    except OSError as e:   # 404 etc: season parquet not published yet (e.g. 2026 preseason)
        print(f"{y}: not available yet ({e}), skip", flush=True)
        if os.path.exists(fp):
            os.remove(fp)
        continue
    pbp = pd.read_parquet(fp, columns=["game_id", "play_id", "play_type", "epa", "wp"])
    pbp = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna()]
    pbp[["game_id", "play_id", "wp"]].to_csv(OUT, mode="w" if first else "a",
                                             header=first, index=False)
    first = False
    os.remove(fp)
    print(f"{y}: {len(pbp)} plays", flush=True)
print("DONE", flush=True)
