"""Per-game starting-pitcher rotation-fatigue feature.

Days of rest = days since that starter's PREVIOUS start (strictly earlier, so
leak-safe). Also his workload in that last start (batters faced), since a heavy
outing on short rest is the real fatigue case. The effect is on run PREVENTION,
so a rested home starter helps the home side; the feature is the home-minus-away
differential of a rest-penalty encoding.

Data: game logs (starter ids + dates) + parsed appearances (last-start workload).
No new download.
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from datetime import datetime

F_DATE, F_VIS, F_HOME, F_VIS_SP, F_HOME_SP = 0, 3, 6, 101, 103


def load_games():
    rows = []
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            try:
                vr, hr = int(r[9]), int(r[10])
            except ValueError:
                continue
            if vr == hr:
                continue
            rows.append({"gid": r[F_HOME] + r[F_DATE] + r[1], "date": r[F_DATE],
                         "season": int(r[F_DATE][:4]),
                         "home_sp": r[F_HOME_SP].strip(), "away_sp": r[F_VIS_SP].strip(),
                         "y": 1.0 if hr > vr else 0.0})
    rows.sort(key=lambda g: (g["date"], g["gid"]))
    return rows


def last_start_bf():
    """(game_id, pitcher) -> batters faced as the starter (workload)."""
    out = {}
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        if r["is_starter"] == "1":
            out[(r["game_id"], r["pitcher"])] = int(r["bf"])
    return out


def days(d1, d2):
    return (datetime.strptime(d1, "%Y%m%d") - datetime.strptime(d2, "%Y%m%d")).days


def build(out="data/retro_events/rest_feature.csv"):
    games = load_games()
    bf = last_start_bf()
    last_start = {}     # pitcher -> (date, bf)
    rows = []
    for g in games:
        def rest_of(sp, gid):
            if not sp or sp not in last_start:
                return None, None
            ld, lbf = last_start[sp]
            dr = days(g["date"], ld)
            return (dr if 0 < dr < 40 else None), lbf
        hr_days, hr_bf = rest_of(g["home_sp"], g["gid"])
        ar_days, ar_bf = rest_of(g["away_sp"], g["gid"])
        rows.append([g["gid"], g["date"], g["season"],
                     hr_days if hr_days is not None else "",
                     ar_days if ar_days is not None else "",
                     hr_bf if hr_bf is not None else "",
                     ar_bf if ar_bf is not None else ""])
        # update last-start
        for sp in (g["home_sp"], g["away_sp"]):
            if sp:
                last_start[sp] = (g["date"], bf.get((g["gid"], sp), 25))

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "season", "home_rest", "away_rest",
                    "home_last_bf", "away_last_bf"])
        w.writerows(rows)

    # distribution
    hr = [int(r[3]) for r in rows if r[3] != ""]
    from collections import Counter
    c = Counter(min(x, 8) for x in hr)
    print(f"wrote {out}: {len(rows):,} games, {len(hr):,} with home-starter rest")
    print("home-starter days-rest distribution (8=8+):",
          {k: c[k] for k in sorted(c)})
    short = sum(1 for x in hr if x < 4) / len(hr)
    print(f"short rest (<4 days): {short*100:.1f}% of starts")


if __name__ == "__main__":
    build()
