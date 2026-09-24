"""Breakthrough program 2026-09-24 -- NFL gap anatomy, QB-in-flux drill-down (build).

The anatomy (bt_nfl_anatomy2.py) puts the largest share of the DEV gap in games
where a team's starter is NOT its established QB. Two competing mechanisms:
  H1 STALE TEAM RATING: team Elo is updated on W/L with no QB adjustment, so it
     embeds the OLD QB; when the QB changes the Elo term is stale by roughly
     (old QB - new QB) and the blend's separate qd term only partly corrects.
  H2 OVER-SHRUNK NEW STARTER: the shipped QB rating shrinks to replacement with
     prior_db ~650 dropbacks, so a new starter's early-stint evidence is heavily
     discounted.
This script records, for every game <= 2015 and each side, the as-of shipped QB
rating of the CURRENT starter and of the team's ESTABLISHED starter (most starts
in the team's trailing 16 games), the current starter's evidence mass and raw
EPA/db, and the stint length. Walk-forward: every value is read BEFORE the game
updates the state. Rows with season >= 2016 are never written.
Output: data/bt_nfl_anatomy_qb.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict

src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=pedP["delta"],
          rep_map=rep_map, decay=SP["decay"], prior_db=SP["prior_db"],
          season_decay=SP["season_decay"])
hist = defaultdict(list)          # team -> list of starter ids (past games)
out = []
prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    if g["season"] > 2015:
        break
    row = {"gid": g["gid"], "season": g["season"]}
    for side, team, qb in (("h", g["home"], g["home_qb"]), ("a", g["away"], g["away_qb"])):
        H = hist[team]
        cnt = defaultdict(int)
        for q in H[-16:]:
            cnt[q] += 1
        estab = max(cnt, key=cnt.get) if cnt else qb
        stint = 0
        for q in reversed(H):
            if q == qb:
                stint += 1
            else:
                break
        st = m.Q.get(qb)
        row[side + "_q_cur"] = m.qb_adj(qb)
        row[side + "_q_estab"] = m.qb_adj(estab)
        row[side + "_is_estab"] = int(qb == estab)
        row[side + "_db"] = st[1] if st else 0.0
        row[side + "_raw"] = (st[0] / st[1]) if (st and st[1] > 0) else float("nan")
        row[side + "_stint"] = stint
        row[side + "_team_starts_cur"] = cnt.get(qb, 0)
    out.append(row)
    m.predict(g)
    m.update(g, 0.5, qbw)
    hist[g["home"]].append(g["home_qb"])
    hist[g["away"]].append(g["away_qb"])
assert max(r["season"] for r in out) <= 2015
with open("data/bt_nfl_anatomy_qb.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    wr.writeheader()
    for r in out:
        wr.writerow(r)
print(f"wrote data/bt_nfl_anatomy_qb.csv ({len(out)} rows, lg_qb {lg_qb:+.4f}, "
      f"rep {lg_qb + pedP['delta']:+.4f}, prior_db {SP['prior_db']:.0f})")
