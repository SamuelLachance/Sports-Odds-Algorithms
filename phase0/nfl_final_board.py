"""FINAL player board: opponent-adjusted CONTINUOUS per-play value ratings.

Per duel: resid = EPA - (defense's as-of allowed EPA/play)  -> magnitude preserved,
opponent adjusted. Player state: decayed (resid_sum, n). Rating = EB shrink toward
position REPLACEMENT (rep = pos mean - 0.5 sd, DEV-era). Display = rating - 2*SE
(SE = sd_pos/sqrt(n_eff+P)) -> small samples sink structurally. 0-100 percentile.
This is the MLB architecture faithfully: full event value vs the opponent, priors,
uncertainty — not binary duels (which reward dink-and-dunk; see Mac Jones incident).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # loaders (games+gid, snaps, OFFW, names, ...)  # noqa: S102

epaP = json.load(open("data/nfl_epa_rating.json"))["params"]

# duel plays with protagonists
plays_by_game = defaultdict(list)
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        plays_by_game[r[ix["game_id"]]].append(
            (PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]]),
             PBP_FIX.get(r[ix["defteam"]], r[ix["defteam"]]),
             float(r[ix["epa"]]), r[ix["passer_player_id"]],
             r[ix["rusher_player_id"]], r[ix["receiver_player_id"]]))

posmap = {}
for (pid, s, w), d in OFFW.items():
    posmap[pid] = d["pos"]

DECAY = 0.998            # per-duel decay (~ half-life 350 duels)
S_DECAY = 0.30           # season boundary
# defense allowed-rate states (same params as shipped epa feature)
dfa = defaultdict(lambda: [0.0, 0.0])
def drate(t):
    st = dfa[t]
    return (st[0] + epaP["prior_n"] * LG_EPA) / (st[1] + epaP["prior_n"])

P_STATE = defaultdict(lambda: [0.0, 0.0])     # pid -> [resid_sum, n] decayed
dev_snap = defaultdict(list)                  # position -> DEV-era rating snapshots
prev = None
for g in games:
    s = g["season"]
    if prev is not None and s != prev:
        for st in P_STATE.values():
            st[0] *= (1 - S_DECAY); st[1] *= (1 - S_DECAY)
        for st in dfa.values():
            st[0] *= (1 - epaP["season_decay"]); st[1] *= (1 - epaP["season_decay"])
    prev = s
    game_team_epa = defaultdict(lambda: [0.0, 0])
    for (pos_t, def_t, epa, pa, ru, re) in plays_by_game.get(g["gid"], ()):
        d_allow = drate(def_t)
        resid = epa - d_allow
        for pid in (pa, ru, re):
            if pid:
                st = P_STATE[pid]
                st[0] = DECAY * st[0] + resid
                st[1] = DECAY * st[1] + 1.0
        game_team_epa[pos_t][0] += epa
        game_team_epa[pos_t][1] += 1
    for t, (sm, n) in game_team_epa.items():
        opp = g["away"] if t == g["home"] else g["home"]
        st = dfa[opp]
        st[0] = epaP["decay"] * st[0] + sm
        st[1] = epaP["decay"] * st[1] + n
    # DEV-era snapshots for normalization (raw rates, players with sample)
    if 2001 <= s <= 2015 and g["week"] == 10:          # one snapshot week per season
        for pid, st in P_STATE.items():
            if st[1] >= 100:
                dev_snap[posmap.get(pid, "?")].append(st[0] / st[1])

NORM = {}
for pos, vals in dev_snap.items():
    v = np.array(vals)
    if len(v) > 50:
        NORM[pos] = (float(v.mean()), float(v.std()))
print("normalization (DEV):", {p: (round(m, 3), round(s, 3)) for p, (m, s) in NORM.items()})

P_EB = 150.0             # replacement-prior weight (duels)
REP_D = 0.5              # replacement = mean - 0.5 sd
ACT = {p for (p, s, w) in OFFW if s == 2025}
rows_out = []
for pid, st in P_STATE.items():
    pos = posmap.get(pid)
    if pos not in NORM or pid not in ACT or st[1] < 30:
        continue
    mu_p, sd_p = NORM[pos]
    rep = mu_p - REP_D * sd_p
    n = st[1]
    rate = (st[0] + P_EB * rep) / (n + P_EB)
    z = (rate - mu_p) / sd_p
    se_z = (1.4 / sd_p) / np.sqrt(n + P_EB)            # per-play sd ~1.4 EPA, in z units
    cons = z - 2.0 * se_z
    rows_out.append((cons, z, n, pid, pos))
rows_out.sort(reverse=True)
with open("data/nfl_player_board_2025.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["player", "gsis_id", "pos", "z", "conservative_z", "n_duels"])
    for cons, z, n, pid, pos in rows_out:
        w.writerow([names.get(pid, pid), pid, pos, round(z, 3), round(cons, 3), int(n)])

def show(pos_set, title, floor=500):
    lst = [(c, z, n, pid) for c, z, n, pid, pos in rows_out if pos in pos_set and n >= floor]
    print(f"\n{title}")
    for i, (c, z, n, pid) in enumerate(lst[:10], 1):
        print(f"  {i:>2} {names.get(pid, pid):<24} z {z:+.2f}  cons {c:+.2f}  n {int(n):,}")

show({"QB"}, "TOP-10 QB (opponent-adjusted continuous value):", 300)
show({"RB", "FB"}, "TOP-10 RB:", 220)
show({"WR"}, "TOP-10 WR:", 90)
show({"TE"}, "TOP-10 TE:", 70)
for nm in ("Lamar Jackson", "Jalen Hurts", "Mac Jones", "Malik Willis", "Joe Milton"):
    m = [(c, z, n, pid, pos) for c, z, n, pid, pos in rows_out
         if names.get(pid, "").startswith(nm.split()[0]) and names.get(pid, "") .endswith(nm.split()[-1])]
    for c, z, n, pid, pos in m:
        qb_ranked = [(cc, pp) for cc, zz, nn, pp, po in rows_out if po == "QB" and nn >= 500]
        rk = next((i for i, (cc, pp) in enumerate(qb_ranked, 1) if pp == pid), "unranked")
        print(f"  check {names.get(pid,pid):<22} rank {rk}  z {z:+.2f} cons {c:+.2f} n {int(n):,}")
