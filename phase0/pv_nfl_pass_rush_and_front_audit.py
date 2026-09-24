"""pv NFL pass_rush_and_front -- step 5: the 2013-2015 snap-truth audits. DEV ONLY.

2013-2015 are the only DEV seasons with snap counts (data/snap_2012.csv is an
empty header). They are used here to measure, not to rate:

 A. PRESENCE PROXY  how often the credit-based presence (credited, or credited
    in the team's previous game) matches defense_snaps > 0, by position and
    snap-share band; how many 'present' rows had no defensive snap.
 B. PER-SNAP EXPOSURE  the same engine with exposure = team plays x
    defense_pct on snap seasons: does a per-snap rating predict the future
    better than the per-team-play rating? Checkpoints in 2014-2015 (2013 is the
    snap warm-up).
 C. OFFENSIVE LINE ATTRIBUTION  what can be pinned on individual linemen:
    (1) the pooled effect of a missing regular OL starter (as-of regulars:
        >= 85% of offensive snaps in >= 60% of the team's earlier games this
        season, >= 3 earlier games; absent = 0 offensive snaps) on the game's
        pressure / sacks allowed RESIDUAL vs the pre-game expectation;
    (2) the reliability of the only individually charged OL events
        (offensive holding + false start per offensive snap).
Output: data/pv_nfl_pass_rush_and_front_audit.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_nfl_pass_rush_and_front as E  # noqa: E402

DATA = E.DATA
OUT = {"split": "DEV 2013-2015 (snap seasons) only"}
RNG = np.random.default_rng(7)


def r_(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


PG = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_player_games.parquet"))
TG = pd.read_csv(os.path.join(DATA, "pv_nfl_pass_rush_and_front_team_games.csv"))
SN = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_snaps.parquet"))
assert PG.season.max() < 2016 and SN.season.max() < 2016
SN = SN[SN.season.between(2013, 2015)]

# ------------------------------------------------------------ A. presence proxy
d = SN[SN.defense_snaps > 0][["game_id", "team", "player_id", "defense_pct", "position"]]
pr = PG[PG.season.between(2013, 2015) & (PG.presence != "")][["game_id", "team", "player_id", "presence"]]
m = d.merge(pr, on=["game_id", "team", "player_id"], how="left")
m["grp"] = m.position.map(lambda q: "DL" if q in ("DE", "DT", "NT") else ("LB" if q in ("LB", "OLB", "ILB", "MLB")
                                                                          else "DB"))
m["band"] = pd.cut(m.defense_pct, [0, .2, .4, .6, .8, 1.01], labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
m["credit"] = (m.presence == "credit").astype(float)
m["any"] = m.presence.notna().astype(float)
tab = m.groupby(["grp", "band"], observed=True)[["credit", "any"]].mean().round(3)
A = {f"{g}|{b}": {"recall_credit_only": float(v.credit), "recall_credit_or_fill": float(v["any"])}
     for (g, b), v in tab.iterrows()}
x = pr.merge(SN[["game_id", "team", "player_id", "defense_snaps"]], on=["game_id", "team", "player_id"], how="left")
A["present_rows_without_def_snaps"] = {
    h: round(float((x[x.presence == h].defense_snaps.fillna(0) == 0).mean()), 3) for h in ("credit", "fill")}
A["n_snap_rows"] = int(len(m))
OUT["A_presence_proxy_vs_snaps"] = A
print("A", json.dumps(A)[:900])

# ------------------------------------------------------------ B. per-snap exposure variant
PGs, TGs, _ = E.build({"exposure": "snap"}, through=2015)
key = ["game_id", "team", "player_id"]
both = PG[PG.season.between(2014, 2015)].merge(
    PGs[key + ["th_pr", "th_sk", "th_st"]].rename(columns={"th_pr": "sn_pr", "th_sk": "sn_sk", "th_st": "sn_st"}),
    on=key).merge(SN[key + ["defense_pct"]], on=key, how="left")
both = both[both.bucket.isin(["DE", "DT", "OLB", "ILB"])]
both = both[both.defense_pct.fillna(0) > 0].sort_values(["player_id", "game_date"])
rows = []
for (p, s), g in both.groupby(["player_id", "season"]):
    if len(g) >= 8:
        f = g.iloc[0]
        rows.append(dict(th_pr=f.th_pr, sn_pr=f.sn_pr, th_st=f.th_st, sn_st=f.sn_st, pct=f.defense_pct,
                         sk_pg=g.g_sack.mean(), pr_pg=g.g_pr.mean(), st_pg=g.g_stop.mean(),
                         pr_psnap=g.g_pr.sum() / (g.defense_pct * 1.0).sum(),
                         st_psnap=g.g_stop.sum() / g.defense_pct.sum()))
cp = pd.DataFrame(rows)
B = {"n_player_seasons_2014_2015": int(len(cp)),
     "future_pressure_per_game": {"team_exposure": round(r_(cp.th_pr, cp.pr_pg), 3),
                                  "snap_exposure": round(r_(cp.sn_pr, cp.pr_pg), 3)},
     "future_sacks_per_game": {"team_exposure": round(r_(cp.th_pr, cp.sk_pg), 3),
                               "snap_exposure": round(r_(cp.sn_pr, cp.sk_pg), 3)},
     "future_pressure_per_snap_share": {"team_exposure": round(r_(cp.th_pr, cp.pr_psnap), 3),
                                        "snap_exposure": round(r_(cp.sn_pr, cp.pr_psnap), 3)},
     "future_stops_per_game": {"team_exposure": round(r_(cp.th_st, cp.st_pg), 3),
                               "snap_exposure": round(r_(cp.sn_st, cp.st_pg), 3)},
     "note": "snap variant = same engine, exposure scaled by defense_pct on 2013+ games only; history before 2013 "
             "is identical, so this isolates 1-2 seasons of snap information"}
OUT["B_per_snap_exposure"] = B
print("B", B)

# ------------------------------------------------------------ C. OL attribution
ol = SN[SN.position.isin(["T", "G", "C", "OL", "OT"])].copy()
ol = ol.merge(TG[["game_id", "posteam", "game_date"]].rename(columns={"posteam": "team"}).drop_duplicates(),
              on=["game_id", "team"])
ol = ol.sort_values("game_date")
team_games = TG[TG.season.between(2013, 2015)][["game_id", "posteam", "season", "game_date"]].sort_values("game_date")
miss = {}
for (t, s), tg in team_games.groupby(["posteam", "season"], sort=False):
    gids = list(tg.game_id)
    sub = ol[(ol.team == t) & ol.game_id.isin(gids)]
    M = sub.pivot_table(index="player_id", columns="game_id", values="offense_pct", aggfunc="sum").reindex(
        columns=gids).fillna(0.0).to_numpy()
    for i, g in enumerate(gids):
        if i < 3:
            miss[(g, t)] = (np.nan, 0)
            continue
        regs = (M[:, :i] >= 0.85).sum(1) >= 0.6 * i          # regulars from EARLIER games only
        miss[(g, t)] = (int(((M[:, i] == 0) & regs).sum()), int(regs.sum()))
U = TG[TG.season.between(2013, 2015)].copy()
U["n_miss"] = [miss.get((g, o), (np.nan, 0))[0] for g, o in zip(U.game_id, U.posteam)]
U["n_regs"] = [miss.get((g, o), (np.nan, 0))[1] for g, o in zip(U.game_id, U.posteam)]
U = U[U.n_regs >= 3]
exp_pr = U.g_n_db * U.L_pr * U.D_pr * U.Oopp_pr * U.S_ht
exp_sk = U.g_n_db * U.L_sk * U.D_sk * U.Oopp_sk
U["res_pr"] = (U.g_pressure - exp_pr) / exp_pr.clip(lower=1e-6)
U["res_sk"] = (U.g_sacks - exp_sk) / exp_sk.clip(lower=1e-6)


def ols_slope(x, y):
    X = np.column_stack([x, np.ones(len(x))])
    b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = (e @ e) / (len(y) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return round(float(b[0]), 4), round(float(se), 4)


C = {"n_team_games": int(len(U)),
     "share_games_with_missing_regular": round(float((U.n_miss > 0).mean()), 3),
     "pressure_residual_per_missing_regular (fraction of expected)": ols_slope(U.n_miss.to_numpy(float),
                                                                               U.res_pr.to_numpy(float)),
     "sack_residual_per_missing_regular (fraction of expected)": ols_slope(U.n_miss.to_numpy(float),
                                                                           U.res_sk.to_numpy(float))}
# individually charged OL events: holding + false start per offensive snap-share game
op = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_olpen.parquet"))
op = op[op.season.between(2013, 2015)]
z = ol.merge(op[["game_id", "team", "player_id", "olpen_core"]], on=["game_id", "team", "player_id"], how="left")
z["olpen_core"] = z.olpen_core.fillna(0.0)
z = z[z.offense_snaps > 0].sort_values(["player_id", "game_date"])
z["k"] = z.groupby(["player_id", "season"]).cumcount()
agg = z.groupby(["player_id", "season", z.k % 2]).agg(p=("olpen_core", "sum"), sn=("offense_snaps", "sum"),
                                                      n=("k", "size")).reset_index()
agg["rate"] = agg.p / agg.sn
w = agg.pivot_table(index=["player_id", "season"], columns="k", values=["rate", "n"])
w = w[(w[("n", 0)] >= 4) & (w[("n", 1)] >= 4)]
rsh = r_(w[("rate", 0)], w[("rate", 1)])
ss = z.groupby(["player_id", "season"]).agg(p=("olpen_core", "sum"), sn=("offense_snaps", "sum"),
                                            n=("k", "size")).reset_index()
ss = ss[ss.n >= 8]
ss["rate"] = ss.p / ss.sn
nx = ss.copy(); nx["season"] -= 1
yy = ss.merge(nx, on=["player_id", "season"], suffixes=("", "_n"))
C["ol_holding_falsestart_per_snap"] = {"split_half_spearman_brown": round(2 * rsh / (1 + rsh), 3),
                                       "n_player_seasons": int(len(w)),
                                       "year_over_year_r": round(r_(yy.rate, yy.rate_n), 3),
                                       "n_pairs": int(len(yy))}
C["limits"] = ("No DEV play records which lineman allowed a pressure or a run stop; before 2013 not even OL "
               "participation exists. Individual OL value on DEV = (unit protection factor, shared with the QB) x "
               "snap share + charged penalties. The unit factor is the measurable object.")
OUT["C_offensive_line_attribution"] = C
print("C", C)
json.dump(OUT, open(os.path.join(DATA, "pv_nfl_pass_rush_and_front_audit.json"), "w"), indent=1, default=float)
print("wrote data/pv_nfl_pass_rush_and_front_audit.json")
