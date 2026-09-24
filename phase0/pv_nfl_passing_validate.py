"""Player-level validation of the NFL passing rating (pv_nfl_passing_build.py). DEV ONLY.

Reads the walk-forward values written by the build and scores them on DEV seasons
2006-2015 only (hard-asserted). Three questions, in the MLB order:

  1. RELIABILITY - is the underlying per-QB stat a stable trait? split-half
     (odd/even games within a QB-season, >= 200 dropbacks) and year-over-year
     (consecutive seasons, >= 200 dropbacks each), with n. Opponent adjustment is
     walk-forward (each game corrected by the opponent's PRE-game defence rating).
  2. PREDICTIVE VALIDITY - does the pre-game rating predict the QB's / his unit's
     FUTURE outcomes better than the shipped QbElo feature?  Starter QB-games.
       next game : QB EPA/dropback (dropback-weighted), team offensive points per
                   drive, team offensive points, team win
       next season: rating at the QB's first game of season s+1 vs his s+1 totals
     paired game-cluster bootstrap for every difference.
  3. FACE VALIDITY - top / bottom lists at the end of DEV and in three seasons.

Output: data/pv_nfl_passing_validation.json
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
RNG = np.random.default_rng(20260924)
NB = 2000

V = pd.read_csv("data/pv_nfl_passing_values.csv")
O = pd.read_csv("data/pv_nfl_passing_outcomes.csv")
T = pd.read_csv("data/pv_nfl_passing_team.csv")
H = json.load(open("data/pv_nfl_passing_hyper.json"))
assert V.season.max() < TEST_ERA and T.season.max() < TEST_ERA
assert len(V) == len(O) and (V.game_id.values == O.game_id.values).all() and (V.qb_id.values == O.qb_id.values).all()
Q = pd.concat([V, O.drop(columns=["game_id", "team", "qb_id"])], axis=1)
Q = Q[(Q.season >= DEV_LO) & (Q.season <= DEV_HI)].copy()
assert Q.season.max() <= DEV_HI
OUT = {"dev": f"{DEV_LO}-{DEV_HI}", "n_qb_games": int(len(Q))}

# final scores for the win outcome (DEV rows only; no odds column is read)
SC = {}
with open("data/nfl_games.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["season"] and DEV_LO <= int(r["season"]) <= DEV_HI and r["home_score"] != "":
            SC[r["game_id"]] = (float(r["home_score"]), float(r["away_score"]))


def wcorr(x, y, w=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    mx = (w * x).sum() / w.sum(); my = (w * y).sum() / w.sum()
    cxy = (w * (x - mx) * (y - my)).sum(); cxx = (w * (x - mx) ** 2).sum(); cyy = (w * (y - my) ** 2).sum()
    return float(cxy / np.sqrt(cxx * cyy))


def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(max(n - 3, 1))
    return [round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)]


# ============================================================ 1. reliability
Q["adj_epa"] = Q.s_epa / Q.n_db - Q.lg_epa - Q.opp_def_epa     # per-game, walk-forward
Q["adj_succ"] = Q.s_succ / Q.n_db - Q.lg_succ - Q.opp_def_succ
Q["adj_sack"] = Q.n_sack / Q.n_db - Q.lg_sack - Q.opp_def_sack
Q["cpoe_g"] = np.where(Q.n_cp > 0, Q.s_cpoe / Q.n_cp.clip(lower=1), np.nan)
Q["adj_cpoe"] = Q.cpoe_g - Q.lg_cpoe - Q.opp_def_cpoe

# the shipped QbElo input: (passing_epa + rushing_epa) / (attempts + sacks + carries), weekly lines
wk = {}
with open("data/nfl_player_stats.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["position"] != "QB" or not (DEV_LO <= int(r["season"]) <= DEV_HI):
            continue
        f = lambda x: float(x) if x not in ("", "NA") else 0.0  # noqa: E731
        k = (r["player_id"], int(r["season"]), int(r["week"]))
        e = f(r["passing_epa"]) + f(r["rushing_epa"]); d = f(r["attempts"]) + f(r["sacks"]) + f(r["carries"])
        e0, d0 = wk.get(k, (0.0, 0.0)); wk[k] = (e0 + e, d0 + d)
Q["ship_num"] = [wk.get((q, s, w), (np.nan, np.nan))[0] for q, s, w in zip(Q.qb_id, Q.season, Q.week)]
Q["ship_den"] = [wk.get((q, s, w), (np.nan, np.nan))[1] for q, s, w in zip(Q.qb_id, Q.season, Q.week)]

WC = H["composite_weights_by_season"]["2015"]["qb_coef"]      # trained 2007-2014 (DEV)
STATS = {
    # name: (per-game value column, per-game weight column)
    "raw EPA/db": ("raw_epa", "n_db"),
    "opp-adj EPA/db": ("adj_epa", "n_db"),
    "opp-adj success": ("adj_succ", "n_db"),
    "opp-adj sack rate": ("adj_sack", "n_db"),
    "INT rate": ("int_rate", "n_att"),
    "CPOE": ("cpoe_g", "n_cp"),
    "opp-adj CPOE": ("adj_cpoe", "n_cp"),
    "shipped input (EPA incl. rush)/(att+sk+car)": ("ship_rate", "ship_den"),
}
Q["raw_epa"] = Q.s_epa / Q.n_db
Q["int_rate"] = np.where(Q.n_att > 0, Q.n_int / Q.n_att.clip(lower=1), np.nan)
Q["ship_rate"] = Q.ship_num / Q.ship_den.replace(0, np.nan)


def season_stat(df, col, wcol):
    m = df[col].notna() & (df[wcol] > 0)
    d = df[m]
    return (d[col] * d[wcol]).sum() / d[wcol].sum() if len(d) else np.nan


def composite_stat(df):
    return (WC["epa"] * season_stat(df, "adj_epa", "n_db") + WC["succ"] * season_stat(df, "adj_succ", "n_db")
            + WC["sack"] * season_stat(df, "adj_sack", "n_db") + WC["cpoe"] * season_stat(df, "adj_cpoe", "n_cp"))


Q = Q.sort_values(["qb_id", "season", "game_date"])
Q["gno"] = Q.groupby(["qb_id", "season"]).cumcount()
qs_db = Q.groupby(["qb_id", "season"]).n_db.sum()
elig = qs_db[qs_db >= 200].index
rel = {}
halves = {}
seasonal = {}
for (qb, s), d in Q.groupby(["qb_id", "season"]):
    if (qb, s) not in elig:
        continue
    a, b = d[d.gno % 2 == 0], d[d.gno % 2 == 1]
    halves[(qb, s)] = {nm: (season_stat(a, c, w), season_stat(b, c, w)) for nm, (c, w) in STATS.items()}
    halves[(qb, s)]["passing composite (season stat, 2015-fold weights)"] = (composite_stat(a), composite_stat(b))
    seasonal[(qb, s)] = {nm: season_stat(d, c, w) for nm, (c, w) in STATS.items()}
    seasonal[(qb, s)]["passing composite (season stat, 2015-fold weights)"] = composite_stat(d)
names = list(STATS) + ["passing composite (season stat, 2015-fold weights)"]
for nm in names:
    xs = np.array([[v[nm][0], v[nm][1]] for v in halves.values()], float)
    xs = xs[np.isfinite(xs).all(1)]
    r_sh = wcorr(xs[:, 0], xs[:, 1])
    pairs = [(seasonal[(q, s)][nm], seasonal[(q, s + 1)][nm]) for (q, s) in seasonal if (q, s + 1) in seasonal]
    pairs = np.array([p for p in pairs if np.isfinite(p).all()], float)
    r_y = wcorr(pairs[:, 0], pairs[:, 1])
    rel[nm] = {"split_half_r": round(r_sh, 3), "split_half_ci": fisher_ci(r_sh, len(xs)),
               "spearman_brown_full_season": round(2 * r_sh / (1 + r_sh), 3), "n_qb_seasons": int(len(xs)),
               "yoy_r": round(r_y, 3), "yoy_ci": fisher_ci(r_y, len(pairs)), "n_yoy_pairs": int(len(pairs))}
    print(f"  {nm:<52} split-half {r_sh:.3f} (SB {2*r_sh/(1+r_sh):.3f}, n={len(xs)})  YoY {r_y:.3f} (n={len(pairs)})")
OUT["reliability"] = rel

# ====================================================== 2. predictive validity
S = Q[Q.is_starter == 1].copy()
S = S.merge(T[["game_id", "posteam", "off_pts", "drives", "u_db", "u_epa"]],
            left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left")
S["ppd"] = S.off_pts / S.drives
S["unit_epa"] = S.u_epa / S.u_db
S["win"] = [(1.0 if (SC[g][0] > SC[g][1]) == (sd == "home") else 0.0) if SC[g][0] != SC[g][1] else 0.5
            for g, sd in zip(S.game_id, S.side)]
S["unit3"] = S.epa3_q + S.epa3_cast
assert S.season.max() <= DEV_HI
RATINGS = {
    "passing_composite": "passing composite (opp-adj EPA+succ+sack+CPOE, walk-forward weights)",
    "epa_q": "opp-adj EPA/db Kalman (QB x pass-defence duel)",
    "epa_raw_q": "un-adjusted EPA/db Kalman (control)",
    "epax_q": "opp-adj EPA incl. designed QB runs",
    "passing_composite3": "passing composite, EPA part cast-stripped (QB x cast x defence filter)",
    "epa3_q": "cast-stripped opp-adj EPA/db (3-state filter), QB part",
    "unit3": "3-state unit forecast: QB part + team cast part",
    "cpoe_q": "CPOE rating alone", "sack_q": "sack-rate rating alone", "succ_q": "success rating alone",
    "qbelo_starter": "SHIPPED QbElo (starter-keyed, as served)",
    "qbelo_listed": "shipped QbElo (listed-QB walk, as cached in X14)",
}
OUTCOMES = {"qb_epa": ("raw_epa", "n_db"), "unit_epa": ("unit_epa", "u_db"), "ppd": ("ppd", None),
            "off_pts": ("off_pts", None), "win": ("win", None)}
gidx = pd.factorize(S.game_id)[0]
G = gidx.max() + 1
BOOT = [RNG.integers(0, G, size=G) for _ in range(NB)]
rows_of = [np.where(gidx == g)[0] for g in range(G)]


def boot_rows(b):
    return np.concatenate([rows_of[g] for g in b])


BIDX = [boot_rows(b) for b in BOOT[:NB]]
# opponent control: residualise every outcome on the opponent's pre-game pass-defence rating
opp = S.opp_def_epa.values


def resid(y, w):
    X = np.column_stack([np.ones(len(y)), opp])
    ww = np.ones(len(y)) if w is None else w
    b = np.linalg.lstsq(X * np.sqrt(ww)[:, None], y * np.sqrt(ww), rcond=None)[0]
    return y - X @ b


pv = {}
REF = "qbelo_starter"
for on, (oc, wc) in OUTCOMES.items():
    m = S[oc].notna().values & ((S[wc] > 0).values if wc else True)
    y = S[oc].values[m]; w = None if wc is None else S[wc].values[m].astype(float)
    yr = resid(y, w)
    res = {}
    bidx = [bi[m[bi]] for bi in BIDX]
    # map boot rows (indices into S) to indices into the masked arrays
    pos = -np.ones(len(S), int); pos[np.where(m)[0]] = np.arange(m.sum())
    bidx = [pos[bi] for bi in bidx]
    ref = S[REF].values[m]
    for rk, desc in RATINGS.items():
        x = S[rk].values[m]
        ok = np.isfinite(x)
        r = wcorr(x[ok], y[ok], None if w is None else w[ok])
        rp = wcorr(x[ok], yr[ok], None if w is None else w[ok])
        d_boot = []
        for bi in bidx:
            bi = bi[ok[bi]]
            wb = None if w is None else w[bi]
            d_boot.append(wcorr(x[bi], yr[bi], wb) - wcorr(ref[bi], yr[bi], wb))
        dlo, dhi = np.percentile(d_boot, [2.5, 97.5])
        res[rk] = {"desc": desc, "r": round(r, 4), "r_opp_controlled": round(rp, 4),
                   "delta_r_vs_shipped_opp_controlled": round(rp - wcorr(ref[ok], yr[ok], None if w is None else w[ok]), 4),
                   "delta_ci": [round(float(dlo), 4), round(float(dhi), 4)], "n": int(ok.sum())}
    # incremental validity: both ratings (z) + opponent in one regression
    xc = S["passing_composite"].values[m]; xs_ = ref
    Z = np.column_stack([np.ones(len(y)), (xc - xc.mean()) / xc.std(), (xs_ - xs_.mean()) / xs_.std(), opp[m]])
    ww = np.ones(len(y)) if w is None else w

    def ols(ix):
        b = np.linalg.lstsq(Z[ix] * np.sqrt(ww[ix])[:, None], y[ix] * np.sqrt(ww[ix]), rcond=None)[0]
        return b[1], b[2]
    b0 = ols(np.arange(len(y)))
    bb = np.array([ols(bi) for bi in bidx[:1000]])
    res["_joint_regression_z_coefs"] = {
        "composite": [round(float(b0[0]), 4)] + [round(float(v), 4) for v in np.percentile(bb[:, 0], [2.5, 97.5])],
        "shipped_qbelo_starter": [round(float(b0[1]), 4)] + [round(float(v), 4) for v in np.percentile(bb[:, 1], [2.5, 97.5])],
        "note": "coef per 1 SD of each rating, same regression, + opponent pass-D control; [point, lo, hi]"}
    pv[on] = res
    print(f"\n  outcome {on} (n={m.sum()})")
    for rk in RATINGS:
        v = res[rk]
        print(f"    {rk:<18} r {v['r']:+.4f}  r|opp {v['r_opp_controlled']:+.4f}  "
              f"d vs shipped {v['delta_r_vs_shipped_opp_controlled']:+.4f} CI{v['delta_ci']}")
    print("    joint z-coefs:", res["_joint_regression_z_coefs"])
OUT["predictive_next_game_starters"] = pv

# ---- acid test: QBs on a NEW team (never played for it before, >= 100 prior dropbacks).
# A player-quality number should travel with the player; teammate quality should not.
A_ = pd.concat([V, O.drop(columns=["game_id", "team", "qb_id"])], axis=1).sort_values(["game_date", "game_id"])
A_["cum"] = A_.groupby("qb_id").n_db.cumsum() - A_.n_db
seen = set(); newteam = []
for q, t in zip(A_.qb_id, A_.team):
    newteam.append((q, t) not in seen); seen.add((q, t))
A_["new_team"] = newteam
A_["first_on_team_season"] = A_.groupby(["qb_id", "team"]).season.transform("min")
mv = A_[(A_.cum >= 100)].copy()
movers = set(zip(mv.qb_id[mv.new_team], mv.team[mv.new_team]))
S = S.merge(A_[["game_id", "qb_id", "cum", "first_on_team_season"]], on=["game_id", "qb_id"], how="left")
S["mover"] = [((q, t) in movers) and (s_ == f_) for q, t, s_, f_ in
              zip(S.qb_id, S.team, S.season, S.first_on_team_season)]
MV = S[S.mover]
mvres = {"n_qb_games": int(len(MV)), "n_qbs": int(MV.qb_id.nunique()),
         "note": "starter games in the QB's first season with a team he had never played for, "
                 ">= 100 career dropbacks before joining; DEV 2006-2015"}
mg = pd.factorize(MV.game_id)[0]
for on, (oc, wc) in (("qb_epa", ("raw_epa", "n_db")), ("ppd", ("ppd", None))):
    y = MV[oc].values; w = None if wc is None else MV[wc].values.astype(float)
    ok = np.isfinite(y)
    rr = {k: round(wcorr(MV[k].values[ok], y[ok], None if w is None else w[ok]), 4) for k in RATINGS}
    bs = {k: [] for k in ("passing_composite", "passing_composite3", "epa3_q", "epa_q")}
    ix = np.where(ok)[0]
    for _ in range(NB):
        b = ix[RNG.integers(0, len(ix), len(ix))]
        wb = None if w is None else w[b]
        ref_ = wcorr(MV[REF].values[b], y[b], wb)
        for k in bs:
            bs[k].append(wcorr(MV[k].values[b], y[b], wb) - ref_)
    mvres[on] = {"r": rr, "delta_vs_shipped_ci": {k: [round(float(np.mean(v)), 4)] + [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])] for k, v in bs.items()}}
    print(f"\n  MOVERS {on} (n={ok.sum()}):", rr)
    print("   delta vs shipped [mean, lo, hi]:", mvres[on]["delta_vs_shipped_ci"])
OUT["acid_test_new_team"] = mvres

# per-season stability of the headline comparison (QB EPA/db and points per drive)
per = {}
for s in range(DEV_LO, DEV_HI + 1):
    d = S[(S.season == s)]
    per[s] = {"qb_epa_r": [round(wcorr(d[k], d.raw_epa, d.n_db), 4) for k in ("passing_composite", REF)],
              "ppd_r": [round(wcorr(d[k], d.ppd), 4) for k in ("passing_composite", REF)], "n": int(len(d))}
OUT["per_season_[composite,shipped]"] = per
print("\n  per season [composite, shipped]:", {s: (v["qb_epa_r"], v["ppd_r"]) for s, v in per.items()})

# next-season: rating at the QB's first game of s+1 vs his season s+1 totals (>= 200 db in s+1)
ns = []
for (qb, s), d in Q.groupby(["qb_id", "season"]):
    if s <= DEV_LO or d.n_db.sum() < 200:
        continue
    prev = Q[(Q.qb_id == qb) & (Q.season == s - 1)]
    if prev.n_db.sum() < 1:
        continue
    f = d.sort_values("game_date").iloc[0]
    st = S[(S.qb_id == qb) & (S.season == s)]
    ns.append({"qb": qb, "s": s, "comp": f.passing_composite, "epa_q": f.epa_q, "epa_raw_q": f.epa_raw_q,
               "comp3": f.passing_composite3, "epa3_q": f.epa3_q,
               "ship": f.qbelo_starter, "ship_l": f.qbelo_listed,
               "y_epa": d.s_epa.sum() / d.n_db.sum(), "w": d.n_db.sum(),
               "y_adj": season_stat(d, "adj_epa", "n_db"),
               "y_ppd": st.off_pts.sum() / st.drives.sum() if len(st) else np.nan})
NS = pd.DataFrame(ns)
nsres = {"n": int(len(NS)), "note": "QB-seasons 2007-2015 with >=200 db and any dropback in s-1"}
for k in ("comp", "comp3", "epa_q", "epa3_q", "epa_raw_q", "ship", "ship_l"):
    ok = NS[k].notna()
    m2 = ok & NS.y_ppd.notna()
    nsres[k] = {"r_epa": round(wcorr(NS[k][ok], NS.y_epa[ok], NS.w[ok]), 4),
                "r_oppadj_epa": round(wcorr(NS[k][ok], NS.y_adj[ok], NS.w[ok]), 4),
                "r_ppd_as_starter": round(wcorr(NS[k][m2], NS.y_ppd[m2]), 4)}
bs = []
idx = np.arange(len(NS))
for _ in range(NB):
    b = RNG.choice(idx, len(idx))
    d = NS.iloc[b]
    bs.append(wcorr(d.comp, d.y_epa, d.w) - wcorr(d.ship, d.y_epa, d.w))
nsres["delta_r_epa_comp_minus_ship_ci"] = [round(float(v), 4) for v in np.percentile(bs, [2.5, 97.5])]
OUT["predictive_next_season"] = nsres
print("\n  next season:", json.dumps(nsres))

# ============================================================ 3. face validity
end = V[(V.season == DEV_HI)].sort_values("game_date").groupby("qb_id").tail(1)
vol = O.merge(V[["game_id", "qb_id", "season"]], on=["game_id", "qb_id"])
vol = vol[(vol.season >= DEV_HI - 1) & (vol.season <= DEV_HI)].groupby("qb_id").n_db.sum()
end = end[end.qb_id.map(vol).fillna(0) >= 400]
cols = ["qb_name", "passing_composite", "qbelo_starter", "epa_q", "cpoe_q", "sack_q", "passing_composite3"]
fv = {"end_2015_top15": end.sort_values("passing_composite", ascending=False)[cols].head(15).round(4).values.tolist(),
      "end_2015_bottom10": end.sort_values("passing_composite")[cols].head(10).round(4).values.tolist(),
      "note": "pre-game value at each QB's last 2015 game; >= 400 dropbacks in 2014-15; EPA/db units"}
d_ = end.assign(gap=end.passing_composite - end.qbelo_starter)
fv["biggest_disagreements_composite_minus_shipped"] = (
    d_.reindex(d_.gap.abs().sort_values(ascending=False).index)[cols + ["gap"]].head(8).round(4).values.tolist())
for s in (2007, 2011):
    e = V[V.season == s].sort_values("game_date").groupby("qb_id").tail(1)
    starts = V[(V.season == s) & (V.is_starter == 1)].groupby("qb_id").size()
    e = e[e.qb_id.map(starts).fillna(0) >= 8]
    fv[f"{s}_top5"] = e.sort_values("passing_composite", ascending=False)[cols].head(5).round(4).values.tolist()
    fv[f"{s}_bottom5"] = e.sort_values("passing_composite")[cols].head(5).round(4).values.tolist()
OUT["face_validity"] = fv
for k, v in fv.items():
    print(f"\n  {k}")
    if isinstance(v, list):
        for row in v:
            print("    ", row)
json.dump(OUT, open("data/pv_nfl_passing_validation.json", "w"), indent=1, default=float)
print("\nwrote data/pv_nfl_passing_validation.json")
