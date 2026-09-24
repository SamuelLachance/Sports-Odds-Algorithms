"""pv NFL pass_rush_and_front -- step 4: PLAYER- and UNIT-level validation. DEV ONLY (2006-2015).

Reads the walk-forward outputs of pv_nfl_pass_rush_and_front.py and asks the
three questions of the program, before any game-level screen:

 1. RELIABILITY  split-half (odd/even present games inside a season,
    Spearman-Brown corrected) and year-over-year correlations of per-game
    rates, player-seasons with >= 8 present games; n reported.
 2. PREDICTIVE VALIDITY  does the PRE-GAME rating predict the player's FUTURE
    RESULT (sacks, the "runs" of pass rush -- not a process proxy) better than
    (a) the sacks-only rate (the ERA analog) and (b) the CURRENT rating --
    the shipped roster_quality DEF 'rush' component (sack + .5 TFL + .5 hit
    per weekly stat line, EWMA .95, 6-week EB prior; phase0/
    nfl_player_rating_system.py + rq_tune). The shipped 11v11 participation
    TrueSkill cannot be a DEV comparator: participation starts 2016.
    Checkpoints (non-overlapping): CP0 = a player's first present game of a
    season -> that season's sacks/game; CP8 = first present game after the
    team's 8th game -> rest-of-season sacks/game. Pearson r, player-clustered
    bootstrap of r differences.
    UNIT level: do the front / protection ratings predict the team's future
    POINTS (and sacks, pressure, run success) beyond the current team
    EPA/play rating (shipped epa_net dynamics)? Next-game partial r, n.
 3. FACE VALIDITY  top / bottom lists, end of 2009 / 2012 / 2015.
 Plus: snap-truth audit of the presence proxy (2013-15), per-snap exposure
 variant, and what is attributable to individual offensive linemen.

Output: data/pv_nfl_pass_rush_and_front_validation.json
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
DEV_LO, DEV_HI = 2006, 2015
FRONT = ["DE", "DT", "OLB", "ILB"]
EDGE = ["DE", "OLB"]
RNG = np.random.default_rng(20260924)
OUT = {"split": "DEV 2006-2015 only; no season >= 2016 read"}


def r_(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else float("nan")


def sb(r):
    return 2 * r / (1 + r)


# ================================================================ load
PG = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_player_games.parquet"))
TG = pd.read_csv(os.path.join(DATA, "pv_nfl_pass_rush_and_front_team_games.csv"))
assert PG.season.max() < 2016 and TG.season.max() < 2016
TGi = TG.set_index(["game_id", "defteam"])
PG = PG.join(TGi[["g_n_db", "g_n_run", "L_pr", "L_sk", "L_ht", "L_st", "L_tf", "Oopp_pr", "Oopp_sk", "Oopp_ht",
                  "Oopp_st", "Oopp_tf", "S_ht"]], on=["game_id", "team"])
PR = PG[PG.presence != ""].copy()                  # present player-games
PR = PR.sort_values(["player_id", "game_date", "game_id"])
# expected events vs an average unit (pre-game normalisers) -- the adjusted-rate denominators
PR["e_pr"] = PR.g_n_db * PR.L_pr * PR.Oopp_pr * PR.S_ht
PR["e_sk"] = PR.g_n_db * PR.L_sk * PR.Oopp_sk
PR["e_ht"] = PR.g_n_db * PR.L_ht * PR.Oopp_ht * PR.S_ht
PR["e_st"] = PR.g_n_run * PR.L_st * PR.Oopp_st
PR["e_tf"] = PR.g_n_run * PR.L_tf * PR.Oopp_tf
# team game index within season (for checkpoints)
TG = TG.sort_values(["game_date", "game_id"])
TG["tgi"] = TG.groupby(["defteam", "season"]).cumcount() + 1
PR = PR.merge(TG[["game_id", "defteam", "tgi"]].rename(columns={"defteam": "team"}), on=["game_id", "team"])

# ---------------------------------------------------------------- comparator: shipped rq 'rush' component
W = pd.read_csv(os.path.join(DATA, "nfl_player_stats_def.csv"),
                usecols=["season", "week", "player_id", "position", "team", "def_sacks", "def_tackles_for_loss",
                         "def_qb_hits", "def_tackles_solo"])
W = W[W.season <= DEV_HI].sort_values(["season", "week"])
W["rush"] = W.def_sacks + 0.5 * W.def_tackles_for_loss + 0.5 * W.def_qb_hits
W["tak"] = W.def_tackles_solo
W["bk"] = W.position.map(E.pos_bucket)
SHIP_DECAY, SHIP_PRIOR = 0.95, 6.0                 # rq_tune adopted decay .95; EB prior 6 weeks
st_ = defaultdict(lambda: [0.0, 0.0, 0.0])         # pid -> [rush, tak, w]
bsum = defaultdict(lambda: [0.0, 0.0, 0.0])        # bucket -> pooled per-appearance sums (earlier seasons)
need = defaultdict(set)                            # (season, week) -> pids needing a pre-week value
for p_, s_, w_ in zip(PR.player_id, PR.season, PR.week):
    need[(s_, w_)].add(p_)
bk_of = dict(zip(PR.player_id, PR.bucket))
ship_pre = {}
frozen, cur = {}, None
wk_groups = {k: g for k, g in W.groupby(["season", "week"], sort=True)}
for key in sorted(set(wk_groups) | set(need)):
    s_ = key[0]
    if s_ != cur:
        frozen = {b: (v[0] / v[2], v[1] / v[2]) for b, v in bsum.items() if v[2] > 50}
        cur = s_
    for p_ in need.get(key, ()):              # PRE-week state (before this week's line is added)
        a = st_.get(p_, [0.0, 0.0, 0.0])
        mu = frozen.get(bk_of.get(p_), (0.3, 1.5))
        ship_pre[(p_, s_, key[1])] = ((a[0] + SHIP_PRIOR * mu[0]) / (a[2] + SHIP_PRIOR),
                                      (a[1] + SHIP_PRIOR * mu[1]) / (a[2] + SHIP_PRIOR))
    grp = wk_groups.get(key)
    if grp is None:
        continue
    for r in grp.itertuples(index=False):
        a = st_[r.player_id]
        a[0] = SHIP_DECAY * a[0] + r.rush; a[1] = SHIP_DECAY * a[1] + r.tak; a[2] = SHIP_DECAY * a[2] + 1
        b = bsum[r.bk]; b[0] += r.rush; b[1] += r.tak; b[2] += 1
PR["ship_rush"] = [ship_pre[(p, s, w)][0] for p, s, w in zip(PR.player_id, PR.season, PR.week)]
PR["ship_tak"] = [ship_pre[(p, s, w)][1] for p, s, w in zip(PR.player_id, PR.season, PR.week)]

# ---------------------------------------------------------------- comparator: raw EB per-game rates (no opp adj)
def raw_eb(col, K, d, s, name):
    vals = np.full(len(PR), np.nan)
    num, den = defaultdict(float), defaultdict(float)
    last_season = {}
    pooled = defaultdict(lambda: [0.0, 0.0]); frozen_ = {}; cur_ = None
    pid, sea, bk, y = PR.player_id.to_numpy(), PR.season.to_numpy(), PR.bucket.to_numpy(), PR[col].to_numpy(float)
    order = np.argsort(PR.game_date.to_numpy(), kind="stable")
    for i in order:
        if sea[i] != cur_:
            frozen_ = {b: v[0] / v[1] for b, v in pooled.items() if v[1] > 50}; cur_ = sea[i]
        p = pid[i]
        if p in last_season and last_season[p] != sea[i]:
            num[p] *= s; den[p] *= s
        last_season[p] = sea[i]
        mu = frozen_.get(bk[i], 0.2)
        vals[i] = (num[p] + K * mu) / (den[p] + K)
        num[p] = d * num[p] + y[i]; den[p] = d * den[p] + 1.0
        pooled[bk[i]][0] += y[i]; pooled[bk[i]][1] += 1
    PR[name] = vals


PR = PR.reset_index(drop=True)
raw_eb("g_sack", 8.0, 0.97, 0.75, "raw_sk_pg")        # sacks-only, per game (the ERA analog)
raw_eb("g_pr", 8.0, 0.97, 0.75, "raw_pr_pg")          # sack + hit, per game, no opponent/scorer adjustment
raw_eb("g_stop", 8.0, 0.94, 0.55, "raw_st_pg")

# ================================================================ 0. WHO OWNS A TEAM-GAME EVENT? (design check)
# per DEV season, fit events = n x L x Defence x Offence x Stadium(scorer) on odd weeks and on even weeks
# separately (ratio fitting, light shrink) and correlate the two halves' log factors: a factor that is a
# persistent trait replicates; bookkeeping (home stats crew) shows up as a replicating Stadium factor.
UN = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_units.parquet"))
UN = UN[UN.season.between(DEV_LO, DEV_HI)]
_gm = pd.read_csv(os.path.join(DATA, "nfl_games.csv"), usecols=["game_id", "home_team"])
UN = UN.merge(_gm, on="game_id"); UN["stad"] = UN.home_team.map(lambda t: E.FR.get(t, t))


def _fit3(x, col, nc, it=60, reg=3.0):
    L = x[col].sum() / x[nc].sum()
    F = {k: {t: 1.0 for t in x[k].unique()} for k in ("defteam", "posteam", "stad")}
    for _ in range(it):
        for k in F:
            e = x[nc] * L
            for k2 in F:
                if k2 != k:
                    e = e * x[k2].map(F[k2])
            num, den = x[col].groupby(x[k]).sum(), e.groupby(x[k]).sum()
            F[k] = {t: (num[t] + reg) / (den[t] + reg) for t in F[k]}
    return F


own = {}
for col, nc in (("hits_ns", "n_db"), ("sacks", "n_db"), ("pressure", "n_db"), ("stop_plays", "n_run"),
                ("tfl_plays", "n_run")):
    acc = {"defteam": [], "posteam": [], "stad": []}
    for s_ in range(DEV_LO, DEV_HI + 1):
        x = UN[UN.season == s_]
        h0, h1 = _fit3(x[x.week % 2 == 0], col, nc), _fit3(x[x.week % 2 == 1], col, nc)
        for k in acc:
            a, b = pd.Series(h0[k]), pd.Series(h1[k]); ix = a.index.intersection(b.index)
            acc[k].append(r_(np.log(a[ix]), np.log(b[ix])))
    own[col] = {{"defteam": "defence", "posteam": "offence", "stad": "stadium_scorer"}[k]: round(float(np.mean(v)), 3)
                for k, v in acc.items()}
OUT["0_split_half_factor_replication_by_season_mean"] = own
print("factor replication:", own)

# ================================================================ 1. RELIABILITY
dev = PR[PR.season.between(DEV_LO, DEV_HI) & PR.bucket.isin(FRONT)].copy()
dev["k"] = dev.groupby(["player_id", "season"]).cumcount()
dev["half"] = dev.k % 2
# shipped composite per game, realised (weekly line) for reliability of the current metric's input
Wg = W.set_index(["player_id", "season", "week"])[["rush", "tak"]]
dev = dev.join(Wg, on=["player_id", "season", "week"])
dev[["rush", "tak"]] = dev[["rush", "tak"]].fillna(0.0)

MEAS = {  # name: (numerator col, denominator col or None=per game)
    "sacks_per_game": ("g_sack", None), "hits_per_game": ("g_hit", None), "pressure_per_game": ("g_pr", None),
    "stops_per_game": ("g_stop", None), "tfl_per_game": ("g_tfl", None),
    "sacks_adj": ("g_sack", "e_sk"), "hits_adj": ("g_hit", "e_ht"), "pressure_adj": ("g_pr", "e_pr"),
    "stops_adj": ("g_stop", "e_st"), "tfl_adj": ("g_tfl", "e_tf"),
    "CURRENT_ship_rush_per_game": ("rush", None), "CURRENT_ship_tackles_per_game": ("tak", None),
}


def rate_tab(df, keys):
    g = df.groupby(keys)
    out = pd.DataFrame({"n": g.size()})
    for nm, (a, b) in MEAS.items():
        out[nm] = g[a].sum() / (g[b].sum() if b else g.size())
    return out


rel = {}
for grpname, buckets in (("front", FRONT), ("edge", EDGE), ("interior", ["DT", "ILB"])):
    d_ = dev[dev.bucket.isin(buckets)]
    tot = d_.groupby(["player_id", "season"]).size()
    keep = tot[tot >= 8].index
    d_ = d_.set_index(["player_id", "season"]).loc[keep].reset_index()
    h = rate_tab(d_, ["player_id", "season", "half"]).reset_index()
    h0 = h[h.half == 0].set_index(["player_id", "season"]); h1 = h[h.half == 1].set_index(["player_id", "season"])
    ix = h0.index.intersection(h1.index)
    sh = {nm: round(sb(r_(h0.loc[ix, nm], h1.loc[ix, nm])), 3) for nm in MEAS}
    ss = rate_tab(d_, ["player_id", "season"])
    a = ss.reset_index(); b = a.copy(); b["season"] -= 1
    yy = a.merge(b, on=["player_id", "season"], suffixes=("", "_next"))
    yoy = {nm: round(r_(yy[nm], yy[nm + "_next"]), 3) for nm in MEAS}
    rel[grpname] = {"split_half_spearman_brown": sh, "n_player_seasons_split": int(len(ix)),
                    "year_over_year_r": yoy, "n_pairs_yoy": int(len(yy))}
OUT["1_reliability"] = rel
print("RELIABILITY (front):", json.dumps(rel["front"], indent=None)[:1500])

# ================================================================ 2. PLAYER PREDICTIVE VALIDITY
def checkpoint_table():
    x = PR[PR.season.between(DEV_LO + 1, DEV_HI) & PR.bucket.isin(FRONT)].copy()
    x = x.sort_values(["player_id", "season", "game_date"])
    rows = []
    for (p, s), g in x.groupby(["player_id", "season"], sort=False):
        if len(g) >= 8:
            f = g.iloc[0]
            rows.append(("CP0", p, s, f.bucket, g.g_sack.mean(), g.g_pr.mean(), g.g_stop.mean(),
                         g.g_sack.sum() / g.g_n_db.sum(), len(g), f))
        later = g[g.tgi >= 9]
        if len(later) >= 4:
            f = later.iloc[0]
            rows.append(("CP8", p, s, f.bucket, later.g_sack.mean(), later.g_pr.mean(), later.g_stop.mean(),
                         later.g_sack.sum() / later.g_n_db.sum(), len(later), f))
    cols = ["th_sk", "th_pr", "th_ht", "th_st", "th_tf", "raw_sk_pg", "raw_pr_pg", "raw_st_pg", "ship_rush",
            "ship_tak", "neff_pr"]
    t = pd.DataFrame([(a, p, s, b, y1, y2, y3, y4, n, *[f[c] for c in cols]) for a, p, s, b, y1, y2, y3, y4, n, f in rows],
                     columns=["cp", "player_id", "season", "bucket", "fut_sacks_pg", "fut_pr_pg", "fut_stops_pg",
                              "fut_sacks_per_db", "fut_n"] + cols)
    return t


CP = checkpoint_table()

# FIP-style weights: future sacks ~ th_sk + th_ht, fit on one half of DEV, applied to the other (cross-fit)
def fitw(t):
    X = np.column_stack([t.th_sk, t.th_ht, np.ones(len(t))]); y = t.fut_sacks_pg.to_numpy()
    w = np.linalg.lstsq(X * np.sqrt(t.fut_n.to_numpy())[:, None], y * np.sqrt(t.fut_n.to_numpy()), rcond=None)[0]
    return w


A_ = CP[CP.season <= 2010]; B_ = CP[CP.season >= 2011]
wA, wB = fitw(A_), fitw(B_)
CP["fip_pr"] = np.where(CP.season <= 2010, wB[0] * CP.th_sk + wB[1] * CP.th_ht, wA[0] * CP.th_sk + wA[1] * CP.th_ht)
OUT["2_fip_weights"] = {"fit_2007_2010": {"th_sk": round(wA[0], 3), "th_ht": round(wA[1], 3)},
                        "fit_2011_2015": {"th_sk": round(wB[0], 3), "th_ht": round(wB[1], 3)},
                        "hit_weight_relative_to_sack": [round(wA[1] / wA[0], 3), round(wB[1] / wB[0], 3)],
                        "note": "weights cross-fit: each half of DEV is scored with the other half's weights"}
print("FIP weights (sack, hit):", OUT["2_fip_weights"])


def boot_diff(t, a, b, y, B=2000):
    """player-clustered bootstrap of r(a,y) - r(b,y)."""
    pids = t.player_id.unique()
    idx = {p: np.where(t.player_id.to_numpy() == p)[0] for p in pids}
    A, Bv, Y = t[a].to_numpy(float), t[b].to_numpy(float), t[y].to_numpy(float)
    ds = []
    for _ in range(B):
        samp = np.concatenate([idx[p] for p in RNG.choice(pids, size=len(pids))])
        ds.append(r_(A[samp], Y[samp]) - r_(Bv[samp], Y[samp]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


pv = {}
PREDS_PASS = ["fip_pr", "th_pr", "th_sk", "th_ht", "raw_sk_pg", "raw_pr_pg", "ship_rush"]
for cp in ("CP0", "CP8"):
    for grp, bk in (("front", FRONT), ("edge", EDGE)):
        t = CP[(CP.cp == cp) & CP.bucket.isin(bk)].dropna(subset=PREDS_PASS)
        res = {"n": int(len(t)), "n_players": int(t.player_id.nunique())}
        for y in ("fut_sacks_pg", "fut_sacks_per_db", "fut_pr_pg"):
            res[y] = {p: round(r_(t[p], t[y]), 3) for p in PREDS_PASS}
        res["fip_pr_minus_CURRENT_ship_rush_r_sacks_ci95"] = boot_diff(t, "fip_pr", "ship_rush", "fut_sacks_pg", 1000)
        res["fip_pr_minus_raw_sacks_r_sacks_ci95"] = boot_diff(t, "fip_pr", "raw_sk_pg", "fut_sacks_pg", 1000)
        res["th_pr_minus_raw_pr_r_sacks_ci95"] = boot_diff(t, "th_pr", "raw_pr_pg", "fut_sacks_pg", 1000)
        pv[f"{cp}_{grp}"] = res
        print(cp, grp, res["n"], "r(fut sacks/g):", res["fut_sacks_pg"],
              "CI fip-ship", res["fip_pr_minus_CURRENT_ship_rush_r_sacks_ci95"])
    # run front
    t = CP[(CP.cp == cp) & CP.bucket.isin(FRONT)].dropna(subset=["th_st", "raw_st_pg", "ship_tak"])
    pv[f"{cp}_front_run"] = {"n": int(len(t)),
                            "fut_stops_pg": {p: round(r_(t[p], t.fut_stops_pg), 3)
                                             for p in ("th_st", "th_tf", "raw_st_pg", "ship_tak", "ship_rush")},
                            "th_st_minus_CURRENT_ship_tak_ci95": boot_diff(t, "th_st", "ship_tak", "fut_stops_pg", 1000)}
    print(cp, "run", pv[f"{cp}_front_run"])
OUT["2_player_predictive_validity"] = pv


# fairness: give the CURRENT comparator its own grid (decay x prior x season carry) on the same checkpoints
def ship_variant(decay, prior, carry):
    stv = defaultdict(lambda: [0.0, 0.0]); bs = defaultdict(lambda: [0.0, 0.0]); pre_ = {}; fz = {}; cu = None
    for key in sorted(set(wk_groups) | set(need)):
        if key[0] != cu:
            fz = {b: v[0] / v[1] for b, v in bs.items() if v[1] > 50}; cu = key[0]
            for a in stv.values():
                a[0] *= carry; a[1] *= carry
        for p_ in need.get(key, ()):
            a = stv.get(p_, [0.0, 0.0]); mu = fz.get(bk_of.get(p_), 0.3)
            pre_[(p_, key[0], key[1])] = (a[0] + prior * mu) / (a[1] + prior)
        grp = wk_groups.get(key)
        if grp is None:
            continue
        for r in grp.itertuples(index=False):
            a = stv[r.player_id]; a[0] = decay * a[0] + r.rush; a[1] = decay * a[1] + 1
            b = bs[r.bk]; b[0] += r.rush; b[1] += 1
    return [pre_[(p, s, w)] for p, s, w in zip(PR.player_id, PR.season, PR.week)]


keep_ship = PR.ship_rush.copy()
fair = []
for dcy in (0.9, 0.95, 0.97, 0.985):
    for pri in (3.0, 6.0, 12.0):
        for car in (1.0, 0.75):
            PR["ship_rush"] = ship_variant(dcy, pri, car)
            c_ = checkpoint_table()
            rr = {}
            for cp in ("CP0", "CP8"):
                for grp, bk in (("front", FRONT), ("edge", EDGE)):
                    t = c_[(c_.cp == cp) & c_.bucket.isin(bk)].dropna(subset=["ship_rush", "th_pr"])
                    rr[f"{cp}_{grp}"] = round(r_(t.ship_rush, t.fut_sacks_pg), 3)
            fair.append({"decay": dcy, "prior": pri, "carry": car, **rr})
PR["ship_rush"] = keep_ship
best_fair = {k: max(fair, key=lambda z: z[k])[k] for k in ("CP0_front", "CP0_edge", "CP8_front", "CP8_edge")}
OUT["2c_current_rating_best_of_grid"] = {
    "best_r_future_sacks_per_game": best_fair,
    "ours_th_pr": {k: pv[k]["fut_sacks_pg"]["th_pr"] for k in best_fair},
    "grid": "decay .90/.95/.97/.985 x prior 3/6/12 x season carry 1/.75 (24 variants; best per checkpoint, "
            "chosen after the fact -- favours the comparator)"}
print("fairness:", OUT["2c_current_rating_best_of_grid"])

# ================================================================ 2b. UNIT-LEVEL PREDICTIVE VALIDITY (points)
ev = pd.read_parquet(os.path.join(DATA, "pv_nfl_events.parquet"), columns=["game_id", "season", "posteam", "defteam",
                                                                           "play_type", "epa", "game_date"],
                     filters=[("season", "<=", DEV_HI)])
assert ev.season.max() <= DEV_HI
ev = ev[ev.play_type.isin(["pass", "run"]) & ev.epa.notna()]
ev["posteam"] = ev.posteam.map(lambda t: E.FR.get(t, t)); ev["defteam"] = ev.defteam.map(lambda t: E.FR.get(t, t))
ge = ev.groupby(["game_id", "posteam", "defteam"]).agg(season=("season", "first"), date=("game_date", "first"),
                                                       epa=("epa", "sum"), n=("epa", "size")).reset_index()
DEC, PN, SDEC, LG = 0.8813581205848026, 754.0438610349224, 0.7130881070009863, -0.0233   # shipped epa_net dynamics
offs, defs = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
pre = {}
cur = None
for date, day in ge.sort_values("date").groupby("date", sort=True):
    s = int(day.season.iloc[0])
    if s != cur and cur is not None:
        for dd in (offs, defs):
            for v in dd.values():
                v[0] *= SDEC; v[1] *= SDEC
    cur = s
    for r in day.itertuples(index=False):
        o, d = offs[r.posteam], defs[r.defteam]
        pre[(r.game_id, r.defteam)] = ((d[0] + PN * LG) / (d[1] + PN), (o[0] + PN * LG) / (o[1] + PN))
    for r in day.itertuples(index=False):
        o, d = offs[r.posteam], defs[r.defteam]
        o[0] = DEC * o[0] + r.epa; o[1] = DEC * o[1] + r.n
        d[0] = DEC * d[0] + r.epa; d[1] = DEC * d[1] + r.n
TG["epa_def_allowed"] = [pre.get((g, d), (np.nan, np.nan))[0] for g, d in zip(TG.game_id, TG.defteam)]
TG["epa_opp_off"] = [pre.get((g, d), (np.nan, np.nan))[1] for g, d in zip(TG.game_id, TG.defteam)]
# the defending team's own offence protection (its row as posteam)
own = TG[["game_id", "posteam", "Oopp_pr", "Oopp_sk", "Oopp_st"]].rename(
    columns={"posteam": "defteam", "Oopp_pr": "own_prot_pr", "Oopp_sk": "own_prot_sk", "Oopp_st": "own_run_st"})
TG = TG.merge(own, on=["game_id", "defteam"], how="left")
U = TG[TG.season.between(DEV_LO, DEV_HI)].copy()
U["pr_rate"] = U.g_pressure / U.g_n_db
U["sk_rate"] = U.g_sacks / U.g_n_db
U["opp_run_succ"] = U.g_rsucc / U.g_n_run
U["log_exp_pr"] = np.log(U.D_pr * U.Oopp_pr * U.S_ht)
U["log_exp_sk"] = np.log(U.D_sk * U.Oopp_sk)
U["logD_pr"], U["logO_pr"] = np.log(U.D_pr), np.log(U.Oopp_pr)
U["logD_sk"], U["logO_sk"] = np.log(U.D_sk), np.log(U.Oopp_sk)
U["logD_st"] = np.log(U.D_st)
U["log_agg_pr"] = np.log(U.agg_pr.clip(lower=1e-3))
U["log_agg_sk"] = np.log(U.agg_sk.clip(lower=1e-3))


def partial_r(df, x, y, ctrl):
    d = df[[x, y] + ctrl].dropna()
    C = np.column_stack([d[ctrl].to_numpy(float), np.ones(len(d))])
    rx = d[x] - C @ np.linalg.lstsq(C, d[x], rcond=None)[0]
    ry = d[y] - C @ np.linalg.lstsq(C, d[y], rcond=None)[0]
    r = r_(rx, ry)
    n = len(d)
    se = (1 - r * r) / np.sqrt(n - len(ctrl) - 2)
    return {"partial_r": round(r, 4), "z": round(r / se, 2), "n": n}


unit = {}
# next-game: the 1v1 structure (both sides matter) and the POINTS question
unit["next_game_r"] = {
    "pressure_rate~log(D*O*S)": round(r_(U.log_exp_pr, U.pr_rate), 4),
    "pressure_rate~logD_only": round(r_(U.logD_pr, U.pr_rate), 4),
    "pressure_rate~logO_only(opp protection)": round(r_(U.logO_pr, U.pr_rate), 4),
    "sack_rate~log(D*O)": round(r_(U.log_exp_sk, U.sk_rate), 4),
    "sack_rate~logD_only": round(r_(U.logD_sk, U.sk_rate), 4),
    "sack_rate~logO_only": round(r_(U.logO_sk, U.sk_rate), 4),
    "sack_rate~CURRENT_epa_def_allowed": round(r_(U.epa_def_allowed, U.sk_rate), 4),
    "points_allowed~CURRENT_epa_def_allowed": round(r_(U.epa_def_allowed, U.g_pts_allowed), 4),
    "points_allowed~logD_pr": round(r_(U.logD_pr, U.g_pts_allowed), 4),
    "points_allowed~logD_sk": round(r_(U.logD_sk, U.g_pts_allowed), 4),
    "points_allowed~logD_st": round(r_(U.logD_st, U.g_pts_allowed), 4),
    "points_allowed~log_agg_pr(player aggregate)": round(r_(U.log_agg_pr, U.g_pts_allowed), 4),
    "n_team_games": int(len(U)),
}
ctrl = ["epa_def_allowed", "epa_opp_off"]
unit["next_game_partial_beyond_CURRENT_epa"] = {
    "points_allowed|logD_pr": partial_r(U, "logD_pr", "g_pts_allowed", ctrl),
    "points_allowed|logD_sk": partial_r(U, "logD_sk", "g_pts_allowed", ctrl),
    "points_allowed|logD_st": partial_r(U, "logD_st", "g_pts_allowed", ctrl),
    "points_allowed|log_agg_pr": partial_r(U, "log_agg_pr", "g_pts_allowed", ctrl),
    "points_allowed|logO_pr(opp protection)": partial_r(U, "logO_pr", "g_pts_allowed", ctrl),
    "points_allowed|logO_sk(opp protection)": partial_r(U, "logO_sk", "g_pts_allowed", ctrl),
    "sack_rate|log(D*O)": partial_r(U, "log_exp_sk", "sk_rate", ctrl),
    "pressure_rate|log(D*O*S)": partial_r(U, "log_exp_pr", "pr_rate", ctrl),
    "opp_run_success|logD_st": partial_r(U, "logD_st", "opp_run_succ", ctrl),
}
# unit rating vs player aggregate, head to head on the unit's next-game pressure / sacks
unit["unit_vs_player_aggregate"] = {
    "pressure_rate~logD_pr": round(r_(U.logD_pr, U.pr_rate), 4),
    "pressure_rate~log_agg_pr": round(r_(U.log_agg_pr, U.pr_rate), 4),
    "sack_rate~logD_sk": round(r_(U.logD_sk, U.sk_rate), 4),
    "sack_rate~log_agg_sk": round(r_(U.log_agg_sk, U.sk_rate), 4),
    "pressure_rate|agg beyond D": partial_r(U, "log_agg_pr", "pr_rate", ["logD_pr", "logO_pr"]),
    "sack_rate|agg beyond D": partial_r(U, "log_agg_sk", "sk_rate", ["logD_sk", "logO_sk"]),
    "early_season_wk1_4_pressure|agg beyond D": partial_r(U[U.week <= 4], "log_agg_pr", "pr_rate",
                                                          ["logD_pr", "logO_pr"]),
}
# rest-of-season checkpoints (team level): rating at team game 9 -> games 9..end
U = U.sort_values(["defteam", "season", "game_date"])
U["tgi"] = U.groupby(["defteam", "season"]).cumcount() + 1
rows = []
for (t, s), g in U.groupby(["defteam", "season"]):
    f = g[g.tgi == 9]
    if len(f) and (g.tgi >= 9).sum() >= 6:
        rest = g[g.tgi >= 9]
        f = f.iloc[0]
        rows.append({"logD_pr": f.logD_pr, "logD_sk": f.logD_sk, "logD_st": f.logD_st, "log_agg_pr": f.log_agg_pr,
                     "epa_def_allowed": f.epa_def_allowed, "own_prot_sk": np.log(f.own_prot_sk),
                     "fut_pts_allowed": rest.g_pts_allowed.mean(), "fut_sk_rate": rest.g_sacks.sum() / rest.g_n_db.sum(),
                     "fut_pr_rate": rest.g_pressure.sum() / rest.g_n_db.sum()})
RS = pd.DataFrame(rows)
unit["rest_of_season_from_game9"] = {
    "n_team_seasons": int(len(RS)),
    "fut_pts_allowed~CURRENT_epa_def_allowed": round(r_(RS.epa_def_allowed, RS.fut_pts_allowed), 3),
    "fut_pts_allowed~logD_pr": round(r_(RS.logD_pr, RS.fut_pts_allowed), 3),
    "fut_pts_allowed~logD_sk": round(r_(RS.logD_sk, RS.fut_pts_allowed), 3),
    "fut_pts_allowed|logD_pr beyond epa": partial_r(RS, "logD_pr", "fut_pts_allowed", ["epa_def_allowed"]),
    "fut_pts_allowed|logD_sk beyond epa": partial_r(RS, "logD_sk", "fut_pts_allowed", ["epa_def_allowed"]),
    "fut_pts_allowed|logD_st beyond epa": partial_r(RS, "logD_st", "fut_pts_allowed", ["epa_def_allowed"]),
    "fut_sk_rate~logD_sk": round(r_(RS.logD_sk, RS.fut_sk_rate), 3),
    "fut_sk_rate~CURRENT_epa_def_allowed": round(r_(RS.epa_def_allowed, RS.fut_sk_rate), 3),
}
# offensive line (protection) -> the offence's own points and sacks next game, beyond its EPA rating
Off = TG[TG.season.between(DEV_LO, DEV_HI)].copy()
Off["logO_sk"], Off["logO_pr"] = np.log(Off.Oopp_sk), np.log(Off.Oopp_pr)
Off["points_scored"] = [np.nan] * len(Off)
pts = TG.set_index(["game_id", "posteam"]).g_pts_allowed.to_dict()     # points scored by posteam = allowed by defteam
Off["points_scored"] = [pts.get((g, o)) for g, o in zip(Off.game_id, Off.posteam)]
Off["sk_allowed_rate"] = Off.g_sacks / Off.g_n_db
unit["offensive_line_protection"] = {
    "sacks_allowed_rate~logO_sk": round(r_(Off.logO_sk, Off.sk_allowed_rate), 4),
    "sacks_allowed_rate|logO_sk beyond CURRENT off EPA": partial_r(Off, "logO_sk", "sk_allowed_rate",
                                                                   ["epa_opp_off", "epa_def_allowed"]),
    "points_scored~logO_sk": round(r_(Off.logO_sk, Off.points_scored), 4),
    "points_scored|logO_sk beyond CURRENT off EPA": partial_r(Off, "logO_sk", "points_scored",
                                                               ["epa_opp_off", "epa_def_allowed"]),
    "points_scored|logO_pr beyond CURRENT off EPA": partial_r(Off, "logO_pr", "points_scored",
                                                               ["epa_opp_off", "epa_def_allowed"]),
    "note": "O factor = the offence's pressure/sacks ALLOWED (opp+scorer adjusted); it is QB+OL together",
}
OUT["2b_unit_predictive_validity"] = unit
print("UNIT:", json.dumps(unit, indent=1)[:4000])

# ================================================================ 3. FACE VALIDITY
NAMES = dict(zip(PG.player_id, PG.name))
face = {}
for s in (2009, 2012, 2015):
    x = PR[(PR.season == s)].sort_values("game_date").groupby("player_id").tail(1)
    edge = x[x.bucket.isin(EDGE + ["DT"]) & (x.neff_pr >= 80)]
    reg = edge[edge.n_games >= 20]
    run = x[x.bucket.isin(FRONT) & (x.neff_pr >= 80) & (x.n_games >= 15)]
    f = lambda d, c: [f"{NAMES.get(p, p)} ({b},{t}) {v:.3f}" for p, b, t, v in zip(d.player_id, d.bucket, d.team, d[c])]  # noqa
    face[str(s)] = {
        "pass_rush_top12": f(edge.sort_values("th_pr", ascending=False).head(12), "th_pr"),
        "pass_rush_bottom8_regulars(n_games>=20)": f(reg.sort_values("th_pr").head(8), "th_pr"),
        "run_stop_top10": f(run.sort_values("th_st", ascending=False).head(10), "th_st"),
        "run_stop_bottom6_regulars": f(run.sort_values("th_st").head(6), "th_st"),
    }
    y = TG[TG.season == s].sort_values("game_date").groupby("posteam").tail(1)
    face[str(s)]["protection_best6(O_sk low)"] = [f"{t} {v:.2f}" for t, v in
                                                   y.sort_values("Oopp_sk")[["posteam", "Oopp_sk"]].head(6).to_numpy()]
    face[str(s)]["protection_worst6"] = [f"{t} {v:.2f}" for t, v in
                                         y.sort_values("Oopp_sk", ascending=False)[["posteam", "Oopp_sk"]].head(6).to_numpy()]
OUT["3_face_validity"] = face
for s, v in face.items():
    print(s, "TOP", v["pass_rush_top12"][:6], "\n   BOTTOM", v["pass_rush_bottom8_regulars(n_games>=20)"][:4],
          "\n   RUN", v["run_stop_top10"][:5])

json.dump(OUT, open(os.path.join(DATA, "pv_nfl_pass_rush_and_front_validation.json"), "w"), indent=1, default=float)
print("wrote data/pv_nfl_pass_rush_and_front_validation.json")
