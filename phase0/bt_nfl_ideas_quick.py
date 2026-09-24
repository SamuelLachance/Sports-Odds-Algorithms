"""Breakthrough program (NFL, ideas leg) - single-shot DEV indications. DEV ONLY.

NOT the screen. Each candidate below has ONE pre-declared construction (no knob
search) and is added to the shipped 14-feature matrix inside the shipped
walk-forward protocol (expanding refit, recency HL=3, C=100, ties dropped, DEV
scored 2006-2015). New columns are z-scored with TRAIN-FOLD statistics only.
The numbers here are indications for ranking ideas; the pre-registered screen
(bar +0.00150, CI excluding 0, harness repro, leak audit, out-of-search window)
is a separate, later step.

Every feature is an as-of walk over the 1999-2015 spine: the value for game i
uses only games/plays/injury reports dated before game i (injury reports are
the pre-game Friday report for game i's week). No odds are read. No season >=
2016 row exists in the cache, is built, fitted or scored.

Output: data/bt_nfl_ideas_quick.json, data/bt_nfl_ideas_feats.npy (+ names json)
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260924)

D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
QH, QA = np.array(D["qh"]), np.array(D["qa"])
TSE = np.array(D["tse_old"])
N = len(G)
SEAS = np.array([g["season"] for g in G])
assert SEAS.max() < TEST_ERA
WK = np.array([g["week"] for g in G])
TYP = np.array([g["type"] for g in G])
Y = np.array([g["y"] for g in G], float)
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}

RAW = {}
for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")):
    if r["season"] and int(r["season"]) < TEST_ERA:
        RAW[r["game_id"]] = r
HS = np.array([float(RAW[g["gid"]]["home_score"]) for g in G])
AS = np.array([float(RAW[g["gid"]]["away_score"]) for g in G])


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------------------------------------------------------- plays ----
gp = defaultdict(lambda: [0.0, 0, 0.0, 0])   # (gid, posteam) -> pass sum,n, run sum,n
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: k for k, c in enumerate(hdr)}
    for r in rd:
        if int(r[ix["game_id"]][:4]) >= TEST_ERA:
            continue
        t = FR.get(r[ix["posteam"]], r[ix["posteam"]])
        a = gp[(r[ix["game_id"]], t)]
        e = float(r[ix["epa"]])
        if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
            a[0] += e; a[1] += 1
        else:
            a[2] += e; a[3] += 1
ga = defaultdict(lambda: [0.0, 0])            # all plays (the shipped epa_net source)
with open("data/nfl_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: k for k, c in enumerate(hdr)}
    for r in rd:
        if int(r[ix["game_id"]][:4]) >= TEST_ERA:
            continue
        a = ga[(r[ix["game_id"]], FR.get(r[ix["posteam"]], r[ix["posteam"]]))]
        a[0] += float(r[ix["epa"]]); a[1] += 1

FEATS = {}

# ======================================= C1 opponent-adjusted EPA channels ===
DEC, PN, SDEC = 0.8813581205848026, 754.0438610349224, 0.7130881070009863
LG_ALL = -0.02334
_ps = sum(v[0] for k, v in gp.items() if 1999 <= int(k[0][:4]) <= 2015)
_pn = sum(v[1] for k, v in gp.items() if 1999 <= int(k[0][:4]) <= 2015)
_rs = sum(v[2] for k, v in gp.items() if 1999 <= int(k[0][:4]) <= 2015)
_rn = sum(v[3] for k, v in gp.items() if 1999 <= int(k[0][:4]) <= 2015)
LGP, LGR = _ps / _pn, _rs / _rn


def opp_adjusted(src, lgv, prior, adjust):
    """shipped EWMA dynamics; if adjust, each observation is corrected by the
    opponent's PRE-game (adjusted) rating. Returns home-minus-away net rating."""
    off = defaultdict(lambda: [0.0, 0.0]); dfn = defaultdict(lambda: [0.0, 0.0])
    rate = lambda st: (st[0] + prior * lgv) / (st[1] + prior)  # noqa: E731
    out = np.zeros(N); prev = None
    for i, g in enumerate(G):
        if prev is not None and g["season"] != prev:
            for dct in (off, dfn):
                for st in dct.values():
                    st[0] *= (1 - SDEC); st[1] *= (1 - SDEC)
        prev = g["season"]
        h, a = g["home"], g["away"]
        out[i] = (rate(off[h]) - rate(dfn[h])) - (rate(off[a]) - rate(dfn[a]))
        upd = []
        for t_off, t_def in ((h, a), (a, h)):
            e = src(g["gid"], t_off)
            if e is None or e[1] == 0:
                continue
            s_, n_ = e
            so = s_ - n_ * (rate(dfn[t_def]) - lgv) if adjust else s_
            sd = s_ - n_ * (rate(off[t_off]) - lgv) if adjust else s_
            upd.append((t_off, t_def, so, sd, n_))
        for t_off, t_def, so, sd, n_ in upd:
            o = off[t_off]; o[0] = DEC * o[0] + so; o[1] = DEC * o[1] + n_
            d_ = dfn[t_def]; d_[0] = DEC * d_[0] + sd; d_[1] = DEC * d_[1] + n_
    return out


src_all = lambda gid, t: ga.get((gid, t))  # noqa: E731
src_pass = lambda gid, t: (gp[(gid, t)][0], gp[(gid, t)][1]) if (gid, t) in gp else None  # noqa: E731
src_run = lambda gid, t: (gp[(gid, t)][2], gp[(gid, t)][3]) if (gid, t) in gp else None  # noqa: E731
raw_all = opp_adjusted(src_all, LG_ALL, PN, False)
adj_all = opp_adjusted(src_all, LG_ALL, PN, True)
raw_pass = opp_adjusted(src_pass, LGP, PN / 2, False)
adj_pass = opp_adjusted(src_pass, LGP, PN / 2, True)
raw_run = opp_adjusted(src_run, LGR, PN / 2, False)
adj_run = opp_adjusted(src_run, LGR, PN / 2, True)
dev_rows = (SEAS >= DEV_LO)
repro = {"epa_net": float(np.corrcoef(raw_all[dev_rows], X14[dev_rows, 2])[0, 1]),
         "pass_net": float(np.corrcoef(raw_pass[dev_rows], X14[dev_rows, 12])[0, 1]),
         "run_net": float(np.corrcoef(raw_run[dev_rows], X14[dev_rows, 13])[0, 1])}
print("raw-walk reproduction corr vs shipped cols:", {k: round(v, 5) for k, v in repro.items()})
FEATS["oppadj_incr_net"] = adj_all - raw_all
FEATS["oppadj_incr_pass"] = adj_pass - raw_pass
FEATS["oppadj_incr_run"] = adj_run - raw_run
FEATS["oppadj_net"] = adj_all
FEATS["oppadj_pass"] = adj_pass
FEATS["oppadj_run"] = adj_run

# ============================================= C2 playoff leverage (MC) =====
DIV = {"AFC East": "BUF MIA NE NYJ", "AFC North": "BAL CIN CLE PIT",
       "AFC South": "HOU IND JAX TEN", "AFC West": "DEN KC LV LAC",
       "NFC East": "DAL NYG PHI WAS", "NFC North": "CHI DET GB MIN",
       "NFC South": "ATL CAR NO TB", "NFC West": "ARI LA SF SEA"}
TDIV = {t: d for d, ts in DIV.items() for t in ts.split()}
TEAMS = sorted(TDIV); TIX = {t: k for k, t in enumerate(TEAMS)}
conf_of = np.array([TDIV[t][:3] for t in TEAMS]); div_of = np.array([TDIV[t] for t in TEAMS])
K_, HFA_, REG_ = 47.43351932834238, 52.14162888646703, 0.3647684062154459
elo = defaultdict(lambda: 1500.0); POST = {}   # team -> rating after its latest game
ELO_PRE = np.zeros((N, 2)); prev = None
for i, g in enumerate(G):
    if prev is not None and g["season"] != prev:
        for t in list(elo):
            elo[t] = 1500.0 + (1 - REG_) * (elo[t] - 1500.0)
    prev = g["season"]
    h, a = g["home"], g["away"]
    ELO_PRE[i] = (elo[h], elo[a])
    dr = elo[h] - elo[a] + (0.0 if g["neutral"] else HFA_)
    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
    elo[h] += K_ * (yy - e); elo[a] -= K_ * (yy - e)
ELO_POST = np.zeros((N, 2))
for i, g in enumerate(G):
    dr = ELO_PRE[i, 0] - ELO_PRE[i, 1] + (0.0 if g["neutral"] else HFA_)
    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
    ELO_POST[i] = (ELO_PRE[i, 0] + K_ * (yy - e), ELO_PRE[i, 1] - K_ * (yy - e))

NS = 4000
LEV_PO = np.zeros(N); LEV_BYE = np.zeros(N)
divs = sorted(set(div_of))
for s in range(2002, DEV_HI + 1):
    reg = [i for i in range(N) if SEAS[i] == s and TYP[i] == "REG"]
    for W in range(11, 18):
        wk_games = [i for i in reg if WK[i] == W]
        if not wk_games:
            continue
        done = [i for i in reg if WK[i] < W]
        rem = [i for i in reg if WK[i] >= W]
        base_w = np.zeros(len(TEAMS)); rat = {}
        for i in done:
            hh, aa = TIX[G[i]["home"]], TIX[G[i]["away"]]
            yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
            base_w[hh] += yy; base_w[aa] += 1 - yy
            rat[G[i]["home"]] = ELO_POST[i, 0]; rat[G[i]["away"]] = ELO_POST[i, 1]
        for t in TEAMS:
            rat.setdefault(t, 1500.0)
        hi = np.array([TIX[G[i]["home"]] for i in rem]); ai = np.array([TIX[G[i]["away"]] for i in rem])
        dr = np.array([rat[G[i]["home"]] - rat[G[i]["away"]] + (0.0 if G[i]["neutral"] else HFA_)
                       for i in rem])
        pe = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        hw = (RNG.random((NS, len(rem))) < pe).astype(float)
        wins = np.tile(base_w, (NS, 1))
        np.add.at(wins.T, hi, hw.T); np.add.at(wins.T, ai, (1 - hw).T)
        noisy = wins + RNG.random(wins.shape) * 1e-3
        po = np.zeros_like(wins, bool); bye = np.zeros_like(wins, bool)
        for cf in ("AFC", "NFC"):
            cidx = np.where(conf_of == cf)[0]
            dwin = np.stack([np.where(div_of == dv)[0][np.argmax(noisy[:, div_of == dv], axis=1)]
                             for dv in divs if dv.startswith(cf)], 1)
            isdw = np.zeros((NS, len(TEAMS)), bool)
            np.put_along_axis(isdw, dwin, True, axis=1)
            wc_score = np.where(isdw[:, cidx], -1e9, noisy[:, cidx])
            wc = cidx[np.argsort(-wc_score, axis=1)[:, :2]]
            pc_ = isdw.copy(); np.put_along_axis(pc_, wc, True, axis=1)
            po |= pc_
            top2 = np.take_along_axis(dwin, np.argsort(-np.take_along_axis(noisy, dwin, 1), 1)[:, :2], 1)
            np.put_along_axis(bye, top2, True, axis=1)
        col = {i: k for k, i in enumerate(rem)}
        for i in wk_games:
            c = col[i]; hh, aa = TIX[G[i]["home"]], TIX[G[i]["away"]]
            won = hw[:, c] == 1.0
            if won.sum() < 30 or (~won).sum() < 30:
                continue
            LEV_PO[i] = (po[won, hh].mean() - po[~won, hh].mean()) - (po[~won, aa].mean() - po[won, aa].mean())
            LEV_BYE[i] = (bye[won, hh].mean() - bye[~won, hh].mean()) - (bye[~won, aa].mean() - bye[won, aa].mean())
FEATS["lev_po"] = LEV_PO
FEATS["lev_bye"] = LEV_BYE

# ===================================== C3 informed preseason prior (fade) ===
teamseas = defaultdict(lambda: {"w": 0.0, "g": 0, "pf": 0.0, "pa": 0.0, "qbs": Counter()})
seas_epa = defaultdict(lambda: [0.0, 0, 0.0, 0])
for i, g in enumerate(G):
    if g["type"] != "REG":
        continue
    s = g["season"]
    for team, opp, pf, pa, qid in ((g["home"], g["away"], HS[i], AS[i], g["home_qb"]),
                                   (g["away"], g["home"], AS[i], HS[i], g["away_qb"])):
        t = teamseas[(team, s)]
        t["w"] += 1.0 if pf > pa else 0.5 if pf == pa else 0.0
        t["g"] += 1; t["pf"] += pf; t["pa"] += pa; t["qbs"][qid] += 1
        e = ga.get((g["gid"], team))
        if e:
            so = seas_epa[(team, s)]; so[0] += e[0]; so[1] += e[1]
            sd = seas_epa[(opp, s)]; sd[2] += e[0]; sd[3] += e[1]
gplayed = defaultdict(int)
PRE_NET = np.zeros(N); PRE_PYL = np.zeros(N); PRE_QBN = np.zeros(N); PRE_PDG = np.zeros(N)
FADE = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25}
for i, g in enumerate(G):
    s = g["season"]
    vals = []
    for team, qid in ((g["home"], g["home_qb"]), (g["away"], g["away_qb"])):
        gpl = gplayed[(team, s)]
        w = FADE.get(gpl + 1, 0.0) if g["type"] == "REG" else 0.0
        t = teamseas.get((team, s - 1)); e = seas_epa.get((team, s - 1))
        if not t or not e or not e[1] or not e[3]:
            vals.append((0.0, 0.0, 0.0, 0.0)); continue
        net = e[0] / e[1] - e[2] / e[3]
        wp = t["w"] / t["g"]
        pyth = t["pf"] ** 2.37 / (t["pf"] ** 2.37 + t["pa"] ** 2.37)
        top = t["qbs"].most_common(1)[0][0]
        vals.append((w * net, w * (wp - pyth), w * (0.0 if qid == top else 1.0),
                     w * (t["pf"] - t["pa"]) / t["g"]))
    PRE_NET[i] = vals[0][0] - vals[1][0]; PRE_PYL[i] = vals[0][1] - vals[1][1]
    PRE_QBN[i] = vals[0][2] - vals[1][2]; PRE_PDG[i] = vals[0][3] - vals[1][3]
    for team in (g["home"], g["away"]):
        gplayed[(team, s)] += 1
FEATS["pre_net"] = PRE_NET; FEATS["pre_pyth_luck"] = PRE_PYL
FEATS["pre_qb_new"] = PRE_QBN; FEATS["pre_pdg"] = PRE_PDG

# ================================== C4 QB shock at QB-change games ==========
ew = defaultdict(lambda: [0.0, 0.0]); prev_qb = {}
SHOCK_CH = np.zeros(N)
for i, g in enumerate(G):
    for side, team, qid, q in ((1, g["home"], g["home_qb"], QH[i]), (-1, g["away"], g["away_qb"], QA[i])):
        st = ew[team]
        if team in prev_qb and prev_qb[team] != qid and st[1] > 0:
            SHOCK_CH[i] += side * (q - st[0] / st[1])
    for team, qid, q in ((g["home"], g["home_qb"], QH[i]), (g["away"], g["away_qb"], QA[i])):
        prev_qb[team] = qid
        st = ew[team]; st[0] = 0.8 * st[0] + q; st[1] = 0.8 * st[1] + 1.0
FEATS["qb_shock_change"] = SHOCK_CH

names = list(FEATS)
np.save("data/bt_nfl_ideas_feats.npy", np.column_stack([FEATS[k] for k in names]))
json.dump({"names": names, "note": "as-of candidate features, pre-2016 spine rows aligned "
           "with data/bt_nfl_ideas_dev_X.npy"}, open("data/bt_nfl_ideas_feats_names.json", "w"))


# ================================================== walk-forward harness ====
def wf(extra=None, replace=None):
    X = X14.copy()
    if replace:
        for c, v in replace.items():
            X[:, c] = v
    per, vec = {}, []
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[tr].max() < s_ and SEAS[te].max() < TEST_ERA
        Xs = X
        if extra is not None:
            E = np.column_stack(extra) if isinstance(extra, list) else extra.reshape(-1, 1)
            mu, sd = E[tr].mean(0), E[tr].std(0)
            sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec)


base_ll, base_per, base_vec = wf()
print(f"baseline DEV wf LL {base_ll:.5f} (recorded 0.62292)")
assert abs(base_ll - 0.62292) < 5e-4


def report(nm, res):
    ll, per, vec = res
    d = base_vec - vec
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if base_per[s] - per[s] > 0)
    print(f"  {nm:<40} {ll:.5f} gain {base_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10")
    return {"ll": round(ll, 6), "gain": round(base_ll - ll, 6), "ci": [round(lo, 6), round(hi, 6)],
            "seasons_pos": pos, "per_season_gain": {s: round(base_per[s] - per[s], 5) for s in per}}


OUT = {"baseline_dev_ll": base_ll, "repro_corr": repro, "results": {}}
Rz = OUT["results"]
print("\nC1 opponent adjustment")
Rz["C1a add oppadj increment (net)"] = report("C1a +incr net", wf(FEATS["oppadj_incr_net"]))
Rz["C1b add oppadj increments (net,pass,run)"] = report(
    "C1b +incr net/pass/run", wf([FEATS["oppadj_incr_net"], FEATS["oppadj_incr_pass"], FEATS["oppadj_incr_run"]]))
Rz["C1c replace cols 2/12/13 with adjusted"] = report(
    "C1c replace 2/12/13", wf(replace={2: adj_all, 12: adj_pass, 13: adj_run}))
Rz["C1d add old team-TS edge (redundancy probe)"] = report("C1d +old tse", wf(TSE))
Rz["C1e C1b + old tse"] = report(
    "C1e C1b + old tse", wf([FEATS["oppadj_incr_net"], FEATS["oppadj_incr_pass"], FEATS["oppadj_incr_run"], TSE]))
print("\nC2 playoff leverage")
Rz["C2a lev_po"] = report("C2a lev_po", wf(FEATS["lev_po"]))
Rz["C2b lev_po + lev_bye"] = report("C2b lev_po+lev_bye", wf([FEATS["lev_po"], FEATS["lev_bye"]]))
print("\nC3 preseason prior (fades over first 4 games)")
Rz["C3a pre_net"] = report("C3a pre_net", wf(FEATS["pre_net"]))
Rz["C3b pre_net+pyth_luck+qb_new"] = report(
    "C3b pre_net+pyl+qbn", wf([FEATS["pre_net"], FEATS["pre_pyth_luck"], FEATS["pre_qb_new"]]))
Rz["C3c pre_pdg"] = report("C3c pre_pdg", wf(FEATS["pre_pdg"]))
print("\nC4 QB shock at change games")
Rz["C4 qb_shock_change"] = report("C4 qb_shock_change", wf(FEATS["qb_shock_change"]))
print("\nALL (C1b + C2a + C3b + C4)")
Rz["ALL"] = report("ALL", wf([FEATS["oppadj_incr_net"], FEATS["oppadj_incr_pass"], FEATS["oppadj_incr_run"],
                             FEATS["lev_po"], FEATS["pre_net"], FEATS["pre_pyth_luck"],
                             FEATS["pre_qb_new"], FEATS["qb_shock_change"]]))
json.dump(OUT, open("data/bt_nfl_ideas_quick.json", "w"), indent=1)
print("wrote data/bt_nfl_ideas_quick.json")
