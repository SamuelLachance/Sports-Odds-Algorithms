"""Breakthrough program (NFL, ideas leg) - single-shot DEV indications, round 2. DEV ONLY.

Same harness/discipline as bt_nfl_ideas_quick.py (shipped walk-forward, DEV
2006-2015 scored, new columns z-scored on the train fold, ONE pre-declared
construction per idea, no knob search). Indications only - not the screen.

  V  volatile-play discount (FIP analog): the three team EPA channels rebuilt
     from per-play EPA winsorised at +-3 (V1) or floored at -3 only (V2,
     turnover-tail only); league anchors recomputed on the clipped scale.
     REPLACES cols 2/12/13.
  L  offence/defence de-netting: epa_net (L1) or all three channels (L2) split
     into an offence diff and a defence diff so the blend can weight the less
     stable defensive EPA separately. REPLACES the net columns.
  Q  opponent-adjusted QB rating: exact increment of the shipped QbElo state
     when each dropback's EPA is corrected by the opponent's as-of pass-defence
     EPA allowed (raw pass channel walk). ADDS one column.
  S  late-season stakes: MC from as-of Elo (weeks 15-17 only): per side
     'locked' (P(PO)>=.999 and bye leverage <.02) and 'eliminated'
     (P(PO)<=.001). S1 adds nostake_away - nostake_home; S2 adds locked and
     eliminated diffs separately.
  P  returning production at season start: share of the team's previous-season
     non-QB offensive usage (targets+carries) and defensive tackles+2*sacks held
     by players who have appeared for the team this season up to and including
     this game (gameday-actives convention), fading over the first 4 games.

No odds read. No season >= 2016 row exists in the cache. Output:
data/bt_nfl_ideas_quick2.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260925)

D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
N = len(G)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
WK = np.array([g["week"] for g in G]); TYP = np.array([g["type"] for g in G])
Y = np.array([g["y"] for g in G], float)
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
RAW = {r["game_id"]: r for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8"))
       if r["season"] and int(r["season"]) < TEST_ERA}
HS = np.array([float(RAW[g["gid"]]["home_score"]) for g in G])
AS = np.array([float(RAW[g["gid"]]["away_score"]) for g in G])
DEC, PN, SDEC = 0.8813581205848026, 754.0438610349224, 0.7130881070009863


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------ per-game EPA aggregates ----
def agg_plays(clipf):
    ga = defaultdict(lambda: [0.0, 0])
    with open("data/nfl_plays.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh); hdr = next(rd); ix = {c: k for k, c in enumerate(hdr)}
        for r in rd:
            if int(r[ix["game_id"]][:4]) >= TEST_ERA:
                continue
            a = ga[(r[ix["game_id"]], FR.get(r[ix["posteam"]], r[ix["posteam"]]))]
            a[0] += clipf(float(r[ix["epa"]])); a[1] += 1
    gp = defaultdict(lambda: [0.0, 0, 0.0, 0])
    with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh); hdr = next(rd); ix = {c: k for k, c in enumerate(hdr)}
        for r in rd:
            if int(r[ix["game_id"]][:4]) >= TEST_ERA:
                continue
            a = gp[(r[ix["game_id"]], FR.get(r[ix["posteam"]], r[ix["posteam"]]))]
            e = clipf(float(r[ix["epa"]]))
            if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
                a[0] += e; a[1] += 1
            else:
                a[2] += e; a[3] += 1
    return ga, gp


def lg_of(ga, gp):
    yrs = lambda k: 1999 <= int(k[0][:4]) <= 2015  # noqa: E731
    la = sum(v[0] for k, v in ga.items() if yrs(k)) / sum(v[1] for k, v in ga.items() if yrs(k))
    lp = sum(v[0] for k, v in gp.items() if yrs(k)) / sum(v[1] for k, v in gp.items() if yrs(k))
    lr = sum(v[2] for k, v in gp.items() if yrs(k)) / sum(v[3] for k, v in gp.items() if yrs(k))
    return la, lp, lr


def walk(src, lgv, prior, split=False):
    """shipped EWMA team-channel dynamics. returns net diff, or (off diff, def diff),
    plus per-game (home def rate, away def rate) for downstream opponent adjustment."""
    off = defaultdict(lambda: [0.0, 0.0]); dfn = defaultdict(lambda: [0.0, 0.0])
    rate = lambda st: (st[0] + prior * lgv) / (st[1] + prior)  # noqa: E731
    o_d = np.zeros(N); d_d = np.zeros(N); defr = np.zeros((N, 2)); prev = None
    for i, g in enumerate(G):
        if prev is not None and g["season"] != prev:
            for dct in (off, dfn):
                for st in dct.values():
                    st[0] *= (1 - SDEC); st[1] *= (1 - SDEC)
        prev = g["season"]
        h, a = g["home"], g["away"]
        o_d[i] = rate(off[h]) - rate(off[a]); d_d[i] = rate(dfn[h]) - rate(dfn[a])
        defr[i] = (rate(dfn[h]), rate(dfn[a]))
        for t_off, t_def in ((h, a), (a, h)):
            e = src(g["gid"], t_off)
            if e is None or e[1] == 0:
                continue
            o = off[t_off]; o[0] = DEC * o[0] + e[0]; o[1] = DEC * o[1] + e[1]
            d_ = dfn[t_def]; d_[0] = DEC * d_[0] + e[0]; d_[1] = DEC * d_[1] + e[1]
    return (o_d - d_d) if not split else (o_d, d_d), defr


def channels(ga, gp, split=False):
    la, lp, lr = lg_of(ga, gp)
    s_all = lambda gid, t: ga.get((gid, t))  # noqa: E731
    s_p = lambda gid, t: (gp[(gid, t)][0], gp[(gid, t)][1]) if (gid, t) in gp else None  # noqa: E731
    s_r = lambda gid, t: (gp[(gid, t)][2], gp[(gid, t)][3]) if (gid, t) in gp else None  # noqa: E731
    a, _ = walk(s_all, la, PN, split)
    p, defr_p = walk(s_p, lp, PN / 2, split)
    r, _ = walk(s_r, lr, PN / 2, split)
    return a, p, r, defr_p, lp


ga0, gp0 = agg_plays(lambda e: e)
n_all, n_pass, n_run, DEFR_P, LGP = channels(ga0, gp0)
dv = SEAS >= DEV_LO
repro = {c: float(np.corrcoef(v[dv], X14[dv, k])[0, 1]) for c, v, k in
         (("epa_net", n_all, 2), ("pass_net", n_pass, 12), ("run_net", n_run, 13))}
print("repro corr:", {k: round(v, 6) for k, v in repro.items()})
assert min(repro.values()) > 0.9999

FE = {}
# ---- V volatile-play discount
for tag, cf in (("V1", lambda e: max(-3.0, min(3.0, e))), ("V2", lambda e: max(-3.0, e))):
    ga_, gp_ = agg_plays(cf)
    a_, p_, r_, _, _ = channels(ga_, gp_)
    FE[tag] = (a_, p_, r_)
# ---- L off/def de-netting (raw)
(ao, ad), (po, pd), (ro, rd_), _, _ = channels(ga0, gp0, split=True)
FE["L"] = (ao, ad, po, pd, ro, rd_)

# ---- Q opponent-adjusted QB increment (exact algebra on the shipped QbElo state)
import nfl_qb_elo as QE  # noqa: E402
qbw = QE.load_qb_weeks()
SP = {"decay": 0.8010595244395629, "prior_db": 650.2103644039794, "season_decay": 0.3915362183690712}
Qdb = defaultdict(float); Qadj = defaultdict(float)
QINC = np.zeros(N); prev = None
for i, g in enumerate(G):
    if prev is not None and g["season"] != prev:
        for q in list(Qdb):
            Qdb[q] *= (1 - SP["season_decay"]); Qadj[q] *= (1 - SP["season_decay"])
    prev = g["season"]
    inc = []
    for qid in (g["home_qb"], g["away_qb"]):
        inc.append(-Qadj[qid] / (Qdb[qid] + SP["prior_db"]))
    QINC[i] = inc[0] - inc[1]
    # update: home QB faced the AWAY pass defence (DEFR_P[i,1]), and vice versa
    for qid, opp_def in ((g["home_qb"], DEFR_P[i, 1]), (g["away_qb"], DEFR_P[i, 0])):
        line = qbw.get((qid, g["season"], g["week"]))
        if line is None:
            continue
        Qdb[qid] = SP["decay"] * Qdb[qid] + line[1]
        Qadj[qid] = SP["decay"] * Qadj[qid] + line[1] * (opp_def - LGP)
FE["Q"] = QINC

# ---- S late-season stakes (MC from as-of Elo; weeks 15-17 REG)
DIV = {"AFC East": "BUF MIA NE NYJ", "AFC North": "BAL CIN CLE PIT",
       "AFC South": "HOU IND JAX TEN", "AFC West": "DEN KC LV LAC",
       "NFC East": "DAL NYG PHI WAS", "NFC North": "CHI DET GB MIN",
       "NFC South": "ATL CAR NO TB", "NFC West": "ARI LA SF SEA"}
TDIV = {t: d for d, ts in DIV.items() for t in ts.split()}
TEAMS = sorted(TDIV); TIX = {t: k for k, t in enumerate(TEAMS)}
conf_of = np.array([TDIV[t][:3] for t in TEAMS]); div_of = np.array([TDIV[t] for t in TEAMS])
divs = sorted(set(div_of))
K_, HFA_, REG_ = 47.43351932834238, 52.14162888646703, 0.3647684062154459
elo = defaultdict(lambda: 1500.0); ELO_POST = np.zeros((N, 2)); prev = None
for i, g in enumerate(G):
    if prev is not None and g["season"] != prev:
        for t in list(elo):
            elo[t] = 1500.0 + (1 - REG_) * (elo[t] - 1500.0)
    prev = g["season"]
    h, a = g["home"], g["away"]
    dr = elo[h] - elo[a] + (0.0 if g["neutral"] else HFA_)
    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
    elo[h] += K_ * (yy - e); elo[a] -= K_ * (yy - e)
    ELO_POST[i] = (elo[h], elo[a])
NS = 4000
LOCK = np.zeros((N, 2)); ELIM = np.zeros((N, 2))
for s in range(2002, DEV_HI + 1):
    reg = [i for i in range(N) if SEAS[i] == s and TYP[i] == "REG"]
    for W in (15, 16, 17):
        wk_games = [i for i in reg if WK[i] == W]
        if not wk_games:
            continue
        done = [i for i in reg if WK[i] < W]; rem = [i for i in reg if WK[i] >= W]
        base_w = np.zeros(len(TEAMS)); rat = {}
        for i in done:
            hh, aa = TIX[G[i]["home"]], TIX[G[i]["away"]]
            yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
            base_w[hh] += yy; base_w[aa] += 1 - yy
            rat[G[i]["home"]] = ELO_POST[i, 0]; rat[G[i]["away"]] = ELO_POST[i, 1]
        hi = np.array([TIX[G[i]["home"]] for i in rem]); ai = np.array([TIX[G[i]["away"]] for i in rem])
        dr = np.array([rat.get(G[i]["home"], 1500.0) - rat.get(G[i]["away"], 1500.0)
                       + (0.0 if G[i]["neutral"] else HFA_) for i in rem])
        hw = (RNG.random((NS, len(rem))) < 1.0 / (1.0 + 10 ** (-dr / 400.0))).astype(float)
        wins = np.tile(base_w, (NS, 1))
        np.add.at(wins.T, hi, hw.T); np.add.at(wins.T, ai, (1 - hw).T)
        noisy = wins + RNG.random(wins.shape) * 1e-3
        po = np.zeros_like(wins, bool); bye = np.zeros_like(wins, bool)
        for cf in ("AFC", "NFC"):
            cidx = np.where(conf_of == cf)[0]
            dwin = np.stack([np.where(div_of == d_)[0][np.argmax(noisy[:, div_of == d_], axis=1)]
                             for d_ in divs if d_.startswith(cf)], 1)
            isdw = np.zeros((NS, len(TEAMS)), bool); np.put_along_axis(isdw, dwin, True, axis=1)
            wc = cidx[np.argsort(-np.where(isdw[:, cidx], -1e9, noisy[:, cidx]), axis=1)[:, :2]]
            pc_ = isdw.copy(); np.put_along_axis(pc_, wc, True, axis=1); po |= pc_
            top2 = np.take_along_axis(dwin, np.argsort(-np.take_along_axis(noisy, dwin, 1), 1)[:, :2], 1)
            np.put_along_axis(bye, top2, True, axis=1)
        col = {i: k for k, i in enumerate(rem)}
        for i in wk_games:
            c = col[i]
            won = hw[:, c] == 1.0
            for k_, t in enumerate((G[i]["home"], G[i]["away"])):
                ti = TIX[t]
                w_side = won if k_ == 0 else ~won
                ppo = po[:, ti].mean()
                lb = abs(bye[w_side, ti].mean() - bye[~w_side, ti].mean()) if 30 < w_side.sum() < NS - 30 else 0.0
                LOCK[i, k_] = 1.0 if (ppo >= 0.999 and lb < 0.02) else 0.0
                ELIM[i, k_] = 1.0 if ppo <= 0.001 else 0.0
NOSTAKE = np.clip(LOCK + ELIM, 0, 1)
FE["S1"] = NOSTAKE[:, 1] - NOSTAKE[:, 0]
FE["S2"] = (LOCK[:, 1] - LOCK[:, 0], ELIM[:, 1] - ELIM[:, 0])
late = (WK >= 15) & (TYP == "REG") & (SEAS >= DEV_LO)
print(f"stakes: DEV wk15-17 games {late.sum()}, side-locked {LOCK[late].sum():.0f}, "
      f"side-eliminated {ELIM[late].sum():.0f}")

# ---- P returning production at season start
use = defaultdict(lambda: defaultdict(float))       # (team, season) -> player -> usage
appear = defaultdict(set)                            # (team, season, week) -> players with a line
with open("data/nfl_player_stats.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        s = int(r["season"])
        if s >= TEST_ERA or r["season_type"] != "REG" or r["position"] == "QB":
            continue
        t = FR.get(r["recent_team"], r["recent_team"])
        v = float(r["targets"] or 0) + float(r["carries"] or 0)
        use[(t, s)][("o", r["player_id"])] += v
        appear[(t, s, int(r["week"]))].add(("o", r["player_id"]))
with open("data/nfl_player_stats_def.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        s = int(r["season"])
        if s >= TEST_ERA or r["season_type"] != "REG":
            continue
        t = FR.get(r["team"], r["team"])
        v = float(r["def_tackles_solo"] or 0) + float(r["def_tackle_assists"] or 0) + 2 * float(r["def_sacks"] or 0)
        use[(t, s)][("d", r["player_id"])] += v
        appear[(t, s, int(r["week"]))].add(("d", r["player_id"]))
seen = defaultdict(set); gpl = defaultdict(int)
FADE = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25}
RET = np.zeros(N)
for i, g in enumerate(G):
    s = g["season"]; vals = []
    for team in (g["home"], g["away"]):
        if g["type"] == "REG":
            seen[(team, s)] |= appear.get((team, s, g["week"]), set())
        w = FADE.get(gpl[(team, s)] + 1, 0.0) if g["type"] == "REG" else 0.0
        u = use.get((team, s - 1))
        if not u or w == 0.0:
            vals.append(0.0); continue
        tot_o = sum(v for k, v in u.items() if k[0] == "o"); tot_d = sum(v for k, v in u.items() if k[0] == "d")
        ro = sum(v for k, v in u.items() if k[0] == "o" and k in seen[(team, s)]) / max(tot_o, 1e-9)
        rdd = sum(v for k, v in u.items() if k[0] == "d" and k in seen[(team, s)]) / max(tot_d, 1e-9)
        vals.append(w * (0.5 * ro + 0.5 * rdd - 0.6))    # 0.6 ~ typical returning share; centring only
    RET[i] = vals[0] - vals[1]
    for team in (g["home"], g["away"]):
        gpl[(team, s)] += 1
FE["P"] = RET


# ------------------------------------------------------------ harness ------
def wf(extra=None, replace=None, drop=None):
    X = X14.copy()
    if replace:
        for c, v in replace.items():
            X[:, c] = v
    if drop:
        X = np.delete(X, drop, axis=1)
    per, vec = {}, []
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[tr].max() < s_ and SEAS[te].max() < TEST_ERA
        Xs = X
        if extra is not None:
            E = np.column_stack(extra) if isinstance(extra, (list, tuple)) else extra.reshape(-1, 1)
            mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec)


base_ll, base_per, base_vec = wf()
print(f"baseline DEV wf LL {base_ll:.5f} (recorded 0.62292)")
assert abs(base_ll - 0.62292) < 5e-4
OUT = {"baseline_dev_ll": base_ll, "repro_corr": repro, "results": {}}


def rep(nm, res):
    ll, per, vec = res
    d = base_vec - vec
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if base_per[s] - per[s] > 0)
    print(f"  {nm:<46} {ll:.5f} gain {base_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10", flush=True)
    OUT["results"][nm] = {"ll": round(ll, 6), "gain": round(base_ll - ll, 6),
                          "ci": [round(lo, 6), round(hi, 6)], "seasons_pos": pos,
                          "per_season_gain": {str(s): round(base_per[s] - per[s], 5) for s in per}}


# harness sensitivity: dropping the biggest DEV-live feature must cost a lot
rep("CONTROL drop qb_delta (col 1)", wf(drop=[1]))
rep("CONTROL drop elo_logit (col 0)", wf(drop=[0]))
rep("V1 winsorised +-3 EPA channels (replace)", wf(replace={2: FE["V1"][0], 12: FE["V1"][1], 13: FE["V1"][2]}))
rep("V2 floor -3 EPA channels (replace)", wf(replace={2: FE["V2"][0], 12: FE["V2"][1], 13: FE["V2"][2]}))
ao, ad, po, pd, ro, rd_ = FE["L"]
rep("L1 epa_net -> off,def (replace)", wf(replace={2: ao}, extra=[ad]))
rep("L2 all 3 channels -> off,def", wf(replace={2: ao, 12: po, 13: ro}, extra=[ad, pd, rd_]))
rep("Q opp-adjusted QB increment (add)", wf(FE["Q"]))
rep("S1 no-stakes asymmetry wk15-17 (add)", wf(FE["S1"]))
rep("S2 locked + eliminated diffs (add)", wf(list(FE["S2"])))
rep("P returning production wk1-4 fade (add)", wf(FE["P"]))
json.dump(OUT, open("data/bt_nfl_ideas_quick2.json", "w"), indent=1)
print("wrote data/bt_nfl_ideas_quick2.json")
