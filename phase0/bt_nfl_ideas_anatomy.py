"""Breakthrough program (NFL, ideas leg) - gap anatomy of candidate channels. DEV ONLY.

Reads the DEV cache written by phase0/bt_nfl_ideas_cache.py (shipped blend,
walk-forward OOF predictions 2006-2015) and asks, for each candidate information
channel, whether the de-vigged CLOSE moves in the direction of that channel
where our model does not:

    r_i = logit(p_close_i) - logit(p_ours_i)       (home perspective)

    probe(x):  OLS r ~ z(x) on DEV rows -> slope, t, share of r-variance
               plus the in-slice gap (our LL - market LL) for binary slices.

MARKET USE: EVALUATION / DIAGNOSTIC ONLY (sanctioned by the 2026-09-24 prereg:
"to locate where the model loses to the close (gap anatomy) on DEV seasons").
No odds enter any feature, any rating, or any training signal. Every candidate
proxy below is computed from pre-game public information only (results, plays,
injury reports, schedules) in strictly as-of walks.

LOCKED SPLIT: NFL DEV 2006-2015. Nothing at season >= 2016 is loaded from the
odds, scored, or printed. The cache itself contains no season >= 2016 rows.

Output: data/bt_nfl_ideas_anatomy.json
"""
from __future__ import annotations

import csv
import glob
import json
import math
from collections import Counter, defaultdict

import numpy as np

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
RNG = np.random.default_rng(20260924)

D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
QH, QA = np.array(D["qh"]), np.array(D["qa"])
N = len(G)
SEAS = np.array([g["season"] for g in G])
assert SEAS.max() < TEST_ERA, "cache contains TEST-era rows"
WK = np.array([g["week"] for g in G])
TYP = np.array([g["type"] for g in G])
Y = np.array([g["y"] for g in G], float)
PRED = np.array([np.nan if g["pred"] is None else g["pred"] for g in G])
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

# ------------------------------------------------------------------ raw spine
RAW = {}
for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")):
    if r["season"] and int(r["season"]) < TEST_ERA:
        RAW[r["game_id"]] = r
HS = np.array([float(RAW[g["gid"]]["home_score"]) for g in G])
AS = np.array([float(RAW[g["gid"]]["away_score"]) for g in G])
HCOACH = [RAW[g["gid"]]["home_coach"] for g in G]
ACOACH = [RAW[g["gid"]]["away_coach"] for g in G]

# ------------------------------------------------ closing line (EVAL ONLY) ---
def imp(v):
    return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))


PC = np.full(N, np.nan)
for i, g in enumerate(G):
    if not (DEV_LO <= g["season"] <= DEV_HI):
        continue
    r = RAW[g["gid"]]
    if r["home_moneyline"] and r["away_moneyline"]:
        qh_, qa_ = imp(float(r["home_moneyline"])), imp(float(r["away_moneyline"]))
        PC[i] = qh_ / (qh_ + qa_)
DEV = (SEAS >= DEV_LO) & (SEAS <= DEV_HI) & ~np.isnan(PRED) & ~np.isnan(PC) & (Y != 0.5)
assert SEAS[DEV].max() < TEST_ERA
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
R = np.where(DEV, lg(np.clip(PC, 1e-6, 1 - 1e-6)) - lg(np.clip(PRED, 1e-6, 1 - 1e-6)), np.nan)


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


LL_O = np.where(DEV, llv(Y, np.nan_to_num(PRED, nan=.5)), np.nan)
LL_M = np.where(DEV, llv(Y, np.nan_to_num(PC, nan=.5)), np.nan)
GAP = LL_O - LL_M
OUT = {"n_dev": int(DEV.sum()), "our_ll": float(np.nanmean(LL_O[DEV])),
       "market_ll": float(np.nanmean(LL_M[DEV])), "gap": float(np.nanmean(GAP[DEV])),
       "r_sd": float(np.nanstd(R[DEV])), "probes": {}, "slices": {}}
print(f"DEV n={OUT['n_dev']} ours {OUT['our_ll']:.5f} close {OUT['market_ll']:.5f} "
      f"gap {OUT['gap']:+.5f}  sd(r) {OUT['r_sd']:.3f}")


def probe(name, x, mask=None, note=""):
    """OLS of the market residual on a standardised candidate, DEV rows (x finite)."""
    m = DEV.copy() if mask is None else (DEV & mask)
    m &= np.isfinite(x)
    xx, rr = x[m], R[m]
    if m.sum() < 30 or xx.std() < 1e-12:
        print(f"  {name:<44} n={m.sum()} degenerate")
        return
    z = (xx - xx.mean()) / xx.std()
    A = np.column_stack([np.ones(len(z)), z])
    b, *_ = np.linalg.lstsq(A, rr, rcond=None)
    res = rr - A @ b
    s2 = res @ res / (len(z) - 2)
    se = math.sqrt(s2 / (z @ z))
    r2 = 1 - (res @ res) / ((rr - rr.mean()) @ (rr - rr.mean()))
    # per-season sign consistency
    signs = []
    for s_ in range(DEV_LO, DEV_HI + 1):
        ms = m & (SEAS == s_)
        if ms.sum() >= 15 and x[ms].std() > 1e-12:
            zs = (x[ms] - x[ms].mean()) / x[ms].std()
            signs.append(float(np.sign(zs @ R[ms])))
    OUT["probes"][name] = {"n": int(m.sum()), "slope_logit_per_sd": round(float(b[1]), 5),
                           "t": round(float(b[1] / se), 2), "r2": round(float(r2), 5),
                           "seasons_pos": int(sum(1 for s in signs if s > 0)),
                           "seasons_n": len(signs), "note": note}
    print(f"  {name:<44} n={m.sum():4d} slope {b[1]:+.4f}/sd t={b[1]/se:+.2f} "
          f"R2={r2:.4f} pos {sum(1 for s in signs if s>0)}/{len(signs)}")


def slice_(name, mask):
    m = DEV & mask
    if m.sum() < 10:
        return
    g = GAP[m]
    se = g.std(ddof=1) / math.sqrt(m.sum())
    OUT["slices"][name] = {"n": int(m.sum()), "gap": round(float(g.mean()), 5),
                           "se": round(float(se), 5),
                           "share_of_total_gap": round(float(g.sum() / np.nansum(GAP[DEV])), 3),
                           "mean_abs_r": round(float(np.abs(R[m]).mean()), 4)}
    print(f"  slice {name:<38} n={m.sum():4d} gap {g.mean():+.4f} (se {se:.4f}) "
          f"share {g.sum()/np.nansum(GAP[DEV]):.3f} |r| {np.abs(R[m]).mean():.3f}")


# =================================================== 0. market-vs-us slope ===
print("\n[0] market logit on our logit, by week bucket (b>1: close sharper than us)")
OUT["slope_by_bucket"] = {}
for nm, m_ in (("wk1-2", (WK <= 2) & (TYP == "REG")), ("wk3-4", (WK >= 3) & (WK <= 4)),
               ("wk5-8", (WK >= 5) & (WK <= 8)), ("wk9-13", (WK >= 9) & (WK <= 13)),
               ("wk14-17", (WK >= 14) & (TYP == "REG")), ("playoffs", TYP != "REG")):
    m = DEV & m_
    a = lg(PRED[m]); c = lg(PC[m])
    A = np.column_stack([np.ones(m.sum()), a])
    b, *_ = np.linalg.lstsq(A, c, rcond=None)
    OUT["slope_by_bucket"][nm] = {"n": int(m.sum()), "a": round(float(b[0]), 4),
                                  "b": round(float(b[1]), 4),
                                  "corr": round(float(np.corrcoef(a, c)[0, 1]), 4)}
    print(f"  {nm:<9} n={m.sum():4d} close = {b[0]:+.3f} + {b[1]:.3f}*ours  corr {np.corrcoef(a, c)[0,1]:.3f}")

# ============================================================ 1. QB shock ====
print("\n[1] QB shock: today's starter vs the QB the team ratings were built on")
prev_qb = {}
ew = {d: defaultdict(lambda: [0.0, 0.0]) for d in (0.6, 0.8, 0.9)}
shock = {d: np.zeros(N) for d in ew}
qchg_h = np.zeros(N); qchg_a = np.zeros(N)
for i, g in enumerate(G):
    for side, team, qid, q in ((1, g["home"], g["home_qb"], QH[i]), (-1, g["away"], g["away_qb"], QA[i])):
        ch = 1.0 if (team in prev_qb and prev_qb[team] != qid) else 0.0
        if side == 1:
            qchg_h[i] = ch
        else:
            qchg_a[i] = ch
        for d, st in ew.items():
            s_ = st[team]
            base = s_[0] / s_[1] if s_[1] > 0 else q
            shock[d][i] += side * (q - base)
    for team, qid, q in ((g["home"], g["home_qb"], QH[i]), (g["away"], g["away_qb"], QA[i])):
        prev_qb[team] = qid
        for d, st in ew.items():
            s_ = st[team]
            s_[0] = d * s_[0] + q
            s_[1] = d * s_[1] + 1.0
qd = X14[:, 1]
anyc = (qchg_h + qchg_a) > 0
slice_("QB change either side", anyc)
slice_("QB change none", ~anyc)
for d in shock:
    probe(f"qb_shock d={d}", shock[d])
    probe(f"qb_shock d={d} | QB-change games", shock[d], anyc)
probe("qb_delta (shipped col 1) all", qd)
probe("qb_delta | QB-change games", qd, anyc)
# does shock carry info beyond qb_delta? residualise on qb_delta within DEV
m = DEV
b_ = np.polyfit(qd[m], shock[0.8][m], 1)
probe("qb_shock d=.8 residualised on qb_delta", shock[0.8] - np.polyval(b_, qd))
OUT["qb_change_rate"] = float(anyc[DEV].mean())
OUT["corr_shock_qdelta"] = float(np.corrcoef(shock[0.8][DEV], qd[DEV])[0, 1])

# ============================================= 2. preseason / early season ===
print("\n[2] preseason information (weeks 1-2 and 1-4)")
# previous-season team summaries from results (REG only) + play EPA
teamseas = defaultdict(lambda: {"w": 0.0, "g": 0, "pf": 0.0, "pa": 0.0, "qbs": Counter(),
                                "coach": None, "last_elo": None})
for i, g in enumerate(G):
    if g["type"] != "REG":
        continue
    s = g["season"]
    for team, pf, pa, qid, co in ((g["home"], HS[i], AS[i], g["home_qb"], HCOACH[i]),
                                  (g["away"], AS[i], HS[i], g["away_qb"], ACOACH[i])):
        t = teamseas[(team, s)]
        t["w"] += 1.0 if pf > pa else 0.5 if pf == pa else 0.0
        t["g"] += 1; t["pf"] += pf; t["pa"] += pa; t["qbs"][qid] += 1; t["coach"] = co
# play EPA per game/team (pre-2016 only)
gepa = defaultdict(lambda: [0.0, 0])
with open("data/nfl_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: k for k, c in enumerate(hdr)}
    for r in rd:
        s = int(r[ix["game_id"]][:4])
        if s >= TEST_ERA:
            continue
        a = gepa[(r[ix["game_id"]], FR.get(r[ix["posteam"]], r[ix["posteam"]]).replace("JAC", "JAX"))]
        a[0] += float(r[ix["epa"]]); a[1] += 1
seas_epa = defaultdict(lambda: [0.0, 0, 0.0, 0])   # (team,season): off sum,n, def sum,n
late_epa = defaultdict(list)
for i, g in enumerate(G):
    if g["type"] != "REG":
        continue
    for t_off, t_def in ((g["home"], g["away"]), (g["away"], g["home"])):
        a = gepa.get((g["gid"], t_off))
        if not a:
            continue
        so = seas_epa[(t_off, g["season"])]; so[0] += a[0]; so[1] += a[1]
        sd_ = seas_epa[(t_def, g["season"])]; sd_[2] += a[0]; sd_[3] += a[1]
        late_epa[(t_off, g["season"])].append((g["week"], a[0] / max(a[1], 1), +1))
        late_epa[(t_def, g["season"])].append((g["week"], a[0] / max(a[1], 1), -1))
print(f"  EPA games covered: {sum(1 for k in gepa)} team-games")


def prev_feats(team, s, qid, coach):
    t = teamseas.get((team, s - 1))
    if not t or t["g"] == 0:
        return None
    wp = t["w"] / t["g"]
    pdg = (t["pf"] - t["pa"]) / t["g"]
    pyth = t["pf"] ** 2.37 / (t["pf"] ** 2.37 + t["pa"] ** 2.37)
    e = seas_epa.get((team, s - 1))
    net = (e[0] / e[1] - e[2] / e[3]) if e and e[1] and e[3] else 0.0
    lt = sorted(late_epa.get((team, s - 1), []))
    lastn = [v * sg for (_, v, sg) in lt if _ >= 11]
    late = float(np.mean(lastn)) if lastn else 0.0
    top_qb = t["qbs"].most_common(1)[0][0] if t["qbs"] else None
    return {"wp": wp, "pdg": pdg, "pyth_luck": wp - pyth, "net_epa": net, "late_net": late,
            "qb_new": 0.0 if qid == top_qb else 1.0,
            "coach_new": 0.0 if coach == t["coach"] else 1.0}


PRE = {k: np.full(N, np.nan) for k in ("wp", "pdg", "pyth_luck", "net_epa", "late_net",
                                       "qb_new", "coach_new")}
first_game = {}
for i, g in enumerate(G):
    fh = prev_feats(g["home"], g["season"], g["home_qb"], HCOACH[i])
    fa = prev_feats(g["away"], g["season"], g["away_qb"], ACOACH[i])
    if fh and fa:
        for k in PRE:
            PRE[k][i] = fh[k] - fa[k]
early2 = (WK <= 2) & (TYP == "REG")
early4 = (WK <= 4) & (TYP == "REG")
slice_("weeks 1-2", early2)
slice_("weeks 3-4", (WK >= 3) & (WK <= 4))
for k in PRE:
    probe(f"prev-season {k} | wk1-2", PRE[k], early2)
    probe(f"prev-season {k} | wk1-4", PRE[k], early4)
probe("prev-season net_epa | wk5-17", PRE["net_epa"], (WK >= 5) & (TYP == "REG"))
probe("prev-season pdg | wk5-17", PRE["pdg"], (WK >= 5) & (TYP == "REG"))

# ================================================ 3. late-season leverage ====
print("\n[3] late-season playoff leverage (Monte Carlo from as-of plain Elo)")
DIV = {"AFC East": "BUF MIA NE NYJ", "AFC North": "BAL CIN CLE PIT",
       "AFC South": "HOU IND JAX TEN", "AFC West": "DEN KC LV LAC",
       "NFC East": "DAL NYG PHI WAS", "NFC North": "CHI DET GB MIN",
       "NFC South": "ATL CAR NO TB", "NFC West": "ARI LA SF SEA"}
TDIV = {t: d for d, ts in DIV.items() for t in ts.split()}
TEAMS = sorted(TDIV)
TIX = {t: k for k, t in enumerate(TEAMS)}
K_, HFA_, REG_ = 47.43351932834238, 52.14162888646703, 0.3647684062154459
elo = defaultdict(lambda: 1500.0)
ELO_PRE = np.zeros((N, 2))
prev_s = None
for i, g in enumerate(G):
    if prev_s is not None and g["season"] != prev_s:
        for t in list(elo):
            elo[t] = 1500.0 + (1 - REG_) * (elo[t] - 1500.0)
    prev_s = g["season"]
    h, a = g["home"], g["away"]
    ELO_PRE[i] = (elo[h], elo[a])
    dr = elo[h] - elo[a] + (0.0 if g["neutral"] else HFA_)
    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
    elo[h] += K_ * (yy - e); elo[a] -= K_ * (yy - e)

LEV_PO = np.full(N, np.nan); LEV_BYE = np.full(N, np.nan)
PPO = np.full((N, 2), np.nan)
NS = 4000
conf_of = np.array([TDIV[t][:3] for t in TEAMS])
div_of = np.array([TDIV[t] for t in TEAMS])
divs = sorted(set(div_of))
for s in range(DEV_LO, DEV_HI + 1):
    reg = [i for i in range(N) if SEAS[i] == s and TYP[i] == "REG"]
    for W in range(11, 18):
        wk_games = [i for i in reg if WK[i] == W]
        if not wk_games:
            continue
        done = [i for i in reg if WK[i] < W]
        rem = [i for i in reg if WK[i] >= W]
        base_w = np.zeros(len(TEAMS))
        for i in done:
            h, a = TIX[G[i]["home"]], TIX[G[i]["away"]]
            yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
            base_w[h] += yy; base_w[a] += 1 - yy
        # ratings frozen at the first game of week W (as-of)
        rat = {}
        for i in wk_games:
            rat[G[i]["home"]] = ELO_PRE[i, 0]; rat[G[i]["away"]] = ELO_PRE[i, 1]
        # bye teams: rating after their last game before week W (as-of)
        for t in TEAMS:
            if t not in rat or not any(G[i]["home"] == t or G[i]["away"] == t for i in wk_games):
                past = [i for i in done if G[i]["home"] == t or G[i]["away"] == t]
                if past:
                    j = past[-1]
                    hh, aa = G[j]["home"], G[j]["away"]
                    dr = ELO_PRE[j, 0] - ELO_PRE[j, 1] + (0.0 if G[j]["neutral"] else HFA_)
                    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
                    yy = 1.0 if HS[j] > AS[j] else 0.5 if HS[j] == AS[j] else 0.0
                    rat[t] = ELO_PRE[j, 0] + K_ * (yy - e) if hh == t else ELO_PRE[j, 1] - K_ * (yy - e)
                else:
                    rat[t] = 1500.0
        hi = np.array([TIX[G[i]["home"]] for i in rem]); ai = np.array([TIX[G[i]["away"]] for i in rem])
        dr = np.array([rat[G[i]["home"]] - rat[G[i]["away"]] + (0.0 if G[i]["neutral"] else HFA_) for i in rem])
        pe = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        U = RNG.random((NS, len(rem)))
        hw = (U < pe).astype(float)
        wins = np.tile(base_w, (NS, 1))
        np.add.at(wins.T, hi, hw.T)
        np.add.at(wins.T, ai, (1 - hw).T)
        noisy = wins + RNG.random(wins.shape) * 1e-3
        po = np.zeros_like(wins, bool); bye = np.zeros_like(wins, bool)
        for cf in ("AFC", "NFC"):
            cidx = np.where(conf_of == cf)[0]
            dwin = []
            for dv in divs:
                if not dv.startswith(cf):
                    continue
                didx = np.where(div_of == dv)[0]
                dwin.append(didx[np.argmax(noisy[:, didx], axis=1)])
            dwin = np.stack(dwin, 1)                        # NS x 4
            isdw = np.zeros((NS, len(TEAMS)), bool)
            np.put_along_axis(isdw, dwin, True, axis=1)
            wc_score = np.where(isdw[:, cidx], -1e9, noisy[:, cidx])
            wc = cidx[np.argsort(-wc_score, axis=1)[:, :2]]
            po_c = isdw.copy(); np.put_along_axis(po_c, wc, True, axis=1)
            po |= po_c
            dw_score = np.take_along_axis(noisy, dwin, 1)
            top2 = np.take_along_axis(dwin, np.argsort(-dw_score, 1)[:, :2], 1)
            np.put_along_axis(bye, top2, True, axis=1)
        col = {i: k for k, i in enumerate(rem)}
        for i in wk_games:
            c = col[i]; h, a = TIX[G[i]["home"]], TIX[G[i]["away"]]
            won = hw[:, c] == 1.0
            if won.sum() < 50 or (~won).sum() < 50:
                continue
            lp_h = po[won, h].mean() - po[~won, h].mean()
            lp_a = po[~won, a].mean() - po[won, a].mean()
            lb_h = bye[won, h].mean() - bye[~won, h].mean()
            lb_a = bye[~won, a].mean() - bye[won, a].mean()
            LEV_PO[i] = lp_h - lp_a
            LEV_BYE[i] = lb_h - lb_a
            PPO[i] = (po[:, h].mean(), po[:, a].mean())
late = (WK >= 11) & (TYP == "REG")
LEV_TOT = LEV_PO + 0.5 * LEV_BYE
probe("lev_po diff | wk11-17", LEV_PO, late)
probe("lev_bye diff | wk11-17", LEV_BYE, late)
probe("lev_total diff | wk11-17", LEV_TOT, late)
probe("lev_total diff | wk15-17", LEV_TOT, (WK >= 15) & (TYP == "REG"))
probe("lev_total diff | wk17", LEV_TOT, (WK == 17) & (TYP == "REG"))
OUT["lev_coverage"] = int(np.isfinite(LEV_TOT[DEV & late]).sum())
slice_("wk17 REG", (WK == 17) & (TYP == "REG"))
slice_("wk15-16 REG", (WK >= 15) & (WK <= 16) & (TYP == "REG"))
slice_("|lev diff|>0.3 wk11-17", late & (np.abs(np.nan_to_num(LEV_TOT)) > 0.3))
slice_("|lev diff|<=0.3 wk11-17", late & (np.abs(np.nan_to_num(LEV_TOT)) <= 0.3))

# ================================================ 4. opponent-adjusted EPA ===
print("\n[4] opponent-adjusted team EPA vs the shipped raw EPA walk")
dec, pn, sdec, LG = 0.8813581205848026, 754.0438610349224, 0.7130881070009863, -0.02334
off = defaultdict(lambda: [0.0, 0.0]); dfn = defaultdict(lambda: [0.0, 0.0])
offA = defaultdict(lambda: [0.0, 0.0]); dfnA = defaultdict(lambda: [0.0, 0.0])
raw_net = np.zeros(N); adj_net = np.zeros(N)


def rate(st):
    return (st[0] + pn * LG) / (st[1] + pn)


prev_s = None
for i, g in enumerate(G):
    if prev_s is not None and g["season"] != prev_s:
        for dct in (off, dfn, offA, dfnA):
            for st in dct.values():
                st[0] *= (1 - sdec); st[1] *= (1 - sdec)
    prev_s = g["season"]
    h, a = g["home"], g["away"]
    raw_net[i] = (rate(off[h]) - rate(dfn[h])) - (rate(off[a]) - rate(dfn[a]))
    adj_net[i] = (rate(offA[h]) - rate(dfnA[h])) - (rate(offA[a]) - rate(dfnA[a]))
    upd = []
    for t_off, t_def in ((h, a), (a, h)):
        e = gepa.get((g["gid"], t_off))
        if not e:
            continue
        s_, n_ = e
        # adjusted observations use the opponent's PRE-game adjusted rating
        upd.append((t_off, t_def, s_, n_, s_ - n_ * (rate(dfnA[t_def]) - LG),
                    s_ - n_ * (rate(offA[t_off]) - LG)))
    for t_off, t_def, s_, n_, s_off_adj, s_def_adj in upd:
        o = off[t_off]; o[0] = dec * o[0] + s_; o[1] = dec * o[1] + n_
        d_ = dfn[t_def]; d_[0] = dec * d_[0] + s_; d_[1] = dec * d_[1] + n_
        o = offA[t_off]; o[0] = dec * o[0] + s_off_adj; o[1] = dec * o[1] + n_
        d_ = dfnA[t_def]; d_[0] = dec * d_[0] + s_def_adj; d_[1] = dec * d_[1] + n_
cc = float(np.corrcoef(raw_net[DEV], X14[DEV, 2])[0, 1])
print(f"  corr(my raw net, shipped col 2 epa_net) = {cc:.4f}")
OUT["epa_raw_repro_corr"] = cc
probe("opp-adj minus raw net EPA", adj_net - raw_net)
probe("opp-adj net EPA (level)", adj_net)
OUT["corr_adj_raw"] = float(np.corrcoef(adj_net[DEV], raw_net[DEV])[0, 1])

# =================================================== 5. injury-report Outs ===
print("\n[5] injury report Out/Doubtful (2009-2015), usage-weighted, QB excluded")
# usage shares from weekly stats: offense (targets + carries), defense (tackles + 2*sacks)
use_off = defaultdict(float); use_def = defaultdict(float)
tm_off = defaultdict(float); tm_def = defaultdict(float)
wk_off = defaultdict(list); wk_def = defaultdict(list)
with open("data/nfl_player_stats.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        s = int(r["season"])
        if s < 2008 or s >= TEST_ERA or r["season_type"] != "REG":
            continue
        if r["position"] == "QB":
            continue
        v = float(r["targets"] or 0) + float(r["carries"] or 0)
        t = FR.get(r["recent_team"], r["recent_team"])
        wk_off[(t, s)].append((int(r["week"]), r["player_id"], v))
with open("data/nfl_player_stats_def.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        s = int(r["season"])
        if s < 2008 or s >= TEST_ERA or r["season_type"] != "REG":
            continue
        v = float(r["def_tackles_solo"] or 0) + float(r["def_tackle_assists"] or 0) + 2 * float(r["def_sacks"] or 0)
        t = FR.get(r["team"], r["team"])
        wk_def[(t, s)].append((int(r["week"]), r["player_id"], v))


def shares(tbl, team, s, week):
    """player usage share on `team` from all REG weeks < week this season, else last season."""
    rows = [(p, v) for (w, p, v) in tbl.get((team, s), []) if w < week]
    if not rows:
        rows = [(p, v) for (w, p, v) in tbl.get((team, s - 1), [])]
    tot = sum(v for _, v in rows)
    if tot <= 0:
        return {}
    acc = defaultdict(float)
    for p, v in rows:
        acc[p] += v
    return {p: v / tot for p, v in acc.items()}


INJ = defaultdict(list)
for f in sorted(glob.glob("data/inj_20*.csv")):
    s = int(f[-8:-4])
    if s >= TEST_ERA:
        continue
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["game_type"] != "REG" or r["report_status"] not in ("Out", "Doubtful"):
            continue
        t = FR.get(r["team"], r["team"])
        INJ[(t, s, int(r["week"]))].append((r["gsis_id"], r["position"]))
OUTOFF = np.full(N, np.nan); OUTDEF = np.full(N, np.nan); OUTOL = np.full(N, np.nan)
for i, g in enumerate(G):
    if g["season"] < 2009 or g["type"] != "REG":
        continue
    v = []
    for team in (g["home"], g["away"]):
        so = shares(wk_off, team, g["season"], g["week"])
        sd_ = shares(wk_def, team, g["season"], g["week"])
        lst = INJ.get((team, g["season"], g["week"]), [])
        v.append((sum(so.get(p, 0.0) for p, pos in lst if pos != "QB"),
                  sum(sd_.get(p, 0.0) for p, pos in lst),
                  sum(1.0 for p, pos in lst if pos in ("T", "G", "C", "OL", "OT", "OG"))))
    OUTOFF[i] = v[0][0] - v[1][0]; OUTDEF[i] = v[0][1] - v[1][1]; OUTOL[i] = v[0][2] - v[1][2]
pre13 = SEAS < 2013
probe("Out/D skill usage share diff (2009-15)", OUTOFF)
probe("Out/D skill usage share diff (2009-12)", OUTOFF, pre13)
probe("Out/D def usage share diff (2009-15)", OUTDEF)
probe("Out/D OL count diff (2009-15)", OUTOL)
probe("Out/D OL count diff (2009-12)", OUTOL, pre13)

# ======================================================= 6. disagreement ====
print("\n[6] who carries the top-disagreement games?")
top = DEV & (np.abs(np.nan_to_num(PC) - np.nan_to_num(PRED)) > 0.12)
OUT["top_dis"] = {"n": int(top.sum()),
                  "share_qb_change": float(anyc[top].mean()), "share_qb_change_all": float(anyc[DEV].mean()),
                  "share_wk1_2": float(early2[top].mean()), "share_wk1_2_all": float(early2[DEV].mean()),
                  "share_wk15_17": float(((WK >= 15) & (TYP == "REG"))[top].mean()),
                  "share_wk15_17_all": float(((WK >= 15) & (TYP == "REG"))[DEV].mean()),
                  "share_playoffs": float((TYP != "REG")[top].mean()),
                  "share_playoffs_all": float((TYP != "REG")[DEV].mean())}
print("  ", OUT["top_dis"])
slice_("top |dp|>0.12", top)
slice_("top |dp|>0.12 & no QB change & wk3-14", top & ~anyc & (WK >= 3) & (WK <= 14))

with open("data/bt_nfl_ideas_anatomy.json", "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1)
print("\nwrote data/bt_nfl_ideas_anatomy.json")
