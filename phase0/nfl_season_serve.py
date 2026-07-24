"""Serve the 2026 NFL season: freeze end-2025 states from every feature walk,
refit the production blend per the adopted training protocol (all games <=2025,
recency half-life 3 seasons, C=100), predict all 272 scheduled games, and
Monte-Carlo the season for projected standings / division / playoff odds.

Feature semantics for future games (preseason serve):
  elo/epa/pass/run/ts/luck  end-2025 states + the SAME season-boundary transform
                            the historical walks apply at every season change
  qb_pedigree               live 2026 depth charts (data/nfl_qb2026.json) through
                            the QB Elo state; unseen rookies -> pedigree prior
  team_hfa                  walk state as-is (no boundary logic in the walk)
  roster_quality            expected lineup = 2025 actives on current roster,
                            avg-snap-share weighted (matches walk semantics)
  rest/early_kick           from the schedule (nflverse rest columns + TZ rule)
  absences                  0 (no injury reports yet); in-season refresh fills

Outputs into site/data/nfl.json: schedule[272] with p_home, proj{} per team,
teams[].glassbox roster averages, serve block in model_card.
"""
from __future__ import annotations

import copy
import csv
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # full feature prelude  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done ({len(games)} games)", flush=True)

# ---------------- pass/run channels + end states (training_eval block) ----------------
aggc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        t_ = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
        a = aggc[r[ix["game_id"]]][t_]
        if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
            a[0] += float(r[ix["epa"]]); a[1] += 1
        else:
            a[2] += float(r[ix["epa"]]); a[3] += 1
ps = pn = rs = rn = 0.0
for gid, tm in aggc.items():
    if int(gid[:4]) in DEV_YEARS:
        for t_, (a_, b_, c_, d_) in tm.items():
            ps += a_; pn += b_; rs += c_; rn += d_
LGP, LGR = ps / pn, rs / rn
dec_, pn_, sd_ = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
offP = defaultdict(lambda: [0.0, 0.0]); offR = defaultdict(lambda: [0.0, 0.0])
dfaP = defaultdict(lambda: [0.0, 0.0]); dfaR = defaultdict(lambda: [0.0, 0.0])
def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
f_pass = np.zeros(len(games)); f_run = np.zeros(len(games)); prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for st_ in (list(offP.values()) + list(offR.values())
                    + list(dfaP.values()) + list(dfaR.values())):
            st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
    prev = g["season"]
    h, a = g["home"], g["away"]
    f_pass[i] = (crate(offP[h], LGP) - crate(dfaP[h], LGP)) - (crate(offP[a], LGP) - crate(dfaP[a], LGP))
    f_run[i] = (crate(offR[h], LGR) - crate(dfaR[h], LGR)) - (crate(offR[a], LGR) - crate(dfaR[a], LGR))
    tm = aggc.get(g["gid"])
    if tm:
        for t_off, opp in ((h, a), (a, h)):
            pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
            o = offP[t_off]; o[0] = dec_ * o[0] + pS; o[1] = dec_ * o[1] + pN
            o = offR[t_off]; o[0] = dec_ * o[0] + rS; o[1] = dec_ * o[1] + rN
            d2 = dfaP[opp]; d2[0] = dec_ * d2[0] + pS; d2[1] = dec_ * d2[1] + pN
            d2 = dfaR[opp]; d2[0] = dec_ * d2[0] + rS; d2[1] = dec_ * d2[1] + rN
X14 = np.column_stack([X_of(F), f_pass, f_run])
v6_hist = np.load("data/nfl_v7_feature.npy")
assert len(v6_hist) == len(games)
X14[:, 7] = v6_hist                    # rows 63+66: ts_edge -> player ratings (v7 cohort)
print(f"[{time.time()-T0:.0f}s] X14 built (col 7 = v6 player ratings)", flush=True)

# ---------------- protocol check: reproduce TEST 0.61947 (walk-forward+recency) ----------------
HL, BC = 3.0, 100.0
p_wf = np.zeros(int(test.sum()))
tidx = np.where(test)[0]
pos = {gi: j for j, gi in enumerate(tidx)}
for s_ in sorted(np.unique(seasons[test])):
    tr = (seasons < s_) & (y != 0.5)
    te = test & (seasons == s_)
    w_ = 0.5 ** ((s_ - 1 - seasons[tr]) / HL)
    m_ = LogisticRegression(C=BC, max_iter=5000).fit(X14[tr], y[tr], sample_weight=w_)
    pp = m_.predict_proba(X14[te])[:, 1]
    for k2, gi in enumerate(np.where(te)[0]):
        p_wf[pos[gi]] = pp[k2]
repro = float(llv(y[test], p_wf).mean())
print(f"[{time.time()-T0:.0f}s] TEST reproduction LL {repro:.5f} (ledger 0.61947)", flush=True)
assert abs(repro - 0.61947) < 0.0005, "feature build does not reproduce the shipped model"

# ---------------- production serving fit (the 2026 step of the walk-forward) ----------------
tr = (y != 0.5)
w_ = 0.5 ** ((2025 - seasons[tr]) / HL)
CLF = LogisticRegression(C=BC, max_iter=5000).fit(X14[tr], y[tr], sample_weight=w_)
print(f"[{time.time()-T0:.0f}s] serving blend fit on {int(tr.sum())} games", flush=True)

# ================= end-2025 state walks (same code as the builders) =================
bp = base["params"]

# Elo ratings
R = {}
prev = None
for g in games:
    if prev is not None and g["season"] != prev:
        for t in R:
            R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - bp["regress"])
    prev = g["season"]
    rh = R.setdefault(g["home"], 1500.0)
    ra = R.setdefault(g["away"], 1500.0)
    h_ = 0.0 if g["neutral"] else bp["hfa"]
    p_ = 1.0 / (1.0 + 10 ** (-((rh + h_) - ra) / 400.0))
    d_ = bp["k"] * (g["y"] - p_)
    R[g["home"]] += d_
    R[g["away"]] -= d_
for t in R:                                   # 2026 season boundary
    R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - bp["regress"])

# QB Elo (pedigree config = shipped qd_ped), then roll into 2026
mQB = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=pedP["delta"],
            rep_map=rep_map, decay=SP["decay"], prior_db=SP["prior_db"],
            season_decay=SP["season_decay"])
prev = None
for g in games:
    if prev is not None and g["season"] != prev:
        mQB.new_season()
    prev = g["season"]
    mQB.predict(g)
    mQB.update(g, 0.5, qbw)
mQB.new_season()
QB26 = {t: v["gsis"] for t, v in json.load(open("data/nfl_qb2026.json"))["qb"].items()}
assert len(QB26) == 32

# EPA net states
epa_off = defaultdict(lambda: [0.0, 0.0]); epa_def = defaultdict(lambda: [0.0, 0.0])
d_e, pn_e, sd_e = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
prev = None
for g in games:
    if prev is not None and g["season"] != prev:
        for st in list(epa_off.values()) + list(epa_def.values()):
            st[0] *= (1 - sd_e); st[1] *= (1 - sd_e)
    prev = g["season"]
    tm = agg.get(g["gid"])
    if tm:
        for t_off, opp in ((g["home"], g["away"]), (g["away"], g["home"])):
            s_, n_ = tm.get(t_off, (0.0, 0))
            o = epa_off[t_off]; o[0] = d_e * o[0] + s_; o[1] = d_e * o[1] + n_
            dd = epa_def[opp]; dd[0] = d_e * dd[0] + s_; dd[1] = d_e * dd[1] + n_
for st in list(epa_off.values()) + list(epa_def.values()):
    st[0] *= (1 - sd_e); st[1] *= (1 - sd_e)
def erate(st): return (st[0] + pn_e * LG_EPA) / (st[1] + pn_e)

# pass/run boundary into 2026 (states already at end-2025 from the block above)
for st in (list(offP.values()) + list(offR.values())
           + list(dfaP.values()) + list(dfaR.values())):
    st[0] *= (1 - sd_); st[1] *= (1 - sd_)

# per-team HFA state (walk has no boundary logic)
hfa_st = {}
for i, g in enumerate(games):
    if not g["neutral"]:
        s = hfa_st.get(g["home"], (0.0, 0.0))
        hfa_st[g["home"]] = (0.991 * s[0] + (g["y"] - pe[i]), 0.991 * s[1] + 1.0)

# fumble luck states
luck_st = defaultdict(lambda: [0.0, 0.0])
d_l, pn_l, sd_l = luckP["decay"], luckP["prior_n"], luckP["season_decay"]
prev = None
for g in games:
    if prev is not None and g["season"] != prev:
        for st in luck_st.values():
            st[0] *= (1 - sd_l); st[1] *= (1 - sd_l)
    prev = g["season"]
    h, a = g["home"], g["away"]
    gt = tv.get(g["gid"])
    if gt and h in gt and a in gt:
        _, fh, flh = gt[h]; _, fa, fla = gt[a]
        tot = fh + fa; rec_h = (fh - flh) + fla
        for team, rec in ((h, rec_h), (a, tot - rec_h)):
            s_ = luck_st[team]
            s_[0] = d_l * s_[0] + (rec - 0.5 * tot)
            s_[1] = d_l * s_[1] + tot
for st in luck_st.values():
    st[0] *= (1 - sd_l); st[1] *= (1 - sd_l)

# TrueSkill unit states (expensive walk)
from mlbwp.trueskill import TrueSkill1v1  # noqa: E402
MU0, SIG0 = 25.0, 25.0 / 3.0
tsm = TrueSkill1v1(beta=tsP["beta"], tau=tsP["tau"], mu0=MU0, sig0=SIG0)
prev = None
for g in games:
    if prev is not None and g["season"] != prev:
        for k2 in list(tsm.mu.keys()):
            tsm.mu[k2] = MU0 + (tsm.mu[k2] - MU0) * (1 - tsP["mu_reg"])
            tsm.var[k2] = min(tsm.var[k2] + tsP["widen"] ** 2, SIG0 * SIG0)
    prev = g["season"]
    for pos_, dfn, success in plays.get(g["gid"], ()):
        if success:
            tsm.update(pos_ + "_OFF", dfn + "_DEF")
        else:
            tsm.update(dfn + "_DEF", pos_ + "_OFF")
for k2 in list(tsm.mu.keys()):
    tsm.mu[k2] = MU0 + (tsm.mu[k2] - MU0) * (1 - tsP["mu_reg"])
    tsm.var[k2] = min(tsm.var[k2] + tsP["widen"] ** 2, SIG0 * SIG0)
print(f"[{time.time()-T0:.0f}s] state walks done", flush=True)

# roster-quality expected lineup: share walk (STATE already end-2025 from prelude)
share2, pos2, team2 = {}, {}, {}
roster2 = defaultdict(set)
for g in games:
    for team in (g["home"], g["away"]):
        tbl = snaps.get((g["gid"], team))
        if not tbl:
            continue
        for pid, (pos_, op, dp) in tbl.items():
            pct = dp if pos_ in DEFPOS else op
            st = share2.setdefault(pid, [0.0, 0.0])
            st[0] = snapP["decay"] * st[0] + pct
            st[1] = snapP["decay"] * st[1] + 1.0
            pos2[pid] = pos_
            if team2.get(pid) != team:
                if team2.get(pid) in roster2:
                    roster2[team2.get(pid)].discard(pid)
                team2[pid] = team
                roster2[team].add(pid)
active25 = set()
for (gid, team), tbl in snaps.items():
    if gid.startswith("2025"):
        active25.update(tbl.keys())

# 2026 re-homing from live rosters: movers count for their NEW team's roster
# quality (consistent with the live QB depth charts); unrostered players drop out
FR26 = {"JAC": "JAX", "WSH": "WAS", "AZ": "ARI", "LAR": "LA"}
r26 = {}
try:
    for r in csv.DictReader(open("data/roster_2026.csv", encoding="utf-8")):
        if r.get("gsis_id") and r.get("team"):
            r26[r["gsis_id"]] = FR26.get(r["team"], r["team"])
except FileNotFoundError:
    pass
moved = dropped = 0
if r26:
    for pid in list(team2):
        g_ = pfr2gsis.get(pid)
        if g_ is None or g_ not in r26:
            if team2[pid] in roster2:
                roster2[team2[pid]].discard(pid)
            dropped += 1
            continue
        tgt = r26[g_]
        if tgt != team2[pid]:
            if team2[pid] in roster2:
                roster2[team2[pid]].discard(pid)
            roster2[tgt].add(pid)
            team2[pid] = tgt
            moved += 1
print(f"[{time.time()-T0:.0f}s] 2026 re-homing: {moved} moved, {dropped} unrostered dropped",
      flush=True)

TS6 = json.load(open("data/nfl_ts_state.json"))
def serve_v6(team):
    tot = 0.0
    for pid in roster2.get(team, ()):
        if pid not in active25:
            continue
        st = share2.get(pid)
        if not st or st[1] <= 0:
            continue
        g_ = pfr2gsis.get(pid)
        s6 = TS6.get(g_) if g_ else None
        if s6 is None:
            continue
        mu6 = 25.0 + (s6[0] - 25.0) * (2.0 / 3.0)      # 2026 boundary
        s26 = min(s6[1] + 1.5 ** 2, (25.0 / 3.0) ** 2)
        tot += (st[0] / st[1]) * max(-3.0, min(3.0, mu6 - 25.0)) * (1.0 / (1.0 + s26))
    return tot

def serve_rq(team):
    tot = 0.0
    for pid in roster2.get(team, ()):
        if pid not in active25:
            continue
        st = share2.get(pid)
        if not st or st[1] <= 0:
            continue
        g_ = pfr2gsis.get(pid)
        r_ = rate6(g_) if g_ else None
        if r_ is not None:
            tot += (st[0] / st[1]) * max(-1.5, min(1.5, r_))
    return tot

# ================= 2026 schedule -> features -> probs =================
TZ = {"BUF": 0, "MIA": 0, "NE": 0, "NYJ": 0, "BAL": 0, "CIN": 0, "CLE": 0, "PIT": 0,
      "IND": 0, "JAX": 0, "ATL": 0, "CAR": 0, "TB": 0, "WAS": 0, "NYG": 0, "PHI": 0,
      "DET": 0, "CHI": 1, "GB": 1, "MIN": 1, "DAL": 1, "HOU": 1, "TEN": 1, "KC": 1,
      "NO": 1, "DEN": 2, "ARI": 2, "SEA": 3, "SF": 3, "LAC": 3, "LV": 3, "LA": 3}
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS"}
sched = []
for r in csv.DictReader(open("data/nfl_games.csv")):
    if r["season"] != "2026":
        continue
    h = FR.get(r["home_team"], r["home_team"])
    a = FR.get(r["away_team"], r["away_team"])
    try:
        kick = int(r["gametime"].split(":")[0]) + int(r["gametime"].split(":")[1]) / 60.0
    except (ValueError, IndexError, AttributeError):
        kick = 13.0
    sched.append({"w": int(r["week"]), "d": r["gameday"], "t": r["gametime"],
                  "home": h, "away": a, "neutral": 1 if r["location"] == "Neutral" else 0,
                  "hrest": int(r["home_rest"]), "arest": int(r["away_rest"]),
                  "kick": kick,
                  "hs": int(r["home_score"]) if r["home_score"] != "" else None,
                  "as": int(r["away_score"]) if r["away_score"] != "" else None})
sched.sort(key=lambda s: (s["d"], s["t"]))
assert len(sched) == 272

eps_ = 1e-12
Xs = np.zeros((len(sched), 14))
for i, s in enumerate(sched):
    h, a = s["home"], s["away"]
    hfa_pts = 0.0 if s["neutral"] else bp["hfa"]
    p_elo = 1.0 / (1.0 + 10 ** (-((R.get(h, 1500.0) + hfa_pts) - R.get(a, 1500.0)) / 400.0))
    early = 0.0
    if not s["neutral"] and TZ[a] - TZ[h] >= 2 and s["kick"] <= 14.0:
        early = 1.0
    Xs[i] = [
        np.log(p_elo / (1 - p_elo)),
        mQB.qb_adj(QB26[h]) - mQB.qb_adj(QB26[a]),
        (erate(epa_off[h]) - erate(epa_def[h])) - (erate(epa_off[a]) - erate(epa_def[a])),
        early,
        0.0 if s["neutral"] else hfa_st.get(h, (0.0, 0.0))[0] / (hfa_st.get(h, (0.0, 0.0))[1] + 180.0),
        max(-7, min(7, s["hrest"] - s["arest"])),
        luck_st[h][0] / (luck_st[h][1] + pn_l) - luck_st[a][0] / (luck_st[a][1] + pn_l),
        serve_v6(h) - serve_v6(a),
        0.0, 0.0, 0.0,
        serve_rq(h) - serve_rq(a),
        (crate(offP[h], LGP) - crate(dfaP[h], LGP)) - (crate(offP[a], LGP) - crate(dfaP[a], LGP)),
        (crate(offR[h], LGR) - crate(dfaR[h], LGR)) - (crate(offR[a], LGR) - crate(dfaR[a], LGR)),
    ]
probs = CLF.predict_proba(Xs)[:, 1]
co = CLF.coef_[0]
for i, (s, p_) in enumerate(zip(sched, probs)):
    s["ph"] = round(float(p_), 3)
    # grouped blend contributions, converted logit -> home prob points for display
    # (pass/run/EPA grouped because their coefs are collinear rotations; the sum is
    # the interpretable quantity)
    x = Xs[i]
    scale = float(p_ * (1 - p_)) * 100.0
    ct = {
        "elo": co[0] * x[0], "qb": co[1] * x[1],
        "units": co[2] * x[2] + co[12] * x[12] + co[13] * x[13],
        "sched": co[3] * x[3] + co[5] * x[5],
        "hfa": co[4] * x[4], "luck": co[6] * x[6], "ts": co[7] * x[7],
        "abs": co[8] * x[8] + co[9] * x[9] + co[10] * x[10],
        "roster": co[11] * x[11],
    }
    s["ct"] = {k: round(v * scale, 1) for k, v in ct.items()}
print(f"[{time.time()-T0:.0f}s] 2026 probs: mean home {probs.mean():.3f} "
      f"min {probs.min():.3f} max {probs.max():.3f}", flush=True)
assert 0.50 < probs.mean() < 0.62 and probs.min() > 0.05 and probs.max() < 0.95

# sanity: serve features must live inside historical feature ranges
recent = seasons >= 2020
for j, nm in enumerate(["lgt", "qd", "epa", "early", "thfa", "rest", "luck", "tse",
                        "ol", "de", "sk", "rq", "pass", "run"]):
    if nm in ("ol", "de", "sk"):
        continue
    lo, hi = X14[recent, j].min(), X14[recent, j].max()
    assert Xs[:, j].min() >= lo - 0.3 * (hi - lo) and Xs[:, j].max() <= hi + 0.3 * (hi - lo), \
        f"serve feature {nm} out of historical range"

# ================= Monte Carlo season sims (FEATURES update INSIDE each sim) =================
# Each sim walks the season chronologically. After every simulated game the
# outcome-driven features evolve exactly like their real walks: Elo (k update),
# per-team HFA (residual EWMA), and — via game stats sampled conditional on the
# simulated winner/loser (distributions fit on 2020-2025) — the EPA net rating,
# the pass/run channel ratings, and the starting QB's rating (QbElo EWMA + EB
# shrink to his pedigree prior). Frozen in-sim: ts_edge/luck (not inferable from
# a winner; fabricating fumbles or play-level duels would be invention), roster
# quality and absences (injuries cannot be honestly simulated), rest/early
# (schedule-exact per game). Per-game MC prob = mean across sims; week 1 equals
# the model prob by construction, late-season honestly widens with state drift.
from nfl_qb_elo import CLIP as QB_CLIP  # noqa: E402

# outcome-conditional per-game stat distributions (2020-2025): pass/run EPA sums
w_p, w_r, l_p, l_r, cnt_p, cnt_r = [], [], [], [], [], []
for g in games:
    if g["season"] < 2020 or g["y"] == 0.5:
        continue
    tm = aggc.get(g["gid"])
    if not tm:
        continue
    for t_, won in ((g["home"], g["y"] == 1.0), (g["away"], g["y"] == 0.0)):
        pS, pN, rS, rN = tm.get(t_, (0.0, 0, 0.0, 0))
        if pN < 10:
            continue
        (w_p if won else l_p).append(pS)
        (w_r if won else l_r).append(rS)
        cnt_p.append(pN); cnt_r.append(rN)
MU_WP, SD_WP = float(np.mean(w_p)), float(np.std(w_p))
MU_WR, SD_WR = float(np.mean(w_r)), float(np.std(w_r))
MU_LP, SD_LP = float(np.mean(l_p)), float(np.std(l_p))
MU_LR, SD_LR = float(np.mean(l_r)), float(np.std(l_r))
N_P, N_R = float(np.mean(cnt_p)), float(np.mean(cnt_r))
print(f"[{time.time()-T0:.0f}s] sim stat dists: win pass {MU_WP:+.1f}±{SD_WP:.1f} "
      f"run {MU_WR:+.1f}±{SD_WR:.1f} | lose pass {MU_LP:+.1f} run {MU_LR:+.1f} "
      f"| plays {N_P:.0f}/{N_R:.0f}", flush=True)

DIVS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAMS = [t for ts in DIVS.values() for t in ts]
tix = {t: i for i, t in enumerate(TEAMS)}
S = 20000
rng = np.random.default_rng(20260723)
co_ = CLF.coef_[0]
DYN = (0, 1, 2, 4, 12, 13)          # elo, qb, epa, thfa, pass, run evolve in-sim
z_frozen = CLF.decision_function(Xs) - sum(co_[j] * Xs[:, j] for j in DYN)
L400 = np.log(10.0) / 400.0

def m32(fn):                         # S x 32 state matrix from a per-team scalar
    return np.tile(np.array([fn(t) for t in TEAMS], float), (S, 1))
Rm = m32(lambda t: R.get(t, 1500.0))
h0 = m32(lambda t: hfa_st.get(t, (0.0, 0.0))[0]); h1 = m32(lambda t: hfa_st.get(t, (0.0, 0.0))[1])
eo_s = m32(lambda t: epa_off[t][0]); eo_n = m32(lambda t: epa_off[t][1])
ed_s = m32(lambda t: epa_def[t][0]); ed_n = m32(lambda t: epa_def[t][1])
po_s = m32(lambda t: offP[t][0]); po_n = m32(lambda t: offP[t][1])
pd_s = m32(lambda t: dfaP[t][0]); pd_n = m32(lambda t: dfaP[t][1])
ro_s = m32(lambda t: offR[t][0]); ro_n = m32(lambda t: offR[t][1])
rd_s = m32(lambda t: dfaR[t][0]); rd_n = m32(lambda t: dfaR[t][1])
q0 = m32(lambda t: (mQB.Q.get(QB26[t]) or [0.0, 0.0])[0])
q1 = m32(lambda t: (mQB.Q.get(QB26[t]) or [0.0, 0.0])[1])
repv = np.array([rep_map.get(QB26[t], mQB.rep) for t in TEAMS])
QDEC, QPDB, QBETA, QLG = SP["decay"], SP["prior_db"], 1.0, lg_qb

def qb_adj_col(t_i):
    q = (q0[:, t_i] + QPDB * repv[t_i]) / (q1[:, t_i] + QPDB)
    return np.clip(QBETA * (q - QLG), -QB_CLIP, QB_CLIP)

W = np.zeros((S, 32))
pmc = np.zeros(len(sched))
for i, s in enumerate(sched):
    h, a = tix[s["home"]], tix[s["away"]]
    hfa_pts = 0.0 if s["neutral"] else bp["hfa"]
    lgt = L400 * (Rm[:, h] + hfa_pts - Rm[:, a])
    f_qb = qb_adj_col(h) - qb_adj_col(a)
    er_oh = (eo_s[:, h] + pn_e * LG_EPA) / (eo_n[:, h] + pn_e)
    er_dh = (ed_s[:, h] + pn_e * LG_EPA) / (ed_n[:, h] + pn_e)
    er_oa = (eo_s[:, a] + pn_e * LG_EPA) / (eo_n[:, a] + pn_e)
    er_da = (ed_s[:, a] + pn_e * LG_EPA) / (ed_n[:, a] + pn_e)
    f_epa = (er_oh - er_dh) - (er_oa - er_da)
    cp = pn_ / 2
    f_pass = ((po_s[:, h] + cp * LGP) / (po_n[:, h] + cp) - (pd_s[:, h] + cp * LGP) / (pd_n[:, h] + cp)) \
           - ((po_s[:, a] + cp * LGP) / (po_n[:, a] + cp) - (pd_s[:, a] + cp * LGP) / (pd_n[:, a] + cp))
    f_run = ((ro_s[:, h] + cp * LGR) / (ro_n[:, h] + cp) - (rd_s[:, h] + cp * LGR) / (rd_n[:, h] + cp)) \
          - ((ro_s[:, a] + cp * LGR) / (ro_n[:, a] + cp) - (rd_s[:, a] + cp * LGR) / (rd_n[:, a] + cp))
    f_hfa = 0.0 if s["neutral"] else h0[:, h] / (h1[:, h] + 180.0)
    z = (z_frozen[i] + co_[0] * lgt + co_[1] * f_qb + co_[2] * f_epa
         + co_[4] * f_hfa + co_[12] * f_pass + co_[13] * f_run)
    p = 1.0 / (1.0 + np.exp(-z))
    pmc[i] = p.mean()
    if s["hs"] is not None:                      # played: condition on the result
        o = np.full(S, 1.0 if s["hs"] > s["as"] else (0.0 if s["hs"] < s["as"] else 0.5))
    else:
        o = (rng.random(S) < p).astype(float)
    # Elo + team-HFA updates (outcome-driven, exact walk equations)
    pe_s = 1.0 / (1.0 + np.exp(-lgt))
    d_ = bp["k"] * (o - pe_s)
    Rm[:, h] += d_; Rm[:, a] -= d_
    if not s["neutral"]:
        h0[:, h] = 0.991 * h0[:, h] + (o - pe_s)
        h1[:, h] = 0.991 * h1[:, h] + 1.0
    # game stats sampled conditional on the outcome -> EPA/pass/run/QB walks
    for t_i, win_mask in ((h, o), (a, 1.0 - o)):
        pS = np.where(win_mask >= 0.5, rng.normal(MU_WP, SD_WP, S), rng.normal(MU_LP, SD_LP, S))
        rS = np.where(win_mask >= 0.5, rng.normal(MU_WR, SD_WR, S), rng.normal(MU_LR, SD_LR, S))
        opp = a if t_i == h else h
        eo_s[:, t_i] = d_e * eo_s[:, t_i] + (pS + rS); eo_n[:, t_i] = d_e * eo_n[:, t_i] + (N_P + N_R)
        ed_s[:, opp] = d_e * ed_s[:, opp] + (pS + rS); ed_n[:, opp] = d_e * ed_n[:, opp] + (N_P + N_R)
        po_s[:, t_i] = dec_ * po_s[:, t_i] + pS; po_n[:, t_i] = dec_ * po_n[:, t_i] + N_P
        pd_s[:, opp] = dec_ * pd_s[:, opp] + pS; pd_n[:, opp] = dec_ * pd_n[:, opp] + N_P
        ro_s[:, t_i] = dec_ * ro_s[:, t_i] + rS; ro_n[:, t_i] = dec_ * ro_n[:, t_i] + N_R
        rd_s[:, opp] = dec_ * rd_s[:, opp] + rS; rd_n[:, opp] = dec_ * rd_n[:, opp] + N_R
        q0[:, t_i] = QDEC * q0[:, t_i] + pS
        q1[:, t_i] = QDEC * q1[:, t_i] + N_P
    W[:, h] += o; W[:, a] += 1.0 - o
for s, pm in zip(sched, pmc):
    s["pmc"] = round(float(pm), 3)
assert abs(sched[0]["pmc"] - sched[0]["ph"]) < 0.005, "week-1 MC must match the model prob"
jit = rng.random((S, 32)) * 1e-3                 # random tiebreaks
Wj = W + jit
div_win = np.zeros(32); po = np.zeros(32)
div_ix = {d: [tix[t] for t in ts] for d, ts in DIVS.items()}
winners = {}
for d, ixs in div_ix.items():
    wsub = Wj[:, ixs]
    win = np.array(ixs)[wsub.argmax(1)]
    winners[d] = win
    for t_i in ixs:
        div_win[t_i] += (win == t_i).mean()
for conf in ("AFC", "NFC"):
    divs_c = [d for d in DIVS if d.startswith(conf)]
    conf_ix = [i for d in divs_c for i in div_ix[d]]
    winmat = np.stack([winners[d] for d in divs_c], 1)      # S x 4 division winners
    is_winner = np.zeros((S, 32), bool)
    for c in range(4):
        is_winner[np.arange(S), winmat[:, c]] = True
    for t_i in conf_ix:
        po[t_i] += is_winner[:, t_i].mean()
    wc = Wj[:, conf_ix] + np.where(is_winner[:, conf_ix], -1000.0, 0.0)
    order = np.argsort(-wc, 1)[:, :3]
    for c in range(3):
        pick = np.array(conf_ix)[order[:, c]]
        for t_i in conf_ix:
            po[t_i] += (pick == t_i).mean()
proj = {}
for t, i in tix.items():
    proj[t] = {"w": round(float(W[:, i].mean()), 1),
               "sd": round(float(W[:, i].std()), 1),
               "div": round(float(div_win[i]) * 100, 1),
               "po": round(float(po[i]) * 100, 1)}
print(f"[{time.time()-T0:.0f}s] sims done; top proj:",
      sorted(proj.items(), key=lambda kv: -kv[1]["w"])[:5], flush=True)

# ================= payload =================
payload = json.load(open("site/data/nfl.json"))
payload["status"] = "season"
payload["schedule"] = [{k: s[k] for k in ("w", "d", "t", "home", "away", "neutral",
                                          "ph", "pmc", "hs", "as", "ct",
                                          "hrest", "arest")} for s in sched]
payload["proj"] = proj
QBN = {t: v["name"] for t, v in json.load(open("data/nfl_qb2026.json"))["qb"].items()}
for t, tm in payload["teams"].items():
    P = [payload["players"][pid] for pid in tm["roster"] if pid in payload["players"]]
    num = den = 0.0
    noff = doff = 0.0; nof = ndf = 0.0
    for p_ in P:
        rt = p_.get("rating")
        if not rt:
            continue
        wgt = max(p_.get("snap_share") or 0.0, 0.05)
        num += wgt * rt["r"]; den += wgt
        if rt["bucket"] in ("QB", "RB", "WR", "TE", "OL", "SK"):
            noff += wgt * rt["r"]; nof += wgt
        else:
            doff += wgt * rt["r"]; ndf += wgt
    tm["glassbox"] = round(num / den, 1) if den else None
    tm["gb_off"] = round(noff / nof, 1) if nof else None
    tm["gb_def"] = round(doff / ndf, 1) if ndf else None
    tm["proj"] = proj.get(t)
    qid = QB26.get(t)
    tm["qb1"] = {"id": qid, "name": QBN.get(t, "")} if qid else None
payload["model_card"]["ratings_model"] = {
    "test_ll": 0.63211, "acc": 64.6, "elo_ll": 0.63824,
    "note": "single-feature model from player ratings alone beats team Elo",
}
payload["model_card"]["serve"] = {
    "fit": "all completed games through 2025, recency half-life 3 seasons, C=100",
    "test_repro_ll": round(repro, 5), "sims": S,
    "qb_source": "nflverse depth charts 2026-07-23",
    "generated": "2026-07-23",
}
json.dump(payload, open("site/data/nfl.json", "w"))
json.dump({"repro_ll": round(repro, 5), "mean_home": round(float(probs.mean()), 4),
           "proj": proj, "coef": [round(float(c), 5) for c in CLF.coef_[0]],
           "intercept": round(float(CLF.intercept_[0]), 5)},
          open("data/nfl_season_2026.json", "w"), indent=1)
print(f"[{time.time()-T0:.0f}s] wrote site/data/nfl.json + data/nfl_season_2026.json")
