"""Breakthrough program 2026-09-24 -- NFL gap anatomy, step 3 (drill-downs).

DEV ONLY (2006-2015). Re-uses the frame of phase0/bt_nfl_anatomy.py (exec of its
frame/slice section; same TEST-free loaders and asserts), then adds:
  A  playoff-state proxies (conference-level clinch / seed-locked / exact-ish
     elimination) for weeks 14-17 -> the resting-starter channel
  B  margin information by era and phase (is the DEV margin gap era-stable?)
  C  offseason information in weeks 1-4 (QB change, new HC, prior-season
     point-differential vs record, draft capital)
  D  QB situation split by phase / experience
  E  a non-overlapping GAP BUDGET: ordered mechanism buckets with paired CIs
  F  outcome-only (market-blind) learnability of the top buckets
Odds are an evaluation benchmark only. Output data/bt_nfl_anatomy2.json.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np

src = open("phase0/bt_nfl_anatomy.py", encoding="utf-8").read()
exec(src.split("def team_diff(key")[0])  # noqa: S102

CONF = {t: ("AFC" if d[0] == "A" else "NFC") for t, d in DIV.items()}

# ---------------------------------------------------------------- A. playoff-state proxies
# standings BEFORE each game, recomputed with conference context
stand = defaultdict(lambda: [0.0, 0])            # (team, season) -> [wins, played]
pstate = {}
for r in raw_all:
    s, wk, gt, gid = int(r["season"]), int(r["week"]), r["game_type"], r["game_id"]
    if gt == "REG":
        out = {}
        for team in (r["home_team"], r["away_team"]):
            Wt, Pt = stand[(team, s)]
            remt = 16 - Pt
            conf = [u for u in CONF if CONF[u] == CONF[team] and u != team]
            rivals = [u for u in conf if DIV[u] == DIV[team]]
            mx = lambda u: stand[(u, s)][0] + (16 - stand[(u, s)][1])
            clinch_div = Wt > max(mx(u) for u in rivals)
            clinch_top1 = all(mx(u) < Wt for u in conf)
            threats_bye = sum(1 for u in conf if mx(u) >= Wt)
            clinch_bye = clinch_div and threats_bye <= 1
            # other divisions' current leaders
            leaders = []
            for dv in {DIV[u] for u in conf if DIV[u] != DIV[team]}:
                mem = [u for u in conf if DIV[u] == dv]
                leaders.append(max(mem, key=lambda u: stand[(u, s)][0]))
            up = any(stand[(u, s)][0] >= Wt and Wt + remt >= stand[(u, s)][0] for u in leaders)
            down = any(stand[(u, s)][0] <= Wt and mx(u) >= Wt for u in leaders)
            locked = clinch_top1 or (clinch_div and not up and not down)
            elim = sum(1 for u in conf if stand[(u, s)][0] > Wt + remt) >= 6
            playoff_in = sum(1 for u in conf if mx(u) >= Wt) <= 5
            out[team] = dict(clinch_div=int(clinch_div), top1=int(clinch_top1), bye=int(clinch_bye),
                             locked=int(locked), elim=int(elim), playoff_in=int(playoff_in),
                             W=Wt, P=Pt)
        pstate[gid] = out
    if gt == "REG":
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        for team, m_ in ((r["home_team"], hs - as_), (r["away_team"], as_ - hs)):
            sd_ = stand[(team, s)]
            sd_[1] += 1
            sd_[0] += 1.0 if m_ > 0 else (0.5 if m_ == 0 else 0.0)

HT = [raw[g["gid"]]["home_team"] for g in G]; AT = [raw[g["gid"]]["away_team"] for g in G]
def ps(i, side, key):
    st_ = pstate.get(G[i]["gid"])
    if st_ is None:
        return 0
    return st_[HT[i] if side == "h" else AT[i]][key]

OUT = {}
A = []
for key, lab in (("locked", "seed locked (clinched #1, or div clinched & cannot move)"),
                 ("bye", "clinched div + bye-safe"), ("clinch_div", "clinched division"),
                 ("playoff_in", "clinched playoff spot (record proxy)"),
                 ("elim", "eliminated (6 conf teams already out of reach)")):
    hk = np.array([ps(i, "h", key) for i in range(n)], bool)
    ak = np.array([ps(i, "a", key) for i in range(n)], bool)
    for wlo, whi in ((14, 15), (16, 16), (17, 17), (16, 17)):
        cond = REG & (W >= wlo) & (W <= whi)
        one = (hk ^ ak) & cond
        orient = np.where(hk, 1.0, -1.0)
        res = summarize(one, f"wk{wlo}-{whi}: one team {lab}", orient)
        res["n_both"] = int((hk & ak & cond).sum())
        A.append(res)
OUT["A_playoff_state"] = A
lock_h = np.array([ps(i, "h", "locked") for i in range(n)], bool)
lock_a = np.array([ps(i, "a", "locked") for i in range(n)], bool)
elim_h = np.array([ps(i, "h", "elim") for i in range(n)], bool)
elim_a = np.array([ps(i, "a", "elim") for i in range(n)], bool)
LATE_LOCK = REG & (W >= 16) & (lock_h ^ lock_a)

# ---------------------------------------------------------------- B. margin info by era / phase
B = []
if "pd_res" in dir():
    q1, q2 = np.nanpercentile(pd_res, [20, 80])
    ext = okpd & ((pd_res <= q1) | (pd_res >= q2))
    orient_pd = np.where(pd_res >= 0, 1.0, -1.0)   # orient to the side margin favours beyond model
    for lab, cond in (("2006-2012", SEA <= 2012), ("2013-2015", SEA >= 2013),
                      ("weeks 4-8", REG & (W >= 4) & (W <= 8)), ("weeks 9-13", REG & (W >= 9) & (W <= 13)),
                      ("weeks 14-17", REG & (W >= 14)), ("playoffs", ~REG)):
        B.append(summarize(ext & cond, f"PD-residual extreme 40%: {lab}", orient_pd))
        B.append(summarize(okpd & ~ext & cond, f"PD-residual middle 60%: {lab}", orient_pd))
    # per-season: correlation of PD residual with outcome residual (y - p_model) and with r
    per = []
    for s in range(2006, 2016):
        mm = okpd & (SEA == s)
        per.append({"season": s, "n": int(mm.sum()),
                    "corr_pdres_outcome_resid": round(float(np.corrcoef(pd_res[mm], (y - p)[mm])[0, 1]), 4),
                    "corr_pdres_r": round(float(np.corrcoef(pd_res[mm], r_dis[mm])[0, 1]), 4)})
    OUT["B_margin_per_season"] = per
OUT["B_margin"] = B

# ---------------------------------------------------------------- C. offseason info, weeks 1-4
picks_val = defaultdict(float)
for rr_ in csv.DictReader(open("data/nfl_draft_picks.csv", encoding="utf-8")):
    s_ = int(rr_["season"])
    if s_ > 2015 or not rr_["pick"]:
        continue
    picks_val[(rr_["team"], s_)] += math.exp(-0.023 * (int(rr_["pick"]) - 1))
TM_FIX = {"SDG": "SD", "STL": "STL", "RAM": "STL", "NWE": "NE", "GNB": "GB", "KAN": "KC", "NOR": "NO",
          "SFO": "SF", "TAM": "TB", "LVR": "OAK", "RAI": "OAK", "CLT": "IND", "HTX": "HOU", "OTI": "TEN",
          "CRD": "ARI", "RAV": "BAL", "LAR": "STL", "LAC": "SD"}
pv = defaultdict(float)
for (t_, s_), v in picks_val.items():
    pv[(TM_FIX.get(t_, t_), s_)] += v
# prior-season PD/g and win pct per team (from team_hist built in the frame)
prior = {}
for t_, H in team_hist.items():
    by = defaultdict(list)
    for g_ in H:
        by[g_["season"]].append(g_["margin"])
    for s_, ms in by.items():
        prior[(t_, s_ + 1)] = (sum(ms) / len(ms), sum(1 for m_ in ms if m_ > 0) / len(ms))
wk14 = REG & (W <= 4)
def diffv(f):
    return np.array([f(HT[i], G[i]["s"]) - f(AT[i], G[i]["s"]) for i in range(n)], float)
pv_d = diffv(lambda t_, s_: pv.get((t_, s_), np.nan))
ppd_d = diffv(lambda t_, s_: prior.get((t_, s_), (np.nan, np.nan))[0])
pwp_d = diffv(lambda t_, s_: prior.get((t_, s_), (np.nan, np.nan))[1])
# pythag gap: prior PD/g beyond what prior win% implies (lucky/unlucky record)
okp = wk14 & ~np.isnan(ppd_d) & ~np.isnan(pwp_d)
bb = np.linalg.lstsq(np.column_stack([np.ones(okp.sum()), pwp_d[okp]]), ppd_d[okp], rcond=None)[0]
pyth_d = np.full(n, np.nan); pyth_d[okp] = ppd_d[okp] - (bb[0] + bb[1] * pwp_d[okp])
C = []
OFFV = {"draft capital this season (diff)": pv_d, "prior-season PD/g (diff)": ppd_d,
        "prior-season win% (diff)": pwp_d, "prior PD/g beyond record (pythag, diff)": pyth_d,
        "QB != last season primary (diff)": np.array(
            [G[i]["h"]["offseason_qb_change"] - G[i]["a"]["offseason_qb_change"] for i in range(n)], float),
        "new head coach (diff)": np.array([G[i]["h"]["new_coach"] - G[i]["a"]["new_coach"] for i in range(n)], float)}
for nm, x in OFFV.items():
    xx = np.where(wk14, x, np.nan)
    a1 = ols_t(xx, r_dis)
    ok_ = ~np.isnan(xx)
    w_, se_ = offset_logit(xx[ok_], logit(p[ok_]), y[ok_])
    C.append({"var": nm, "n": int(ok_.sum()), "r_on_v_slope": round(a1[0], 4), "r_on_v_t": round(a1[1], 2),
              "corr_r": round(a1[3], 4), "outcome_offset_coef": round(float(w_[1]), 4),
              "outcome_offset_t": round(float(w_[1] / se_), 2)})
# joint R2 of r within weeks 1-4 on the offseason block
Mo = np.column_stack([np.nan_to_num(np.where(wk14, x, 0.0)) for x in OFFV.values()])[wk14]
Aj = np.column_stack([np.ones(Mo.shape[0]), Mo])
bj = np.linalg.lstsq(Aj, r_dis[wk14], rcond=None)[0]
ej = r_dis[wk14] - Aj @ bj
OUT["C_offseason_wk1_4"] = {"vars": C, "joint_R2_of_r": round(float(1 - ej.var() / r_dis[wk14].var()), 4),
                            "n": int(wk14.sum()), "wk1_4_gap": summarize(wk14, "weeks 1-4")}

# ---------------------------------------------------------------- D. QB situation by phase
D = []
nonest = lambda t: t["is_estab"] == 0
for lab, cond in (("weeks 1-4", REG & (W <= 4)), ("weeks 5-17", REG & (W >= 5)), ("playoffs", ~REG)):
    D.append(team_slice(lambda t, g, sd: nonest(t) and t["qb_starts"] < 8, f"non-estab QB <8 starts, {lab}", cond=cond))
    D.append(team_slice(lambda t, g, sd: nonest(t) and t["qb_starts"] >= 8, f"non-estab QB >=8 starts, {lab}", cond=cond))
D.append(team_slice(lambda t, g, sd: t["qb_changed"] == 1 and t["returning"] == 0 and t["n_played"] >= 1,
                    "in-season QB switch (differs from last game, not returning)", cond=REG & (W >= 2)))
D.append(team_slice(lambda t, g, sd: nonest(t) and t["qb_changed"] == 0, "non-estab QB, SAME as last game (2nd+ start of a new stint)"))
OUT["D_qb"] = D

# ---------------------------------------------------------------- E. gap budget (non-overlapping, ordered)
qb_any = np.array([(g["h"]["is_estab"] == 0) or (g["a"]["is_estab"] == 0) for g in G])
qb_switch = np.array([(g["h"]["qb_changed"] == 1) or (g["a"]["qb_changed"] == 1) for g in G])
ext_pd = okpd & ((pd_res <= np.nanpercentile(pd_res, 20)) | (pd_res >= np.nanpercentile(pd_res, 80)))
buckets = [("late-season motivation: wk16-17, exactly one team seed-locked", LATE_LOCK),
           ("offseason: weeks 1-2", REG & (W <= 2)),
           ("QB: either starter non-established or changed from last game", qb_any | qb_switch),
           ("margin: season PD/g far from model (extreme 40%)", ext_pd),
           ("playoffs", ~REG),
           ("short week / off bye (either)", (hr <= 5) | (ar <= 5) | (hr >= 13) | (ar >= 13)),
           ("primetime", kick >= 19.0)]
taken = np.zeros(n, bool)
E = []
for lab, msk in buckets:
    m_ = msk & ~taken
    res = summarize(m_, lab)
    E.append(res)
    taken |= m_
E.append(summarize(~taken, "remainder (none of the above)"))
OUT["E_gap_budget"] = {"total_gap": round(float(d.mean()), 5), "total_nats": round(float(d.sum()), 3),
                       "buckets": E}

# ---------------------------------------------------------------- F. market-blind learnability
# can the OUTCOMES alone (no odds) recover the late-lock effect? LOSO offset logistic on a
# signed indicator (+1 home locked, -1 away locked) restricted to weeks 16-17
lock_sig = np.where(REG & (W >= 16), lock_h.astype(float) - lock_a.astype(float), 0.0)
g_lock = loso_gain(lock_sig)
w_l, se_l = offset_logit(lock_sig, logit(p), y)
elim_sig = np.where(REG & (W >= 14), elim_h.astype(float) - elim_a.astype(float), 0.0)
g_elim = loso_gain(elim_sig)
w_e, se_e = offset_logit(elim_sig, logit(p), y)
OUT["F_learnability"] = {
    "late_lock_signed": {"n_nonzero": int((lock_sig != 0).sum()), "coef": round(float(w_l[1]), 4),
                         "t": round(float(w_l[1] / se_l), 2), "loso_gain_all_games": round(g_lock[0], 5),
                         "loso_ci": g_lock[1],
                         "market_mean_r_toward_locked": round(float((lock_sig * r_dis)[lock_sig != 0].mean()), 4)},
    "elim_signed_wk14plus": {"n_nonzero": int((elim_sig != 0).sum()), "coef": round(float(w_e[1]), 4),
                             "t": round(float(w_e[1] / se_e), 2), "loso_gain_all_games": round(g_elim[0], 5),
                             "loso_ci": g_elim[1]}}
# top-5% disagreement composition by budget bucket
top = np.abs(r_dis) >= np.percentile(np.abs(r_dis), 95)
comp = {}
taken = np.zeros(n, bool)
for lab, msk in buckets:
    m_ = msk & ~taken
    comp[lab] = {"share_of_top5": round(float(m_[top].mean()), 3), "share_of_all": round(float(m_.mean()), 3)}
    taken |= m_
OUT["top5_composition"] = comp

# direction profile: games where the model is much higher on HOME than the close
dn = r_dis <= -0.4
prof = {"n": int(dn.sum())}
for c_ in ("lgt", "qd", "epa", "thfa", "rest", "luck", "pass", "run"):
    xc = np.array([g["X"][c_] for g in G])
    prof[c_] = {"mean_bucket": round(float(xc[dn].mean()), 4), "mean_all": round(float(xc.mean()), 4)}
prof["home_nonestab_share"] = round(float(np.mean([G[i]["h"]["is_estab"] == 0 for i in np.where(dn)[0]])), 3)
prof["home_nonestab_share_all"] = round(float(np.mean([g["h"]["is_estab"] == 0 for g in G])), 3)
prof["home_locked_late_share"] = round(float((lock_h & REG & (W >= 16))[dn].mean()), 3)
prof["home_locked_late_share_all"] = round(float((lock_h & REG & (W >= 16)).mean()), 3)
OUT["model_much_higher_on_home_profile"] = prof

# primetime: sharper market or information?
OUT["primetime_check"] = {
    "primetime": summarize(kick >= 19.0, "primetime"),
    "primetime_excl_budget_1_4": summarize((kick >= 19.0) & ~(LATE_LOCK | (REG & (W <= 2)) | qb_any | qb_switch | ext_pd),
                                           "primetime, none of buckets 1-4"),
    "nonprime_excl_budget_1_4": summarize((kick < 19.0) & ~(LATE_LOCK | (REG & (W <= 2)) | qb_any | qb_switch | ext_pd),
                                          "non-primetime, none of buckets 1-4")}

json.dump(OUT, open("data/bt_nfl_anatomy2.json", "w"), indent=1)
print("wrote data/bt_nfl_anatomy2.json")
