"""Clone audit (Hsu-papers takeaway #1): pick-level kappa/overlap diagnostics.

Measures whether the 14-feature GLASSBOX model makes different PICKS than
(a) tuned team Elo alone, (b) the ratings-only v7 TrueSkill model, and
(c) the de-vigged closing moneyline — and decomposes the LL gap-to-close
into 'we disagree and lose' vs 'we agree and both miss'.

Pure post-hoc diagnostic: odds are EVALUATION ONLY (constitution intact),
no DEV look consumed, no model change.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)

# ---- shipped 14-col matrix (passrun cols + v7 ratings col) ----
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
X14[:, 7] = np.load("data/nfl_v7_feature.npy")

# ---- main-model walk-forward TEST probs (the adopted trainer) ----
HL, BC = 3.0, 100.0
p_main = np.full(len(games), np.nan)
for s_ in sorted(np.unique(seasons[test])):
    tr = (seasons < s_) & (y != 0.5)
    te = test & (seasons == s_)
    w_ = 0.5 ** ((s_ - 1 - seasons[tr]) / HL)
    m_ = LogisticRegression(C=BC, max_iter=5000).fit(X14[tr], y[tr], sample_weight=w_)
    p_main[te] = m_.predict_proba(X14[te])[:, 1]
ll_main = float(llv(y[test], p_main[test]).mean())
print(f"[{time.time()-T0:.0f}s] main model TEST LL {ll_main:.5f} (repro check vs 0.61919)", flush=True)
assert abs(ll_main - 0.61919) < 0.0005, "main TEST repro drifted"

# ---- ratings-only probs (single v7 column, its own protocol: train 2016+, eval 2022-25) ----
v7col = X14[:, 7:8]
p_rat = np.full(len(games), np.nan)
for s_ in range(2022, 2026):
    tr = (seasons >= 2016) & (seasons < s_) & (y != 0.5)
    te = seasons == s_
    m_ = LogisticRegression(C=1e6, max_iter=3000).fit(v7col[tr], y[tr])
    p_rat[te] = m_.predict_proba(v7col[te])[:, 1]

# ---- closing moneylines (chronological join, no-vig) ----
raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
def imp(ml):
    if not ml:
        return None
    v = float(ml)
    return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))
mkt = np.full(len(games), np.nan)
for i, r in enumerate(raw):
    ph_, pa_ = imp(r["home_moneyline"]), imp(r["away_moneyline"])
    if ph_ and pa_:
        mkt[i] = ph_ / (ph_ + pa_)

# ---- Elo probs ----
p_elo = pe  # tuned 1999+ Elo walk from the prelude

def kappa2(a, b):
    po = float((a == b).mean())
    pa1, pb1 = float(a.mean()), float(b.mean())
    pe_ = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe_) / (1 - pe_) if pe_ < 1 else 1.0

def pair_audit(p1, p2, mask, n1, n2):
    """Pick-level agreement on games in mask (ties excluded from accuracy)."""
    m = mask & ~np.isnan(p1) & ~np.isnan(p2)
    mm = m & (y != 0.5)
    a1, a2 = p1[mm] > 0.5, p2[mm] > 0.5
    yy = y[mm] > 0.5
    agree = a1 == a2
    out = {
        "pair": f"{n1} vs {n2}", "n": int(mm.sum()),
        "agreement": round(float(agree.mean()), 4),
        "kappa": round(kappa2(a1, a2), 4),
        f"acc_{n1}": round(float((a1 == yy).mean()), 4),
        f"acc_{n2}": round(float((a2 == yy).mean()), 4),
        "n_discord": int((~agree).sum()),
        "acc_concord": round(float((a1[agree] == yy[agree]).mean()), 4),
        f"discord_won_by_{n1}": round(float((a1[~agree] == yy[~agree]).mean()), 4) if (~agree).any() else None,
    }
    print(f"  {out['pair']:<22} n={out['n']:<5} agree {out['agreement']:.1%}  "
          f"kappa {out['kappa']:.3f}  acc {out[f'acc_{n1}']:.4f}/{out[f'acc_{n2}']:.4f}  "
          f"discord n={out['n_discord']}  {n1} wins discords {out[f'discord_won_by_{n1}']}", flush=True)
    return out

print(f"\nCLONE AUDIT — panel A: main TEST 2016-2025 (n={int(test.sum())})", flush=True)
res = {"main_test_ll": round(ll_main, 5)}
res["main_vs_elo"] = pair_audit(p_main, p_elo, test, "main", "elo")
res["main_vs_close"] = pair_audit(p_main, mkt, test, "main", "close")
res["elo_vs_close"] = pair_audit(p_elo, mkt, test, "elo", "close")

m22 = test & (seasons >= 2022)
print(f"\npanel B: 2022-2025 (ratings-only protocol TEST, n={int(m22.sum())})", flush=True)
res["main_vs_ratings"] = pair_audit(p_main, p_rat, m22, "main", "ratings")
res["ratings_vs_close"] = pair_audit(p_rat, mkt, m22, "ratings", "close")

# ---- gap-to-close decomposition: agree/disagree splits (LL includes ties) ----
mc = test & ~np.isnan(mkt) & ~np.isnan(p_main)
am, ac = p_main[mc] > 0.5, mkt[mc] > 0.5
agree = am == ac
ll_m = llv(y[mc], p_main[mc]); ll_c = llv(y[mc], mkt[mc])
gap = float(ll_m.mean() - ll_c.mean())
n_all = int(mc.sum())
dec = {}
for name, sel in (("concordant", agree), ("discordant", ~agree)):
    n_ = int(sel.sum())
    dm, dc = float(ll_m[sel].mean()), float(ll_c[sel].mean())
    contrib = (dm - dc) * n_ / n_all
    dec[name] = {"n": n_, "ll_main": round(dm, 5), "ll_close": round(dc, 5),
                 "gap_here": round(dm - dc, 5), "contrib_to_total_gap": round(contrib, 5)}
    print(f"\n  {name}: n={n_}  LL main {dm:.5f} vs close {dc:.5f}  "
          f"(gap {dm-dc:+.5f}, contributes {contrib:+.5f} of total {gap:+.5f})", flush=True)
res["gap_decomposition"] = {"total_gap": round(gap, 5), **dec}
corr = float(np.corrcoef(p_main[mc], mkt[mc])[0, 1])
mad = float(np.abs(p_main[mc] - mkt[mc]).mean())
res["prob_level"] = {"corr_main_close": round(corr, 4), "mean_abs_diff": round(mad, 4)}
print(f"\n  prob-level: corr(main, close) {corr:.4f}  mean|diff| {mad:.4f}", flush=True)

# accuracy of each side WITHIN the discordant set, plus close's overall edge there
mm2 = mc & (y != 0.5)
am2, ac2 = p_main[mm2] > 0.5, mkt[mm2] > 0.5
dd = am2 != ac2
if dd.any():
    yy2 = y[mm2] > 0.5
    res["discord_detail"] = {
        "n": int(dd.sum()),
        "main_wins": int((am2[dd] == yy2[dd]).sum()),
        "close_wins": int((ac2[dd] == yy2[dd]).sum()),
    }
    print(f"  discordant picks (no ties): {res['discord_detail']['n']} games — "
          f"main right {res['discord_detail']['main_wins']}, "
          f"close right {res['discord_detail']['close_wins']}", flush=True)

json.dump(res, open("data/nfl_clone_audit.json", "w"), indent=1)
print("\nwrote data/nfl_clone_audit.json", flush=True)
