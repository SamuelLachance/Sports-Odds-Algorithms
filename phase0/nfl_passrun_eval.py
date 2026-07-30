"""Pass/run split ratings + mismatch — the last untested structural idea.

Channels from duel-file protagonists (passer/receiver -> pass; rusher-only -> run).
Per team: off-pass, off-run, def-pass-allowed, def-run-allowed EPA/play EWMAs
(same dynamics as the shipped aggregate EPA feature, channel league rates from DEV).

Variants (DEV CV selects, ONE TEST look, keep if better):
  split-replace  f_epa -> [pass_net, run_net]
  split-add      current + both nets
  mismatch       + interaction terms (pass-off strength x opp pass-def weakness, both sides)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

# ---- per-game-team channel aggregates from the duel file ----
aggc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))  # gid->team->[pS,pN,rS,rN]
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        t_ = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
        epa = float(r[ix["epa"]])
        is_pass = bool(r[ix["passer_player_id"]] or r[ix["receiver_player_id"]])
        a = aggc[r[ix["game_id"]]][t_]
        if is_pass:
            a[0] += epa; a[1] += 1
        else:
            a[2] += epa; a[3] += 1

def lg_ch():
    ps = pn = rs = rn = 0.0
    for gid, tm in aggc.items():
        if int(gid[:4]) in DEV_YEARS:
            for t_, (a, b, c, d) in tm.items():
                ps += a; pn += b; rs += c; rn += d
    return ps / pn, rs / rn
LGP, LGR = lg_ch()
print(f"DEV league EPA/play: pass {LGP:+.4f}  run {LGR:+.4f}")

dec, pn_, sd = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
offP = defaultdict(lambda: [0.0, 0.0]); offR = defaultdict(lambda: [0.0, 0.0])
dfaP = defaultdict(lambda: [0.0, 0.0]); dfaR = defaultdict(lambda: [0.0, 0.0])
def rate(st_, lg_):
    return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)

f_pass = np.zeros(len(games)); f_run = np.zeros(len(games))
f_miP = np.zeros(len(games)); f_miR = np.zeros(len(games))
prev = None
SDP = 0.09; SDR = 0.07          # rough z-scales for interactions (EPA/play spread)
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for st_ in (list(offP.values()) + list(offR.values())
                    + list(dfaP.values()) + list(dfaR.values())):
            st_[0] *= (1 - sd); st_[1] *= (1 - sd)
    prev = g["season"]
    h, a = g["home"], g["away"]
    ophh, opha = rate(offP[h], LGP), rate(offP[a], LGP)
    orhh, orha = rate(offR[h], LGR), rate(offR[a], LGR)
    dphh, dpha = rate(dfaP[h], LGP), rate(dfaP[a], LGP)
    drhh, drha = rate(dfaR[h], LGR), rate(dfaR[a], LGR)
    f_pass[i] = (ophh - dphh) - (opha - dpha)
    f_run[i] = (orhh - drhh) - (orha - drha)
    # mismatch: my pass-off strength x their pass-def weakness (z-ish), minus mirror
    f_miP[i] = ((ophh - LGP) / SDP) * ((dpha - LGP) / SDP) \
        - ((opha - LGP) / SDP) * ((dphh - LGP) / SDP)
    f_miR[i] = ((orhh - LGR) / SDR) * ((drha - LGR) / SDR) \
        - ((orha - LGR) / SDR) * ((drhh - LGR) / SDR)
    tm = aggc.get(g["gid"])
    if tm:
        for t_off, opp in ((h, a), (a, h)):
            pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
            o = offP[t_off]; o[0] = dec * o[0] + pS; o[1] = dec * o[1] + pN
            o = offR[t_off]; o[0] = dec * o[0] + rS; o[1] = dec * o[1] + rN
            d_ = dfaP[opp]; d_[0] = dec * d_[0] + pS; d_[1] = dec * d_[1] + pN
            d_ = dfaR[opp]; d_[0] = dec * d_[0] + rS; d_[1] = dec * d_[1] + rN

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

X_cur = X_of(F)
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")
F_rep = dict(F)
X_norep = np.column_stack([F["lgt"], F["qd"], f_pass, f_run, F["early"], F["thfa"],
                           F["rest"], F["luck"], F["tse"], F["ol"], F["de"],
                           F["sk"], F["rq"]])
CAND = {
    "split-replace": X_norep,
    "split-add": np.column_stack([X_cur, f_pass, f_run]),
    "mismatch-add": np.column_stack([X_cur, f_miP, f_miR]),
    "split+mismatch": np.column_stack([X_cur, f_pass, f_run, f_miP, f_miR]),
}
best = None
for name, X in CAND.items():
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, name, X); tag = "  <-- best"
    print(f"  {name:<16} DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, name, Xn = best
if s >= base_cv:
    print("\nno DEV improvement -> rejected at tryouts, no TEST look")
    json.dump({"verdict": "dev_rejected", "dev_cv_base": round(base_cv, 5),
               "dev_cv": round(s, 5), "variant": name},
              open("data/nfl_passrun.json", "w"), indent=1)
    raise SystemExit(0)

fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(Xn[fitm], y[fitm])
p0 = c0.predict_proba(X_cur[test])[:, 1]
p1 = c1.predict_proba(Xn[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"\nmain TEST ({name}): current {llv(yt, p0).mean():.5f} -> {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"variant": name, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_passrun.json", "w"), indent=1)
