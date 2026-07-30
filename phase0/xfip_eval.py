"""Is xFIP a better pitcher core than FIP, and/or does it add over 0.67431?

xFIP swaps a starter's actual HR for expected HR = (fly balls) x league HR/FB.
Since league total expected-HR == league total HR by construction, xFIP sits on
the same scale as FIP, so the same FipPitcherElo (params + league constant) rates
it. We build BOTH pitcher-Elo variants and:
  A) head-to-head: recalibrated holdout LL of FIP-Elo vs xFIP-Elo (better core?)
  B) increment: add (logit_xFIP - logit_FIP) -- the HR-normalisation signal --
     over the current full model [logit_fip + ts_z + bp_z + power_z] (=0.67431).
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]


def load_fb():
    fb = {}
    for r in csv.DictReader(open("data/retro_events/flyballs.csv")):
        fb[(r["game_id"], r["pitcher"])] = int(r["fb"])
    return fb


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def recal(lf, y, dm, tm):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lf[dm].reshape(-1, 1), y[dm])
    return m.predict_proba(lf[tm].reshape(-1, 1))[:, 1]


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    games = load_games()
    fb = load_fb()

    # league HR/FB over starter starts (FB_total = fly balls + HR)
    tot_hr = tot_fb = 0
    for g in games:
        for sp, ln in ((g["home_sp"], g["home_line"]), (g["away_sp"], g["away_line"])):
            if ln:
                hr = ln[1]
                tot_hr += hr
                tot_fb += fb.get((g["game_id"], sp), 0) + hr
    lg_hrfb = tot_hr / tot_fb
    print(f"league HR/FB = {lg_hrfb:.4f}  (starts with a line drive the model)")

    # build xFIP lines: expHR = (fb + HR) * lg_hrfb, replacing HR
    def xline(gid, sp, ln):
        if not ln:
            return None
        outs, hr, bb, hbp, so = ln
        exp_hr = (fb.get((gid, sp), 0) + hr) * lg_hrfb
        return (outs, exp_hr, bb, hbp, so)

    lgf = league_fip_core(games)
    mf = FipPitcherElo(lg_fip=lgf, **PARAMS)      # FIP
    mx = FipPitcherElo(lg_fip=lgf, **PARAMS)      # xFIP
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lg_iso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, FZ.load_lineups(), lg_iso)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            mf.new_season(); mx.new_season()
        prev = g["season"]
        pf = mf.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        px = mx.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        rows.append((g["season"], pf, px, g["y"], g["game_id"]))
        mf.update(g)
        gx = dict(g, home_line=xline(g["game_id"], g["home_sp"], g["home_line"]),
                  away_line=xline(g["game_id"], g["away_sp"], g["away_line"]))
        mx.update(gx)

    arr = {r[4]: r for r in rows}
    seasons = np.array([r[0] for r in rows])
    pf = np.array([r[1] for r in rows]); px = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows], float)
    dm = np.isin(seasons, list(DEV)); tm = np.isin(seasons, TEST)
    yt = y[tm]

    # A) head-to-head recalibrated
    p_fip = recal(logit(pf), y, dm, tm)
    p_xfip = recal(logit(px), y, dm, tm)
    print(f"\n=== A) HEAD-TO-HEAD  n={int(tm.sum()):,}  (recalibrated pitcher core) ===")
    print(f"  FIP-Elo   LL={ll(p_fip, yt):.5f}")
    print(f"  xFIP-Elo  LL={ll(p_xfip, yt):.5f}")
    d, lo, hi = boot(ll_vec(p_fip, yt), ll_vec(p_xfip, yt))
    print(f"  FIP minus xFIP: {d:+.5f}  CI[{lo:+.5f},{hi:+.5f}]  "
          f"{'xFIP BETTER' if lo > 0 else ('FIP BETTER' if hi < 0 else 'tie (n.s.)')}")

    # B) increment over the current full model
    j = [(s, pf_i, px_i, yy, ts[gid], bpd[gid], pdiff.get(gid))
         for s, pf_i, px_i, yy, gid in rows if gid in ts and pdiff.get(gid) is not None]
    seas = np.array([x[0] for x in j])
    lf = logit(np.array([x[1] for x in j])); lx = logit(np.array([x[2] for x in j]))
    yb = np.array([x[3] for x in j], float)
    tse = np.array([x[4] for x in j]); bpe = np.array([x[5] for x in j]); pwe = np.array([x[6] for x in j])
    dmb = np.isin(seas, list(DEV)); tmb = np.isin(seas, TEST)

    def z(v):
        return (v - v[dmb].mean()) / (v[dmb].std() or 1.0)
    tsz, bpz, pwz, xz = z(tse), z(bpe), z(pwe), z(lx - lf)
    ybt = yb[tmb]
    base_cols = [lf, tsz, bpz, pwz]
    base = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(base_cols)[dmb], yb[dmb])
    p_base = base.predict_proba(np.column_stack(base_cols)[tmb])[:, 1]
    plus = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack(base_cols + [xz])[dmb], yb[dmb])
    p_plus = plus.predict_proba(np.column_stack(base_cols + [xz])[tmb])[:, 1]
    d2, lo2, hi2 = boot(ll_vec(p_base, ybt), ll_vec(p_plus, ybt))
    print(f"\n=== B) INCREMENT  n={int(tmb.sum()):,}  base [fip+ts+bp+power] LL={ll(p_base, ybt):.5f} (=0.67431) ===")
    print(f"  + (logit_xFIP - logit_FIP)  LL={ll(p_plus, ybt):.5f}  d={d2:+.5f} CI[{lo2:+.5f},{hi2:+.5f}]  "
          f"{'HELPS' if lo2 > 0 else ('HURTS' if hi2 < 0 else 'n.s.')}  coef={plus.coef_[0][-1]:+.4f}")


if __name__ == "__main__":
    main()
