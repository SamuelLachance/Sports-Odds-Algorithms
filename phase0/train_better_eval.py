"""'Train better' -- implement all four ideas and test each with logistic + xgboost.

A. BASELINE            frozen features, binary target, uniform weights, fixed split.
B. RECENCY / REGIME    (B1) recency-weighted refit; (B2) walk-forward expanding window
                       (retrain each test season on all prior seasons -> uses 2016+ data).
C. MARGIN TARGET       train on run-margin (Skellam-style) -> map to win prob via a link.
D. RE-TUNED ENGINE     re-optimize the FIP-Elo core hyperparameters (walk-forward random
                       search on DEV), regenerate lf, refit the blend.

Features: [lf, bpz, siz, tsz, pwz, brz]. Split: DEV 2000-2015, TEST 2016-2024 ex-2020.
"""
import sys, math, csv, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from mlbwp.ingest import league_fip_core, load_games
from mlbwp.rating import FipPitcherElo
from mlbwp.train import PARAMS

FEATS = ["lf", "bpz", "siz", "tsz", "pwz", "brz"]
XGB = dict(max_depth=3, n_estimators=400, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=30,
           eval_metric="logloss", verbosity=0, n_jobs=-1)
eps = 1e-15
def logit(p): p = min(max(p, eps), 1 - eps); return math.log(p / (1 - p))
def LLv(y, p): p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

games = load_games(); lg_fip = league_fip_core(games)
def gkey(g):
    dh = g["game_id"][11:] if len(g["game_id"]) > 11 else "0"
    d = g["date"]; return (f"{d[:4]}-{d[4:6]}-{d[6:8]}", g["home"], g["away"], dh)

def compute_lf(params):
    m = FipPitcherElo(lg_fip=lg_fip, **params); out = {}; prev = None
    for g in games:
        if prev is not None and g["season"] != prev: m.new_season()
        prev = g["season"]
        out[gkey(g)] = logit(m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"]))
        m.update(g)
    return out

# margin (home-away runs) from Retrosheet game logs, keyed like model_probs
import glob
margin = {}
for fp in glob.glob("data/retrosheet/gl*.txt"):
    for r in csv.reader(open(fp)):
        if len(r) < 11: continue
        d = r[0]; iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        try: margin[(iso, r[6], r[3], r[1])] = int(r[10]) - int(r[9])
        except ValueError: pass

# base feature frame from model_probs (has bpz/siz/tsz/pwz/brz), override lf from engine
df = pd.read_csv("data/model_probs.csv", dtype={"dh": str})
df = df.dropna(subset=FEATS).reset_index(drop=True)
keys = list(zip(df.date, df.home, df.away, df.dh))
df["margin"] = [margin.get(k, np.nan) for k in keys]
df = df.dropna(subset=["margin"]).reset_index(drop=True)
keys = list(zip(df.date, df.home, df.away, df.dh))
lf0 = compute_lf(PARAMS)
df["lf"] = [lf0[k] for k in keys]
dev = (df.season <= 2015).values
test = ((df.season >= 2016) & (df.season != 2020)).values
y = df.y.values.astype(float); season = df.season.values
print(f"games {len(df)} | DEV {dev.sum()} TEST {test.sum()} | margin joined {(~df.margin.isna()).sum()}")

def run(Xtr, ytr, Xte, wtr=None):
    lr = LogisticRegression(C=1e6, max_iter=3000).fit(Xtr, ytr, sample_weight=wtr)
    xg = XGBClassifier(**XGB).fit(Xtr, ytr, sample_weight=wtr)
    return lr.predict_proba(Xte)[:, 1], xg.predict_proba(Xte)[:, 1]

def LL(p): return LLv(y[test], p).mean()
results = {}

# ---------- A. baseline ----------
X = df[FEATS].values
pl, px = run(X[dev], y[dev], X[test])
results["A. baseline"] = (LL(pl), LL(px))

# ---------- B1. recency-weighted (fixed split), lambda picked on DEV tail ----------
tr_in = dev & (season <= 2013); tr_val = dev & (season >= 2014)
best_lam, best_vll = 0.0, 9
for lam in [0.0, 0.05, 0.1, 0.2, 0.4]:
    w = np.exp(-lam * (2013 - season[tr_in]))
    lr = LogisticRegression(C=1e6, max_iter=3000).fit(X[tr_in], y[tr_in], sample_weight=w)
    vll = LLv(y[tr_val], lr.predict_proba(X[tr_val])[:, 1]).mean()
    if vll < best_vll: best_vll, best_lam = vll, lam
w = np.exp(-best_lam * (2015 - season[dev]))
pl, px = run(X[dev], y[dev], X[test], wtr=w)
results[f"B1. recency-weighted (lam={best_lam})"] = (LL(pl), LL(px))

# ---------- B2. walk-forward expanding window (retrain each test season) ----------
pl_wf = np.zeros(test.sum()); px_wf = np.zeros(test.sum())
tidx = np.where(test)[0]; pos = {i: j for j, i in enumerate(tidx)}
for s in sorted(np.unique(season[test])):
    tr = (season < s) & (season != 2020)
    te = test & (season == s)
    a, b = run(X[tr], y[tr], X[te])
    for k, gi in enumerate(np.where(te)[0]):
        pl_wf[pos[gi]] = a[k]; px_wf[pos[gi]] = b[k]
results["B2. walk-forward (expanding)"] = (LLv(y[test], pl_wf).mean(), LLv(y[test], px_wf).mean())

# ---------- C. margin target -> win prob via link ----------
mg = df.margin.values
lin = LinearRegression().fit(X[dev], mg[dev])
xgr = XGBRegressor(objective="reg:squarederror", **{k: v for k, v in XGB.items() if k != "eval_metric"}).fit(X[dev], mg[dev])
def link_probs(reg):
    md = reg.predict(X[dev]).reshape(-1, 1); mt = reg.predict(X[test]).reshape(-1, 1)
    link = LogisticRegression(C=1e6, max_iter=2000).fit(md, y[dev])
    return link.predict_proba(mt)[:, 1]
results["C. margin target"] = (LL(link_probs(lin)), LL(link_probs(xgr)))

# ---------- D. re-tuned FIP-Elo engine ----------
dev_games = [g for g in games if g["season"] <= 2015]
def dev_fip_ll(params):
    m = FipPitcherElo(lg_fip=lg_fip, **params); ps, ys, prev = [], [], None
    for g in dev_games:
        if prev is not None and g["season"] != prev: m.new_season()
        prev = g["season"]
        if g["season"] >= 2002:
            ps.append(m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])); ys.append(g["y"])
        m.update(g)
    ps = np.array(ps); ys = np.array(ys)
    return LLv(ys, ps).mean()
rng = np.random.default_rng(20260722); t0 = time.time()
best = (dev_fip_ll(PARAMS), dict(PARAMS)); base_dev = best[0]; N = 300
for i in range(N):
    cand = dict(
        k_team=float(np.exp(rng.uniform(np.log(1.0), np.log(12.0)))),
        hfa=float(rng.uniform(10, 45)), team_regress=float(rng.uniform(0.2, 0.8)),
        beta=float(rng.uniform(15, 60)), decay=float(rng.uniform(0.85, 0.99)),
        prior_ip=float(rng.uniform(20, 150)), season_decay=float(rng.uniform(0.0, 0.1)))
    s = dev_fip_ll(cand)
    if s < best[0]: best = (s, cand)
print(f"\nD: engine re-tune {N} trials in {time.time()-t0:.0f}s | DEV FIP-Elo LL {base_dev:.5f} -> {best[0]:.5f}")
lfD = compute_lf({k: v for k, v in best[1].items()})
XD = X.copy(); XD[:, 0] = np.array([lfD[k] for k in keys])
pl, px = run(XD[dev], y[dev], XD[test])
results["D. re-tuned engine"] = (LL(pl), LL(px))

# ---------- results ----------
print(f"\n{'='*60}\n{'CONFIG':<34}{'LOGISTIC':>12}{'XGBOOST':>12}\n{'-'*60}")
b_lr, b_xg = results["A. baseline"]
for name, (l, x) in results.items():
    dl = "" if name.startswith("A.") else f"  ({l-b_lr:+.5f})"
    print(f"{name:<34}{l:>12.5f}{x:>12.5f}{dl}")
print(f"{'-'*60}\nbaseline logistic {b_lr:.5f} is the number to beat; xgboost shown for contrast")
