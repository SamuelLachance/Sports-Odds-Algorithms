"""Test the Hubacek DECORRELATION loss: does forcing distance from the opening line
improve betting CLV/yield, or just wreck accuracy?

Refit the no-lineup blend [1, lf, bpz, siz] -> sigmoid on a TRAIN split by minimizing
    mean[(p - y)^2]  -  c * mean[(p - m)^2]        (m = no-vig opening market prob)
i.e. accuracy minus c*distance-from-market (the paper's modified loss, c in [0.4,0.8]
optimal, c=1 destroys it). Sweep c, evaluate on a held-out TEST split: log loss,
Pearson corr(model, market), flat betting yield and CLV vs the opening line.

Train 2016-2019, test 2021 (2020 COVID-short skipped; 2021 is beyond the base model's
2016-24 selection window). Odds = evaluation only.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

ODDS = "data/odds_mlb.csv"
PROBS = "data/model_probs.csv"
EPS = 1e-12


def load(train_years, test_years):
    odds = defaultdict(list)
    for r in csv.DictReader(open(ODDS)):
        odds[(r["date"], r["home"], r["away"])].append(r)
    used = defaultdict(int)
    tr, te = [], []
    mp = [r for r in csv.DictReader(open(PROBS))]
    for r in sorted(mp, key=lambda r: (r["date"], r["home"], r["away"], r["dh"])):
        s = int(r["season"])
        if s not in train_years and s not in test_years:
            continue
        k = (r["date"], r["home"], r["away"])
        lst = odds.get(k, [])
        if used[k] >= len(lst):
            continue
        o = lst[used[k]]; used[k] += 1
        ho, ao = float(o["home_open"]), float(o["away_open"])
        m = (1 / ho) / (1 / ho + 1 / ao)          # no-vig opening market prob (home)
        rec = dict(x=[1.0, float(r["lf"]), float(r["bpz"]), float(r["siz"])],
                   y=int(r["y"]), m=m, ho=ho, ao=ao,
                   hc=float(o["home_close"]), ac=float(o["away_close"]))
        (tr if s in train_years else te).append(rec)
    return tr, te


def probs(beta, X):
    z = X @ beta
    return 1.0 / (1.0 + np.exp(-z))


def fit(tr, c):
    X = np.array([r["x"] for r in tr]); y = np.array([r["y"] for r in tr], float)
    m = np.array([r["m"] for r in tr])

    def loss(b):
        p = probs(b, X)
        return np.mean((p - y) ** 2) - c * np.mean((p - m) ** 2)
    b0 = np.zeros(X.shape[1])
    res = minimize(loss, b0, method="L-BFGS-B")
    return res.x


def boot_ci(x, n=10000, seed=7):
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def evaluate(beta, te, phi=0.10):
    X = np.array([r["x"] for r in te])
    p = np.clip(probs(beta, X), EPS, 1 - EPS)
    y = np.array([r["y"] for r in te], float)
    m = np.array([r["m"] for r in te])
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    rho = float(np.corrcoef(p, m)[0, 1])
    flat, clv = [], []
    for pi, r in zip(p, te):
        if abs(pi - 0.5) < phi:
            continue
        for o_open, o_close, ps, win in [
                (r["ho"], r["hc"], pi, r["y"] == 1),
                (r["ao"], r["ac"], 1 - pi, r["y"] == 0)]:
            if ps > 1.0 / o_open:
                flat.append((o_open - 1) if win else -1.0)
                clv.append(1.0 / o_close - 1.0 / o_open)
                break
    ymean, ylo, yhi = boot_ci(flat)
    return dict(ll=ll, rho=rho, n=len(flat), y=ymean, ylo=ylo, yhi=yhi,
                clv=float(np.mean(clv)) if clv else 0.0)


def run(train_years, test_years, tag):
    tr, te = load(set(train_years), set(test_years))
    print(f"\n=== {tag}: train {sorted(train_years)} ({len(tr)}g) -> test {sorted(test_years)} ({len(te)}g), phi=0.10 ===")
    print(f"{'c':>5} {'testLL':>8} {'corr':>6} {'bets':>5} {'yield/bet':>10} {'95% CI':>18} {'CLV(pp)':>8}")
    for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        e = evaluate(fit(tr, c), te)
        sig = "PROFIT" if e["ylo"] > 0 else ("LOSS" if e["yhi"] < 0 else "n.s.")
        print(f"{c:>5.1f} {e['ll']:>8.4f} {e['rho']:>6.3f} {e['n']:>5} {e['y']*100:>+9.2f}% "
              f"[{e['ylo']*100:>+6.2f},{e['yhi']*100:>+6.2f}] {e['clv']*100:>+7.2f} {sig}")


def main():
    run(range(2016, 2019), [2019], "SPLIT A")
    run(range(2016, 2020), [2021], "SPLIT B")
    run([2016, 2017, 2018], [2019, 2021], "SPLIT C (pooled test)")


if __name__ == "__main__":
    main()
