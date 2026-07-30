"""Betting-queue rank 5: does the devig method matter for the MLB two-way market?

Three ways to strip vig from a 2-way price (home decimal h, away decimal a,
raw implied qh=1/h, qa=1/a, overround V = qh+qa-1):

  multiplicative  qh/(qh+qa)          — the incumbent everywhere in this repo
  additive        qh - V/2            — each outcome sheds half the vig
  shin            insurance-weighted  — Shin (1993): vig concentrates on
                                        longshots (favorite-longshot bias)

If additive/Shin fit outcomes better, our close-line benchmark (the "market
LL" every model chases) and the Hubacek q-weights are slightly mis-set.
Evaluation-layer only — no model input, no one-look cost.

Out: data/devig_method_test.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "phase0")
from ev_gate_audit import load  # noqa: E402

EPS = 1e-9


def shin_probs(qh, qa):
    """Shin devig for 2 outcomes: p_i = (sqrt(z^2+4(1-z)q_i^2/B)-z)/(2(1-z)),
    z found by bisection so probabilities sum to 1. B = qh+qa (booksum)."""
    B = qh + qa
    if B <= 1.0:                       # no overround: nothing to strip
        return qh / B

    def total(z):
        s = 0.0
        for q in (qh, qa):
            s += (np.sqrt(z * z + 4 * (1 - z) * q * q / B) - z) / (2 * (1 - z))
        return s

    lo, hi = 0.0, 0.2
    if total(hi) > 1.0:                # extreme overround: cap z
        hi = 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if total(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    return (np.sqrt(z * z + 4 * (1 - z) * qh * qh / B) - z) / (2 * (1 - z))


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def devig_all(rows, hkey, akey):
    y = np.array([r["y"] for r in rows], dtype=float)
    out = {}
    mult, add, shin = [], [], []
    for r in rows:
        qh, qa = 1.0 / r[hkey], 1.0 / r[akey]
        V = qh + qa - 1.0
        mult.append(qh / (qh + qa))
        add.append(min(max(qh - V / 2, EPS), 1 - EPS))
        shin.append(shin_probs(qh, qa))
    for name, p in (("multiplicative", mult), ("additive", add), ("shin", shin)):
        p = np.array(p)
        out[name] = {"ll": round(ll(y, p), 6)}
        # favorite-side calibration: does the method over/under-rate favorites?
        fav = p > 0.5
        out[name]["fav_pred"] = round(float(p[fav].mean()), 5)
        out[name]["fav_real"] = round(float(y[fav].mean()), 5)
        out[name]["fav_gap"] = round(float(p[fav].mean() - y[fav].mean()), 5)
    return out, len(rows)


def main():
    rows = load()
    res = {}
    for tag, hk, ak in (("open", "ho", "ao"), ("close", "hc", "ac")):
        res[tag], n = devig_all(rows, hk, ak)
        res[tag]["n"] = n
    # paired significance: per-game LL difference additive vs multiplicative (close)
    y = np.array([r["y"] for r in rows], dtype=float)
    pm, pa = [], []
    for r in rows:
        qh, qa = 1.0 / r["hc"], 1.0 / r["ac"]
        V = qh + qa - 1.0
        pm.append(qh / (qh + qa))
        pa.append(min(max(qh - V / 2, EPS), 1 - EPS))
    pm, pa = np.clip(np.array(pm), EPS, 1 - EPS), np.clip(np.array(pa), EPS, 1 - EPS)
    dm = -(y * np.log(pm) + (1 - y) * np.log(1 - pm))
    da = -(y * np.log(pa) + (1 - y) * np.log(1 - pa))
    diff = dm - da                         # >0 means additive better
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    res["paired_add_minus_mult_close"] = {
        "mean_delta_ll": round(float(diff.mean()), 7),
        "z": round(float(diff.mean() / se), 2) if se > 0 else 0.0,
    }
    json.dump(res, open("data/devig_method_test.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
