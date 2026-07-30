"""Hubacek-style betting backtest of the market-blind model vs OPENING odds.

Pipeline (matches the paper + Samuel's diagram):
  model prob (at open, no lineup)  x  opening moneyline
     -> value bet where p > 1/o and |p-.5| > phi   (E[R]=p(o-1)-(1-p))
     -> per-round allocation: flat, or Markowitz/Sharpe (f_i propto E_i/Var_i,
        Var_i=p(1-p)o^2), remainder to CASH (0% risk-free)
     -> realized profit per round, and CLV audit: did the OPENING price we took
        beat the CLOSING price? (closing implied prob < our taken implied prob)

Odds are EVALUATION only; the model never sees them. Reports flat yield (per unit
staked), Markowitz %/round, hit rate, value-bet count, and CLV -- with bootstrap CIs.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

ODDS = "data/odds_mlb.csv"
PROBS = "data/model_probs.csv"


def load(prob_col="recbp_p", years=range(2016, 2022)):
    odds = defaultdict(list)
    for r in csv.DictReader(open(ODDS)):
        odds[(r["date"], r["home"], r["away"])].append(r)
    used = defaultdict(int)
    rows = []
    mp = [r for r in csv.DictReader(open(PROBS)) if int(r["season"]) in years]
    for r in sorted(mp, key=lambda r: (r["date"], r["home"], r["away"], r["dh"])):
        k = (r["date"], r["home"], r["away"])
        lst = odds.get(k, [])
        if used[k] >= len(lst) or r[prob_col] == "":
            continue
        o = lst[used[k]]
        used[k] += 1
        rows.append({
            "date": r["date"], "season": int(r["season"]), "p": float(r[prob_col]),
            "ho": float(o["home_open"]), "ao": float(o["away_open"]),
            "hc": float(o["home_close"]), "ac": float(o["away_close"]),
            "y": int(r["y"]),
        })
    return rows


def value_bet(g, phi, min_edge):
    """Return (side, dec_open, dec_close, win, p_side) for the positive-EV side at the
    opening price, or None. side: 1=home, 0=away."""
    p = g["p"]
    if abs(p - 0.5) < phi:
        return None
    cands = []
    # home
    ev_h = p * (g["ho"] - 1) - (1 - p)
    if ev_h > min_edge:
        cands.append((ev_h, 1, g["ho"], g["hc"], g["y"] == 1, p))
    ev_a = (1 - p) * (g["ao"] - 1) - p
    if ev_a > min_edge:
        cands.append((ev_a, 0, g["ao"], g["ac"], g["y"] == 0, 1 - p))
    if not cands:
        return None
    ev, side, o_open, o_close, win, p_side = max(cands)   # never both sides; take best EV
    return dict(side=side, o_open=o_open, o_close=o_close, win=win, p=p_side, ev=ev)


def boot_ci(x, n=10000, seed=7):
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def backtest(rows, phi=0.0, min_edge=0.0, gamma=2.0, label=""):
    bets = []
    by_round = defaultdict(list)
    for g in rows:
        b = value_bet(g, phi, min_edge)
        if b:
            b["date"] = g["date"]
            bets.append(b)
            by_round[g["date"]].append(b)
    if not bets:
        print(f"  [{label}] no value bets"); return
    # flat yield: 1 unit per bet, profit = (o-1) if win else -1
    flat = [(b["o_open"] - 1) if b["win"] else -1.0 for b in bets]
    fy, flo, fhi = boot_ci(flat)
    hit = np.mean([b["win"] for b in bets])
    # CLV: taken opening price vs closing price for the SAME side
    clv_beat = np.mean([1.0 if b["o_open"] > b["o_close"] else 0.0 for b in bets])
    clv_prob = np.mean([1.0 / b["o_close"] - 1.0 / b["o_open"] for b in bets])  # implied-prob move toward us
    # Markowitz per round: f_i propto E_i/Var_i, budget 1/round, remainder cash
    round_ret = []
    for d, bs in by_round.items():
        w = []
        for b in bs:
            o, p = b["o_open"], b["p"]
            var = p * (1 - p) * o * o
            ev = p * (o - 1) - (1 - p)
            w.append(max(ev, 0.0) / (gamma * var) if var > 0 else 0.0)
        s = sum(w)
        if s > 1:                      # cap exposure at the 1-unit round budget
            w = [x / s for x in w]
        ret = sum(wi * ((b["o_open"] - 1) if b["win"] else -1.0) for wi, b in zip(w, bs))
        round_ret.append(ret)
    mk, mlo, mhi = boot_ci(round_ret)
    n_bets, n_rounds = len(bets), len(by_round)
    frac_games = n_bets / len(rows)
    print(f"  [{label}] bets={n_bets} ({frac_games*100:.0f}% of games), rounds={n_rounds}, hit={hit*100:.1f}%")
    print(f"      flat yield/bet : {fy*100:+.2f}%  CI[{flo*100:+.2f},{fhi*100:+.2f}]  "
          f"{'PROFIT' if flo>0 else 'n.s.' if fhi>0 else 'LOSS'}")
    print(f"      Markowitz/round: {mk*100:+.2f}%  CI[{mlo*100:+.2f},{mhi*100:+.2f}]")
    print(f"      CLV: beat close {clv_beat*100:.1f}% of bets, mean CLV {clv_prob*100:+.2f} prob-pts "
          f"({'positive' if clv_prob>0 else 'negative'})")


def main():
    prob_col = sys.argv[1] if len(sys.argv) > 1 else "recbp_p"
    rows = load(prob_col)
    print(f"=== Hubacek backtest vs OPENING odds | model={prob_col} | 2016-2021 | n={len(rows)} joined games ===")
    # reference: how often does the model even disagree enough to bet, at various gates
    for phi, me in [(0.0, 0.0), (0.0, 0.02), (0.10, 0.0), (0.10, 0.02), (0.15, 0.02)]:
        backtest(rows, phi=phi, min_edge=me, label=f"phi={phi} min_edge={me}")
    print("--- clean 2021 only (beyond the model's 2016-24 selection window) ---")
    r21 = [g for g in rows if g["season"] == 2021]
    for phi, me in [(0.0, 0.0), (0.10, 0.02)]:
        backtest(r21, phi=phi, min_edge=me, label=f"2021 phi={phi} min_edge={me}")


if __name__ == "__main__":
    main()
