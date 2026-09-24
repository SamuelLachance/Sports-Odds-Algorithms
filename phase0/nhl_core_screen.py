"""DEV screen round 2: the rating CORE and the context around it.

Round 1 (phase0/nhl_xg_screen.py) asked whether a better-measured xG rating
helps. It does not — score-adjusted, 5v5-only and offence/defence-split ratings
all tie the raw one, and the harness demonstrably sees the xG feature itself
(+0.00304 on DEV). Together with ledger rows 6, 8, 9 and 10, that closes the
"add a better team/player quality signal" family: the xG rating already carries
what the shot data holds.

So this round leaves quality alone and attacks what the model does NOT model:

  HFA_DRIFT   home-ice advantage as a rolling quantity, not a constant. The
              shipped Elo uses ha=30 for all 16 seasons, but NHL home-ice has
              decayed sharply over exactly this window (the shipped model's own
              home_win_rate is 0.541 pooled). The NFL model already carries a
              rolling HFA (data/nfl_rolling_hfa.json); NHL never got one.

  FINISH      finishing talent: a rating on goals-above-expected. The model is
              xG-only by design (row 5: update on expected, never on realized),
              which is right for VOLUME but discards the residual — a team that
              persistently converts above its xG is not fully priced. This is
              the hockey analog of the MLB power ratings.

  ST          special teams entered BESIDE the 5v5 rating rather than pooled
              into it. Round 1 replaced xG with 5v5-only and tied; it never
              asked whether the special-teams part carries separate signal.

  FAST_SLOW   two Elo timescales (a responsive k and a stable k) blended, rather
              than one k=8. Row 2 killed Glicko-2, but that tested a different
              UNCERTAINTY model, not a different memory length.

Same protocol as round 1: DEV only, leave-one-season-out over 2011-12..2017-18,
same +0.00100 bar with a CI excluding zero, at most ONE candidate to TEST.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from datetime import date

import numpy as np

sys.path.insert(0, "phase0")
from nhl_xg_screen import BAR_DELTA, load_adj, loso, margin_rating  # noqa: E402
from nhl_features_eval import XG_HA, XG_K, XG_REGRESS, build_features  # noqa: E402
from nhl_glicko2_eval import (DEV_END, DEV_WARM_BEFORE, llv,  # noqa: E402
                              load_games, run_elo)


def hfa_drift(games, window=400):
    """Trailing home-win rate in logits, as of each game (leak-safe: the window
    closes before the game it describes). A constant ha cannot track a league
    whose home edge moved over the sample."""
    n = len(games)
    out = np.zeros(n)
    hist = deque(maxlen=window)
    for i, g in enumerate(games):
        if len(hist) >= 100:
            r = float(np.mean(hist))
            r = min(max(r, 0.35), 0.65)
            out[i] = np.log(r / (1 - r))
        hist.append(g["y"])
    return out


def finish_rating(games, adj, flavour="raw", k=0.04, regress=0.35):
    """Elo-style rating on goals-above-expected. Positive = converts above xG."""
    n = len(games)
    feat = np.full(n, np.nan)
    R = defaultdict(float)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] *= (1 - regress)
        prev = g["season"]
        h, a = g["home"], g["away"]
        feat[i] = R[h] - R[a]
        tx = adj.get(g["game_id"])
        if not tx or h not in tx or a not in tx:
            feat[i] = np.nan
            continue
        # goals above expected, each side, this game
        gh = g["hg"] - tx[h][flavour][0]
        ga = g["ag"] - tx[a][flavour][0]
        R[h] += k * (gh - R[h])
        R[a] += k * (ga - R[a])
    return feat


def st_rating(games, adj, k=XG_K / 100.0, regress=XG_REGRESS):
    """Special-teams xG rating: the non-5v5 remainder of each team's xG, rated
    separately from the even-strength play."""
    n = len(games)
    feat = np.full(n, np.nan)
    R = defaultdict(float)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] *= (1 - regress)
        prev = g["season"]
        h, a = g["home"], g["away"]
        feat[i] = R[h] - R[a]
        tx = adj.get(g["game_id"])
        if not tx or h not in tx or a not in tx:
            feat[i] = np.nan
            continue
        hst = (tx[h]["raw"][0] - tx[h]["ev"][0]) - (tx[h]["raw"][1] - tx[h]["ev"][1])
        ast = (tx[a]["raw"][0] - tx[a]["ev"][0]) - (tx[a]["raw"][1] - tx[a]["ev"][1])
        R[h] += k * (hst - R[h])
        R[a] += k * (ast - R[a])
    return feat


def elo_logit_at(games, k, ha, regress):
    out = run_elo(games, k=k, ha=ha, regress=regress)
    p = np.clip(np.array([o[1] for o in out]), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def main():
    games = load_games()
    # run_elo's loader drops goals; re-attach them for the finishing rating
    import csv
    goals = {}
    for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")):
        goals[int(r["game_id"])] = (int(r["home_goals"]), int(r["away_goals"]))
    for g in games:
        g["hg"], g["ag"] = goals.get(g["game_id"], (0, 0))

    rep = json.load(open("data/nhl_glicko2_report.json"))
    cfg = rep["elo_cfg"]
    elogit = elo_logit_at(games, **cfg)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    F = build_features(games)
    adj = load_adj()
    xg = margin_rating(games, adj, "raw")

    fast = elo_logit_at(games, k=cfg["k"] * 2.5, ha=cfg["ha"], regress=cfg["regress"])
    slow = elo_logit_at(games, k=cfg["k"] * 0.4, ha=cfg["ha"], regress=cfg["regress"])

    cands = {
        "SHIPPED": [],
        "HFA_DRIFT": [hfa_drift(games)],
        "FINISH": [finish_rating(games, adj)],
        "ST": [st_rating(games, adj)],
        "FAST_SLOW": [fast, slow],
        "ALL": [hfa_drift(games), finish_rating(games, adj), st_rating(games, adj)],
    }

    ok = ~np.isnan(xg)
    for cols in cands.values():
        for c in cols:
            ok &= ~np.isnan(c)
    ds = sorted({int(s) for s in seas if DEV_WARM_BEFORE <= s <= DEV_END})
    m = ok & np.isin(seas, ds)
    print(f"DEV screen round 2: {int(m.sum())} games, {len(ds)} folds\n")

    base = [F["rest_diff"], F["b2b_home"], F["b2b_away"], xg]
    preds, lls = {}, {}
    for name, cols in cands.items():
        X = np.column_stack([elogit] + base + cols)[m]
        preds[name] = loso(X, y[m], seas[m], ds)
        lls[name] = float(llv(y[m], preds[name]).mean())

    ref = preds["SHIPPED"]
    rng = np.random.default_rng(11)
    print(f"{'candidate':<11} {'DEV LL':>9} {'vs SHIPPED':>11}  95% CI            verdict")
    rows = {}
    for name in cands:
        d = llv(y[m], ref) - llv(y[m], preds[name])
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
        delta = float(d.mean())
        clears = delta >= BAR_DELTA and lo > 0
        verdict = ("CLEARS BAR" if clears else "SIG but under bar" if lo > 0
                   else "SIG WORSE" if hi < 0 else "n.s.")
        rows[name] = {"dev_ll": round(lls[name], 5), "delta": round(delta, 5),
                      "ci": [round(lo, 5), round(hi, 5)], "clears_bar": bool(clears),
                      "verdict": verdict}
        print(f"{name:<11} {lls[name]:>9.5f} {delta:>+11.5f}  "
              f"[{lo:+.5f},{hi:+.5f}]  {verdict}")

    win = [n for n, r in rows.items() if r["clears_bar"] and n != "SHIPPED"]
    best = min(win, key=lambda n: rows[n]["dev_ll"]) if win else None
    print(f"\nbar: delta >= {BAR_DELTA:+.5f} AND CI excludes zero")
    print("TEST look earned by: " + (best or "NOBODY - no TEST look taken"))
    json.dump({"n_dev": int(m.sum()), "folds": ds, "bar": BAR_DELTA, "rows": rows,
               "earns_test_look": best},
              open("data/nhl_core_screen.json", "w"), indent=1)
    print("wrote data/nhl_core_screen.json")


if __name__ == "__main__":
    main()
