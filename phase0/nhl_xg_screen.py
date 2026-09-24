"""DEV screen: does a better-MEASURED xG rating beat the raw one?

Pre-registered in documents/nhl_xg_measurement_prereg_2026_08_11.md. Read that
first — the bar, the candidate list and the falsification condition were fixed
before this ran.

DEV ONLY. Leave-one-season-out across DEV seasons 2011-12..2017-18: for each DEV
season the blend is fit on the other DEV seasons and scored on the held-out one,
then the held-out predictions are pooled. TEST is never touched here; the single
TEST look, if any candidate earns one, is taken separately and gets a ledger row.

Candidates all replace the xg feature inside the shipped 4-feature blend (Elo
logit + rest + b2b + xG). Everything else is held fixed, so a difference is
attributable to the xG measurement and nothing else.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "phase0")
from nhl_features_eval import (XG_HA, XG_K, XG_REGRESS, TEAM_FIX,  # noqa: E402
                               build_features)
from nhl_glicko2_eval import (DEV_END, DEV_WARM_BEFORE, llv,  # noqa: E402
                              load_games, run_elo)

FLAVOURS = ("raw", "adj", "ev", "adjev")
BAR_DELTA = 0.00100          # pre-registered; see the prereg document


def load_adj(path="data/nhl_team_xg_adj.csv"):
    """game_id -> {team: {flavour: (xgf, xga)}}."""
    out = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        t = TEAM_FIX.get(r["team"], r["team"])
        out[int(r["nhl_game_id"])][t] = {
            f: (float(r[f"xgf_{f}"]), float(r[f"xga_{f}"])) for f in FLAVOURS}
    return out


def margin_rating(games, adj, flavour):
    """The shipped construction, on a chosen xG flavour: one rating per team,
    Elo-style, updated on the error in expected goal MARGIN."""
    n = len(games)
    feat = np.full(n, np.nan)
    R = defaultdict(float)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] *= (1 - XG_REGRESS)
        prev = g["season"]
        h, a = g["home"], g["away"]
        feat[i] = (R[h] + XG_HA) - R[a]
        tx = adj.get(g["game_id"])
        if not tx or h not in tx or a not in tx:
            feat[i] = np.nan
            continue
        hxgf = tx[h][flavour][0]
        axgf = tx[a][flavour][0]
        err = (hxgf - axgf) - (R[h] - R[a] + XG_HA)
        R[h] += XG_K / 100.0 * err
        R[a] -= XG_K / 100.0 * err
    return feat


def split_rating(games, adj, flavour):
    """Separate offence and defence ratings: a team that generates 3.0 and allows
    2.5 is not the same team as one that generates 2.0 and allows 1.5, though the
    margin rating scores them identically. Returns the home side's offensive edge
    and defensive edge as two features."""
    n = len(games)
    off_edge = np.full(n, np.nan)
    def_edge = np.full(n, np.nan)
    O, D = defaultdict(float), defaultdict(float)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in O:
                O[t] *= (1 - XG_REGRESS)
            for t in D:
                D[t] *= (1 - XG_REGRESS)
        prev = g["season"]
        h, a = g["home"], g["away"]
        off_edge[i] = (O[h] - D[a]) + XG_HA / 2
        def_edge[i] = (D[h] - O[a]) - XG_HA / 2
        tx = adj.get(g["game_id"])
        if not tx or h not in tx or a not in tx:
            off_edge[i] = def_edge[i] = np.nan
            continue
        hxgf = tx[h][flavour][0]
        axgf = tx[a][flavour][0]
        # home offence vs away defence produced hxgf; away offence vs home
        # defence produced axgf. Split each error evenly between the two sides.
        e_h = hxgf - (O[h] - D[a] + XG_HA / 2)
        e_a = axgf - (O[a] - D[h] - XG_HA / 2)
        k = XG_K / 100.0 / 2
        O[h] += k * e_h
        D[a] -= k * e_h
        O[a] += k * e_a
        D[h] -= k * e_a
    return off_edge, def_edge


def loso(X, y, seas, dev_seasons):
    """Leave-one-season-out predictions pooled over DEV. Returns (p, index)."""
    from sklearn.linear_model import LogisticRegression
    p = np.full(len(y), np.nan)
    for s in dev_seasons:
        tr = np.isin(seas, [d for d in dev_seasons if d != s])
        te = seas == s
        clf = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def main():
    games = load_games()
    rep = json.load(open("data/nhl_glicko2_report.json"))
    e_out = run_elo(games, **rep["elo_cfg"])
    pe = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(pe / (1 - pe))
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    F = build_features(games)
    adj = load_adj()

    cands = {}
    for f in FLAVOURS:
        cands[f.upper()] = [margin_rating(games, adj, f)]
    cands["SPLIT"] = list(split_rating(games, adj, "raw"))
    cands["SPLITADJ"] = list(split_rating(games, adj, "adj"))

    # ONE mask for every candidate so the comparison is paired game-for-game.
    ok = np.ones(len(games), bool)
    for cols in cands.values():
        for c in cols:
            ok &= ~np.isnan(c)
    dev_seasons = sorted({int(s) for s in seas
                          if DEV_WARM_BEFORE <= s <= DEV_END})
    m = ok & np.isin(seas, dev_seasons)
    print(f"DEV screen: {int(m.sum())} games over seasons "
          f"{dev_seasons[0]}..{dev_seasons[-1]}  ({len(dev_seasons)} folds)\n")

    base_cols = [F["rest_diff"], F["b2b_home"], F["b2b_away"]]
    out, preds = {}, {}
    for name, cols in cands.items():
        X = np.column_stack([elogit] + base_cols + cols)[m]
        p = loso(X, y[m], seas[m], dev_seasons)
        preds[name] = p
        out[name] = float(llv(y[m], p).mean())

    ref = preds["RAW"]
    rng = np.random.default_rng(11)
    print(f"{'candidate':<10} {'DEV LL':>9} {'vs RAW':>10}  95% CI            verdict")
    rows = {}
    for name in cands:
        d = llv(y[m], ref) - llv(y[m], preds[name])
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
        delta = float(d.mean())
        clears = delta >= BAR_DELTA and lo > 0
        verdict = ("CLEARS BAR" if clears
                   else "SIG but under bar" if lo > 0
                   else "SIG WORSE" if hi < 0 else "n.s.")
        rows[name] = {"dev_ll": round(out[name], 5), "delta_vs_raw": round(delta, 5),
                      "ci": [round(lo, 5), round(hi, 5)], "clears_bar": bool(clears),
                      "verdict": verdict}
        print(f"{name:<10} {out[name]:>9.5f} {delta:>+10.5f}  "
              f"[{lo:+.5f},{hi:+.5f}]  {verdict}")

    winners = [n for n, r in rows.items() if r["clears_bar"] and n != "RAW"]
    best = min(winners, key=lambda n: rows[n]["dev_ll"]) if winners else None
    print(f"\nbar: delta >= {BAR_DELTA:+.5f} AND CI excludes zero")
    print("TEST look earned by: " + (best if best else
          "NOBODY - channel screened and dropped, no TEST look taken"))
    json.dump({"n_dev": int(m.sum()), "folds": dev_seasons, "bar": BAR_DELTA,
               "rows": rows, "earns_test_look": best},
              open("data/nhl_xg_screen.json", "w"), indent=1)
    print("wrote data/nhl_xg_screen.json")


if __name__ == "__main__":
    main()
