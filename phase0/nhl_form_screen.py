"""DEV screen round 3: the model's FORM, not its inputs.

Rounds 1 and 2 added or re-measured nine signals and every one came back flat.
Together with ledger rows 6/8/9/10 that is thirteen nulls, all of the same shape:
another number bolted onto a linear-in-logit blend. So this round changes the
blend instead, leaving the inputs exactly as they ship.

  NONLIN      curvature in the Elo logit (squared and cubed terms). ~23% of NHL
              games reach overtime, where the result is much closer to a coin
              flip than team strength implies. A linear-in-logit blend can only
              rescale the whole curve; it cannot compress the tails, so if OT/SO
              flattens strong favourites the model should be systematically
              overconfident out there and curvature should pay.

  EARLY       interaction of the Elo logit with how much the ratings actually
              know: games played so far this season, after the pre-season
              regression toward the mean. In October both ratings are freshly
              shrunk and equally uninformative, yet the blend trusts them as
              much as it does in April.

  OT_RATE     trailing share of each team's games that reached overtime. Some
              teams genuinely play low-event, close games; their opponents' win
              probabilities should sit nearer 0.5 for reasons no strength
              rating captures.

  DENSITY     games in the last 5 days, per side — fatigue deeper than the
              back-to-back flag already in the model.

  TRIP        length of the current homestand / road trip. Rest and b2b measure
              the gap between games, not the accumulated cost of being away.

Same protocol: DEV only, leave-one-season-out over 2011-12..2017-18, the same
pre-registered +0.00100 bar with a CI excluding zero, at most ONE TEST look.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict, deque
from datetime import date

import numpy as np

sys.path.insert(0, "phase0")
from nhl_xg_screen import BAR_DELTA, load_adj, loso, margin_rating  # noqa: E402
from nhl_features_eval import build_features  # noqa: E402
from nhl_glicko2_eval import (DEV_END, DEV_WARM_BEFORE, llv,  # noqa: E402
                              load_games, run_elo)


def context(games):
    """Per-game schedule context, all leak-safe (state read before it updates)."""
    n = len(games)
    gp = np.zeros(n)          # min games played this season, scaled 0..1
    dens = np.zeros(n)        # home minus away games in the last 5 days
    trip = np.zeros(n)        # home homestand length minus away road-trip length
    played = defaultdict(int)
    recent = defaultdict(lambda: deque())
    streak = defaultdict(int)   # + = consecutive home games, - = consecutive away
    prev = None
    for i, g in enumerate(games):
        d = date.fromisoformat(g["date"])
        if prev is not None and g["season"] != prev:
            played.clear()
            recent.clear()
            streak.clear()
        prev = g["season"]
        h, a = g["home"], g["away"]
        gp[i] = min(played[h], played[a]) / 40.0
        for t in (h, a):
            while recent[t] and (d - recent[t][0]).days > 5:
                recent[t].popleft()
        dens[i] = len(recent[h]) - len(recent[a])
        trip[i] = min(max(streak[h], 0), 6) - min(max(-streak[a], 0), 6)
        played[h] += 1
        played[a] += 1
        recent[h].append(d)
        recent[a].append(d)
        streak[h] = streak[h] + 1 if streak[h] > 0 else 1
        streak[a] = streak[a] - 1 if streak[a] < 0 else -1
    return gp, dens, trip


def ot_rate(games, window=41):
    """Trailing share of each team's games that reached OT, home minus away."""
    n = len(games)
    out = np.zeros(n)
    hist = defaultdict(lambda: deque(maxlen=window))
    for i, g in enumerate(games):
        h, a = g["home"], g["away"]
        rh = float(np.mean(hist[h])) if len(hist[h]) >= 10 else 0.23
        ra = float(np.mean(hist[a])) if len(hist[a]) >= 10 else 0.23
        out[i] = rh + ra - 0.46          # both teams' tendency toward close games
        ot = 1.0 if g.get("ot") else 0.0
        hist[h].append(ot)
        hist[a].append(ot)
    return out


def main():
    games = load_games()
    last = {}
    for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")):
        last[int(r["game_id"])] = r["last_period"]
    for g in games:
        g["ot"] = last.get(g["game_id"], "REG") != "REG"

    rep = json.load(open("data/nhl_glicko2_report.json"))
    out = run_elo(games, **rep["elo_cfg"])
    p = np.clip(np.array([o[1] for o in out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    F = build_features(games)
    xg = margin_rating(games, load_adj(), "raw")
    gp, dens, trip = context(games)
    otr = ot_rate(games)

    cands = {
        "SHIPPED": [],
        "NONLIN": [elogit ** 2, elogit ** 3],
        "EARLY": [elogit * gp, gp],
        "OT_RATE": [otr, elogit * otr],
        "DENSITY": [dens],
        "TRIP": [trip],
        "ALL": [elogit ** 2, elogit ** 3, elogit * gp, gp, otr, elogit * otr,
                dens, trip],
    }

    ok = ~np.isnan(xg)
    ds = sorted({int(s) for s in seas if DEV_WARM_BEFORE <= s <= DEV_END})
    m = ok & np.isin(seas, ds)
    print(f"DEV screen round 3: {int(m.sum())} games, {len(ds)} folds")
    print(f"  OT/SO share of DEV games: {np.mean([g['ot'] for g in games])[()]:.1%}\n"
          if False else
          f"  OT/SO share of games: {float(np.mean([1.0 if g['ot'] else 0.0 for g in games])):.1%}\n")

    base = [F["rest_diff"], F["b2b_home"], F["b2b_away"], xg]
    preds, lls = {}, {}
    for name, cols in cands.items():
        X = np.column_stack([elogit] + base + cols)[m]
        preds[name] = loso(X, y[m], seas[m], ds)
        lls[name] = float(llv(y[m], preds[name]).mean())

    ref = preds["SHIPPED"]
    rng = np.random.default_rng(11)
    print(f"{'candidate':<10} {'DEV LL':>9} {'vs SHIPPED':>11}  95% CI            verdict")
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
        print(f"{name:<10} {lls[name]:>9.5f} {delta:>+11.5f}  "
              f"[{lo:+.5f},{hi:+.5f}]  {verdict}")

    win = [n for n, r in rows.items() if r["clears_bar"] and n != "SHIPPED"]
    best = min(win, key=lambda n: rows[n]["dev_ll"]) if win else None
    print(f"\nbar: delta >= {BAR_DELTA:+.5f} AND CI excludes zero")
    print("TEST look earned by: " + (best or "NOBODY - no TEST look taken"))
    json.dump({"n_dev": int(m.sum()), "folds": ds, "bar": BAR_DELTA, "rows": rows,
               "earns_test_look": best},
              open("data/nhl_form_screen.json", "w"), indent=1)
    print("wrote data/nhl_form_screen.json")


if __name__ == "__main__":
    main()
