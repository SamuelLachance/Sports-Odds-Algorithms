"""NHL theoretical-floor estimate -- the cell error_map.py had to leave blank.

READ-ONLY. No model is touched, nothing is fitted that is ever served. It re-runs
the shipped NHL DEV harness exactly as phase0/error_map.py::nhl_league does
(nhl_depth_eval.Ctx + LOSO with the shipped Elo/xG params), joins the DEV games
to the newly sourced closing moneylines in data/odds_nhl.csv, and calls the SAME
floor_block() error_map uses for MLB and NFL, so the three numbers are directly
comparable.

ONE-LOOK PROTOCOL
  NHL DEV 2011-12 .. 2017-18, TEST >= 2018-19 never loaded, never scored.
  Hard-asserted below; no TEST-era number is computed, printed or written here.

MARKET-BLINDNESS
  Odds appear ONLY as an evaluation benchmark and as the proxy for the true
  probability in the entropy floor. They are never a feature and never a
  training signal. This is the explicitly sanctioned evaluation use.

Usage:  python -X utf8 phase0/nhl_floor.py   ->  data/nhl_floor.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

import error_map as EM  # noqa: E402

OUT = "data/nhl_floor.json"
ODDS = "data/odds_nhl.csv"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def dev_predictions():
    """Shipped NHL DEV LOSO out-of-fold probabilities + their game rows."""
    import nhl_depth_eval as Hd
    assert Hd.DEV_END < Hd.TEST_START, "NHL DEV/TEST split moved"
    ctx = Hd.Ctx()
    y = ctx.yv()
    X = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p = ctx.folds.loso_pred(X, y)
    seas = ctx.seasons[ctx.mask]
    assert seas.max() <= Hd.DEV_END < Hd.TEST_START, "TEST-era NHL game scored"
    ll = EM.llv(y, p)
    log(f"NHL DEV LOSO ll {ll.mean():.6f} n={len(y)} (recorded 0.6724)")
    assert abs(ll.mean() - 0.6724) < 5e-4, "NHL baseline not reproduced"
    G = [g for g, keep in zip(ctx.games, ctx.mask) if keep]
    return p, y, seas, G, int(Hd.DEV_END), int(Hd.TEST_START)


def odds_index():
    rows = list(csv.DictReader(open(ODDS, encoding="utf-8")))
    idx = defaultdict(list)
    for r in rows:
        if r["home_close"] and r["away_close"]:
            idx[(r["date"], r["home"], r["away"])].append(r)
    return rows, idx


def main():
    p, y, seas, G, dev_end, test_start = dev_predictions()
    rows, oidx = odds_index()

    # ---- join (date, home, away). Ambiguity is impossible in NHL (no
    # doubleheaders) but is guarded anyway, mirroring the MLB joiner.
    keep, pm, ambiguous, miss = [], [], 0, 0
    for i, g in enumerate(G):
        m = oidx.get((g["date"], g["home"], g["away"]))
        if not m:
            miss += 1
            continue
        if len(m) != 1:
            ambiguous += 1
            continue
        r = m[0]
        qh, qa = 1.0 / float(r["home_close"]), 1.0 / float(r["away_close"])
        keep.append(i)
        pm.append(qh / (qh + qa))
        # independent correctness check: SBR's own final score must agree with
        # the NHL API result for this game, or the join is wrong.
        assert int(r["home_win"]) == int(round(float(y[i]))), \
            f"result mismatch on {g['date']} {g['away']}@{g['home']}"
    idx = np.array(keep)
    pm = np.array(pm)
    s_j = seas[idx]
    assert s_j.max() <= dev_end < test_start, "market join reached a TEST-era season"

    # ---- per-season join coverage
    per = {}
    for s_ in sorted(set(seas.tolist())):
        tot = int((seas == s_).sum())
        got = int((s_j == s_).sum())
        per[str(int(s_))] = {"dev_games": tot, "joined": got,
                             "join_rate": round(got / tot, 4)}

    note = (f"data/odds_nhl.csv (SportsbookReviewsOnline NHL archive, scraped by "
            f"phase0/nhl_odds_load.py) covers 2010-11..2022-23; the DEV era is "
            f"{int(seas.min())}-{int(seas.max())} and {len(idx)} of {len(G)} DEV "
            f"games joined ({len(idx)/len(G):.4f}). {miss} unmatched, {ambiguous} "
            f"ambiguous. Every joined game's SBR final score agrees with the NHL "
            f"API result (asserted, 0 mismatches).")

    block = EM.floor_block("nhl", p[idx], pm, y[idx], note)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": "NHL theoretical-floor estimate on DEV, using the newly sourced "
                "historical closing moneyline. Completes the error_map.py floor table.",
        "protocol": {
            "read_only": True, "models_touched": 0, "test_looks_spent": 0,
            "market_blind_models": True,
            "dev_seasons": [int(seas.min()), int(seas.max())],
            "test_never_touched": [test_start, 20252026],
            "odds_use": ("EVALUATION ONLY -- benchmark + proxy for the true probability "
                         "in the entropy floor. No odds enter any feature or training "
                         "signal. Sanctioned use."),
            "harness": "phase0/nhl_depth_eval.py Ctx + LOSO, shipped Elo/xG params "
                       "(identical to phase0/error_map.py::nhl_league)",
        },
        "source": {
            "url_pattern": "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl-odds-<slug>",
            "loader": "phase0/nhl_odds_load.py",
            "file": ODDS, "rows_total": len(rows),
            "format": "season page renders the whole season as one HTML table, two "
                      "rows per game (V then H), American Open/Close moneylines",
        },
        "join": {"dev_games": len(G), "joined": len(idx),
                 "join_rate": round(len(idx) / len(G), 4),
                 "unmatched": miss, "ambiguous": ambiguous,
                 "result_mismatches": 0, "per_season": per},
        "market_sanity": {
            "mean_devigged_home_prob": EM.r6(float(pm.mean())),
            "devigged_home_favourite_rate": EM.r6(float((pm > 0.5).mean())),
            "realized_home_win_rate": EM.r6(float(y[idx].mean())),
            "mean_closing_overround": EM.r6(float(np.mean(
                [1 / float(r["home_close"]) + 1 / float(r["away_close"])
                 for r in rows if r["home_close"] and r["away_close"]]))),
        },
        "floor": block,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, OUT)
    log(f"wrote {OUT}")

    f = block
    print(f"\n=== NHL floor  n={f['n']}")
    print(f"  our DEV LL      {f['our_ll']}")
    print(f"  market LL       {f['market_ll']}")
    print(f"  entropy floor   {f['entropy_floor']}")
    print(f"  us -> market    {f['gap_us_to_market']}  CI {f['gap_us_to_market_ci']}")
    print(f"  market -> floor {f['gap_market_to_floor']}  CI {f['gap_market_to_floor_ci']}")
    mc = f["market_calibration"]
    print(f"  close recal a={mc['recal_intercept']} b={mc['recal_slope']} "
          f"(se {mc['recal_slope_se']}, z vs 1 = {mc['recal_slope_z_vs_1']})")
    print("  decile calibration of the devigged close:")
    for d in mc["deciles"]:
        print(f"    {d['decile']:<24} n={d['n']:<5} pred={d['mean_pred']:.4f} "
              f"real={d['realized']:.4f} gap={d['gap']:+.4f} z={d['z']:+.2f}")


if __name__ == "__main__":
    main()
