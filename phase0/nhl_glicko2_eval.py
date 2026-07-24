"""NHL Glicko-2 vs Elo — tuned on DEV only, scored once on a locked TEST split.

Protocol fixed before any result was seen (mirrors the MLB eval):

  DEV   seasons 2010-11 .. 2017-18  (eval from 2011-12 so ratings are warm)
  TEST  seasons 2018-19 .. 2025-26  (scored once)

Regular season only for the core protocol (playoffs are a different intensity
regime, handled separately downstream). Both systems tuned on DEV with the same
grid discipline. Every prediction uses only strictly-earlier games: within a
rating period all games are predicted before any of them updates a rating.

Hockey adaptation (from the retired-oracle prior): shootout games update the
rating at PARTIAL credit (so_credit) because a shootout is a coin flip
(historically ~50/50), but evaluation always scores the true binary result.

Data: data/nhl_games.csv (public NHL API spine). Market-blind — no odds.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, "phase0")
from glicko2 import Glicko2  # noqa: E402

EPS = 1e-15
DEV_START, DEV_END = 20102011, 20172018       # season ints
TEST_START, TEST_END = 20182019, 20252026
DEV_WARM_BEFORE = 20112012                     # eval DEV from here on


def load_games(reg_only=True):
    rows = []
    for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")):
        if reg_only and r["type"] != "2":
            continue
        rows.append({
            "game_id": int(r["game_id"]), "date": r["date"], "season": int(r["season"]),
            "home": r["home"], "away": r["away"],
            "y": int(r["home_win"]), "so": r["last_period"] == "SO"})
    rows.sort(key=lambda g: (g["date"], g["home"]))
    return rows


def llv(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run_glicko(games, period_days, so_credit, **kw):
    """Return per-game (season, home-win-prob, y) using leak-safe rating periods."""
    gl = Glicko2(**kw)
    out = []
    prev_season = None
    buf = []            # (home, away, update_score) queued for current period
    buf_start = None

    def flush():
        if buf:
            gl.update_period([(h, a, s) for (h, a, s) in buf])
            buf.clear()

    for g in games:
        if prev_season is not None and g["season"] != prev_season:
            flush()
            gl.new_season()
            buf_start = None
        prev_season = g["season"]
        d = date.fromisoformat(g["date"])
        if buf_start is None:
            buf_start = d
        elif (d - buf_start).days >= period_days:
            flush()
            buf_start = d
        p = gl.expect_home(g["home"], g["away"])       # pre-period ratings
        out.append((g["season"], p, g["y"]))
        upd = g["y"]
        if g["so"]:
            upd = so_credit if g["y"] == 1 else (1.0 - so_credit)
        buf.append((g["home"], g["away"], upd))
    flush()
    return out


def run_elo(games, k, ha, regress):
    R = {}
    out = []
    prev_season = None
    for g in games:
        if prev_season is not None and g["season"] != prev_season:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - regress)
        prev_season = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        p = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        out.append((g["season"], p, g["y"]))
        R[g["home"]] += k * (g["y"] - p)
        R[g["away"]] += k * ((1 - g["y"]) - (1 - p))
    return out


def score(out, lo, hi, warm=None):
    y = np.array([o[2] for o in out if lo <= o[0] <= hi and (warm is None or o[0] >= warm)])
    p = np.array([o[1] for o in out if lo <= o[0] <= hi and (warm is None or o[0] >= warm)])
    return float(llv(y, p).mean()), len(y)


def main():
    games = load_games()
    print(f"{len(games):,} regular-season games "
          f"{games[0]['date']}..{games[-1]['date']}", flush=True)
    hw = np.mean([g["y"] for g in games])
    print(f"home-win base rate {hw:.4f} (const LL {float(llv(np.array([g['y'] for g in games]), np.full(len(games), hw)).mean()):.5f})", flush=True)

    # ---- tune Glicko-2 on DEV ----
    best = (9e9, None)
    grid = list(itertools.product(
        [7, 14, 21],                    # period_days
        [0.5, 0.6],                     # so_credit
        [16.0, 24.0, 32.0],             # home_adv
        [0.20, 0.30, 0.40],             # season_regress
        [0.3, 0.5]))                    # tau
    for pd_, so_, ha_, sr_, tau_ in grid:
        out = run_glicko(games, pd_, so_, home_adv=ha_, season_regress=sr_, tau=tau_)
        ll, _ = score(out, DEV_START, DEV_END, warm=DEV_WARM_BEFORE)
        if ll < best[0]:
            best = (ll, dict(period_days=pd_, so_credit=so_, home_adv=ha_,
                             season_regress=sr_, tau=tau_))
    print(f"\nGlicko-2 best DEV {best[0]:.5f}  {best[1]}", flush=True)

    # ---- tune Elo on DEV ----
    ebest = (9e9, None)
    for k_, ha_, sr_ in itertools.product([4, 6, 8, 12], [15, 20, 25, 30], [0.20, 0.30, 0.40]):
        out = run_elo(games, k_, ha_, sr_)
        ll, _ = score(out, DEV_START, DEV_END, warm=DEV_WARM_BEFORE)
        if ll < ebest[0]:
            ebest = (ll, dict(k=k_, ha=ha_, regress=sr_))
    print(f"Elo best DEV {ebest[0]:.5f}  {ebest[1]}", flush=True)

    # ---- ONE TEST look ----
    g_out = run_glicko(games, **best[1])
    e_out = run_elo(games, **ebest[1])
    gll, n = score(g_out, TEST_START, TEST_END)
    ell, _ = score(e_out, TEST_START, TEST_END)
    d = (np.array([o[1] for o in e_out if TEST_START <= o[0] <= TEST_END]),
         np.array([o[1] for o in g_out if TEST_START <= o[0] <= TEST_END]))
    yt = np.array([o[2] for o in g_out if TEST_START <= o[0] <= TEST_END])
    diff = llv(yt, d[0]) - llv(yt, d[1])
    rng = np.random.default_rng(7)
    bs = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    acc = float(((np.array([o[1] for o in g_out if TEST_START <= o[0] <= TEST_END]) > 0.5) == (yt > 0.5)).mean())
    print(f"\n=== TEST (n={n}) ===")
    print(f"  Glicko-2  {gll:.5f}  (acc {acc:.4f})")
    print(f"  Elo       {ell:.5f}")
    print(f"  Glicko-Elo delta {gll-ell:+.5f}  CI [{lo:+.5f},{hi:+.5f}]  "
          f"({'inside noise' if lo < 0 < hi else 'SIG'})")
    json.dump({"glicko_test": round(gll, 5), "elo_test": round(ell, 5),
               "glicko_dev": round(best[0], 5), "glicko_cfg": best[1],
               "elo_dev": round(ebest[0], 5), "elo_cfg": ebest[1],
               "test_n": n, "test_acc": round(acc, 4),
               "delta_ci": [round(float(lo), 5), round(float(hi), 5)],
               "home_win_rate": round(float(hw), 4)},
              open("data/nhl_glicko2_report.json", "w"), indent=1)
    print("wrote data/nhl_glicko2_report.json")


if __name__ == "__main__":
    main()
