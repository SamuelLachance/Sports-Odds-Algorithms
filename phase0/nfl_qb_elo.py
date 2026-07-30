"""NFL step 2: QB-adjusted Elo — the FIP-pitcher-adjustment analog.

Team Elo + a starting-QB adjustment from the QB's own history alone:
  qb_rate = EWMA(passing_epa + rushing_epa) / EWMA(dropbacks)   (leak-free, as-of)
  q       = (db*rate + prior_db*lg_rate) / (db + prior_db)      (empirical Bayes)
  adj     = beta * (q - lg_rate), clipped                        (Elo points)

Team ratings update from outcomes; QB EWMAs update from that game's weekly stat
line AFTER the prediction. lg_rate computed on DEV years only. Joint random search
(k, hfa, regress, beta, decay, prior_db, season_decay) on DEV; TEST scored once,
paired bootstrap vs the frozen plain-Elo baseline.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, ll, load_games, run_elo  # noqa: E402

CLIP = 150.0


def load_games_qb():
    games = load_games()
    qb = {}
    for r in csv.DictReader(open("data/nfl_games.csv")):
        if r["home_score"] == "":
            continue
        qb[(r["gameday"], r["home_team"], r["away_team"])] = (
            r["home_qb_id"], r["away_qb_id"], int(r["season"]), int(r["week"]))
    # re-key onto the franchise-mapped game dicts by original order match
    raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
    raw.sort(key=lambda r: (r["gameday"],))
    assert len(raw) == len(games)
    for g, r in zip(games, raw):
        g["home_qb"], g["away_qb"] = r["home_qb_id"], r["away_qb_id"]
        g["week"] = int(r["week"])
    return games


def load_qb_weeks():
    """(player_id, season, week) -> (epa_total, dropbacks)"""
    out = {}
    def f(x):
        try: return float(x)
        except (ValueError, TypeError): return 0.0
    # 2026 successor file (same NEW schema as _2025: sacks_suffered) is read the
    # day it lands; today it does not exist -> list is identical to before
    srcs = [("data/nfl_player_stats.csv", "sacks"),
            ("data/nfl_player_stats_2025.csv", "sacks_suffered")]
    if os.path.exists("data/nfl_player_stats_2026.csv"):
        srcs.append(("data/nfl_player_stats_2026.csv", "sacks_suffered"))
    for path, sack_col in srcs:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("position") != "QB":
                continue
            key = (r["player_id"], int(r["season"]), int(r["week"]))
            epa = f(r.get("passing_epa")) + f(r.get("rushing_epa"))
            db = f(r.get("attempts")) + f(r.get(sack_col)) + f(r.get("carries"))
            if key in out:
                e0, d0 = out[key]; out[key] = (e0 + epa, d0 + db)
            else:
                out[key] = (epa, db)
    return out


class QbElo:
    def __init__(self, *, k, hfa, regress, beta, decay, prior_db, season_decay, lg_rate,
                 rep_delta=0.0, rep_map=None):
        self.k, self.hfa, self.regress = k, hfa, regress
        self.beta, self.decay, self.prior_db = beta, decay, prior_db
        self.season_decay, self.lg = season_decay, lg_rate
        # Empirical-Bayes shrinkage target: an unproven QB is pulled toward
        # REPLACEMENT level (lg + rep_delta, delta<0), not league average.
        # rep_map optionally gives a PER-QB target (draft-pedigree prior).
        self.rep = lg_rate + rep_delta
        self.rep_map = rep_map or {}
        self.R = {}
        self.Q = {}                       # qb_id -> [epa_sum, db_sum]

    def qb_adj(self, qid):
        rep = self.rep_map.get(qid, self.rep)
        s = self.Q.get(qid)
        if not s or s[1] <= 0:
            # never seen: pure prior = (pedigree-adjusted) replacement level
            adj = self.beta * (rep - self.lg)
            return max(-CLIP, min(CLIP, adj))
        rate = s[0] / s[1]
        q = (s[1] * rate + self.prior_db * rep) / (s[1] + self.prior_db)
        adj = self.beta * (q - self.lg)
        return max(-CLIP, min(CLIP, adj))

    def predict(self, g):
        rh = self.R.setdefault(g["home"], 1500.0)
        ra = self.R.setdefault(g["away"], 1500.0)
        h = 0.0 if g["neutral"] else self.hfa
        diff = (rh + h + self.qb_adj(g["home_qb"])) - (ra + self.qb_adj(g["away_qb"]))
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, g, p, qbw):
        d = self.k * (g["y"] - p)
        self.R[g["home"]] += d
        self.R[g["away"]] -= d
        for qid in (g["home_qb"], g["away_qb"]):
            line = qbw.get((qid, g["season"], g["week"]))
            if line is None:
                continue
            s = self.Q.setdefault(qid, [0.0, 0.0])
            s[0] = self.decay * s[0] + line[0]
            s[1] = self.decay * s[1] + line[1]

    def new_season(self):
        for t in self.R:
            self.R[t] = 1500.0 + (self.R[t] - 1500.0) * (1.0 - self.regress)
        for s in self.Q.values():
            s[0] *= (1.0 - self.season_decay)
            s[1] *= (1.0 - self.season_decay)


def run_qb(games, qbw, *, score_from, score_to=None, **kw):
    m = QbElo(**kw)
    ps, ys, prev = [], [], None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        p = m.predict(g)
        if g["season"] >= score_from and (score_to is None or g["season"] <= score_to):
            ps.append(p); ys.append(g["y"])
        m.update(g, p, qbw)
    return np.array(ps), np.array(ys)


def main():
    games = load_games_qb()
    qbw = load_qb_weeks()
    print(f"games {len(games)} | qb-week lines {len(qbw)}")
    # join rate: how many game-starters have a stat line for that (season, week)?
    hits = sum(1 for g in games for q in (g["home_qb"], g["away_qb"])
               if (q, g["season"], g["week"]) in qbw)
    print(f"starter weekly-line join: {hits}/{2*len(games)} = {hits/(2*len(games)):.1%}")

    # league rate from DEV years only
    num = sum(v[0] for (pid, s, w), v in qbw.items() if s in DEV_YEARS)
    den = sum(v[1] for (pid, s, w), v in qbw.items() if s in DEV_YEARS)
    lg = num / den
    print(f"league EPA/dropback (DEV era): {lg:+.4f}")

    base = json.load(open("data/nfl_elo_base.json"))
    dev = [g for g in games if g["season"] in DEV_YEARS]
    rng = np.random.default_rng(20260723)
    N = 1200
    t0 = time.time()
    best = None
    for i in range(N):
        cand = dict(
            k=float(np.exp(rng.uniform(np.log(2.0), np.log(80.0)))),
            hfa=float(rng.uniform(0.0, 120.0)),
            regress=float(rng.uniform(0.0, 0.8)),
            beta=float(np.exp(rng.uniform(np.log(20.0), np.log(3000.0)))),
            decay=float(rng.uniform(0.80, 1.0)),
            prior_db=float(np.exp(rng.uniform(np.log(20.0), np.log(1500.0)))),
            season_decay=float(rng.uniform(0.0, 0.6)),
        )
        p, y = run_qb(dev, qbw, score_from=DEV_SCORE_FROM, lg_rate=lg, **cand)
        s = ll(p, y)
        if best is None or s < best[0]:
            best = (s, cand)
        if (i + 1) % 300 == 0:
            print(f"  ...{i+1}/{N}  best DEV LL {best[0]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nQB-Elo best DEV LL {best[0]:.5f}  (plain Elo DEV {base['dev_ll']})")
    print("  params:", {k: round(v, 4) for k, v in best[1].items()})

    # ---- TEST once, paired vs frozen plain-Elo baseline ----
    pq, y = run_qb(games, qbw, score_from=TEST_YEARS.start, score_to=TEST_YEARS.stop - 1,
                   lg_rate=lg, **best[1])
    pe, ye = run_elo(games, score_from=TEST_YEARS.start, score_to=TEST_YEARS.stop - 1,
                     **base["params"])
    assert np.array_equal(y, ye)
    eps = 1e-12
    def v(p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = v(pe) - v(pq)                    # >0 => QB-Elo better
    rng2 = np.random.default_rng(7)
    n = len(d)
    bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = ("QB-Elo SIG better" if lo > 0 else
           ("plain Elo SIG better" if hi < 0 else "inside noise"))
    print(f"\nTEST 2016-2025 (n={n}, scored once):")
    print(f"  plain Elo LL {ll(pe, y):.5f}")
    print(f"  QB-Elo    LL {ll(pq, y):.5f}")
    print(f"  paired delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
    json.dump({"qb_elo": {"dev_ll": round(best[0], 5), "params": best[1],
                          "test_ll": round(float(ll(pq, y)), 5), "lg_rate": round(lg, 5)},
               "plain_elo_test_ll": round(float(ll(pe, y)), 5),
               "delta": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
               "verdict": sig},
              open("data/nfl_qb_elo.json", "w"), indent=1)
    print("wrote data/nfl_qb_elo.json")


if __name__ == "__main__":
    main()
