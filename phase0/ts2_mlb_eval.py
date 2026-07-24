"""MLB TrueSkill-2 screens — DEV-ONLY 5-fold CV of the full 6-feature blend.

The shipped engine (mlbwp/trueskill.py) is vanilla 1v1 TrueSkill per PA with a
single global tau. TrueSkill-2 adds, among other things, experience-indexed
dynamics (new players' ratings move faster than veterans') and informed debut
priors (new players are below average on entry). This screen tests both on the
per-PA batter-vs-pitcher engine WITHOUT touching mlbwp/trueskill.py:

  VARIANT 1 — experience-indexed dynamics:
      tau_p^2 = TAU^2 * (1 + ek * exp(-n_p / N0)),  n_p = career PAs so far.
      Doses (ek, N0): (2, 300), (5, 300), (5, 1000).
  VARIANT 2 — debut prior: new players enter at mu0 - d (batters AND pitchers).
      Doses d: 0.5, 1.0.
  VARIANT 3 — best of 1 + best of 2 combined, run only if either shows any
      positive delta on its own.

Protocol (matches freeze_trueskill / weather_umpire_eval):
  * walk ALL PAs in strict (date, game_id) order; read each game's feature from
    ratings BEFORE its PAs are applied (leak-safe), exactly like trueskill_pa.py
  * per-game feature = lineup-mean offense vs starter defense differential
  * blend = logistic on [logit_fip, ts_z, bp_z, pw_z, siera_z, baserun_z], all
    non-TS columns built by freeze_trueskill's own helpers and held fixed
  * DEV = seasons 2002-2015 only; 5-fold CV KFold(5, shuffle=True, random_state=7)
  * metric = pooled out-of-fold DEV log loss; gate: improvement >= 0.0020
  * TEST (2016-2024 ex 2020) is NEVER touched here.

Baseline check: the parameterized engine at ek=0, d=0 must reproduce a fresh
walk of the shipped TrueSkill1v1 to max|feature diff| < 1e-9 (hard assert).

Writes data/ts2_mlb_screen.json.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.siera import SieraRater  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402
from mlbwp.trueskill import MU0, SIG0, BETA, TAU, TrueSkill1v1, _cdf, _pdf  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402
import trueskill_pa as TP  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
GATE = 0.0020
ROOKIE_N = 300
OUT = "data/ts2_mlb_screen.json"


class TS2(object):
    """Parameterized TrueSkill-2-style variant of mlbwp.trueskill.TrueSkill1v1.

    ek, n0 — experience-indexed dynamics: per-player process noise
             tau_p^2 = tau^2 * (1 + ek * exp(-n_p / n0)); ek=0 => shipped engine.
    debut  — new players enter at mu0 - debut instead of mu0; debut=0 => shipped.

    The update body is copied verbatim from TrueSkill1v1.update so that ek=0,
    debut=0 is bit-identical to the shipped engine (asserted in main()). Also
    tracks the sigma of each player at the moment n hits 100 / 1000 (diagnostic).
    """

    def __init__(self, ek=0.0, n0=300.0, debut=0.0,
                 beta=BETA, tau=TAU, mu0=MU0, sig0=SIG0):
        self.beta, self.tau, self.mu0, self.sig0 = beta, tau, mu0, sig0
        self.ek, self.n0, self.debut = float(ek), float(n0), float(debut)
        start = mu0 - self.debut
        self.mu = defaultdict(lambda: start)
        self.var = defaultdict(lambda: sig0 * sig0)
        self.n = defaultdict(int)
        self.tau2 = tau ** 2                      # ** to match self.tau ** 2 exactly
        self.sig_sum = {100: 0.0, 1000: 0.0}
        self.sig_cnt = {100: 0, 1000: 0}

    def _tau2(self, p):
        if self.ek == 0.0:
            return self.tau2
        return self.tau2 * (1.0 + self.ek * math.exp(-self.n[p] / self.n0))

    def update(self, winner, loser):
        vw = self.var[winner] + self._tau2(winner)
        vl = self.var[loser] + self._tau2(loser)
        c2 = 2 * self.beta ** 2 + vw + vl
        c = math.sqrt(c2)
        t = (self.mu[winner] - self.mu[loser]) / c
        denom = _cdf(t)
        v = _pdf(t) / denom if denom > 1e-12 else -t
        w = min(max(v * (v + t), 0.0), 1.0)
        self.mu[winner] += vw / c * v
        self.mu[loser] -= vl / c * v
        self.var[winner] = vw * (1 - vw / c2 * w)
        self.var[loser] = vl * (1 - vl / c2 * w)
        nw = self.n[winner] + 1
        self.n[winner] = nw
        nl = self.n[loser] + 1
        self.n[loser] = nl
        if nw == 100 or nw == 1000:
            self.sig_sum[nw] += math.sqrt(self.var[winner])
            self.sig_cnt[nw] += 1
        if nl == 100 or nl == 1000:
            self.sig_sum[nl] += math.sqrt(self.var[loser])
            self.sig_cnt[nl] += 1

    def sigma_at(self, n):
        return self.sig_sum[n] / self.sig_cnt[n] if self.sig_cnt[n] else None


def walk(order, pa_games, lineups, starters, engines, track_rookies=False):
    """Walk every PA in strict (date, game_id) order through ALL engines at once.

    Per game: read the lineup-vs-starter differential from each engine's ratings
    BEFORE the game's PAs, then apply the PAs — identical to trueskill_pa.py
    build_feature (same row condition, same averaging order, same edge sign).
    Returns one {game_id: edge} dict per engine, plus rookie-PA coverage counts
    (n < ROOKIE_N at PA time; n is identical across engines so engine 0 is used).
    """
    feats = [dict() for _ in engines]
    rk = {"total": 0, "bat": 0, "pit": 0, "either": 0}
    base = engines[0]
    for gid in order:
        g = pa_games[gid]
        ln = lineups.get(gid)
        sp = starters.get(gid)
        if ln and sp and ln[1] and ln[0] and sp[0] and sp[1]:
            for k, ts in enumerate(engines):
                mu = ts.mu
                home_bat = [mu[b] for b in ln[1]]
                away_bat = [mu[b] for b in ln[0]]
                home_off = sum(home_bat) / len(home_bat)
                away_off = sum(away_bat) / len(away_bat)
                # home offensive edge minus away offensive edge (trueskill_pa.py)
                feats[k][gid] = (home_off - mu[sp[1]]) - (away_off - mu[sp[0]])
        # apply this game's PAs to every engine (batter reached => batter wins)
        for batter, pitcher, on_base in g["pa"]:
            if track_rookies:
                rk["total"] += 1
                br = base.n[batter] < ROOKIE_N
                pr = base.n[pitcher] < ROOKIE_N
                if br:
                    rk["bat"] += 1
                if pr:
                    rk["pit"] += 1
                if br or pr:
                    rk["either"] += 1
            if on_base:
                for ts in engines:
                    ts.update(batter, pitcher)
            else:
                for ts in engines:
                    ts.update(pitcher, batter)
    return feats, rk


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def blend_inputs():
    """The 5 fixed blend inputs, built exactly like freeze_trueskill.main():
    FIP-Elo probability walk + siera edge + bullpen/power/baserun differentials."""
    games = load_games()
    lu = FZ.load_lineups()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lgiso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, lu, lgiso)
    brdiff, _ = FZ.baserun_diffs(games, FZ.load_baserun(), lu)
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    sr = SieraRater(decay=PARAMS["decay"], prior=FZ.SIERA_PRIOR, lg=slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rec, sdiff, last = {}, {}, None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        gid = g["game_id"]
        rec[gid] = (g["season"], m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"]), g["y"])
        sdiff[gid] = sr.edge(g["home_sp"], g["away_sp"])
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)
    return games, rec, sdiff, bpd, pdiff, brdiff


def dev_matrix(games, rec, sdiff, bpd, pdiff, brdiff, ts_base):
    """DEV-only rows with the same join condition as freeze_trueskill: game has a
    TS feature and non-None power + baserun diffs. The join set is identical for
    every variant (feature presence depends only on lineups/starters, not doses)."""
    gids = [g["game_id"] for g in games
            if g["game_id"] in ts_base
            and pdiff.get(g["game_id"]) is not None
            and brdiff.get(g["game_id"]) is not None
            and rec[g["game_id"]][0] in DEV]
    lf = logit(np.array([rec[g][1] for g in gids]))
    y = np.array([rec[g][2] for g in gids], float)

    def z(vals):
        v = np.asarray(vals, float)
        return (v - v.mean()) / v.std()
    bpz = z([bpd[g] for g in gids])
    pwz = z([pdiff[g] for g in gids])
    siz = z([sdiff[g] for g in gids])
    brz = z([brdiff[g] for g in gids])
    return gids, lf, y, bpz, pwz, siz, brz


def cv_ll(lf, tsz, bpz, pwz, siz, brz, y):
    """Pooled out-of-fold DEV log loss of the full 6-column blend.
    Folds are identical across variants (same n, same seed)."""
    X = np.column_stack([lf, tsz, bpz, pwz, siz, brz])
    oof = np.empty(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(X):
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(X[tr], y[tr])
        oof[va] = mdl.predict_proba(X[va])[:, 1]
    return ll(oof, y)


def main():
    t0 = time.time()
    print("building fixed blend inputs (FIP-Elo, siera, bullpen, power, baserun)...", flush=True)
    games, rec, sdiff, bpd, pdiff, brdiff = blend_inputs()
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print("loading PA data + lineups + starters...", flush=True)
    pa_games = TP.load_pa_by_game()
    lineups = TP.load_lineups()
    starters = TP.load_starters()
    order = sorted(pa_games, key=lambda g: (pa_games[g]["date"], g))
    print(f"  {len(pa_games):,} games with PAs  ({time.time() - t0:.0f}s)", flush=True)

    doses1 = [(2.0, 300.0), (5.0, 300.0), (5.0, 1000.0)]
    doses2 = [0.5, 1.0]
    engines = [TrueSkill1v1(), TS2()]              # fresh shipped walk + param baseline
    names = ["baseline_fresh", "baseline_param"]
    params = [{}, {"ek": 0, "N0": None, "d": 0}]
    for ek, n0 in doses1:
        engines.append(TS2(ek=ek, n0=n0))
        names.append(f"v1_dyn_ek{ek:g}_N{n0:g}")
        params.append({"ek": ek, "N0": n0, "d": 0})
    for d in doses2:
        engines.append(TS2(debut=d))
        names.append(f"v2_debut_d{d:g}")
        params.append({"ek": 0, "N0": None, "d": d})

    print(f"walking {sum(len(pa_games[g]['pa']) for g in pa_games):,} PAs through "
          f"{len(engines)} engines...", flush=True)
    feats, rk = walk(order, pa_games, lineups, starters, engines, track_rookies=True)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # ---- baseline exactness: param engine at ek=0,d=0 vs fresh shipped walk ----
    f_fresh, f_param = feats[0], feats[1]
    assert set(f_fresh) == set(f_param), "baseline feature game sets differ"
    repro_max = max(abs(f_fresh[g] - f_param[g]) for g in f_fresh)
    assert repro_max < 1e-9, f"param baseline does not reproduce shipped engine: {repro_max}"
    # vs shipped CSV (values rounded to 4dp there, so tolerance is rounding-level)
    shipped = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv"))
               if r[0] != "game_id"}
    common = set(shipped) & set(f_fresh)
    csv_max = max(abs(shipped[g] - f_fresh[g]) for g in common) if common else None
    print(f"baseline repro: max|diff| fresh-vs-param = {repro_max:.2e}  "
          f"vs shipped csv (4dp) = {csv_max:.2e} over {len(common):,}/{len(shipped):,} games", flush=True)

    # ---- DEV matrix + CV ----
    gids, lf, y, bpz, pwz, siz, brz = dev_matrix(games, rec, sdiff, bpd, pdiff, brdiff, f_fresh)
    print(f"DEV blend rows: {len(gids):,} (seasons 2002-2015)", flush=True)

    def ts_col(fd):
        v = np.array([fd[g] for g in gids])
        return (v - v.mean()) / v.std()

    base_ll = cv_ll(lf, ts_col(f_fresh), bpz, pwz, siz, brz, y)
    print(f"BASELINE DEV CV LL = {base_ll:.5f}", flush=True)

    results = {}
    for name, fd in zip(names[2:], feats[2:]):
        v_ll = cv_ll(lf, ts_col(fd), bpz, pwz, siz, brz, y)
        results[name] = v_ll
        print(f"  {name:22s} CV LL = {v_ll:.5f}  delta = {base_ll - v_ll:+.5f}", flush=True)

    # ---- variant 3: best of 1 + best of 2, only if either shows any positive delta ----
    v1_names = names[2:2 + len(doses1)]
    v2_names = names[2 + len(doses1):]
    best1 = min(v1_names, key=lambda n: results[n])
    best2 = min(v2_names, key=lambda n: results[n])
    combo_name = None
    sig_engines = list(zip(names[1:], engines[1:]))    # all TS2 engines
    if base_ll - results[best1] > 0 or base_ll - results[best2] > 0:
        i1, i2 = v1_names.index(best1), v2_names.index(best2)
        ek, n0 = doses1[i1]
        d = doses2[i2]
        combo_name = f"v3_combo_ek{ek:g}_N{n0:g}_d{d:g}"
        print(f"running {combo_name} (best-of-1 + best-of-2 second walk)...", flush=True)
        combo = TS2(ek=ek, n0=n0, debut=d)
        cf, _ = walk(order, pa_games, lineups, starters, [combo])
        v_ll = cv_ll(lf, ts_col(cf[0]), bpz, pwz, siz, brz, y)
        results[combo_name] = v_ll
        params.append({"ek": ek, "N0": n0, "d": d})
        names.append(combo_name)
        sig_engines.append((combo_name, combo))
        print(f"  {combo_name:22s} CV LL = {v_ll:.5f}  delta = {base_ll - v_ll:+.5f}", flush=True)
    else:
        print("variant 3 skipped: neither variant 1 nor variant 2 showed a positive delta", flush=True)

    # ---- diagnostics ----
    rookie = {k: rk[k] / rk["total"] for k in ("bat", "pit", "either")}
    sigma_tbl = {nm: {"n0": round(eng.sig0, 4),
                      "n100": round(eng.sigma_at(100), 4) if eng.sigma_at(100) else None,
                      "n1000": round(eng.sigma_at(1000), 4) if eng.sigma_at(1000) else None}
                 for nm, eng in sig_engines}

    deltas = {n: base_ll - results[n] for n in results}
    best_name = max(deltas, key=deltas.get)
    gate_met = deltas[best_name] >= GATE

    out = {
        "screen": "MLB per-PA TrueSkill-2: experience-indexed dynamics + debut prior",
        "protocol": {
            "dev_seasons": "2002-2015", "cv": "KFold(5, shuffle=True, random_state=7)",
            "metric": "pooled out-of-fold DEV log loss of the 6-col blend "
                      "[logit_fip, ts_z, bp_z, pw_z, siera_z, baserun_z]",
            "gate": GATE, "n_dev_games": len(gids), "test_touched": False,
        },
        "baseline": {
            "cv_ll": round(base_ll, 6),
            "repro_max_abs_diff_vs_fresh_walk": repro_max,
            "max_abs_diff_vs_shipped_csv_4dp": csv_max,
            "n_feature_games": len(f_fresh),
        },
        "variants": [
            {"name": n, "params": p, "cv_ll": round(results[n], 6),
             "delta": round(deltas[n], 6)}
            for n, p in zip(names[2:], params[2:]) if n in results
        ],
        "gate_met": bool(gate_met),
        "best_variant": best_name,
        "best_delta": round(deltas[best_name], 6),
        "diagnostics": {
            "rookie_pa_share_n_lt_300": {k: round(v, 4) for k, v in rookie.items()},
            "total_pas": rk["total"],
            "mean_sigma_at_career_n": sigma_tbl,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}", flush=True)
    print(f"gate (>= +{GATE:.4f}): {'MET' if gate_met else 'NOT MET'} — "
          f"best {best_name} delta {deltas[best_name]:+.5f}", flush=True)


if __name__ == "__main__":
    main()
