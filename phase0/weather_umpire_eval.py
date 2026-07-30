"""Do WEATHER (temp, wind) and the HOME-PLATE UMPIRE add over the full model (0.67322)?

Both are things the sharp market prices that our model ignores. They mainly move the
RUN ENVIRONMENT, which for a WIN probability is second-order (a higher-scoring game
lets talent show through more / less), so we test both the main effects AND the
environment x favourite interaction, plus a directional wind x fly-ball-pitcher term.

Features (all as-of / leak-safe; weather is a pre-game observable, the umpire rating
is accumulated as-of):
  ump_env    home-plate umpire's run-environment deviation (EB-shrunk, as-of)
  temp_dev   (game temp - 70F), 0 if dome/missing
  wind_out   wind speed signed +out / -in / 0 cross
  *_x_logit  each run-env signal x the model's current logit (compression/expansion)
  wind_fb    wind_out x (home starter fly-ball share - away), directional
Data: Retrosheet event-file info records (umphome/temp/winddir/windspeed) + game-log
total runs. Increment over [xfip+ts+bp+power+siera+baserun] on the locked holdout.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.siera import SieraRater  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]
UMP_PRIOR = 150            # games of EB shrink of an umpire's run env to league
FB_PRIOR = 120.0
_OUT = {"tocf", "torf", "tolf"}
_IN = {"fromcf", "fromlf", "fromrf"}


def parse_events():
    """game_id -> (ump, temp, wind_out). game_id = home+YYYYMMDD+gamenum (Retrosheet)."""
    out = {}
    for path in sorted(glob.glob("data/retro_events/*.EV*")):
        gid = None; ump = None; temp = None; wdir = None; wspd = None
        for line in open(path, encoding="latin-1"):
            f = line.rstrip("\n").split(",")
            if f[0] == "id":
                if gid is not None:
                    out[gid] = (ump, temp, wdir, wspd)
                gid = f[1]; ump = None; temp = None; wdir = None; wspd = None
            elif f[0] == "info" and len(f) >= 3:
                if f[1] == "umphome":
                    ump = f[2]
                elif f[1] == "temp":
                    try:
                        temp = int(f[2])
                    except ValueError:
                        temp = None
                elif f[1] == "winddir":
                    wdir = f[2]
                elif f[1] == "windspeed":
                    try:
                        wspd = int(f[2])
                    except ValueError:
                        wspd = None
        if gid is not None:
            out[gid] = (ump, temp, wdir, wspd)
    return out


def total_runs():
    tr = {}
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 11:
                continue
            try:
                gid = r[6] + r[0] + r[1]
                tr[gid] = int(r[9]) + int(r[10])
            except (ValueError, IndexError):
                continue
    return tr


def wind_out(wdir, wspd):
    if not wspd or wdir is None:
        return 0.0
    if wdir in _OUT:
        return float(wspd)
    if wdir in _IN:
        return -float(wspd)
    return 0.0


def temp_dev(t):
    return float(t - 70) if (t is not None and 30 <= t <= 115) else 0.0


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    games = load_games()
    lu = FZ.load_lineups()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lgiso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, lu, lgiso)
    brdiff, _ = FZ.baserun_diffs(games, FZ.load_baserun(), lu)
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    B = json.load(open("mlbwp/artifacts/blend.json"))
    fb = B["full"]
    ev = parse_events()
    tr = total_runs()
    lg_fb_share = slg["fb"] / (slg["gb"] + slg["fb"] + slg["pu"])

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    ump_sum = defaultdict(float); ump_n = defaultdict(int)
    glob_sum = 0.0; glob_n = 0
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        gid = g["game_id"]
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        lf = logit(np.array([p]))[0]
        sd = sr.edge(g["home_sp"], g["away_sp"])
        ump, temp, wdir, wspd = ev.get(gid, (None, None, None, None))

        # full-model logit (needs all lineup features present)
        ts, pw, br = tsf.get(gid), pdiff.get(gid), brdiff.get(gid)
        ready = ts is not None and pw is not None and br is not None
        if ready:
            def z(v, mu, sd_):
                return (v - B[mu]) / B[sd_]
            fl = (fb["b0"] + fb["b1"] * lf + fb["b2"] * z(ts, "ts_mu", "ts_sd")
                  + fb["b3"] * z(bpd[gid], "bp_mu", "bp_sd") + fb["b4"] * z(pw, "pw_mu", "pw_sd")
                  + fb["b5"] * z(sd, "si_mu", "si_sd") + fb["b6"] * z(br, "br_mu", "br_sd"))
            glob = glob_sum / glob_n if glob_n > 200 else 8.7
            ump_env = ((ump_sum[ump] + UMP_PRIOR * glob) / (ump_n[ump] + UMP_PRIOR) - glob) if ump else 0.0

            def fbsh(pid):
                e = sr.E[pid]; bip = e[2] + e[3] + e[4]
                return (e[3] + FB_PRIOR * lg_fb_share) / (bip + FB_PRIOR)
            wfb = wind_out(wdir, wspd) * (fbsh(g["home_sp"]) - fbsh(g["away_sp"]))
            rows.append((g["season"], fl, g["y"], ump_env, temp_dev(temp),
                         wind_out(wdir, wspd), wfb))

        # advance
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fbb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fbb, pu, bf)
        rtot = tr.get(gid)
        if ump and rtot is not None:
            ump_sum[ump] += rtot; ump_n[ump] += 1; glob_sum += rtot; glob_n += 1

    a = np.array(rows, float)
    seas = a[:, 0]; fl = a[:, 1]; y = a[:, 2]
    dm = np.isin(seas, list(DEV)); tm = np.isin(seas, TEST)

    def zc(col):
        v = a[:, col]
        return (v - v[dm].mean()) / (v[dm].std() or 1.0)
    ump_z, temp_z, wind_z, wfb_z = zc(3), zc(4), zc(5), zc(6)
    lf_x = fl                                   # model logit itself, for interactions
    umpx = ump_z * lf_x; windx = wind_z * lf_x; tempx = temp_z * lf_x
    yt = y[tm]

    def fit(cols):
        X = np.column_stack([fl] + cols)
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(X[dm], y[dm])
        return mdl, mdl.predict_proba(X[tm])[:, 1]
    _, p_base = fit([])
    base_ll = ll(p_base, yt)
    # face validity: umpire run-env extremes
    finals = [(u, ump_sum[u] / ump_n[u]) for u in ump_n if ump_n[u] > 300]
    finals.sort(key=lambda x: -x[1])
    print(f"n={len(yt):,}  base full LL={base_ll:.5f} (=0.67322)")
    print("umpire runs/game extremes: high " + ", ".join(f"{u} {v:.2f}" for u, v in finals[:3])
          + " | low " + ", ".join(f"{u} {v:.2f}" for u, v in finals[-3:]))
    tests = [
        ("umpire env", [ump_z]),
        ("umpire env x logit", [umpx]),
        ("temp dev", [temp_z]),
        ("wind out", [wind_z]),
        ("wind x flyball(dir)", [wfb_z]),
        ("weather env x logit", [tempx, windx]),
        ("ALL weather+umpire", [ump_z, temp_z, wind_z, wfb_z, umpx, tempx, windx]),
    ]
    for name, cols in tests:
        _, p_a = fit(cols)
        d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_a, yt))
        sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
        print(f"  + {name:22s} LL={ll(p_a, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig}")


if __name__ == "__main__":
    main()
