"""Emit the model's per-game probability + blend features -> data/model_probs.csv.

Walks every engine as-of (identical to the shipped model), applies the FROZEN blend,
and records for each game:
  recbp_p  the no-lineup tier prob (xFIP-Elo + bullpen + SIERA) -- the realistic
           "at open" number, since opening lines post before lineups.
  full_p   the full-tier prob with the actual lineup (ceiling, for reference).
  lf,bpz,siz,tsz,pwz,brz  the standardized blend features (for refitting a
           decorrelated final layer in the Hubacek backtest).
Market-free: no odds are used to produce these; the odds are joined later, downstream.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.siera import SieraRater  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402

OUT = "data/model_probs.csv"
EPS = 1e-15


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


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
    rb, fb = B["recbp"], B["full"]

    def z(v, mu, sd):
        return (v - B[mu]) / B[sd]

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        gid = g["game_id"]
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        lf = logit(p)
        bpe = bpd.get(gid, 0.0)
        sie = sr.edge(g["home_sp"], g["away_sp"])
        bpz = z(bpe, "bp_mu", "bp_sd")
        siz = z(sie, "si_mu", "si_sd")
        recbp_p = sig(rb["b0"] + rb["b1"] * lf + rb["b3"] * bpz + rb.get("b5", 0.0) * siz)

        ts, pw, br = tsf.get(gid), pdiff.get(gid), brdiff.get(gid)
        if ts is not None and pw is not None and br is not None:
            tsz = z(ts, "ts_mu", "ts_sd"); pwz = z(pw, "pw_mu", "pw_sd"); brz = z(br, "br_mu", "br_sd")
            full_p = sig(fb["b0"] + fb["b1"] * lf + fb["b2"] * tsz + fb["b3"] * bpz
                         + fb["b4"] * pwz + fb["b5"] * siz + fb["b6"] * brz)
        else:
            tsz = pwz = brz = ""
            full_p = ""
        d = g["date"]
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        rows.append({
            "date": iso, "season": g["season"], "dh": gid[11:] if len(gid) > 11 else "0",
            "home": g["home"], "away": g["away"], "y": int(g["y"]),
            "lf": round(lf, 6), "bpz": round(bpz, 6), "siz": round(siz, 6),
            "tsz": round(tsz, 6) if tsz != "" else "",
            "pwz": round(pwz, 6) if pwz != "" else "",
            "brz": round(brz, 6) if brz != "" else "",
            "recbp_p": round(recbp_p, 6),
            "full_p": round(full_p, 6) if full_p != "" else "",
        })
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fbb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fbb, pu, bf)

    cols = ["date", "season", "dh", "home", "away", "y",
            "lf", "bpz", "siz", "tsz", "pwz", "brz", "recbp_p", "full_p"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    nfull = sum(1 for r in rows if r["full_p"] != "")
    print(f"wrote {OUT}: {len(rows)} games ({nfull} with full/lineup prob)")


if __name__ == "__main__":
    main()
