"""Player-value program, NHL -- FULL (DEV + TEST) run of component CREATION.

Wraps the unmodified DEV scripts phase0/pv_nhl_creation_build.py (per player-game
facts) and phase0/pv_nhl_creation_rate.py (walk-forward EB rates) and runs them
through 2025-26 with the DEV hyperparameters unchanged (half-life 80 player-games,
'fringe' prior, the split-half constants K in pv_nhl_creation_rate.K -- nothing is
re-fit). Only module globals are re-pointed:

  loaders        load_events / load_rosters with allow_test=True
  DEV_MAX_GID    lifted (the DEV guards would otherwise refuse TEST rows)
  output paths   data/pv_nhl_full_creation_* (the DEV files are never written)

Missing shift charts (pv_nhl_full_common.noshift_games, TEST only): the player-games
of such a game carry zero TOI and zero on-ice counts in the fact table, so for the
rate step every TOI-denominated count and minute of those rows is set to 0 and the
per-player decay index advances only over OBSERVED games (decayed_prior below). On a
history with no missing game (all of DEV) the arithmetic is identical to
pv_nhl_creation_rate.decayed_prior, operation for operation.

    python phase0/pv_nhl_full_creation.py build    # facts  -> data/pv_nhl_full_creation_pg.parquet
    python phase0/pv_nhl_full_creation.py rate     # rates  -> data/pv_nhl_full_creation_wf.parquet
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import pv_nhl_full_common as FC
import pv_nhl_io as IO
import pv_nhl_creation_build as CB
import pv_nhl_creation_rate as CR


def _load_events_full(columns=None, allow_test=False, seasons=None):
    return IO.load_events(columns=columns, allow_test=True, seasons=seasons)


def _load_rosters_full(allow_test=False):
    return IO.load_rosters(allow_test=True)


class _JsonScrub:
    """json proxy for pv_nhl_creation_build.main: its QA block totals individual vs
    on-ice 5v5 GOALS (an outcome count); over DEV+TEST that field is dropped before it
    is written or printed. Everything else passes through."""
    DROP = ("goals_ind_vs_onice_ev",)

    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if k not in self.DROP}
        return obj

    def dump(self, obj, fh, **kw):
        return json.dump(self._clean(obj), fh, **kw)

    def dumps(self, obj, **kw):
        return json.dumps(self._clean(obj), **kw)


def patch_build():
    CB.json = _JsonScrub()
    CB.load_events = _load_events_full
    CB.load_rosters = _load_rosters_full
    CB.DEV_MAX_GID = FC.NO_GUARD
    CB.SHIFT_CACHE = FC.full_path("pv_nhl_creation_shifts_dev.parquet").replace(
        "_shifts_dev", "_shifts")
    CB.OUT = FC.full_path("pv_nhl_creation_pg.parquet")
    CB.QA_OUT = FC.full_path("pv_nhl_creation_build_qa.json")


# ------------------------------------------------------------------ rate ---
_ORIG_LOAD_PG = CR.load_pg
_ORIG_DP = CR.decayed_prior
RATE_COLS = (CR.EV_IND + CR.EV_ON + CR.PP_ALL + CR.ALL_SIT + CR.OFF
             + ["toi_ev", "toi_pp", "toi_all", "off_toi"])


def load_pg_full(reg_only=False):
    d = _ORIG_LOAD_PG(reg_only)
    ns = FC.noshift_games()
    obs = ~d.gid.isin(ns).to_numpy()
    d["_obs"] = obs.astype(np.float64)
    if (~obs).any():
        for c in RATE_COLS:
            v = d[c].to_numpy(np.float64).copy()
            v[~obs] = 0.0
            d[c] = v
    return d


def decayed_prior_obs(d, cols, lam):
    """pv_nhl_creation_rate.decayed_prior with the decay index counted over OBSERVED
    games only: S_i = sum_{j<i, obs} lam^(c_i-1-c_j) x_j, c = number of observed
    earlier games. With every game observed c == cumcount and each operation is the
    original one (DEV bitwise)."""
    if "_obs" not in d.columns:
        return _ORIG_DP(d, cols, lam)
    obs = d["_obs"].to_numpy(np.float64)
    key = d.pid.to_numpy()
    c = pd.Series(obs).groupby(key).cumsum().to_numpy() - obs
    out = {}
    if lam >= 1.0:
        for col in cols:
            x = d[col].to_numpy(np.float64) * obs
            cs = pd.Series(x).groupby(key).cumsum().to_numpy()
            out[col] = cs - x
        return pd.DataFrame(out, index=d.index)
    assert float(c.max()) * -np.log(lam) < 600.0
    wneg = lam ** (-c)
    for col in cols:
        x = d[col].to_numpy(np.float64) * obs
        cs = pd.Series(x * wneg).groupby(key).cumsum().to_numpy() - x * wneg
        s = cs * lam ** (c - 1)
        s[c == 0] = 0.0
        out[col] = s
    return pd.DataFrame(out, index=d.index)


def patch_rate():
    CR.PG = FC.full_path("pv_nhl_creation_pg.parquet")
    CR.DEV_MAX_GID = FC.NO_GUARD
    CR.load_pg = load_pg_full
    CR.decayed_prior = decayed_prior_obs


def run_rate():
    t0 = time.time()
    patch_rate()
    # DEV defaults of pv_nhl_creation_rate.__main__ (the DEV file was written with them)
    o = CR.build(80.0, "fringe", False)
    for c in o.columns:
        if o[c].dtype == np.float64:
            o[c] = o[c].astype(np.float32)
    out = FC.full_path("pv_nhl_creation_wf.parquet")
    o.to_parquet(out, index=False)
    print(f"wrote {out}: {len(o):,} rows, {o.gid.nunique():,} games "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    FC.noshift_games()
    if cmd in ("build", "all"):
        patch_build()
        CB.main()
    if cmd in ("rate", "all"):
        run_rate()
