"""Player-value program, NHL -- FULL (DEV + TEST) run of component DEFENSE_ONICE.

Wraps the unmodified DEV scripts
  phase0/bt_nhl_rapmel_stints.py (stage 'split' logic: shift rows per season)
  phase0/pv_nhl_defense_onice_stints.py   5v5 personnel intervals with goals / xG
  phase0/pv_nhl_defense_onice_fit.py      walk-forward on-ice RAPM refits
  phase0/pv_nhl_defense_onice_values.py   per player-game values (latest fit < date)
and runs them through 2025-26. Hyperparameters are the DEV ones, unchanged:
the ridge lambda / recency half-life grid and fit windows of the fit module (3 prior
seasons + the current season before each pre-season / monthly cutoff), VET_SECONDS,
REG_SECONDS, and the value configuration chosen on DEV
(data/pv_nhl_defense_onice_eval_predict_month.json -> in_sample_best; the compose
feature reads d_xga = d_xg_120000_730). The RAPM refits for TEST cutoffs use prior
TEST seasons' on-ice events -- walk-forward serving, allowed by the protocol; no fit
is ever scored.

Outcome hygiene: the DEV stint QA and fit logs print goal totals and league goal
rates of each window. Over TEST windows those are outcome statistics, so the
wrappers drop them (stint QA keys with goals / xG totals; fit-info 'goals' and
'league') before anything is written or printed.

Missing shift charts: a game with no shift chart has no 5v5 interval, so it simply
adds nothing to any fit or to career 5v5 time. The m5 minutes EWMA (not a compose
input) is not updated from such a game (it would otherwise read 0 minutes).

Outputs (never the DEV files): data/pv_nhl_full_rapmel_sh_<season>.npy,
data/pv_nhl_full_defense_onice_{st_<s>.npz, toi5_<s>.csv, fit_<name>.npz,
fits_wf.json, values.csv, team.csv}.

    python phase0/pv_nhl_full_defense_onice.py split|stints|fit|values|all

Re-run note: the DEV walk-forward fit files cannot be reproduced to the bit even by
re-running pv_nhl_defense_onice_fit.py itself (multithreaded OpenBLAS Cholesky /
LU: ~1 ulp run-to-run; max |diff| 9e-14 over all 46 DEV fits, 3.3e-16 on the compose
key d_xg_120000_730). The per-game values file the compose step reads
(values.csv, 5 decimals) is text-identical on all 337,332 DEV rows
(phase0/pv_nhl_full_verify.py).
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

import pv_nhl_full_common as FC
import pv_nhl_io as IO
import pv_nhl_defense_onice_stints as ST
import pv_nhl_defense_onice_fit as F
import pv_nhl_defense_onice_values as V

os.chdir(FC.ROOT)                     # the DEV scripts use data/... relative paths
PFX_FULL = "data/pv_nhl_full_defense_onice_"
SH_FULL = "data/pv_nhl_full_rapmel_sh_{}.npy"
SH_DEV = "data/bt_nhl_rapmel_sh_{}.npy"


def _events_full(columns=None, allow_test=False, seasons=None):
    return IO.load_events(columns=columns, allow_test=True, seasons=seasons)


def _rosters_full(allow_test=False):
    return IO.load_rosters(allow_test=True)


class _JsonScrub:
    """json proxy: drops outcome totals from QA / fit-info dicts (see docstring)."""
    DROP = {"goals", "league", "goals1551", "goals1551_in_kept", "goals_all_nonSO",
            "kept_goals", "kept_xg", "share_1551_goals_in_kept"}

    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if k not in self.DROP}
        return obj

    def dump(self, obj, fh, **kw):
        return json.dump(self._clean(obj), fh, **kw)

    def dumps(self, obj, **kw):
        return json.dumps(self._clean(obj), **kw)

    def load(self, fh, **kw):
        return json.load(fh, **kw)


# ------------------------------------------------------------------ split ---
def split():
    """bt_nhl_rapmel_stints.stage_split with the game set = every regular-season
    game (the DEV original used dev_games(): every DEV regular-season game)."""
    g = pd.read_csv("data/nhl_games.csv", usecols=["game_id", "season", "type"])
    g = g[g.type == 2]
    season_of = dict(zip(g.game_id.astype(np.int64), g.season.astype(int)))
    ids = np.array(sorted(season_of), dtype=np.int64)
    teams = {}
    parts = defaultdict(list)
    for ch in pd.read_csv("data/nhl_shifts.csv", chunksize=2_000_000,
                          dtype={"team": str}, low_memory=False):
        ch = ch[pd.to_numeric(ch.game_id, errors="coerce").notna()]
        ch["game_id"] = pd.to_numeric(ch.game_id).astype(np.int64)
        ch = ch[ch.game_id.isin(ids)].copy()
        if not len(ch):
            continue
        for c in ("player_id", "period", "start_s", "end_s"):
            ch[c] = pd.to_numeric(ch[c]).astype(np.int64)
        for t in ch.team.unique():
            teams.setdefault(t, len(teams))
        tc = ch.team.map(teams).astype(np.int64).to_numpy()
        arr = np.column_stack([ch.game_id.to_numpy(np.int64),
                               ch.player_id.to_numpy(np.int64), tc,
                               ch.period.to_numpy(np.int64),
                               ch.start_s.to_numpy(np.int64),
                               ch.end_s.to_numpy(np.int64)])
        ss = np.array([season_of[x] for x in arr[:, 0]])
        for s in np.unique(ss):
            parts[int(s)].append(arr[ss == s])
    rep = {}
    for s in FC.ALL_SEASONS:
        a = np.concatenate(parts[s])
        np.save(SH_FULL.format(s), a)
        if s in FC.DEV_SEASONS:
            ref = np.load(SH_DEV.format(s))
            rep[s] = {"rows": int(len(a)),
                      "cols_used_equal": bool(a.shape == ref.shape and np.array_equal(
                          a[:, [0, 1, 3, 4, 5]], ref[:, [0, 1, 3, 4, 5]])),
                      "team_idx_equal": bool(a.shape == ref.shape
                                             and np.array_equal(a[:, 2], ref[:, 2]))}
        else:
            rep[s] = {"rows": int(len(a)), "games": int(len(np.unique(a[:, 0])))}
        print(s, rep[s], flush=True)
    json.dump({str(k): v for k, v in rep.items()},
              open(PFX_FULL + "split_check.json", "w"), indent=1)
    assert all(rep[s]["cols_used_equal"] for s in FC.DEV_SEASONS)


# ----------------------------------------------------------------- stints ---
class _NpProxy:
    """numpy stand-in for pv_nhl_defense_onice_stints: np.load of the round-1 DEV
    shift arrays is redirected to the full-season arrays written by split()."""

    def __getattr__(self, k):
        return getattr(np, k)

    def load(self, path, *a, **kw):
        path = str(path)
        if "bt_nhl_rapmel_sh_" in path:
            path = path.replace("data/bt_nhl_rapmel_sh_", "data/pv_nhl_full_rapmel_sh_")
        return np.load(path, *a, **kw)


def patch_stints():
    ST.SEASONS = list(FC.ALL_SEASONS)
    ST.DEV_MAX_GID = FC.NO_GUARD
    ST.OUT = PFX_FULL
    ST.load_rosters = _rosters_full
    ST.load_events = _events_full
    ST.np = _NpProxy()
    ST.json = _JsonScrub()


def _stint_worker(season):
    os.chdir(FC.ROOT)
    patch_stints()
    return ST.build_season(season)


def stints(procs=8):
    with Pool(procs) as pool:
        res = pool.map(_stint_worker, FC.ALL_SEASONS, chunksize=1)
    res = [_JsonScrub()._clean(q) for q in res]     # build_season returns the raw QA
    json.dump(res, open(PFX_FULL + "st_qa.json", "w"), indent=1)


# -------------------------------------------------------------------- fit ---
def patch_fit():
    F.SEASONS = list(FC.ALL_SEASONS)
    F.EVAL_SEASONS = list(FC.ALL_SEASONS[1:])
    F.PFX = PFX_FULL
    F.DEV_MAX_GID = FC.NO_GUARD
    F.load_rosters = _rosters_full
    F.json = _JsonScrub()


def _fit_worker(spec):
    os.chdir(FC.ROOT)
    patch_fit()
    info = F.fit_one(spec)
    return {k: v for k, v in info.items() if k not in _JsonScrub.DROP}


def fit(procs=4, only_missing=False):
    # 4 BLAS threads per worker (the run that produced the files). This machine's
    # OpenBLAS at its default 24 threads needs ~9 s per 2k x 2k solve; the solutions
    # differ across thread counts by ~1 ulp only (as does a re-run of the DEV script).
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    patch_fit()
    specs = F.wf_specs()
    if only_missing:
        specs = [s for s in specs if not os.path.exists(f"{PFX_FULL}fit_{s['name']}.npz")]
    print(f"{len(specs)} walk-forward fits", flush=True)
    with Pool(procs) as pool:
        infos = pool.map(_fit_worker, specs, chunksize=1)
    json.dump({"specs": specs, "fits": infos}, open(f"{PFX_FULL}fits_wf.json", "w"),
              indent=1)


# ----------------------------------------------------------------- values ---
def _configs_dev():
    d = json.load(open("data/pv_nhl_defense_onice_eval_predict_month.json"))["families"]["g"]
    return {col: d[fam]["in_sample_best"] for col, fam in V.FAMS.items()}


def patch_values():
    patch_fit()
    V.PFX = PFX_FULL
    V.DEV_MAX_GID = FC.NO_GUARD
    V.load_rosters = _rosters_full
    V.configs = _configs_dev
    V._NOSHIFT = FC.noshift_games()
    src = inspect.getsource(V.cmd_values)
    old = ("            x = sec.get((gids[r], p), 0.0) / 60.0\n")
    new = ("            if gids[r] in _NOSHIFT:\n"
           "                continue\n"
           "            x = sec.get((gids[r], p), 0.0) / 60.0\n")
    assert src.count(old) == 1
    src = src.replace(old, new).replace("def cmd_values():", "def cmd_values_full():")
    exec(compile(src, V.__file__ + " [full: m5 skips no-shift games]", "exec"), V.__dict__)


def values():
    patch_values()
    V.cmd_values_full()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if cmd in ("split", "all"):
        split()
    if cmd in ("stints", "all"):
        stints()
    if cmd in ("fit", "all"):
        fit()
    if cmd == "fit_missing":
        fit(only_missing=True)
    if cmd in ("values", "all"):
        values()
    print(f"{cmd} done {time.time()-t0:.0f}s", flush=True)
