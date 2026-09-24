"""Player-value program, NHL -- FULL (DEV + TEST) feature extension: shared helpers.

Pre-registered TEST protocol (data/pv_nhl_serve_prereg.json -> test_protocol):
  * every component rating walks forward continuously through 2018-19..2025-26 with
    its DEV hyperparameters UNCHANGED (read from the DEV parameter files, never
    re-fit);
  * compose weights for every TEST season are FROZEN at the weights the DEV walk-
    forward assigns to 2018-19 (ridge fit on DEV 2011-12..2017-18 regular season);
  * the extended pipeline reproduces every DEV feature bitwise.

This module holds what the phase0/pv_nhl_full_*.py wrappers share:
  full_path(name)     data/pv_nhl_<x>  ->  data/pv_nhl_full_<x>  (the DEV outputs are
                      never written by the wrappers; every output carries "_full_")
  noshift_games()     games with NO row in data/nhl_shifts.csv (shift chart absent;
                      0 in DEV; 57 in 2024-25 and 525 in 2025-26). For those games the
                      dressed lineup (pbp rosterSpots) is known, so every pre-game value
                      exists, but the shift-derived facts (time on ice by strength,
                      on-ice counts) are unobserved. The wrappers treat such a game as
                      MISSING for every TOI-denominated accumulator: it adds neither
                      counts nor minutes and does not advance that accumulator's per-
                      game decay (exactly the state a server holds when a shift chart
                      never arrives). Because DEV has no such game this is a no-op on
                      DEV (asserted), so the DEV reproduction is unaffected.

NO OUTCOME STATISTIC OF ANY TEST GAME IS COMPUTED ANYWHERE IN THE WRAPPERS. Ratings
update on earlier TEST events (serving behaviour); nothing is scored.
Market-blind: no odds are read.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

NO_GUARD = 10 ** 12            # replaces DEV_MAX_GID inside patched modules (full run)
DEV_SEASONS = [20102011, 20112012, 20122013, 20132014, 20142015, 20152016, 20162017,
               20172018]
TEST_SEASONS = [20182019, 20192020, 20202021, 20212022, 20222023, 20232024, 20242025,
                20252026]
ALL_SEASONS = DEV_SEASONS + TEST_SEASONS
FREEZE_SEASON = 20182019       # the DEV walk-forward's weights for this season are
                               # used for every TEST season
NOSHIFT_CSV = os.path.join(ROOT, "data", "pv_nhl_full_noshift_games.csv")


def D(p):
    return os.path.join(ROOT, "data", p)


def full_path(name):
    """'pv_nhl_creation_wf.parquet' -> absolute data/pv_nhl_full_creation_wf.parquet."""
    base = os.path.basename(name)
    assert base.startswith("pv_nhl_") and not base.startswith("pv_nhl_full_"), base
    return D("pv_nhl_full_" + base[len("pv_nhl_"):])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def noshift_games(rebuild=False):
    """Rostered games with NO row at all in data/nhl_shifts.csv -- the definition
    pv_nhl_events.py uses when it leaves the roster toi_s blank ("blank when the game
    has no shift rows"); cross-checked against that blank here. (DEV game 2013020971
    has one zero-length shift row: the DEV pipeline treated it as an observed game
    with zero TOI, so it stays observed here.) Cached to
    data/pv_nhl_full_noshift_games.csv."""
    if os.path.exists(NOSHIFT_CSV) and not rebuild:
        t = pd.read_csv(NOSHIFT_CSV)
        return set(int(g) for g in t.gid)
    have = set()
    for ch in pd.read_csv(D("nhl_shifts.csv"), chunksize=3_000_000, usecols=["game_id"],
                          dtype={"game_id": str}):
        g = pd.to_numeric(ch.game_id, errors="coerce").dropna().astype(np.int64)
        have |= set(g.unique().tolist())
    ro = pd.read_csv(D("pv_nhl_rosters.csv"), usecols=["gid", "season", "toi_s"])
    g = ro.groupby("gid").agg(season=("season", "first"),
                              toi_blank=("toi_s", lambda x: bool(x.isna().all())))
    g = g.reset_index()
    miss = g[~g.gid.isin(have)].sort_values("gid")
    assert (g.toi_blank == ~g.gid.isin(have)).all(), "roster toi_s blank != no shift rows"
    assert not (miss.gid < DEV_MAX_GID).any(), "a DEV game lacks shifts: DEV no-op broken"
    miss[["gid", "season"]].to_csv(NOSHIFT_CSV, index=False)
    return set(int(x) for x in miss.gid)


def frames_equal(a: pd.DataFrame, b: pd.DataFrame):
    """Bitwise equality of two frames (same columns, order, dtypes, values; NaN==NaN)."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, f"shape/columns differ {a.shape} {b.shape}"
    bad = []
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            if x.dtype != y.dtype:
                bad.append(f"{c}: dtype {x.dtype} vs {y.dtype}")
                continue
            xv = x.view(np.int32 if x.dtype == np.float32 else np.int64)
            yv = y.view(np.int32 if y.dtype == np.float32 else np.int64)
            nan_x, nan_y = np.isnan(x), np.isnan(y)
            ok = ((xv == yv) | (nan_x & nan_y)).all()
        else:
            xs = pd.Series(x).astype(object)
            ys = pd.Series(y).astype(object)
            ok = bool(((xs == ys) | (xs.isna() & ys.isna())).all())
        if not ok:
            bad.append(c)
    return (len(bad) == 0), bad
