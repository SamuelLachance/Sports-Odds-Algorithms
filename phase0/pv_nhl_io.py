"""Player-value program, NHL -- DEV-guarded loaders for the event archive.

    from pv_nhl_io import load_events, load_rosters
    ev = load_events()                     # DEV games only (gid < 2018000000)
    ro = load_rosters()                    # DEV games only

Both loaders return DEV rows by default and refuse TEST rows unless the caller
passes allow_test=True explicitly -- which, under the locked-split protocol, only
the lead engineer's single TEST look (or serving) may do. Every DEV-returning
call asserts that no TEST gid slipped through.

Events come from data/pv_nhl_events.parquet when it is at least as new as the
CSV (pyarrow row-group filter, fast); otherwise from the CSV, read in chunks that
stop at the first TEST gid (the file is sorted by gid, DEV first).
"""
from __future__ import annotations

import os

import pandas as pd

DEV_MAX_GID = 2018000000
EV_CSV = "data/pv_nhl_events.csv"
EV_PQ = "data/pv_nhl_events.parquet"
RO_CSV = "data/pv_nhl_rosters.csv"


def _root(p):
    if os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), p)


def load_events(columns=None, allow_test=False, seasons=None):
    """Event rows. DEV only unless allow_test=True. `seasons`: optional iterable
    of season ints (e.g. {20152016, 20162017}) to subset further."""
    csvp, pqp = _root(EV_CSV), _root(EV_PQ)
    cols = None if columns is None else list(dict.fromkeys(
        ["gid"] + (["season"] if seasons is not None else []) + list(columns)))
    if os.path.exists(pqp) and (not os.path.exists(csvp)
                                or os.path.getmtime(pqp) >= os.path.getmtime(csvp)):
        filt = None if allow_test else [("gid", "<", DEV_MAX_GID)]
        df = pd.read_parquet(pqp, columns=cols, filters=filt)
    else:
        parts = []
        for ch in pd.read_csv(csvp, usecols=cols, chunksize=500_000,
                              low_memory=False):
            if not allow_test:
                stop = (ch.gid >= DEV_MAX_GID).any()
                ch = ch[ch.gid < DEV_MAX_GID]
                parts.append(ch)
                if stop:
                    break
            else:
                parts.append(ch)
        df = pd.concat(parts, ignore_index=True)
    if seasons is not None:
        if "season" not in df.columns:
            raise ValueError("seasons filter needs the 'season' column")
        df = df[df.season.isin(set(int(s) for s in seasons))].reset_index(drop=True)
    if not allow_test:
        assert len(df) == 0 or int(df.gid.max()) < DEV_MAX_GID, "TEST gid leaked"
    return df


def load_rosters(allow_test=False):
    df = pd.read_csv(_root(RO_CSV))
    if not allow_test:
        df = df[df.gid < DEV_MAX_GID].reset_index(drop=True)
        assert len(df) == 0 or int(df.gid.max()) < DEV_MAX_GID
    return df
