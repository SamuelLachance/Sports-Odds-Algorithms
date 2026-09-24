"""Player-value program, NHL -- ONE shared, strictly walk-forward fill of missing xG.

Amendment 1 of data/pv_nhl_serve_prereg.json. The event archive carries MoneyPuck
xG for almost every unblocked shot attempt (goal / shot-on-goal / missed-shot);
~0.05% have none. phase0/pv_nhl_defense_onice_stints.py used to fill those with
the WHOLE-SEASON mean xG of the event type, which contains shots dated after the
walk-forward RAPM fit cutoffs (a leak of ~7e-5 on d_xga). This module replaces it.

Rule (the only one; the DEV builder and the full DEV+TEST wrapper both call it,
through pv_nhl_defense_onice_stints.build_season):

  a missing xG of a shot of type T dated d is filled with the mean xG of type T
  over ALL regular-season, non-shootout shots of type T with a KNOWN xG dated
  STRICTLY BEFORE d -- a running mean across seasons from the first date of the
  archive (2010-10-07). Nothing dated d or later enters the fill.

  First dates: while no earlier date holds a known-xG shot of type T (only the
  archive's first date, 2010-10-07), the fill is the declared league constant
  FALLBACK[T] below. These constants are declared, not estimated from the archive
  (MoneyPuck-scale orders of magnitude: a goal's xG averages ~0.2, a saved or
  missed unblocked attempt ~0.06). In this archive the first missing xG is on
  2010-10-08, one completed date later, so the constants are never used; every
  caller counts their uses (QA key xg_fill_fallback) so that stays checked.

Determinism: the per-date sums are taken in archive row order and cumulated in
date order, so a DEV date gets the bit-identical fill whether the loader is the
DEV-guarded one or the full one (no TEST date precedes a DEV date).

Market-blind. Nothing here reads an outcome beyond the shot rows themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNBLOCKED = ("goal", "shot-on-goal", "missed-shot")
FALLBACK = {"goal": 0.20, "shot-on-goal": 0.06, "missed-shot": 0.06}   # declared


class PriorMeanTable:
    """Per type T: sorted dates carrying >= 1 known-xG shot of type T, and the
    cumulative xG sum / count through each of those dates (inclusive)."""

    def __init__(self, per_type):
        self.per_type = per_type          # T -> (dates int64, csum float64, ccnt int64)

    def lookup(self, types, dates):
        """Fill value for each (type, date): mean over dates strictly < date.
        Returns (values float64, used_fallback bool)."""
        types = np.asarray(types, dtype=object)
        dates = np.asarray(dates, dtype=np.float64)
        out = np.full(len(types), np.nan)
        fb = np.zeros(len(types), bool)
        for T in UNBLOCKED:
            m = (types == T) & np.isfinite(dates)
            if not m.any():
                continue
            ud, cs, cn = self.per_type[T]
            k = np.searchsorted(ud, dates[m].astype(np.int64), side="left")  # # dates < d
            has = k > 0
            v = np.full(int(m.sum()), FALLBACK[T])
            v[has] = cs[k[has] - 1] / cn[k[has] - 1]
            out[m] = v
            fb[np.nonzero(m)[0][~has]] = True
        return out, fb


def prior_mean_table(load_events_fn, date_of_gid, max_season):
    """Build the running-mean table from every regular-season, non-shootout
    unblocked shot with a known xG in seasons <= max_season.

    load_events_fn : the caller's event loader (DEV-guarded in the DEV builder, the
                     full loader in the full wrapper) -- signature of
                     pv_nhl_io.load_events(columns=..., seasons=...)
    date_of_gid    : dict gid -> yyyymmdd int (regular-season games)
    """
    ev = load_events_fn(columns=["season", "gtype", "ptype", "ev", "xg"])
    ev = ev[(ev.season <= max_season) & (ev.gtype == 2) & (ev.ptype != "SO")
            & ev.ev.isin(UNBLOCKED) & ev.xg.notna()]
    d = ev.gid.map(date_of_gid)
    ev = ev[d.notna()].assign(date=d[d.notna()].astype(np.int64))
    per = {}
    for T in UNBLOCKED:
        e = ev[ev.ev == T]
        g = e.groupby("date", sort=True).xg.agg(["sum", "count"])
        per[T] = (g.index.to_numpy(np.int64), np.cumsum(g["sum"].to_numpy(np.float64)),
                  np.cumsum(g["count"].to_numpy(np.int64)))
    return PriorMeanTable(per)


def fill_missing_xg(ev, date_of_gid, table):
    """In place on `ev` (columns gid, ev, xg): fill missing xG of unblocked attempts
    from `table` (strictly earlier dates). Returns (n_filled, n_fallback)."""
    unb = ev.ev.isin(UNBLOCKED).to_numpy()
    miss = unb & ev.xg.isna().to_numpy()
    if not miss.any():
        return 0, 0
    sub = ev.loc[miss]
    vals, fb = table.lookup(sub.ev.to_numpy(), sub.gid.map(date_of_gid).to_numpy(np.float64))
    ev.loc[miss, "xg"] = vals
    return int(np.isfinite(vals).sum()), int(fb.sum())
