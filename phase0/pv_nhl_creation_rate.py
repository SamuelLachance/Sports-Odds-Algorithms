"""Player-value program, NHL -- component "creation", STEP 2: walk-forward ratings.

For every DEV skater-game this computes the player's creation rates using ONLY
the games he played strictly before it (EWMA over his own games, half-life H
player-games) and shrinks each rate toward a walk-forward prior mean
(empirical Bayes, one stabilisation constant k per component and position):

    rate_hat = (C + k * mu) / (T + k)          C, T = decayed counts, minutes

Components (per 60 of the stated strength; ev = 5v5, pp = power play):
  individual creation   ixg icf iff isf g a1 a2 ireb irush irc     (ev)
                        ixg icf g a1 a2                            (pp)
                        pdrawn (per 60 all situations)
  on-ice comparators    on_gf on_xgf on_cf (ev), on_gf on_xgf (pp),
                        rel_gf rel_xgf (on minus off-ice, ev)
Composites (goals per 60, no fitted weights):
  cre_ev   = g + a1 + a2          (EV points/60: goals he scored + goals he set up)
  cre_evx  = cre_ev + ixg         (adds the shot-generation process term)
  cre_pp   = g + a1 + a2 at PP
The stabilisation constants K come from DEV split-half reliability of the raw
rates (k = T_half (1 - r) / r, pv_nhl_creation_validate.py 'reliability'); they
are design constants, not player ratings.

Prior mean mu (walk-forward, by date, per position group F/D):
  'all'      TOI-weighted league mean of every earlier game
  'fringe'   TOI-weighted mean of earlier games played by players whose decayed
             prior EV TOI was < FRINGE_MIN minutes (the population that the
             prior actually stands in for: call-ups, rookies)

Also attached: the walk-forward stint-RAPM offence value from the round-1
rapmel fits (data/bt_nhl_rapmel_fitv_{pre,jan}_<s>.csv), taking the latest fit
whose cutoff date is strictly before the game date -- the incumbent on-ice xG
rating this component is compared against.

    python phase0/pv_nhl_creation_rate.py [--half-life 80] [--prior fringe]
           [--reg-only] [--out data/pv_nhl_creation_wf.parquet]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PG = os.path.join(ROOT, "data/pv_nhl_creation_pg.parquet")
FRINGE_MIN = 300.0

# count -> TOI column it is a rate of
EV_IND = ["ixg_ev", "icf_ev", "iff_ev", "isf_ev", "g_ev", "a1_ev", "a2_ev",
          "ireb_ev", "irush_ev", "irc_ev"]
EV_ON = ["on_gf_ev", "on_xgf_ev", "on_cf_ev"]
PP_ALL = ["ixg_pp", "icf_pp", "g_pp", "a1_pp", "a2_pp", "on_gf_pp", "on_xgf_pp"]
ALL_SIT = ["pdrawn"]
OFF = ["off_gf", "off_xgf"]

# split-half stabilisation constants, minutes (DEV 2010-18, regular season,
# odd/even games within season, players >= 250 EV / 50 PP minutes per half)
K = {
    "F": {"ixg_ev": 349, "icf_ev": 100, "iff_ev": 134, "isf_ev": 180, "g_ev": 1296,
          "a1_ev": 1305, "a2_ev": 3072, "ireb_ev": 2226, "irush_ev": 1441,
          "irc_ev": 1995, "on_gf_ev": 713, "on_xgf_ev": 295, "on_cf_ev": 138,
          "off_gf": 700, "off_xgf": 300,
          "ixg_pp": 216, "icf_pp": 31, "g_pp": 491, "a1_pp": 388, "a2_pp": 900,
          "on_gf_pp": 396, "on_xgf_pp": 188, "pdrawn": 351},
    "D": {"ixg_ev": 312, "icf_ev": 125, "iff_ev": 177, "isf_ev": 225, "g_ev": 2266,
          "a1_ev": 2546, "a2_ev": 5101, "ireb_ev": 3417, "irush_ev": 2541,
          "irc_ev": 2833, "on_gf_ev": 3243, "on_xgf_ev": 623, "on_cf_ev": 248,
          "off_gf": 700, "off_xgf": 300,
          "ixg_pp": 184, "icf_pp": 37, "g_pp": 619, "a1_pp": 919, "a2_pp": 1500,
          "on_gf_pp": 985, "on_xgf_pp": 241, "pdrawn": 547},
}


def toi_col(c):
    if c in OFF:
        return "off_toi"
    if c.endswith("_pp"):
        return "toi_pp"
    if c in ALL_SIT:
        return "toi_all"
    return "toi_ev"


def load_pg(reg_only=False):
    d = pd.read_parquet(PG)
    assert int(d.gid.max()) < DEV_MAX_GID, "TEST gid leaked"
    if reg_only:
        d = d[d.gtype == 2]
    d = d.copy()
    d["grp"] = np.where(d.pos == "D", "D", "F")
    d["toi_all"] = d[["toi_ev", "toi_pp", "toi_sh", "toi_eo", "toi_xx"]].sum(axis=1)
    d["off_toi"] = (d.tm_toi_ev - d.toi_ev).clip(lower=0)
    d["off_gf"] = (d.tm_gf_ev - d.on_gf_ev).clip(lower=0)
    d["off_xgf"] = (d.tm_xgf_ev - d.on_xgf_ev).clip(lower=0)
    d = d.sort_values(["pid", "date", "gid"], kind="stable").reset_index(drop=True)
    return d


def decayed_prior(d, cols, lam):
    """Per-row exclusive decayed sums over the player's earlier games:
    S_i = sum_{j<i} lam^(i-1-j) x_j (rows sorted by pid, date)."""
    k = d.groupby("pid").cumcount().to_numpy().astype(float)
    out = {}
    if lam >= 1.0:
        for c in cols:
            x = d[c].to_numpy(np.float64)
            cs = pd.Series(x).groupby(d.pid.to_numpy()).cumsum().to_numpy()
            out[c] = cs - x
        return pd.DataFrame(out, index=d.index)
    # exact closed form: S_i = lam^(i-1) * sum_{j<i} lam^(-j) x_j ; DEV careers
    # are < 1,000 games so lam^(-k) stays far inside float64 range
    assert float(k.max()) * -np.log(lam) < 600.0
    wneg = lam ** (-k)
    key = d.pid.to_numpy()
    for c in cols:
        x = d[c].to_numpy(np.float64)
        cs = pd.Series(x * wneg).groupby(key).cumsum().to_numpy() - x * wneg
        s = cs * lam ** (k - 1)
        s[k == 0] = 0.0
        out[c] = s
    return pd.DataFrame(out, index=d.index)


def prior_means(d, cols, prior_T_ev, mode):
    """Walk-forward per-group prior mean rate (per minute) of each count, from
    games on dates strictly before the row's date."""
    use = np.ones(len(d), bool) if mode == "all" else (prior_T_ev < FRINGE_MIN)
    res = {}
    base = d[["date", "grp"]].copy()
    tois = sorted({toi_col(c) for c in cols})
    num = pd.DataFrame({c: np.where(use, d[c].to_numpy(np.float64), 0.0)
                        for c in cols + tois})
    num["date"], num["grp"] = d.date.to_numpy(), d.grp.to_numpy()
    daily = num.groupby(["grp", "date"]).sum().sort_index()
    cum = daily.groupby(level=0).cumsum() - daily            # strictly earlier dates
    cum = cum.reset_index()
    m = base.merge(cum, on=["grp", "date"], how="left")
    for c in cols:
        t = toi_col(c)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[c] = (m[c] / (m[t] / 60.0)).to_numpy()
    return pd.DataFrame(res, index=d.index)


def rapm_values():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "data/bt_nhl_rapmel_fitv_*_20*.csv")):
        if "drop" in os.path.basename(f):
            continue
        v = pd.read_csv(f, usecols=["cutoff_date", "pid", "off", "rel", "v"])
        rows.append(v)
    r = pd.concat(rows, ignore_index=True)
    r["rapm_off_s"] = r.off * r.rel
    return r.rename(columns={"off": "rapm_off", "v": "rapm_net_v"})[
        ["cutoff_date", "pid", "rapm_off", "rapm_off_s", "rapm_net_v"]]


def attach_rapm(d):
    r = rapm_values().sort_values("cutoff_date")
    cuts = np.array(sorted(r.cutoff_date.unique()))
    # latest cutoff strictly before the game date
    pos = np.searchsorted(cuts, d.date.to_numpy(), side="left") - 1
    cut = np.where(pos >= 0, cuts[np.clip(pos, 0, None)], None)
    tmp = pd.DataFrame({"cutoff_date": cut, "pid": d.pid.to_numpy()})
    m = tmp.merge(r, on=["cutoff_date", "pid"], how="left")
    assert (m.cutoff_date.dropna().to_numpy() < d.date.to_numpy()[m.cutoff_date.notna()]).all()
    for c in ("rapm_off", "rapm_off_s", "rapm_net_v"):
        d[c] = m[c].to_numpy()
    d["rapm_cutoff"] = m.cutoff_date.to_numpy()
    return d


def build(half_life=80.0, prior="fringe", reg_only=False, d=None):
    """d: optional pre-loaded load_pg() frame (the walk-forward audit passes a
    truncated / blanked copy)."""
    d = load_pg(reg_only) if d is None else d
    lam = 1.0 if not np.isfinite(half_life) else 0.5 ** (1.0 / half_life)
    cnts = EV_IND + EV_ON + PP_ALL + ALL_SIT + OFF
    tois = ["toi_ev", "toi_pp", "toi_all", "off_toi"]
    S = decayed_prior(d, cnts + tois, lam)
    T = {t: S[t].to_numpy() / 60.0 for t in tois}                 # minutes
    mu = prior_means(d, cnts, T["toi_ev"], prior)
    out = d[["gid", "date", "season", "gtype", "team", "side", "pid", "pos", "grp"]].copy()
    out["n_prior_games"] = d.groupby("pid").cumcount().to_numpy()
    out["prior_min_ev"] = T["toi_ev"]
    out["prior_min_pp"] = T["toi_pp"]
    out["prior_min_all"] = T["toi_all"]
    # expected minutes per game (decayed average), for lineup aggregation
    ng = decayed_prior(d.assign(one=1.0), ["one"], lam)["one"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["exp_min_ev"] = np.where(ng > 0, T["toi_ev"] / ng, np.nan)
        out["exp_min_pp"] = np.where(ng > 0, T["toi_pp"] / ng, np.nan)
    kk = {c: d.grp.map({g: K[g][c] for g in K}).to_numpy(float) for c in cnts}
    for c in cnts:
        t = T[toi_col(c)]
        m = mu[c].to_numpy()
        m = np.where(np.isfinite(m), m, 0.0)
        out[c + "_hat"] = (S[c].to_numpy() + kk[c] * m) / (t + kk[c]) * 60.0
    out["rel_gf_hat"] = out.on_gf_ev_hat - out.off_gf_hat
    out["rel_xgf_hat"] = out.on_xgf_ev_hat - out.off_xgf_hat
    out["fin_ev_hat"] = out.g_ev_hat - out.ixg_ev_hat
    out["cre_ev"] = out.g_ev_hat + out.a1_ev_hat + out.a2_ev_hat
    out["cre_evx"] = out.cre_ev + out.ixg_ev_hat
    out["cre_pp"] = out.g_pp_hat + out.a1_pp_hat + out.a2_pp_hat
    out = attach_rapm(out)
    out = out.sort_values(["gid", "side", "pid"]).reset_index(drop=True)
    assert int(out.gid.max()) < DEV_MAX_GID
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-life", type=float, default=80.0)
    ap.add_argument("--prior", choices=["all", "fringe"], default="fringe")
    ap.add_argument("--reg-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/pv_nhl_creation_wf.parquet"))
    a = ap.parse_args()
    o = build(a.half_life, a.prior, a.reg_only)
    for c in o.columns:
        if o[c].dtype == np.float64:
            o[c] = o[c].astype(np.float32)
    o.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(o):,} rows, {o.gid.nunique():,} games, "
          f"half_life={a.half_life} prior={a.prior} reg_only={a.reg_only}")
