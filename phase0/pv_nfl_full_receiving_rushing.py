"""Player-value program (pv), NFL FULL-SPAN extension: RECEIVING + RUSHING. DISPLAY ONLY.

Extends pv_nfl_receiving_rushing_{xmodels,build}.py (DEV <= 2015) through the latest
pulled season with FROZEN DEV hyper-parameters (see pv_nfl_full_common):

  stage xmodels  the unmodified pv_nfl_receiving_rushing_xmodels.py, run with
                 --through <latest> --out data/pv_nfl_full_rr_plays.parquet. Its
                 situation baselines (xcomp / xyac / xepa per target, xyds / xsucc /
                 xepa per carry) are walk-forward by construction: every (season,
                 week) block is fitted on plays strictly before it with the DEV
                 LightGBM settings. The script prints nothing once --through > 2015.
  stage walk     the continuous-outcome TrueSkill walk of pv_nfl_receiving_rushing_build.py:
                 its GaussTS filter and prev_league_mean are extracted verbatim with
                 `ast`; the empirical-Bayes moments and the selected (vs, tau, rho,
                 widen) per component are READ from the DEV run
                 (data/pv_nfl_receiving_rushing_params.json, through=2015) -- never
                 re-estimated. The residual definitions below are the build's
                 lines, unchanged. Usage (target / carry share) uses the build's
                 frozen EWMA constants.

Components kept for display (the ones the DEV compose step used, plus their yards
split): xepat + epat = v_e, EPA per target above average (opportunity quality +
execution, QB- and defence-adjusted); xypt / yptoe = the same split in yards;
epar = EPA per designed carry over expected (blocking unit + defence adjusted);
ryds = rush yards over expected per carry (capped -5..20).

Output data/pv_nfl_full_rr_current.parquet: one row per player, his rating state as
of his NEXT game (the filter's own season transition applied), usage shares and
per-season target / carry COUNTS. No realised yards, EPA or success is stored.

--verify-dev: re-walks DEV (<= 2015) only and compares the per-event predictions with
the DEV run's data/pv_nfl_receiving_rushing_events.parquet (a DEV-only check).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import pv_nfl_full_common as C

BUILD = "pv_nfl_receiving_rushing_build"
XMOD = "pv_nfl_receiving_rushing_xmodels"
PLAYS_OUT = "pv_nfl_full_rr_plays.parquet"
CURRENT_OUT = "pv_nfl_full_rr_current.parquet"
DEV_LAST = 2015
KEEP_COMPS = ("xepat", "epat", "xypt", "yptoe", "epar", "ryds")


def stage_xmodels(last: int) -> None:
    out = os.path.join(C.DATA, PLAYS_OUT)
    C.log(f"xmodels: walk-forward situation models through {last} -> {PLAYS_OUT} (silent past 2015)")
    C.exec_script(XMOD, ["--through", last, "--out", out])
    assert os.path.exists(out), "xmodels wrote nothing"


def frozen_code():
    """GaussTS + prev_league_mean from the build file, verbatim."""
    src, sha = C.read_source(BUILD)
    ns = {"np": np, "pd": pd}
    exec(compile(C.ast_extract(src, ["GaussTS", "prev_league_mean"]), C.src_path(BUILD), "exec"), ns)
    C.record_provenance(BUILD, sha)
    return ns


def load_plays(through: int, path: str) -> pd.DataFrame:
    PL = pd.read_parquet(path)
    PL = PL[PL.season <= through].reset_index(drop=True)
    assert PL.season.max() <= through
    PL["passer_id"] = PL["passer_id"].astype(object)
    return PL


def residuals(PL: pd.DataFrame, ns: dict) -> None:
    """pv_nfl_receiving_rushing_build.py lines 106-147, unchanged (subset of components)."""
    ns["PL"] = PL
    prev_league_mean = ns["prev_league_mean"]
    T = PL.kind.to_numpy() == "T"
    Cc = PL.complete_pass.to_numpy() == 1
    PL["y_t"] = np.where(T, np.where(Cc, PL.air_yards + PL.yac_c, 0.0), np.nan)
    PL["x_ypt"] = np.where(T, PL.xcomp * (PL.air_yards + PL.xyac), np.nan)
    PL["ypc_c"] = np.where(~T, PL.yds_c, np.nan)
    PL["r_epat"] = np.where(T, PL.epa - PL.xepa, np.nan)
    PL["r_xypt"] = PL.x_ypt - prev_league_mean("x_ypt")
    PL["r_yptoe"] = PL.y_t - PL.x_ypt
    PL["xepa_t"] = np.where(T, PL.xepa, np.nan)
    PL["r_xepat"] = PL.xepa_t - prev_league_mean("xepa_t")
    PL["r_ryds"] = np.where(~T, PL.yds_c - PL.xyds, np.nan)
    PL["r_epar"] = np.where(~T, PL.epa - PL.xepa, np.nan)


COMPS = {  # the build's COMPS entries for the kept components
    "epat": ("T", "r_epat", ("P", "Q", "D")),
    "xypt": ("T", "r_xypt", ("P", "Q", "D")),
    "yptoe": ("T", "r_yptoe", ("P", "Q", "D")),
    "xepat": ("T", "r_xepat", ("P", "Q", "D")),
    "ryds": ("R", "r_ryds", ("P", "O", "D")),
    "epar": ("R", "r_epar", ("P", "O", "D")),
}


def make_filter(ns, comp, MOM, prm):
    """the build's make_filter, reading the frozen DEV moments."""
    m = MOM[comp]
    roles = COMPS[comp][2]
    V = {r: m["V" + r] * (prm.get("vs", 1.0) if r == "P" else 1.0) for r in roles}
    return ns["GaussTS"](m["R"], V, prm["tau_f"], prm["rho"], prm["widen_f"])


def walk(PL, ns, frozen, comps):
    """the build's run_comp over every event; returns the filters (final states) and
    the per-event one-step predictions (p_)."""
    KEYS = {"P": PL.player_id.astype(str).to_numpy(),
            "Q": ("Q:" + PL.passer_id.astype(str)).to_numpy(),
            "O": ("O:" + PL.posteam.astype(str)).to_numpy(),
            "D": ("D:" + PL.defteam.astype(str)).to_numpy()}
    GIDX = PL.gidx.to_numpy()
    SEAS = PL.season.to_numpy().astype(int)
    KIND = PL.kind.to_numpy()
    filters, preds = {}, {}
    for comp in comps:
        kind, col, roles = COMPS[comp]
        f = make_filter(ns, comp, frozen["moments"], frozen["params"][comp])
        y = PL[col].to_numpy()
        idx = np.where((KIND == kind) & ~np.isnan(y))[0]
        pred = np.full(len(PL), np.nan)
        R = f.R
        for i in idx:
            g, s = GIDX[i], SEAS[i]
            sts = [f.touch(KEYS[r][i], r, g, s) for r in roles]
            pred[i] = f.update(sts, y[i], R)
        filters[comp], preds[comp] = f, pred
        C.log(f"  walk {comp}: {len(idx):,} events")
    return filters, preds


def usage(PL, frozen):
    """the build's usage EWMA (per appearance, halved at a season change), final state."""
    T = PL.kind.to_numpy() == "T"
    PL["_isT"] = T.astype(float)
    PL["_isR"] = (~T).astype(float)
    agg = PL.groupby(["gidx", "season", "player_id", "posteam"], sort=True).agg(
        y_tgt=("_isT", "sum"), y_car=("_isR", "sum")).reset_index()
    tt = agg.groupby(["gidx", "posteam"])[["y_tgt", "y_car"]].sum()
    agg = agg.join(tt.rename(columns={"y_tgt": "team_tgt", "y_car": "team_car"}), on=["gidx", "posteam"])
    U = frozen["usage"]
    dec, pn, pri = U["decay"], U["prior_n"], U["prior"]
    u = defaultdict(lambda: {"tsh": 0.0, "tsh_w": 0.0, "csh": 0.0, "csh_w": 0.0, "season": None, "g": 0})
    for pid, season, nt, tt_, nc, tc_ in agg[["player_id", "season", "y_tgt", "team_tgt", "y_car", "team_car"]] \
            .itertuples(index=False):
        s = u[pid]
        if s["season"] is not None and s["season"] != season:
            for k in ("tsh", "csh"):
                s[k] *= 0.5; s[k + "_w"] *= 0.5
        s["season"] = season
        for k, num, den in (("tsh", nt, tt_), ("csh", nc, tc_)):
            s[k] = dec * s[k] + (num / den if den > 0 else 0.0)
            s[k + "_w"] = dec * s[k + "_w"] + 1.0
        s["g"] += 1
    return u, pn, pri


def current_table(PL, filters, u, pn, pri, last):
    """one row per player: component state for his NEXT game in season `last`."""
    NEXT = 10 ** 9                               # a game index after every real one
    team = PL.sort_values(["gidx", "play_id"]).groupby("player_id").posteam.last()
    last_season = PL.groupby("player_id").season.max()
    cnt = PL.assign(t=(PL.kind == "T").astype(int), c=(PL.kind == "R").astype(int)) \
        .groupby(["player_id", "season"])[["t", "c"]].sum()
    rows = []
    for pid in team.index:
        rec = {"player_id": pid, "team": C.fr(team[pid]), "last_season": int(last_season[pid])}
        for comp, f in filters.items():
            if pid in f.st:
                s = f.touch(pid, "P", NEXT, last)
                rec["mu_" + comp] = s[0]; rec["sd_" + comp] = float(np.sqrt(s[1])); rec["n_" + comp] = s[4]
            else:
                rec["mu_" + comp] = 0.0; rec["sd_" + comp] = float(np.sqrt(f.V["P"])); rec["n_" + comp] = 0.0
        s = u.get(pid)
        if s is not None:
            half = 0.5 if (s["season"] is not None and s["season"] != last) else 1.0
            for k in ("tsh", "csh"):
                rec["pre_" + k] = (half * s[k] + pn * pri[k]) / (half * s[k + "_w"] + pn)
            rec["games"] = s["g"]
        for yr, tag in ((last, "cur"), (last - 1, "prev")):
            rec[f"tgt_{tag}"] = int(cnt.t.get((pid, yr), 0)) if (pid, yr) in cnt.index else 0
            rec[f"car_{tag}"] = int(cnt.c.get((pid, yr), 0)) if (pid, yr) in cnt.index else 0
        pc = cnt.loc[pid] if pid in cnt.index.get_level_values(0) else None
        rec["tgt_career"] = int(pc.t.sum()) if pc is not None else 0
        rec["car_career"] = int(pc.c.sum()) if pc is not None else 0
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["v_e"] = out.mu_xepat + out.mu_epat
    out["v_t"] = out.mu_xypt + out.mu_yptoe
    return out


def stage_walk(last: int, verify_dev: bool = False):
    frozen = C.read_json("pv_nfl_receiving_rushing_params.json")
    assert int(frozen["through"]) == DEV_LAST, "frozen params must come from the DEV run"
    ns = frozen_code()
    through = DEV_LAST if verify_dev else last
    # --verify-dev walks the DEV run's own plays file, isolating the walk from the
    # (re-fitted, LightGBM) baselines
    src = "pv_nfl_receiving_rushing_plays.parquet" if verify_dev else PLAYS_OUT
    PL = load_plays(through, os.path.join(C.DATA, src))
    residuals(PL, ns)
    filters, preds = walk(PL, ns, frozen, KEEP_COMPS)
    if verify_dev:
        ev = pd.read_parquet(os.path.join(C.DATA, "pv_nfl_receiving_rushing_events.parquet"),
                             columns=["kind", "game_id", "play_id"] + ["p_" + c for c in KEEP_COMPS])
        m = PL[["kind", "game_id", "play_id"]].assign(**{"q_" + c: preds[c] for c in KEEP_COMPS}) \
            .merge(ev, on=["kind", "game_id", "play_id"], how="inner")
        assert len(m) == len(ev), (len(m), len(ev))
        diffs = {}
        for c in KEEP_COMPS:
            a, b = m["q_" + c].to_numpy(), m["p_" + c].to_numpy()
            ok = bool((np.isfinite(a) == np.isfinite(b)).all())
            d = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
            diffs[c] = d if ok else float("inf")
            C.log(f"  verify DEV {c}: nan-pattern equal {ok}, max |diff| {d:.2e}")
        return diffs
    u, pn, pri = usage(PL, frozen)
    cur = current_table(PL, filters, u, pn, pri, last)
    C.write_parquet(cur, CURRENT_OUT)
    C.record_provenance(BUILD, C.sha256_file(C.src_path(BUILD)), {"receiving_rushing": {"through_season": last}})
    C.log(f"wrote {CURRENT_OUT}: {len(cur):,} players (state as of season {last})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["xmodels", "walk", "all"], default="all")
    ap.add_argument("--verify-dev", action="store_true")
    a = ap.parse_args()
    last = C.events_span()["season"]
    if a.stage in ("xmodels", "all") and not a.verify_dev:
        stage_xmodels(last)
    if a.stage in ("walk", "all"):
        stage_walk(last, verify_dev=a.verify_dev)
    return None


if __name__ == "__main__":
    main()
