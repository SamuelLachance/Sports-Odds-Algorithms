"""Player-value program, NHL -- AMENDMENT 1 rebuild (data/pv_nhl_serve_prereg.json ->
amendment_1): the two walk-forward fixes, rebuilt downstream for DEV and for the full
DEV + TEST range, into NEW files only (no original file is written).

The fixes live in the component code itself, one implementation each, shared by the
DEV builders and the full wrappers:
  * missing unblocked-shot xG: phase0/pv_nhl_xg_impute.py, called by
    pv_nhl_defense_onice_stints.build_season (mean xG of the type over known-xG shots
    dated strictly before the shot's date; was: the whole-season mean);
  * faceoff league mean rating: pv_nhl_duels._fo_engine commits it per DATE (was:
    per faceoff, so a game read same-date games with a lower gid).

Outputs
  DEV   data/pv_nhl_fix_defense_onice_*        stints, toi5, 46 walk-forward fits, values
        data/pv_nhl_fix_discrete_duels*        duels build (frozen DEV params)
        data/pv_nhl_fix_compose_*              panel, weights, validation, game features
  FULL  data/pv_nhl_fix_full_defense_onice_*   (shift arrays: the existing, unaffected
                                                data/pv_nhl_full_rapmel_sh_<s>.npy)
        data/pv_nhl_fix_full_discrete_duels*
        data/pv_nhl_fix_full_compose_*         (creation / finishing inputs: the existing
                                                data/pv_nhl_full_{creation,finishing}_*;
                                                neither reads stints, the xG fill or
                                                faceoff ratings)
        data/pv_nhl_fix_full_serve_lv_last.csv S1 input, pv_nhl_serve_screen.lv_last
  data/pv_nhl_fix_rebuild.json                 checks, change vs the originals, sha256

Everything else is the DEV specification, unchanged: RAPM grid / windows / cutoffs,
value configs (data/pv_nhl_defense_onice_eval_predict_month.json in_sample_best),
duels params (data/pv_nhl_discrete_duels_params.json), compose spec. Compose weights
for every TEST season = W[2018-19] refit on the FIXED DEV panel (DEV rows only), via
pv_nhl_full_compose.walk_forward_weights_frozen.

BLAS: the fit stages must run with OPENBLAS_NUM_THREADS=1 (asserted): single-thread
BLAS is deterministic, so the DEV walk-forward fits of the DEV and full runs are
bit-identical (the multithreaded originals differed by ~1 ulp run to run).

TEST outcome hygiene: exactly the full wrappers' (stint QA / fit-info goal totals
scrubbed, TEST panel labels NaN). `verify` reads no TEST outcome; for TEST rows it
compares FEATURE values only (fixed vs original). Market-blind.

    python phase0/pv_nhl_fix_rebuild.py dev_stints|dev_fit|dev_values|dev_duels|dev_compose
    python phase0/pv_nhl_fix_rebuild.py full_stints|full_fit|full_values|full_duels|full_compose
    python phase0/pv_nhl_fix_rebuild.py verify
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pv_nhl_full_common as FC  # noqa: E402

os.chdir(FC.ROOT)                      # the component scripts use data/... paths
DEV_DEF = "data/pv_nhl_fix_defense_onice_"
FULL_DEF = "data/pv_nhl_fix_full_defense_onice_"
ORIG_DEV_DEF = "data/pv_nhl_defense_onice_"
ORIG_FULL_DEF = "data/pv_nhl_full_defense_onice_"
DEV_DUELS = os.path.join(FC.ROOT, "data", "pv_nhl_fix_discrete_duels")
MANIFEST = FC.D("pv_nhl_fix_rebuild.json")
PROCS = int(os.environ.get("PV_FIX_PROCS", "12"))
SCRUB = {"goals", "league", "goals1551", "goals1551_in_kept", "goals_all_nonSO",
         "kept_goals", "kept_xg", "share_1551_goals_in_kept"}
# full-run component files that the fixes do not touch (read, never written)
UNCHANGED_FULL = {"pv_nhl_creation_wf.parquet", "pv_nhl_creation_pg.parquet",
                  "pv_nhl_finishing_and_saving_player_games.parquet",
                  "pv_nhl_finishing_and_saving_team_games.csv"}


def fixd(name):
    """'pv_nhl_x' -> absolute data/pv_nhl_fix_x (DEV fixed file)."""
    assert name.startswith("pv_nhl_") and not name.startswith(("pv_nhl_fix_", "pv_nhl_full_"))
    return FC.D("pv_nhl_fix_" + name[len("pv_nhl_"):])


def fixfull(name):
    """'pv_nhl_x' -> absolute data/pv_nhl_fix_full_x (full fixed file)."""
    assert name.startswith("pv_nhl_") and not name.startswith(("pv_nhl_fix_", "pv_nhl_full_"))
    return FC.D("pv_nhl_fix_full_" + name[len("pv_nhl_"):])


class FCProxy:
    """Stand-in for pv_nhl_full_common inside a full wrapper module: full_path() and
    D() are re-pointed to the fixed files; everything else is the real module."""

    def __init__(self, dmap=None, unchanged=()):
        self._dmap = dict(dmap or {})
        self._unchanged = set(unchanged)

    def __getattr__(self, k):
        return getattr(FC, k)

    def full_path(self, name):
        base = os.path.basename(name)
        assert base.startswith("pv_nhl_") and not base.startswith("pv_nhl_full_"), base
        if base in self._unchanged:
            return FC.full_path(base)
        return fixfull(base)

    def D(self, p):
        return FC.D(self._dmap.get(p, p))


def _configs_dev():
    """The DEV value configuration, unchanged (chosen on DEV before the fix)."""
    import pv_nhl_defense_onice_values as V
    d = json.load(open(ORIG_DEV_DEF + "eval_predict_month.json"))["families"]["g"]
    return {col: d[fam]["in_sample_best"] for col, fam in V.FAMS.items()}


# ------------------------------------------------------------------ stints ---
def _patch_stints(mode):
    import pv_nhl_defense_onice_stints as ST
    if mode == "full":
        import pv_nhl_full_defense_onice as FDO
        FDO.PFX_FULL = FULL_DEF
        FDO.patch_stints()             # full loaders, shift-array redirect, QA scrub
    else:
        ST.OUT = DEV_DEF
    assert ST.OUT == (FULL_DEF if mode == "full" else DEV_DEF)
    return ST


def _stint_worker(arg):
    mode, season = arg
    os.chdir(FC.ROOT)
    return _patch_stints(mode).build_season(season)


def stints(mode):
    seasons = FC.ALL_SEASONS if mode == "full" else FC.DEV_SEASONS
    with Pool(min(PROCS, len(seasons))) as pool:
        res = pool.map(_stint_worker, [(mode, s) for s in seasons], chunksize=1)
    if mode == "full":
        res = [{k: v for k, v in q.items() if k not in SCRUB} for q in res]
    pfx = FULL_DEF if mode == "full" else DEV_DEF
    json.dump(res, open(pfx + "st_qa.json", "w"), indent=1)


# --------------------------------------------------------------------- fit ---
def _patch_fit(mode):
    import pv_nhl_defense_onice_fit as F
    if mode == "full":
        import pv_nhl_full_defense_onice as FDO
        FDO.PFX_FULL = FULL_DEF
        FDO.patch_fit()                # all seasons, full rosters, no guard, scrub
    else:
        F.PFX = DEV_DEF
    assert F.PFX == (FULL_DEF if mode == "full" else DEV_DEF)
    return F


def _fit_worker(arg):
    mode, spec = arg
    os.chdir(FC.ROOT)
    info = _patch_fit(mode).fit_one(spec)
    if mode == "full":
        info = {k: v for k, v in info.items() if k not in SCRUB}
    return info


def fit(mode):
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "run with OPENBLAS_NUM_THREADS=1"
    F = _patch_fit(mode)
    specs = F.wf_specs()
    print(f"{len(specs)} walk-forward fits ({mode})", flush=True)
    with Pool(PROCS) as pool:
        infos = pool.map(_fit_worker, [(mode, s) for s in specs], chunksize=1)
    pfx = FULL_DEF if mode == "full" else DEV_DEF
    json.dump({"specs": specs, "fits": infos}, open(pfx + "fits_wf.json", "w"), indent=1)


# ------------------------------------------------------------------ values ---
def values(mode):
    import pv_nhl_defense_onice_values as V
    if mode == "full":
        import pv_nhl_full_defense_onice as FDO
        FDO.PFX_FULL = FULL_DEF
        FDO.patch_values()             # V.PFX, F.PFX -> FULL_DEF; DEV configs; m5 rule
        assert V.PFX == FULL_DEF and V.F.PFX == FULL_DEF
        V.cmd_values_full()
    else:
        _patch_fit("dev")
        V.PFX = DEV_DEF
        V.configs = _configs_dev
        V.cmd_values()


# ------------------------------------------------------------------- duels ---
def duels(mode):
    import pv_nhl_duels as D
    import pv_nhl_duels_build as B
    assert B.PARAMS_JSON.endswith("pv_nhl_discrete_duels_params.json")   # frozen DEV
    if mode == "full":
        import pv_nhl_full_duels as FD
        FD.FC = FCProxy()
        FD.main()
    else:
        D.OUT = DEV_DUELS
        B.run("build", False)


# ----------------------------------------------------------------- compose ---
def compose(mode):
    if mode == "full":
        import pv_nhl_full_compose as FCo
        FCo.FC = FCProxy(dmap={"pv_nhl_compose_weights.json": "pv_nhl_fix_compose_weights.json",
                               "pv_nhl_full_serve_lv_last.csv":
                                   "pv_nhl_fix_full_serve_lv_last.csv"},
                         unchanged=UNCHANGED_FULL)
        FCo.main()
        return
    import pv_nhl_compose_panel as P
    import pv_nhl_compose_core as C
    import pv_nhl_compose_fit as CF
    remap = {n: "pv_nhl_fix_" + n[len("pv_nhl_"):] for n in (
        "pv_nhl_discrete_duels.csv", "pv_nhl_defense_onice_values.csv",
        "pv_nhl_compose_skater_games.parquet", "pv_nhl_compose_goalie_games.parquet",
        "pv_nhl_compose_team_games.parquet")}

    def dfix(p):
        return FC.D(remap.get(p, p))
    P.D = dfix
    P.OUT_SK = fixd("pv_nhl_compose_skater_games.parquet")
    P.OUT_GK = fixd("pv_nhl_compose_goalie_games.parquet")
    P.OUT_TG = fixd("pv_nhl_compose_team_games.parquet")
    P.main()
    C.D = dfix
    CF.D = dfix
    CF.OUT_W = fixd("pv_nhl_compose_weights.json")
    CF.OUT_V = fixd("pv_nhl_compose_validation.json")
    CF.OUT_PV = fixd("pv_nhl_compose_player_values.parquet")
    CF.OUT_GV = fixd("pv_nhl_compose_goalie_values.csv")
    CF.OUT_GF = fixd("pv_nhl_compose_game_features.csv")
    CF.main()


# ------------------------------------------------------------------ verify ---
DEVG = FC.DEV_MAX_GID


def _npz_equal(a, b):
    x, y = np.load(a), np.load(b)
    return bool(sorted(x.files) == sorted(y.files) and all(
        x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and np.array_equal(x[k], y[k])
        for k in x.files))


def _text_prefix(dev_path, full_path, gid_col=0):
    la = open(dev_path, encoding="utf-8").read().splitlines()
    lb = open(full_path, encoding="utf-8").read().splitlines()
    rest = lb[len(la):]
    return {"rows_dev": len(la) - 1, "rows_full": len(lb) - 1,
            "dev_prefix_text_identical": bool(la == lb[:len(la)]),
            "remaining_rows_all_test": bool(all(int(float(r.split(",")[gid_col])) >= DEVG
                                                for r in rest))}


def _parquet_dev_equal(dev_path, full_path):
    a = pd.read_parquet(dev_path)
    b = pd.read_parquet(full_path)
    b = b[b.gid < DEVG].reset_index(drop=True)
    ok, bad = FC.frames_equal(a, b)
    return {"rows": int(len(a)), "bitwise": bool(ok), "bad_cols": bad[:10]}


def _absdiff(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    d = d[np.isfinite(d)]
    return {"max": float(d.max()) if len(d) else 0.0, "mean": float(d.mean()) if len(d) else 0.0,
            "share_changed": float((d > 0).mean()) if len(d) else 0.0, "n": int(len(d))}


def check_xg_fill_walk_forward(cutoff=20150115):
    """Perturbation test of the shared fill on DEV: scramble every known xG dated on or
    after `cutoff`; every fill for a shot dated before `cutoff` must be bit-identical."""
    import pv_nhl_io as IO
    import pv_nhl_defense_onice_stints as ST
    import pv_nhl_xg_impute as XI
    dates = ST.game_dates()
    base = IO.load_events(columns=["season", "gtype", "ptype", "ev", "xg"])
    rng = np.random.default_rng(5)

    def loader_pert(columns=None, **kw):
        e = base.copy()
        d = e.gid.map(dates)
        late = (d >= cutoff).to_numpy() & e.xg.notna().to_numpy()
        e.loc[late, "xg"] = rng.uniform(0.0, 1.0, int(late.sum()))
        return e

    def loader(columns=None, **kw):
        return base.copy()
    t0 = XI.prior_mean_table(loader, dates, 20172018)
    t1 = XI.prior_mean_table(loader_pert, dates, 20172018)
    sh = base[(base.gtype == 2) & (base.ptype != "SO") & base.ev.isin(XI.UNBLOCKED)
              & base.xg.isna()]
    sd = sh.gid.map(dates).to_numpy(np.float64)
    v0, fb0 = t0.lookup(sh.ev.to_numpy(), sd)
    v1, _ = t1.lookup(sh.ev.to_numpy(), sd)
    pre = sd < cutoff
    post = sd > cutoff
    return {"cutoff": cutoff, "missing_xg_shots_dev": int(len(sh)),
            "fills_before_cutoff": int(pre.sum()),
            "fills_before_cutoff_bit_identical": bool(np.array_equal(v0[pre], v1[pre])),
            "fills_after_cutoff_changed_share": float(np.mean(v0[post] != v1[post])),
            "fallback_uses_dev": int(fb0.sum()), "pass": bool(np.array_equal(v0[pre], v1[pre]))}


def check_stints_walk_forward(season=20142015, cutoff=20150115):
    """End-to-end: rebuild one DEV season's stints with every known xG dated on/after
    `cutoff` scaled by 1.37 (scratch output); every interval dated before the cutoff
    must be bit-identical to the fixed DEV stints (the old season-mean fill failed
    this whenever a pre-cutoff 5v5 shot lacked xG)."""
    import tempfile
    import pv_nhl_io as IO
    import pv_nhl_defense_onice_stints as ST
    dates = ST.game_dates()
    orig = IO.load_events

    def pert(columns=None, allow_test=False, seasons=None):
        e = orig(columns=columns, allow_test=allow_test, seasons=seasons)
        if "xg" in e.columns:
            d = e.gid.map(dates)
            late = (d >= cutoff).to_numpy() & e.xg.notna().to_numpy()
            e.loc[late, "xg"] = e.loc[late, "xg"] * 1.37
        return e
    tmp = tempfile.mkdtemp(prefix="pv_fix_")
    ST.load_events = pert
    ST.OUT = os.path.join(tmp, "pert_")
    ST.build_season(season)
    ST.load_events = orig
    a = np.load(f"{DEV_DEF}st_{season}.npz")
    b = np.load(os.path.join(tmp, f"pert_st_{season}.npz"))
    pre = a["date"] < cutoff
    same = all(np.array_equal(a[k][pre], b[k][pre]) for k in a.files)
    post_changed = bool(not np.array_equal(a["h_xg"][~pre], b["h_xg"][~pre]))
    return {"season": season, "cutoff": cutoff, "intervals_before_cutoff": int(pre.sum()),
            "pre_cutoff_bit_identical": bool(same), "post_cutoff_xg_changed": post_changed,
            "pass": bool(same and post_changed)}


def check_mbar_per_date():
    """Faceoff league mean: one value per date (every roster row of a date reads the
    same committed mean), from a fresh engine run on DEV."""
    import pv_nhl_duels as D
    import pv_nhl_duels_build as B
    import json as _j
    ev, ro, games = D.load_base(False)
    hi, ai, _ = D.team_season_index(ev, games)
    f = D.fo_frame(ev, hi, ai)
    pids = sorted(set(ro.pid.astype(np.int64)))
    pid_index = {p: i for i, p in enumerate(pids)}
    arr, fs, gpos, r, ptr, rop = B.fo_setup(f, games, ro, pid_index)
    prm = _j.load(open(B.PARAMS_JSON))["fo"]
    prm = {k: prm[k] for k in ("sigma0", "q_season", "mu_c", "mu_w", "mu_d", "lr")}
    pw, th, sm, sv, sn, sbar = D.run_fo(arr, ptr, rop, len(pid_index), prm)
    dmap = dict(zip(games.gid, games.date))
    t = pd.DataFrame({"date": r.gid.map(dmap).to_numpy(), "bar": sbar})
    nun = t.groupby("date").bar.nunique()
    return {"dates": int(len(nun)), "dates_with_more_than_one_mbar": int((nun > 1).sum()),
            "pass": bool((nun == 1).all())}


def verify():
    t0 = time.time()
    R = {"amendment": "data/pv_nhl_serve_prereg.json -> amendment_1",
         "note": "DEV rows: fixed full vs fixed DEV (reproduction) and fixed vs original "
                 "(change). TEST rows: feature values only; no TEST outcome is read."}
    # ---------------------------------------------------------------- walk-forward
    R["wf_xg_fill_perturbation"] = check_xg_fill_walk_forward()
    R["wf_stints_perturbation"] = check_stints_walk_forward()
    R["wf_faceoff_mbar_per_date"] = check_mbar_per_date()
    print("walk-forward", json.dumps({k: R[k]["pass"] for k in R if k.startswith("wf_")}),
          flush=True)
    # ---------------------------------------------------------------- stints
    st = {}
    xg_keys = ("h_xg", "a_xg")
    for s in FC.DEV_SEASONS:
        o = np.load(f"{ORIG_DEV_DEF}st_{s}.npz")
        x = np.load(f"{DEV_DEF}st_{s}.npz")
        other_same = all(np.array_equal(o[k], x[k]) for k in o.files if k not in xg_keys)
        dx = np.r_[np.abs(o["h_xg"] - x["h_xg"]), np.abs(o["a_xg"] - x["a_xg"])]
        st[str(s)] = {
            "fixed_vs_orig_non_xg_arrays_identical": bool(other_same),
            "fixed_vs_orig_xg_max_abs": float(dx.max()),
            "fixed_vs_orig_xg_intervals_changed": int((dx > 0).sum()),
            "full_vs_dev_fixed_identical": _npz_equal(f"{DEV_DEF}st_{s}.npz",
                                                      f"{FULL_DEF}st_{s}.npz"),
            "toi5_full_vs_dev_byte_identical": filecmp.cmp(
                f"{DEV_DEF}toi5_{s}.csv", f"{FULL_DEF}toi5_{s}.csv", shallow=False),
            "toi5_fixed_vs_orig_byte_identical": filecmp.cmp(
                f"{DEV_DEF}toi5_{s}.csv", f"{ORIG_DEV_DEF}toi5_{s}.csv", shallow=False)}
    R["stints"] = st
    qa = json.load(open(DEV_DEF + "st_qa.json"))
    R["stints_dev_xg_fills"] = {str(q["season"]): {"filled": q["xg_filled"],
                                                   "fallback": q["xg_fill_fallback"]}
                                for q in qa}
    qf = json.load(open(FULL_DEF + "st_qa.json"))
    R["stints_full_xg_fallback_uses"] = int(sum(q["xg_fill_fallback"] for q in qf))
    # TEST-season stints: feature arrays only (the xG columns are inputs, not scored)
    stt = {}
    for s in FC.TEST_SEASONS:
        o = np.load(f"{ORIG_FULL_DEF}st_{s}.npz")
        x = np.load(f"{FULL_DEF}st_{s}.npz")
        stt[str(s)] = {"non_xg_arrays_identical": bool(all(
            np.array_equal(o[k], x[k]) for k in o.files if k not in xg_keys)),
            "toi5_identical": filecmp.cmp(f"{ORIG_FULL_DEF}toi5_{s}.csv",
                                          f"{FULL_DEF}toi5_{s}.csv", shallow=False)}
    R["stints_test_structure_unchanged"] = stt
    # ---------------------------------------------------------------- fits
    wf = json.load(open(DEV_DEF + "fits_wf.json"))
    fit_ok, key_orig = True, {"max": 0.0}
    for sp in wf["specs"]:
        nm = sp["name"]
        fit_ok &= _npz_equal(f"{DEV_DEF}fit_{nm}.npz", f"{FULL_DEF}fit_{nm}.npz")
        a = np.load(f"{DEV_DEF}fit_{nm}.npz")
        b = np.load(f"{ORIG_DEV_DEF}fit_{nm}.npz")
        if np.array_equal(a["pids"], b["pids"]):
            key_orig["max"] = max(key_orig["max"], float(np.max(np.abs(
                a["d_xg_120000_730"] - b["d_xg_120000_730"]))))
    R["fits"] = {"n_dev": len(wf["specs"]), "full_vs_dev_fixed_bit_identical": bool(fit_ok),
                 "fixed_vs_orig_max_abs_d_xg_120000_730": key_orig["max"]}
    # ---------------------------------------------------------------- values
    R["values_csv"] = _text_prefix(f"{DEV_DEF}values.csv", f"{FULL_DEF}values.csv")
    vo = pd.read_csv(f"{ORIG_DEV_DEF}values.csv")
    vx = pd.read_csv(f"{DEV_DEF}values.csv")
    assert (vo.gid.to_numpy() == vx.gid.to_numpy()).all() and (vo.pid.to_numpy() == vx.pid.to_numpy()).all()
    R["values_change_dev"] = {c: _absdiff(vo[c], vx[c]) for c in ("d_xga", "d_ga", "d_gax", "d_ca")}
    # ---------------------------------------------------------------- duels
    R["duels_csv"] = _text_prefix(DEV_DUELS + ".csv", fixfull("pv_nhl_discrete_duels.csv"))
    fo_d = pd.read_parquet(DEV_DUELS + "_fo.parquet")
    fo_f = pd.read_parquet(fixfull("pv_nhl_discrete_duels_fo.parquet"))
    fo_f = fo_f[fo_f.gid < DEVG].reset_index(drop=True)
    R["duels_fo_bitwise"] = bool(FC.frames_equal(fo_d, fo_f)[0])
    fo_o = pd.read_parquet(FC.D("pv_nhl_discrete_duels_fo.parquet"))
    R["duels_fo_p_winner_unchanged_vs_orig"] = bool(FC.frames_equal(fo_o, fo_d)[0])
    do = pd.read_csv(FC.D("pv_nhl_discrete_duels.csv"))
    dx = pd.read_csv(DEV_DUELS + ".csv")
    assert (do.gid.to_numpy() == dx.gid.to_numpy()).all() and (do.pid.to_numpy() == dx.pid.to_numpy()).all()
    R["duels_change_dev"] = {c: _absdiff(do[c], dx[c]) for c in ("fo_rating", "fo_g60", "dd_g60")}
    # ---------------------------------------------------------------- compose
    for n in ("skater", "goalie", "team"):
        R[f"compose_{n}_games"] = _parquet_dev_equal(
            fixd(f"pv_nhl_compose_{n}_games.parquet"), fixfull(f"pv_nhl_compose_{n}_games.parquet"))
    fd, ff = fixd("pv_nhl_compose_game_features.csv"), fixfull("pv_nhl_compose_game_features.csv")
    R["game_features"] = _text_prefix(fd, ff)
    raw = open(ff, "rb").read().splitlines(keepends=True)
    dev_slice = b"".join([raw[0]] + [r for r in raw[1:] if int(r.split(b",")[0]) < DEVG])
    R["game_features"]["dev_slice_sha256"] = hashlib.sha256(dev_slice).hexdigest()
    R["game_features"]["dev_file_sha256"] = FC.sha(fd)
    R["game_features"]["dev_slice_hash_matches_dev_file"] = (
        R["game_features"]["dev_slice_sha256"] == R["game_features"]["dev_file_sha256"])
    R["game_features"]["full_file_sha256"] = FC.sha(ff)
    W = json.load(open(fixfull("pv_nhl_compose_weights.json")))
    Wd = json.load(open(fixd("pv_nhl_compose_weights.json")))
    Wo = json.load(open(FC.D("pv_nhl_compose_weights.json")))
    R["weights"] = {
        "goalie_rule_fixed_dev": Wd["goalie_rule"],
        "dev_seasons_equal_dev_file": all(
            W["walk_forward"][s]["w"] == Wd["walk_forward"][s]["w"]
            and W["walk_forward"][s]["intercept"] == Wd["walk_forward"][s]["intercept"]
            for s in map(str, FC.DEV_SEASONS)),
        "test_seasons_all_frozen": all(
            W["walk_forward"][str(s)]["w"] == W["walk_forward"][str(FC.FREEZE_SEASON)]["w"]
            for s in FC.TEST_SEASONS),
        "frozen_w_fixed": W["frozen_test_weights"]["w"],
        "frozen_n_train_games": W["frozen_test_weights"]["n_train_games"],
        "frozen_w_original": json.load(open(FC.D("pv_nhl_full_compose_weights.json")))[
            "frozen_test_weights"]["w"],
        "w_20172018_fixed_vs_orig_max_abs": max(
            abs(Wd["walk_forward"]["20172018"]["w"][c] - Wo["walk_forward"]["20172018"]["w"][c])
            for c in Wd["walk_forward"]["20172018"]["w"])}
    # feature change vs the originals
    import pv_nhl_serve_screen as SS
    Fo = pd.read_csv(FC.D("pv_nhl_compose_game_features.csv"))
    Fx = pd.read_csv(fd)
    assert (Fo.gid.to_numpy() == Fx.gid.to_numpy()).all()
    Lo, Lx = SS.lv_last(Fo), SS.lv_last(Fx).reindex(SS.lv_last(Fo).index)
    R["dev_feature_change"] = {
        "lv_home": _absdiff(Fo.lv_home, Fx.lv_home), "lv_away": _absdiff(Fo.lv_away, Fx.lv_away),
        "lv_last_diff (S1 column)": _absdiff(
            Lo.lv_last_home.fillna(0) - Lo.lv_last_away.fillna(0),
            Lx.lv_last_home.fillna(0) - Lx.lv_last_away.fillna(0)),
        "wLU_ev_def_home": _absdiff(Fo.wLU_ev_def_home, Fx.wLU_ev_def_home),
        "wLU_duels_home": _absdiff(Fo.wLU_duels_home, Fx.wLU_duels_home),
        "sd_lv_home_orig": float(Fo.lv_home.std())}
    Fxf = pd.read_csv(ff)
    Lf = SS.lv_last(Fxf)
    Ld = SS.lv_last(Fx)
    R["serve_lv_last_dev_equal"] = bool(
        np.array_equal(Ld.lv_last_home.to_numpy(), Lf.reindex(Ld.index).lv_last_home.to_numpy(),
                       equal_nan=True)
        and np.array_equal(Ld.lv_last_away.to_numpy(), Lf.reindex(Ld.index).lv_last_away.to_numpy(),
                           equal_nan=True))
    # TEST features: magnitude of the change only (no outcome anywhere)
    Fof = pd.read_csv(FC.D("pv_nhl_full_compose_game_features.csv"))
    Tt_o = Fof[Fof.gid >= DEVG].set_index("gid")
    Tt_x = Fxf[Fxf.gid >= DEVG].set_index("gid").reindex(Tt_o.index)
    R["test_feature_change_features_only"] = {
        "rows": int(len(Tt_o)),
        "lv_home": _absdiff(Tt_o.lv_home, Tt_x.lv_home),
        "lv_away": _absdiff(Tt_o.lv_away, Tt_x.lv_away)}
    sl = pd.read_csv(FC.D("pv_nhl_fix_full_serve_lv_last.csv"))
    R["serve_lv_last_file"] = {"rows": int(len(sl)), "games_in_feature_file": int(len(Fxf)),
                               "sha256": FC.sha(FC.D("pv_nhl_fix_full_serve_lv_last.csv")),
                               "blank_sides_test": int(sl[sl.gid >= DEVG].lv_last_home.isna().sum()
                                                       + sl[sl.gid >= DEVG].lv_last_away.isna().sum())}
    checks = [R["wf_xg_fill_perturbation"]["pass"], R["wf_stints_perturbation"]["pass"],
              R["wf_faceoff_mbar_per_date"]["pass"],
              all(v["fixed_vs_orig_non_xg_arrays_identical"] and v["full_vs_dev_fixed_identical"]
                  and v["toi5_full_vs_dev_byte_identical"] for v in st.values()),
              R["stints_full_xg_fallback_uses"] == 0,
              R["fits"]["full_vs_dev_fixed_bit_identical"],
              R["values_csv"]["dev_prefix_text_identical"],
              R["duels_csv"]["dev_prefix_text_identical"], R["duels_fo_bitwise"],
              R["compose_skater_games"]["bitwise"], R["compose_goalie_games"]["bitwise"],
              R["compose_team_games"]["bitwise"],
              R["game_features"]["dev_prefix_text_identical"],
              R["game_features"]["dev_slice_hash_matches_dev_file"],
              R["serve_lv_last_dev_equal"], R["weights"]["dev_seasons_equal_dev_file"],
              R["weights"]["test_seasons_all_frozen"],
              R["weights"]["goalie_rule_fixed_dev"].startswith("G2 REJECTED")]
    R["ALL_CHECKS_PASS"] = bool(all(checks))
    R["files"] = {"dev_features": fd, "dev_features_sha256": FC.sha(fd),
                  "full_features": ff, "full_features_sha256": FC.sha(ff),
                  "full_serve_lv_last": FC.D("pv_nhl_fix_full_serve_lv_last.csv")}
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(MANIFEST, "w"), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))


STAGES = {"dev_stints": lambda: stints("dev"), "full_stints": lambda: stints("full"),
          "dev_fit": lambda: fit("dev"), "full_fit": lambda: fit("full"),
          "dev_values": lambda: values("dev"), "full_values": lambda: values("full"),
          "dev_duels": lambda: duels("dev"), "full_duels": lambda: duels("full"),
          "dev_compose": lambda: compose("dev"), "full_compose": lambda: compose("full"),
          "verify": verify}

if __name__ == "__main__":
    t_ = time.time()
    for cmd in sys.argv[1:]:
        STAGES[cmd]()
        print(f"{cmd} done {time.time()-t_:.0f}s", flush=True)
