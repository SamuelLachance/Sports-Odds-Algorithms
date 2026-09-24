"""Player-value program, NHL -- FULL (DEV + TEST) extension: DEV bitwise reproduction
checks and STRUCTURAL coverage of the TEST features.

Reads only feature / rating files. No TEST outcome (goals, results, labels) is read,
and no statistic relating any TEST feature to any outcome is computed.

Checks, every one on the DEV rows (gid < 2018000000) of the full files against the
DEV files written by the DEV pipeline:
  creation      pg / wf parquet                       bitwise (float32 bit patterns)
  finishing     player-games / entity-games / shots   bitwise; team-games lg_xg_pre bitwise
  duels         discrete_duels.csv                    text-identical rows; fo parquet bitwise
  defense       stints npz + toi5 csv                 array- / byte-identical
                every DEV walk-forward fit npz        array-identical (all arrays)
                values.csv                            text-identical rows
  compose       skater / goalie / team panel          bitwise
                game features csv                     text-identical rows; the DEV slice
                                                      of the full file hashes to the
                                                      pre-registered sha256
  serve         lv_last from the full file == lv_last from the DEV file on DEV games
Structural TEST coverage: games with a feature row, missing lv, sides without a
previous game (lv_last blank -> 0 by the S1 rule), games without shift charts.

    python phase0/pv_nhl_full_verify.py      -> data/pv_nhl_full_verify.json
"""
from __future__ import annotations

import ast
import filecmp
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

import pv_nhl_full_common as FC
from nhl_features_eval import TEAM_FIX

DEVG = FC.DEV_MAX_GID
OUT = FC.D("pv_nhl_full_verify.json")


def dev_rows(df):
    return df[df.gid < DEVG].reset_index(drop=True)


def cmp_parquet(name, cols=None):
    a = pd.read_parquet(FC.D(name), columns=cols)
    b = dev_rows(pd.read_parquet(FC.full_path(name), columns=cols))
    ok, bad = FC.frames_equal(a, b)
    return {"rows_dev": int(len(a)), "rows_full_dev": int(len(b)), "bitwise": bool(ok),
            "bad_cols": bad[:10]}


def cmp_text_prefix(name, gid_col=0):
    la = open(FC.D(name), encoding="utf-8").read().splitlines()
    lb = open(FC.full_path(name), encoding="utf-8").read().splitlines()
    rest = lb[len(la):]
    rest_test = all(int(float(r.split(",")[gid_col])) >= DEVG for r in rest)
    dev_in_full = sum(1 for r in lb[1:] if int(float(r.split(",")[gid_col])) < DEVG)
    return {"rows_dev": len(la) - 1, "rows_full": len(lb) - 1,
            "dev_rows_in_full": dev_in_full,
            "dev_prefix_text_identical": bool(la == lb[:len(la)]),
            "remaining_rows_all_test": bool(rest_test)}


def lv_last_fn():
    src = open(os.path.join(FC.HERE, "pv_nhl_serve_screen.py"), encoding="utf-8").read()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
          and n.name == "lv_last"][0]
    ns = {"pd": pd, "TEAM_FIX": TEAM_FIX}
    exec(compile(ast.Module([fn], []), "pv_nhl_serve_screen.py:lv_last", "exec"), ns)
    return ns["lv_last"]


def main():
    t0 = time.time()
    R = {"note": "DEV rows of every full file vs the DEV pipeline's files; TEST: structural "
                 "coverage only (no outcome read, nothing scored)"}
    # ---------------------------------------------------------- components
    R["creation_pg"] = cmp_parquet("pv_nhl_creation_pg.parquet")
    R["creation_wf"] = cmp_parquet("pv_nhl_creation_wf.parquet")
    for n in ("player_games", "entity_games", "shots"):
        R[f"finishing_{n}"] = cmp_parquet(f"pv_nhl_finishing_and_saving_{n}.parquet")
    a = pd.read_csv(FC.D("pv_nhl_finishing_and_saving_team_games.csv"),
                    usecols=["gid", "side_h", "lg_xg_pre"])
    b = dev_rows(pd.read_csv(FC.full_path("pv_nhl_finishing_and_saving_team_games.csv"),
                             usecols=["gid", "side_h", "lg_xg_pre"]))
    R["finishing_team_games_lg_xg_pre"] = {"bitwise": bool(FC.frames_equal(a, b)[0]),
                                           "rows": int(len(a))}
    R["duels_csv"] = cmp_text_prefix("pv_nhl_discrete_duels.csv")
    R["duels_fo"] = cmp_parquet("pv_nhl_discrete_duels_fo.parquet")
    dv = json.load(open(FC.full_path("pv_nhl_discrete_duels_values.json")))
    dref = json.load(open(FC.D("pv_nhl_discrete_duels_values.json")))
    R["duels_goal_values"] = {
        "dev_seasons_identical": all(json.dumps(dv[s]["fo"]) == json.dumps(dref[s]["fo"])
                                     and json.dumps(dv[s]["pen"]) == json.dumps(dref[s]["pen"])
                                     and json.dumps(dv[s]["tkgv"]) == json.dumps(dref[s]["tkgv"])
                                     for s in map(str, FC.DEV_SEASONS)),
        "test_seasons_all_equal_frozen_20182019": all(
            {k: dv[str(s)][k] for k in ("fo", "pen", "tkgv")}
            == {k: dv[str(FC.TEST_SEASONS[0])][k] for k in ("fo", "pen", "tkgv")}
            for s in FC.TEST_SEASONS),
        "frozen_fit_on": dv[str(FC.TEST_SEASONS[0])]["fit_on"]}
    st = {}
    for s in FC.DEV_SEASONS:
        x = np.load(FC.D(f"pv_nhl_defense_onice_st_{s}.npz"))
        y = np.load(FC.full_path(f"pv_nhl_defense_onice_st_{s}.npz"))
        st[str(s)] = {"stints_identical": bool(sorted(x.files) == sorted(y.files) and all(
            x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]) for k in x.files)),
            "toi5_byte_identical": filecmp.cmp(FC.D(f"pv_nhl_defense_onice_toi5_{s}.csv"),
                                               FC.full_path(f"pv_nhl_defense_onice_toi5_{s}.csv"),
                                               shallow=False)}
    R["defense_stints"] = st
    # The RAPM solves run on multithreaded OpenBLAS (cho_factor / np.linalg.solve):
    # re-running even the DEV script reproduces its own fit arrays only to ~1 ulp, so
    # the fits are compared by structure (players, flags, seconds: exact) and by the
    # max abs difference of the rating arrays; the per-game values the compose step
    # reads (values.csv, 5 decimals) are compared text-exactly below.
    fits = {}
    struct_ok, mx, mx_xga = True, 0.0, 0.0
    wf = json.load(open(FC.D("pv_nhl_defense_onice_fits_wf.json")))
    for sp in wf["specs"]:
        nm = sp["name"]
        x = np.load(FC.D(f"pv_nhl_defense_onice_fit_{nm}.npz"))
        y = np.load(FC.full_path(f"pv_nhl_defense_onice_fit_{nm}.npz"))
        struct_ok &= sorted(x.files) == sorted(y.files) and all(
            np.array_equal(x[k], y[k]) for k in ("pids", "isD", "new", "Traw", "gpids"))
        for k in x.files:
            if x[k].dtype.kind == "f" and x[k].shape == y[k].shape:
                d = float(np.nanmax(np.abs(x[k] - y[k]))) if x[k].size else 0.0
                mx = max(mx, d)
                if k in ("d_xg_120000_730", "u_xg_120000_730"):
                    mx_xga = max(mx_xga, d)
        fits[nm] = True
    R["defense_dev_fits"] = {"n": len(fits), "structure_identical": bool(struct_ok),
                             "max_abs_diff_all_rating_arrays": mx,
                             "max_abs_diff_compose_key_d_xg_120000_730": mx_xga,
                             "note": "multithreaded OpenBLAS: ~1 ulp run-to-run; see "
                                     "defense_values_csv for the compose input"}
    R["defense_team_csv_not_a_compose_input"] = cmp_text_prefix("pv_nhl_defense_onice_team.csv")
    R["defense_values_csv"] = cmp_text_prefix("pv_nhl_defense_onice_values.csv")
    # ---------------------------------------------------------- compose
    R["compose_skater_games"] = cmp_parquet("pv_nhl_compose_skater_games.parquet")
    R["compose_goalie_games"] = cmp_parquet("pv_nhl_compose_goalie_games.parquet")
    R["compose_team_games"] = cmp_parquet("pv_nhl_compose_team_games.parquet")
    feat = "pv_nhl_compose_game_features.csv"
    R["game_features"] = cmp_text_prefix(feat)
    raw = open(FC.full_path(feat), "rb").read().splitlines(keepends=True)   # CRLF kept
    dev_slice = b"".join([raw[0]] + [r for r in raw[1:] if int(r.split(b",")[0]) < DEVG])
    pre = json.load(open(FC.D("pv_nhl_serve_prereg.json")))
    R["game_features"]["dev_slice_sha256"] = hashlib.sha256(dev_slice).hexdigest()
    R["game_features"]["prereg_sha256"] = pre["features"]["sha256"]
    R["game_features"]["dev_file_sha256_now"] = FC.sha(FC.D(feat))
    R["game_features"]["dev_slice_hash_matches_prereg"] = (
        R["game_features"]["dev_slice_sha256"] == pre["features"]["sha256"])
    Fd = pd.read_csv(FC.D(feat))
    Ff = pd.read_csv(FC.full_path(feat))
    Fdd = dev_rows(Ff)
    R["game_features"]["dev_lv_values_equal"] = bool(
        np.array_equal(Fd.lv_home.to_numpy(), Fdd.lv_home.to_numpy())
        and np.array_equal(Fd.lv_away.to_numpy(), Fdd.lv_away.to_numpy())
        and np.array_equal(Fd.gid.to_numpy(), Fdd.gid.to_numpy()))
    lvl = lv_last_fn()
    Ld = lvl(Fd)
    Lf = lvl(Ff).reindex(Ld.index)
    R["serve_lv_last_dev_equal"] = bool(
        np.array_equal(Ld.lv_last_home.to_numpy(), Lf.lv_last_home.to_numpy(), equal_nan=True)
        and np.array_equal(Ld.lv_last_away.to_numpy(), Lf.lv_last_away.to_numpy(),
                           equal_nan=True))
    # ---------------------------------------------------------- weights
    W = json.load(open(FC.full_path("pv_nhl_compose_weights.json")))
    Wd = json.load(open(FC.D("pv_nhl_compose_weights.json")))
    R["weights"] = {
        "dev_seasons_equal_dev_file": all(
            W["walk_forward"][s]["w"] == Wd["walk_forward"][s]["w"]
            and W["walk_forward"][s]["intercept"] == Wd["walk_forward"][s]["intercept"]
            for s in map(str, FC.DEV_SEASONS)),
        "test_seasons_all_frozen": all(
            W["walk_forward"][str(s)]["w"] == W["walk_forward"][str(FC.FREEZE_SEASON)]["w"]
            for s in FC.TEST_SEASONS),
        "frozen": W["frozen_test_weights"]}
    # ---------------------------------------------------------- coverage (structural)
    g = pd.read_csv(FC.D("nhl_games.csv"), usecols=["game_id", "season", "type"])
    gt = g[g.game_id >= DEVG]
    Ft = Ff[Ff.gid >= DEVG]
    ns = FC.noshift_games()
    LLf = pd.read_csv(FC.D("pv_nhl_full_serve_lv_last.csv"))
    LLt = LLf[LLf.gid >= DEVG]
    cov = {}
    for s in FC.TEST_SEASONS:
        gs = gt[gt.season == s]
        fs = Ft[Ft.season == s]
        ls = LLt[LLt.season == s]
        cov[str(s)] = {
            "games_regular": int((gs.type == 2).sum()), "games_playoff": int((gs.type == 3).sum()),
            "feature_rows_regular": int((fs.gtype == 2).sum()),
            "feature_rows_playoff": int((fs.gtype == 3).sum()),
            "lv_missing": int(fs.lv_home.isna().sum() + fs.lv_away.isna().sum()),
            "lv_last_blank_sides": int(ls.lv_last_home.isna().sum() + ls.lv_last_away.isna().sum()),
            "games_without_shift_chart": int(fs.gid.isin(ns).sum()),
            "n_skaters_not_18_sides": int((fs.n_sk_home != 18).sum() + (fs.n_sk_away != 18).sum()),
            "starting_goalie_missing_sides": int(fs.g_start_home.isna().sum()
                                                 + fs.g_start_away.isna().sum())}
    blank = LLt[LLt.lv_last_home.isna() | LLt.lv_last_away.isna()][["gid", "date", "home", "away"]]
    R["test_coverage"] = {"per_season": cov,
                          "test_games_total": int(len(gt)),
                          "test_feature_rows": int(len(Ft)),
                          "test_games_without_feature_row": int(len(set(gt.game_id) - set(Ft.gid))),
                          "lv_last_blank_games": blank.astype(str).to_dict("records"),
                          "games_without_shift_chart_total": int(Ft.gid.isin(ns).sum())}
    checks = [R["creation_pg"]["bitwise"], R["creation_wf"]["bitwise"],
              R["finishing_player_games"]["bitwise"], R["finishing_entity_games"]["bitwise"],
              R["finishing_shots"]["bitwise"], R["finishing_team_games_lg_xg_pre"]["bitwise"],
              R["duels_csv"]["dev_prefix_text_identical"], R["duels_fo"]["bitwise"],
              R["duels_goal_values"]["dev_seasons_identical"],
              all(v["stints_identical"] and v["toi5_byte_identical"] for v in st.values()),
              R["defense_dev_fits"]["structure_identical"]
              and R["defense_dev_fits"]["max_abs_diff_all_rating_arrays"] < 1e-12,
              R["defense_values_csv"]["dev_prefix_text_identical"],
              R["compose_skater_games"]["bitwise"], R["compose_goalie_games"]["bitwise"],
              R["compose_team_games"]["bitwise"],
              R["game_features"]["dev_prefix_text_identical"],
              R["game_features"]["dev_slice_hash_matches_prereg"],
              R["serve_lv_last_dev_equal"], R["weights"]["dev_seasons_equal_dev_file"],
              R["weights"]["test_seasons_all_frozen"]]
    R["ALL_DEV_CHECKS_PASS"] = bool(all(checks))
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(OUT, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in R.items() if k != "test_coverage"}, indent=1,
                     default=str))
    print(json.dumps(R["test_coverage"]["per_season"], indent=1))


if __name__ == "__main__":
    main()
