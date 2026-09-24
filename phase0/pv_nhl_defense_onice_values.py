"""pv_nhl_defense_onice STEP 4 -- per-player-per-game walk-forward values, team-game
aggregates, and a DEV game-level read through the shipped harness.

values  data/pv_nhl_defense_onice_values.csv -- one row per dressed skater per DEV
        regular-season game (2010-11..2017-18). Each game reads the latest walk-
        forward fit (phase0/pv_nhl_defense_onice_fit.py: pre-season or monthly
        refit) whose cutoff <= the game date; every such fit was trained on
        intervals dated strictly before its cutoff (asserted there). 2010-11 has no
        prior fit -> values blank (fit = "none").
          d_xga   on-ice xG-against RAPM defence, xG/60 prevented vs an average
                  established regular of his position (positive = good)
          d_ga    on-ice goals-against RAPM (heavy ridge), goals/60 prevented
          d_gax   goals-against RAPM shrunk toward the xGA solution, goals/60
          d_ca    on-ice shot-attempts-against RAPM, attempts/60 prevented
          known   1 if the player was in the fit window (else the unseen value
                  of his position), new = career 5v5 < 20,000 s before the cutoff
          m5      EWMA (0.8/0.2) of his own PRIOR 5v5 minutes per game (walk-
                  forward; first game F 10.0 / D 14.0)
        Configs (ridge lambda, half-life) = the family's best pooled future-GA
        deviance on the DEV monthly evaluation (eval_predict_month.json): a DEV
        design choice made on player-level outcomes, never on game results.
teams   data/pv_nhl_defense_onice_team.csv -- per team-game: D_* = sum over tonight's
        dressed skaters of d * m5 / 60 (5v5 goals (xG, attempts) prevented per game
        implied by the lineup), and dD_* = D_* minus the mean D_* of the team's
        previous 10 dressed lineups re-valued with tonight's ratings and m5.
game    DEV game-level read (nhl_depth_eval Ctx/Folds, LOSO 2011-12..2017-18,
        n = 7,929): harness sanity S1 (shipped 0.672398) and S2 (drop-xG cost
        0.00303) first, then each pre-declared feature BESIDE the shipped blend:
          G1 D_xga diff   G2 D_ga diff   G3 D_gax diff   G4 dD_xga diff
          G5 D_ca diff
        Home-minus-away differences. Bar (prereg): >= +0.00100, CI lo > 0.

PROTOCOL: DEV only; every frame asserted gid < 2018000000; market-blind.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_io import DEV_MAX_GID, load_rosters  # noqa: E402
import pv_nhl_defense_onice_fit as F  # noqa: E402

PFX = F.PFX
FAMS = {"d_xga": "XGA", "d_ga": "GA", "d_gax": "GAX", "d_ca": "CA"}
M_DECAY = 0.8
M_INIT = {"F": 10.0, "D": 14.0}
DP_WIN = 10


def configs():
    d = json.load(open(f"{PFX}eval_predict_month.json"))["families"]["g"]
    return {col: d[fam]["in_sample_best"] for col, fam in FAMS.items()}


def all_fits():
    specs = F.wf_specs()
    return sorted((sp["cutoff"], sp["name"]) for sp in specs)


def cmd_values():
    t0 = time.time()
    cfg = configs()
    print("configs", cfg, flush=True)
    ro = load_rosters()
    ro = ro[(ro.gtype == 2) & (ro.pos != "G")].copy()
    g = pd.read_csv("data/nhl_games.csv", usecols=["game_id", "date", "type"])
    g = g[(g.game_id < DEV_MAX_GID) & (g.type == 2)]
    dmap = dict(zip(g.game_id, g.date.str.replace("-", "").astype(int)))
    ro["date"] = ro.gid.map(dmap)
    ro = ro[ro.date.notna()].copy()
    ro["date"] = ro.date.astype(np.int64)
    ro["posFD"] = np.where(ro.pos == "D", "D", "F")
    assert ro.gid.max() < DEV_MAX_GID
    fits = all_fits()
    cut = np.array([c for c, _ in fits])
    which = np.searchsorted(cut, ro.date.to_numpy(), side="right") - 1
    ro["fit"] = [fits[k][1] if k >= 0 else "none" for k in which]
    for col in list(FAMS) + ["known", "new", "Tprior"]:
        ro[col] = np.nan
    for k, (c, name) in enumerate(fits):
        m = (which == k)
        if not m.any():
            continue
        assert ro.date.to_numpy()[m].min() >= c
        z = np.load(f"{PFX}fit_{name}.npz")
        pids = z["pids"]
        q = ro.pid.to_numpy()[m]
        i = np.clip(np.searchsorted(pids, q), 0, len(pids) - 1)
        ok = pids[i] == q
        isD = ro.posFD.to_numpy()[m] == "D"
        idx = np.nonzero(m)[0]
        for col, key in cfg.items():
            uF, uD = z["u" + key[1:]]
            v = np.where(ok, z[key][i], np.where(isD, uD, uF))
            ro.iloc[idx, ro.columns.get_loc(col)] = v
        ro.iloc[idx, ro.columns.get_loc("known")] = ok.astype(float)
        ro.iloc[idx, ro.columns.get_loc("new")] = np.where(ok, z["new"][i], True).astype(float)
        ro.iloc[idx, ro.columns.get_loc("Tprior")] = np.where(ok, z["Traw"][i], 0.0)
    # m5: walk-forward EWMA of own prior 5v5 minutes per game
    toi = F.toi5_all()
    sec = dict(zip(zip(toi.gid.to_numpy(), toi.pid.to_numpy()), toi.sec5.to_numpy()))
    ro = ro.sort_values(["date", "gid", "side", "pid"]).reset_index(drop=True)
    m5 = {}
    last_gid = {}
    m5col = np.empty(len(ro))
    gids = ro.gid.to_numpy()
    pids = ro.pid.to_numpy()
    posv = ro.posFD.to_numpy()
    # read all of a game's players before any update from that game
    starts = np.r_[0, np.nonzero(np.diff(gids))[0] + 1, len(ro)]
    for a, b in zip(starts[:-1], starts[1:]):
        for r in range(a, b):
            p = pids[r]
            if p in m5:
                assert last_gid[p] != gids[r]
            m5col[r] = m5.get(p, M_INIT[posv[r]])
        for r in range(a, b):
            p = pids[r]
            x = sec.get((gids[r], p), 0.0) / 60.0
            m5[p] = M_DECAY * m5[p] + (1 - M_DECAY) * x if p in m5 else x
            last_gid[p] = gids[r]
    ro["m5"] = m5col
    cols = ["gid", "date", "season", "team", "side", "pid", "pos", "fit", "known", "new",
            "Tprior", "m5"] + list(FAMS)
    out = ro[cols]
    out.to_csv(f"{PFX}values.csv", index=False, float_format="%.5f")
    print(f"values rows {len(out):,} ({time.time()-t0:.0f}s)", flush=True)

    # team-game aggregates + deviation from the previous 10 lineups
    recent = defaultdict(lambda: deque(maxlen=DP_WIN))
    rows = []
    vals_by_pid_now = {}
    out = out.sort_values(["date", "gid", "side"]).reset_index(drop=True)
    for (date, gid, side), grp in out.groupby(["date", "gid", "side"], sort=True):
        team = grp.team.iloc[0]
        rec = {"gid": gid, "date": date, "season": grp.season.iloc[0], "team": team,
               "side": side, "n_sk": len(grp), "fit": grp.fit.iloc[0]}
        cur = {}
        for col in FAMS:
            v = grp[col].to_numpy()
            rec[f"D_{col[2:]}"] = float(np.nansum(v * grp.m5.to_numpy() / 60.0)) \
                if not np.isnan(v).all() else np.nan
        # dD: previous lineups re-valued with tonight's fit and each player's m5
        if grp.fit.iloc[0] != "none":
            z = _fit_cache(grp.fit.iloc[0])
            prev = list(recent[team])
            if prev:
                tonight = dict(zip(grp.pid.to_numpy(), grp.m5.to_numpy()))
                for col, key in configs_cached().items():
                    if col not in ("d_xga", "d_ga"):
                        continue
                    tot = []
                    for lineup in prev:
                        tot.append(sum(_val(z, key, p, ps) * m5_now(p, ps, tonight)
                                       for p, ps in lineup) / 60.0)
                    rec[f"dD_{col[2:]}"] = rec[f"D_{col[2:]}"] - float(np.mean(tot))
            rec["n_prev"] = len(prev)
        recent[team].append(list(zip(grp.pid.to_numpy(), grp.pos.to_numpy())))
        _M5_NOW.update(dict(zip(grp.pid.to_numpy(), grp.m5.to_numpy())))
        rows.append(rec)
    td = pd.DataFrame(rows)
    assert td.gid.max() < DEV_MAX_GID
    td.to_csv(f"{PFX}team.csv", index=False, float_format="%.5f")
    print(f"team-game rows {len(td):,} ({time.time()-t0:.0f}s)", flush=True)


_FC = {}
_CFG = None
_M5_NOW = {}


def configs_cached():
    global _CFG
    if _CFG is None:
        _CFG = configs()
    return _CFG


def _fit_cache(name):
    if name not in _FC:
        z = np.load(f"{PFX}fit_{name}.npz")
        d = {k: z[k] for k in z.files}
        d["_idx"] = {int(p): i for i, p in enumerate(d["pids"])}
        _FC.clear()
        _FC[name] = d
    return _FC[name]


def _val(z, key, pid, pos):
    i = z["_idx"].get(int(pid))
    if i is not None:
        return float(z[key][i])
    uF, uD = z["u" + key[1:]]
    return float(uD if pos == "D" else uF)


def m5_now(pid, pos, tonight):
    """Tonight's m5 for a player dressed tonight, else his m5 as read before his
    latest earlier game (never an update from tonight or later)."""
    if pid in tonight:
        return tonight[pid]
    return _M5_NOW.get(pid, M_INIT["D" if pos == "D" else "F"])


# -------------------------------------------------------------------- game ---
def cmd_game():
    import nhl_depth_eval as Hd
    from nhl_features_eval import TEAM_FIX
    from nhl_glicko2_eval import llv
    t0 = time.time()
    ctx = Hd.Ctx()
    games = ctx.games
    y = ctx.yv()
    folds = ctx.folds
    mask = ctx.mask
    Hd.assert_dev_only(ctx.seasons[mask])
    res = {"protocol": {"n_dev": int(mask.sum()), "eval": "LOSO 2011-12..2017-18",
                        "market_blind": True, "test_seasons_touched": 0,
                        "arms_pre_declared": ["G1 D_xga", "G2 D_ga", "G3 D_gax",
                                              "G4 dD_xga", "G5 D_ca"]}}
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    s1 = float(llv(y, p0).mean())
    pnx = folds.loso_pred(X0[:, :5], y)
    s2 = float(llv(y, pnx).mean() - s1)
    res["S1"] = {"shipped_dev_loso_ll": round(s1, 6), "target": 0.672398,
                 "pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929)}
    res["S2"] = {"drop_xg_cost": round(s2, 5), "target": 0.00303,
                 "pass": bool(abs(s2 - 0.00303) <= 0.0002)}
    print(res["S1"], res["S2"], flush=True)
    td = pd.read_csv(f"{PFX}team.csv")
    assert td.gid.max() < DEV_MAX_GID
    td["team_n"] = td.team.map(lambda t: TEAM_FIX.get(t, t))
    key = {(int(g), t): i for i, (g, t) in enumerate(zip(td.gid, td.team_n))}
    cols = ["D_xga", "D_ga", "D_gax", "dD_xga", "D_ca"]
    feat = {c: np.full(len(games), np.nan) for c in cols}
    miss = 0
    for gi, gm in enumerate(games):
        h = key.get((gm["game_id"], TEAM_FIX.get(gm["home"], gm["home"])))
        a = key.get((gm["game_id"], TEAM_FIX.get(gm["away"], gm["away"])))
        if h is None or a is None:
            miss += 1
            continue
        for c in cols:
            feat[c][gi] = td[c].iat[h] - td[c].iat[a]
    res["coverage_missing_games"] = miss
    arms = {"G1": "D_xga", "G2": "D_ga", "G3": "D_gax", "G4": "dD_xga", "G5": "D_ca"}
    for arm, c in arms.items():
        f = feat[c][mask]
        cov = float(np.mean(~np.isnan(f)))
        f = np.nan_to_num(f)
        X = np.column_stack([X0, f])
        p = folds.loso_pred(X, y)
        st = Hd.fold_stats(y, folds, p0, p)
        st["coverage"] = round(cov, 4)
        st["feature_sd"] = round(float(np.std(f)), 5)
        w = Hd.fit_logit(X, y)
        st["coef_full"] = round(float(w[-1]), 4)
        res[arm] = {"feature": c, **st}
        print(arm, c, st, flush=True)
    json.dump(res, open(f"{PFX}game.json", "w"), indent=1)
    print(f"wrote {PFX}game.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "values"
    if cmd == "values":
        cmd_values()
    elif cmd == "game":
        cmd_game()
    else:
        raise SystemExit("usage: values | game")
