"""NHL breakthrough DEV screen nhl_gmar -- day-of personnel composite.

Candidate: starting-goalie LEVEL inside the W/L Elo with a reliability-calibrated
EB prior (construction G, phase0/bt_nhl_gmar_gl.py), plus the absent-regular
deficit valued in minutes ABOVE REPLACEMENT beside the blend (construction S).

Binding protocol: documents/breakthrough_program_prereg_2026_09_24.md
  * market-blind: odds appear ONLY in the evaluation diagnostic (model-minus-close
    on DEV games), never in an input or a training target;
  * DEV only: every rating runs on nhl_depth_eval.dev_games() (hard-asserts season
    <= 2017-18); every data file is filtered to game_id < 2018000000 on read;
    every scored mask passes nhl_depth_eval.assert_dev_only; no 2018-19+ game is
    ever loaded, scored or printed;
  * walk-forward: W1 (runtime read-before-write guards), W2 (future-perturbation
    invariance at three cutoffs), W3 (m_E / rho / xbar from completed seasons or
    strictly earlier dates, asserted), W4 (identical eval-mask hash, no NaN
    personnel feature), W5 (only tonight's starter identity + dressed membership
    are same-game inputs; both public pre-game).

Harness: phase0/nhl_depth_eval.py Ctx, leave-one-season-out over 2011-12..2017-18
(7 folds, n = 7,929; ctx.mask kept exactly for every arm).  Gains vs the shipped
blend p0 = folds.loso_pred(X0, y).  Grids are NESTED (pair_loss pattern of
nhl_depth_eval.axis_a); the bar is judged ONLY on the nested prediction.

Arms
  A0 shipped blend (reproduction, S1)
  A1 GL 'new' replacing column 1, gamma nested in {0,60,120,180}
  A2 shipped blend + mar_diff
  A3 GL 'new' (gamma nested) + mar_diff beside         <- PRIMARY (only bar arm)
  A4 GL probe c=64 role prior in column 1 + anatomy SKATER block (anchor)
  A5 GL 'new' + D INSIDE the W/L Elo, (gamma, delta) nested jointly (diagnostic)
  A6 A3 with the starter replaced by a pre-game projection (serving diagnostic)

Outputs: data/bt_nhl_nhl_gmar.json (mirrored to data/bt_nhl_gmar.json, the name
in the spec) and data/bt_nhl_gmar_feats.csv (per-game DEV features).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, DEV_WARM_BEFORE, TEST_START, llv  # noqa: E402
from nhl_features_eval import TEAM_FIX  # noqa: E402
import bt_nhl_anatomy_build as B  # noqa: E402
import bt_nhl_gmar_gl as GL  # noqa: E402

OUT = "data/bt_nhl_nhl_gmar.json"
OUT_MIRROR = "data/bt_nhl_gmar.json"
OUT_FEATS = "data/bt_nhl_gmar_feats.csv"
MAX_GID = 2018000000
assert DEV_END == 20172018 and TEST_START == 20182019 and DEV_WARM_BEFORE == 20112012

GAMMAS = [0.0, 60.0, 120.0, 180.0]          # Elo per goal/game (pre-declared grid)
DELTAS = [0.0, 0.5, 1.0, 2.0]               # Elo per minute above replacement (A5)
PROBE_C = 64.0

# construction S constants (fixed a priori)
WIN = 10          # team games defining usual regulars
REG_MIN = 7       # regular = dressed in >= 7 of the last 10
M_DECAY = 0.8     # m <- 0.8 m + 0.2 toi
RHO_INIT = 11.5   # replacement TOI before ANY fill-in data (2010-10-07 only; warm-up)

NB10K = 10000
BOOT_SEED = 7
LATE_FROM = 20152016

FACE_DATE = "2015-12-01"
PRICE, CONDON = 8471679, 8477237


def norm(t):
    return TEAM_FIX.get(t, t)


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ loading --
def load_inputs(games):
    ids = set(g["game_id"] for g in games)
    assert max(ids) < MAX_GID, "TEST-era game id in DEV list"
    starters = B.load_starters(ids)          # filters game_id < MAX_GID on read
    gg, goalie_ids = B.load_goalie_games(ids)  # filters game_id < MAX_GID on read
    dressed = B.load_dressed(ids, goalie_ids)  # filters game_id < MAX_GID on read
    for d in (starters, gg, dressed):
        assert not d or max(d) < MAX_GID
    return starters, gg, dressed


# ------------------------------------------------ construction S (MAR) -------
def build_mar(games, dressed, anatomy=False):
    """Absent-regular deficit per game and side.

    new mode (anatomy=False):
      m_p  : player's trailing TOI (min), EWMA over his own prior dressed games on
             any team (m <- 0.8 m + 0.2 toi, initialised at his first game)
      U(T) : skaters with >= 7 appearances in T's last 10 games (window runs across
             seasons, no reset; need = ceil(0.7 * games) while T has < 10); a player
             leaves U(T) once his most recent prior game was for another team
      rho  : running mean TOI of fill-in player-games (dressed, not in U of his team
             that night) over games on strictly earlier dates
      D    : sum over absent p in U of max(0, m_p - rho)
    anatomy mode (S4b): season reset, 5-game minimum, regular TOI = mean over the
      window, raw team keys, no departure rule; returns raw absent TOI (min).
    Only tonight's dressed MEMBERSHIP is read before the game; tonight's toi_s is
    used only in the update step.
    """
    n = len(games)
    D = {"home": np.zeros(n), "away": np.zeros(n)}
    RAW = {"home": np.full(n, np.nan), "away": np.full(n, np.nan)}
    NABS = {"home": np.zeros(n), "away": np.zeros(n)}
    MISS = {"home": np.zeros(n, dtype=bool), "away": np.zeros(n, dtype=bool)}
    rho_arr = np.empty(n)
    hist = defaultdict(lambda: deque(maxlen=WIN))
    m = {}
    m_w = {}                 # W1 guard: index of last write of m_p
    last_team = {}
    rho_sum, rho_cnt = 0.0, 0
    pend_sum, pend_cnt = 0.0, 0
    cur_date, last_committed = None, ""
    prev_season = None
    for i, g in enumerate(games):
        gid = g["game_id"]
        dstr = g["date"]
        if dstr != cur_date:
            if cur_date is not None:
                assert cur_date < dstr
                rho_sum += pend_sum
                rho_cnt += pend_cnt
                pend_sum, pend_cnt = 0.0, 0
                last_committed = cur_date
            cur_date = dstr
        assert last_committed < dstr                    # W3: rho strictly earlier dates
        if anatomy and prev_season is not None and g["season"] != prev_season:
            for t in hist:
                hist[t].clear()
        prev_season = g["season"]
        rho = (rho_sum / rho_cnt) if rho_cnt else RHO_INIT
        rho_arr[i] = rho
        gdr = dressed.get(gid, {})
        pre = {}
        for side in ("home", "away"):
            key = g[side] if anatomy else norm(g[side])
            dr = gdr.get(key)
            lh = hist[key]
            ng = len(lh)
            cnt = defaultdict(int)
            tsum = defaultdict(float)
            for lu in lh:
                for pid, toi in lu.items():
                    cnt[pid] += 1
                    tsum[pid] += toi
            need = REG_MIN if ng >= WIN else int(np.ceil(0.7 * ng))
            if anatomy:
                U = [p for p in cnt if cnt[p] >= need]
            else:
                U = [p for p in cnt if cnt[p] >= need and last_team.get(p) == key]
            pre[side] = (key, dr, set(U))
            if not dr:
                MISS[side][i] = True
                continue
            if anatomy:
                if ng < 5:
                    continue
                regs = {p: tsum[p] / cnt[p] / 60.0 for p in cnt if cnt[p] >= need}
                miss = [p for p in regs if p not in dr]
                RAW[side][i] = sum(regs[p] for p in miss)
                NABS[side][i] = len(miss)
            else:
                miss = [p for p in U if p not in dr]
                for p in miss:
                    assert m_w[p] < i, "W1 violated (m_p)"
                RAW[side][i] = sum(m[p] for p in miss)
                D[side][i] = sum(max(0.0, m[p] - rho) for p in miss)
                NABS[side][i] = len(miss)
        # ---------------- updates AFTER reading ----------------
        for side in ("home", "away"):
            key, dr, Uset = pre[side]
            if not dr:
                continue
            for pid, toi in dr.items():
                tm = toi / 60.0
                if pid not in Uset:
                    pend_sum += tm
                    pend_cnt += 1
                m[pid] = tm if pid not in m else M_DECAY * m[pid] + (1 - M_DECAY) * tm
                m_w[pid] = i
                last_team[pid] = key
            hist[key].append(dr)
    return {"D_h": D["home"], "D_a": D["away"], "raw_h": RAW["home"], "raw_a": RAW["away"],
            "nabs_h": NABS["home"], "nabs_a": NABS["away"],
            "miss_h": MISS["home"], "miss_a": MISS["away"], "rho": rho_arr}


# ------------------------------------------------------------- feature set ---
def build_features(games, starters, gg, dressed, full=True):
    """Every personnel feature this screen uses (full-length over games)."""
    F = {}
    S = build_mar(games, dressed)
    F.update({k: S[k] for k in ("D_h", "D_a", "raw_h", "raw_a", "nabs_h", "nabs_a",
                                "rho")})
    F["miss_h"], F["miss_a"] = S["miss_h"], S["miss_a"]
    F["mar_diff"] = S["D_a"] - S["D_h"]
    for gm in GAMMAS:
        r = GL.gl_elo(games, starters, gg, gm, mode="new",
                      snap=(FACE_DATE, [PRICE, CONDON]) if gm == 0.0 else None)
        F[f"gl_p_{gm:g}"] = r["p"]
        F[f"R_h_{gm:g}"], F[f"R_a_{gm:g}"] = r["R_h"], r["R_a"]
        if gm == 0.0:
            F["G_h"], F["G_a"] = r["G_h"], r["G_a"]
            F["gk_h"], F["gk_a"] = r["gk_h"], r["gk_a"]
            F["fb_h"], F["fb_a"] = r["fb_h"], r["fb_a"]
            F["_me"], F["_snap"] = r["me"], r["snap"]
    if full:
        for gm in GAMMAS:
            for dl in DELTAS:
                if dl == 0.0:
                    F[f"gld_p_{gm:g}_{dl:g}"] = F[f"gl_p_{gm:g}"]
                    continue
                F[f"gld_p_{gm:g}_{dl:g}"] = GL.gl_elo(
                    games, starters, gg, gm, mode="new",
                    d_home=S["D_h"], d_away=S["D_a"], delta=dl)["p"]
        proj = GL.project_starters(games, starters)
        F["_proj"] = proj
        for gm in GAMMAS:
            r = GL.gl_elo(games, starters, gg, gm, mode="new", starter_override=proj)
            F[f"gl6_p_{gm:g}"] = r["p"]
            if gm == 0.0:
                F["fb6_h"], F["fb6_a"] = r["fb_h"], r["fb_a"]
                F["G6_h"], F["G6_a"] = r["G_h"], r["G_a"]
        F["probe_p"] = GL.gl_elo(games, starters, gg, PROBE_C, mode="probe",
                                 role_prior=True)["p"]
    return F


# ---------------------------------------------------------------- stats -------
def boot10k(d, seed=BOOT_SEED, nb=NB10K, chunk=1000):
    rng = np.random.default_rng(seed)
    n = len(d)
    out = []
    for _ in range(nb // chunk):
        out.append(d[rng.integers(0, n, size=(chunk, n))].mean(axis=1))
    bs = np.concatenate(out)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


def arm_stats(y, folds, p0, p, seas):
    st = Hd.fold_stats(y, folds, p0, p)               # 4,000 draws, seed 7 (harness)
    d = llv(y, p0) - llv(y, p)
    st["boot_ci_4k_seed7"] = st.pop("boot_ci")
    st["boot_ci_10k"] = boot10k(d)
    st["gain_exact"] = float(d.mean())
    st["dev_ll"] = round(float(llv(y, p).mean()), 6)
    late = seas >= LATE_FROM
    Hd.assert_dev_only(seas[late])
    dl = d[late]
    st["late_dev_2015_18"] = {"n": int(late.sum()), "gain": round(float(dl.mean()), 5),
                              "boot_ci_10k": boot10k(dl)}
    st["sig"] = ("SIG" if st["boot_ci_10k"][0] > 0 else
                 ("SIG WORSE" if st["boot_ci_10k"][1] < 0 else "n.s."))
    return st


def nested(cells, y, folds):
    """cells: ordered dict cell -> X.  Returns nested pred, picks, per-cell LOSO."""
    fl = folds.list
    pairs = [(a, b) for i, a in enumerate(fl) for b in fl[i + 1:]]
    loso, pair_loss = {}, {}
    for c, X in cells.items():
        assert not np.isnan(X).any(), f"NaN in design for cell {c}"
        loso[c] = folds.loso_pred(X, y)
        pl = {}
        w0 = None
        for a, b in pairs:
            tr = (folds.seasons != a) & (folds.seasons != b)
            w = Hd.fit_logit(X[tr], y[tr], Hd.BLEND_C, w0)
            w0 = w
            for s, o in ((a, b), (b, a)):
                mo = folds.idx[o]
                pl[(s, o)] = float(llv(y[mo], Hd._sig(X[mo] @ w)).mean())
        pair_loss[c] = pl
    pred = np.empty(len(y))
    picks = {}
    for s in fl:
        inner = [o for o in fl if o != s]
        tot = sum(folds.n[o] for o in inner)

        def score(c):
            return sum(folds.n[o] * pair_loss[c][(s, o)] for o in inner) / tot
        cstar = min(cells, key=score)             # ties -> first in grid order
        picks[s] = cstar
        X = cells[cstar]
        tr = folds.seasons != s
        w = Hd.fit_logit(X[tr], y[tr], Hd.BLEND_C)
        pred[folds.idx[s]] = Hd._sig(X[folds.idx[s]] @ w)
    cell_ll = {c: float(llv(y, loso[c]).mean()) for c in cells}
    best = min(cells, key=lambda c: cell_ll[c])
    return pred, picks, loso, cell_ll, best


# ------------------------------------------------------------- W2 --------------
def perturb(games, starters, gg, dressed, cutoff, seed):
    """Permute y, goalie rows (GA/xGA), dressed lists (membership + TOI) and
    starters among games on/after the cutoff date."""
    rng = np.random.default_rng(seed)
    after = [i for i, g in enumerate(games) if g["date"] >= cutoff]
    perm = rng.permutation(after)
    g2 = [dict(g) for g in games]
    for i, j in zip(after, perm):
        g2[i]["y"] = games[j]["y"]
    st2, gg2, dr2 = dict(starters), dict(gg), dict(dressed)
    for i, j in zip(after, perm):
        a, b = games[i]["game_id"], games[j]["game_id"]
        st2[a] = starters.get(b, {})
        gg2[a] = gg.get(b, [])
        dr2[a] = dressed.get(b, {})
    changed = sum(1 for i, j in zip(after, perm) if i != j)
    return g2, st2, gg2, dr2, len(after), changed


def feats_equal_before(F1, F2, games, cutoff):
    pre = np.array([g["date"] < cutoff for g in games])
    bad = []
    nchk = 0
    for k, v in F1.items():
        if k.startswith("_"):
            continue
        a, b = np.asarray(v), np.asarray(F2[k])
        nchk += 1
        if a.dtype.kind == "f":
            same = np.array_equal(a[pre], b[pre], equal_nan=True)
        else:
            same = np.array_equal(a[pre], b[pre])
        if not same:
            bad.append(k)
    # also: the post-cutoff part must actually differ somewhere (non-vacuous test)
    post_diff = sum(1 for k, v in F1.items() if not k.startswith("_")
                    and np.asarray(v).dtype.kind == "f"
                    and not np.array_equal(np.asarray(v)[~pre], np.asarray(F2[k])[~pre],
                                           equal_nan=True))
    return {"n_pre_games": int(pre.sum()), "n_features_checked": nchk,
            "mismatched_features": bad, "n_features_changed_after_cutoff": post_diff,
            "pass": (not bad) and post_diff > 0}


# ------------------------------------------------------------------- main -----
def main():
    t0 = time.time()
    ctx = Hd.Ctx()
    games = ctx.games
    assert max(g["season"] for g in games) <= DEV_END
    y = ctx.yv()
    folds = ctx.folds
    mask = ctx.mask
    seas = ctx.seasons[mask]
    Hd.assert_dev_only(seas)
    res = {"candidate": "nhl_gmar",
           "title": "Day-of personnel composite: starting-goalie level inside the W/L "
                    "Elo (reliability-calibrated EB prior) + absent-regular deficit in "
                    "minutes above replacement beside the blend",
           "protocol": {"dev_seasons_scored": sorted(set(seas.tolist())),
                        "n_dev": int(mask.sum()), "eval": "LOSO over DEV seasons, nested grids",
                        "market_blind": True, "test_seasons_touched": 0,
                        "bar": "nested A3 gain >= +0.00100 AND 95% CI lo > 0 AND S1-S4, W1-W5 pass"}}

    starters, gg, dressed = load_inputs(games)
    print(f"DEV games {len(games)}  masked {int(mask.sum())}  loaded in {time.time()-t0:.1f}s",
          flush=True)

    # ============================ HARNESS SANITY ============================
    san = {}
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    l0 = llv(y, p0)
    s1 = float(l0.mean())
    san["S1"] = {"shipped_dev_loso_ll": round(s1, 6), "n": int(len(y)),
                 "target": 0.672398, "pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929)}
    pnx = folds.loso_pred(X0[:, :5], y)
    s2 = float(llv(y, pnx).mean() - s1)
    san["S2"] = {"drop_xg_cost": round(s2, 5), "target": 0.00303, "tol": 0.0002,
                 "recorded_0811_sweep": 0.00304, "ledger_row5_TEST_recorded": 0.00316,
                 "pass": bool(abs(s2 - 0.00303) <= 0.0002)}
    print(f"S1 shipped DEV LOSO {s1:.6f} n={len(y)}  S2 drop-xG cost {s2:+.5f}", flush=True)

    F = build_features(games, starters, gg, dressed, full=True)
    print(f"features built {time.time()-t0:.1f}s", flush=True)

    ref = Hd.run_elo_arr(games, *Hd.SHIPPED_ELO)
    s3d = float(np.abs(F["gl_p_0"] - ref).max())
    Xg0 = X0.copy()
    Xg0[:, 1] = logit(F["gl_p_0"])[mask]
    pg0 = folds.loso_pred(Xg0, y)
    s3b = float(np.abs(pg0 - p0).max())
    san["S3"] = {"gamma0_vs_run_elo_arr_maxabs": s3d, "blend_pred_maxabs_diff": s3b,
                 "blend_ll": round(float(llv(y, pg0).mean()), 6),
                 "pass": bool(s3d < 1e-12 and s3b < 1e-12)}

    # S4a: probe c=64 role prior, CI via the probe's RNG stream (4th boot_mean call)
    Xp = X0.copy()
    Xp[:, 1] = logit(F["probe_p"])[mask]
    pp = folds.loso_pred(Xp, y)
    dp = l0 - llv(y, pp)
    rng = np.random.default_rng(20260924)
    for _ in range(3):
        rng.integers(0, len(dp), size=(4000, len(dp)))
    bs = dp[rng.integers(0, len(dp), size=(4000, len(dp)))].mean(axis=1)
    ci4 = [round(float(x), 5) for x in np.percentile(bs, [2.5, 97.5])]
    del bs
    s4a_ll = round(float(llv(y, pp).mean()), 6)
    san["S4a"] = {"probe_dev_ll": s4a_ll, "gain": round(float(dp.mean()), 5), "ci": ci4,
                  "target": {"dev_ll": 0.672103, "gain": 0.00029, "ci": [-0.00013, 0.00073]},
                  "pass": bool(s4a_ll == 0.672103 and round(float(dp.mean()), 5) == 0.00029
                               and ci4 == [-0.00013, 0.00073])}

    # S4b: anatomy bookkeeping reproduces bt_nhl_anatomy_feats abs_toi
    SA = build_mar(games, dressed, anatomy=True)
    af = pd.read_csv("data/bt_nhl_anatomy_feats.csv")
    af = af[af.game_id < MAX_GID].set_index("game_id")
    gids = [g["game_id"] for g in games]
    af = af.loc[gids]
    assert af.season.max() <= DEV_END
    eq = []
    for side, k in (("home", "raw_h"), ("away", "raw_a")):
        a = af[f"abs_toi_{side}"].to_numpy()
        b = SA[k]
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=0, atol=1e-6)
        eq.append(both_nan | close)
    eq_all = eq[0] & eq[1]
    eq_m = eq_all[mask]
    san["S4b"] = {"share_games_equal_all_dev": round(float(eq_all.mean()), 6),
                  "share_games_equal_masked": round(float(eq_m.mean()), 6),
                  "n_games": int(len(eq_all)), "n_mismatch": int((~eq_all).sum()),
                  "pass": bool(eq_all.mean() >= 0.999)}
    for k, v in san.items():
        print(f"  {k}: pass={v['pass']}  {v}", flush=True)
    res["harness_sanity"] = san

    # ============================ WALK-FORWARD ============================
    wf = {"W1": {"statement": "single ordered pass; every per-goalie (N, X, appearance "
                              "count), per-player (m_p, last team), per-team (R, 10-game "
                              "window, previous starter) state for game i is read before "
                              "any of game i's result, GA, xGA or TOI is applied; enforced "
                              "at runtime by write-index guards (assert last write < i) in "
                              "gl_elo and build_mar",
                 "runtime_guards_raised": 0, "pass": True}}
    w2 = {}
    for ci_, cut in enumerate(("2012-02-01", "2015-01-15", "2017-12-01")):
        g2, st2, gg2, dr2, n_after, n_changed = perturb(games, starters, gg, dressed, cut,
                                                         seed=100 + ci_)
        F2 = build_features(g2, st2, gg2, dr2, full=True)
        r = feats_equal_before(F, F2, games, cut)
        r.update({"n_games_permuted_after": n_after, "n_moved": n_changed})
        w2[cut] = r
        print(f"  W2 cutoff {cut}: pass={r['pass']} pre={r['n_pre_games']} "
              f"checked={r['n_features_checked']} changed_after={r['n_features_changed_after_cutoff']}"
              f" mismatched={r['mismatched_features']}", flush=True)
    wf["W2"] = {"cutoffs": w2, "pass": all(v["pass"] for v in w2.values()),
                "note": "y, goalie GA/xGA rows, dressed lists (membership+TOI) and starters "
                        "permuted among games on/after each cutoff; all personnel features "
                        "(GL p for every gamma, G, D, raw absent TOI, rho, A5 cells, A6 "
                        "projection arm, probe arm) rebuilt and compared bitwise before it"}
    me = F["_me"]
    wf["W3"] = {"m_E_by_season": {str(k): {"m_E": round(v["m_E"], 6),
                                           "evidence_xga": round(v["evidence_xga"], 1),
                                           "from_seasons": v["from_seasons"],
                                           "fallback": v["fallback"]} for k, v in me.items()},
                "m_E_uses_completed_seasons_only": all(all(x < int(k) for x in v["from_seasons"])
                                                       for k, v in me.items()),
                "rho_xbar": "committed only on date change; assert last_committed_date < "
                            "today inside both loops",
                "pass": all(all(x < int(k) for x in v["from_seasons"]) for k, v in me.items())}
    pers = {k: F[k][mask] for k in ("G_h", "G_a", "D_h", "D_a", "mar_diff", "rho")}
    for gm in GAMMAS:
        pers[f"gl_p_{gm:g}"] = F[f"gl_p_{gm:g}"][mask]
        pers[f"gl6_p_{gm:g}"] = F[f"gl6_p_{gm:g}"][mask]
    nan_any = {k: int(np.isnan(v).sum()) for k, v in pers.items() if np.isnan(v).any()}
    mask_hash = hashlib.sha1(mask.tobytes()).hexdigest()
    y_hash = hashlib.sha1(y.tobytes()).hexdigest()

    # ============================ ARMS ============================
    arms = {}
    arm_hash = {}
    base = [X0[:, 2], X0[:, 3], X0[:, 4], X0[:, 5]]
    mar = F["mar_diff"][mask]

    def rec(name, p, extra=None):
        st = arm_stats(y, folds, p0, p, seas)
        if extra:
            st.update(extra)
        arms[name] = st
        arm_hash[name] = mask_hash
        print(f"  {name:<44} LL {st['dev_ll']:.6f} gain {st['gain_exact']:+.5f} "
              f"CI10k [{st['boot_ci_10k'][0]:+.5f},{st['boot_ci_10k'][1]:+.5f}] "
              f"{st['n_pos_folds']}/{st['n_folds']} folds  late {st['late_dev_2015_18']['gain']:+.5f}",
              flush=True)
        return st

    rec("A0 shipped blend (reproduction)", p0)

    def picks_json(picks):
        return {str(s): (list(c) if isinstance(c, tuple) else c) for s, c in picks.items()}

    # A1 goalie only
    cells1 = {gm: Hd.design([logit(F[f"gl_p_{gm:g}"])[mask]] + base) for gm in GAMMAS}
    pA1, pk1, loso1, ll1, best1 = nested(cells1, y, folds)
    rec("A1 GL new col1 (gamma nested)", pA1,
        {"picks": picks_json(pk1), "cell_loso_ll": {f"{k:g}": round(v, 6) for k, v in ll1.items()},
         "best_single_cell_OPTIMISTIC": {"gamma": best1,
                                         "gain": round(s1 - ll1[best1], 5)}})
    # A2 skater only
    pA2 = folds.loso_pred(np.column_stack([X0, mar]), y)
    wA2 = Hd.fit_logit(np.column_stack([X0, mar]), y)
    rec("A2 shipped + mar_diff", pA2,
        {"coef_mar_diff_all_dev_descriptive": round(float(wA2[-1]), 5)})
    # A3 PRIMARY
    cells3 = {gm: np.column_stack([Hd.design([logit(F[f"gl_p_{gm:g}"])[mask]] + base), mar])
              for gm in GAMMAS}
    pA3, pk3, loso3, ll3, best3 = nested(cells3, y, folds)
    stA3 = rec("A3 GL new (gamma nested) + mar_diff [PRIMARY]", pA3,
               {"picks": picks_json(pk3),
                "cell_loso_ll": {f"{k:g}": round(v, 6) for k, v in ll3.items()},
                "best_single_cell_OPTIMISTIC": {"gamma": best3,
                                                "gain": round(s1 - ll3[best3], 5)}})
    # A4 probe anchor
    def zdiff(a, h):
        d = af[a].to_numpy()[mask] - af[h].to_numpy()[mask]
        z = (d - np.nanmean(d)) / np.nanstd(d)
        return np.nan_to_num(z, nan=0.0)
    skb = [zdiff("abs_toi_away", "abs_toi_home"), zdiff("abs_top4_away", "abs_top4_home"),
           zdiff("new_n_away", "new_n_home")]
    XA4 = np.column_stack([Xp] + skb)
    rec("A4 probe c=64 col1 + anatomy SKATER block", folds.loso_pred(XA4, y),
        {"note": "anchor; skater block z-scored on the DEV mask (NaN->0) as specified"})
    # A5 D inside Elo
    cells5 = {(gm, dl): Hd.design([logit(F[f"gld_p_{gm:g}_{dl:g}"])[mask]] + base)
              for gm in GAMMAS for dl in DELTAS}
    pA5, pk5, loso5, ll5, best5 = nested(cells5, y, folds)
    rec("A5 GL new + D inside Elo ((gamma,delta) nested)", pA5,
        {"picks": picks_json(pk5),
         "cell_loso_ll": {f"{k[0]:g},{k[1]:g}": round(v, 6) for k, v in ll5.items()},
         "best_single_cell_OPTIMISTIC": {"gamma_delta": list(best5),
                                         "gain": round(s1 - ll5[best5], 5)}})
    # A6 projection
    pA6 = np.empty(len(y))
    for s in folds.list:
        gm = pk3[s]
        X = np.column_stack([Hd.design([logit(F[f"gl6_p_{gm:g}"])[mask]] + base), mar])
        tr = folds.seasons != s
        w = Hd.fit_logit(X[tr], y[tr])
        pA6[folds.idx[s]] = Hd._sig(X[folds.idx[s]] @ w)
    proj = F["_proj"]
    agree = []
    for g, k in zip(games, mask):
        if not k:
            continue
        st = starters.get(g["game_id"], {})
        pr = proj.get(g["game_id"], {})
        for fl_ in (1, 0):
            agree.append(pr.get(fl_) == st.get(fl_))
    rec("A6 A3 with projected starter (serving diag)", pA6,
        {"gamma_per_season_from_A3": picks_json(pk3),
         "projection_hit_rate_masked": round(float(np.mean(agree)), 4)})

    wf["W4"] = {"eval_mask_sha1": mask_hash, "y_sha1": y_hash,
                "identical_across_arms": len(set(arm_hash.values())) == 1,
                "n_masked": int(mask.sum()),
                "nan_personnel_features_on_masked_games": nan_any,
                "pass": len(set(arm_hash.values())) == 1 and not nan_any}
    wf["W5"] = {"statement": "the only same-game inputs are tonight's starting-goalie "
                             "identity (first-shot goalie; 99.99% = sole dressed goalie; "
                             "confirmed at warmups) and tonight's dressed-skater membership "
                             "(announced lineup). Tonight's GA/xGA/TOI/result enter only "
                             "the post-game update. nhl_games.csv win_goalie, max-shots "
                             "starters, all-season RAPM/shift ratings and odds are never read "
                             "by any feature path.", "pass": True}
    res["walk_forward"] = wf
    res["arms"] = arms

    # ============================ GL diagnostics ============================
    Gh, Ga = F["G_h"][mask], F["G_a"][mask]
    dG = Gh - Ga
    corr = {}
    for gm in GAMMAS:
        Rst = np.concatenate([F[f"R_h_{gm:g}"][mask], F[f"R_a_{gm:g}"][mask]])
        Gst = np.concatenate([Gh, Ga])
        corr[f"{gm:g}"] = round(float(np.corrcoef(Gst, Rst)[0, 1]), 4)
    snap = F["_snap"]
    face = {"date": snap["date"], "xbar": round(snap["xbar"], 4)}
    gp_, gc_ = snap[str(PRICE)], snap[str(CONDON)]
    face["Price"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in gp_.items()}
    face["Condon"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in gc_.items()}
    dGf = gp_["G_goals"] - gc_["G_goals"]
    face["G_price_minus_condon_goals"] = round(dGf, 4)
    face["elo_logit_gap_by_gamma"] = {f"{gm:g}": round(gm * dGf * np.log(10) / 400, 4)
                                      for gm in GAMMAS}
    face["ideas_screen_reference_logit"] = 0.09
    fbm = (F["fb_h"] | F["fb_a"])[mask]
    missm = (F["miss_h"] | F["miss_a"])[mask]
    rho_m = F["rho"][mask]
    res["gl_diagnostics"] = {
        "sd_G_diff_goals": round(float(dG.std()), 4),
        "sd_G_team_goals": round(float(np.concatenate([Gh, Ga]).std()), 4),
        "share_absG_diff_gt_0.05": round(float((np.abs(dG) > 0.05).mean()), 4),
        "corr_G_teamR_by_gamma": corr,
        "face_validity": face,
        "starter_fallback_games_masked": int(fbm.sum()),
        "starter_fallback_games_all_dev": int((F["fb_h"] | F["fb_a"]).sum()),
        "m_E": wf["W3"]["m_E_by_season"],
    }
    res["mar_diagnostics"] = {
        "missing_dressed_side_games_masked": int(missm.sum()),
        "missing_share_masked": round(float(missm.mean()), 5),
        "missing_under_0.5pct": bool(missm.mean() < 0.005),
        "rho_first_masked_game": round(float(rho_m[0]), 3),
        "rho_last_masked_game": round(float(rho_m[-1]), 3),
        "rho_by_season_start": {str(s): round(float(rho_m[seas == s][0]), 3)
                                for s in folds.list},
        "mean_D_side": round(float(np.concatenate([F["D_h"][mask], F["D_a"][mask]]).mean()), 3),
        "share_side_D_pos": round(float((np.concatenate([F["D_h"][mask],
                                                          F["D_a"][mask]]) > 0).mean()), 4),
        "sd_mar_diff": round(float(mar.std()), 3),
        "mean_absent_regulars_side": round(float(np.concatenate(
            [F["nabs_h"][mask], F["nabs_a"][mask]]).mean()), 3),
    }
    print(json.dumps(res["gl_diagnostics"], indent=None)[:1500], flush=True)
    print(json.dumps(res["mar_diagnostics"]), flush=True)

    # ======================= evaluation-only market diagnostic =======================
    from bt_nhl_anatomy import join_odds   # DEV-only join (date < 2018-07-01)
    Gm = [g for g, k in zip(games, mask) if k]
    pc, _po, _ = join_odds(Gm, y)
    ok = ~np.isnan(pc)
    lc = llv(y[ok], pc[ok])
    afm = af.loc[[g["game_id"] for g in Gm]]
    nm = ((afm.g_notmodal_home == 1) | (afm.g_notmodal_away == 1)).to_numpy()[ok]
    lk = (afm.abs_toi_home.notna() & afm.abs_toi_away.notna()).to_numpy()
    t4 = (np.nan_to_num(afm.abs_top4_home.to_numpy()) + np.nan_to_num(afm.abs_top4_away.to_numpy()))
    top4 = (lk & (t4 >= 1))[ok]
    mkt = {"n_joined": int(ok.sum()), "close_ll": round(float(lc.mean()), 6),
           "note": "EVALUATION ONLY: multiplicative de-vig of the SBR close; odds never "
                   "enter a feature or a fit"}
    for nm_, p_ in (("A0", p0), ("A1", pA1), ("A2", pA2), ("A3", pA3), ("A6", pA6)):
        lm = llv(y[ok], p_[ok])
        dd = lm - lc
        mkt[nm_] = {"model_minus_close_all": round(float(dd.mean()), 5),
                    "non1_goalie": {"n": int(nm.sum()),
                                    "model_minus_close": round(float(dd[nm].mean()), 5)},
                    "top4_absent": {"n": int(top4.sum()),
                                    "model_minus_close": round(float(dd[top4].mean()), 5)}}
    g0 = mkt["A0"]["model_minus_close_all"]
    mkt["A3_gap_closed_share"] = round((g0 - mkt["A3"]["model_minus_close_all"]) / g0, 4)
    res["market_eval_diagnostic"] = mkt
    print(json.dumps(mkt), flush=True)

    # ============================ VERDICT ============================
    sanity_pass = all(v["pass"] for v in san.values())
    wf_pass = all(v["pass"] for v in wf.values())
    gain = stA3["gain_exact"]
    lo = stA3["boot_ci_10k"][0]
    clears = bool(gain >= 0.00100 and lo > 0 and stA3["boot_ci_4k_seed7"][0] > 0
                  and sanity_pass and wf_pass)
    res["verdict"] = {"primary_arm": "A3", "nested_gain": round(gain, 6),
                      "boot_ci_10k": stA3["boot_ci_10k"],
                      "boot_ci_4k_seed7": stA3["boot_ci_4k_seed7"],
                      "sanity_pass": sanity_pass, "walk_forward_pass": wf_pass,
                      "clears_bar": clears,
                      "baseline_dev_ll": round(s1, 6), "candidate_dev_ll": stA3["dev_ll"]}
    res["seconds"] = round(time.time() - t0, 1)
    print(f"\nVERDICT A3 nested gain {gain:+.5f} CI10k {stA3['boot_ci_10k']} "
          f"sanity={sanity_pass} wf={wf_pass} clears_bar={clears}  ({res['seconds']}s)",
          flush=True)

    # per-game DEV feature cache
    fd = pd.DataFrame({"game_id": gids, "date": [g["date"] for g in games],
                       "season": ctx.seasons, "home": [g["home"] for g in games],
                       "away": [g["away"] for g in games], "in_eval_mask": mask.astype(int),
                       "gk_home": F["gk_h"], "gk_away": F["gk_a"],
                       "G_home": F["G_h"], "G_away": F["G_a"],
                       "starter_fallback_home": F["fb_h"].astype(int),
                       "starter_fallback_away": F["fb_a"].astype(int),
                       "D_home": F["D_h"], "D_away": F["D_a"], "mar_diff": F["mar_diff"],
                       "absent_regulars_home": F["nabs_h"], "absent_regulars_away": F["nabs_a"],
                       "dressed_missing_home": F["miss_h"].astype(int),
                       "dressed_missing_away": F["miss_a"].astype(int), "rho": F["rho"]})
    assert fd.season.max() <= DEV_END and fd.game_id.max() < MAX_GID
    fd.to_csv(OUT_FEATS + ".tmp", index=False)
    os.replace(OUT_FEATS + ".tmp", OUT_FEATS)
    for path in (OUT, OUT_MIRROR):
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=float)
        os.replace(path + ".tmp", path)
    print(f"wrote {OUT}, {OUT_MIRROR}, {OUT_FEATS}")


if __name__ == "__main__":
    main()
