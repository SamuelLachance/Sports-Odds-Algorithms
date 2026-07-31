"""Special-teams PP/PK xG split — DEV-only screen (locked-split protocol).

The shipped xG team rating (nhl_features_eval.build_features) updates one
rating per team on ALL-situations xG margin. Power-play offense and
penalty-kill defense are distinct skills that the single rating blurs; a team
with an elite PP but average 5v5 game gets the same credit as the reverse.

Data: data/nhl_shots.csv has per-shot skater counts (home_sk/away_sk, goalies
NOT counted per MoneyPuck convention). A power-play shot = shooter's side has
MORE skaters AND at most 5 (<=5 excludes pulled-goalie 6v5 / extra-attacker
delayed-penalty situations, which are endgame desperation, not special-teams
skill). Shorthanded shots by the disadvantaged team are deliberately excluded.

--build : aggregate per-game per-team PP xGF (and shot counts) from
          data/nhl_shots.csv -> data/nhl_st_xg.csv (atomic write).
          A team's PK xGA in a game is by construction the opponent's PP xGF.

--screen: two Elo-style walks updating on EXPECTED goals only (never realized),
          mirroring the shipped xg_diff design exactly:

            pred_pp(team@side vs opp) = mu_side + pp[team] + pk[opp]
            err = obs_pp_xgf - pred ;  pp[team] += K*err ; pk[opp] += K*err

          pp[t]  = PP xG created per game above league norm (higher = better PP)
          pk[t]  = PP xG CONCEDED per game above norm (higher = leakier PK)
          mu_side = slow online EWMA of the side-specific league mean (home
          teams draw more penalties; two means absorb that instead of a tuned
          HA constant). Season boundary regresses pp/pk by ST_REGRESS = the
          shipped xg regress (0.30). All reads pre-game, updates after.

          Features (pre-game differentials, home perspective):
            st_pp_diff = pp[home] - pp[away]        (PP-offense edge)
            st_pk_diff = pk[away] - pk[home]        (opp PK leakier than ours)
            st_net     = st_pp_diff + st_pk_diff    (expected ST xG swing)

DEV screen protocol (NO TEST look — TEST seasons >= 2018-19 never touched):
  base_z = frozen shipped-blend coefs applied to DEV features (as
  nhl_goalie2_eval builds it). DEV eval era 2011-12..2017-18 (warm from
  2010-11) is split temporally: first half FIT, second half SCORE. K is chosen
  on the fit half only (internal temporal sub-split), then base logistic
  (base_z) vs augmented (base_z + ST features) are fit on the fit half and
  scored once on the score half; bootstrap CI on per-game LL deltas. A random
  5-fold out-of-fold screen over full DEV is reported as robustness.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import DEV_END, DEV_WARM_BEFORE, llv, load_games, run_elo  # noqa: E402
from nhl_features_eval import TEAM_FIX, build_features  # noqa: E402

SHOTS = "data/nhl_shots.csv"
ST_CSV = "data/nhl_st_xg.csv"
REPORT = "data/nhl_st_report.json"

ST_REGRESS = 0.30       # season regression, mirror shipped xg_cfg
MU_DECAY = 0.995        # league-mean EWMA decay per team-game
K_GRID = [0.02, 0.04, 0.06, 0.10, 0.15]


def norm(t):
    return TEAM_FIX.get(t, t)


# ---------------------------------------------------------------- build stage
def build():
    """Per-game per-team PP xGF from per-shot rows. PK xGA = opponent's PP xGF."""
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # gid -> team -> [pp_xgf, n]
    sides = {}                                                 # gid -> {team: is_home}
    n_shots = n_pp = 0
    with open(SHOTS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n_shots += 1
            period = int(r["period"])
            if period > 4:                       # shootout safety
                continue
            hs, as_ = int(r["home_sk"]), int(r["away_sk"])
            is_home = r["is_home"] == "1"
            sk_for, sk_ag = (hs, as_) if is_home else (as_, hs)
            if not (sk_for > sk_ag and sk_for <= 5):
                continue                          # not a classic PP shot
            n_pp += 1
            gid = int(r["nhl_game_id"])
            team = norm(r["shooter_team"])
            cell = agg[gid][team]
            cell[0] += float(r["xg"]); cell[1] += 1
            sides.setdefault(gid, {})[team] = int(is_home)
    rows = []
    for gid in sorted(agg):
        for team, (xgf, n) in sorted(agg[gid].items()):
            rows.append({"nhl_game_id": gid, "team": team,
                         "is_home": sides[gid].get(team, ""),
                         "pp_xgf": round(xgf, 4), "pp_shots": n})
    tmp = ST_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["nhl_game_id", "team", "is_home",
                                           "pp_xgf", "pp_shots"])
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, ST_CSV)
    seasons = sorted({gid // 1000000 for gid in agg})
    print(f"[build] {n_shots:,} shots -> {n_pp:,} PP shots (adv & <=5 skaters), "
          f"{len(rows):,} team-game rows, seasons {seasons[0]}..{seasons[-1]}")
    print(f"[build] wrote {ST_CSV}")


# --------------------------------------------------------------- rating walk
def load_st():
    """gid -> {team: pp_xgf}. Games with shots data but no PP shots for a team
    legitimately mean 0 PP xG that night, so default missing team to 0.0 IF the
    game is covered at all."""
    out = {}
    for r in csv.DictReader(open(ST_CSV, encoding="utf-8")):
        out.setdefault(int(r["nhl_game_id"]), {})[r["team"]] = float(r["pp_xgf"])
    return out


def st_walk(games, k):
    """Leak-safe as-of walk. Returns (pp_diff, pk_diff, covered_mask, ratings)."""
    st = load_st()
    covered_gids = set(st)                       # any game in the shots file era
    # a game is "covered" if its season appears in the shots data at all
    seasons_cov = {g // 1000000 for g in covered_gids}
    pp = defaultdict(float)
    pk = defaultdict(float)
    mu = {1: 0.45, 0: 0.45}                      # side -> EWMA league mean, mild seed
    n = len(games)
    pp_diff = np.full(n, np.nan)
    pk_diff = np.full(n, np.nan)
    prev_season = None
    for i, g in enumerate(games):
        if prev_season is not None and g["season"] != prev_season:
            for t in pp: pp[t] *= (1 - ST_REGRESS)
            for t in pk: pk[t] *= (1 - ST_REGRESS)
        prev_season = g["season"]
        if (g["season"] // 10000) in seasons_cov:
            pp_diff[i] = pp[g["home"]] - pp[g["away"]]
            pk_diff[i] = pk[g["away"]] - pk[g["home"]]
        gv = st.get(g["game_id"])
        if gv is None:
            if g["game_id"] in covered_gids or (g["season"] // 10000) not in seasons_cov:
                continue
            gv = {}                              # covered era, zero PP shots at all
        for team, opp, side in ((g["home"], g["away"], 1), (g["away"], g["home"], 0)):
            obs = gv.get(team, 0.0)
            pred = mu[side] + pp[team] + pk[opp]
            err = obs - pred
            pp[team] += k * err
            pk[opp] += k * err
            mu[side] = MU_DECAY * mu[side] + (1 - MU_DECAY) * obs
    return pp_diff, pk_diff, dict(pp), dict(pk)


# -------------------------------------------------------------- screen stage
def fit_score(base_z, feats, y, fit_m, score_m):
    """Refit base and augmented logistics on fit_m, score on score_m."""
    Xb = base_z.reshape(-1, 1)
    mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[fit_m], y[fit_m])
    p0 = mb.predict_proba(Xb[score_m])[:, 1]
    b_ll = llv(y[score_m], p0)
    if feats is None:
        return b_ll, None, None
    X = np.column_stack([base_z] + feats)
    ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[fit_m], y[fit_m])
    p1 = ms.predict_proba(X[score_m])[:, 1]
    return b_ll, llv(y[score_m], p1), [float(c) for c in ms.coef_[0][1:]]


def boot_ci(d, seed=7):
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def screen():
    model = json.load(open("data/nhl_model.json"))
    games = load_games()
    # hard guard: nothing after DEV is ever scored in this script
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    b = model["blend"]; c = b["coefs"]
    base_z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
              + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
              + c["xg"] * np.nan_to_num(F["xg_diff"]))

    dev = (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    dev_idx = np.where(dev)[0]                   # games are date-sorted
    half = len(dev_idx) // 2
    fit_m = np.zeros(len(games), bool); fit_m[dev_idx[:half]] = True
    sc_m = np.zeros(len(games), bool); sc_m[dev_idx[half:]] = True

    # ---- K selection on the FIT half only (internal temporal sub-split)
    sub = dev_idx[:half]
    sub_fit = np.zeros(len(games), bool); sub_fit[sub[:len(sub) // 2]] = True
    sub_sc = np.zeros(len(games), bool); sub_sc[sub[len(sub) // 2:]] = True
    best_k, best_ll = None, np.inf
    print("K selection (fit-half internal split, st_net feature):", flush=True)
    for k in K_GRID:
        ppd, pkd, _, _ = st_walk(games, k)
        net = np.nan_to_num(ppd + pkd)
        _, s_ll, _ = fit_score(base_z, [net], y, sub_fit, sub_sc)
        m = float(s_ll.mean())
        print(f"  K={k:<5} inner-score LL {m:.5f}", flush=True)
        if m < best_ll:
            best_ll, best_k = m, k
    print(f"  chosen K={best_k}", flush=True)

    ppd, pkd, pp_r, pk_r = st_walk(games, best_k)
    net = ppd + pkd
    have = ~np.isnan(net)
    print(f"\nfeature coverage on DEV eval era: {have[dev].mean():.1%}  "
          f"std(st_net)={np.nanstd(net[dev]):.4f}  "
          f"std(pp_diff)={np.nanstd(ppd[dev]):.4f}  std(pk_diff)={np.nanstd(pkd[dev]):.4f}",
          flush=True)

    # ---- sanity: top-5 PP / PK ratings at end of the walk (post-2017-18 state
    # regressed through TEST-era games too, so print DEV-frozen ones instead)
    ppd_dev, pkd_dev, pp_dev, pk_dev = st_walk(
        [g for g in games if g["season"] <= DEV_END], best_k)
    top_pp = sorted(pp_dev.items(), key=lambda kv: -kv[1])[:5]
    top_pk = sorted(pk_dev.items(), key=lambda kv: kv[1])[:5]
    print("top-5 PP (end of 2017-18):", [(t, round(v, 3)) for t, v in top_pp], flush=True)
    print("top-5 PK (end of 2017-18):", [(t, round(v, 3)) for t, v in top_pk], flush=True)

    res = {"k": best_k, "coverage_dev": round(float(have[dev].mean()), 4),
           "top_pp_2018": [(t, round(v, 4)) for t, v in top_pp],
           "top_pk_2018": [(t, round(v, 4)) for t, v in top_pk]}

    def report(name, feats, key):
        fm = fit_m & have; sm = sc_m & have
        b_ll, s_ll, coefs = fit_score(base_z, feats, y, fm, sm)
        d = b_ll - s_ll
        lo, hi = boot_ci(d)
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        bm, sm_ = float(b_ll.mean()), float(s_ll.mean())
        print(f"  {name:<24} n={int(sm.sum()):<5} base {bm:.5f} -> {sm_:.5f}  "
              f"delta {bm-sm_:+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}  "
              f"coefs {[round(x,3) for x in coefs]}", flush=True)
        res[key] = {"n": int(sm.sum()), "base": round(bm, 5), "with": round(sm_, 5),
                    "delta": round(bm - sm_, 5), "ci": [round(lo, 5), round(hi, 5)],
                    "sig": sig, "coefs": [round(x, 4) for x in coefs]}

    print("\nDEV screen — temporal fit-half / score-half (2011-12..2017-18 only):",
          flush=True)
    report("st_net (1f)", [np.nan_to_num(net)], "net_holdout")
    report("st_pp + st_pk (2f)", [np.nan_to_num(ppd), np.nan_to_num(pkd)], "split_holdout")

    # ---- robustness: random 5-fold out-of-fold over full DEV eval era
    print("\nrobustness — 5-fold OOF over full DEV:", flush=True)
    m_all = dev & have
    idx = np.where(m_all)[0]
    for name, feats, key in (("st_net (1f)", [np.nan_to_num(net)], "net_5fold"),
                             ("st_pp + st_pk (2f)",
                              [np.nan_to_num(ppd), np.nan_to_num(pkd)], "split_5fold")):
        d_all = np.zeros(len(idx))
        kf = KFold(n_splits=5, shuffle=True, random_state=7)
        for tr, te in kf.split(idx):
            fm = np.zeros(len(games), bool); fm[idx[tr]] = True
            sm2 = np.zeros(len(games), bool); sm2[idx[te]] = True
            b_ll, s_ll, _ = fit_score(base_z, feats, y, fm, sm2)
            d_all[te] = b_ll - s_ll
        lo, hi = boot_ci(d_all)
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<24} n={len(idx):<5} OOF delta {d_all.mean():+.5f} "
              f"CI [{lo:+.5f},{hi:+.5f}] {sig}", flush=True)
        res[key] = {"n": len(idx), "delta": round(float(d_all.mean()), 5),
                    "ci": [round(lo, 5), round(hi, 5)], "sig": sig}

    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, REPORT)
    print(f"\nwrote {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--screen", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.screen:
        screen()
    if not (a.build or a.screen):
        ap.error("pass --build and/or --screen")


if __name__ == "__main__":
    main()
