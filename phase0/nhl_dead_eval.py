"""Dead-game weighting — the last untested NHL channel. DEV-only screen.

Hypothesis: late-season games where a team's playoff fate is effectively
decided (mathematically eliminated, or locked into the playoffs) carry weaker
signal — tanking for draft position, resting stars before the playoffs. Two
distinct uses are screened:

  A) PREDICTION: a dead team underperforms its rating TONIGHT
     -> home_dead - away_dead (and a soft continuous version) over base_z.
  B) EVIDENCE : rating updates from dead games are noisier
     -> Elo walk down-weighting K (x0.5 / x0.25) when either team is dead,
        DEV LL of the raw Elo probs vs the standard walk.

Standings are reconstructed as-of from data/nhl_games.csv ALONE (2 pts win,
1 pt OT/SO loss, 0 reg loss). All deadness reads use standings ENTERING the
game's calendar date (strictly earlier games only — leak-safe).

Deadness definitions (per team, pre-game; GR = season_len - games_played):
  ELIMINATED: >= 8 conference teams already hold MORE points than my max
              attainable (pts + 2*GR). Those teams cannot lose points, so top-8
              is unreachable  (equivalent to gap-to-8th > 2*GR).
  LOCKED    : <= 7 conference teams could still reach my CURRENT points
              (their pts + 2*their_GR >= my pts). Even losing out I stay top-8.
  DEAD      : eliminated OR locked.
  SOFT      : margin to the relevant cutline (8th place if outside, 9th place
              if inside) divided by 2*GR, clipped to [0,1]. 1.0 = fully decided,
              early-season values ~0 by construction.

Honest approximations (all conservative or second-order):
  * Playoff qualification = top-8 by conference points. True pre-2013-14; from
    2013-14 the real rule is top-3 per division + 2 wildcards, which almost
    always coincides with the conference top-8 (rare crossover exceptions).
  * Tiebreakers (ROW etc.) ignored: ELIMINATED requires strictly-more points,
    LOCKED lets challengers tie — both flags fire slightly LATE, never early.
  * Remaining head-to-head games double-counted in max-attainable (two teams
    can't both win the same game) — again makes both flags conservative.
  * Season length: 82, except 48 (2012-13 lockout) and 56 (2020-21). 2019-20
    uses the pre-game expectation of 82 (the COVID stoppage was not knowable).
  * Conference map hardcoded both sides of the 2013-14 realignment
    (pre: WPG/ATL East, CBJ+DET West; post: CBJ+DET East, WPG West).

--build : reconstruct standings, write per-game deadness -> data/nhl_dead.csv
--screen: DEV-only screens (2011-12..2017-18 eval era, warm from 2010-11).
          A) frozen shipped-blend base_z, temporal fit-half/score-half plus
             5-fold OOF robustness (mirrors nhl_st_eval discipline; no
             hyperparameters to select — definitions fixed a priori above).
          B) engine screen: run_elo variant with K*mult on dead games,
             DEV LL vs standard K, bootstrap CI on per-game deltas.
          NO TEST-era (>= 2018-19) evaluation anywhere in this file.
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
from nhl_glicko2_eval import (DEV_END, DEV_WARM_BEFORE, llv,  # noqa: E402
                              load_games, run_elo)
from nhl_features_eval import build_features  # noqa: E402

DEAD_CSV = "data/nhl_dead.csv"
REPORT = "data/nhl_dead_report.json"

SEASON_LEN = {20122013: 48, 20202021: 56}          # default 82 (2019-20: expected 82)

EAST_PRE = {"ATL", "BOS", "BUF", "CAR", "FLA", "MTL", "NJD", "NYI", "NYR",
            "OTT", "PHI", "PIT", "TBL", "TOR", "WSH", "WPG"}   # <= 2012-13
EAST_POST = {"BOS", "BUF", "CAR", "CBJ", "DET", "FLA", "MTL", "NJD", "NYI",
             "NYR", "OTT", "PHI", "PIT", "TBL", "TOR", "WSH"}  # >= 2013-14


def conf_of(team, season):
    east = EAST_PRE if season <= 20122013 else EAST_POST
    return "E" if team in east else "W"


# ---------------------------------------------------------------- build stage
def team_state(team, season, pts, gp, conf_teams, slen):
    """(elim, lock, soft) from standings ENTERING today. Pure function of
    already-played games."""
    gr = max(slen - gp[team], 0)
    max_pts = pts[team] + 2 * gr
    others = [t for t in conf_teams if t != team]
    ahead_forever = sum(1 for t in others if pts[t] > max_pts)
    elim = ahead_forever >= 8
    could_catch = sum(1 for t in others
                      if pts[t] + 2 * max(slen - gp[t], 0) >= pts[team])
    lock = could_catch <= 7
    # soft decidedness: margin to the live cutline over max swing
    ranked = sorted((pts[t] for t in conf_teams), reverse=True)
    p8 = ranked[7] if len(ranked) >= 8 else 0
    p9 = ranked[8] if len(ranked) >= 9 else 0
    margin = (pts[team] - p9) if pts[team] >= p8 else (p8 - pts[team])
    soft = min(1.0, max(margin, 0) / max(2 * gr, 1))
    return int(elim), int(lock), round(soft, 4)


def build():
    games = load_games()
    # load_games keeps only the SO flag; the loser point needs OT too
    loser_pt = {int(r["game_id"]): r["last_period"] != "REG"
                for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8"))}
    by_season = defaultdict(list)
    for g in games:
        by_season[g["season"]].append(g)

    rows_out = []
    for season in sorted(by_season):
        sg = sorted(by_season[season], key=lambda g: (g["date"], g["home"]))
        teams = sorted({g["home"] for g in sg} | {g["away"] for g in sg})
        conf_members = {c: [t for t in teams if conf_of(t, season) == c]
                        for c in ("E", "W")}
        slen = SEASON_LEN.get(season, 82)
        pts = defaultdict(int)
        gp = defaultdict(int)
        # process date-by-date: flags for ALL of a date's games use standings
        # entering that date, then the date's results are applied
        i = 0
        while i < len(sg):
            j = i
            while j < len(sg) and sg[j]["date"] == sg[i]["date"]:
                j += 1
            for g in sg[i:j]:
                row = {"game_id": g["game_id"], "date": g["date"],
                       "season": season, "home": g["home"], "away": g["away"]}
                for side in ("home", "away"):
                    t = g[side]
                    e_, l_, s_ = team_state(t, season, pts, gp,
                                            conf_members[conf_of(t, season)], slen)
                    row[f"{side}_elim"], row[f"{side}_lock"] = e_, l_
                    row[f"{side}_soft"] = s_
                rows_out.append(row)
            for g in sg[i:j]:                      # apply results after flags
                w = g["home"] if g["y"] == 1 else g["away"]
                lo = g["away"] if g["y"] == 1 else g["home"]
                pts[w] += 2
                if loser_pt[g["game_id"]]:
                    pts[lo] += 1
                gp[g["home"]] += 1
                gp[g["away"]] += 1
            i = j

    tmp = DEAD_CSV + ".tmp"
    fields = ["game_id", "date", "season", "home", "away", "home_elim",
              "home_lock", "home_soft", "away_elim", "away_lock", "away_soft"]
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    os.replace(tmp, DEAD_CSV)
    n_dead = sum(1 for r in rows_out
                 if r["home_elim"] or r["home_lock"] or r["away_elim"] or r["away_lock"])
    print(f"[build] {len(rows_out):,} games, {n_dead:,} with a dead team "
          f"({n_dead/len(rows_out):.1%} all-era)  -> {DEAD_CSV}")


# -------------------------------------------------------------- screen stage
def load_dead(games):
    idx = {}
    for r in csv.DictReader(open(DEAD_CSV, encoding="utf-8")):
        idx[int(r["game_id"])] = r
    n = len(games)
    he = np.zeros(n); hl = np.zeros(n); hs = np.zeros(n)
    ae = np.zeros(n); al = np.zeros(n); as_ = np.zeros(n)
    for i, g in enumerate(games):
        r = idx[g["game_id"]]
        he[i], hl[i], hs[i] = float(r["home_elim"]), float(r["home_lock"]), float(r["home_soft"])
        ae[i], al[i], as_[i] = float(r["away_elim"]), float(r["away_lock"]), float(r["away_soft"])
    return he, hl, hs, ae, al, as_


def run_elo_dead(games, k, ha, regress, dead_mask, mult):
    """run_elo clone with K*mult on flagged games (evidence down-weighting)."""
    R = {}
    out = []
    prev_season = None
    for i, g in enumerate(games):
        if prev_season is not None and g["season"] != prev_season:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - regress)
        prev_season = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        p = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        out.append((g["season"], p, g["y"]))
        kk = k * mult if dead_mask[i] else k
        R[g["home"]] += kk * (g["y"] - p)
        R[g["away"]] += kk * ((1 - g["y"]) - (1 - p))
    return out


def fit_score(base_z, feats, y, fit_m, score_m):
    Xb = base_z.reshape(-1, 1)
    mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[fit_m], y[fit_m])
    b_ll = llv(y[score_m], mb.predict_proba(Xb[score_m])[:, 1])
    X = np.column_stack([base_z] + feats)
    ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[fit_m], y[fit_m])
    s_ll = llv(y[score_m], ms.predict_proba(X[score_m])[:, 1])
    return b_ll, s_ll, [float(c) for c in ms.coef_[0][1:]]


def boot_ci(d, seed=7):
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def screen():
    model = json.load(open("data/nhl_model.json"))
    games = load_games()
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    he, hl, hs, ae, al, as_ = load_dead(games)
    h_dead = np.maximum(he, hl)
    a_dead = np.maximum(ae, al)
    either_dead = np.maximum(h_dead, a_dead)
    either_elim = np.maximum(he, ae)

    dev = (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    res = {}

    # ---- prevalence + sanity (DEV era only) ----
    months = np.array([g["date"][5:7] for g in games])
    print("deadness prevalence, DEV era 2011-12..2017-18 (share of games with a "
          "dead team):", flush=True)
    print(f"  overall: {either_dead[dev].mean():.1%}  "
          f"(elim-only {np.maximum(he, ae)[dev].mean():.1%}, "
          f"lock-only {np.maximum(hl, al)[dev].mean():.1%})", flush=True)
    prev_m = {}
    for m in ("10", "11", "12", "01", "02", "03", "04"):
        mm = dev & (months == m)
        if mm.sum():
            prev_m[m] = round(float(either_dead[mm].mean()), 4)
            print(f"  month {m}: {either_dead[mm].mean():6.1%}  (n={int(mm.sum())})",
                  flush=True)
    res["prevalence_dev"] = {"overall": round(float(either_dead[dev].mean()), 4),
                            "by_month": prev_m}

    print("\nsanity — first ELIMINATED date (famous tank/bad seasons, DEV era):",
          flush=True)
    sanity = {}
    for team, season in (("BUF", 20142015), ("ARI", 20142015), ("ARI", 20162017),
                         ("EDM", 20142015), ("EDM", 20152016), ("TOR", 20142015),
                         ("BUF", 20132014)):
        first = next((g["date"] for i, g in enumerate(games)
                      if g["season"] == season
                      and ((g["home"] == team and he[i]) or (g["away"] == team and ae[i]))),
                     None)
        sanity[f"{team}_{season}"] = first
        print(f"  {team} {season}: {first or 'never flagged'}", flush=True)
    print("sanity — first LOCKED date (juggernauts):", flush=True)
    for team, season in (("WSH", 20152016), ("WSH", 20162017), ("NSH", 20172018)):
        first = next((g["date"] for i, g in enumerate(games)
                      if g["season"] == season
                      and ((g["home"] == team and hl[i]) or (g["away"] == team and al[i]))),
                     None)
        sanity[f"{team}_{season}_lock"] = first
        print(f"  {team} {season}: {first or 'never flagged'}", flush=True)
    res["sanity_first_flag"] = sanity

    # ---- base_z: frozen shipped blend (nhl_goalie2/st pattern) ----
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    b = model["blend"]; c = b["coefs"]
    base_z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
              + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
              + c["xg"] * np.nan_to_num(F["xg_diff"]))

    dev_idx = np.where(dev)[0]                    # date-sorted
    half = len(dev_idx) // 2
    fit_m = np.zeros(len(games), bool); fit_m[dev_idx[:half]] = True
    sc_m = np.zeros(len(games), bool); sc_m[dev_idx[half:]] = True

    dead_diff = h_dead - a_dead
    elim_diff = he - ae
    lock_diff = hl - al
    soft_diff = hs - as_

    def report(name, feats, key):
        b_ll, s_ll, coefs = fit_score(base_z, feats, y, fit_m, sc_m)
        d = b_ll - s_ll
        lo, hi = boot_ci(d)
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        bm, sm_ = float(b_ll.mean()), float(s_ll.mean())
        print(f"  {name:<26} n={int(sc_m.sum()):<5} base {bm:.5f} -> {sm_:.5f}  "
              f"delta {bm-sm_:+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}  "
              f"coefs {[round(x, 3) for x in coefs]}", flush=True)
        res[key] = {"n": int(sc_m.sum()), "base": round(bm, 5), "with": round(sm_, 5),
                    "delta": round(bm - sm_, 5), "ci": [round(lo, 5), round(hi, 5)],
                    "sig": sig, "coefs": [round(x, 4) for x in coefs]}

    print("\nA) PREDICTION screen — temporal fit-half/score-half "
          "(DEV 2011-12..2017-18):", flush=True)
    report("dead_diff (1f)", [dead_diff], "pred_dead_holdout")
    report("elim_diff + lock_diff (2f)", [elim_diff, lock_diff], "pred_split_holdout")
    report("soft stakes diff (1f)", [soft_diff], "pred_soft_holdout")

    print("\nA-robustness — 5-fold OOF over full DEV:", flush=True)
    idx = dev_idx
    for name, feats, key in (("dead_diff (1f)", [dead_diff], "pred_dead_5fold"),
                             ("elim+lock (2f)", [elim_diff, lock_diff], "pred_split_5fold"),
                             ("soft diff (1f)", [soft_diff], "pred_soft_5fold")):
        d_all = np.zeros(len(idx))
        kf = KFold(n_splits=5, shuffle=True, random_state=7)
        for tr, te in kf.split(idx):
            fm = np.zeros(len(games), bool); fm[idx[tr]] = True
            sm2 = np.zeros(len(games), bool); sm2[idx[te]] = True
            b_ll, s_ll, _ = fit_score(base_z, feats, y, fm, sm2)
            d_all[te] = b_ll - s_ll
        lo, hi = boot_ci(d_all)
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<26} n={len(idx):<5} OOF delta {d_all.mean():+.5f} "
              f"CI [{lo:+.5f},{hi:+.5f}] {sig}", flush=True)
        res[key] = {"n": len(idx), "delta": round(float(d_all.mean()), 5),
                    "ci": [round(lo, 5), round(hi, 5)], "sig": sig}

    # ---- B) EVIDENCE screen: dead-game K down-weighting (raw Elo, DEV LL) ----
    print("\nB) EVIDENCE screen — Elo K down-weighting on dead games "
          "(raw Elo probs, DEV LL):", flush=True)
    ll_std = llv(y[dev], p[dev])
    print(f"  standard K={model['elo_cfg']['k']:<18} DEV LL {float(ll_std.mean()):.5f} "
          f"(n={int(dev.sum())})", flush=True)
    res["elo_std_dev_ll"] = round(float(ll_std.mean()), 5)
    for cond_name, cond in (("either_dead", either_dead), ("either_elim", either_elim)):
        for mult in (0.5, 0.25):
            out = run_elo_dead(games, dead_mask=cond > 0, mult=mult, **model["elo_cfg"])
            pw = np.clip(np.array([o[1] for o in out]), 1e-9, 1 - 1e-9)
            ll_w = llv(y[dev], pw[dev])
            d = ll_std - ll_w
            lo, hi = boot_ci(d)
            sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
            print(f"  {cond_name} K*{mult:<4}        DEV LL {float(ll_w.mean()):.5f}  "
                  f"delta {float(d.mean()):+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}",
                  flush=True)
            res[f"elo_{cond_name}_x{mult}"] = {
                "dev_ll": round(float(ll_w.mean()), 5),
                "delta_vs_std": round(float(d.mean()), 5),
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
