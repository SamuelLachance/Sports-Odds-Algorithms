"""Breakthrough program - NHL gap anatomy on DEV ONLY (evaluation use of odds).

Where does the shipped 4-feature blend lose to the de-vigged closing line on
DEV seasons 2011-12..2017-18? Odds are used ONLY to evaluate (constitution rule
1) - nothing here fits a model to the market. Every metric is computed on the
explicit DEV mask; TEST seasons are never scored.

Shipped-blend predictions are leave-one-season-out over DEV (the
phase0/nhl_xg_screen.py loso harness), so the in-sample blend is not flattered.

Writes data/bt_nhl_ideas_anat.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict, deque

import numpy as np

sys.path.insert(0, "phase0")
from nhl_features_eval import build_features  # noqa: E402
from nhl_glicko2_eval import llv, load_games, run_elo  # noqa: E402
from nhl_xg_screen import loso  # noqa: E402

DEV_SEASONS = [20112012, 20122013, 20132014, 20142015, 20152016, 20162017, 20172018]
GFIX = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}


def goalie_rows():
    rows = defaultdict(list)
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        rows[int(r["nhl_game_id"])].append(
            (GFIX.get(r["team"], r["team"]), int(r["goalie_id"]), float(r["xga"]),
             float(r["gsax"])))
    return rows


def spine_to_mp(team, season):
    # MoneyPuck uses ARI for Phoenix years and WPG/ATL as-is
    if team == "PHX":
        return "ARI"
    return team


def main():
    games = [g for g in load_games() if g["season"] <= 20172018]   # DEV era only
    assert max(g["season"] for g in games) <= 20172018
    model = json.load(open("data/nhl_model.json"))
    e_out = run_elo(games, **model["elo_cfg"])
    pe = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(pe / (1 - pe))
    F = build_features(games)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    n = len(games)

    # ---- goalie starter / usual starter (pre-game state) ----
    gr = goalie_rows()
    last20 = defaultdict(lambda: deque(maxlen=20))
    tgp = defaultdict(int)
    prev = None
    backup = np.zeros((n, 2))          # home, away backup-start flag
    gstarts = np.zeros((n, 2))         # starter's starts in last 20
    known = np.zeros(n, bool)
    team_gp = np.zeros((n, 2))
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            tgp.clear()
        prev = g["season"]
        rows = gr.get(g["game_id"], [])
        st = {}
        for side in ("home", "away"):
            mt = spine_to_mp(g[side], g["season"])
            lst = [(xga, gk) for t, gk, xga, _ in rows if t == mt]
            st[side] = max(lst)[1] if lst else None
        ok = st["home"] is not None and st["away"] is not None
        known[i] = ok
        for j, side in enumerate(("home", "away")):
            t = g[side]
            team_gp[i, j] = tgp[t]
            dq = last20[t]
            if ok and len(dq) >= 5:
                c = Counter(dq)
                usual = c.most_common(1)[0][0]
                backup[i, j] = float(st[side] != usual)
                gstarts[i, j] = c.get(st[side], 0)
        for side in ("home", "away"):
            if st[side] is not None:
                last20[g[side]].append(st[side])
            tgp[g[side]] += 1

    # ---- scratch/absence (row-10 construction, as-of) ----
    from nhl_scratch_eval import build_features_scratch
    absent, nmiss = build_features_scratch(games)

    # ---- odds join (evaluation only) ----
    odds = {}
    for r in csv.DictReader(open("data/odds_nhl.csv", encoding="utf-8")):
        try:
            ho, ao = float(r["home_close"]), float(r["away_close"])
        except ValueError:
            continue
        if ho <= 1 or ao <= 1:
            continue
        ih, ia = 1 / ho, 1 / ao
        odds[(r["date"], r["home"], r["away"])] = ih / (ih + ia)
    pc = np.array([odds.get((g["date"], g["home"], g["away"]), np.nan) for g in games])

    # ---- DEV mask, LOSO shipped blend ----
    xg = F["xg_diff"]
    dev = np.isin(seas, DEV_SEASONS) & ~np.isnan(xg)
    X = np.column_stack([elogit, F["rest_diff"], F["b2b_home"], F["b2b_away"], xg])
    pm_all = np.full(n, np.nan)
    pm_all[dev] = loso(X[dev], y[dev], seas[dev], DEV_SEASONS)
    m = dev & ~np.isnan(pc)
    assert set(np.unique(seas[m])) <= set(DEV_SEASONS)
    pm, pcl, ym = pm_all[m], pc[m], y[m]
    lm, lc = llv(ym, pm), llv(ym, pcl)
    d = lm - lc
    print(f"DEV games with close: {m.sum()} of {dev.sum()} dev; "
          f"model {lm.mean():.5f} close {lc.mean():.5f} gap {d.mean():+.5f}")
    out = {"n": int(m.sum()), "model_ll": float(lm.mean()), "close_ll": float(lc.mean()),
           "gap": float(d.mean()), "slices": {}}

    def sl(name, mask):
        mask = mask.astype(bool)
        if mask.sum() < 30:
            return
        share = d[mask].sum() / d.sum()
        rec = {"n": int(mask.sum()), "frac": round(float(mask.mean()), 4),
               "model": round(float(lm[mask].mean()), 5),
               "close": round(float(lc[mask].mean()), 5),
               "gap": round(float(d[mask].mean()), 5),
               "share_of_total_gap": round(float(share), 3)}
        out["slices"][name] = rec
        print(f"  {name:<38} n={rec['n']:>5} ({rec['frac']:.1%})  model {rec['model']:.5f} "
              f"close {rec['close']:.5f}  gap {rec['gap']:+.5f}  share {rec['share_of_total_gap']:+.2f}")

    tg = team_gp[m]
    mingp = tg.min(axis=1)
    print("\nby season-game number (min of two teams):")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 90)):
        sl(f"gp {lo}-{hi}", (mingp >= lo) & (mingp < hi))
    print("\nby season:")
    for s in DEV_SEASONS:
        sl(f"season {s}", seas[m] == s)
    bk = backup[m]
    kn = known[m]
    print("\nby actual starting-goalie status (starter != team's modal last-20 starter):")
    sl("no backup", kn & (bk.sum(1) == 0))
    sl("home backup only", kn & (bk[:, 0] == 1) & (bk[:, 1] == 0))
    sl("away backup only", kn & (bk[:, 0] == 0) & (bk[:, 1] == 1))
    sl("both backups", kn & (bk.sum(1) == 2))
    print("\nby b2b:")
    b2h, b2a = F["b2b_home"][m], F["b2b_away"][m]
    sl("no b2b", (b2h == 0) & (b2a == 0))
    sl("home b2b", b2h == 1)
    sl("away b2b", b2a == 1)
    sl("away b2b & away backup", (b2a == 1) & (bk[:, 1] == 1))
    sl("away b2b & away usual", (b2a == 1) & (bk[:, 1] == 0))
    sl("home b2b & home backup", (b2h == 1) & (bk[:, 0] == 1))
    print("\nby absent regulars (row-10 feature, TOI minutes):")
    ab = absent[m]
    sl("absent nan (masked early season)", np.isnan(ab))
    sl("|absent|<5", np.abs(ab) < 5)
    sl("5<=|absent|<20", (np.abs(ab) >= 5) & (np.abs(ab) < 20))
    sl("|absent|>=20", np.abs(ab) >= 20)
    print("\nby month:")
    mon = np.array([int(g["date"][5:7]) for g in games])[m]
    for mo in (10, 11, 12, 1, 2, 3, 4):
        sl(f"month {mo:02d}", mon == mo)
    print("\nby model-close disagreement |logit diff|:")
    lgm = np.log(pm / (1 - pm)); lgc = np.log(pcl / (1 - pcl))
    r = lgc - lgm
    for lo, hi in ((0, .1), (.1, .2), (.2, .35), (.35, 5)):
        sl(f"|dlogit| {lo}-{hi}", (np.abs(r) >= lo) & (np.abs(r) < hi))

    # ---- what does the close's deviation from us track? (diagnostic OLS) ----
    cols = {
        "backup_home": bk[:, 0] * kn, "backup_away": bk[:, 1] * kn,
        "absent_diff(awayminushome,min)": np.nan_to_num(ab),
        "b2b_home": b2h, "b2b_away": b2a,
        "early_gp<10": (mingp < 10).astype(float),
        "model_logit": lgm,
    }
    Z = np.column_stack([np.ones(m.sum())] + list(cols.values()))
    beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
    res = r - Z @ beta
    se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * res.var())
    print("\nOLS of (close logit - model logit) on game conditions (DEV):")
    diag = {}
    for nm, b_, s_ in zip(["const"] + list(cols), beta, se):
        print(f"  {nm:<34} {b_:+.4f}  t={b_ / s_:+.2f}")
        diag[nm] = [round(float(b_), 4), round(float(b_ / s_), 2)]
    print(f"  R2 = {1 - res.var() / r.var():.3f}; sd(close-model logit) = {r.std():.4f}")
    out["residual_ols"] = diag
    out["residual_sd"] = float(r.std())
    out["residual_r2"] = float(1 - res.var() / r.var())

    # How much LL is the residual worth, and how much of it is explained?
    # (i) close itself vs model = the gap; (ii) model logit + fitted conditions part
    fit = lgm + Z @ beta - (Z[:, -1] * beta[-1]) - beta[0]  # conditions-only shift
    pf = 1 / (1 + np.exp(-fit))
    print(f"\nmodel + conditions-shift (in-sample diagnostic, NOT a model): "
          f"{llv(ym, pf).mean():.5f} vs model {lm.mean():.5f}")
    out["cond_shift_ll"] = float(llv(ym, pf).mean())
    json.dump(out, open("data/bt_nhl_ideas_anat.json", "w"), indent=1)
    print("wrote data/bt_nhl_ideas_anat.json")


if __name__ == "__main__":
    main()
