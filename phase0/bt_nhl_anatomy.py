"""NHL gap anatomy -- step 2: WHERE does the shipped blend lose to the close?

Breakthrough program (documents/breakthrough_program_prereg_2026_09_24.md).
DEV ONLY (2011-12 .. 2017-18). No TEST-era season is loaded into a rating,
joined to odds, scored or printed: the game list is truncated by
nhl_depth_eval.dev_games() and every scored mask passes assert_dev_only().

MODEL: the shipped 4-feature blend (tuned Elo logit + rest_diff + b2b_home +
b2b_away + xG-Elo diff) reproduced EXACTLY as the recorded DEV harness does it
(phase0/nhl_depth_eval.py Ctx + leave-one-season-out, SHIPPED_ELO/SHIPPED_XG):
each DEV season is predicted by a blend fit on the other six DEV seasons.
Asserted to reproduce the recorded DEV LOSO log-loss 0.67239 (n=7,929) and,
on the 7,927 odds-joined games, nhl_floor.json (model 0.672423 / close 0.670717).

MARKET: data/odds_nhl.csv (SBR archive), multiplicative de-vig of the OPEN and
the CLOSE moneyline. EVALUATION ONLY -- never a feature, never a target.

Per game:  d_close = LL(model) - LL(close)   (positive = model loses)
           d_open  = LL(model) - LL(open)
           mv      = LL(open)  - LL(close)   (what the line learned after opening)
Slices from data/bt_nhl_anatomy_feats.csv (phase0/bt_nhl_anatomy_build.py).

Output: data/bt_nhl_anatomy.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, TEST_START, llv  # noqa: E402

OUT = "data/bt_nhl_anatomy.json"
RNG = np.random.default_rng(20260924)
NB = 4000


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def reproduce():
    ctx = Hd.Ctx()
    y = ctx.yv()
    X = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p = ctx.folds.loso_pred(X, y)
    seas = ctx.seasons[ctx.mask]
    Hd.assert_dev_only(seas)
    G = [g for g, k in zip(ctx.games, ctx.mask) if k]
    ll = float(llv(y, p).mean())
    print(f"reproduced shipped DEV LOSO: LL {ll:.6f} n={len(y)} "
          f"(recorded 0.67239 on 7,929 in nhl_dev_screens_2026_08_11.md)")
    assert len(y) == 7929 and abs(ll - 0.67239) < 2e-5, "shipped DEV harness not reproduced"
    # also the pieces, for the 'what does the market know' regression
    Xcols = {"elo_logit": X[:, 1], "rest_diff": X[:, 2], "b2b_home": X[:, 3],
             "b2b_away": X[:, 4], "xg_diff": X[:, 5]}
    return p, y, seas, G, Xcols


def join_odds(G, y):
    idx = defaultdict(list)
    for r in csv.DictReader(open("data/odds_nhl.csv", encoding="utf-8")):
        if r["date"] >= "2018-07-01":          # DEV era ends 2018-06; never read TEST
            continue
        idx[(r["date"], r["home"], r["away"])].append(r)
    pc = np.full(len(G), np.nan)
    po = np.full(len(G), np.nan)
    over = np.full(len(G), np.nan)
    for i, g in enumerate(G):
        m = idx.get((g["date"], g["home"], g["away"]))
        if not m or len(m) != 1:
            continue
        r = m[0]
        assert int(r["home_win"]) == int(y[i]), "result mismatch -> bad join"
        if r["home_close"] and r["away_close"]:
            qh, qa = 1 / float(r["home_close"]), 1 / float(r["away_close"])
            pc[i] = qh / (qh + qa)
            over[i] = qh + qa
        if r["home_open"] and r["away_open"]:
            qh, qa = 1 / float(r["home_open"]), 1 / float(r["away_open"])
            if 0.9 < qh + qa < 1.2:
                po[i] = qh / (qh + qa)
    return pc, po, over


def boot_mean(x):
    if len(x) < 2:
        return [float("nan")] * 2
    bs = x[RNG.integers(0, len(x), size=(NB, len(x)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


def boot_diff(a, b):
    """CI of mean(a) - mean(b), independent resampling of the two groups."""
    if len(a) < 2 or len(b) < 2:
        return [float("nan")] * 2
    ba = a[RNG.integers(0, len(a), size=(NB, len(a)))].mean(axis=1)
    bb = b[RNG.integers(0, len(b), size=(NB, len(b)))].mean(axis=1)
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


class Anatomy:
    def __init__(self, dc, do, mv, dlt, ok_open):
        self.dc, self.do, self.mv, self.dlt = dc, do, mv, dlt
        self.ok_open = ok_open
        self.total = float(dc.sum())
        self.N = len(dc)

    def slice(self, mask, label):
        m = np.asarray(mask, bool)
        n = int(m.sum())
        if n < 30:
            return None
        dc = self.dc[m]
        rest = self.dc[~m]
        mo = m & self.ok_open
        out = {
            "slice": label, "n": n, "n_share": round(n / self.N, 4),
            "gap_close": round(float(dc.mean()), 5), "ci": boot_mean(dc),
            "gap_share": round(float(dc.sum()) / self.total, 3),
            "excess_vs_rest": round(float(dc.mean() - rest.mean()), 5)
            if len(rest) else None,
            "excess_ci": boot_diff(dc, rest) if len(rest) > 30 else None,
            "gap_open": round(float(self.do[mo].mean()), 5) if mo.sum() > 30 else None,
            "open_to_close": round(float(self.mv[mo].mean()), 5) if mo.sum() > 30 else None,
            "oc_ci": boot_mean(self.mv[mo]) if mo.sum() > 30 else None,
            "mean_mkt_minus_model_logit": round(float(self.dlt[m].mean()), 4),
            "mean_abs_mkt_minus_model_logit": round(float(np.abs(self.dlt[m]).mean()), 4),
        }
        return out


def fmt(r):
    if r is None:
        return ""
    ex = r["excess_vs_rest"]
    exs = (f"{ex:+.5f} [{r['excess_ci'][0]:+.5f},{r['excess_ci'][1]:+.5f}]"
           if ex is not None else "")
    go = f"{r['gap_open']:+.5f}" if r["gap_open"] is not None else "   n/a  "
    oc = f"{r['open_to_close']:+.5f}" if r["open_to_close"] is not None else "   n/a  "
    return (f"  {r['slice']:<34} n={r['n']:>5} ({r['n_share']:.3f})  "
            f"gap {r['gap_close']:+.5f} [{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] "
            f"share {r['gap_share']:+.2f}  excess {exs}  "
            f"| vsOpen {go} open->close {oc}  |dlogit| {r['mean_abs_mkt_minus_model_logit']:.3f} "
            f"dlogit {r['mean_mkt_minus_model_logit']:+.3f}")


def main():
    p, y, seas, G, Xcols = reproduce()
    pc, po, over = join_odds(G, y)
    ok = ~np.isnan(pc)
    print(f"joined close: {int(ok.sum())}/{len(G)}; open usable {int((ok & ~np.isnan(po)).sum())}")

    # restrict everything to joined games
    keep = np.where(ok)[0]
    p, y, seas, pc, po, over = p[keep], y[keep], seas[keep], pc[keep], po[keep], over[keep]
    G = [G[i] for i in keep]
    Xcols = {k: v[keep] for k, v in Xcols.items()}
    Hd.assert_dev_only(seas)
    ok_open = ~np.isnan(po)

    ll_m = llv(y, p)
    ll_c = llv(y, pc)
    print(f"on joined games: model {ll_m.mean():.6f} close {ll_c.mean():.6f} "
          f"(recorded nhl_floor.json 0.672423 / 0.670717, n=7927)")
    assert len(y) == 7927 and abs(ll_m.mean() - 0.672423) < 2e-6
    assert abs(ll_c.mean() - 0.670717) < 2e-6
    ll_o = np.where(ok_open, llv(y, np.nan_to_num(po, nan=0.5)), np.nan)
    dc = ll_m - ll_c
    do = ll_m - ll_o
    mv = ll_o - ll_c
    dlt = logit(pc) - logit(p)
    A = Anatomy(dc, do, mv, dlt, ok_open)

    head = {
        "n": len(y), "model_ll": round(float(ll_m.mean()), 6),
        "close_ll": round(float(ll_c.mean()), 6),
        "gap_close": round(float(dc.mean()), 6), "gap_close_ci": boot_mean(dc),
        "n_open": int(ok_open.sum()),
        "model_ll_on_open_games": round(float(ll_m[ok_open].mean()), 6),
        "open_ll": round(float(np.nanmean(ll_o)), 6),
        "close_ll_on_open_games": round(float(ll_c[ok_open].mean()), 6),
        "gap_open": round(float(np.nanmean(do)), 6), "gap_open_ci": boot_mean(do[ok_open]),
        "open_to_close_gain": round(float(np.nanmean(mv)), 6),
        "open_to_close_ci": boot_mean(mv[ok_open]),
        "sd_logit_mkt_minus_model": round(float(dlt.std()), 4),
        "corr_logit_model_close": round(float(np.corrcoef(logit(p), logit(pc))[0, 1]), 4),
        "sd_logit_model": round(float(logit(p).std()), 4),
        "sd_logit_close": round(float(logit(pc).std()), 4),
        "sd_logit_open": round(float(logit(po[ok_open]).std()), 4),
    }
    # outcome weights: y ~ a + b*logit(model) + c*logit(close)  (descriptive)
    X2 = Hd.design([logit(p), logit(pc)])
    w2 = Hd.fit_logit(X2, y)
    head["outcome_weights_model_close"] = [round(float(x), 3) for x in w2]
    Xo = Hd.design([logit(p[ok_open]), logit(po[ok_open])])
    head["outcome_weights_model_open"] = [round(float(x), 3)
                                          for x in Hd.fit_logit(Xo, y[ok_open])]
    head["note_weights"] = ("logit weights of y on [1, logit model, logit close], "
                            "in-sample DEV, descriptive only")
    print("\n=== HEADLINE (DEV 2011-12..2017-18, joined games) ===")
    for k, v in head.items():
        print(f"  {k:<32} {v}")

    # ---- slices -------------------------------------------------------------
    F = pd.read_csv("data/bt_nhl_anatomy_feats.csv").set_index("game_id")
    gids = [g["game_id"] for g in G]
    F = F.loc[gids]
    assert F.season.max() <= DEV_END < TEST_START
    f = {c: F[c].to_numpy() for c in F.columns if c not in ("date", "home", "away")}
    b2bh, b2ba = Xcols["b2b_home"], Xcols["b2b_away"]
    fav_p = np.maximum(pc, 1 - pc)
    home_fav = pc >= 0.5
    mgp = np.minimum(f["gp_home"], f["gp_away"])

    S = {}

    def add(group, mask, label):
        r = A.slice(mask, label)
        if r is not None:
            S.setdefault(group, []).append(r)

    for s in sorted(set(seas.tolist())):
        add("season", seas == s, f"season {s}")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 90)):
        add("phase_min_gp", (mgp >= lo) & (mgp < hi), f"min team gp {lo}-{hi - 1}")
    for mth, lab in ((10, "Oct"), (11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"),
                     (3, "Mar"), (4, "Apr")):
        add("month", f["month"] == mth, lab)
    for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.0)):
        add("fav_size_close", (fav_p >= lo) & (fav_p < hi), f"close fav p {lo:.2f}-{hi:.2f}")
    add("fav_side", home_fav, "home favourite (close)")
    add("fav_side", ~home_fav, "away favourite (close)")
    q = np.quantile(dlt, [0.1, 0.25, 0.5, 0.75, 0.9])
    edges = [-np.inf] + list(q) + [np.inf]
    for a_, b_ in zip(edges[:-1], edges[1:]):
        add("mkt_minus_model_logit", (dlt >= a_) & (dlt < b_),
            f"dlogit [{a_:+.3f},{b_:+.3f})")
    add("rest", (b2bh == 0) & (b2ba == 0), "no b2b")
    add("rest", (b2bh == 1) & (b2ba == 0), "home b2b only")
    add("rest", (b2bh == 0) & (b2ba == 1), "away b2b only")
    add("rest", (b2bh == 1) & (b2ba == 1), "both b2b")
    rd = np.nan_to_num(f["rest_home"], nan=3) - np.nan_to_num(f["rest_away"], nan=3)
    add("rest", np.abs(rd) >= 2, "|rest diff| >= 2 days")

    # goalie
    bh, ba = f["g_backup_home"], f["g_backup_away"]
    anyb = (bh == 1) | (ba == 1)
    known = ~np.isnan(bh) & ~np.isnan(ba)
    add("goalie", known & ~anyb, "both usual starters (share>=.5)")
    add("goalie", known & (bh == 1) & (ba != 1), "home backup only")
    add("goalie", known & (ba == 1) & (bh != 1), "away backup only")
    add("goalie", known & (bh == 1) & (ba == 1), "both backups")
    sh_h, sh_a = f["g_share_home"], f["g_share_away"]
    add("goalie", (sh_h < 0.3) | (sh_a < 0.3), "any clear backup (share<.3)")
    add("goalie", (sh_h < 0.15) | (sh_a < 0.15), "any rare starter (share<.15)")
    ch = (f["g_changed_home"] == 1) | (f["g_changed_away"] == 1)
    add("goalie", ch, "starter changed vs prev game (any)")
    add("goalie", (f["g_changed_home"] == 0) & (f["g_changed_away"] == 0), "same starters as prev game")
    add("goalie", (f["g_career_home"] < 15) | (f["g_career_away"] < 15), "inexperienced starter (<15 starts)")
    gdl = np.nan_to_num(f["g_delta_home"]) - np.nan_to_num(f["g_delta_away"])
    add("goalie", np.abs(gdl) >= 0.25, "|goalie delta diff| >= .25 GSAx/g")
    add("goalie_x_b2b", (b2bh == 1) & (bh == 1), "home b2b & home backup")
    add("goalie_x_b2b", (b2bh == 1) & (bh == 0), "home b2b & home usual")
    add("goalie_x_b2b", (b2ba == 1) & (ba == 1), "away b2b & away backup")
    add("goalie_x_b2b", (b2ba == 1) & (ba == 0), "away b2b & away usual")
    add("goalie_x_b2b", (b2bh == 0) & (b2ba == 0) & anyb, "no b2b & any backup")

    # lineup absences
    at_h, at_a = f["abs_toi_home"], f["abs_toi_away"]
    lk = ~np.isnan(at_h) & ~np.isnan(at_a)
    atd = np.abs(np.nan_to_num(at_h) - np.nan_to_num(at_a))
    add("lineup", lk & (atd < 5), "|absent TOI diff| < 5 min")
    add("lineup", lk & (atd >= 5) & (atd < 20), "|absent TOI diff| 5-20 min")
    add("lineup", lk & (atd >= 20) & (atd < 40), "|absent TOI diff| 20-40 min")
    add("lineup", lk & (atd >= 40), "|absent TOI diff| >= 40 min")
    t4 = np.nan_to_num(f["abs_top4_home"]) + np.nan_to_num(f["abs_top4_away"])
    add("lineup", lk & (t4 == 0), "no top-4-TOI regular missing")
    add("lineup", lk & (t4 >= 1), "any top-4-TOI regular missing")
    add("lineup", lk & (t4 >= 2), "2+ top-4-TOI regulars missing")
    nn = np.nan_to_num(f["new_n_home"]) + np.nan_to_num(f["new_n_away"])
    add("lineup", lk & (nn >= 3), "3+ skaters new to lineup (call-ups/trades)")
    add("lineup", ~lk, "lineup history unavailable (<5 gp)")

    # standings / stakes
    dead_any = ((f["home_elim"] == 1) | (f["away_elim"] == 1)
                | (f["home_lock"] == 1) | (f["away_lock"] == 1))
    add("stakes", dead_any, "either team eliminated or locked")
    add("stakes", (f["home_elim"] == 1) | (f["away_elim"] == 1), "either eliminated")
    add("stakes", (f["home_lock"] == 1) | (f["away_lock"] == 1), "either locked in")
    late = np.maximum(f["gp_home"], f["gp_away"]) >= 70
    add("stakes", late & ~dead_any, "gp>=70, both alive")
    add("stakes", late & dead_any, "gp>=70, a dead team")
    ph, pa = f["ptspct_home"], f["ptspct_away"]
    add("stakes", (mgp >= 20) & ((ph < 0.45) | (pa < 0.45)), "gp>=20 & a bad team (pts%<.45)")

    # form / model-vs-results
    fh, fa = f["form10_home"], f["form10_away"]
    add("form", np.abs(np.nan_to_num(fh) - np.nan_to_num(fa)) >= 0.4, "|L10 win% diff| >= .4")
    # context
    add("context", f["trav_away_km"] >= 2000, "away travelled >= 2000 km")
    add("context", np.abs(np.nan_to_num(f["tz_away"])) >= 2, "away tz shift >= 2")
    # teams: which team-season carries most gap
    team_rows = []
    for t in sorted(set([g["home"] for g in G] + [g["away"] for g in G])):
        m = np.array([g["home"] == t or g["away"] == t for g in G])
        r = A.slice(m, f"team {t}")
        if r:
            team_rows.append(r)
    S["team"] = team_rows

    print("\n=== SLICES  gap = LL(model)-LL(close), + = model loses; "
          "excess = slice gap - rest gap ===")
    for grp, rows in S.items():
        if grp == "team":
            continue
        print(f"\n[{grp}]")
        for r in rows:
            print(fmt(r))
    print("\n[team] (top/bottom 6 by gap)")
    ts = sorted(team_rows, key=lambda r: r["gap_close"], reverse=True)
    for r in ts[:6] + ts[-6:]:
        print(fmt(r))

    # ---- what does the market price that the model does not? (descriptive) ---
    # OLS of dlogit = logit(close) - logit(model) on market-blind descriptors.
    desc = {
        "home_backup": np.nan_to_num(bh), "away_backup": np.nan_to_num(ba),
        "goalie_delta_diff": gdl,
        "abs_toi_diff(a-h)": np.nan_to_num(at_a) - np.nan_to_num(at_h),
        "abs_top4_diff(a-h)": np.nan_to_num(f["abs_top4_away"]) - np.nan_to_num(f["abs_top4_home"]),
        "b2b_home": b2bh, "b2b_away": b2ba,
        "early(mgp<10)": (mgp < 10).astype(float),
        "home_dead": ((f["home_elim"] == 1) | (f["home_lock"] == 1)).astype(float),
        "away_dead": ((f["away_elim"] == 1) | (f["away_lock"] == 1)).astype(float),
        "form10_diff": np.nan_to_num(fh, nan=0.5) - np.nan_to_num(fa, nan=0.5),
        "ptspct_diff": np.nan_to_num(ph, nan=0.5) - np.nan_to_num(pa, nan=0.5),
        "model_logit": logit(p),
    }
    names = list(desc)
    Xd = np.column_stack([np.ones(len(y))] + [desc[k] for k in names])
    beta, *_ = np.linalg.lstsq(Xd, dlt, rcond=None)
    res = dlt - Xd @ beta
    r2 = 1 - res.var() / dlt.var()
    # per-term SE (OLS, homoskedastic) for orientation
    s2 = res.var() * len(y) / (len(y) - Xd.shape[1])
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    reg = {"r2": round(float(r2), 4), "terms": {}}
    print(f"\n=== OLS: logit(close)-logit(model) on descriptors (DEV, descriptive) "
          f"R2={r2:.4f} ===")
    for k, b_, s_ in zip(["const"] + names, beta, se):
        reg["terms"][k] = [round(float(b_), 4), round(float(s_), 4)]
        print(f"  {k:<22} {b_:+.4f}  (se {s_:.4f}, t {b_ / s_:+.1f})")
    # partial R2 of each block
    blocks = {"goalie": ["home_backup", "away_backup", "goalie_delta_diff"],
              "lineup": ["abs_toi_diff(a-h)", "abs_top4_diff(a-h)"],
              "rest": ["b2b_home", "b2b_away"],
              "stakes": ["home_dead", "away_dead"],
              "standings_form": ["form10_diff", "ptspct_diff"],
              "early": ["early(mgp<10)"],
              "model_level": ["model_logit"]}
    reg["drop_block_r2_loss"] = {}
    for bn, cols in blocks.items():
        keepc = [k for k in names if k not in cols]
        Xk = np.column_stack([np.ones(len(y))] + [desc[k] for k in keepc])
        bk, *_ = np.linalg.lstsq(Xk, dlt, rcond=None)
        rk = 1 - (dlt - Xk @ bk).var() / dlt.var()
        reg["drop_block_r2_loss"][bn] = round(float(r2 - rk), 4)
        print(f"  drop {bn:<16} R2 loss {r2 - rk:.4f}")

    # does the OUTCOME confirm the explained part of the market move?
    # y ~ logit(model) + explained dlogit + residual dlogit (descriptive only)
    expl = Xd @ beta - beta[0]
    X4 = Hd.design([logit(p), expl, res])
    w4 = Hd.fit_logit(X4, y)
    ll4 = float(llv(y, 1 / (1 + np.exp(-(X4 @ w4)))).mean())
    reg["outcome_on_explained_residual"] = {
        "weights[const,model,explained,resid]": [round(float(x), 3) for x in w4],
        "insample_ll": round(ll4, 6)}
    print(f"  outcome weights [const, model, explained dlogit, residual dlogit] "
          f"{[round(float(x), 3) for x in w4]}  in-sample LL {ll4:.5f}")

    # Gap decomposition by |dlogit| concentration: how much of the total gap
    # sits in the games where market and model disagree most?
    order = np.argsort(-np.abs(dlt))
    cum = np.cumsum(dc[order]) / dc.sum()
    conc = {f"top{int(fr * 100)}pct_disagreement": round(float(cum[int(fr * len(dc)) - 1]), 3)
            for fr in (0.05, 0.1, 0.2, 0.3, 0.5)}
    print(f"\n=== share of total gap in the top-x% |market-model| disagreement games ===\n  {conc}")

    out = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
           "protocol": {"dev_only": True, "dev_seasons": sorted(set(int(s) for s in seas)),
                        "test_touched": False, "market_use": "evaluation only",
                        "model": "shipped 4-feature blend, DEV LOSO via nhl_depth_eval.Ctx; "
                                 "reproduces recorded 0.672423"},
           "headline": head, "slices": S, "market_minus_model_regression": reg,
           "gap_concentration": conc}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
