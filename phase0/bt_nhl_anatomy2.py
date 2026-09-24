"""NHL gap anatomy -- step 3: is the market RIGHT to react, and where does the
open->close gain live?  DEV ONLY, descriptive.

For each market-blind day-of descriptor D (from data/bt_nhl_anatomy_feats.csv):
  (a) MARKET reaction beyond the model : OLS  logit(close)-logit(model) ~ D + logit(model)
  (b) OUTCOME beyond the model         : y ~ [1, logit(model), D], leave-one-DEV-
      season-out, paired dLL vs y ~ [1, logit(model)] LOSO, bootstrap CI
  (c) OUTCOME beyond the close         : y ~ [1, logit(close), D] in-sample coef/z
      (~0 => the close already prices D correctly)
If (b) > 0 and (c) ~ 0 the close has the information and it is recoverable from
a public, market-blind descriptor. Odds are evaluation-only throughout.

Also: where the open->close gain (the whole DEV gap) concentrates.
Output: data/bt_nhl_anatomy2.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, TEST_START, llv  # noqa: E402
from bt_nhl_anatomy import reproduce, join_odds, logit, boot_mean  # noqa: E402

OUT = "data/bt_nhl_anatomy2.json"
RNG = np.random.default_rng(924)


def loso_ll(X, y, seas):
    p = np.empty(len(y))
    for s in sorted(set(seas.tolist())):
        tr, te = seas != s, seas == s
        w = Hd.fit_logit(X[tr], y[tr])
        p[te] = Hd._sig(X[te] @ w)
    return p


def main():
    p, y, seas, G, Xcols = reproduce()
    pc, po, _ = join_odds(G, y)
    keep = np.where(~np.isnan(pc))[0]
    p, y, seas, pc, po = p[keep], y[keep], seas[keep], pc[keep], po[keep]
    G = [G[i] for i in keep]
    Xcols = {k: v[keep] for k, v in Xcols.items()}
    Hd.assert_dev_only(seas)
    lm, lc, lo_ = logit(p), logit(pc), logit(po)
    dlt = lc - lm

    F = pd.read_csv("data/bt_nhl_anatomy_feats.csv").set_index("game_id")
    F = F.loc[[g["game_id"] for g in G]]
    assert F.season.max() <= DEV_END < TEST_START
    f = {c: F[c].to_numpy() for c in F.columns if c not in ("date", "home", "away")}
    z = np.nan_to_num

    modal_h = f["g_rtg_home"] - f["g_delta_home"]
    modal_a = f["g_rtg_away"] - f["g_delta_away"]
    q80 = np.nanquantile(np.concatenate([modal_h, modal_a]), 0.8)
    elite_out_h = ((modal_h >= q80) & (f["g_notmodal_home"] == 1)).astype(float)
    elite_out_a = ((modal_a >= q80) & (f["g_notmodal_away"] == 1)).astype(float)
    lc_h = np.log1p(z(f["g_career_home"]))
    lc_a = np.log1p(z(f["g_career_away"]))

    D = {
        # goalie (tonight's starter, confirmed pre-game)
        "backup_diff (a-h)": z(f["g_backup_away"]) - z(f["g_backup_home"]),
        "share_diff (h-a)": z(f["g_share_home"], nan=.55) - z(f["g_share_away"], nan=.55),
        "notmodal_diff (a-h)": z(f["g_notmodal_away"]) - z(f["g_notmodal_home"]),
        "log_career_starts_diff (h-a)": lc_h - lc_a,
        "gsax_delta_diff (h-a)": z(f["g_delta_home"]) - z(f["g_delta_away"]),
        "gsax_level_diff (h-a)": z(f["g_rtg_home"]) - z(f["g_rtg_away"]),
        "elite_usual_out (a-h)": elite_out_a - elite_out_h,
        # skaters (tonight's dressed lineup vs trailing-10 regulars)
        "absent_toi_diff (a-h) min": z(f["abs_toi_away"]) - z(f["abs_toi_home"]),
        "absent_top4_diff (a-h)": z(f["abs_top4_away"]) - z(f["abs_top4_home"]),
        "absent_n_diff (a-h)": z(f["abs_n_away"]) - z(f["abs_n_home"]),
        "new_skaters_diff (a-h)": z(f["new_n_away"]) - z(f["new_n_home"]),
        # standings / stakes / form (all as-of, pre-game)
        "ptspct_diff (h-a)": z(f["ptspct_home"], nan=.5) - z(f["ptspct_away"], nan=.5),
        "form10_diff (h-a)": z(f["form10_home"], nan=.5) - z(f["form10_away"], nan=.5),
        "dead_diff (a-h)": (((f["away_elim"] == 1) | (f["away_lock"] == 1)).astype(float)
                            - ((f["home_elim"] == 1) | (f["home_lock"] == 1)).astype(float)),
        "elim_diff (a-h)": (f["away_elim"] == 1).astype(float) - (f["home_elim"] == 1).astype(float),
        "lock_diff (a-h)": (f["away_lock"] == 1).astype(float) - (f["home_lock"] == 1).astype(float),
        "both_b2b": ((Xcols["b2b_home"] == 1) & (Xcols["b2b_away"] == 1)).astype(float),
        "gconsec_diff (h-a)": z(f["gconsec_home"]) - z(f["gconsec_away"]),
    }

    one = np.ones(len(y))
    p_base = loso_ll(np.column_stack([one, lm]), y, seas)
    ll_base = llv(y, p_base)
    rows = {}
    print(f"n={len(y)}  base (recalibrated model LOSO) LL {ll_base.mean():.6f}")
    print(f"{'descriptor':<30} {'sd':>6} | {'mkt coef':>9} {'t':>6} | "
          f"{'out coef':>9} {'dLL LOSO':>9} {'95% CI':>20} | {'vs close coef':>13} {'z':>5}")
    for name, d in D.items():
        sd = float(d.std())
        if sd == 0:
            continue
        Xa = np.column_stack([one, d, lm])
        ba, *_ = np.linalg.lstsq(Xa, dlt, rcond=None)
        ra = dlt - Xa @ ba
        s2 = ra.var() * len(y) / (len(y) - 3)
        sea = np.sqrt(np.diag(s2 * np.linalg.inv(Xa.T @ Xa)))
        Xb = np.column_stack([one, lm, d])
        pb = loso_ll(Xb, y, seas)
        dd = ll_base - llv(y, pb)
        ci = boot_mean(dd)
        wb = Hd.fit_logit(Xb, y)
        Xc = np.column_stack([one, lc, d])
        wc = Hd.fit_logit(Xc, y)
        pcc = Hd._sig(Xc @ wc)
        W = pcc * (1 - pcc)
        cov = np.linalg.inv((Xc * W[:, None]).T @ Xc)
        zc = wc[2] / np.sqrt(cov[2, 2])
        rows[name] = {"sd": round(sd, 4), "nonzero_share": round(float((d != 0).mean()), 3),
                      "market_coef": round(float(ba[1]), 4), "market_t": round(float(ba[1] / sea[1]), 1),
                      "outcome_coef_beyond_model": round(float(wb[2]), 4),
                      "dLL_loso_beyond_model": round(float(dd.mean()), 5), "ci": ci,
                      "outcome_coef_beyond_close": round(float(wc[2]), 4),
                      "z_beyond_close": round(float(zc), 2)}
        print(f"{name:<30} {sd:>6.3f} | {ba[1]:>+9.4f} {ba[1] / sea[1]:>+6.1f} | "
              f"{wb[2]:>+9.4f} {dd.mean():>+9.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] | "
              f"{wc[2]:>+13.4f} {zc:>+5.1f}")

    # joint: all day-of lineup descriptors together (goalie + skaters)
    joint_sets = {
        "GOALIE block (backup, share, career, gsax delta)":
            ["backup_diff (a-h)", "share_diff (h-a)", "log_career_starts_diff (h-a)",
             "gsax_delta_diff (h-a)"],
        "SKATER block (absent toi, top4, new)":
            ["absent_toi_diff (a-h) min", "absent_top4_diff (a-h)", "new_skaters_diff (a-h)"],
        "STAKES block (elim, lock)": ["elim_diff (a-h)", "lock_diff (a-h)"],
        "GOALIE+SKATER": ["backup_diff (a-h)", "share_diff (h-a)", "log_career_starts_diff (h-a)",
                          "gsax_delta_diff (h-a)", "absent_toi_diff (a-h) min",
                          "absent_top4_diff (a-h)", "new_skaters_diff (a-h)"],
    }
    joint = {}
    print("\nJOINT blocks over recalibrated model (LOSO dLL, in-sample market R2 gain):")
    for jn, cols in joint_sets.items():
        Xb = np.column_stack([one, lm] + [D[c] for c in cols])
        pb = loso_ll(Xb, y, seas)
        dd = ll_base - llv(y, pb)
        ci = boot_mean(dd)
        Xa = np.column_stack([one, lm] + [D[c] for c in cols])
        ba, *_ = np.linalg.lstsq(Xa, dlt, rcond=None)
        X0 = np.column_stack([one, lm])
        b0, *_ = np.linalg.lstsq(X0, dlt, rcond=None)
        r2 = (1 - (dlt - Xa @ ba).var() / dlt.var()) - (1 - (dlt - X0 @ b0).var() / dlt.var())
        joint[jn] = {"dLL_loso": round(float(dd.mean()), 5), "ci": ci,
                     "market_r2_gain": round(float(r2), 4)}
        print(f"  {jn:<52} dLL {dd.mean():+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]  "
              f"market R2 gain {r2:.4f}")

    # ---- where does the open->close gain live? ------------------------------
    llo, llc, llm = llv(y, po), llv(y, pc), llv(y, p)
    mv = llo - llc
    tot = mv.sum()
    anyb = (z(f["g_backup_home"]) == 1) | (z(f["g_backup_away"]) == 1)
    top4 = (z(f["abs_top4_home"]) + z(f["abs_top4_away"])) >= 1
    usual = (~anyb) & (~top4) & ~np.isnan(f["abs_top4_home"]) & ~np.isnan(f["g_backup_home"])
    blocks = {
        "any backup goalie": anyb,
        "no backup goalie": ~anyb,
        "any top-4 skater absent": top4,
        "backup OR top-4 absent": anyb | top4,
        "fully usual lineups (no backup, no top-4 absent)": usual,
    }
    oc = {}
    print("\nOPEN->CLOSE gain concentration (DEV total "
          f"{mv.mean():+.5f}/game) and model-vs-close / model-vs-open per block:")
    for bn, m in blocks.items():
        r = {"n": int(m.sum()), "open_to_close": round(float(mv[m].mean()), 5),
             "oc_ci": boot_mean(mv[m]), "share_of_oc_gain": round(float(mv[m].sum() / tot), 3),
             "model_minus_close": round(float((llm - llc)[m].mean()), 5),
             "mc_ci": boot_mean((llm - llc)[m]),
             "model_minus_open": round(float((llm - llo)[m].mean()), 5),
             "mo_ci": boot_mean((llm - llo)[m]),
             "mean_abs_line_move_logit": round(float(np.abs(lc - lo_)[m].mean()), 4)}
        oc[bn] = r
        print(f"  {bn:<50} n={r['n']:>5}  open->close {r['open_to_close']:+.5f} "
              f"[{r['oc_ci'][0]:+.5f},{r['oc_ci'][1]:+.5f}] share {r['share_of_oc_gain']:+.2f}"
              f" | model-close {r['model_minus_close']:+.5f} [{r['mc_ci'][0]:+.5f},{r['mc_ci'][1]:+.5f}]"
              f" | model-open {r['model_minus_open']:+.5f} | |move| {r['mean_abs_line_move_logit']:.3f}")

    # line move direction vs goalie news: does the close move AGAINST backups?
    mvl = lc - lo_
    bd = D["backup_diff (a-h)"]
    Xm = np.column_stack([one, bd, D["absent_toi_diff (a-h) min"], D["absent_top4_diff (a-h)"]])
    bm, *_ = np.linalg.lstsq(Xm, mvl, rcond=None)
    rm = mvl - Xm @ bm
    s2 = rm.var() * len(y) / (len(y) - Xm.shape[1])
    sem = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
    move = {k: [round(float(b), 4), round(float(b / s), 1)] for k, b, s in
            zip(["const", "backup_diff", "absent_toi_diff", "absent_top4_diff"], bm, sem)}
    print(f"\nline MOVE (logit close - logit open) ~ day-of descriptors: {move}  "
          f"R2 {1 - rm.var() / mvl.var():.4f}")

    # ---- elite-goalie-out case study -----------------------------------------
    eo = (elite_out_h + elite_out_a) > 0
    r = {"n": int(eo.sum()), "model_minus_close": round(float((llm - llc)[eo].mean()), 5),
         "ci": boot_mean((llm - llc)[eo]),
         "open_to_close": round(float(mv[eo].mean()), 5)}
    print(f"\nelite usual goalie (top-20% EB GSAx) NOT starting, either side: {r}")
    # per-team-season: MTL 2015-16 (Price injured) as the canonical case
    ts = {}
    for t in ("MTL", "CBJ", "NSH", "PHI", "TOR", "CHI"):
        for s in sorted(set(seas.tolist())):
            m = np.array([(g["home"] == t or g["away"] == t) for g in G]) & (seas == s)
            if m.sum() < 30:
                continue
            ts[f"{t} {s}"] = round(float((llm - llc)[m].mean()), 5)
    worst = sorted(ts.items(), key=lambda kv: -kv[1])[:8]
    print(f"worst team-seasons (model-close): {worst}")

    json.dump({"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
               "protocol": {"dev_only": True, "test_touched": False,
                            "market_use": "evaluation only; descriptive"},
               "descriptors": rows, "joint_blocks": joint, "open_to_close": oc,
               "line_move_regression": move, "elite_goalie_out": r,
               "worst_team_seasons": worst},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
