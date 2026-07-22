"""Phase 0.1 — measure the starting-pitcher prize without building a model.

FiveThirtyEight published two probabilities per game:
  elo_prob1     — team Elo ratings ONLY
  rating_prob1  — ratings + preseason projections + STARTING PITCHER adjustment

The gap between them on the same games is an upper bound on what a
starting-pitcher layer can buy us, because rating_prob1 also carries preseason
projections we will never have. If that upper bound is small, the entire
premise of the rebuild is dead and we ship a plain rating instead.

Everything here is a read-off. No model is fitted to the outcome except a
diagnostic regression that recovers the optimal weight on the pitcher term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CSV = "data/mlb_elo.csv"
EPS = 1e-15


def ll_vec(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p: np.ndarray, y: np.ndarray) -> float:
    return float(ll_vec(p, y).mean())


def paired_boot(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 7):
    """CI for mean(a - b); positive means b is better (lower loss)."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main() -> None:
    df = pd.read_csv(CSV, low_memory=False)
    print(f"raw rows: {len(df):,}")

    # Completed regular-season games where BOTH systems published a probability.
    d = df[
        df["score1"].notna() & df["score2"].notna()
        & df["elo_prob1"].notna() & df["rating_prob1"].notna()
        & df["playoff"].isna()
    ].copy()
    d["date"] = pd.to_datetime(d["date"])
    # Ties are impossible in modern MLB but exist in 19th-century data.
    d = d[d["score1"] != d["score2"]]
    d["y"] = (d["score1"] > d["score2"]).astype(float)
    print(f"usable regular-season games with both systems: {len(d):,}")
    print(f"season range: {int(d.season.min())}..{int(d.season.max())}")

    # The pitcher adjustment only exists in the modern predictions era.
    has_adj = d["pitcher1_adj"].notna() & d["pitcher2_adj"].notna()
    print(f"games with explicit pitcher adjustments: {has_adj.sum():,} "
          f"(seasons {int(d.loc[has_adj,'season'].min())}..{int(d.loc[has_adj,'season'].max())})")

    def report(sub: pd.DataFrame, label: str) -> dict:
        y = sub["y"].to_numpy(float)
        pe = sub["elo_prob1"].to_numpy(float)
        pr = sub["rating_prob1"].to_numpy(float)
        le, lr = ll(pe, y), ll(pr, y)
        gap, lo, hi = paired_boot(ll_vec(pe, y), ll_vec(pr, y))
        print(f"\n=== {label}  (n={len(sub):,}) ===")
        print(f"  elo_prob1     (ratings only)      LL = {le:.5f}")
        print(f"  rating_prob1  (+proj +pitcher)    LL = {lr:.5f}")
        print(f"  GAP (elo - rating)                   = {gap:+.5f} nats")
        print(f"  95% paired bootstrap CI              = [{lo:+.5f}, {hi:+.5f}]")
        print(f"  home win rate {y.mean():.4f} | constant-0.5 LL {ll(np.full_like(y,0.5),y):.5f}")
        return {"n": len(sub), "elo": le, "rating": lr, "gap": gap, "lo": lo, "hi": hi}

    modern = d[has_adj]
    overall = report(modern, "MODERN ERA — pitcher adjustments present")

    # Separate the pitcher effect from the preseason-projection effect.
    # Projections dominate early (little in-season evidence yet) and fade;
    # the pitcher term is constant all year. So May-onward is the cleaner read.
    apr = modern[modern["date"].dt.month <= 4]
    rest = modern[modern["date"].dt.month >= 5]
    report(apr, "APRIL ONLY (projection effect strongest)")
    late = report(rest, "MAY ONWARD (projection effect decayed)")

    # --- diagnostic: how much of the gap is the pitcher term specifically? ---
    m = modern.copy()
    m["elo_logit"] = np.log(np.clip(m.elo_prob1, EPS, 1 - EPS) /
                            np.clip(1 - m.elo_prob1, EPS, 1 - EPS))
    m["sp_diff"] = m["pitcher1_adj"] - m["pitcher2_adj"]

    from sklearn.linear_model import LogisticRegression
    X = m[["elo_logit", "sp_diff"]].to_numpy(float)
    y = m["y"].to_numpy(float)
    lr_fit = LogisticRegression(max_iter=2000, C=1e6).fit(X, y)
    b_elo, b_sp = lr_fit.coef_[0]
    print("\n=== diagnostic regression: outcome ~ elo_logit + sp_diff ===")
    print(f"  beta(elo_logit) = {b_elo:.4f}   (1.0 = Elo already well calibrated)")
    print(f"  beta(sp_diff)   = {b_sp:.4f}   (>0 = pitcher term carries real signal)")
    print(f"  sp_diff sd      = {m.sp_diff.std():.4f}")
    p_fit = lr_fit.predict_proba(X)[:, 1]
    gap2, lo2, hi2 = paired_boot(ll_vec(m.elo_prob1.to_numpy(float), y), ll_vec(p_fit, y))
    print(f"  IN-SAMPLE elo+sp vs elo:  {gap2:+.5f} nats  CI [{lo2:+.5f}, {hi2:+.5f}]")
    print("  (in-sample and therefore OPTIMISTIC — an upper bound on the SP term alone)")

    # --- verdict against the pre-registered decision table ---
    g = late["gap"]
    print("\n" + "=" * 62)
    print(f"DECISION INPUT: May-onward gap = {g:+.5f} nats "
          f"(CI [{late['lo']:+.5f}, {late['hi']:+.5f}])")
    if g >= 0.008:
        v = "FULL PLAN — Phase 4 justified"
    elif g >= 0.005:
        v = "FULL PLAN, Phase 4 capped at 10 days hard"
    elif g >= 0.003:
        v = "PHASES 0-3 ONLY — Phase 3 dev result decides if Phase 4 ever runs"
    else:
        v = "STOP MODELLING — ship Elo + site, write the negative-result page"
    print(f"VERDICT: {v}")
    print("=" * 62)


if __name__ == "__main__":
    main()
