"""Head-to-head: mitchellz4 XGBoost vs Elo, on a test set big enough to discriminate.

The repo evaluates on 1,002 games, where the 95% CI half-width on log loss is
~0.009 — wider than the entire gap between a good model and a mediocre one. Here
the split is moved back so ~5,000 games are scored, and Elo is computed on exactly
the same games from the same dataframe, enabling a paired bootstrap.

Elo is built from the dataframe's own team columns and outcomes, walking forward,
so both models see identical information ordering.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
from load_mz import safe_load  # noqa: E402

MZ = r"C:\Users\Admin\AppData\Local\Temp\mz"
EPS = 1e-15
TEST_N = 5000


def ll_vec(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    import xgboost as xgb

    df = safe_load(MZ + r"\mlb_data\dataframe.pkl")
    df = df.sort_values(by="date").reset_index(drop=True)
    print(f"rows={len(df):,}  cols={df.shape[1]:,}")

    # keep raw identities for the Elo baseline before they get encoded
    home_ab = df["home_team_abbr"].astype(str).to_numpy()
    away_ab = df["away_team_abbr"].astype(str).to_numpy()
    dates = pd.to_datetime(df["date"], unit="s")
    seasons = dates.dt.year.to_numpy()
    y_all = df["home_team_win"].to_numpy(float)

    # --- Elo, walk-forward over the same ordering ---
    r = defaultdict(lambda: 1500.0)
    p_elo = np.zeros(len(df))
    K, HFA, REV = 4.0, 24.0, 0.25
    last_season = None
    for i in range(len(df)):
        if last_season is not None and seasons[i] != last_season:
            for t in r:
                r[t] = 1500.0 + (r[t] - 1500.0) * (1 - REV)
        last_season = seasons[i]
        h, a = home_ab[i], away_ab[i]
        e = 1.0 / (1.0 + 10 ** (-((r[h] + HFA - r[a]) / 400.0)))
        p_elo[i] = e
        d = K * (y_all[i] - e)
        r[h] += d
        r[a] -= d

    # --- XGBoost exactly as the repo builds it, larger test window ---
    d2 = df.copy()
    for c in [c for c in d2.columns if "object" in str(d2[c].dtype)]:
        d2[c] = d2.groupby(c, group_keys=False)["home_team_win"].apply(
            lambda s: s.rolling(180).mean()).shift(1)
    X = d2.drop(columns=["home_team_win", "game_id"])
    y = d2.home_team_win

    split = len(df) - TEST_N
    val = split - 1500
    params = {"learning_rate": 0.03, "max_depth": 3, "colsample_bytree": .6,
              "min_child_weight": 2.0, "subsample": 0.6, "reg_alpha": 6.08}
    m = xgb.XGBClassifier(**params, n_estimators=2000, early_stopping_rounds=20,
                          eval_metric="logloss")
    m.fit(X[:val], y[:val], eval_set=[(X[val:split], y[val:split])], verbose=False)
    p_xgb = m.predict_proba(X[split:])[:, 1]

    yt = y_all[split:]
    pe = p_elo[split:]
    print(f"train={val:,}  valid={split-val:,}  test={len(yt):,}  best_iter={m.best_iteration}")
    print(f"test window: {dates.iloc[split].date()} .. {dates.iloc[-1].date()}"
          f"   home win rate={yt.mean():.4f}")

    print(f"\n{'model':<34}{'log loss':>10}{'Brier':>9}{'acc':>8}{'pred sd':>9}")
    print("-" * 70)
    rows = [("mitchellz4 XGBoost", p_xgb), ("Elo (k=4, hfa=24)", pe),
            ("constant home rate", np.full_like(yt, yt.mean())),
            ("constant 0.5", np.full_like(yt, 0.5))]
    for name, p in rows:
        print(f"{name:<34}{ll(p, yt):>10.5f}{float(np.mean((p-yt)**2)):>9.5f}"
              f"{float(np.mean((p>0.5)==(yt>0.5))):>8.4f}{float(np.std(p)):>9.4f}")

    d, lo, hi = boot(ll_vec(pe, yt), ll_vec(p_xgb, yt))
    print(f"\nXGBoost minus Elo: {-d:+.5f} nats  (negative = XGBoost better)")
    print(f"  95% paired bootstrap CI on (Elo - XGB): [{lo:+.5f}, {hi:+.5f}]")
    print(f"  significant: {'YES' if (lo > 0 or hi < 0) else 'NO — indistinguishable'}")


if __name__ == "__main__":
    main()
