"""Backtest mitchellz4/MLB-Model — replicate its own split and score it properly.

The repo reports Brier, accuracy and AUC but never log loss, and evaluates on the
last 1,002 games. Both are measured here, alongside the baselines the repo omits.

Note on the target encoding in xgb_model.py:

    df[x] = df.groupby(x, group_keys=False)['home_team_win']
              .apply(lambda s: s.rolling(180).mean()).shift(1)

`.shift(1)` is applied to the RECOMBINED series, not within each group, so a row
receives the rolling mean belonging to the previous ROW of the dataframe — a
different game, usually a different team. The pickle is already date-sorted, so
this is not future leakage; it is a misalignment that feeds the model another
team's number. Both variants are evaluated below so the cost is visible.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
from load_mz import safe_load  # noqa: E402

MZ = r"C:\Users\Admin\AppData\Local\Temp\mz"
EPS = 1e-15


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def encode(df: pd.DataFrame, within_group_shift: bool) -> pd.DataFrame:
    d = df.copy()
    cols = [c for c in d.columns if "object" in str(d[c].dtype)]
    for c in cols:
        g = d.groupby(c, group_keys=False)["home_team_win"]
        if within_group_shift:
            # correct: exclude the current game, per team
            d[c] = g.apply(lambda s: s.shift(1).rolling(180).mean())
        else:
            # exactly as the repo does it
            d[c] = g.apply(lambda s: s.rolling(180).mean()).shift(1)
    return d


def run(df_raw: pd.DataFrame, within_group_shift: bool, label: str):
    import xgboost as xgb

    df = encode(df_raw, within_group_shift)
    df = df.sort_values(by="date").copy().reset_index(drop=True)
    X = df.drop(columns=["home_team_win", "game_id"])
    y = df.home_team_win

    # the repo's exact split
    X_train, y_train = X[:-7389], y[:-7389]
    X_valid, y_valid = X[-7389:-2000], y[-7389:-2000]
    X_test, y_test = X[-1002:], y[-1002:]

    params = {"learning_rate": 0.03, "max_depth": 3, "colsample_bytree": .6,
              "min_child_weight": 2.0, "subsample": 0.6, "reg_alpha": 6.08}
    m = xgb.XGBClassifier(**params, n_estimators=1000, early_stopping_rounds=10,
                          eval_metric="logloss")
    m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    p = m.predict_proba(X_test)[:, 1]
    yt = y_test.to_numpy(float)
    print(f"\n--- {label} ---")
    print(f"  train={len(y_train):,}  valid={len(y_valid):,}  test={len(yt):,}"
          f"  best_iter={m.best_iteration}")
    print(f"  LOG LOSS         {ll(p, yt):.5f}")
    print(f"  Brier            {float(np.mean((p-yt)**2)):.5f}")
    print(f"  accuracy         {float(np.mean((p>0.5)==(yt>0.5))):.4f}")
    print(f"  pred sd          {p.std():.4f}   range [{p.min():.3f}, {p.max():.3f}]")
    return p, yt


def main():
    df = safe_load(MZ + r"\mlb_data\dataframe.pkl")
    print(f"dataframe: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print(f"features per training row: {df.shape[1]-2:,}  "
          f"(ratio {(df.shape[1]-2)/df.shape[0]:.3f} features per sample)")

    p_repo, yt = run(df, within_group_shift=False, label="AS THE REPO DOES IT")
    p_fix, _ = run(df, within_group_shift=True, label="target encoding FIXED (shift within group)")

    home_rate = float(yt.mean())
    print("\n=== baselines on the same 1,002 test games ===")
    print(f"  home win rate      {home_rate:.4f}")
    print(f"  constant home rate {ll(np.full_like(yt, home_rate), yt):.5f}")
    print(f"  constant 0.5       {ll(np.full_like(yt, 0.5), yt):.5f}")

    n = len(yt)
    se = float(np.std((p_repo - yt) ** 2)) / np.sqrt(n)
    print(f"\n  n={n}: 95% CI half-width on an absolute log loss is ~"
          f"{1.96*np.std(-(yt*np.log(np.clip(p_repo,EPS,1-EPS))+(1-yt)*np.log(np.clip(1-p_repo,EPS,1-EPS))))/np.sqrt(n):.5f}")
    print("  -> at this sample size the test cannot distinguish models "
          "separated by less than ~0.01 nats.")


if __name__ == "__main__":
    main()
