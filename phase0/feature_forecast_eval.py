"""Can forecasting a rating FORWARD (LTSF-style) beat just using its current value?

Our game features are team ratings = time series. For a future-board game we must use a
STALE rating (today's, frozen until game day). This tests two things on a strong proxy
rating (538 team Elo, mlb_elo.csv, same family as ours):

  1. STALENESS COST: how much TEST log loss degrades if the rating is k games old.
     (This is the ceiling any feature-forecaster could ever recover.)
  2. PERSISTENCE vs MOMENTUM FORECAST: does linearly extrapolating the recent trend
     beat just holding the stale value? If not, the rating is a martingale -> forecasting
     forward gains nothing, which is exactly what LTSF would try to do.

Locked split: fit logistic on 2010-2015, evaluate 2016-2024 ex-2020.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/mlb_elo.csv", usecols=["date", "season", "team1", "team2",
                                              "elo1_pre", "elo2_pre", "score1", "score2"])
df = df[(df.season >= 2010) & (df.season <= 2024)].dropna(subset=["elo1_pre", "elo2_pre", "score1", "score2"])
df = df.sort_values("date").reset_index(drop=True)
df["gid"] = np.arange(len(df))
df["y"] = (df.score1 > df.score2).astype(int)     # team1 = home
print(f"{len(df)} games, home win rate {df.y.mean():.3f}")

# long format: one row per team appearance, to build per-team Elo trajectories
h = df[["gid", "date", "season", "team1", "elo1_pre"]].rename(columns={"team1": "team", "elo1_pre": "elo"})
a = df[["gid", "date", "season", "team2", "elo2_pre"]].rename(columns={"team2": "team", "elo2_pre": "elo"})
h["side"] = "H"; a["side"] = "A"
lg = pd.concat([h, a]).sort_values(["team", "date"]).reset_index(drop=True)

eps = 1e-9
def LL(y, p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

def diff_for(colname):
    """rejoin a per-team-appearance rating column into home-minus-away per game."""
    hh = lg[lg.side == "H"].set_index("gid")[colname]
    aa = lg[lg.side == "A"].set_index("gid")[colname]
    d = (hh - aa).reindex(df.gid)
    return d.values

def fit_eval(feat):
    m = ~np.isnan(feat)
    X = feat.reshape(-1, 1); y = df.y.values; dev = (df.season <= 2015).values & m; test = (df.season >= 2016).values & (df.season != 2020).values & m
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X[dev], y[dev])
    p = lr.predict_proba(X[test])[:, 1]
    return LL(y[test], p), int(test.sum())

print("\n(1) STALENESS COST -- use each team's Elo from k games earlier:")
print(f"  {'k games stale':>14}  {'test LL':>9}  {'vs fresh':>9}")
base_ll = None
for k in [0, 1, 3, 5, 10, 20, 40]:
    lg["elo_k"] = lg.groupby("team")["elo"].shift(k)
    ll, n = fit_eval(diff_for("elo_k"))
    if k == 0: base_ll = ll
    print(f"  {k:>14}  {ll:.5f}  {ll-base_ll:+.5f}")

print("\n(2) PERSISTENCE vs MOMENTUM FORECAST at k=10 games stale:")
# persistence: hold value from 10 games ago
lg["persist"] = lg.groupby("team")["elo"].shift(10)
# momentum forecast: extrapolate the trend over games [t-20 .. t-10] forward 10 games
lg["e10"] = lg.groupby("team")["elo"].shift(10)
lg["e20"] = lg.groupby("team")["elo"].shift(20)
lg["momentum"] = lg["e10"] + (lg["e10"] - lg["e20"])          # linear extrapolation forward 10
ll_p, n = fit_eval(diff_for("persist"))
ll_m, _ = fit_eval(diff_for("momentum"))
ll_fresh, _ = fit_eval(diff_for("elo")) if False else (base_ll, 0)
print(f"  fresh (as-of, unattainable for future games)  LL {base_ll:.5f}")
print(f"  persistence (hold 10-games-ago value)         LL {ll_p:.5f}")
print(f"  momentum forecast (extrapolate trend)         LL {ll_m:.5f}   ({'WORSE' if ll_m>ll_p else 'better'} than persistence by {ll_m-ll_p:+.5f})")
