"""Random forest vs the linear logistic blend on our six rating features.

Same protocol as the GBM test: DEV 2000-2015, TEST 2016-2024 ex-2020, CV-tuned on DEV,
isotonic-calibrated (RF probs are pulled toward 0.5), paired bootstrap on per-game LL.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

F = ["lf", "bpz", "siz", "tsz", "pwz", "brz"]
df = pd.read_csv("data/model_probs.csv")
df = df[df.recbp_p.notna()].reset_index(drop=True)
X = df[F].values; y = df.y.values
dev = (df.season <= 2015).values
test = ((df.season >= 2016) & (df.season != 2020)).values
Xtr, ytr, Xte, yte = X[dev], y[dev], X[test], y[test]
print(f"DEV {dev.sum()}  TEST {test.sum()}")

eps = 1e-9
def LLvec(y, p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

lr = LogisticRegression(C=1e6, max_iter=3000).fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]

grid = {"max_depth": [5, 8, 12], "min_samples_leaf": [50, 200, 500], "max_features": ["sqrt", None]}
rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
gs = GridSearchCV(rf, grid, scoring="neg_log_loss", cv=5, n_jobs=-1)
gs.fit(Xtr, ytr)
p_rf = gs.best_estimator_.predict_proba(Xte)[:, 1]

cal = CalibratedClassifierCV(gs.best_estimator_, method="isotonic", cv=5)
cal.fit(Xtr, ytr)
p_rfc = cal.predict_proba(Xte)[:, 1]

print(f"\nbest RF params: {gs.best_params_}")
print(f"\n{'model':<28}{'TEST log loss':>14}")
for name, p in [("logistic blend (baseline)", p_lr), ("RF (tuned)", p_rf), ("RF + isotonic calib", p_rfc)]:
    print(f"  {name:<26}{LLvec(yte, p).mean():>14.5f}")

rng = np.random.default_rng(7)
for tag, p_alt in [("RF tuned", p_rf), ("RF calib", p_rfc)]:
    d = LLvec(yte, p_lr) - LLvec(yte, p_alt); n = len(d)
    bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "RF SIG better" if lo > 0 else ("RF SIG worse" if hi < 0 else "n.s.")
    print(f"\n{tag} vs logistic: delta LL {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}] -> {sig}")
