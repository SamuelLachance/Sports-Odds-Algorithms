"""Different TRAINING METHOD, same features: does gradient boosting beat the linear
logistic blend on our six rating features?

Features = [lf, bpz, siz, tsz, pwz, brz] (the inputs behind recbp_p). If GBM finds
nonlinear structure / interactions the linear blend misses, it wins on TEST log loss.
GBM is CV-tuned on DEV (given its best shot) and calibration is checked. Locked split:
DEV 2000-2015, TEST 2016-2024 ex-2020. Paired bootstrap on the per-game LL delta.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

F = ["lf", "bpz", "siz", "tsz", "pwz", "brz"]
df = pd.read_csv("data/model_probs.csv")
df = df[df.recbp_p.notna()].reset_index(drop=True)
X = df[F].values; y = df.y.values
dev = (df.season <= 2015).values
test = ((df.season >= 2016) & (df.season != 2020)).values
Xtr, ytr, Xte, yte = X[dev], y[dev], X[test], y[test]
print(f"DEV {dev.sum()}  TEST {test.sum()}  features {F}")

eps = 1e-9
def LLvec(y, p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# --- baseline: linear logistic blend ---
lr = LogisticRegression(C=1e6, max_iter=3000).fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]

# --- shipped production blend, for reference ---
p_ship = df.loc[test, "recbp_p"].values

# --- GBM: CV-tuned on DEV ---
grid = {
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [2, 3],
    "l2_regularization": [0.0, 1.0],
    "max_leaf_nodes": [15, 31],
}
gb = HistGradientBoostingClassifier(max_iter=600, early_stopping=True,
                                    validation_fraction=0.15, random_state=0)
gs = GridSearchCV(gb, grid, scoring="neg_log_loss", cv=5, n_jobs=-1)
gs.fit(Xtr, ytr)
p_gb = gs.best_estimator_.predict_proba(Xte)[:, 1]

# --- GBM + isotonic calibration (fair log-loss shot) ---
cal = CalibratedClassifierCV(gs.best_estimator_, method="isotonic", cv=5)
cal.fit(Xtr, ytr)
p_gbc = cal.predict_proba(Xte)[:, 1]

print(f"\nbest GBM params: {gs.best_params_}")
print(f"\n{'model':<28}{'TEST log loss':>14}")
for name, p in [("logistic blend (baseline)", p_lr), ("shipped recbp_p (ref)", p_ship),
                ("GBM (tuned)", p_gb), ("GBM + isotonic calib", p_gbc)]:
    print(f"  {name:<26}{LLvec(yte, p).mean():>14.5f}")

# --- paired bootstrap: logistic - GBM (positive => GBM better) ---
rng = np.random.default_rng(7)
for tag, p_alt in [("GBM tuned", p_gb), ("GBM calib", p_gbc)]:
    d = LLvec(yte, p_lr) - LLvec(yte, p_alt)
    n = len(d)
    bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "GBM SIG better" if lo > 0 else ("GBM SIG worse" if hi < 0 else "n.s.")
    print(f"\n{tag} vs logistic: delta LL {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}] -> {sig}")
