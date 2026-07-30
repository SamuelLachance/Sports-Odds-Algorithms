"""The 'most complex ever' soft-voting system on our six rating features.

~21 diverse supervised probabilistic classifiers (the full applicable slice of the
taxonomy: linear family, Bayesian, NB, kNN, SVM-RBF, trees, LDA/QDA, GP, RF/ExtraTrees/
AdaBoost/HistGBM/XGB/LGBM/CatBoost, MLP), each held-out isotonic-calibrated, combined by
(a) equal-weight soft vote, (b) inverse-loss weighted vote, (c) logistic STACKING.

Protocol: DEV 2000-2015 split into DEV_fit (fit base learners) + DEV_blend (held out ->
calibrate + weight + fit stacker). Evaluate on TEST 2016-2024 ex-2020. Baseline to beat =
plain logistic on full DEV (0.67333). Slow kernels (GP, SVM-RBF) are subsampled.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier

F = ["lf", "bpz", "siz", "tsz", "pwz", "brz"]
df = pd.read_csv("data/model_probs.csv"); df = df[df.recbp_p.notna()].reset_index(drop=True)
X = df[F].values.astype(float); y = df.y.values
dev = (df.season <= 2015).values
test = ((df.season >= 2016) & (df.season != 2020)).values
Xdev, ydev = X[dev], y[dev]; Xte, yte = X[test], y[test]
rng = np.random.default_rng(0)
perm = rng.permutation(len(Xdev)); cut = int(0.75 * len(perm))
fit_i, bl_i = perm[:cut], perm[cut:]
Xf, yf = Xdev[fit_i], ydev[fit_i]; Xb, yb = Xdev[bl_i], ydev[bl_i]
sc = StandardScaler().fit(Xf)
Xf_s, Xb_s, Xte_s = sc.transform(Xf), sc.transform(Xb), sc.transform(Xte)
eps = 1e-6
def LL(y, p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
def LLv(y, p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

base_ll = LL(yte, LogisticRegression(C=1e6, max_iter=3000).fit(Xdev, ydev).predict_proba(Xte)[:, 1])
print(f"BASELINE plain logistic (full DEV): TEST LL {base_ll:.5f}\n")

def sub(n):
    idx = rng.permutation(len(Xf))[:n]; return Xf_s[idx], yf[idx]

def prob(est, scaled=True, subn=None):
    """fit on DEV_fit, return (blend_probs, test_probs) raw."""
    if subn:
        xf, yff = sub(subn)
    else:
        xf, yff = (Xf_s if scaled else Xf), yf
    est.fit(xf, yff)
    XB = Xb_s if scaled else Xb; XT = Xte_s if scaled else Xte
    if hasattr(est, "predict_proba"):
        return est.predict_proba(XB)[:, 1], est.predict_proba(XT)[:, 1]
    p = est.predict(XB); pt = est.predict(XT)          # BayesianRidge regression
    return np.clip(p, 0, 1), np.clip(pt, 0, 1)

MODELS = [
    ("logistic",        lambda: LogisticRegression(C=1e6, max_iter=3000), True, None),
    ("ridge-logit",     lambda: LogisticRegression(C=1.0, max_iter=3000), True, None),
    ("lasso-logit",     lambda: LogisticRegression(C=1.0, penalty="l1", solver="liblinear"), True, None),
    ("elasticnet-logit",lambda: LogisticRegression(C=1.0, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=3000), True, None),
    ("poly2-logit",     lambda: make_pipeline(PolynomialFeatures(2), LogisticRegression(C=1.0, max_iter=5000)), True, None),
    ("bayesian-ridge",  lambda: BayesianRidge(), True, None),
    ("gaussian-nb",     lambda: GaussianNB(), True, None),
    ("lda",             lambda: LinearDiscriminantAnalysis(), True, None),
    ("qda",             lambda: QuadraticDiscriminantAnalysis(), True, None),
    ("knn",             lambda: KNeighborsClassifier(n_neighbors=300), True, None),
    ("svm-rbf",         lambda: SVC(C=1.0, gamma="scale", probability=True), True, 8000),
    ("gaussian-process",lambda: GaussianProcessClassifier(), True, 2000),
    ("decision-tree",   lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=200), True, None),
    ("random-forest",   lambda: RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=200, n_jobs=-1), True, None),
    ("extra-trees",     lambda: ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=200, n_jobs=-1), True, None),
    ("adaboost",        lambda: AdaBoostClassifier(n_estimators=200), True, None),
    ("hist-gbm",        lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=400, early_stopping=True), True, None),
    ("xgboost",         lambda: xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8, eval_metric="logloss", verbosity=0), True, None),
    ("lightgbm",        lambda: lgb.LGBMClassifier(num_leaves=15, n_estimators=300, learning_rate=0.05, subsample=0.8, verbose=-1), True, None),
    ("catboost",        lambda: CatBoostClassifier(depth=3, iterations=300, learning_rate=0.05, verbose=0), True, None),
    ("mlp",             lambda: MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=500, early_stopping=True), True, None),
]

blend_raw, test_raw, names, weights = [], [], [], []
print(f"{'model':<20}{'cal TEST LL':>12}")
for name, ctor, scaled, subn in MODELS:
    try:
        pb, pt = prob(ctor(), scaled, subn)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=eps, y_max=1 - eps).fit(pb, yb)
        cal_te = iso.transform(pt)
        bll = LL(yb, iso.transform(pb))
        blend_raw.append(pb); test_raw.append(pt); names.append(name)
        weights.append(1.0 / bll); test_cal = cal_te
        MODELS_cal = test_cal
        # stash calibrated test prob on the model tuple via parallel list
        prob._cal = getattr(prob, "_cal", []); prob._cal.append(cal_te)
        print(f"{name:<20}{LL(yte, cal_te):>12.5f}{'  (best base)' if LL(yte,cal_te)<base_ll else ''}")
    except Exception as e:
        print(f"{name:<20}   FAILED: {str(e)[:50]}")

cal_test = np.array(prob._cal)                # (n_models, n_test) calibrated
Braw = np.column_stack(blend_raw); Traw = np.column_stack(test_raw)
w = np.array(weights); w /= w.sum()

equal = cal_test.mean(axis=0)
wtd = (cal_test * w[:, None]).sum(axis=0)
stacker = LogisticRegression(C=1.0, max_iter=5000).fit(Braw, yb)
stack = stacker.predict_proba(Traw)[:, 1]

print(f"\n{'='*46}\nBASELINE plain logistic         TEST LL {base_ll:.5f}")
print(f"{'-'*46}")
rng2 = np.random.default_rng(7)
for tag, p in [("equal-weight soft vote", equal), ("inverse-loss weighted vote", wtd), ("logistic STACKING", stack)]:
    d = LLv(yte, np.full_like(p, 0)) * 0  # placeholder
    d = LLv(yte, p_base := LogisticRegression(C=1e6, max_iter=3000).fit(Xdev, ydev).predict_proba(Xte)[:, 1]) - LLv(yte, p)
    n = len(d); bs = d[rng2.integers(0, n, size=(8000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "BEATS logistic" if lo > 0 else ("worse" if hi < 0 else "n.s. (= logistic)")
    print(f"{tag:<30} TEST LL {LL(yte,p):.5f}   delta {d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] -> {sig}")
