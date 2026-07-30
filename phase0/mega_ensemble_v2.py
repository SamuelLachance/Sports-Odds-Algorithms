"""Fair version: full-DEV cross-validated stacking (the ensemble's best possible shot).

Each base learner gets proper out-of-fold predictions on the FULL DEV set (cross_val_predict,
5-fold) -> no train/blend split penalty. Stacker + calibration + weights all use those OOF
preds. Drops GP and SVM-RBF (the two worst/slowest bases; excluding them only helps the
ensemble). Baseline = plain logistic on full DEV. Locked TEST 2016-2024 ex-2020.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier

F = ["lf", "bpz", "siz", "tsz", "pwz", "brz"]
df = pd.read_csv("data/model_probs.csv"); df = df[df.recbp_p.notna()].reset_index(drop=True)
X = df[F].values.astype(float); y = df.y.values
dev = (df.season <= 2015).values; test = ((df.season >= 2016) & (df.season != 2020)).values
Xdev, ydev, Xte, yte = X[dev], y[dev], X[test], y[test]
eps = 1e-6
def LL(y, p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
def LLv(y, p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

pbase = LogisticRegression(C=1e6, max_iter=3000).fit(Xdev, ydev).predict_proba(Xte)[:, 1]
base_ll = LL(yte, pbase)
print(f"BASELINE plain logistic (full DEV): TEST LL {base_ll:.5f}\n")

def M():
    return [
        ("logistic",   make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=3000))),
        ("ridge",      make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))),
        ("lasso",      make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l1", solver="liblinear"))),
        ("elasticnet", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=3000))),
        ("poly2-logit",make_pipeline(StandardScaler(), PolynomialFeatures(2), LogisticRegression(C=1.0, max_iter=5000))),
        ("bayes-ridge",make_pipeline(StandardScaler(), BayesianRidge())),
        ("gaussian-nb",make_pipeline(StandardScaler(), GaussianNB())),
        ("lda",        LinearDiscriminantAnalysis()),
        ("qda",        QuadraticDiscriminantAnalysis()),
        ("knn",        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=300))),
        ("dtree",      DecisionTreeClassifier(max_depth=4, min_samples_leaf=200)),
        ("rf",         RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=200, n_jobs=-1)),
        ("extratrees", ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=200, n_jobs=-1)),
        ("adaboost",   AdaBoostClassifier(n_estimators=200)),
        ("hist-gbm",   HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=400)),
        ("xgboost",    xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8, eval_metric="logloss", verbosity=0)),
        ("lightgbm",   lgb.LGBMClassifier(num_leaves=15, n_estimators=300, learning_rate=0.05, subsample=0.8, verbose=-1)),
        ("catboost",   CatBoostClassifier(depth=3, iterations=300, learning_rate=0.05, verbose=0)),
        ("mlp",        make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=500, early_stopping=True))),
    ]

def oof_and_test(est):
    if hasattr(est, "predict_proba"):
        oof = cross_val_predict(est, Xdev, ydev, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
        est.fit(Xdev, ydev); te = est.predict_proba(Xte)[:, 1]
    else:
        oof = np.clip(cross_val_predict(est, Xdev, ydev, cv=5, n_jobs=-1), 0, 1)
        est.fit(Xdev, ydev); te = np.clip(est.predict(Xte), 0, 1)
    return oof, te

OOF, TE, names, w = [], [], [], []
print(f"{'model':<14}{'cal TEST LL':>12}")
for name, est in M():
    try:
        oof, te = oof_and_test(est)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=eps, y_max=1 - eps).fit(oof, ydev)
        cte = iso.transform(te)
        OOF.append(oof); TE.append(cte); names.append(name); w.append(1.0 / LL(ydev, iso.transform(oof)))
        print(f"{name:<14}{LL(yte, cte):>12.5f}{'  < logistic' if LL(yte,cte)<base_ll else ''}")
    except Exception as e:
        print(f"{name:<14}   FAILED {str(e)[:45]}")

TEc = np.array(TE); Om = np.column_stack(OOF)
w = np.array(w); w /= w.sum()
equal = TEc.mean(0)
wtd = (TEc * w[:, None]).sum(0)
stack = LogisticRegression(C=1.0, max_iter=5000).fit(Om, ydev).predict_proba(
    np.column_stack([iso.transform(t) if False else t for t in TE]))[:, 1]
# stacker on raw OOF -> apply to raw (uncalibrated) test preds
rawte = []
for name, est in M():
    try:
        _, te = oof_and_test(est); rawte.append(te)
    except Exception:
        pass
stack = LogisticRegression(C=1.0, max_iter=5000).fit(Om, ydev).predict_proba(np.column_stack(rawte))[:, 1]

print(f"\n{'='*50}\nBASELINE plain logistic          TEST LL {base_ll:.5f}\n{'-'*50}")
rng = np.random.default_rng(7)
for tag, p in [("equal-weight soft vote", equal), ("inverse-loss weighted vote", wtd), ("logistic STACKING", stack)]:
    d = LLv(yte, pbase) - LLv(yte, p); n = len(d)
    bs = d[rng.integers(0, n, size=(8000, n))].mean(axis=1); lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "BEATS logistic" if lo > 0 else ("worse, sig" if hi < 0 else "n.s. (ties logistic)")
    print(f"{tag:<28} TEST LL {LL(yte,p):.5f}  delta {d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] -> {sig}")
