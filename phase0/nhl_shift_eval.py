"""Does the per-shift TrueSkill team aggregate improve the NHL game model?

Tests home_agg - away_agg (pre-game on-ice skater rating differential, from
nhl_shift_trueskill.py) as an increment over the shipped Elo+rest+b2b+xG blend,
on the locked split (DEV-fit / TEST-scored). If it clears the gate it ships and
the player ratings become a real feature; if not, the shift-TS is at most a
(deployment-confounded) display artifact.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import DEV_WARM_BEFORE, DEV_END, TEST_START, TEST_END, llv, load_games, run_elo  # noqa: E402
from nhl_features_eval import build_features  # noqa: E402


def main():
    model = json.load(open("data/nhl_model.json"))
    games = load_games()
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])

    sf = {int(r["game_id"]): (float(r["home_agg"]) - float(r["away_agg"]))
          for r in csv.DictReader(open("data/nhl_shift_features.csv", encoding="utf-8"))}
    shift_diff = np.array([sf.get(g["game_id"], np.nan) for g in games])
    have = ~np.isnan(shift_diff)

    b = model["blend"]; c = b["coefs"]
    base_z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
              + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
              + c["xg"] * np.nan_to_num(F["xg_diff"]))

    dev = have & (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    test = have & (seas >= TEST_START) & (seas <= TEST_END)

    # base model (frozen blend logit) vs base + shift feature (refit both coefs on DEV)
    Xb = base_z.reshape(-1, 1)
    Xs = np.column_stack([base_z, np.nan_to_num(shift_diff)])
    mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[dev], y[dev])
    ms = LogisticRegression(C=1e6, max_iter=2000).fit(Xs[dev], y[dev])
    p0 = mb.predict_proba(Xb[test])[:, 1]
    p1 = ms.predict_proba(Xs[test])[:, 1]
    b_ll = float(llv(y[test], p0).mean()); s_ll = float(llv(y[test], p1).mean())
    d = llv(y[test], p0) - llv(y[test], p1)
    rng = np.random.default_rng(7)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
    print(f"coverage: {have.mean():.1%}  TEST n={int(test.sum())}")
    print(f"  shipped blend         {b_ll:.5f}")
    print(f"  + shift-TS team agg    {s_ll:.5f}  (delta {b_ll-s_ll:+.5f}  "
          f"CI [{lo:+.5f},{hi:+.5f}]  {sig})  coef {ms.coef_[0][1]:+.3f}")
    print(f"  -> {'ADOPT' if s_ll < b_ll - 0.0007 else 'below gate'}")
    json.dump({"base_ll": round(b_ll, 5), "shift_ll": round(s_ll, 5),
               "delta": round(b_ll - s_ll, 5), "ci": [round(float(lo), 5), round(float(hi), 5)],
               "coef": round(float(ms.coef_[0][1]), 4), "sig": sig,
               "n_test": int(test.sum()), "coverage": round(float(have.mean()), 4)},
              open("data/nhl_shift_eval.json", "w"), indent=1)
    print("wrote data/nhl_shift_eval.json")


if __name__ == "__main__":
    main()
