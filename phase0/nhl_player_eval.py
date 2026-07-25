"""Combined player-rating feature test: shift-TrueSkill, RAPM, and their blend
as increments over the shipped NHL model (Elo+rest+b2b+xG), locked split.

Both player-rating methods are decoded into a pre-game team aggregate (home minus
away). We test each alone and together; the blend lets the logistic take whatever
signal each carries. Display follows log-loss: whatever clears the gate ships.
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


def load_feat(path, hcol, acol):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d[int(r["game_id"])] = float(r[hcol]) - float(r[acol])
    return d


def main():
    model = json.load(open("data/nhl_model.json"))
    games = load_games()
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    b = model["blend"]; c = b["coefs"]
    base_z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
              + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
              + c["xg"] * np.nan_to_num(F["xg_diff"]))

    ts = load_feat("data/nhl_shift_features.csv", "home_agg", "away_agg")
    rp = load_feat("data/nhl_rapm_features.csv", "home_rapm", "away_rapm")
    ts_d = np.array([ts.get(g["game_id"], np.nan) for g in games])
    rp_d = np.array([rp.get(g["game_id"], np.nan) for g in games])
    have = ~np.isnan(ts_d) & ~np.isnan(rp_d)
    dev = have & (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    test = have & (seas >= TEST_START) & (seas <= TEST_END)

    def run(name, extra):
        Xb = base_z.reshape(-1, 1)
        X = np.column_stack([base_z] + extra)
        mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[dev], y[dev])
        ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[dev], y[dev])
        p0 = mb.predict_proba(Xb[test])[:, 1]
        p1 = ms.predict_proba(X[test])[:, 1]
        b_ll = float(llv(y[test], p0).mean()); s_ll = float(llv(y[test], p1).mean())
        d = llv(y[test], p0) - llv(y[test], p1)
        rng = np.random.default_rng(7)
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        coefs = [round(float(x), 3) for x in ms.coef_[0][1:]]
        print(f"  {name:<24} base {b_ll:.5f} -> {s_ll:.5f}  delta {b_ll-s_ll:+.5f} "
              f"CI [{lo:+.5f},{hi:+.5f}] {sig}  coefs {coefs}", flush=True)
        return {"name": name, "base": round(b_ll, 5), "with": round(s_ll, 5),
                "delta": round(b_ll - s_ll, 5), "ci": [round(float(lo), 5), round(float(hi), 5)],
                "sig": sig, "coefs": coefs}

    corr = float(np.corrcoef(ts_d[have], rp_d[have])[0, 1])
    print(f"coverage {have.mean():.1%}  TEST n={int(test.sum())}  corr(shiftTS, RAPM)={corr:+.3f}")
    res = {"corr": corr, "n_test": int(test.sum())}
    res["shift_ts"] = run("shift-TS agg", [np.nan_to_num(ts_d)])
    res["rapm"] = run("RAPM agg", [np.nan_to_num(rp_d)])
    res["blend"] = run("shift-TS + RAPM", [np.nan_to_num(ts_d), np.nan_to_num(rp_d)])
    json.dump(res, open("data/nhl_player_eval.json", "w"), indent=1)
    print("wrote data/nhl_player_eval.json")


if __name__ == "__main__":
    main()
