"""THE single NHL TEST look for the servable player-value model (S1).

Pre-registered in data/pv_nhl_serve_prereg.json (arms, bar, test_protocol,
amendment_1). S1 = the shipped 4-feature blend + ONE column: the difference in
lineup value of the skaters each team dressed in its PREVIOUS game (walk-forward
player values: finishing, individual creation, assists, faceoffs, penalties,
on-ice xG-against; compose weights frozen at the DEV fit).

Protocol = the shipped NHL TEST protocol (phase0/nhl_features_eval.py main):
blend coefficients fit on DEV regular season 2011-12..2017-18, scored ONCE on
TEST 2018-19..2025-26, identical game mask for both arms, paired bootstrap 10k.
Run once. Whatever it prints gets NHL ledger row 11.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
from nhl_features_eval import build_features  # noqa: E402
from nhl_glicko2_eval import (DEV_END, DEV_WARM_BEFORE, TEST_END, TEST_START,  # noqa: E402
                              llv, load_games, run_elo)

LV = "data/pv_nhl_fix_full_serve_lv_last.csv"
OUT = "data/pv_nhl_serve_test.json"


def main():
    pre = json.load(open("data/pv_nhl_serve_prereg.json"))
    assert "amendment_1" in pre and pre["primary"] == "S1"
    lv_sha = hashlib.sha256(open(LV, "rb").read()).hexdigest()

    rep = json.load(open("data/nhl_glicko2_report.json"))
    games = load_games()
    e_out = run_elo(games, **rep["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    gid = np.array([g["game_id"] for g in games])
    F = build_features(games)
    xg = F["xg_diff"]
    dev = (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    test = (seas >= TEST_START) & (seas <= TEST_END)

    L = pd.read_csv(LV).set_index("gid")
    Lg = L.reindex(gid)
    have = Lg.lv_last_home.notna().to_numpy() & Lg.lv_last_away.notna().to_numpy()
    d_lv = (Lg.lv_last_home.fillna(0.0) - Lg.lv_last_away.fillna(0.0)).to_numpy(float)
    d_lv = np.where(have, d_lv, 0.0)                      # pre-registered: no previous game -> 0

    ok = ~np.isnan(xg)                                    # the shipped blend's own mask
    m_dev, m_test = dev & ok, test & ok
    base = [elogit, F["rest_diff"], F["b2b_home"], F["b2b_away"], np.nan_to_num(xg)]
    X0 = np.column_stack(base)
    X1 = np.column_stack(base + [d_lv])
    c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X0[m_dev], y[m_dev])
    c1 = LogisticRegression(C=1e6, max_iter=5000).fit(X1[m_dev], y[m_dev])
    p0 = c0.predict_proba(X0[m_test])[:, 1]
    p1 = c1.predict_proba(X1[m_test])[:, 1]
    yt = y[m_test]
    l0, l1 = llv(yt, p0), llv(yt, p1)
    dvec = l0 - l1
    rng = np.random.default_rng(20260924)
    bs = dvec[rng.integers(0, len(dvec), size=(10000, len(dvec)))].mean(axis=1)
    lo, hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    st = seas[m_test]
    per = {int(s): round(float(dvec[st == s].mean()), 5) for s in np.unique(st)}
    res = {"n_test": int(m_test.sum()), "n_dev_fit": int(m_dev.sum()),
           "test_sides_without_previous_game": int((~have[m_test]).sum()),
           "baseline_test_ll": round(float(l0.mean()), 5), "s1_test_ll": round(float(l1.mean()), 5),
           "delta": round(float(dvec.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "seasons_better": sum(1 for v in per.values() if v > 0), "per_season": per,
           "coef_lv_last": round(float(c1.coef_[0][-1]), 4),
           "verdict": "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s."),
           "lv_file_sha256": lv_sha}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
