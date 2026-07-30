"""Flepp-style luck-gap test on MLB: does the market over-rate teams whose
RECORD outruns their Pythagorean process — and does conditioning value bets
on it improve the pilot?

  luck(team)   = season-to-date win% − pythag win% (RS^1.83/(RS^1.83+RA^1.83)),
                 as-of (pre-game), EB-damped early season (prior 20 games at 0)
  gap_diff     = luck(home) − luck(away)

Flepp's gate: logistic y ~ [logit(p_market), gap_diff]. A NEGATIVE coefficient
= market over-rates the luckier team (their finding). Tested at both the open
and the close (2010-2021, n≈27k). Overlay: the 15%/20% EV bet sets split by
whether OUR side is the relatively unlucky team (fading luck) — ROI/CLV per arm.
Odds evaluation-layer only.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from ev_gate_audit import exp_vs_real, load, roi

F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R = 0, 3, 6, 9, 10
PYTH_EXP = 1.83
PRIOR_G = 20.0


def build_luck(years):
    """(date, home, away) -> (luck_home, luck_away), as-of, leak-safe."""
    st = defaultdict(lambda: [0, 0, 0.0, 0.0])      # team -> [W, L, RS, RA]
    out = {}
    prev_season = None
    rows = []
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        yr = int(path[-8:-4])
        if yr not in years:
            continue
        with open(path, newline="", encoding="latin-1") as fh:
            for r in csv.reader(fh):
                if len(r) < 11:
                    continue
                try:
                    vr, hr = int(r[F_VIS_R]), int(r[F_HOME_R])
                except ValueError:
                    continue
                rows.append((r[F_DATE], yr, r[F_VIS], r[F_HOME], vr, hr))
    rows.sort()
    for date, yr, vis, home, vr, hr in rows:
        if prev_season is not None and yr != prev_season:
            st.clear()
        prev_season = yr

        def luck(team):
            w, l, rs, ra = st[team]
            g = w + l
            if g == 0 or rs + ra == 0:
                return 0.0
            wp = w / g
            pyth = rs ** PYTH_EXP / (rs ** PYTH_EXP + ra ** PYTH_EXP)
            return (wp - pyth) * g / (g + PRIOR_G)   # EB-damped
        d8 = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        out[(d8, home, vis)] = (luck(home), luck(vis))
        hw = 1 if hr > vr else 0
        st[home][0] += hw; st[home][1] += 1 - hw
        st[home][2] += hr; st[home][3] += vr
        st[vis][0] += 1 - hw; st[vis][1] += hw
        st[vis][2] += vr; st[vis][3] += hr
    return out


def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    years = range(2010, 2022)
    rows = load(years=years)
    luck = build_luck(set(years))
    miss = 0
    for g in rows:
        lk = luck.get((g["date"], "", ""))
    # join needs team codes — reload probs keys
    mp_keys = {}
    for r in csv.DictReader(open("data/model_probs.csv")):
        if int(r["season"]) in years:
            mp_keys[(r["date"], r["home"], r["away"])] = True
    # rebuild rows WITH teams (load() drops them) — re-run the join inline
    odds = defaultdict(list)
    for r in csv.DictReader(open("data/odds_mlb.csv")):
        odds[(r["date"], r["home"], r["away"])].append(r)
    used = defaultdict(int)
    J = []
    mp = [r for r in csv.DictReader(open("data/model_probs.csv"))
          if int(r["season"]) in years]
    for r in sorted(mp, key=lambda r: (r["date"], r["home"], r["away"], r["dh"])):
        k = (r["date"], r["home"], r["away"])
        lst = odds.get(k, [])
        if used[k] >= len(lst) or r["full_p"] == "":
            continue
        o = lst[used[k]]
        used[k] += 1
        lk = luck.get(k)
        if lk is None:
            miss += 1
            continue
        ho, ao = float(o["home_open"]), float(o["away_open"])
        hc, ac = float(o["home_close"]), float(o["away_close"])
        p_open = (1 / ho) / (1 / ho + 1 / ao)
        p_close = (1 / hc) / (1 / hc + 1 / ac)
        J.append({"date": r["date"], "season": int(r["season"]),
                  "p": float(r["full_p"]), "y": int(r["y"]),
                  "ho": ho, "ao": ao, "hc": hc, "ac": ac,
                  "p_open": p_open, "p_close": p_close,
                  "gap": lk[0] - lk[1]})
    print(f"joined {len(J):,} games ({miss} without luck state)")
    y = np.array([g["y"] for g in J])
    gap = np.array([g["gap"] for g in J])
    print(f"gap_diff: std {gap.std():.4f}, p95 |gap| {np.percentile(np.abs(gap),95):.3f}")

    res = {}
    for base_name in ("p_open", "p_close"):
        pb = np.array([g[base_name] for g in J])
        X = np.column_stack([logit(pb), gap])
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        coef = float(m.coef_[0][1])
        rng = np.random.default_rng(7)
        bs = []
        for _ in range(400):
            s_ = rng.choice(len(y), size=len(y), replace=True)
            bs.append(float(LogisticRegression(C=1e6, max_iter=500)
                            .fit(X[s_], y[s_]).coef_[0][1]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  y ~ logit({base_name}) + luck_gap:  coef {coef:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {sig}  "
              f"({'market OVER-rates lucky teams' if coef < 0 else 'no over-rating'})")
        res[base_name] = {"coef": round(coef, 4),
                          "ci": [round(float(lo), 4), round(float(hi), 4)], "sig": sig}

    # ---- overlay on the EV bet sets: fading luck vs riding luck ----
    ev_rows = []
    for g in J:
        ev_rows.append({"date": g["date"], "season": g["season"], "p": g["p"],
                        "ho": g["ho"], "ao": g["ao"], "hc": g["hc"], "ac": g["ac"],
                        "y": g["y"], "gap": g["gap"]})
    for gate in (0.15, 0.20):
        bets = []
        for g in ev_rows:
            p = g["p"]
            cands = []
            ev_h = p * (g["ho"] - 1) - (1 - p)
            ev_a = (1 - p) * (g["ao"] - 1) - p
            if ev_h > gate:
                cands.append((ev_h, g["ho"], g["hc"], p, g["y"] == 1, -g["gap"]))
            if ev_a > gate:
                cands.append((ev_a, g["ao"], g["ac"], 1 - p, g["y"] == 0, g["gap"]))
            if cands:
                ev, o_o, o_c, p_s, win, opp_luck = max(cands)
                # opp_luck > 0 => the OPPONENT of our side has been luckier =>
                # we are fading luck (betting the unlucky side)
                bets.append({"p": p_s, "win": win, "odds": o_o,
                             "clv": (1 / o_o) < (1 / o_c) - 1e-12,
                             "fade": opp_luck > 0.01, "ride": opp_luck < -0.01,
                             "season": g["season"]})
        for arm_name, key in (("fade-luck", "fade"), ("ride-luck", "ride")):
            arm = [b for b in bets if b[key]]
            n, e, r_, z = exp_vs_real(arm)
            if n:
                print(f"  gate {gate:.0%} {arm_name:<10} n={n:<4} ROI {roi(arm):+7.2%}  "
                      f"CLV+ {np.mean([b['clv'] for b in arm]):.1%}  z {z:+.2f}")
                res[f"gate{int(gate*100)}_{key}"] = {
                    "n": n, "roi": round(roi(arm), 4),
                    "clv_pos": round(float(np.mean([b["clv"] for b in arm])), 3),
                    "z": round(z, 2)}
    json.dump(res, open("data/luck_gap_eval.json", "w"), indent=1)
    print("wrote data/luck_gap_eval.json")


if __name__ == "__main__":
    main()
