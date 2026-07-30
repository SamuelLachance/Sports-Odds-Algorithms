"""Tune the rating system FOR the model: the never-searched knobs + structural variants.

Grid: rating decay x EB prior (full walks) x clamp (cheap) — hand-set day one, never
tuned despite roster_quality being a SIG feature. Variants per walk:
  net    current form (one net sum)
  split  offense-sum and defense-sum as TWO features
  posw   player share scaled by WOWY position value (QB 1.57 ... OL 0.15 pts/game)
DEV CV selects across everything; winner gets ONE TEST compare; keep if better.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # noqa: S102
exec("#" + full_src.split("# pass 1")[1].split("# ---------------- screens")[0])  # noqa: S102

PRIMARY = {"QB": "dak", "SK": "eff", "DL": "rush", "LB": "rush", "DB": "ball"}
HAND = {"QB": [("dak", 0.50), ("negsk", 0.25), ("repa", 0.25)],
        "SK": [("eff", 0.40), ("yacr", 0.15), ("wopr", 0.45)],
        "DL": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
        "LB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
        "DB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)]}
POSW = {"QB": 1.57, "RB": 1.13, "FB": 1.13, "HB": 1.13, "WR": 0.97, "TE": 0.21}
POSW_D = {"DL": 0.66, "LB": 1.22, "DB": 0.71}
OLW = 0.15

def posw_of(pos):
    if pos in OLPOS:
        return OLW
    b = ("DL" if pos in ("DE", "DT", "NT", "EDGE") else
         "LB" if pos in ("LB", "ILB", "OLB", "MLB") else
         "DB" if pos in ("CB", "S", "FS", "SS", "DB") else None)
    if b:
        return POSW_D[b]
    return POSW.get(pos, 0.5)


def make_rate(prior):
    def rate(pid):
        st = STATE.get(pid)
        if st is None:
            return None
        b = st.get("bucket")
        if b not in Z:
            return None
        if st.get(PRIMARY[b] + "_w", 0.0) < 5.0:
            return None
        tot, any_ = 0.0, False
        for k, wt in HAND[b]:
            w = st.get(k + "_w", 0.0)
            if w <= 1 or k not in Z[b]:
                continue
            mu, sd = Z[b][k]
            v = (st.get(k, 0.0) + prior * mu) / (w + prior)
            tot += wt * (v - mu) / sd
            any_ = True
        return tot if any_ else None
    return rate


def walk(decay_rating, prior):
    globals()["DECAY"] = decay_rating          # used by upd() inside apply_week
    rate = make_rate(prior)
    STATE.clear()
    share2, pos2, team2 = {}, {}, {}
    roster2 = defaultdict(set)
    done2 = set()
    # per-game raw sums (clamped later cheaply at 3 clamp levels x variants)
    out = {c: {"net": np.zeros(len(games)), "off_h": np.zeros(len(games)),
               "def_h": np.zeros(len(games)), "off_a": np.zeros(len(games)),
               "def_a": np.zeros(len(games)), "posw": np.zeros(len(games))}
           for c in (1.5, 2.0, 3.0)}

    for i, g in enumerate(games):
        s, w = g["season"], g["week"]
        if s >= 2013:
            for side, team in (("h", g["home"]), ("a", g["away"])):
                tbl = snaps.get((g["gid"], team))
                if tbl is None:
                    continue
                for pid in tbl:
                    st = share2.get(pid)
                    if not st or st[1] <= 0:
                        continue
                    sh = st[0] / st[1]
                    g_ = pfr2gsis.get(pid)
                    r_ = rate(g_) if g_ else None
                    if r_ is None:
                        continue
                    p = pos2.get(pid, "")
                    isdef = p in DEFPOS
                    for c in out:
                        v = sh * max(-c, min(c, r_))
                        key = ("def_" if isdef else "off_") + side
                        out[c][key][i] += v
                        out[c]["posw"][i] += (v * posw_of(p)) * (1 if side == "h" else -1)
                        out[c]["net"][i] += v * (1 if side == "h" else -1)
        for team in (g["home"], g["away"]):
            for pid in OFF_TW.get((team, s, w), []) + DEF_TW.get((team, s, w), []):
                if (pid, s, w) in done2:
                    continue
                done2.add((pid, s, w))
                apply_week(pid, s, w)
        for team in (g["home"], g["away"]):
            tbl = snaps.get((g["gid"], team))
            if not tbl:
                continue
            for pid, (pos, op, dp) in tbl.items():
                pct = dp if pos in DEFPOS else op
                st = share2.setdefault(pid, [0.0, 0.0])
                st[0] = snapP["decay"] * st[0] + pct
                st[1] = snapP["decay"] * st[1] + 1.0
                pos2[pid] = pos
                if team2.get(pid) != team:
                    if team2.get(pid) in roster2:
                        roster2[team2.get(pid)].discard(pid)
                    team2[pid] = team
                    roster2[team].add(pid)
    return out


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=5000).fit(X[fitm], y[fitm])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()


BASE11 = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                          f_ol, f_def, f_sk])
X_cur = np.column_stack([BASE11, f_rq])
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}\n")

best = None
for dr in (0.85, 0.90, 0.95):
    for pr in (3.0, 6.0, 12.0):
        out = walk(dr, pr)
        for c, d_ in out.items():
            cands = {
                "net": np.column_stack([BASE11, d_["net"]]),
                "split": np.column_stack([BASE11, d_["off_h"] - d_["off_a"],
                                          d_["def_h"] - d_["def_a"]]),
                "posw": np.column_stack([BASE11, d_["posw"]]),
            }
            for name, X in cands.items():
                s = cv_ll(X)
                if best is None or s < best[0]:
                    best = (s, dr, pr, c, name, X)
        print(f"  decay {dr} prior {pr}: best so far {best[0]:.5f} "
              f"({best[0]-base_cv:+.5f}, {best[4]} c={best[3]})", flush=True)

s, dr, pr, c, name, Xn = best
print(f"\nDEV winner: {name} decay={dr} prior={pr} clamp={c}  CV {s:.5f} ({s-base_cv:+.5f})")
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(Xn[fitm], y[fitm])
p0 = c0.predict_proba(X_cur[test])[:, 1]
p1 = c1.predict_proba(Xn[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"main TEST: current {llv(yt, p0).mean():.5f} -> {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"winner": {"variant": name, "decay": dr, "prior": pr, "clamp": c},
           "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_rq_tune.json", "w"), indent=1)
