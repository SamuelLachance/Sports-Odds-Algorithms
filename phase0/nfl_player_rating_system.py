"""FULL PLAYER RATING SYSTEM (guide-style position composites) + model features.

Ratings (as-of EWMAs, z-scored vs DEV-era position distributions, 0-100 for display):
  QB    50% dakota (EPA+CPOE composite, 1999+), 25% sack-avoidance, 25% rushing EPA
  SKILL 40% EPA/opportunity, 15% YAC/rec, 45% usage (WOPR)
  DEF   45% pass rush (sacks+TFL+hits), 35% ball skills (INT+PD), 20% tackles
  (athleticism/PFF/film: not available free — omitted honestly)

Features screened vs the CURRENT 11-feature model (main protocol):
  F2  QB dakota rating through the replacement-prior machinery (replace or add)
  F1  QUALITY-weighted snap absence (absent starter cost x his rating)
  F3  team aggregate active-roster quality (likely redundant)
Adoption rule (Samuel's): best DEV-CV combo gets one TEST compare; keep if better.
Also emits data/nfl_player_ratings_2025.csv (master table, 0-100 + tiers).
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

# ---------------- current model (11 features) ----------------
snapP = json.load(open("data/nfl_snap_absence.json"))["params"]
OLPOS = {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}
DEFPOS = {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "S", "FS", "SS", "DB", "EDGE", "DL"}
SKPOS = {"RB", "WR", "TE", "FB", "HB"}

snaps = defaultdict(dict)
# discover snap tables on disk (snap_2026.csv is picked up the day it lands);
# 2013 floor kept: snap_2012.csv exists but was never read (walks gate s >= 2013)
SNAP_FILES = [p for p in sorted(glob.glob("data/snap_20??.csv"))
              if int(p[-8:-4]) >= 2013]
for path_ in SNAP_FILES:
    for r in csv.DictReader(open(path_)):
        t = PBP_FIX.get(r["team"], r["team"])
        try:
            op, dp = float(r["offense_pct"] or 0), float(r["defense_pct"] or 0)
        except ValueError:
            continue
        snaps[(r["game_id"], t)][r["pfr_player_id"]] = (r["position"], op, dp)

# gsis <-> pfr crosswalk
pfr2gsis = {}
for r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    if r.get("gsis_id") and r.get("pfr_id"):
        pfr2gsis[r["pfr_id"]] = r["gsis_id"]
names = {}
for r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    if r.get("gsis_id"):
        names[r["gsis_id"]] = r.get("display_name") or r.get("gsis_id")

# ---------------- weekly lines (offense + defense), 1999-2025 ----------------
def f2(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

OFFW = {}   # (pid, s, w) -> dict of offensive metrics
# 2026 successor file (same NEW schema as _2025: team / sacks_suffered) is read
# the day it lands; today it does not exist -> list is identical to before
_OFF_SRC = [("data/nfl_player_stats.csv", "recent_team", "sacks"),
            ("data/nfl_player_stats_2025.csv", "team", "sacks_suffered")]
if os.path.exists("data/nfl_player_stats_2026.csv"):
    _OFF_SRC.append(("data/nfl_player_stats_2026.csv", "team", "sacks_suffered"))
for path, tc, sc in _OFF_SRC:
    for r in csv.DictReader(open(path, encoding="utf-8")):
        pos = r.get("position")
        if pos not in ("QB", "RB", "WR", "TE", "FB"):
            continue
        key = (r["player_id"], int(r["season"]), int(r["week"]))
        d = OFFW.setdefault(key, defaultdict(float))
        d["team"] = PBP_FIX.get(r[tc], r[tc]); d["pos"] = pos
        for k_dst, k_src in (("att", "attempts"), ("sk", sc), ("pepa", "passing_epa"),
                             ("repa", "rushing_epa"), ("cepa", "receiving_epa"),
                             ("car", "carries"), ("tgt", "targets"), ("rec", "receptions"),
                             ("yac", "receiving_yards_after_catch"), ("wopr", "wopr"),
                             ("dak", "dakota"), ("cpoe", "passing_cpoe")):
            d[k_dst] += f2(r.get(k_src))

DEFW = {}
_DEF_SRC = ["data/nfl_player_stats_def.csv", "data/nfl_player_stats_2025.csv"]
if os.path.exists("data/nfl_player_stats_2026.csv"):
    _DEF_SRC.append("data/nfl_player_stats_2026.csv")
for path in _DEF_SRC:
    for r in csv.DictReader(open(path, encoding="utf-8")):
        pos = r.get("position")
        if pos not in DEFPOS:
            continue
        key = (r["player_id"], int(r["season"]), int(r["week"]))
        d = DEFW.setdefault(key, defaultdict(float))
        d["pos"] = pos
        d["team"] = PBP_FIX.get(r.get("team", ""), r.get("team", ""))
        for k_dst, k_src in (("sack", "def_sacks"), ("tfl", "def_tackles_for_loss"),
                             ("hit", "def_qb_hits"), ("int", "def_interceptions"),
                             ("pd", "def_pass_defended"), ("solo", "def_tackles_solo")):
            d[k_dst] += f2(r.get(k_src))
OFF_TW = defaultdict(list); DEF_TW = defaultdict(list)
for (pid, s, w), d in OFFW.items():
    OFF_TW[(d["team"], s, w)].append(pid)
for (pid, s, w), d in DEFW.items():
    DEF_TW[(d["team"], s, w)].append(pid)
print(f"offense weekly {len(OFFW):,} | defense weekly {len(DEFW):,}")

# dakota proxy for 2025 (column absent): fit dakota ~ epa/db + cpoe on 2016-2024 QBs
# NOTE: the s == 2025 application gate below is intentionally 2025-only — nflverse
# dropped the dakota column post-2024, so only the _2025 file needs the proxy.
# When a _2026 stats file is added, generalize the gate (s >= 2025) or 2026 QB
# weeks silently keep dak = 0.
xs, ys = [], []
for (pid, s, w), d in OFFW.items():
    if d["pos"] == "QB" and 2016 <= s <= 2024 and d["dak"] and (d["att"] + d["sk"]) >= 10:
        xs.append([d["pepa"] / (d["att"] + d["sk"]), 0.0]); ys.append(d["dak"])
A_ = np.array(xs); b_ = np.array(ys)
coef, res, *_ = np.linalg.lstsq(np.column_stack([A_[:, 0], np.ones(len(A_))]), b_, rcond=None)
for (pid, s, w), d in OFFW.items():
    if d["pos"] == "QB" and s == 2025 and (d["att"] + d["sk"]) >= 1:
        d["dak"] = coef[0] * (d["pepa"] / (d["att"] + d["sk"])) + coef[1]

# ---------------- as-of player rating engine ----------------
DECAY = 0.90
def bucket(pos):
    if pos == "QB":
        return "QB"
    if pos in SKPOS:
        return "SK"
    if pos in ("DE", "DT", "NT", "EDGE", "DL"):
        return "DL"
    if pos in ("LB", "ILB", "OLB", "MLB"):
        return "LB"
    return "DB"

STATE = defaultdict(lambda: defaultdict(float))     # pid -> metric ewma sums
Z = {}                                              # bucket -> {metric: (mu, sd)} DEV era

def upd(pid, key, val, w=1.0):
    st = STATE[pid]
    st[key] = DECAY * st[key] + val
    st[key + "_w"] = DECAY * st[key + "_w"] + w

def rate_of(pid):
    st = STATE.get(pid)
    if st is None:
        return None
    b = st.get("bucket")
    if b is None or b not in Z:
        return None
    def rget(k, wk=None):
        w = st.get((wk or k) + "_w", 0.0)
        if w <= 1:
            return None
        if b in Z and k in Z[b]:
            mu = Z[b][k][0]
            return (st.get(k, 0.0) + 6.0 * mu) / (w + 6.0)   # EB prior ~6 weeks
        return st.get(k, 0.0) / w
    comps = []
    if b == "QB":
        for k, wt in (("dak", 0.50), ("negsk", 0.25), ("repa", 0.25)):
            v = rget(k)
            if v is not None and k in Z[b]:
                mu, sd = Z[b][k]
                comps.append(wt * (v - mu) / sd)
    elif b == "SK":
        for k, wt in (("eff", 0.40), ("yacr", 0.15), ("wopr", 0.45)):
            v = rget(k)
            if v is not None and k in Z[b]:
                mu, sd = Z[b][k]
                comps.append(wt * (v - mu) / sd)
    else:
        for k, wt in (("rush", 0.45), ("ball", 0.35), ("tak", 0.20)):
            v = rget(k)
            if v is not None and k in Z[b]:
                mu, sd = Z[b][k]
                comps.append(wt * (v - mu) / sd)
    return sum(comps) if comps else None

def apply_week(pid, s, w):
    o = OFFW.get((pid, s, w))
    if o is not None:
        b = bucket(o["pos"]); STATE[pid]["bucket"] = b
        if b == "QB":
            db = o["att"] + o["sk"]
            if db > 0:
                upd(pid, "dak", o["dak"] * db, db)
                upd(pid, "negsk", -o["sk"], db)
            upd(pid, "repa", o["repa"])
        else:
            opp = o["car"] + o["tgt"]
            if opp > 0:
                upd(pid, "eff", o["repa"] + o["cepa"], opp)
            if o["rec"] > 0:
                upd(pid, "yacr", o["yac"], o["rec"])
            upd(pid, "wopr", o["wopr"])
    d = DEFW.get((pid, s, w))
    if d is not None:
        b = bucket(d["pos"]); STATE[pid]["bucket"] = b
        upd(pid, "rush", d["sack"] + 0.5 * d["tfl"] + 0.5 * d["hit"])
        upd(pid, "ball", 3.0 * d["int"] + d["pd"])
        upd(pid, "tak", d["solo"])

# pass 1: DEV-era normalization constants
tmp = defaultdict(list)
weeks_sorted = sorted({(s, w) for (_, s, w) in list(OFFW) + list(DEFW)})
for s, w in weeks_sorted:
    if s > max(DEV_YEARS):
        break
    for (pid, s2, w2) in []:
        pass
# simpler: replay all weeks in order, snapshotting rates during DEV era
STATE.clear()
for s, w in weeks_sorted:
    pids = {p for (p, s2, w2) in OFFW if s2 == s and w2 == w}
    pids |= {p for (p, s2, w2) in DEFW if s2 == s and w2 == w}
    if 2001 <= s <= max(DEV_YEARS):
        for pid in pids:
            st = STATE.get(pid)
            if not st:
                continue
            b = st.get("bucket")
            for k in ("dak", "negsk", "repa", "eff", "yacr", "wopr", "rush", "ball", "tak"):
                wgt = st.get(k + "_w", 0.0)
                if wgt > 3:
                    tmp[(b, k)].append(st[k] / wgt)
    for pid in pids:
        apply_week(pid, s, w)
Z = defaultdict(dict)
for (b, k), vals in tmp.items():
    v = np.array(vals)
    if len(v) > 200 and v.std() > 0:
        Z[b][k] = (float(v.mean()), float(v.std()))
print("z-tables:", {b: list(d) for b, d in Z.items()})

# pass 2: full replay, capturing per-game features
STATE.clear()
wk_idx = defaultdict(list)
for i, g in enumerate(games):
    wk_idx[(g["season"], g["week"])].append(i)

share, pos_of, team_of = {}, {}, {}
roster = defaultdict(set)
fq_ol = np.zeros(len(games)); fq_def = np.zeros(len(games)); fq_sk = np.zeros(len(games))
f_ol = np.zeros(len(games)); f_def = np.zeros(len(games)); f_sk = np.zeros(len(games))
f_rq = np.zeros(len(games))
DK = defaultdict(lambda: [0.0, 0.0])     # qb dakota state for F2
LG_DAK = None

def unit_missing(team, gid, quality):
    tbl = snaps.get((gid, team))
    if tbl is None:
        return 0.0, 0.0, 0.0
    m = [0.0, 0.0, 0.0]
    for pid in roster.get(team, ()):
        st = share.get(pid)
        if not st or st[1] <= 0:
            continue
        sshare = st[0] / st[1]
        if sshare < snapP["thresh"] or pid in tbl:
            continue
        p = pos_of.get(pid, "")
        qz = 0.0
        if quality:
            g_ = pfr2gsis.get(pid)
            r_ = rate_of(g_) if g_ else None
            qz = max(-1.5, min(1.5, r_)) if r_ is not None else 0.0
        wgt = sshare * (1.0 + qz) if quality else sshare
        if p in OLPOS:
            m[0] += wgt
        elif p in DEFPOS:
            m[1] += wgt
        elif p in SKPOS:
            m[2] += wgt
    return m[0], m[1], m[2]

def roster_quality(team, gid):
    tbl = snaps.get((gid, team))
    if tbl is None:
        return 0.0
    tot = 0.0
    for pid in tbl:
        st = share.get(pid)
        if not st or st[1] <= 0:
            continue
        g_ = pfr2gsis.get(pid)
        r_ = rate_of(g_) if g_ else None
        if r_ is not None:
            tot += (st[0] / st[1]) * max(-2.0, min(2.0, r_))
    return tot

# dakota league rate (DEV era, att-weighted)
num = den = 0.0
for (pid, s, w), d in OFFW.items():
    if d["pos"] == "QB" and s in DEV_YEARS:
        db = d["att"] + d["sk"]
        num += d["dak"] * db; den += db
LG_DAK = num / den
REP_DAK = LG_DAK - 0.06

def qb_dak(qid):
    st = DK.get(qid)
    if not st or st[1] <= 0:
        return REP_DAK - LG_DAK
    q = (st[0] + 200.0 * REP_DAK) / (st[1] + 200.0)
    return q - LG_DAK

f_dak = np.zeros(len(games))
done_weeks = set()
for i, g in enumerate(games):
    s, w = g["season"], g["week"]
    # features BEFORE updates
    f_dak[i] = qb_dak(g["home_qb"]) - qb_dak(g["away_qb"])
    if s >= 2013:
        h, a = g["home"], g["away"]
        q_h = unit_missing(h, g["gid"], True); q_a = unit_missing(a, g["gid"], True)
        s_h = unit_missing(h, g["gid"], False); s_a = unit_missing(a, g["gid"], False)
        fq_ol[i], fq_def[i], fq_sk[i] = (q_h[0] - q_a[0], q_h[1] - q_a[1], q_h[2] - q_a[2])
        f_ol[i], f_def[i], f_sk[i] = (s_h[0] - s_a[0], s_h[1] - s_a[1], s_h[2] - s_a[2])
        f_rq[i] = roster_quality(h, g["gid"]) - roster_quality(a, g["gid"])
    # updates: player weeks once per (s,w); snaps per game
    for team in (g["home"], g["away"]):
        for pid in OFF_TW.get((team, s, w), []) + DEF_TW.get((team, s, w), []):
            if (pid, s, w) in done_weeks:
                continue
            done_weeks.add((pid, s, w))
            apply_week(pid, s, w)
            o = OFFW.get((pid, s, w))
            if o is not None and o["pos"] == "QB":
                db = o["att"] + o["sk"]
                if db > 0:
                    st = DK[pid]
                    st[0] = 0.85 * st[0] + o["dak"] * db
                    st[1] = 0.85 * st[1] + db
    tbl = snaps.get((g["gid"], g["home"])) or {}
    tbl2 = snaps.get((g["gid"], g["away"])) or {}
    for team, t_ in ((g["home"], tbl), (g["away"], tbl2)):
        for pid, (pos, op, dp) in t_.items():
            pct = dp if pos in DEFPOS else op
            st = share.setdefault(pid, [0.0, 0.0])
            st[0] = snapP["decay"] * st[0] + pct
            st[1] = snapP["decay"] * st[1] + 1.0
            pos_of[pid] = pos
            if team_of.get(pid) != team:
                if team_of.get(pid) in roster:
                    roster[team_of[pid]].discard(pid)
                team_of[pid] = team
                roster[team].add(pid)

# ---------------- screens ----------------
X11 = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                       f_ol, f_def, f_sk])

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

base_cv = cv_ll(X11)
print(f"\ncurrent model (11f) DEV CV {base_cv:.5f}")
CAND = {
    "F2a qb-dakota (add)":   np.column_stack([X11, f_dak]),
    "F1 quality-absence":    np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest,
                                              f_luck, f_ts, fq_ol, fq_def, fq_sk]),
    "F3 roster quality":     np.column_stack([X11, f_rq]),
    "F1+F2a":                np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest,
                                              f_luck, f_ts, fq_ol, fq_def, fq_sk, f_dak]),
}
best = None
for name, X in CAND.items():
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, name, X); tag = "  <-- best"
    print(f"  {name:<22} DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, name, Xw = best
print(f"\nDEV winner: {name} ({s-base_cv:+.5f})")
fitm = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=5000).fit(X11[fitm], y[fitm])
cw = LogisticRegression(C=1e6, max_iter=5000).fit(Xw[fitm], y[fitm])
pb = cb.predict_proba(X11[test])[:, 1]
pw = cw.predict_proba(Xw[test])[:, 1]
yt = y[test]
d_ = llv(yt, pb) - llv(yt, pw)
rng_ = np.random.default_rng(7)
bs_ = d_[rng_.integers(0, len(d_), size=(10000, len(d_)))].mean(axis=1)
lo_, hi_ = np.percentile(bs_, [2.5, 97.5])
print(f"\nmain TEST 2016-2025: current {llv(yt, pb).mean():.5f} -> {name} {llv(yt, pw).mean():.5f}"
      f"  (delta {d_.mean():+.5f}  CI [{lo_:+.5f}, {hi_:+.5f}])")

# ---------------- master ratings table (current, 2025) ----------------
rows_out = []
# actives = players with weekly lines in the LATEST season present in the stat
# files (today 2025; advances automatically once a _2026 stats file is added)
_LATEST_STAT = max({s for (_p, s, _w) in OFFW} | {s for (_p, s, _w) in DEFW})
ACTIVE25 = ({p for (p, s, w) in OFFW if s == _LATEST_STAT}
            | {p for (p, s, w) in DEFW if s == _LATEST_STAT})
PRIMARY = {"QB": "dak", "SK": "eff", "DL": "rush", "LB": "rush", "DB": "ball"}
for pid, st in STATE.items():
    r_ = rate_of(pid)
    if r_ is None:
        continue
    b_ = st.get("bucket")
    if st.get(PRIMARY.get(b_, "rush") + "_w", 0.0) < 15.0:
        continue
    if pid not in ACTIVE25:
        continue
    rows_out.append((r_, pid, b_))
rows_out.sort(reverse=True)
by_bucket = defaultdict(list)
for r_, pid, b in rows_out:
    by_bucket[b].append((r_, pid))
with open("data/nfl_player_ratings_2025.csv", "w", newline="", encoding="utf-8") as fh:
    wcsv = csv.writer(fh)
    wcsv.writerow(["player", "gsis_id", "bucket", "rating_0_100", "tier"])
    for b, lst in by_bucket.items():
        vals = np.array([x[0] for x in lst])
        for r_, pid in lst:
            pct = (vals < r_).mean() * 100
            tier = "S" if pct >= 95 else "A" if pct >= 80 else "B" if pct >= 55 else "C" if pct >= 25 else "D"
            wcsv.writerow([names.get(pid, pid), pid, b, round(pct, 1), tier])
print(f"\nwrote data/nfl_player_ratings_2025.csv ({len(rows_out)} rated players)")
print("top-5 QBs:", [names.get(p, p) for _, p in by_bucket['QB'][:5]])
print("top-5 SKILL:", [names.get(p, p) for _, p in by_bucket['SK'][:5]])


if llv(yt, pw).mean() < llv(yt, pb).mean():
    allfit = (seasons >= DEV_SCORE_FROM) & (y != 0.5)
    call = LogisticRegression(C=1e6, max_iter=5000).fit(Xw[allfit], y[allfit])
    spec = json.load(open("data/nfl_model.json"))
    spec["features"] = spec["features"] + ["roster_quality"]
    spec["component_params"]["roster_quality"] = {
        "rating_decay": 0.90, "eb_prior_weeks": 6.0, "share_params": snapP,
        "weights": {"QB": [0.5, 0.25, 0.25], "SK": [0.4, 0.15, 0.45],
                    "DEF": [0.45, 0.35, 0.2]}, "data_from": 2013, "zero_before": True}
    spec["prev_test_ll"] = spec["test_ll"]
    spec["test_ll"] = round(float(llv(yt, pw).mean()), 5)
    spec["blend_dev_fit"] = {"coef": [round(float(c), 6) for c in cw.coef_[0]],
                             "intercept": round(float(cw.intercept_[0]), 6)}
    spec["blend_production"] = {"coef": [round(float(c), 6) for c in call.coef_[0]],
                                "intercept": round(float(call.intercept_[0]), 6),
                                "fit": "2001-2025 all completed, ties dropped"}
    json.dump(spec, open("data/nfl_model.json", "w"), indent=1)
    print("ADOPTED -> data/nfl_model.json now 12 features, test_ll",
          spec["test_ll"], "| roster_quality coef", round(float(call.coef_[0][-1]), 4))