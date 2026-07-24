"""Participation TrueSkill: every play is a match between the 11 offensive and 11
defensive players actually on the field (nflverse pbp_participation, 2016-2025).
Bayesian (mu, sigma) update for all 22 after each play; offense "wins" iff EPA > 0.

Design (per the spec, knobs DEV-tuned vs the ratings-only game model —
phase0/nfl_ratings_only_model.py):
  outcome        GRADED: |EPA| < 0.55 = DRAW (TrueSkill draw update — near-zero
                 plays carry no win/lose signal), else offense wins iff EPA > 0;
                 processed strictly chronologically
  snap context   weight = 4*wp*(1-wp) leverage (garbage time fades smoothly)
                 x 1.25 on money downs (3rd/4th) x 1.5 late-and-close (Q4/OT, one
                 score) — every snap counts by how much it mattered
  recency        weekly tau 0.30 (9x the old forgetting): recent snaps dominate
  noise          beta = sigma0 (NFL plays are noisy -> large beta)
  offseason      mu shrunk 1/3 toward the mean, sigma widened (capped at sigma0)
  position-aware ratings LADDERS are global (o-vs-d is what a play measures) but the
                 DISPLAY is normalized within position group (QB/RB/WR/TE/OL/DL/LB/DB)
                 so a great CB and a great LT are never compared to each other
  small samples  conservative skill = mu - 3*sigma, plus shrink-to-50 below the
                 qualification floor; under 50 rated snaps -> no rating shown

Outputs:
  data/nfl_player_ts.csv            full ladder (all rated players)
  data/nfl_player_ratings_2025.csv  site-schema replacement (2025 actives)
  site/data/nfl.json                players[].rating swapped in, team averages redone
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from math import erf, exp, pi, sqrt

T0 = time.time()
MU0, SIG0 = 25.0, 25.0 / 3.0
SIG0_2 = SIG0 * SIG0
BETA = 60.0                      # DEV-swept: single plays are ~7x noisier than the
                                 # prior scale suggested; big beta = smooth learning
                                 # (U-curve peak 45-60; the sweep's biggest win)
TAU2 = 0.30 ** 2                 # weekly drift — DEV-tuned vs the ratings-only
                                 # game model (recent snaps dominate the posterior)
SEASON_SHRINK = 1.0 / 3.0        # offseason: mu 1/3 of the way back to MU0
SEASON_WIDEN2 = 1.5 ** 2
SIG2_FLOOR = 0.5 ** 2
DOWN_MULT = 1.25                 # money downs (3rd/4th) count more
LATE_MULT = 1.5                  # 4th quarter / OT within one score counts more
PLAYOFF_MULT = 1.5               # postseason snaps count more
DEAD_MULT = 0.5                  # both teams eliminated (can't reach 7 wins) counts half
OPP_K = 0.3                      # your update scales with the OPPOSING unit's strength
                                 # vs the RUNNING league mean (anchoring to the fixed
                                 # prior inflates ratings; league-anchored is clean and
                                 # generalized better) — the engine's biggest tuning win
FUS_K = 4.0                      # QB weekly stat fusion: obs = 25 + k*z(weekly EPA/db)
FUS_R = 36.0                     # Kalman obs noise — DEV-swept, biggest v6 win (-0.0092)
DRAW_BAND = 0.40                 # |EPA| below this = a DRAW (re-tuned at beta 60):
                                 # near-zero plays carry almost no win/lose information;
                                 # dose-response peaked at 0.55 (DEV -0.0088, engine's
                                 # biggest single win)
MIN_SNAPS_SHOW = 50

POSMAP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
    "LT": "OL", "RT": "OL", "LG": "OL", "RG": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "SAF": "DB", "DB": "DB",
}
QUAL = {"QB": 150, "RB": 150, "WR": 150, "TE": 120,
        "OL": 250, "DL": 200, "LB": 200, "DB": 200}
OFFG = {"QB", "RB", "WR", "TE", "OL"}


def pdf(x): return exp(-0.5 * x * x) / sqrt(2 * pi)
def cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def v_win(t):
    c = cdf(t)
    if c < 1e-12:
        return -t                # asymptote of the truncated-normal mean
    return pdf(t) / c


# ---- load full snap context (wp leverage, down, quarter, score) ----
ctx_of = {}
try:
    with open("data/nfl_play_ctx2.csv") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                ctx_of[(r["game_id"], int(float(r["play_id"])))] = (
                    float(r["wp"]) if r["wp"] else 0.5,
                    int(float(r["down"])) if r["down"] else 1,
                    int(float(r["qtr"])) if r["qtr"] else 1,
                    abs(float(r["score_differential"])) if r["score_differential"] else 0.0,
                )
            except ValueError:
                continue
    print(f"[{time.time()-T0:.0f}s] snap context: {len(ctx_of):,} plays", flush=True)
except FileNotFoundError:
    print("no snap context; every play weighted 1.0", flush=True)

# ---- game importance per gid: playoffs boosted, dead games (both teams unable
# to reach 7 wins on PRE-game records) halved ----
gmult_of = {}
_wins = defaultdict(float); _played = defaultdict(int); _prev = None
_rows = [r for r in csv.DictReader(open("data/nfl_games.csv"))
         if r["home_score"] != "" and int(r["season"]) >= 2016]
_rows.sort(key=lambda r: r["gameday"])
for r in _rows:
    if _prev is not None and r["season"] != _prev:
        _wins.clear(); _played.clear()
    _prev = r["season"]
    h, a = r["home_team"], r["away_team"]
    if r["game_type"] != "REG":
        gmult_of[r["game_id"]] = PLAYOFF_MULT
    else:
        sched_len = 16 if int(r["season"]) < 2021 else 17
        dead_h = _wins[h] + (sched_len - _played[h]) < 7
        dead_a = _wins[a] + (sched_len - _played[a]) < 7
        gmult_of[r["game_id"]] = DEAD_MULT if (dead_h and dead_a) else 1.0
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        yv = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        _wins[h] += yv; _wins[a] += 1.0 - yv
        _played[h] += 1; _played[a] += 1
print(f"[{time.time()-T0:.0f}s] game importance: "
      f"{sum(1 for v in gmult_of.values() if v == PLAYOFF_MULT)} playoff, "
      f"{sum(1 for v in gmult_of.values() if v == DEAD_MULT)} dead games", flush=True)

# ---- QB weekly observations (for the fusion) ----
import sys as _sys
_sys.path.insert(0, "phase0")
from nfl_qb_elo import load_qb_weeks  # noqa: E402
import numpy as _np  # noqa: E402
qbw = load_qb_weeks()
lg_qb = json.load(open("data/nfl_qb_elo.json"))["qb_elo"]["lg_rate"]
_rates = [ln[0] / ln[1] for (q_, s_, w_), ln in qbw.items()
          if 2016 <= s_ <= 2021 and ln[1] >= 10]
SD_QB = float(_np.std(_rates)) if _rates else 0.15
QBOBS = {}
for (q_, s_, w_), ln in qbw.items():
    if s_ >= 2016 and ln[1] >= 10:
        z_ = max(-2.5, min(2.5, (ln[0] / ln[1] - lg_qb) / SD_QB))
        QBOBS[(q_, str(s_), f"{w_:02d}")] = z_
print(f"[{time.time()-T0:.0f}s] QB weekly obs: {len(QBOBS):,}", flush=True)
QB_POS = set()
for _r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    if _r.get("gsis_id") and (_r.get("position") or "") == "QB":
        QB_POS.add(_r["gsis_id"])

# ---- load plays, strictly chronological ----
plays = []
with open("data/nfl_rapm_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); next(rd)
    for gid, pid_, off, dfn, epa in rd:
        plays.append((gid, int(float(pid_)), off, dfn, float(epa)))
plays.sort(key=lambda p: (p[0], p[1]))
print(f"[{time.time()-T0:.0f}s] {len(plays):,} plays loaded", flush=True)

mu = {}
sig2 = {}
snaps = defaultdict(int)
last_week = {}
cur_season = None
cur_gid = None
lg_mu = MU0
n_upd = 0
for gid, pid_, off, dfn, epa in plays:
    if gid != cur_gid:                       # refresh the league anchor per game
        lg_mu = (sum(mu.values()) / len(mu)) if mu else MU0
        cur_gid = gid
    season, week = gid[:4], gid[5:7]
    if cur_season is not None and season != cur_season:
        for p in mu:
            mu[p] = MU0 + (mu[p] - MU0) * (1.0 - SEASON_SHRINK)
            sig2[p] = min(sig2[p] + SEASON_WIDEN2, SIG0_2)
    cur_season = season
    O = off.split(";")
    D = dfn.split(";")
    if not (6 <= len(O) <= 12 and 6 <= len(D) <= 12):
        continue                                     # malformed participation row
    wk = (season, week)
    for p in O + D:
        if p not in mu:
            mu[p] = MU0; sig2[p] = SIG0_2
        elif last_week.get(p) != wk:
            if p in QB_POS:                          # fuse the completed week's line
                _old = last_week.get(p)
                _z = QBOBS.get((p, _old[0], _old[1])) if _old else None
                if _z is not None:
                    _obs = MU0 + FUS_K * _z
                    _K = sig2[p] / (sig2[p] + FUS_R)
                    mu[p] += _K * (_obs - mu[p])
                    sig2[p] = max(sig2[p] * (1.0 - _K), SIG2_FLOOR)
            sig2[p] = min(sig2[p] + TAU2, SIG0_2)
        last_week[p] = wk
    # context weight: smooth WP leverage (garbage time fades naturally),
    # money downs, late-and-close, and GAME IMPORTANCE — DEV-tuned vs the
    # ratings-only model
    c_ = ctx_of.get((gid, pid_))
    if c_ is None:
        wgt = 1.0
    else:
        wp, down, qtr, adiff = c_
        wgt = (4.0 * wp * (1.0 - wp)) ** 0.5    # sqrt leverage (DEV-swept exponent)
        if down >= 3:
            wgt *= DOWN_MULT
        if qtr >= 4 and adiff <= 8:
            wgt *= LATE_MULT
    wgt *= gmult_of.get(gid, 1.0)
    sum_O = sum(mu[p] for p in O); sum_D = sum(mu[p] for p in D)
    c2 = (len(O) + len(D)) * BETA * BETA + sum(sig2[p] for p in O) + sum(sig2[p] for p in D)
    c = sqrt(c2)
    t = (sum_O - sum_D) / c
    # beating GREAT opponents matters more: each side's update scales with the
    # opposing unit's mean rating (the biggest DEV win of the engine, -0.0124)
    w_O = wgt * min(2.0, max(0.3, 1.0 + OPP_K * (sum_D / len(D) - lg_mu)))
    w_D = wgt * min(2.0, max(0.3, 1.0 + OPP_K * (sum_O / len(O) - lg_mu)))
    if abs(epa) < DRAW_BAND:                     # near-zero play -> proper draw update
        e_ = 0.05
        den = cdf(e_ - t) - cdf(-e_ - t)
        if den > 1e-9:
            vd = (pdf(-e_ - t) - pdf(e_ - t)) / den
            wd = max(0.0, min(1.0, vd * vd + ((e_ - t) * pdf(e_ - t)
                                              + (e_ + t) * pdf(-e_ - t)) / den))
            for p in O:
                mu[p] += w_O * (sig2[p] / c) * vd
                sig2[p] = max(sig2[p] * (1.0 - w_O * (sig2[p] / c2) * wd), SIG2_FLOOR)
            for p in D:
                mu[p] -= w_D * (sig2[p] / c) * vd
                sig2[p] = max(sig2[p] * (1.0 - w_D * (sig2[p] / c2) * wd), SIG2_FLOOR)
        for p in O + D:
            snaps[p] += 1
        n_upd += 1
        continue
    if epa > 0:
        win, lose, tt, w_win_, w_lose_ = O, D, t, w_O, w_D
    else:
        win, lose, tt, w_win_, w_lose_ = D, O, -t, w_D, w_O
    v = v_win(tt)
    w = v * (v + tt)
    for p in win:
        mu[p] += w_win_ * (sig2[p] / c) * v
        sig2[p] = max(sig2[p] * (1.0 - w_win_ * (sig2[p] / c2) * w), SIG2_FLOOR)
    for p in lose:
        mu[p] -= w_lose_ * (sig2[p] / c) * v
        sig2[p] = max(sig2[p] * (1.0 - w_lose_ * (sig2[p] / c2) * w), SIG2_FLOOR)
    for p in O + D:
        snaps[p] += 1
    n_upd += 1
print(f"[{time.time()-T0:.0f}s] rated {n_upd:,} plays, {len(mu):,} players", flush=True)

# ---- players metadata + 2025 actives ----
names, pos_of = {}, {}
for r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    g = r.get("gsis_id")
    if not g:
        continue
    names[g] = r.get("display_name") or g
    pos_of[g] = POSMAP.get(r.get("position") or "", None)
active25 = set()
for gid, pid_, off, dfn, _ in plays:
    if gid.startswith("2025"):
        active25.update(off.split(";")); active25.update(dfn.split(";"))

# ---- display conversion: conservative skill (TrueSkill leaderboard exposure,
# mu - 3 sigma, so uncertainty is punished hard), within-position z among actives ----
cons = {p: mu[p] - 3.0 * sqrt(sig2[p]) for p in mu}
stats = {}
for pg, floor in QUAL.items():
    vals = [cons[p] for p in mu
            if pos_of.get(p) == pg and p in active25 and snaps[p] >= floor]
    m_ = sum(vals) / len(vals)
    sd_ = sqrt(sum((v - m_) ** 2 for v in vals) / max(len(vals) - 1, 1))
    stats[pg] = (m_, sd_, len(vals))
print("position ladders (active, qualified):",
      {k: (round(v[0], 2), round(v[1], 2), v[2]) for k, v in stats.items()}, flush=True)

def rating_of(p):
    pg = pos_of.get(p)
    if pg is None or snaps[p] < MIN_SNAPS_SHOW or pg not in stats:
        return None
    m_, sd_, _ = stats[pg]
    z = (cons[p] - m_) / sd_ if sd_ > 1e-9 else 0.0
    shrink = min(snaps[p] / QUAL[pg], 1.0)           # small samples pull to 50
    r = 50.0 + 15.0 * z * shrink
    return max(1.0, min(99.0, r))

rows = []
for p in mu:
    r_ = rating_of(p)
    if r_ is None:
        continue
    rows.append({"gsis_id": p, "player": names.get(p, p), "bucket": pos_of[p],
                 "mu": round(mu[p], 3), "sigma": round(sqrt(sig2[p]), 3),
                 "snaps": snaps[p], "cons": round(cons[p], 3),
                 "rating_0_100": round(r_, 1), "active_2025": int(p in active25)})
rows.sort(key=lambda r: -r["rating_0_100"])
# tiers: percentile within position among 2025 actives
by_pg = defaultdict(list)
for r in rows:
    if r["active_2025"]:
        by_pg[r["bucket"]].append(r["rating_0_100"])
for v in by_pg.values():
    v.sort()
def tier_of(r):
    v = by_pg.get(r["bucket"])
    if not v or not r["active_2025"]:
        return "D"
    import bisect
    pct = bisect.bisect_left(v, r["rating_0_100"]) / len(v)
    return "S" if pct >= 0.90 else "A" if pct >= 0.70 else "B" if pct >= 0.40 \
        else "C" if pct >= 0.15 else "D"
for r in rows:
    r["tier"] = tier_of(r)

with open("data/nfl_player_ts.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)
with open("data/nfl_player_ratings_2025.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=["player", "gsis_id", "bucket",
                                        "rating_0_100", "tier", "n_eff"])
    wr.writeheader()
    for r in rows:
        if r["active_2025"]:
            wr.writerow({"player": r["player"], "gsis_id": r["gsis_id"],
                         "bucket": r["bucket"], "rating_0_100": r["rating_0_100"],
                         "tier": r["tier"], "n_eff": r["snaps"]})
print(f"[{time.time()-T0:.0f}s] wrote {len(rows)} rated players "
      f"({sum(r['active_2025'] for r in rows)} active 2025)", flush=True)

# litmus: top-10 QBs by rating (actives) — sample-size discipline check
qbs = [r for r in rows if r["bucket"] == "QB" and r["active_2025"]][:10]
for i, r in enumerate(qbs, 1):
    print(f"  QB{i:>2} {r['player']:<22} r {r['rating_0_100']:>5} "
          f"mu {r['mu']:>6} sig {r['sigma']:<5} snaps {r['snaps']}", flush=True)

# ---- swap into the site payload ----
by_id = {r["gsis_id"]: r for r in rows}
payload = json.load(open("site/data/nfl.json"))
n_set = n_null = 0
for pid, pl in payload["players"].items():
    r = by_id.get(pid)
    if r:
        pl["rating"] = {"r": r["rating_0_100"], "tier": r["tier"],
                        "n_eff": float(r["snaps"]), "bucket": r["bucket"],
                        "mu": r["mu"], "sigma": r["sigma"]}
        n_set += 1
    else:
        pl["rating"] = None
        n_null += 1
for t, tm in payload["teams"].items():
    num = den = noff = nof = ndef = ndf = 0.0
    for pid in tm["roster"]:
        pl = payload["players"].get(pid)
        if not pl or not pl.get("rating"):
            continue
        rt = pl["rating"]
        wgt = max(pl.get("snap_share") or 0.0, 0.05)
        num += wgt * rt["r"]; den += wgt
        if rt["bucket"] in OFFG:
            noff += wgt * rt["r"]; nof += wgt
        else:
            ndef += wgt * rt["r"]; ndf += wgt
    tm["glassbox"] = round(num / den, 1) if den else None
    tm["gb_off"] = round(noff / nof, 1) if nof else None
    tm["gb_def"] = round(ndef / ndf, 1) if ndf else None
json.dump({p: [round(mu[p], 4), round(sig2[p], 4), snaps[p]] for p in mu},
          open("data/nfl_ts_state.json", "w"))
payload["model_card"]["ratings"] = {
    "system": "per-play participation TrueSkill (11v11), context-weighted",
    "plays": n_upd, "seasons": "2016-2025",
    "outcome": "offense beats defense iff EPA > 0",
    "beta": BETA, "tau_week": sqrt(TAU2),
    "snap_weight": "sqrt leverage x money downs x late-and-close x importance x opp-quality",
    "qb_fusion": "weekly EPA/dropback line Kalman-merged (k=4, R=36)",
    "display": "mu - 3 sigma, z-scored within position group",
}
json.dump(payload, open("site/data/nfl.json", "w"))
print(f"[{time.time()-T0:.0f}s] payload: {n_set} rated, {n_null} NR; team avgs redone")
