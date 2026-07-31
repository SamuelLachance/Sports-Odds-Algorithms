"""MLB lineup-absence feature — value of USUAL regulars missing from tonight's
announced starting lineup. DEV SCREEN ONLY (2002-2015). NO TEST-era evaluation
lives in this file; the one TEST look is spent by the orchestrator.

NHL analog: ledger row 10 (+0.00077 CI[-0.00009,+0.00164]) with a weak TOI value
proxy. MLB has exact historical starting lineups (data/retro_events/lineups.csv,
Retrosheet: side 0 = visitor, 1 = home, 9 batters each) and a real per-player
value scale (the per-PA TrueSkill mu walk, mlbwp/trueskill.py — the same engine
the shipped ts_edge feature uses).

  regular(team, t) = batter with EWMA start-share >= MIN_SHARE and >= MIN_SEEN
                     lineup appearances (pitchers batting 9th in NL parks never
                     reach the share bar — a 5-man rotation tops out ~0.2)
  absence(team)    = sum over regulars NOT in tonight's starting 9 of
                     (mu_p - share-weighted mean mu of the team's regulars)
  feature          = away_absence - home_absence   (positive = home stronger)
  variants         = n_missing diff; share-weighted value diff

No-leak argument: starting lineups are announced 1-4h pre-game; the historical
actual lineup = the announced one modulo rare post-announcement scratches (same
argument as NHL rows 9/10). TrueSkill mu is read as-of BEFORE the game's PAs;
share/seen/tgame state updates strictly AFTER the feature is read. First
SEASON_MASK team-games each season are masked (roster churn).

Base model: shipped full_p from data/model_probs.csv (join by reconstructing
game_id = home + yyyymmdd + dh, exactly how phase0/model_probs.py emitted dh).
NOTE full_p already prices tonight's ACTUAL lineup (ts_edge averages mu over the
announced 9), so this screens the increment of the absence-vs-usual DEVIATION
beyond actual-lineup mean strength.

Screen harness: 5 season-grouped CV folds WITHIN 2002-2015, logistic
recalibration logit(full_p) vs logit(full_p)+feature_z per fold, pooled OOF
mean-LL delta + paired bootstrap CI. Gate: <= -0.0002 nothing, >= +0.0005
material.

Usage: --build (walk PAs + lineups -> data/retro_events/absence_feature.csv)
       --screen (DEV-only increment screen; hard-capped at season 2015)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from sys import intern

sys.path.insert(0, ".")

FEAT = "data/retro_events/absence_feature.csv"
REPORT = "data/mlb_absence_dev.json"

SH_DECAY = 0.90        # start-share EWMA (per team-game)
MIN_SHARE = 0.5
MIN_SEEN = 10          # lineup appearances before a player can be a "regular"
SEASON_MASK = 10       # first N team-games each season: feature = NaN
SEASON_SHARE_SHRINK = 0.5

DEV_LO, DEV_HI = 2002, 2015   # the screen NEVER touches a season > DEV_HI
N_FOLDS = 5
EPS = 1e-15

F_DATE, F_VIS, F_HOME = 0, 3, 6


def load_spine():
    """Game-log spine: [(gid, date, season, home, away)] sorted by (date, gid).
    Includes ties (state must update on every played game)."""
    rows = []
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            gid = r[F_HOME] + r[F_DATE] + r[1]
            rows.append((gid, r[F_DATE], int(r[F_DATE][:4]), r[F_HOME], r[F_VIS]))
    rows.sort(key=lambda x: (x[1], x[0]))
    return rows


def load_pa_by_game():
    pa = defaultdict(list)
    for r in csv.reader(open("data/retro_events/pa.csv")):
        if r[0] == "game_id":
            continue
        pa[r[0]].append((intern(r[4]), intern(r[5]), r[6] == "1"))
    return pa


def load_lineups():
    lu = defaultdict(lambda: {0: [], 1: []})
    for r in csv.reader(open("data/retro_events/lineups.csv")):
        if r[0] == "game_id":
            continue
        lu[r[0]][int(r[1])].append(intern(r[3]))
    return lu


def build():
    from mlbwp.trueskill import TrueSkill1v1
    spine = load_spine()
    print(f"spine: {len(spine):,} games", flush=True)
    pa = load_pa_by_game()
    lu = load_lineups()
    print(f"PA games: {len(pa):,}  lineup games: {len(lu):,}", flush=True)

    ts = TrueSkill1v1()
    share = defaultdict(dict)                     # team -> pid -> EWMA start share
    seen = defaultdict(lambda: defaultdict(int))  # team -> pid -> appearances
    tgame = defaultdict(int)                      # team -> games this season
    prev_season = None
    out = []

    for gid, date, season, home, away in spine:
        if prev_season is not None and season != prev_season:
            for t in share:
                for pid in share[t]:
                    share[t][pid] *= SEASON_SHARE_SHRINK
            tgame.clear()
        prev_season = season

        ln = lu.get(gid)
        # ---- read features BEFORE any state update ----
        vals = {}
        for side, team in ((1, home), (0, away)):
            lineup = set(ln[side]) if ln else set()
            if not lineup or tgame[team] < SEASON_MASK:
                vals[side] = None
                continue
            regs = [(p, s) for p, s in share[team].items()
                    if s >= MIN_SHARE and seen[team][p] >= MIN_SEEN]
            if not regs:
                vals[side] = None
                continue
            wsum = sum(s for _, s in regs)
            avg_mu = sum(s * ts.mu[p] for p, s in regs) / wsum
            a_val = a_wval = 0.0
            a_n = 0
            for p, s in regs:
                if p not in lineup:
                    a_val += ts.mu[p] - avg_mu
                    a_wval += s * (ts.mu[p] - avg_mu)
                    a_n += 1
            vals[side] = (a_val, a_n, a_wval)

        if vals[1] is not None and vals[0] is not None:
            out.append([gid, date, season,
                        round(vals[0][0] - vals[1][0], 4),   # away - home value
                        vals[0][1] - vals[1][1],             # away - home count
                        round(vals[0][2] - vals[1][2], 4),   # share-weighted
                        round(vals[1][0], 4), round(vals[0][0], 4)])

        # ---- update state strictly AFTER reading (leak-safe) ----
        for b, p, on_base in pa.get(gid, ()):
            if on_base:
                ts.update(b, p)
            else:
                ts.update(p, b)
        if ln:
            for side, team in ((1, home), (0, away)):
                lineup = set(ln[side])
                if not lineup:
                    continue
                tgame[team] += 1
                for pid in set(share[team]) | lineup:
                    d = 1.0 if pid in lineup else 0.0
                    share[team][pid] = SH_DECAY * share[team].get(pid, 0.0) + (1 - SH_DECAY) * d
                for pid in lineup:
                    seen[team][pid] += 1

    with open(FEAT + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "season", "abs_val", "abs_n", "abs_wval",
                    "home_abs", "away_abs"])
        w.writerows(out)
    os.replace(FEAT + ".tmp", FEAT)
    print(f"wrote {FEAT}: {len(out):,} games with feature", flush=True)


def screen():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    feats = {}
    for r in csv.DictReader(open(FEAT, encoding="utf-8")):
        feats[r["game_id"]] = (float(r["abs_val"]), float(r["abs_n"]),
                               float(r["abs_wval"]))

    rows = []
    n_dev_probs = 0
    for r in csv.DictReader(open("data/model_probs.csv")):
        season = int(r["season"])
        if season < DEV_LO or season > DEV_HI:      # HARD CAP: no TEST-era rows
            continue
        if r["full_p"] == "":
            continue
        n_dev_probs += 1
        gid = r["home"] + r["date"].replace("-", "") + r["dh"]
        f = feats.get(gid)
        if f is None:
            continue
        rows.append((season, float(r["full_p"]), int(r["y"]), f))
    assert all(DEV_LO <= s <= DEV_HI for s, *_ in rows)
    print(f"DEV games with full_p: {n_dev_probs:,}; joined feature: {len(rows):,} "
          f"({len(rows)/max(n_dev_probs,1):.1%} coverage)")

    seas = np.array([r[0] for r in rows])
    p = np.clip(np.array([r[1] for r in rows]), EPS, 1 - EPS)
    base_z = np.log(p / (1 - p))
    y = np.array([r[2] for r in rows], float)
    F = {"abs_val": np.array([r[3][0] for r in rows]),
         "abs_n": np.array([r[3][1] for r in rows]),
         "abs_wval": np.array([r[3][2] for r in rows])}

    v = F["abs_val"]
    print(f"feature stats (abs_val, away-home): std {v.std():.3f} mu-units, "
          f"share |v|>0.5: {np.mean(np.abs(v) > 0.5):.1%}, "
          f"share any missing regular: {np.mean(F['abs_n'] != 0) + np.mean((F['abs_n'] == 0) & (np.abs(v) > 1e-9)):.1%}")
    any_miss = np.mean((np.abs(F["abs_val"]) > 1e-9) | (F["abs_n"] != 0))
    print(f"share of games with any absence signal: {any_miss:.1%}")

    # 5 season-grouped folds within DEV
    dev_seasons = sorted(set(seas.tolist()))
    groups = np.array_split(np.array(dev_seasons), N_FOLDS)

    def llv(yy, pp):
        pp = np.clip(pp, EPS, 1 - EPS)
        return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

    res = {"n": int(len(rows)), "coverage": round(len(rows) / max(n_dev_probs, 1), 4),
           "dev_years": f"{DEV_LO}-{DEV_HI}", "folds": N_FOLDS, "variants": {}}
    for name, feat in F.items():
        d_base = np.empty(len(y)); d_with = np.empty(len(y))
        coefs = []
        for grp in groups:
            te = np.isin(seas, grp); tr = ~te
            mu, sd = feat[tr].mean(), feat[tr].std()
            fz = (feat - mu) / (sd if sd > 0 else 1.0)
            Xb = base_z.reshape(-1, 1)
            X = np.column_stack([base_z, fz])
            mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[tr], y[tr])
            ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y[tr])
            d_base[te] = llv(y[te], mb.predict_proba(Xb[te])[:, 1])
            d_with[te] = llv(y[te], ms.predict_proba(X[te])[:, 1])
            coefs.append(float(ms.coef_[0][1]))
        d = d_base - d_with                     # positive = feature helps
        rng = np.random.default_rng(7)
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<9} OOF base {d_base.mean():.5f} -> {d_with.mean():.5f}  "
              f"delta {d.mean():+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}  "
              f"coef/fold {' '.join(f'{c:+.4f}' for c in coefs)}")
        res["variants"][name] = {
            "base_ll": round(float(d_base.mean()), 5),
            "with_ll": round(float(d_with.mean()), 5),
            "delta": round(float(d.mean()), 6),
            "ci": [round(float(lo), 6), round(float(hi), 6)], "sig": sig,
            "fold_coefs": [round(c, 4) for c in coefs]}
    with open(REPORT + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(REPORT + ".tmp", REPORT)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--screen"
    if mode == "--build":
        build()
    else:
        screen()
