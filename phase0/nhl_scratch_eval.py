"""Scratch/injury absence feature — minutes of usual regulars MISSING tonight.

The last untested NHL channel. Team-level player-rating aggregates were null
(ledger 8: the xG team rating already prices average lineup strength). What the
team rating CANNOT know pre-game is tonight's DEVIATION from the usual lineup:
a star scratched/injured = the team is weaker than its rating says.

Historical dressed rosters need no new pull: every skater with a shift in
data/nhl_shifts.csv was in that night's confirmed pre-game lineup (announced
30-60 min before puck drop — same no-leak argument as the goalie-starter test,
ledger row 9; late warmup scratches are rare noise).

  regular(team, t)  = player with EWMA dressed-share >= 0.5, >= 5 appearances,
                      and trailing TOI in [600s, 2400s] (excludes goalies)
  absent_toi(team)  = sum of trailing TOI/60 of regulars NOT dressed tonight
  feature           = away_absent - home_absent   (positive = home stronger)
  variant           = n_missing_regulars diff

First 8 team-games of each season are masked (roster churn makes "usual
lineup" unknowable there via this proxy). All state updates happen AFTER the
feature is read (leak-safe). One TEST look -> ledger row 10.

Usage: --build (cache dressed table)  --stats (feature stats, no eval)
       --eval (THE one look)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "phase0")

DRESSED = "data/nhl_dressed.csv"
SH_DECAY = 0.90        # dressed-share EWMA (per team-game)
TOI_DECAY = 0.80       # trailing TOI EWMA (updated only when dressed)
MIN_SHARE = 0.5
MIN_SEEN = 5
TOI_LO, TOI_HI = 600.0, 2400.0
SEASON_MASK = 8        # first N team-games each season: feature = NaN
SEASON_SHARE_SHRINK = 0.5


def build_dressed():
    """One pass over the 536MB shifts table -> per-game per-player TOI cache."""
    toi = defaultdict(float)                      # (gid, team, pid) -> seconds
    with open("data/nhl_shifts.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            if not row or not row[0].isdigit():
                continue
            try:
                dur = int(row[5]) - int(row[4])
            except ValueError:
                continue
            if dur > 0:
                toi[(int(row[0]), row[2], row[1])] += dur
    with open(DRESSED + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "team", "player_id", "toi_s"])
        for (gid, team, pid), s in sorted(toi.items()):
            w.writerow([gid, team, pid, int(s)])
    os.replace(DRESSED + ".tmp", DRESSED)
    print(f"wrote {DRESSED}: {len(toi):,} player-games")


def build_features_scratch(games):
    import numpy as np
    rows = defaultdict(dict)                      # gid -> team -> {pid: toi}
    for r in csv.DictReader(open(DRESSED, encoding="utf-8")):
        rows[int(r["game_id"])].setdefault(r["team"], {})[r["player_id"]] = float(r["toi_s"])

    share = defaultdict(dict)                     # team -> pid -> EWMA dressed
    ptoi = defaultdict(dict)                      # team -> pid -> EWMA toi
    seen = defaultdict(lambda: defaultdict(int))  # team -> pid -> appearances
    tgame = defaultdict(int)                      # team -> games this season
    prev_season = None
    absent = np.full(len(games), np.nan)
    nmiss = np.full(len(games), np.nan)

    for i, g in enumerate(games):
        if prev_season is not None and g["season"] != prev_season:
            for t in share:
                for pid in share[t]:
                    share[t][pid] *= SEASON_SHARE_SHRINK
            tgame.clear()
        prev_season = g["season"]

        vals = {}
        for side in ("home", "away"):
            t = g[side]
            dressed = rows.get(g["game_id"], {}).get(t, {})
            if not dressed or tgame[t] < SEASON_MASK:
                vals[side] = (np.nan, np.nan)
            else:
                a_toi, a_n = 0.0, 0
                for pid, sh in share[t].items():
                    tv = ptoi[t].get(pid, 0.0)
                    if (sh >= MIN_SHARE and seen[t][pid] >= MIN_SEEN
                            and TOI_LO <= tv <= TOI_HI and pid not in dressed):
                        a_toi += tv / 60.0
                        a_n += 1
                vals[side] = (a_toi, a_n)

        if not (np.isnan(vals["home"][0]) or np.isnan(vals["away"][0])):
            absent[i] = vals["away"][0] - vals["home"][0]
            nmiss[i] = vals["away"][1] - vals["home"][1]

        # ---- update AFTER reading (leak-safe) ----
        for side in ("home", "away"):
            t = g[side]
            dressed = rows.get(g["game_id"], {}).get(t, {})
            if not dressed:
                continue                          # no shift data: state untouched
            tgame[t] += 1
            for pid in set(share[t]) | set(dressed):
                d = 1.0 if pid in dressed else 0.0
                share[t][pid] = SH_DECAY * share[t].get(pid, 0.0) + (1 - SH_DECAY) * d
            for pid, s_ in dressed.items():
                ptoi[t][pid] = TOI_DECAY * ptoi[t].get(pid, s_) + (1 - TOI_DECAY) * s_
                seen[t][pid] += 1
    return absent, nmiss


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--stats"
    if mode == "--build":
        build_dressed()
        return

    import numpy as np
    from nhl_glicko2_eval import DEV_END, DEV_WARM_BEFORE, TEST_END, TEST_START, llv, load_games, run_elo
    games = load_games()
    absent, nmiss = build_features_scratch(games)
    seas = np.array([g["season"] for g in games])
    have = ~np.isnan(absent)
    print(f"coverage {have.mean():.1%}  n={int(have.sum())}")
    print(f"absent_toi diff: std {np.nanstd(absent):.2f} min, "
          f"share |absent|>10min: {np.mean(np.abs(absent[have]) > 10):.1%}")
    print(f"n_missing diff:  std {np.nanstd(nmiss):.2f}, "
          f"share any missing regular: "
          f"{np.mean(np.abs(nmiss[have]) > 0):.1%}")
    if mode != "--eval":
        return

    import json as _json
    from sklearn.linear_model import LogisticRegression
    from nhl_features_eval import build_features
    model = _json.load(open("data/nhl_model.json"))
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    y = np.array([g["y"] for g in games])
    b = model["blend"]; c = b["coefs"]
    base_z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
              + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
              + c["xg"] * np.nan_to_num(F["xg_diff"]))
    dev = have & (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    test = have & (seas >= TEST_START) & (seas <= TEST_END)

    def run(name, feat):
        Xb = base_z.reshape(-1, 1)
        X = np.column_stack([base_z, np.nan_to_num(feat)])
        mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[dev], y[dev])
        ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[dev], y[dev])
        p0 = mb.predict_proba(Xb[test])[:, 1]; p1 = ms.predict_proba(X[test])[:, 1]
        b_ll = float(llv(y[test], p0).mean()); s_ll = float(llv(y[test], p1).mean())
        d = llv(y[test], p0) - llv(y[test], p1)
        rng = np.random.default_rng(7)
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<24} base {b_ll:.5f} -> {s_ll:.5f}  delta {b_ll-s_ll:+.5f} "
              f"CI [{lo:+.5f},{hi:+.5f}] {sig}  coef {ms.coef_[0][1]:+.4f}")
        return {"base": round(b_ll, 5), "with": round(s_ll, 5),
                "delta": round(b_ll - s_ll, 5),
                "ci": [round(float(lo), 5), round(float(hi), 5)], "sig": sig,
                "coef": round(float(ms.coef_[0][1]), 4)}

    print(f"\nscratch-absence increment over shipped model (TEST n={int(test.sum())}):")
    res = {"n_test": int(test.sum()), "coverage": round(float(have.mean()), 4)}
    res["absent_toi"] = run("absent regulars TOI", absent)
    res["n_missing"] = run("n missing regulars", nmiss)
    _json.dump(res, open("data/nhl_scratch_report.json", "w"), indent=1)
    print("wrote data/nhl_scratch_report.json")


if __name__ == "__main__":
    main()
