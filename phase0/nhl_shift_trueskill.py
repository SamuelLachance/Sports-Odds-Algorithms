"""Per-shift participation TrueSkill for NHL skaters — the NFL snap-engine analog.

Shifts (data/nhl_shifts.csv) tell us who is on the ice at every moment; shots
(data/nhl_shots.csv) are the events with an xGoal outcome. So each shot is a
match between the shooting team's on-ice skaters and the defending team's on-ice
skaters, exactly as each NFL snap was an 11v11 match graded by EPA:

  outcome   xGoal thresholded — offense WINS if xg > HI (dangerous chance
            created), defense WINS if xg < LO (chance suppressed), DRAW between.
            Goalie excluded (their save skill is the separate GSAx feature).
  update    closed-form TrueSkill team-vs-team with a draw branch; per-game tau
            drift; season carryover shrink + sigma inflation.

Every skater accrues one two-way rating (offense = create danger, defense =
suppress it, on one latent scale). Aggregated per team pre-game it becomes a
lineup-aware strength feature, tested as an increment over the shipped 0.66418.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from math import erf, exp, pi, sqrt

import numpy as np

sys.path.insert(0, "phase0")

MU0, SIG0 = 25.0, 25.0 / 3.0
SIG0_2 = SIG0 * SIG0
BETA = SIG0
SIG2_FLOOR = 0.5 ** 2
V_CLAMP = 3.0
TAU2 = (SIG0 / 12.0) ** 2          # per-game drift
DRAW_EPS = 0.10
SEASON_SHRINK = 1.0 / 3.0
SEASON_WIDEN2 = 1.5 ** 2
HI, LO = 0.050, 0.020             # xGoal win/loss thresholds, centered on median 0.033


def pdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def v_win(t):
    c = cdf(t)
    return -t if c < 1e-12 else pdf(t) / c


def load_shifts():
    """game_id -> list of (player, team, period, start_s, end_s)."""
    sh = defaultdict(list)
    with open("data/nhl_shifts.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if not r or not r[0].isdigit():
                continue
            sh[int(r[0])].append((int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])))
    return sh


def index_shifts(shifts):
    """(team, period) -> list of (start_s, end_s, player) for fast on-ice lookup."""
    idx = defaultdict(list)
    for (p, t, pe, s0, s1) in shifts:
        idx[(t, pe)].append((s0, s1, p))
    return idx


def on_ice(idx, period, sec, team):
    """Skaters on the ice for `team` at (period, sec)."""
    return [p for (s0, s1, p) in idx.get((team, period), ()) if s0 <= sec < s1]


def main():
    # game meta (date, home, away, season) for chronological walk + team codes
    meta = {}
    for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")):
        meta[int(r["game_id"])] = (r["date"], r["home"], r["away"], int(r["season"]))
    shifts = load_shifts()
    goalies = set()
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        try:
            goalies.add(int(r["goalie_id"]))
        except (KeyError, ValueError):
            pass
    print(f"{len(goalies):,} goalie ids (excluded from skater matches)", flush=True)
    shots = defaultdict(list)
    for r in csv.DictReader(open("data/nhl_shots.csv", encoding="utf-8")):
        shots[int(r["nhl_game_id"])].append(
            (int(r["period"]), int(r["per_sec"]), r["shooter_team"],
             float(r["xg"]), int(r["goal"])))
    games = sorted((g for g in shots if g in meta and g in shifts),
                   key=lambda g: (meta[g][0], g))
    print(f"{len(games):,} games with both shifts and shots", flush=True)

    mu, s2, lastseen = {}, {}, {}
    toi = defaultdict(float)         # running TOI share proxy (shots-on-ice count)
    team_of = {}                     # player -> most-recent team (for lineup lookup)
    prev_season = None
    n_matches = 0
    n_draw = 0
    feat_rows = []                   # (game_id, home_agg, away_agg) pre-game, leak-safe
    for gi, gid in enumerate(games):
        date, home, away, season = meta[gid]
        if prev_season is not None and season != prev_season:
            for p in mu:
                mu[p] = MU0 + (mu[p] - MU0) * (1 - SEASON_SHRINK)
                s2[p] = min(s2[p] + SEASON_WIDEN2, SIG0_2)
        prev_season = season
        gidx = index_shifts(shifts[gid])
        # pre-game team aggregate: TOI-weighted mean of on-ice skaters' pre-game
        # mu (leak-safe — reads ratings before this game's shots update them).
        def team_agg(tm):
            skaters = {p for (t, pe), lst in gidx.items() if t == tm
                       for (_, _, p) in lst if p not in goalies}
            num = den = 0.0
            for p in skaters:
                if p in mu:
                    wt = min(toi.get(p, 0.0), 3000.0) + 50.0   # prior-shrunk weight
                    num += wt * (mu[p] - MU0); den += wt
            return num / den if den else 0.0
        feat_rows.append((gid, round(team_agg(home), 4), round(team_agg(away), 4)))
        for period, sec, steam, xg, goal in shots[gid]:
            dteam = away if steam == home else home
            O = [p for p in on_ice(gidx, period, sec, steam) if p not in goalies]
            D = [p for p in on_ice(gidx, period, sec, dteam) if p not in goalies]
            if not (2 <= len(O) <= 6 and 2 <= len(D) <= 6):
                continue
            for p in O + D:
                if p not in mu:
                    mu[p] = MU0; s2[p] = SIG0_2
                elif lastseen.get(p) != gid:
                    s2[p] = min(s2[p] + TAU2, SIG0_2)
                lastseen[p] = gid
                toi[p] += 1
            sO = sum(mu[p] for p in O); sD = sum(mu[p] for p in D)
            c2 = (len(O) + len(D)) * BETA * BETA + sum(s2[p] for p in O) + sum(s2[p] for p in D)
            c = sqrt(c2)
            t = (sO - sD) / c
            is_draw = LO <= xg <= HI
            n_matches += 1
            if is_draw:
                n_draw += 1
                e = DRAW_EPS
                den = cdf(e - t) - cdf(-e - t)
                if den > 1e-9:
                    vd = (pdf(-e - t) - pdf(e - t)) / den
                    wd = vd * vd + ((e - t) * pdf(e - t) + (e + t) * pdf(-e - t)) / den
                    wd = max(0.0, min(1.0, wd))
                    for p in O:
                        mu[p] += (s2[p] / c) * vd
                        s2[p] = max(s2[p] * (1 - (s2[p] / c2) * wd), SIG2_FLOOR)
                    for p in D:
                        mu[p] -= (s2[p] / c) * vd
                        s2[p] = max(s2[p] * (1 - (s2[p] / c2) * wd), SIG2_FLOOR)
                continue
            off_win = xg > HI
            if off_win:
                win, lose, tt = O, D, t
            else:
                win, lose, tt = D, O, -t
            v = v_win(tt); w = v * (v + tt)
            for p in win:
                mu[p] += (s2[p] / c) * v
                s2[p] = max(s2[p] * (1 - (s2[p] / c2) * w), SIG2_FLOOR)
            for p in lose:
                mu[p] -= (s2[p] / c) * v
                s2[p] = max(s2[p] * (1 - (s2[p] / c2) * w), SIG2_FLOOR)
        if (gi + 1) % 2000 == 0:
            print(f"  {gi+1:,}/{len(games):,} games, {n_matches:,} matches "
                  f"({n_draw/max(n_matches,1):.0%} draws), {len(mu):,} skaters", flush=True)

    # export final ratings (display: mu - 3 sigma z-scored within the pool)
    rated = {p: (mu[p], sqrt(s2[p]), toi[p]) for p in mu if toi[p] >= 200}
    vals = np.array([m - 3 * sqrt(v) for p, (m, sd, tn) in rated.items()
                     for v in [sd * sd]])
    z = (vals - vals.mean()) / (vals.std() + 1e-9)
    out = {str(p): {"mu": round(m, 3), "sigma": round(sd, 3), "n": int(tn),
                    "rating": round(float(50 + 15 * zz), 1)}
           for (p, (m, sd, tn)), zz in zip(rated.items(), z)}
    json.dump(out, open("data/nhl_shift_ratings.json", "w"))
    print(f"\n{len(out):,} rated skaters (>=200 on-ice shots) -> data/nhl_shift_ratings.json")
    top = sorted(out.items(), key=lambda x: -x[1]["rating"])[:10]
    print("top skaters (by shift-TS rating):", [(p, v["rating"]) for p, v in top])
    with open("data/nhl_shift_features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "home_agg", "away_agg"])
        w.writerows(feat_rows)
    print(f"wrote data/nhl_shift_features.csv: {len(feat_rows):,} games")


if __name__ == "__main__":
    main()
