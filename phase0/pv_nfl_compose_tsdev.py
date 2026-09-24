"""Player-value program (pv), NFL, COMPOSE comparator: a DEV analog of the shipped 11v11 TrueSkill. DEV ONLY.

The shipped column 7 (v7 11v11 participation TrueSkill) is identically 0 on every DEV game: its
participation lists start in 2016 (TEST). To compare the composed player values against "the
current roster rating" on DEV, this script builds the closest thing DEV data allows -- the same
shared-credit logic (everyone on the field shares each unit's EPA outcome, opponent-adjusted,
walk-forward), aggregated to the game:

  observation per team-game side: y = the offence's EPA per scrimmage play, n plays
  model: y = mu_league + sum_i (f_i/11) a_i - sum_j (f_j/11) d_j + e,  Var(e) = sigma2 / n
  participants and weights f (sum ~ 11 per side):
      2013-2015: the snap table, f = offense_pct / defense_pct (true snap-share participation)
      1999-2012: no participation data -> the players CREDITED on that side of the ball (passer,
                 targets, ball carriers / any defensive credit or penalty), equal shares 11/n
  update: Gaussian assumed-density filter (the continuous-outcome TrueSkill), each player takes a
          Kalman share of the surprise; per-appearance drift tau2; at a player's first game of a
          season mu *= (1-rs) and his variance widens back toward v0 by the same factor.
  pre-game roster rating of team t in game g (the "roster quality" feature) uses EXACTLY the
          lineups of pv_nfl_compose_build (data/pv_nfl_compose_lineups.parquet): presence-weighted
          mean offence rating + presence-weighted mean defence rating (EPA/play net), in the
          LINEUP variant (2013+: absent regulars removed) and the EXPECTED variant.
Differences from v7 (disclosed): per-game not per-play, Gaussian not win/draw/loss, no weekly QB
fusion, no salary prior (contracts start 2016), credited-player membership before 2013.

Hyper-parameters (v0, tau2, rs) are chosen by the filter's own one-step-ahead predictive
log-likelihood of y on the WARM-UP seasons 2001-2005 (burn-in 1999-2000), never on DEV outcomes,
never on game results; sigma2 = per-play EPA variance of the warm-up seasons.
TEST discipline: all reads filtered to season < 2016 and asserted; nothing about 2016+ is touched.
Output: data/pv_nfl_compose_tsdev_team_games.parquet, data/pv_nfl_compose_tsdev_params.json
"""
from __future__ import annotations

import itertools
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA = 2016
SNAP_ERA = 2013
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
ETA = 0.05          # league-mean EWMA per game date


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def fr(t):
    return FR.get(t, t)


def load():
    cols = ["game_id", "season", "game_date", "home_team", "away_team", "posteam", "defteam", "play_type",
            "epa", "two_point_attempt", "qb_kneel", "qb_spike", "passer_id", "target_player_id", "rusher_id"]
    ev = pq.read_table(os.path.join(DATA, "pv_nfl_events.parquet"), columns=cols,
                       filters=[("season", "<", TEST_ERA)]).to_pandas()
    assert ev.season.max() < TEST_ERA
    for c in ("home_team", "away_team", "posteam", "defteam"):
        ev[c] = ev[c].astype(object).map(fr)
    s = ev[ev.play_type.isin(["pass", "run"]) & (ev.two_point_attempt.fillna(0) != 1)
           & (ev.qb_kneel.fillna(0) != 1) & (ev.qb_spike.fillna(0) != 1) & ev.epa.notna()]
    side = s.groupby(["game_id", "posteam"]).agg(y=("epa", "mean"), n=("epa", "size"),
                                                  opp=("defteam", "first")).reset_index()
    sig2 = float(s[s.season.between(1999, 2005)].epa.var())
    off = defaultdict(set)
    for c in ("passer_id", "target_player_id", "rusher_id"):
        sub = s[s[c].notna()][["game_id", "posteam", c]]
        for g, t, p in sub.itertuples(index=False):
            off[(g, t)].add(p)
    games = ev.drop_duplicates("game_id")[["game_id", "season", "game_date", "home_team", "away_team"]]
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    PRC = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_credits.parquet"),
                          columns=["game_id", "team", "player_id", "season"])
    PRC = PRC[PRC.season < TEST_ERA]
    assert PRC.season.max() < TEST_ERA
    dfn = defaultdict(set)
    for g, t, p in PRC[["game_id", "team", "player_id"]].itertuples(index=False):
        dfn[(g, fr(t))].add(p)
    CV = pd.read_parquet(os.path.join(DATA, "pv_nfl_coverage_player_games.parquet"),
                         columns=["game_id", "season", "team", "player_id"])
    CV = CV[CV.season < TEST_ERA]
    assert CV.season.max() < TEST_ERA
    for g, t, p in CV[["game_id", "team", "player_id"]].itertuples(index=False):
        dfn[(g, fr(t))].add(p)
    SN = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_snaps.parquet"))
    SN = SN[SN.season < TEST_ERA]
    assert SN.season.max() < TEST_ERA
    snO, snD = defaultdict(dict), defaultdict(dict)
    for g, t, p, op, dp in SN[["game_id", "team", "player_id", "offense_pct", "defense_pct"]].itertuples(index=False):
        t = fr(t)
        if (op or 0) > 0:
            snO[(g, t)][p] = float(op)
        if (dp or 0) > 0:
            snD[(g, t)][p] = float(dp)
    return games, side, sig2, off, dfn, snO, snD


def participants(g, t, o, season, off, dfn, snO, snD):
    """(offence {pid: f}, defence {pid: f}) for t's offence vs o's defence."""
    if season >= SNAP_ERA and (g, t) in snO and (g, o) in snD:
        return snO[(g, t)], snD[(g, o)]
    O = off.get((g, t), set()); D = dfn.get((g, o), set())
    return ({p: 11.0 / len(O) for p in O} if O else {}), ({p: 11.0 / len(D) for p in D} if D else {})


def run(params, games, side, sig2, off, dfn, snO, snD, lineups=None, score=(2001, 2005), stop=None):
    v0, tau2, rs = params
    A = {}; Dd = {}                      # pid -> [mu, var, last_season]
    mu_lg = 0.0
    ll, n_ll = 0.0, 0
    sided = {(g, t): (y, n, o) for g, t, y, n, o in side[["game_id", "posteam", "y", "n", "opp"]].itertuples(index=False)}
    out = []
    for date, dg in games.groupby("game_date", sort=True):
        yr = int(dg.season.iat[0])
        if stop is not None and yr > stop:
            break
        # ---- pre-game roster ratings (read before any update of this date)
        if lineups is not None:
            for g in dg.itertuples(index=False):
                for sd, t in (("home", g.home_team), ("away", g.away_team)):
                    rows = lineups.get((g.game_id, t), ())
                    res = {}
                    for var in ("lin", "exp"):
                        so = sw_o = sd_ = sw_d = 0.0
                        for p, a, absent, role in rows:
                            w = 0.0 if (var == "lin" and absent) else a
                            if role == "O":
                                st = A.get(p); so += w * (st[0] if st else 0.0); sw_o += w
                            elif role == "D":
                                st = Dd.get(p); sd_ += w * (st[0] if st else 0.0); sw_d += w
                        res[var] = (so / sw_o if sw_o else 0.0, sd_ / sw_d if sw_d else 0.0)
                    out.append((g.game_id, yr, t, sd, res["lin"][0], res["lin"][1], res["exp"][0], res["exp"][1]))
        # ---- updates
        ys = []
        for g in dg.itertuples(index=False):
            for t, o in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
                ob = sided.get((g.game_id, t))
                if ob is None:
                    continue
                y, n, _ = ob
                P, Q = participants(g.game_id, t, o, yr, off, dfn, snO, snD)
                if not P or not Q:
                    ys.append(y)
                    continue
                sts = []
                for tbl, pl, sgn in ((A, P, 1.0), (Dd, Q, -1.0)):
                    for p, f in pl.items():
                        st = tbl.get(p)
                        if st is None:
                            st = [0.0, v0, yr]; tbl[p] = st
                        elif st[2] != yr:
                            st[0] *= (1 - rs); st[1] = (1 - rs) ** 2 * st[1] + (1 - (1 - rs) ** 2) * v0; st[2] = yr
                        st[1] += tau2
                        sts.append((st, sgn * f / 11.0))
                pred = mu_lg + sum(c * st[0] for st, c in sts)
                S = sig2 / n + sum(c * c * st[1] for st, c in sts)
                r = y - pred
                if score[0] <= yr <= score[1]:
                    ll += -0.5 * np.log(2 * np.pi * S) - r * r / (2 * S); n_ll += 1
                for st, c in sts:
                    k = c * st[1] / S
                    st[0] += k * r
                    st[1] -= c * c * st[1] * st[1] / S
                ys.append(y)
        if ys:
            mu_lg += ETA * (float(np.mean(ys)) - mu_lg)
    return (ll / max(n_ll, 1)), n_ll, out


def main():
    games, side, sig2, off, dfn, snO, snD = load()
    log(f"games {len(games)}; sigma2 per play (warm-up) {sig2:.4f}")
    grid = list(itertools.product([0.02, 0.05, 0.1, 0.2, 0.4], [0.0, 0.0005, 0.002], [0.25, 0.5]))
    res = []
    for p in grid:
        ll, n, _ = run(p, games, side, sig2, off, dfn, snO, snD, stop=2005)
        res.append((ll, p)); log(f"  v0 {p[0]:<5} tau2 {p[1]:<7} rs {p[2]:<4} warm-up mean pred LL {ll:.5f} (n {n})")
    best = max(res)[1]
    log("best warm-up params", best)
    EM = pd.read_parquet(os.path.join(DATA, "pv_nfl_compose_lineups.parquet"))
    assert EM.season.max() < TEST_ERA
    lineups = defaultdict(list)
    for g, t, p, a, ab, rl in EM[["game_id", "team", "player_id", "a", "absent", "role"]].itertuples(index=False):
        lineups[(g, t)].append((p, a, ab, rl))
    _, _, out = run(best, games, side, sig2, off, dfn, snO, snD, lineups=lineups)
    T = pd.DataFrame(out, columns=["game_id", "season", "team", "side", "ts_off_lin", "ts_def_lin",
                                   "ts_off_exp", "ts_def_exp"])
    assert T.season.max() < TEST_ERA
    T["ts_lin"] = T.ts_off_lin + T.ts_def_lin
    T["ts_exp"] = T.ts_off_exp + T.ts_def_exp
    T.to_parquet(os.path.join(DATA, "pv_nfl_compose_tsdev_team_games.parquet"), index=False)
    json.dump({"sigma2_per_play": sig2, "grid_warmup_mean_pred_ll": [[list(p), ll] for ll, p in res],
               "best": {"v0": best[0], "tau2": best[1], "rs": best[2]}, "eta_league": ETA,
               "selection": "one-step-ahead predictive LL of team-game offensive EPA/play, warm-up 2001-2005"},
              open(os.path.join(DATA, "pv_nfl_compose_tsdev_params.json"), "w"), indent=1)
    log(f"wrote data/pv_nfl_compose_tsdev_team_games.parquet ({len(T)} team-games)")


if __name__ == "__main__":
    main()
