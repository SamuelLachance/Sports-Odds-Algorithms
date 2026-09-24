"""Player-value program (pv), NFL, COMPOSE step 1: one value per player, lineup aggregates. DEV ONLY.

Combines the four component ratings (passing, receiving_rushing, pass_rush_and_front, coverage)
into per-unit lineup aggregates in EPA per game, and from them (step 2, weights) one value per
player in points per game. Every input is a PRE-GAME walk-forward value written by a component
build; nothing here looks at a game's own outcome except as the regression target of step 2.

UNITS (each in EPA per game, "as if every credited event were worth its linear-weight EPA"):
  QB    starter's passing composite (EPA/dropback above league, opponent-neutral) x league dropbacks
  REC   usage-weighted mean receiver rating v_e (EPA/target above average: opportunity quality +
        execution, QB- and defence-adjusted) of the lineup x league targets per game
  RUSH  usage-weighted mean rushing rating (EPA/carry over expected, line+defence-adjusted) x carries
  PROT  offence protection unit (QB+OL; nobody records which lineman lost): sacks, non-sack hits and
        run stops ALLOWED relative to an average unit, x league rates x |EPA per event|
  PR    pass-rush production of the lineup's defenders: sum theta_sack*L_sk*DB*|e_sack|
        + theta_hit*L_ht*DB*|e_hit|  (theta = share of an average unit's production)
  RF    run-front production: sum theta_stop * L_st * runs * |e_stop|
  BALL  ball / penalty production of the lineup: sum_k v_k * mu_k(group) * theta_k * attempts,
        k in {pass defensed, interception, coverage penalty} (coverage linear weights, warm-up)
  COV   EPA saved on attributed targets: sum cv_eff * attempts
League volumes (dropbacks, attempts, designed runs per team-game) and the event EPA constants are
AS-OF: means of seasons strictly before the game's season. Exception, disclosed: the non-sack QB
hit constant for 2006 uses 2006 plays (hits are not logged before 2006); it is a unit conversion
that step 2's weight regression re-scales anyway.

LINEUPS (who is counted for a team in a game):
  expected roster E(t,g): every player with pre-game presence EWMA a >= 0.05 for team t
      (a <- 0.5*a + 0.5*member each team game, updated AFTER the game). member = credited with
      an event in the game (dropback, target, carry, any defensive credit or penalty) before 2013;
      in the snap table (offense_pct or defense_pct > 0) from 2013 (snap files start 2013 on DEV;
      data/snap_2012.csv is empty).
  LINEUP variant (the emitted feature):
      2013-2015 (snap era): players who actually played. Every E member keeps weight a, except a
          REGULAR (presence a >= 0.5 AND pre-game snap share when active >= 0.5, EWMA 0.8 per
          appearance) who is ABSENT from the game's snap table: weight 0, and his vacated
          production is refilled at his bucket's new-player (prior) level. Membership only; the
          realized in-game snap % is never used as a weight (it is post-game: blowout backups,
          in-game injuries). Non-regulars keep their expected weight whether or not they played,
          so garbage-time backups cannot leak the score into the lineup.
      1999-2012 (no snap data): the depth/usage proxy = the expected roster E (no day-of info).
  EXPECTED variant: E with weight a in every season (no day-of information at all).
  QB: always the actual starter (first dropback passer; announced pre-game).

Per-player component values used for a game are the component rows OF THAT GAME (pre-game states)
when the player has one, else the player's most recent earlier pre-game row (stale by at most his
last appearance's update; never a later row -- later rows can embed later-fit weights).

TEST discipline: every input is filtered to season < 2016 at read time and asserted; nothing about
any season >= 2016 is read, computed or printed. No odds column is read (nfl_games.csv is read with
an explicit usecols list of identity + score columns).

Outputs:
  data/pv_nfl_compose_team_games.parquet   one row per team-game: unit aggregates (lineup and
                                           expected variants), starter, absence bookkeeping
  data/pv_nfl_compose_player_games.parquet one row per (team-game, lineup member): per-unit
                                           contributions (EPA/game, pre-game) and weights
  data/pv_nfl_compose_lineups.parquet      every expected-roster member per team-game: pre-game
                                           presence a, day-of absence flag (2013+), role O/D
  data/pv_nfl_compose_constants.json       as-of league volumes and EPA constants by season
"""
from __future__ import annotations

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
RHO = 0.5            # presence EWMA per team game (the pass-rush engine's rho)
A_MIN = 0.05         # expected-roster floor
REG = 0.5            # regular threshold for the day-of absence check (presence EWMA) ...
SHARE_REG = 0.5      # ... AND pre-game snap share when active >= 0.5 (plays at least half the snaps)
SH_DECAY = 0.80      # snap-share EWMA per appearance (the shipped snap-absence decay, rounded)
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
UNITS = ["QB", "REC", "RUSH", "PROT", "PR", "RF", "BALL", "COV"]


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def fr(t):
    return FR.get(t, t)


# ============================================================================ load ==========
def load_events():
    cols = ["game_id", "season", "season_type", "week", "game_date", "home_team", "away_team",
            "posteam", "defteam", "play_type", "epa", "qb_dropback", "qb_scramble", "qb_kneel",
            "qb_spike", "sack", "qb_hit", "two_point_attempt", "down", "ydstogo", "yards_gained",
            "touchdown", "passer_id", "target_player_id", "rusher_id"]
    ev = pq.read_table(os.path.join(DATA, "pv_nfl_events.parquet"), columns=cols,
                       filters=[("season", "<", TEST_ERA)]).to_pandas()
    assert ev.season.max() < TEST_ERA
    for c in ("home_team", "away_team", "posteam", "defteam"):
        ev[c] = ev[c].astype(object).map(fr)
    return ev


def team_volumes(ev):
    s = ev[ev.play_type.isin(["pass", "run"]) & (ev.two_point_attempt.fillna(0) != 1)].copy()
    s["is_db"] = ((s.qb_dropback == 1) & (s.qb_spike.fillna(0) != 1)).astype(int)
    s["is_att"] = ((s.play_type == "pass") & (s.sack.fillna(0) != 1) & (s.qb_spike.fillna(0) != 1)).astype(int)
    s["is_run"] = ((s.play_type == "run") & (s.qb_scramble.fillna(0) != 1) & (s.qb_kneel.fillna(0) != 1)).astype(int)
    s["is_sack"] = ((s.sack == 1) & (s.is_db == 1)).astype(int)
    s["is_hit_ns"] = ((s.qb_hit == 1) & (s.sack != 1) & (s.is_db == 1)).astype(int)
    yg = s.yards_gained.fillna(0).to_numpy(float)
    togo = s.ydstogo.fillna(10).to_numpy(float)
    dn = s.down.fillna(1).to_numpy(float)
    need = np.where(dn == 1, 0.4 * togo, np.where(dn == 2, 0.6 * togo, togo))
    ok = (yg >= need) | (s.touchdown.fillna(0).to_numpy() == 1)
    s["is_stop"] = ((s.is_run == 1) & ~ok).astype(int)
    s["is_scrim"] = ((s.qb_kneel.fillna(0) != 1) & (s.qb_spike.fillna(0) != 1)).astype(int)
    s["epa0"] = s.epa.fillna(0.0) * s.is_scrim
    tg = s.groupby(["game_id", "posteam"]).agg(
        season=("season", "first"), n_db=("is_db", "sum"), n_att=("is_att", "sum"),
        n_run=("is_run", "sum"), n_play=("is_scrim", "sum"), epa_off=("epa0", "sum")).reset_index()
    return s, tg


def event_constants(s, seasons):
    """EPA constants per season from strictly earlier seasons (hits: 2006+; 2006 uses itself)."""
    by = {}
    for yr, d in s.groupby("season"):
        db = d[d.is_db == 1]
        runs = d[d.is_run == 1]
        nsdb = db[db.is_sack == 0]
        by[int(yr)] = {
            "sack_sum": float(db.epa[db.is_sack == 1].sum()), "sack_n": int((db.is_sack == 1).sum()),
            "db_sum": float(db.epa.sum()), "db_n": int(len(db)),
            "hit_sum": float(nsdb.epa[nsdb.is_hit_ns == 1].sum()), "hit_n": int((nsdb.is_hit_ns == 1).sum()),
            "nsdb_sum": float(nsdb.epa.sum()), "nsdb_n": int(len(nsdb)),
            "stop_sum": float(runs.epa[runs.is_stop == 1].sum()), "stop_n": int((runs.is_stop == 1).sum()),
            "run_sum": float(runs.epa.sum()), "run_n": int(len(runs)),
        }
    out = {}
    for yr in seasons:
        prior = [y for y in by if y < yr] or [yr]
        hprior = [y for y in by if y < yr and y >= 2006 and by[y]["hit_n"] > 0] or [max(yr, 2006)]

        def m(keys, ys):
            a, b = keys
            return sum(by[y][a] for y in ys) / max(sum(by[y][b] for y in ys), 1)
        e_sack = m(("sack_sum", "sack_n"), prior) - m(("db_sum", "db_n"), prior)
        # before 2006 hits are not logged (L_ht is NaN there and the hit terms are zeroed downstream)
        e_hit = m(("hit_sum", "hit_n"), hprior) - m(("nsdb_sum", "nsdb_n"), hprior)
        e_stop = m(("stop_sum", "stop_n"), prior) - m(("run_sum", "run_n"), prior)
        out[yr] = {"e_sack": e_sack, "e_hit": e_hit, "e_stop": e_stop,
                   "hit_from": f"{min(hprior)}-{max(hprior)}", "from": f"{min(prior)}-{max(prior)}"}
    return out


def league_volumes(tg, seasons):
    """as-of per team-game volumes: mean of the 3 seasons strictly before (first season: itself)."""
    m = tg.groupby("season")[["n_db", "n_att", "n_run", "n_play"]].mean()
    out = {}
    for yr in seasons:
        prior = [y for y in m.index if yr - 3 <= y < yr] or [yr]
        out[yr] = {k: float(m.loc[prior, k].mean()) for k in ("n_db", "n_att", "n_run", "n_play")}
    return out


def main():
    ev = load_events()
    s, tg = team_volumes(ev)
    games = ev.drop_duplicates("game_id")[["game_id", "season", "season_type", "week", "game_date",
                                             "home_team", "away_team"]].copy()
    games["season"] = games.season.astype(int)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(games.season.unique().tolist())
    assert max(seasons) < TEST_ERA
    sc = pd.read_csv(os.path.join(DATA, "nfl_games.csv"),
                     usecols=["game_id", "season", "home_score", "away_score", "location"])
    sc = sc[sc.season < TEST_ERA]
    assert sc.season.max() < TEST_ERA
    games = games.merge(sc[["game_id", "home_score", "away_score", "location"]], on="game_id", how="left")
    assert games.home_score.notna().all()
    K = event_constants(s, seasons)
    LV = league_volumes(tg, seasons)
    log(f"spine {len(games)} games {seasons[0]}-{seasons[-1]}; e_sack(2006) {K[2006]['e_sack']:.3f} "
        f"e_hit(2006) {K[2006]['e_hit']:.3f} e_stop(2006) {K[2006]['e_stop']:.3f}; "
        f"DB(2006) {LV[2006]['n_db']:.1f} ATT {LV[2006]['n_att']:.1f} RUN {LV[2006]['n_run']:.1f}")
    vol = {(g, t): (a, b, c, d, e) for g, t, a, b, c, d, e in
           tg[["game_id", "posteam", "n_db", "n_att", "n_run", "n_play", "epa_off"]].itertuples(index=False)}

    # ------------------------------------------------------------------ component tables
    PV = pd.read_csv(os.path.join(DATA, "pv_nfl_passing_values.csv"))
    PV = PV[PV.season < TEST_ERA]
    assert PV.season.max() < TEST_ERA
    PV["team"] = PV.team.map(fr)
    PV["comp_f"] = PV.passing_composite.fillna(PV.epa_q)   # 1999 (burn-in) has no composite
    starter = {}
    qb_rows = {}
    for r in PV.itertuples(index=False):
        qb_rows[(r.game_id, r.qb_id)] = r.comp_f
        if r.is_starter == 1:
            starter[(r.game_id, r.team)] = (r.qb_id, r.comp_f)
    qb_members = PV.groupby(["game_id", "team"]).qb_id.apply(list).to_dict()

    RR = pd.read_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_values.parquet"),
                         columns=["game_id", "season", "player_id", "team", "pre_tsh", "pre_csh",
                                  "mu_xepat", "mu_epat", "mu_epar", "pre_games"])
    RR = RR[RR.season < TEST_ERA]
    assert RR.season.max() < TEST_ERA
    RR["team"] = RR.team.astype(object).map(fr)
    RR["v_e"] = RR.mu_xepat + RR.mu_epat
    rr_rows = {(g, p): (a, b, c, d) for g, p, a, b, c, d in
               RR[["game_id", "player_id", "pre_tsh", "pre_csh", "v_e", "mu_epar"]].itertuples(index=False)}
    rr_members = RR.groupby(["game_id", "team"]).player_id.apply(list).to_dict()

    PR = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_player_games.parquet"),
                         columns=["game_id", "season", "team", "player_id", "bucket", "presence",
                                  "th_sk", "th_ht", "th_st", "n_games"])
    PR = PR[PR.season < TEST_ERA]
    assert PR.season.max() < TEST_ERA
    pr_rows = {(g, p): (a, b, c) for g, p, a, b, c in
               PR[["game_id", "player_id", "th_sk", "th_ht", "th_st"]].itertuples(index=False)}
    bucket = dict(zip(PR.player_id, PR.bucket))
    # new-player (prior) production by bucket and season, as-of: first-game rows of strictly earlier seasons
    newp = PR[PR.n_games == 0].groupby(["season", "bucket"])[["th_sk", "th_ht", "th_st"]].mean()
    PRC = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_credits.parquet"),
                          columns=["game_id", "team", "player_id", "season"])
    PRC = PRC[PRC.season < TEST_ERA]
    assert PRC.season.max() < TEST_ERA
    def_credit_members = PRC.groupby(["game_id", "team"]).player_id.apply(set).to_dict()
    TGF = pd.read_csv(os.path.join(DATA, "pv_nfl_pass_rush_and_front_team_games.csv"))
    TGF = TGF[TGF.season < TEST_ERA]
    assert TGF.season.max() < TEST_ERA
    prot = {(g, fr(t)): (a, b, c, la, lb, lc) for g, t, a, b, c, la, lb, lc in
            TGF[["game_id", "posteam", "Oopp_sk", "Oopp_ht", "Oopp_st", "L_sk", "L_ht", "L_st"]].itertuples(index=False)}

    CV = pd.read_parquet(os.path.join(DATA, "pv_nfl_coverage_player_games.parquet"),
                         columns=["game_id", "season", "team", "player_id", "group", "att", "pre_th_pd",
                                  "pre_th_int", "pre_th_cpen", "pre_cv_eff", "e_pd", "e_int", "e_cpen"])
    CV = CV[CV.season < TEST_ERA]
    assert CV.season.max() < TEST_ERA
    CV["team"] = CV.team.astype(object).map(fr)
    cv_rows = {(g, p): (a, b, c, d) for g, p, a, b, c, d in
               CV[["game_id", "player_id", "pre_th_pd", "pre_th_int", "pre_th_cpen", "pre_cv_eff"]].itertuples(index=False)}
    cgroup = dict(zip(CV.player_id, CV.group))
    for (g, t), ids in CV.groupby(["game_id", "team"]).player_id.apply(set).to_dict().items():
        def_credit_members.setdefault((g, t), set()).update(ids)
    # per-player-present expected rate per opponent attempt, by group, as-of (seasons strictly before)
    cvs = CV[CV.att > 0].groupby(["season", "group"])[["e_pd", "e_int", "e_cpen", "att"]].sum()
    LW = json.load(open(os.path.join(DATA, "pv_nfl_coverage_priors.json")))["linear_weights_defence_positive"]

    def mu_rate(season, grp):
        prior = [y for y in range(season - 3, season) if (y, grp) in cvs.index] or \
                [y for y in range(1999, season + 1) if (y, grp) in cvs.index][:1]
        if not prior:
            return (0.0, 0.0, 0.0)
        tot = cvs.loc[[(y, grp) for y in prior]].sum()
        return (tot.e_pd / tot.att, tot.e_int / tot.att, tot.e_cpen / tot.att)
    MU = {(yr, gp): mu_rate(yr, gp) for yr in seasons for gp in ("DB", "LB", "DL")}

    def rep_th(season, b):
        prior = [(y, b) for y in range(1999, season) if (y, b) in newp.index] or \
                [(y, b) for y in range(1999, season + 1) if (y, b) in newp.index][:1]
        if not prior:
            return (0.0, 0.0, 0.0)
        return tuple(newp.loc[prior].mean().to_numpy())
    buckets = sorted(PR.bucket.dropna().unique())
    REP = {(yr, b): rep_th(yr, b) for yr in seasons for b in buckets}

    SN = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_snaps.parquet"))
    SN = SN[SN.season < TEST_ERA]
    assert SN.season.max() < TEST_ERA and SN.season.min() >= SNAP_ERA
    SN["team"] = SN.team.astype(object).map(fr)
    snap_members = {}
    snap_pct = {}
    for g, t, p, op, dp in SN[["game_id", "team", "player_id", "offense_pct", "defense_pct"]].itertuples(index=False):
        if (op or 0) > 0 or (dp or 0) > 0:
            snap_members.setdefault((g, t), set()).add(p)
            snap_pct[(g, p)] = (float(op or 0), float(dp or 0))
    TEAMS_OF = {g: (h, a) for g, h, a in games[["game_id", "home_team", "away_team"]].itertuples(index=False)}
    RR_BY_GAME = RR.groupby("game_id").player_id.apply(list).to_dict()
    PR_BY_GAME = PR.groupby("game_id").player_id.apply(list).to_dict()
    CV_BY_GAME = CV.groupby("game_id").player_id.apply(list).to_dict()
    log(f"components: QB rows {len(PV)}, rec/rush {len(RR)}, front {len(PR)}, coverage {len(CV)}, "
        f"snap team-games {len(snap_members)}")

    # ------------------------------------------------------------------ walk
    last_rr, last_pr, last_cv = {}, {}, {}
    snap_sh = {}                    # pid -> [decayed off pct, decayed def pct, decayed count] over appearances
    pres = defaultdict(dict)        # team -> {pid: a}
    team_of = {}
    trows, prow, erows = [], [], []
    role = {}                       # pid -> 'O' / 'D' (side of the ball of his latest membership)
    for date, dg in games.groupby("game_date", sort=True):
        date_updates = []
        for g in dg.itertuples(index=False):
            yr = int(g.season)
            k = K[yr]; lv = LV[yr]
            DBb, ATTb, RUNb = lv["n_db"], lv["n_att"], lv["n_run"]
            c_sk_base = DBb * abs(k["e_sack"]); c_ht_base = DBb * abs(k["e_hit"]); c_st_base = RUNb * abs(k["e_stop"])
            for side, t, o in (("home", g.home_team, g.away_team), ("away", g.away_team, g.home_team)):
                E = {p: a for p, a in pres[t].items() if a >= A_MIN}
                snap_era = yr >= SNAP_ERA and (g.game_id, t) in snap_members
                played = snap_members.get((g.game_id, t), set()) if snap_era else None
                pf = prot.get((g.game_id, t))
                L_sk, L_ht, L_st = (pf[3], pf[4], pf[5]) if pf else (0.06, 0.068, 0.548)
                # NaN league rates: hits before 2006, every rate in the first 1999 weeks (burn-in)
                L_sk, L_ht, L_st = (x if x == x else 0.0 for x in (L_sk, L_ht, L_st))
                st = starter.get((g.game_id, t))
                qb_id, qb_comp = st if st else (None, 0.0)
                acc = {v: {u: 0.0 for u in UNITS} for v in ("lin", "exp")}
                rec_num = {"lin": 0.0, "exp": 0.0}; rec_den = {"lin": 0.0, "exp": 0.0}
                rsh_num = {"lin": 0.0, "exp": 0.0}; rsh_den = {"lin": 0.0, "exp": 0.0}
                n_abs = 0; abs_a = 0.0
                for p, a in E.items():
                    w_exp = a
                    sh = snap_sh.get(p)
                    share = max(sh[0], sh[1]) / sh[2] if sh and sh[2] > 0 else 0.0
                    absent = snap_era and a >= REG and share >= SHARE_REG and p not in played
                    w_lin = 0.0 if absent else a
                    if absent:
                        n_abs += 1; abs_a += a
                    erows.append((g.game_id, yr, t, side, p, a, int(absent), role.get(p, "")))
                    rr =rr_rows.get((g.game_id, p)) or last_rr.get(p)
                    pr = pr_rows.get((g.game_id, p)) or last_pr.get(p)
                    cv = cv_rows.get((g.game_id, p)) or last_cv.get(p)
                    contrib = {}
                    if rr is not None and p != qb_id:
                        tsh, csh, ve, epar = rr
                        for v, w in (("lin", w_lin), ("exp", w_exp)):
                            rec_num[v] += w * tsh * ve; rec_den[v] += w * tsh
                            rsh_num[v] += w * csh * epar; rsh_den[v] += w * csh
                        contrib["REC"] = tsh * TGT_PER_ATT * ATTb * ve
                        contrib["RUSH"] = csh * RUNb * epar
                    if pr is not None:
                        thsk, thht, thst = pr
                        thht = thht if thht == thht else 0.0
                        contrib["PR"] = thsk * L_sk * c_sk_base + thht * L_ht * c_ht_base
                        contrib["RF"] = thst * L_st * c_st_base
                    if cv is not None:
                        thpd, thint, thcp, cve = cv
                        grp = cgroup.get(p, "DB")
                        mpd, mint, mcp = MU[(yr, grp)]
                        contrib["BALL"] = ATTb * (LW["pd"] * mpd * thpd + LW["int"] * mint * thint
                                                  + LW["cpen"] * mcp * thcp)
                        contrib["COV"] = ATTb * cve
                    for u in ("PR", "RF", "BALL", "COV"):
                        if u in contrib:
                            acc["lin"][u] += w_lin * contrib[u]; acc["exp"][u] += w_exp * contrib[u]
                    if absent:                                 # vacated production refilled at prior level
                        b = bucket.get(p)
                        if pr is not None and b is not None:
                            rsk, rht, rst = REP[(yr, b)]
                            acc["lin"]["PR"] += a * (rsk * L_sk * c_sk_base + rht * L_ht * c_ht_base)
                            acc["lin"]["RF"] += a * rst * L_st * c_st_base
                        if cv is not None:
                            grp = cgroup.get(p, "DB")
                            mpd, mint, mcp = MU[(yr, grp)]
                            acc["lin"]["BALL"] += a * ATTb * (LW["pd"] * mpd + LW["int"] * mint + LW["cpen"] * mcp)
                    if contrib or absent:
                        prow.append((g.game_id, yr, t, side, p, a, int(absent), int(p == qb_id),
                                     contrib.get("REC", np.nan), contrib.get("RUSH", np.nan),
                                     contrib.get("PR", np.nan), contrib.get("RF", np.nan),
                                     contrib.get("BALL", np.nan), contrib.get("COV", np.nan)))
                for v in ("lin", "exp"):
                    acc[v]["QB"] = qb_comp * DBb
                    acc[v]["REC"] = TGT_PER_ATT * ATTb * rec_num[v] / rec_den[v] if rec_den[v] > 1e-9 else 0.0
                    acc[v]["RUSH"] = RUNb * rsh_num[v] / rsh_den[v] if rsh_den[v] > 1e-9 else 0.0
                    if pf:
                        o_sk, o_ht, o_st = (x if x == x else 1.0 for x in (pf[0], pf[1], pf[2]))
                        acc[v]["PROT"] = ((1 - o_sk) * L_sk * c_sk_base + (1 - o_ht) * L_ht * c_ht_base
                                          + (1 - o_st) * L_st * c_st_base)
                vv = vol.get((g.game_id, t), (np.nan,) * 5)
                row = {"game_id": g.game_id, "season": yr, "season_type": g.season_type, "week": int(g.week),
                       "game_date": g.game_date, "team": t, "opp": o, "side": side,
                       "neutral": int(g.location == "Neutral"),
                       "pts_for": float(g.home_score if side == "home" else g.away_score),
                       "pts_against": float(g.away_score if side == "home" else g.home_score),
                       "n_db": vv[0], "n_att": vv[1], "n_run": vv[2], "n_play": vv[3], "epa_off": vv[4],
                       "starter": qb_id, "qb_comp": qb_comp, "snap_era": int(snap_era),
                       "n_expected": len(E), "n_absent_reg": n_abs, "absent_a": abs_a}
                for v in ("lin", "exp"):
                    for u in UNITS:
                        row[f"{u}_{v}"] = acc[v][u]
                trows.append(row)
                # membership for the presence update (applied after the whole date)
                if snap_era:
                    mem = set(played)
                    rl = {p: ("O" if snap_pct[(g.game_id, p)][0] >= snap_pct[(g.game_id, p)][1] else "D")
                          for p in mem}
                else:
                    mo = set(qb_members.get((g.game_id, t), [])) | set(rr_members.get((g.game_id, t), []))
                    md = def_credit_members.get((g.game_id, t), set())
                    mem = mo | md
                    rl = {p: "O" for p in mo}
                    rl.update({p: "D" for p in md - mo})
                date_updates.append((t, mem, rl))
        # reveal this date's pre-game component rows as the latest known states (after the date)
        gids = set(dg.game_id)
        for gid in gids:
            for p in RR_BY_GAME.get(gid, ()):
                last_rr[p] = rr_rows[(gid, p)]
            for p in PR_BY_GAME.get(gid, ()):
                last_pr[p] = pr_rows[(gid, p)]
            for p in CV_BY_GAME.get(gid, ()):
                last_cv[p] = cv_rows[(gid, p)]
        for gid in gids:
            for t in (TEAMS_OF[gid]):
                for p in snap_members.get((gid, t), ()):
                    op, dp = snap_pct[(gid, p)]
                    st_ = snap_sh.setdefault(p, [0.0, 0.0, 0.0])
                    st_[0] = SH_DECAY * st_[0] + op; st_[1] = SH_DECAY * st_[1] + dp
                    st_[2] = SH_DECAY * st_[2] + 1.0
        for t, mem, rl in date_updates:
            role.update(rl)
            for p in mem:
                ot = team_of.get(p)
                if ot is not None and ot != t:
                    pres[ot].pop(p, None)
                team_of[p] = t
            d = pres[t]
            for p in set(d) | mem:
                d[p] = RHO * d.get(p, 0.0) + (1 - RHO) * (1.0 if p in mem else 0.0)
            for p in [p for p, a in d.items() if a < 0.01]:
                del d[p]
    TG = pd.DataFrame(trows)
    PGm = pd.DataFrame(prow, columns=["game_id", "season", "team", "side", "player_id", "a", "absent",
                                      "is_starter_qb", "c_REC", "c_RUSH", "c_PR", "c_RF", "c_BALL", "c_COV"])
    assert TG.season.max() < TEST_ERA and PGm.season.max() < TEST_ERA
    TG.to_parquet(os.path.join(DATA, "pv_nfl_compose_team_games.parquet"), index=False)
    EM = pd.DataFrame(erows, columns=["game_id", "season", "team", "side", "player_id", "a", "absent", "role"])
    assert EM.season.max() < TEST_ERA
    EM.to_parquet(os.path.join(DATA, "pv_nfl_compose_lineups.parquet"), index=False)
    PGm.to_parquet(os.path.join(DATA, "pv_nfl_compose_player_games.parquet"), index=False)
    json.dump({"event_epa_constants": {str(k): v for k, v in K.items()},
               "league_volumes": {str(k): v for k, v in LV.items()},
               "coverage_linear_weights": LW, "tgt_per_att": TGT_PER_ATT,
               "coverage_rate_per_att": {f"{k[0]}|{k[1]}": v for k, v in MU.items()},
               "params": {"rho": RHO, "a_min": A_MIN, "regular": REG, "share_regular": SHARE_REG,
                          "share_decay": SH_DECAY, "snap_era": SNAP_ERA}},
              open(os.path.join(DATA, "pv_nfl_compose_constants.json"), "w"), indent=1)
    dv = TG[TG.season >= 2006]
    log(f"team-games {len(TG)} (DEV {len(dv)}); player-game rows {len(PGm)}")
    log("DEV unit aggregate SD (lineup variant, EPA/game):",
        {u: round(float(dv[f'{u}_lin'].std()), 3) for u in UNITS})
    sn = TG[TG.snap_era == 1]
    log(f"snap-era team-games {len(sn)}: mean regulars absent {sn.n_absent_reg.mean():.2f}, "
        f"mean expected roster {sn.n_expected.mean():.1f}")
    log("wrote data/pv_nfl_compose_team_games.parquet, _player_games.parquet, _constants.json")


TGT_PER_ATT = 1.0   # targets == non-sack pass attempts (the receiver rating is per target)

if __name__ == "__main__":
    main()
