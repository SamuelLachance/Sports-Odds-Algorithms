"""Player-value program, NFL - COVERAGE on attributed events (walk-forward engine).

What public play-by-play CAN attribute to a pass defender (nflverse pbp, DEV 2006-2015):
  * interceptions             interception_player_id (100% of INTs)
  * passes defensed           pass_defense_1/2_player_id (breakups; ~30% of incompletions,
                              ~89% of INTs also carry a PD credit for the interceptor)
  * the tackler after a catch solo/assist/with-assist tackle ids on completions (~90% of catches)
  * coverage penalties        penalty_player_id on DPI / illegal contact / defensive holding
                              (no_play rows, ~4.3k/10 seasons, 99% carry a player id)
What it CANNOT attribute: the defender in coverage on a completion (only the tackler, who may be a
rallying safety or LB), completions that end untackled (7.8k DEV touchdowns, 3.4k out of bounds),
incompletions with no PD credit (~70%: overthrows, drops, throwaways), targets NOT thrown because
the receiver was covered (deterrence - the shutdown-corner effect), snaps / alignment / man vs zone
(no participation data before 2016). PD and tackle counts are recorded by the HOME scorer and drift
by season; this engine carries a walk-forward stadium (scorer) factor and league rates as EWMAs.

MLB principle ported: rate the defender ONLY on credited events, opponent-QB-adjusted, scorer-
adjusted, empirical-Bayes shrunk (gamma-Poisson per event type), strictly walk-forward (all state
for a game day is read before any game of that day updates it).

Per defender (DB/LB rated; DL tracked), per event type k in {pd, int, tc, cpen}:
  exp_k = att * mu_{pos,k} * Q_k(opp primary passer) * S_k(stadium)    (S only for pd, tc)
  theta_k = (a_k + sum_decayed obs_k) / (a_k + sum_decayed exp_k)
Composite coverage value (EPA saved per opponent pass attempt, defence-positive, linear weights):
  CV = sum_k v_k * mu_{pos,k} * (theta_k - 1),  v_k = -(mean EPA of a k-play - mean EPA of attempt)
Efficiency views on ATTRIBUTED targets (AT = pd + int + tc):
  DSP = defended-share over expectation (QB-adjusted), beta-shrunk
  EPAP = EPA per attributed target over QB expectation, shrunk (defence-positive)
Comparators (the "current rating" analogs computable on DEV):
  PM  = on-field plus-minus: EWMA of the team's opp-QB-adjusted pass EPA allowed per dropback in
        games the player was active (the 11v11 shared-credit logic without participation data)
  R6  = the shipped weekly composite for defenders (45% rush, 35% ball=3*INT+PD, 20% solo tackles,
        6-game EB prior, z by bucket), rebuilt from the same pbp credits

Hyper-parameters: fixed a priori, except the gamma prior strengths a_k which are set by method of
moments on the WARM-UP seasons 1999-2005 only (pass 1), then the engine re-runs from 1999 (pass 2).

TEST discipline: default run reads seasons <= 2015 only (pyarrow filter + assert) and prints only
DEV/warm-up numbers. `--all-seasons` extends the walk to every pulled season for serving and writes
data/pv_nfl_coverage_serve_*.parquet WITHOUT printing or computing any summary statistic.

Outputs (default run):
  data/pv_nfl_coverage_player_games.parquet  one row per defender-game, PRE-game walk-forward values
                                             + that game's observed credits (for validation)
  data/pv_nfl_coverage_unit_games.parquet    one row per team-game, PRE-game unit aggregates
  data/pv_nfl_coverage_priors.json           warm-up MoM prior strengths, linear weights
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TEST_ERA = 2016
EVENTS = "data/pv_nfl_events.parquet"
KINDS = ("pd", "int", "tc", "cpen")
COV_PEN = {"Defensive Pass Interference", "Illegal Contact", "Defensive Holding"}
GROUPS = ("DB", "LB", "DL")

# ---- fixed a-priori dynamics -------------------------------------------------------------------
P_DECAY, P_SEASON = 0.97, 0.75          # player: per appearance, at a season boundary
Q_DECAY, Q_SEASON, Q_PRIOR = 0.98, 0.85, 30.0   # opposing passer factor (events)
S_DECAY, S_SEASON, S_PRIOR = 0.97, 0.90, 60.0   # stadium scorer factor (events), per hosted game
D_DECAY, D_SEASON, D_PRIOR = 0.90, 0.60, 20.0   # team-defence factor (only to de-confound Q and S)
L_DECAY = 0.998                          # league/position per-attempt rates, per defence-game
DS_PRIOR, EPA_PRIOR = 80.0, 80.0         # QB-level offsets on attributed targets (targets)
PM_PRIOR = 300.0                         # QB-level offset on dropback EPA (dropbacks)
AT_PRIOR = 100.0                         # player involvement rate shrink (team attempts)
# player-level efficiency shrink (targets / dropbacks) is set per group by warm-up MoM (pass 1)
R6_DECAY, R6_PRIOR = 0.95, 6.0           # shipped composite dynamics
UNIT_DECAY = 0.6                         # availability weight per team game


def pos_group(p):
    if p in ("CB", "DB", "S", "SS", "FS", "SAF"):
        return "DB"
    if p in ("LB", "ILB", "OLB", "MLB"):
        return "LB"
    if p in ("DE", "DT", "NT", "DL"):
        return "DL"
    return None


# =============================================================================== data prep ======
def load(all_seasons=False):
    cols = ["game_id", "play_id", "season", "season_type", "week", "game_date", "home_team",
            "away_team", "posteam", "defteam", "play_type", "epa", "yards_gained", "sack",
            "qb_spike", "two_point_attempt", "complete_pass", "interception", "pass_touchdown",
            "penalty", "penalty_type", "penalty_team", "penalty_player_id", "passer_id",
            "interception_player_id", "pass_defense_1_player_id", "pass_defense_2_player_id",
            "solo_tackle_1_player_id", "solo_tackle_1_team", "solo_tackle_2_player_id",
            "solo_tackle_2_team", "assist_tackle_1_player_id", "assist_tackle_1_team",
            "assist_tackle_2_player_id", "assist_tackle_2_team", "tackle_with_assist_1_player_id",
            "tackle_with_assist_1_team", "tackle_with_assist_2_player_id",
            "tackle_with_assist_2_team", "sack_credit", "qb_hit_1_player_id", "qb_hit_2_player_id",
            ]
    filt = None if all_seasons else [("season", "<", TEST_ERA)]
    df = pq.read_table(EVENTS, columns=cols, filters=filt).to_pandas()
    if not all_seasons:
        assert df.season.max() < TEST_ERA
    for c in df.columns:
        if str(df[c].dtype) in ("string", "str"):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def build_tables(df):
    """Return (games, team_rows, events) - all per game/defence, no cross-game information."""
    pt = df.play_type.values
    scrim = np.isin(pt, ["pass", "run", "no_play"])
    d = df[scrim].copy()
    d["is_att"] = ((d.play_type == "pass") & (d.sack.fillna(0) != 1) & (d.qb_spike.fillna(0) != 1)
                   & (d.two_point_attempt.fillna(0) != 1))
    d["is_db"] = (d.play_type == "pass") & (d.two_point_attempt.fillna(0) != 1) & (d.qb_spike.fillna(0) != 1)
    d["is_cpen"] = ((d.penalty.fillna(0) == 1) & (d.penalty_team == d.defteam)
                    & d.penalty_type.isin(COV_PEN) & d.play_type.isin(["no_play", "pass"])
                    & d.penalty_player_id.notna())
    d["epa0"] = d.epa.fillna(0.0).astype(float)
    ev = []  # (game_id, defteam, player, kind, credit, epa)

    def add(mask, idcol, kind, credit=1.0, teamcol=None):
        m = mask & d[idcol].notna()
        if teamcol is not None:
            m &= (d[teamcol] == d.defteam)
        x = d.loc[m, ["game_id", "defteam", idcol, "epa0"]]
        cr = credit if np.isscalar(credit) else credit[m.values]
        ev.append(pd.DataFrame({"game_id": x.game_id.values, "defteam": x.defteam.values,
                                "pid": x[idcol].values, "kind": kind,
                                "credit": np.broadcast_to(cr, len(x)).astype(float),
                                "epa": x.epa0.values}))

    att = d.is_att
    inter = att & (d.interception.fillna(0) == 1)
    add(inter, "interception_player_id", "int")
    for c in ("pass_defense_1_player_id", "pass_defense_2_player_id"):
        m = att & (d[c] != d.interception_player_id.where(inter, "__none__"))
        add(m, c, "pd")
    comp = att & (d.complete_pass.fillna(0) == 1) & (d.interception.fillna(0) != 1)
    tk_cols = [("solo_tackle_1_player_id", "solo_tackle_1_team"),
               ("solo_tackle_2_player_id", "solo_tackle_2_team"),
               ("tackle_with_assist_1_player_id", "tackle_with_assist_1_team"),
               ("tackle_with_assist_2_player_id", "tackle_with_assist_2_team"),
               ("assist_tackle_1_player_id", "assist_tackle_1_team"),
               ("assist_tackle_2_player_id", "assist_tackle_2_team")]
    ntk = np.zeros(len(d))
    for c, t in tk_cols:
        ntk += (d[c].notna() & (d[t] == d.defteam)).values
    frac = np.where(ntk > 0, 1.0 / np.maximum(ntk, 1), 0.0)
    for c, t in tk_cols:
        add(comp, c, "tc", frac, t)
    add(d.is_cpen, "penalty_player_id", "cpen")
    # activity + R6 ingredients (all scrimmage plays)
    for c, t in tk_cols[:2]:
        add(pd.Series(True, index=d.index), c, "solo", 1.0, t)
    for c, t in tk_cols[2:]:
        add(pd.Series(True, index=d.index), c, "act", 1.0, t)
    run_loss = (d.play_type == "run") & (d.yards_gained.fillna(0) < 0)
    for c, t in tk_cols:
        add(run_loss, c, "tfl", frac, t)
    for c in ("qb_hit_1_player_id", "qb_hit_2_player_id"):
        add(d.play_type == "pass", c, "hit")
    sc = d.loc[d.sack_credit.notna(), ["game_id", "defteam", "sack_credit", "epa0"]]
    rows = []
    for gid, dt, s, e in sc.itertuples(index=False):
        for part in str(s).split(";"):
            pid, w = part.rsplit(":", 1)
            rows.append((gid, dt, pid, "sack", float(w), e))
    ev.append(pd.DataFrame(rows, columns=["game_id", "defteam", "pid", "kind", "credit", "epa"]))
    pen = (d.penalty.fillna(0) == 1) & (d.penalty_team == d.defteam) & ~d.is_cpen
    add(pen, "penalty_player_id", "act")
    E = pd.concat(ev, ignore_index=True)

    # ---- team (defence) per game
    g = d.groupby(["game_id", "defteam"], sort=False)
    T = pd.DataFrame({
        "att": g.is_att.sum() + g.is_cpen.sum(),
        "n_att": g.is_att.sum(),
        "comp": g.apply(lambda x: float(((x.complete_pass.fillna(0) == 1) & x.is_att).sum())),
    })
    dbk = d[d.is_db]
    gd = dbk.groupby(["game_id", "defteam"], sort=False)
    T["dropbacks"] = gd.size()
    T["pass_epa"] = gd.epa0.sum()
    T["pass_yds"] = gd.yards_gained.sum()           # net: sacks carry negative yards
    T["sacks"] = gd.sack.sum()
    T["pass_td"] = gd.pass_touchdown.sum()
    T["ints"] = gd.interception.sum()
    T = T.fillna(0.0)
    # primary opposing passer = most attempts
    pp = (d[d.is_att & d.passer_id.notna()].groupby(["game_id", "defteam", "passer_id"]).size()
          .reset_index(name="n").sort_values("n", ascending=False)
          .drop_duplicates(["game_id", "defteam"]).set_index(["game_id", "defteam"]).passer_id)
    T["opp_qb"] = pp
    T = T.reset_index()
    # game meta
    Gm = (df.groupby("game_id", sort=False)
          .agg(season=("season", "first"), week=("week", "first"), season_type=("season_type", "first"),
               game_date=("game_date", "first"), home=("home_team", "first"), away=("away_team", "first"))
          .reset_index())
    T = T.merge(Gm, on="game_id", how="left")
    T["offteam"] = np.where(T.defteam == T.home, T.away, T.home)
    # final scores (schedule file; join on game_id, roles not codes) -> points allowed per defence
    smax = int(df.season.max())
    sch = pd.read_csv("data/nfl_games.csv", usecols=["game_id", "season", "home_score", "away_score"])
    sch = sch[sch.season <= smax].set_index("game_id")
    hs = T.game_id.map(sch.home_score); as_ = T.game_id.map(sch.away_score)
    T["pts_allowed"] = np.where(T.defteam == T.home, as_, hs)   # defence at home allows the away score
    return Gm, T, E


# ============================================================================== the engine ======
class Engine:
    def __init__(self, prior_a, lw, posmap, eff_prior=None):
        self.a = prior_a                  # {(group, kind): a}
        self.k = eff_prior or {(g_, m_): v_ for g_ in GROUPS for m_, v_ in
                               (("ds", 80.0), ("epa", 80.0), ("pm", 600.0))}
        self.grp = {(g_, m_): [0.0, 0.0] for g_ in GROUPS for m_ in ("ds", "epa", "at")}
        self.lw = lw                      # {kind: v_k} defence-positive EPA per event vs avg attempt
        self.pos = posmap
        # league per-attempt rates by group (EWMA numer/denom); initial guesses overwritten fast
        self.mu = {(gname, k): [0.0, 0.0] for gname in GROUPS for k in KINDS}
        self.lgT = {k: [0.0, 0.0] for k in KINDS}   # team-level league events per attempt
        self.lg_ds = [0.0, 0.0]           # league defended share (defended, AT)
        self.lg_epa = [0.0, 0.0]          # league EPA per attributed target
        self.lg_pm = [0.0, 0.0]           # league pass EPA per dropback
        self.lg_r6 = {}
        self.Q = defaultdict(lambda: {k: [0.0, 0.0] for k in KINDS + ("ds", "epa", "pm")})
        self.S = defaultdict(lambda: {k: [0.0, 0.0] for k in ("pd", "tc")})
        self.D = defaultdict(lambda: {k: [0.0, 0.0] for k in KINDS})
        self.P = defaultdict(lambda: defaultdict(float))
        self.unit = defaultdict(dict)     # team -> {pid: availability weight}
        self.last_team = {}
        self.season = None

    # ---- factor reads (pre-game) ----
    def mu_rate(self, gname, k):
        n, dd = self.mu[(gname, k)]
        return n / dd if dd > 0 else 1e-3

    def lg_rate(self, k):
        n, dd = self.lgT[k]
        return n / dd if dd > 0 else 1e-3

    def qf(self, qb, k):
        o, e = self.Q[qb][k] if qb is not None else (0.0, 0.0)
        return (Q_PRIOR + o) / (Q_PRIOR + e)

    def sf(self, st, k):
        o, e = self.S[st][k]
        return (S_PRIOR + o) / (S_PRIOR + e)

    def df_(self, tm, k):
        o, e = self.D[tm][k]
        return (D_PRIOR + o) / (D_PRIOR + e)

    def q_add(self, qb, k):
        """QB additive offsets for efficiency views (DS, EPA/AT, PM)."""
        if qb is None:
            return 0.0
        o, e = self.Q[qb][k]
        pr = {"ds": DS_PRIOR, "epa": EPA_PRIOR, "pm": PM_PRIOR}[k]
        return o / (e + pr)

    def new_season(self):
        for st in self.P.values():
            for key in list(st.keys()):
                if key.startswith(("o_", "e_", "ds", "at", "ep", "pm", "r6")):
                    st[key] *= P_SEASON
        for dct, f in ((self.Q, Q_SEASON), (self.S, S_SEASON), (self.D, D_SEASON)):
            for v in dct.values():
                for k in v:
                    v[k][0] *= f; v[k][1] *= f

    # ---- player rating read ----
    def player(self, pid):
        st = self.P.get(pid)
        gname = self.pos.get(pid)
        out = {"group": gname}
        if gname is None:
            return None
        th = {}
        for k in KINDS:
            a = self.a[(gname, k)]
            o = st.get("o_" + k, 0.0) if st else 0.0
            e = st.get("e_" + k, 0.0) if st else 0.0
            th[k] = (a + o) / (a + e)
            out["th_" + k] = th[k]
        # ball = pd + int pooled
        a = self.a[(gname, "pd")] + self.a[(gname, "int")]
        o = (st.get("o_pd", 0.0) + st.get("o_int", 0.0)) if st else 0.0
        e = (st.get("e_pd", 0.0) + st.get("e_int", 0.0)) if st else 0.0
        out["th_ball"] = (a + o) / (a + e)
        out["cv"] = sum(self.lw[k] * self.mu_rate(gname, k) * (th[k] - 1.0) for k in KINDS)
        # efficiency on attributed targets, shrunk toward the position-group mean residual
        at = st.get("at", 0.0) if st else 0.0
        out["at_n"] = at
        gds, gep, gat = self.grp[(gname, "ds")], self.grp[(gname, "epa")], self.grp[(gname, "at")]
        cds = gds[0] / gds[1] if gds[1] else 0.0
        cep = gep[0] / gep[1] if gep[1] else 0.0
        mat = gat[0] / gat[1] if gat[1] else 0.05
        out["dsp"] = ((st.get("ds_o", 0.0) - st.get("ds_e", 0.0) - at * cds)
                      / (at + self.k[(gname, "ds")])) if st else 0.0
        out["epap"] = (-(st.get("ep_o", 0.0) - st.get("ep_e", 0.0) - at * cep)
                       / (at + self.k[(gname, "epa")])) if st else 0.0
        att_ = st.get("att", 0.0) if st else 0.0
        out["at_rate"] = (at + AT_PRIOR * mat) / (att_ + AT_PRIOR)
        out["cv_eff"] = out["at_rate"] * out["epap"]      # EPA saved per team attempt via his targets
        # plus-minus (defence-positive: negative EPA allowed over expectation is good)
        pmn = st.get("pm_n", 0.0) if st else 0.0
        out["pm"] = (-st.get("pm_r", 0.0) / (pmn + self.k[(gname, "pm")])) if st else 0.0
        out["pm_n"] = pmn
        # R6 composite
        r6 = 0.0
        if st and gname in self.lg_r6 and st.get("r6_w", 0.0) > 1.0:
            for key, wt in (("rush", .45), ("ball", .35), ("tak", .20)):
                mu_, sd_ = self.lg_r6[gname][key]
                v = (st.get("r6_" + key, 0.0) + R6_PRIOR * mu_) / (st.get("r6_w", 0.0) + R6_PRIOR)
                r6 += wt * (v - mu_) / sd_
        out["r6"] = r6
        out["games"] = st.get("games", 0.0) if st else 0.0
        out["exp_ball"] = e
        return out

    # ---- per-date step ----
    def step(self, date_games, Tg, Eg, collect):
        """date_games: list of game ids on one date. Reads all state first, then updates."""
        pre = []
        for gid in date_games:
            rows = Tg.get(gid, [])
            for tr in rows:
                season = tr["season"]
                st = tr["home"]
                qb = tr["opp_qb"] if isinstance(tr["opp_qb"], str) else None
                att = tr["att"]
                F = {k: self.qf(qb, k) for k in KINDS}
                SF = {k: self.sf(st, k) for k in ("pd", "tc")}
                DF = {k: self.df_(tr["defteam"], k) for k in KINDS}
                qds, qep, qpm = self.q_add(qb, "ds"), self.q_add(qb, "epa"), self.q_add(qb, "pm")
                lgds = self.lg_ds[0] / self.lg_ds[1] if self.lg_ds[1] else 0.3
                lgep = self.lg_epa[0] / self.lg_epa[1] if self.lg_epa[1] else 0.0
                lgpm = self.lg_pm[0] / self.lg_pm[1] if self.lg_pm[1] else 0.0
                ev = Eg.get((gid, tr["defteam"]), {})
                # unit (pre-game) aggregate from availability weights
                team = tr["defteam"]
                U = defaultdict(float)
                for pid, w in self.unit[team].items():
                    if w < 0.05:
                        continue
                    r = self.player(pid)
                    if r is None:
                        continue
                    gname = r["group"]
                    for tag, grp in (("dblb", ("DB", "LB")), ("db", ("DB",))):
                        if gname in grp:
                            U[tag + "_cv"] += w * r["cv"]
                            U[tag + "_ball"] += w * self.mu_rate(gname, "pd") * (r["th_ball"] - 1.0)
                            U[tag + "_int"] += w * self.mu_rate(gname, "int") * (r["th_int"] - 1.0)
                            U[tag + "_tc"] += w * self.mu_rate(gname, "tc") * (r["th_tc"] - 1.0)
                            U[tag + "_cpen"] += w * self.mu_rate(gname, "cpen") * (r["th_cpen"] - 1.0)
                            wat = w * r["at_rate"]
                            U[tag + "_dsp_n"] += wat * r["dsp"]; U[tag + "_epap_n"] += wat * r["epap"]
                            U[tag + "_at_w"] += wat
                            U[tag + "_val"] += wat * r["epap"]
                            U[tag + "_pm_n"] += w * r["pm"]; U[tag + "_w"] += w
                    if gname in ("DB", "LB", "DL"):
                        U["all_r6_n"] += w * r["r6"]; U["all_w"] += w
                        if gname in ("DB", "LB"):
                            U["dblb_r6_n"] += w * r["r6"]
                urow = {"game_id": gid, "season": season, "week": tr["week"], "defteam": team,
                        "offteam": tr["offteam"], "opp_qb": qb, "home": st,
                        "qf_pd": F["pd"], "qf_int": F["int"], "qf_tc": F["tc"], "sf_pd": SF["pd"],
                        "q_pm": qpm, "lg_pm": lgpm}
                for tag in ("dblb", "db"):
                    urow[tag + "_cv"] = U[tag + "_cv"]; urow[tag + "_ball"] = U[tag + "_ball"]
                    urow[tag + "_int"] = U[tag + "_int"]; urow[tag + "_tc"] = U[tag + "_tc"]
                    urow[tag + "_cpen"] = U[tag + "_cpen"]
                    urow[tag + "_dsp"] = U[tag + "_dsp_n"] / U[tag + "_at_w"] if U[tag + "_at_w"] else 0.0
                    urow[tag + "_epap"] = U[tag + "_epap_n"] / U[tag + "_at_w"] if U[tag + "_at_w"] else 0.0
                    urow[tag + "_pm"] = U[tag + "_pm_n"] / U[tag + "_w"] if U[tag + "_w"] else 0.0
                    urow[tag + "_val"] = U[tag + "_val"]
                    urow[tag + "_n"] = U[tag + "_w"]
                urow["all_r6"] = U["all_r6_n"] / U["all_w"] if U["all_w"] else 0.0
                urow["dblb_r6"] = U["dblb_r6_n"] / U["dblb_w"] if U["dblb_w"] else 0.0
                for c in ("att", "n_att", "comp", "dropbacks", "pass_epa", "pass_yds", "sacks",
                          "pass_td", "ints"):
                    urow["y_" + c] = tr[c]
                if collect is not None:
                    collect["unit"].append(urow)
                # players active this game (credited on a defensive scrimmage play)
                plist = []
                for pid, cnt in ev.items():
                    gname = self.pos.get(pid)
                    if gname is None:
                        continue
                    r = self.player(pid)
                    e_k = {}
                    for k in KINDS:
                        base = att * self.mu_rate(gname, k) * F[k]
                        if k in SF:
                            base *= SF[k]
                        e_k[k] = base
                    plist.append((pid, gname, cnt, e_k, r))
                    if collect is not None:
                        row = {"game_id": gid, "season": season, "week": tr["week"], "team": team,
                               "opp": tr["offteam"], "home_game": int(team == st), "player_id": pid,
                               "group": gname, "att": att}
                        for k2, v2 in r.items():
                            if k2 != "group":
                                row["pre_" + k2] = v2
                        for k in KINDS:
                            row["o_" + k] = cnt.get(k, 0.0); row["e_" + k] = e_k[k]
                        row["o_at"] = cnt.get("pd", 0.0) + cnt.get("int", 0.0) + cnt.get("tc", 0.0)
                        row["o_def"] = cnt.get("pd", 0.0) + cnt.get("int", 0.0)
                        row["o_atepa"] = cnt.get("atepa", 0.0)
                        row["e_ds"] = row["o_at"] * (lgds + qds)
                        row["e_atepa"] = row["o_at"] * (lgep + qep)
                        row["team_pass_epa"] = tr["pass_epa"]; row["team_dropbacks"] = tr["dropbacks"]
                        row["pm_exp"] = tr["dropbacks"] * (lgpm + qpm)
                        row["solo"] = cnt.get("solo", 0.0)
                        collect["player"].append(row)
                pre.append((tr, qb, st, att, F, SF, DF, qds, qep, qpm, lgds, lgep, lgpm, plist))
        # ------------------------------------------------ updates (after every read of the date)
        for (tr, qb, st, att, F, SF, DF, qds, qep, qpm, lgds, lgep, lgpm, plist) in pre:
            team = tr["defteam"]
            tot = defaultdict(float)
            evall = Eg.get((tr["game_id"], team), {})
            for pid, cnt in evall.items():          # every credited id, mapped position or not
                for k in KINDS:
                    tot[k] += cnt.get(k, 0.0)
            # team-level obs vs league expectation
            for k in KINDS:
                lg = self.lg_rate(k) * att
                sfk = SF.get(k, 1.0)
                if qb is not None:
                    q = self.Q[qb][k]
                    q[0] = Q_DECAY * q[0] + tot[k]; q[1] = Q_DECAY * q[1] + lg * sfk * DF[k]
                dd = self.D[team][k]
                dd[0] = D_DECAY * dd[0] + tot[k]; dd[1] = D_DECAY * dd[1] + lg * sfk * F[k]
                if k in SF:
                    s_ = self.S[st][k]
                    s_[0] = S_DECAY * s_[0] + tot[k]; s_[1] = S_DECAY * s_[1] + lg * F[k] * DF[k]
            at_tot = tot["pd"] + tot["int"] + tot["tc"]
            atepa_tot = sum(cnt.get("atepa", 0.0) for cnt in evall.values())
            if qb is not None:
                q = self.Q[qb]
                q["ds"][0] = Q_DECAY * q["ds"][0] + (tot["pd"] + tot["int"] - at_tot * lgds)
                q["ds"][1] = Q_DECAY * q["ds"][1] + at_tot
                q["epa"][0] = Q_DECAY * q["epa"][0] + (atepa_tot - at_tot * lgep)
                q["epa"][1] = Q_DECAY * q["epa"][1] + at_tot
                q["pm"][0] = Q_DECAY * q["pm"][0] + (tr["pass_epa"] - tr["dropbacks"] * lgpm)
                q["pm"][1] = Q_DECAY * q["pm"][1] + tr["dropbacks"]
            # league
            self.lg_ds = [L_DECAY * self.lg_ds[0] + tot["pd"] + tot["int"], L_DECAY * self.lg_ds[1] + at_tot]
            self.lg_epa = [L_DECAY * self.lg_epa[0] + atepa_tot, L_DECAY * self.lg_epa[1] + at_tot]
            self.lg_pm = [L_DECAY * self.lg_pm[0] + tr["pass_epa"], L_DECAY * self.lg_pm[1] + tr["dropbacks"]]
            seen = defaultdict(float); gobs = defaultdict(float)
            for pid, gname, cnt, e_k, r in plist:
                seen[gname] += 1
                for k in KINDS:
                    gobs[(gname, k)] += cnt.get(k, 0.0)
            for gname in GROUPS:
                for k in KINDS:
                    m = self.mu[(gname, k)]
                    m[0] = L_DECAY * m[0] + gobs[(gname, k)]
                    m[1] = L_DECAY * m[1] + att * seen[gname]
            for k in KINDS:
                self.lgT[k][0] = L_DECAY * self.lgT[k][0] + tot[k]
                self.lgT[k][1] = L_DECAY * self.lgT[k][1] + att
            # players
            pm_resid = tr["pass_epa"] - tr["dropbacks"] * (lgpm + qpm)
            for pid, gname, cnt, e_k, r in plist:
                s = self.P[pid]
                for key in list(s.keys()):
                    if key.startswith(("o_", "e_", "ds_", "at", "ep_", "pm_")):
                        s[key] *= P_DECAY
                for k in KINDS:
                    s["o_" + k] += cnt.get(k, 0.0); s["e_" + k] += e_k[k]
                at = cnt.get("pd", 0.0) + cnt.get("int", 0.0) + cnt.get("tc", 0.0)
                s["at"] += at
                s["ds_o"] += cnt.get("pd", 0.0) + cnt.get("int", 0.0)
                s["ds_e"] += at * (lgds + qds)
                s["ep_o"] += cnt.get("atepa", 0.0); s["ep_e"] += at * (lgep + qep)
                s["pm_r"] += pm_resid; s["pm_n"] += tr["dropbacks"]
                s["att"] += att
                s["games"] += 1
                for m_, num in (("ds", cnt.get("pd", 0.0) + cnt.get("int", 0.0) - at * (lgds + qds)),
                                ("epa", cnt.get("atepa", 0.0) - at * (lgep + qep))):
                    gr = self.grp[(gname, m_)]
                    gr[0] = L_DECAY * gr[0] + num; gr[1] = L_DECAY * gr[1] + at
                gr = self.grp[(gname, "at")]
                gr[0] = L_DECAY * gr[0] + at; gr[1] = L_DECAY * gr[1] + att
                # R6 (per game, decayed)
                for key in ("r6_rush", "r6_ball", "r6_tak", "r6_w"):
                    s[key] *= R6_DECAY
                s["r6_rush"] += cnt.get("sack", 0.0) + 0.5 * cnt.get("tfl", 0.0) + 0.5 * cnt.get("hit", 0.0)
                s["r6_ball"] += 3.0 * cnt.get("int", 0.0) + cnt.get("pd", 0.0)
                s["r6_tak"] += cnt.get("solo", 0.0)
                s["r6_w"] += 1.0
                # availability (unit membership): player leaves his previous team's pool
                old = self.last_team.get(pid)
                if old is not None and old != team:
                    self.unit[old].pop(pid, None)
                self.last_team[pid] = team
            for pid in list(self.unit[team].keys()):
                self.unit[team][pid] *= UNIT_DECAY
                if self.unit[team][pid] < 0.01:
                    del self.unit[team][pid]
            for pid, gname, cnt, e_k, r in plist:
                self.unit[team][pid] = self.unit[team].get(pid, 0.0) + (1 - UNIT_DECAY)


def run(Gm, T, E, prior_a, lw, posmap, r6z, seasons_max, collect=True, eff_prior=None):
    eng = Engine(prior_a, lw, posmap, eff_prior)
    eng.lg_r6 = r6z
    # seed league rates with a neutral small pseudo-count so the first games have finite rates
    for gname, base in (("DB", (0.030, 0.006, 0.12, 0.010)), ("LB", (0.010, 0.002, 0.12, 0.002)),
                        ("DL", (0.004, 0.001, 0.03, 0.0003))):
        for k, v in zip(KINDS, base):
            eng.mu[(gname, k)] = [v * 200.0, 200.0]
    eng.lg_ds = [30.0, 100.0]; eng.lg_epa = [0.0, 100.0]; eng.lg_pm = [0.0, 100.0]
    for k, v in zip(KINDS, (0.14, 0.03, 0.55, 0.015)):
        eng.lgT[k] = [v * 200.0, 200.0]
    Tg = defaultdict(list)
    for r in T.to_dict("records"):
        Tg[r["game_id"]].append(r)
    Eg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for gid, dt, pid, kind, cr, epa in E.itertuples(index=False):
        c = Eg[(gid, dt)][pid]
        c[kind] += cr
        if kind in ("pd", "int", "tc"):
            c["atepa"] += cr * epa
        c["act"] += 0.0
    G = Gm[Gm.season <= seasons_max].sort_values(["game_date", "game_id"])
    out = {"player": [], "unit": []} if collect else None
    for date, grp in G.groupby("game_date", sort=True):
        s = int(grp.season.iloc[0])
        if eng.season is not None and s != eng.season:
            eng.new_season()
        eng.season = s
        eng.step(list(grp.game_id), Tg, Eg, out)
    return eng, out


def posmap_from(E):
    pl = pd.read_csv("data/nfl_players.csv", usecols=["gsis_id", "position"], dtype=str)
    m = {r.gsis_id: pos_group(r.position) for r in pl.itertuples(index=False)}
    return {p: m.get(p) for p in E.pid.unique() if m.get(p) is not None}


def linear_weights(d, seasons):
    """EPA linear weights of each credited event vs an average attempt (warm-up seasons only)."""
    x = d[d.season.isin(seasons)]
    att = x[(x.play_type == "pass") & (x.sack.fillna(0) != 1) & (x.qb_spike.fillna(0) != 1)
            & (x.two_point_attempt.fillna(0) != 1)]
    base = att.epa.mean()
    pdm = att.pass_defense_1_player_id.notna() & (att.interception.fillna(0) != 1)
    comp = (att.complete_pass.fillna(0) == 1) & (att.interception.fillna(0) != 1)
    tkd = comp & (att.solo_tackle_1_player_id.notna() | att.tackle_with_assist_1_player_id.notna()
                  | att.assist_tackle_1_player_id.notna())
    cp = x[(x.penalty.fillna(0) == 1) & (x.penalty_team == x.defteam) & x.penalty_type.isin(COV_PEN)
           & (x.play_type == "no_play")]
    w = {"int": -(att[att.interception.fillna(0) == 1].epa.mean() - base),
         "pd": -(att[pdm].epa.mean() - base),
         "tc": -(att[tkd].epa.mean() - base),
         "cpen": -(cp.epa.mean() - base)}
    return {k: float(v) for k, v in w.items()}, float(base)


def mom_priors(player_rows, seasons):
    """Gamma prior strength a = 1/var(true theta) by method of moments, per group and kind."""
    P = pd.DataFrame(player_rows)
    P = P[P.season.isin(seasons)]
    out = {}
    for gname in GROUPS:
        x = P[P.group == gname].groupby(["player_id", "season"])[
            [f"o_{k}" for k in KINDS] + [f"e_{k}" for k in KINDS]].sum()
        for k in KINDS:
            o, e = x[f"o_{k}"], x[f"e_{k}"]
            keep = e > 1.0
            if keep.sum() < 50:                     # too little warm-up mass (e.g. DL coverage
                out[f"{gname}|{k}"] = {"a": 50.0, "var_true": None,       # penalties): heavy default
                                       "n_player_seasons": int(keep.sum())}
                continue
            o, e = o[keep], e[keep]
            r = o / e
            # weighted MoM: E[(r-1)^2] - E[1/e] (Poisson noise around theta) with ratio re-centred
            rbar = o.sum() / e.sum()
            r = r / rbar
            var_true = float(np.average((r - 1.0) ** 2, weights=e) - np.average(1.0 / (e * rbar), weights=e))
            var_true = max(var_true, 0.01)
            out[f"{gname}|{k}"] = {"a": float(min(1.0 / var_true, 200.0)), "var_true": var_true,
                                   "n_player_seasons": int(keep.sum())}
    return out


def mom_eff_priors(player_rows, seasons):
    """Player-level shrink for DS, EPA/AT (targets) and PM (dropbacks): k = noise var / true var."""
    P = pd.DataFrame(player_rows)
    P = P[P.season.isin(seasons)]
    out = {}
    for gname in GROUPS:
        x = P[P.group == gname]
        s = x.groupby(["player_id", "season"]).agg(nat=("o_at", "sum"), d=("o_def", "sum"), eds=("e_ds", "sum"),
                                                     ep=("o_atepa", "sum"), eep=("e_atepa", "sum"),
                                                     pr=("team_pass_epa", "sum"), pe=("pm_exp", "sum"),
                                                     nd=("team_dropbacks", "sum"))
        s = s[s.nat >= 10]
        r = (s.d - s.eds) / s.nat
        pbar = (s.d / s.nat).mean()
        vt = r.var() - (pbar * (1 - pbar) / s.nat).mean()
        out[f"{gname}|ds"] = float(pbar * (1 - pbar) / max(vt, 1e-4))
        g = x[x.o_at > 0]
        pv = float((((g.o_atepa - g.e_atepa) ** 2) / g.o_at).mean())
        re = (s.ep - s.eep) / s.nat
        vt = re.var() - (pv / s.nat).mean()
        out[f"{gname}|epa"] = float(pv / max(vt, 1e-3))
        g = x[x.team_dropbacks > 0]
        pv = float((((g.team_pass_epa - g.pm_exp) ** 2) / g.team_dropbacks).mean())
        rp = (s.pr - s.pe) / s.nd
        vt = rp.var() - (pv / s.nd).mean()
        out[f"{gname}|pm"] = float(pv / max(vt, 1e-4))
    return {k: float(min(max(v, 5.0), 5000.0)) for k, v in out.items()}


def r6_ztables(player_rows, seasons):
    P = pd.DataFrame(player_rows)
    P = P[P.season.isin(seasons)]
    z = {}
    for gname in GROUPS:
        x = P[P.group == gname]
        # per player-season per-game rates (R6 ingredients reconstructed)
        by = x.groupby(["player_id", "season"]).agg(n=("o_pd", "size"), pd_=("o_pd", "sum"),
                                                     int_=("o_int", "sum"), solo=("solo", "sum"),
                                                     rush=("r6rush", "sum"))
        by = by[by.n >= 4]
        ball = (3 * by.int_ + by.pd_) / by.n
        tak = by.solo / by.n
        rush = by.rush / by.n
        z[gname] = {"rush": (float(rush.mean()), float(rush.std())), "ball": (float(ball.mean()), float(ball.std())),
                    "tak": (float(tak.mean()), float(tak.std()))}
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true",
                    help="serving run over every pulled season; writes silently, prints nothing")
    args = ap.parse_args()
    WARM = list(range(1999, 2006))
    df = load(all_seasons=args.all_seasons)
    Gm, T, E = build_tables(df)
    posmap = posmap_from(E)
    lw, base = linear_weights(df, WARM)
    # R6 rush ingredient per player-game for z-tables
    # pass 1: provisional priors, warm-up only, to set MoM priors + R6 z-tables
    prov = {(gname, k): {"pd": 10.0, "int": 20.0, "tc": 10.0, "cpen": 10.0}[k] for gname in GROUPS for k in KINDS}
    r6z0 = {gname: {"rush": (0.2, 0.3), "ball": (0.5, 0.5), "tak": (2.0, 1.5)} for gname in GROUPS}
    _, out1 = run(Gm, T, E, prov, lw, posmap, r6z0, seasons_max=max(WARM))
    rushmap = E[E.kind.isin(["sack", "tfl", "hit"])].copy()
    rushmap["w"] = np.where(rushmap.kind == "sack", 1.0, 0.5) * rushmap.credit
    rush = rushmap.groupby(["game_id", "pid"]).w.sum().to_dict()
    for r in out1["player"]:
        r["r6rush"] = rush.get((r["game_id"], r["player_id"]), 0.0)
    pri = mom_priors(out1["player"], WARM)
    effp = mom_eff_priors(out1["player"], WARM)
    r6z = r6_ztables(out1["player"], WARM)
    prior_a = {(gname, k): pri[f"{gname}|{k}"]["a"] for gname in GROUPS for k in KINDS}
    eff_prior = {(gname, m): effp[f"{gname}|{m}"] for gname in GROUPS for m in ("ds", "epa", "pm")}
    smax = int(df.season.max()) if args.all_seasons else TEST_ERA - 1
    eng, out = run(Gm, T, E, prior_a, lw, posmap, r6z, seasons_max=smax, eff_prior=eff_prior)
    P = pd.DataFrame(out["player"])
    U = pd.DataFrame(out["unit"])
    U["pts_allowed"] = U.merge(T[["game_id", "defteam", "pts_allowed"]], on=["game_id", "defteam"],
                               how="left").pts_allowed.values
    if args.all_seasons:
        P.to_parquet("data/pv_nfl_coverage_serve_player_games.parquet", index=False)
        U.to_parquet("data/pv_nfl_coverage_serve_unit_games.parquet", index=False)
        return
    assert P.season.max() < TEST_ERA and U.season.max() < TEST_ERA
    P.to_parquet("data/pv_nfl_coverage_player_games.parquet", index=False)
    U.to_parquet("data/pv_nfl_coverage_unit_games.parquet", index=False)
    json.dump({"note": "set on warm-up seasons 1999-2005 only; engine hyper-parameters in source",
               "linear_weights_defence_positive": lw, "mean_epa_attempt_warmup": base,
               "gamma_prior_strength": pri, "efficiency_prior_strength": effp, "r6_ztables_warmup": r6z,
               "dynamics": {"P_DECAY": P_DECAY, "P_SEASON": P_SEASON, "Q": [Q_DECAY, Q_SEASON, Q_PRIOR],
                            "S": [S_DECAY, S_SEASON, S_PRIOR], "D": [D_DECAY, D_SEASON, D_PRIOR],
                            "L_DECAY": L_DECAY, "DS_PRIOR": DS_PRIOR, "EPA_PRIOR": EPA_PRIOR,
                            "PM_PRIOR": PM_PRIOR, "R6": [R6_DECAY, R6_PRIOR], "UNIT_DECAY": UNIT_DECAY}},
              open("data/pv_nfl_coverage_priors.json", "w"), indent=1)
    print("linear weights (defence-positive EPA vs avg attempt, warm-up):", {k: round(v, 3) for k, v in lw.items()})
    print("MoM prior strengths (warm-up):", {k: round(v["a"], 1) for k, v in pri.items()})
    print("MoM efficiency priors (warm-up):", {k: round(v, 1) for k, v in effp.items()})
    print("player-game rows", len(P), "unit rows", len(U), "seasons", P.season.min(), "-", P.season.max())


if __name__ == "__main__":
    sys.exit(main())
