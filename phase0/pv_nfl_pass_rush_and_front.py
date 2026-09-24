"""Player-value program (pv), NFL, component `pass_rush_and_front` -- step 2: WALK-FORWARD RATINGS.

The MLB principle applied to the defensive front:
  1. Rate each defender ONLY on events credited to him: sacks, non-sack QB hits
     (pass rush), run tackles for loss and run STOPS (tackles on runs that fail
     the yards rule) -- no on-field plus-minus.
  2. Every credited event is one side of a matchup: the rusher vs the offence
     that protects (QB + OL, rated as a unit because nobody records which
     lineman lost), the run defender vs the offence's run game. Expected events
     for a defender = his exposure x league rate x OPPONENT factor x SCORER
     factor (QB hits are logged by the home stats crew; on DEV the stadium
     factor of non-sack hits replicates across split halves BETTER than the
     defence factor, 0.33 vs 0.16 -- validation json section 0). Sacks (-0.14)
     and stops (0.04) show no scorer effect and carry no scorer factor.
  3. Walk-forward and empirical-Bayes: every factor is a decayed ratio
     (credits + K*prior) / (expected + K*1) updated only AFTER the game it
     describes; games are processed in date batches, so no game sees any game
     played the same day or later. Priors: position-bucket mean from strictly
     earlier seasons (players), 1.0 (unit / scorer factors).
  4. Output per player per game: the PRE-GAME ratings (theta = the player's
     production as a share of an average unit's, against an average opponent
     and scorer), exposure n_eff, and an as-of expected-presence weight that
     aggregates the players into a unit value without looking at who played.

Exposure. Nothing records who was on the field before 2013 (snap files start
2013 on DEV). The rating's exposure is the TEAM's dropbacks (pass rush) or
designed runs (run front) in games the player is PRESENT, where present =
credited with any defensive event in the game, or credited in his team's
previous game of the same season (one-game fill, walk-forward; on 2013-15 snap
truth it lifts recall for 20-40%-snap linemen from 0.77 to ~0.9). `--exposure
snap` switches seasons with snap files to exposure = team plays x defense_pct
(sub-analysis only; the main rating is consistent across all seasons).

Offensive line. OL players are never credited with protection events. The
unit factors O_pr/O_sk/O_st (pressure / sacks / run stops allowed per
opportunity, opponent- and scorer-adjusted) ARE the line's rating, shared with
its QB (sack avoidance) and run game. Written per team-game; distributed to
individual linemen by snap share only where snaps exist (2013-15 on DEV), see
pv_nfl_pass_rush_and_front_ol.csv.

Columns. th_* are PRE-GAME ratings (walk-forward: state before this date).
g_* columns are the game's REALIZED credits / unit counts (post-game outcomes,
kept only so validation can score the ratings) -- never a feature for that
game. presence: 'credit' | 'fill' | '' (expected-roster row, not present);
a_exp = pre-game expected-presence weight used by the agg_* unit aggregates.
Positions come from nfl_players.csv (latest listed position -- identity
metadata that only sets the EB prior bucket).

Outputs (default: seasons <= 2015 only; --through >= 2016 is for the lead
engineer's serving build and prints nothing about TEST seasons):
  data/pv_nfl_pass_rush_and_front_player_games.parquet (+ .csv)
  data/pv_nfl_pass_rush_and_front_team_games.csv
  data/pv_nfl_pass_rush_and_front_ol_games.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA = 2016
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}

# kinds: credit column, unit numerator column, exposure ('db' dropbacks / 'run' designed runs), scorer factor?
KINDS = {"pr": ("pr", "pressure", "db", True),
         "sk": ("sack", "sacks", "db", False),
         "ht": ("hit", "hits_ns", "db", True),
         "st": ("stop", "stop_plays", "run", False),
         "tf": ("tfl", "tfl_plays", "run", False)}

DEFAULT = dict(
    # player (DEV grid, pv_nfl_pass_rush_and_front_tune.json: pass-rush kinds optimum K 12 / d .97 /
    # s .70-.85; run kinds want faster dynamics -- stops are partly ROLE, which changes fast)
    K_p=12.0, d_p=0.97, s_p=0.75,
    K_p_st=16.0, d_p_st=0.94, s_p_st=0.55, K_p_tf=16.0, d_p_tf=0.94, s_p_tf=0.55,
    # unit (defence / offence factors)
    K_u=25.0, d_u=0.96, s_u=0.55,
    # scorer (stadium) factor for credited hits
    K_s=40.0, d_s=0.98, s_s=0.85,
    # league rate: weight (in opportunities) of the previous season's rate at a season's start
    M_l=300.0,
    fill=1,                # one-game presence fill
    rho=0.5,               # expected-presence EWMA per team game
    exposure="team",       # or 'snap' (seasons with snap files only)
)

BUCKET = {"DE": "DE", "DT": "DT", "NT": "DT", "DL": "DT", "OLB": "OLB", "ILB": "ILB", "MLB": "ILB",
          "LB": "ILB", "CB": "CB", "DB": "S", "S": "S", "SS": "S", "FS": "S", "SAF": "S"}


def pos_bucket(pos):
    return BUCKET.get(pos, "OTH")


def load_inputs(through):
    pg = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_credits.parquet"))
    u = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_units.parquet"))
    sn = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_snaps.parquet"))
    pg = pg[pg.season <= through].copy()
    u = u[u.season <= through].copy()
    sn = sn[sn.season <= through].copy()
    assert pg.season.max() <= through and u.season.max() <= through
    pg["pr"] = pg.sack + pg.hit
    pl = pd.read_csv(os.path.join(DATA, "nfl_players.csv"), usecols=["gsis_id", "display_name", "position"])
    pos = dict(zip(pl.gsis_id, pl.position))
    names = dict(zip(pl.gsis_id, pl.display_name))
    g = pd.read_csv(os.path.join(DATA, "nfl_games.csv"), usecols=["game_id", "home_team", "season"])
    g = g[g.season <= through]
    stad = {r.game_id: FR.get(r.home_team, r.home_team) for r in g.itertuples()}
    return pg, u, sn, pos, names, stad


def build(params=None, through=TEST_ERA - 1, inputs=None, emit_all=True):
    P = dict(DEFAULT)
    P.update(params or {})
    pg, u, sn, POS, NAMES, STAD = inputs if inputs is not None else load_inputs(through)

    # ------------------------------------------------ per-game lookups
    games = (u.groupby("game_id").agg(date=("game_date", "first"), season=("season", "first"),
                                      week=("week", "first"), stype=("season_type", "first")).reset_index()
             .sort_values(["date", "game_id"]))
    urow = {(r.game_id, r.defteam): r for r in u.itertuples(index=False)}
    GT = defaultdict(list)
    for (g_, t_) in urow:
        GT[g_].append(t_)
    cred = defaultdict(dict)                       # (gid, team) -> {pid: row}
    ccols = ["sack", "hit", "pr", "stop", "tfl", "rtk", "pen_presnap", "pen_rtp"]
    arr = pg[["game_id", "team", "player_id"] + ccols].to_numpy()
    for r in arr:
        cred[(r[0], r[1])][r[2]] = r[3:]
    snap = {}
    if len(sn):
        for r in sn[["game_id", "team", "player_id", "defense_pct", "defense_snaps"]].itertuples(index=False):
            snap[(r.game_id, r.team, r.player_id)] = float(r.defense_pct) if r.defense_snaps > 0 else 0.0
    snap_games = set(sn.game_id) if len(sn) else set()

    kinds = list(KINDS)
    # ------------------------------------------------ state
    # league
    L_cur = {k: [0.0, 0.0] for k in kinds}         # season-to-date (events, opps)
    L_prev = {k: None for k in kinds}              # previous season rate (None = not recorded)
    # unit factors: D[k][team] = [num, den]; O[k][team]; S[k][stadium]
    D = {k: defaultdict(lambda: [0.0, 0.0]) for k in kinds}
    O = {k: defaultdict(lambda: [0.0, 0.0]) for k in kinds}
    S = {k: defaultdict(lambda: [0.0, 0.0]) for k in kinds}
    # players: PL[pid][k] = [num, den]; raw EWMA sums for comparators
    PL = defaultdict(lambda: {k: [0.0, 0.0] for k in kinds})
    PLN = defaultdict(float)                       # decayed present games (n)
    PL_last_season = {}
    # position priors from strictly earlier seasons: POSS[bucket][k] = [num, den] pooled
    POSS = defaultdict(lambda: {k: [0.0, 0.0] for k in kinds})
    POS_frozen = {}
    # rosters / presence
    ROSTER = defaultdict(dict)                     # team -> {pid: a_exp}
    PTEAM = {}
    last_cred = {}                                 # (team) -> set of pids credited in the team's last game
    team_season = {}

    def fac(store, key, K):
        n, d = store[key]
        return (n + K) / (d + K)

    def league(k, min_n=1000.0):
        e, n = L_cur[k]
        prev = L_prev[k]
        if prev is None:
            return e / n if n >= min_n else None
        return (e + P["M_l"] * prev) / (n + P["M_l"])

    def theta(pid, k, prior):
        n, d = PL[pid][k]
        K = P.get(f"K_p_{k}", P["K_p"])
        return (n + K * prior) / (d + K)

    def pos_prior(b, k):
        f = POS_frozen.get(b, POS_frozen.get("ALL", {}))
        return f.get(k, 0.05)

    out_rows, team_rows, ol_rows = [], [], []
    cur_season = None
    for date, day in games.groupby("date", sort=True):
        season = int(day.season.iloc[0])
        if season != cur_season:
            # ---- season boundary: close league rates, freeze position priors, carry decay
            if cur_season is not None:
                for k in kinds:
                    e, n = L_cur[k]
                    rate = e / n if n > 0 else 0.0
                    lv = [v for v in (L_prev[k],) if v]
                    L_prev[k] = rate if (rate > 0.2 * (lv[0] if lv else rate) and rate > 0.003) else None
                    L_cur[k] = [0.0, 0.0]
                for k in kinds:
                    for st in D[k].values():
                        st[0] *= P["s_u"]; st[1] *= P["s_u"]
                    for st in O[k].values():
                        st[0] *= P["s_u"]; st[1] *= P["s_u"]
                    for st in S[k].values():
                        st[0] *= P["s_s"]; st[1] *= P["s_s"]
                for pid, st in PL.items():
                    for k in kinds:
                        sk_ = P.get(f"s_p_{k}", P["s_p"])
                        st[k][0] *= sk_; st[k][1] *= sk_
                    PLN[pid] *= P["s_p"]
            # position priors: pooled from all seasons strictly before this one
            POS_frozen.clear()
            allk = {k: [0.0, 0.0] for k in kinds}
            for b, dct in POSS.items():
                POS_frozen[b] = {}
                for k in kinds:
                    n_, d_ = dct[k]
                    POS_frozen[b][k] = n_ / d_ if d_ > 50 else None
                    allk[k][0] += n_; allk[k][1] += d_
            POS_frozen["ALL"] = {k: (allk[k][0] / allk[k][1] if allk[k][1] > 50 else None) for k in kinds}
            for k in kinds:
                if POS_frozen["ALL"][k] is None:
                    POS_frozen["ALL"][k] = POS_frozen["ALL"].get("sk") or 0.05
            for b in list(POS_frozen):
                for k in kinds:
                    if POS_frozen[b][k] is None:
                        # a kind never recorded before (QB hits before 2006): shares are comparable
                        # across kinds, so start from the bucket's sack share
                        alt = POS_frozen[b].get("sk") if k in ("pr", "ht") else None
                        POS_frozen[b][k] = alt if alt is not None else POS_frozen["ALL"][k]
            last_cred.clear()
            cur_season = season
        Lk = {k: league(k) for k in kinds}

        # ---------------- PRE-GAME values for every game on this date
        updates = []
        for gid in day.game_id:
            # both defensive sides of this game
            dts = GT[gid]
            for dteam in dts:
                ur = urow[(gid, dteam)]
                oteam = ur.posteam
                stadium = STAD.get(gid, dteam)
                n_db, n_run = float(ur.n_db), float(ur.n_run)
                creds = cred.get((gid, dteam), {})
                # presence
                present = {}
                for pid, c in creds.items():
                    present[pid] = "credit"
                if P["fill"]:
                    for pid in last_cred.get(dteam, set()):
                        if pid not in present and PTEAM.get(pid) == dteam:
                            present[pid] = "fill"
                # expected roster (pre-game)
                ros = ROSTER[dteam]
                # unit factors pre-game
                Dv = {k: fac(D[k], dteam, P["K_u"]) for k in kinds}
                Ov = {k: fac(O[k], oteam, P["K_u"]) for k in kinds}   # opponent offence allowed factor
                Sv = {k: (fac(S[k], stadium, P["K_s"]) if KINDS[k][3] else 1.0) for k in kinds}
                # offence's own protection factors (this team as offence) for the team table
                agg = {k: 0.0 for k in kinds}
                aggw = 0.0
                emit = set(present) | {p for p, a in ros.items() if a >= 0.05}
                for pid in emit:
                    b = pos_bucket(POS.get(pid))
                    th = {k: theta(pid, k, pos_prior(b, k)) for k in kinds}
                    a = ros.get(pid, 0.0)
                    if a >= 0.05:          # aggregate = pre-game roster only; presence in THIS game never enters
                        for k in kinds:
                            agg[k] += a * th[k]
                        aggw += a
                    sp = snap.get((gid, dteam, pid)) if gid in snap_games else None
                    c = creds.get(pid)
                    out_rows.append((gid, season, int(ur.week), date, dteam, oteam, pid, b,
                                     present.get(pid, ""), sp, a,
                                     th["pr"], th["sk"], th["ht"], th["st"], th["tf"],
                                     PL[pid]["pr"][1], PL[pid]["st"][1], PLN[pid],
                                     *(c[:5] if c is not None else (0.0,) * 5)))
                team_rows.append((gid, season, int(ur.week), date, dteam, oteam, stadium,
                                  Dv["pr"], Dv["sk"], Dv["ht"], Dv["st"], Dv["tf"],
                                  Ov["pr"], Ov["sk"], Ov["ht"], Ov["st"], Ov["tf"], Sv["ht"],
                                  agg["pr"], agg["sk"], agg["ht"], agg["st"], agg["tf"], aggw,
                                  *[Lk[k] if Lk[k] is not None else np.nan for k in kinds],
                                  n_db, n_run, float(ur.sacks), float(ur.hits_ns), float(ur.pressure),
                                  float(ur.stop_plays), float(ur.tfl_plays), float(ur.rsucc),
                                  float(ur.pts_for) if ur.pts_for == ur.pts_for else np.nan))
                updates.append((gid, dteam, oteam, stadium, ur, present, creds, Dv, Ov, Sv))

        # ---------------- UPDATE after the whole date batch
        # the league normaliser used for UPDATES may include this date's games (post-game information;
        # pre-game values above used only earlier dates)
        for (_g, _d, _o, _s, ur, *_rest) in updates:
            for k, (ccol, ucol, expo, use_s) in KINDS.items():
                n_ = float(ur.n_db) if expo == "db" else float(ur.n_run)
                L_cur[k][0] += float(getattr(ur, ucol)); L_cur[k][1] += n_
        Lk = {k: league(k, 300.0) for k in kinds}
        for (gid, dteam, oteam, stadium, ur, present, creds, Dv, Ov, Sv) in updates:
            n_db, n_run = float(ur.n_db), float(ur.n_run)
            season_ = int(ur.season)
            for k, (ccol, ucol, expo, use_s) in KINDS.items():
                if Lk[k] is None:
                    continue
                n = n_db if expo == "db" else n_run
                y = float(getattr(ur, ucol))
                base = n * Lk[k]
                # unit factors (decay then add)
                for store, key, other in ((D[k], dteam, Ov[k] * Sv[k]), (O[k], oteam, Dv[k] * Sv[k])):
                    st = store[key]
                    st[0] = P["d_u"] * st[0] + y
                    st[1] = P["d_u"] * st[1] + base * other
                if use_s:
                    st = S[k][stadium]
                    st[0] = P["d_s"] * st[0] + y
                    st[1] = P["d_s"] * st[1] + base * Dv[k] * Ov[k]
                # players
                for pid, how in present.items():
                    c = creds.get(pid)
                    val = 0.0
                    if c is not None:
                        val = float(c[["sack", "hit", "pr", "stop", "tfl"].index(ccol)])
                    expo_n = n
                    if P["exposure"] == "snap" and gid in snap_games:
                        sp = snap.get((gid, dteam, pid))
                        if sp is None or sp <= 0:
                            continue
                        expo_n = n * sp
                    e = expo_n * Lk[k] * Ov[k] * Sv[k]
                    st = PL[pid][k]
                    dk_ = P.get(f"d_p_{k}", P["d_p"])
                    st[0] = dk_ * st[0] + val
                    st[1] = dk_ * st[1] + e
                    b = pos_bucket(POS.get(pid))
                    pst = POSS[b][k]
                    pst[0] += val; pst[1] += e
            for pid in present:
                PLN[pid] = P["d_p"] * PLN[pid] + 1.0
            # presence / rosters
            for pid in present:
                old = PTEAM.get(pid)
                if old is not None and old != dteam:
                    ROSTER[old].pop(pid, None)
                PTEAM[pid] = dteam
            ros = ROSTER[dteam]
            for pid in list(ros):
                ros[pid] = P["rho"] * ros[pid] + (1 - P["rho"]) * (1.0 if pid in present else 0.0)
                if ros[pid] < 0.02 and pid not in present:
                    del ros[pid]
            for pid in present:
                if pid not in ros:
                    ros[pid] = 1 - P["rho"]
            last_cred[dteam] = set(creds)

    pcols = ["game_id", "season", "week", "game_date", "team", "opp", "player_id", "bucket", "presence",
             "snap_pct", "a_exp", "th_pr", "th_sk", "th_ht", "th_st", "th_tf", "neff_pr", "neff_st",
             "n_games", "g_sack", "g_hit", "g_pr", "g_stop", "g_tfl"]
    PGo = pd.DataFrame(out_rows, columns=pcols)
    tcols = ["game_id", "season", "week", "game_date", "defteam", "posteam", "stadium",
             "D_pr", "D_sk", "D_ht", "D_st", "D_tf", "Oopp_pr", "Oopp_sk", "Oopp_ht", "Oopp_st", "Oopp_tf",
             "S_ht", "agg_pr", "agg_sk", "agg_ht", "agg_st", "agg_tf", "agg_w",
             "L_pr", "L_sk", "L_ht", "L_st", "L_tf",
             "g_n_db", "g_n_run", "g_sacks", "g_hits_ns", "g_pressure", "g_stops", "g_tfl", "g_rsucc",
             "g_pts_allowed"]
    TGo = pd.DataFrame(team_rows, columns=tcols)
    return PGo, TGo, NAMES


def ol_table(TGo, through):
    """Distribute the offence's pre-game protection factors to linemen by snap share (snap seasons only)."""
    sn = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_snaps.parquet"))
    sn = sn[(sn.season <= through) & sn.position.isin(["T", "G", "C", "OL", "OT"]) & (sn.offense_snaps > 0)]
    olp = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_olpen.parquet"))
    olp = olp[olp.season <= through]
    # the offence's factors live on the OPPONENT's defensive row (Oopp_* of the row where posteam = team)
    off = TGo.rename(columns={"posteam": "team"})[["game_id", "team", "Oopp_pr", "Oopp_sk", "Oopp_st"]]
    m = sn.merge(off, on=["game_id", "team"], how="left")
    m = m.merge(olp[["game_id", "team", "player_id", "olpen_hold", "olpen_fs"]], on=["game_id", "team", "player_id"],
                how="left").fillna({"olpen_hold": 0.0, "olpen_fs": 0.0})
    m["share"] = m.offense_pct / 5.0
    for k in ("pr", "sk", "st"):
        # value: expected events PREVENTED vs an average unit, per 100 dropbacks, x the player's share
        m[f"ol_val_{k}"] = (1.0 - m[f"Oopp_{k}"]) * m.share
    return m[["game_id", "season", "week", "team", "player_id", "position", "offense_snaps", "offense_pct",
              "Oopp_pr", "Oopp_sk", "Oopp_st", "ol_val_pr", "ol_val_sk", "ol_val_st", "olpen_hold", "olpen_fs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", type=int, default=TEST_ERA - 1)
    ap.add_argument("--exposure", default="team")
    a = ap.parse_args()
    quiet = a.through >= TEST_ERA
    PGo, TGo, NAMES = build({"exposure": a.exposure}, through=a.through)
    PGo["name"] = PGo.player_id.map(NAMES)
    suf = "" if a.exposure == "team" else "_snapexpo"
    PGo.to_parquet(os.path.join(DATA, f"pv_nfl_pass_rush_and_front_player_games{suf}.parquet"), index=False)
    if a.exposure == "team":
        PGo.to_csv(os.path.join(DATA, "pv_nfl_pass_rush_and_front_player_games.csv"), index=False,
                   float_format="%.6g")
        TGo.to_csv(os.path.join(DATA, "pv_nfl_pass_rush_and_front_team_games.csv"), index=False, float_format="%.6g")
        ol_table(TGo, a.through).to_csv(os.path.join(DATA, "pv_nfl_pass_rush_and_front_ol_games.csv"), index=False,
                                        float_format="%.6g")
    if quiet:
        print("built (TEST seasons included; no statistics printed)")
        return
    print(f"player-game rows {len(PGo):,} | team-game rows {len(TGo):,}")
    print(PGo[PGo.season >= 2006].groupby("bucket")[["th_pr", "th_st"]].mean().round(4))


if __name__ == "__main__":
    sys.exit(main())
