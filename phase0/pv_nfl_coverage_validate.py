"""Player-value program, NFL - COVERAGE: player-level validation on DEV (2006-2015) ONLY.

Reads the walk-forward outputs of phase0/pv_nfl_coverage.py (seasons <= 2015 only; asserted).
  1. RELIABILITY   split-half (odd/even appearances within a season) and year-over-year, per stat,
                   DB and LB, with n; YoY split into team-STAYERS vs team-MOVERS (portability: a stat
                   that measures the player should survive a change of team; one that measures the
                   team - plus-minus - should not).
  2. PREDICTIVE    (a) player: the rating he carries into season s+1 (state after season s, strictly
                   pre-game) vs his own s+1 outcomes and his team's s+1 pass-defence outcomes in the
                   games he played; (b) UNIT: pre-game unit aggregate vs that game's opponent passing
                   OUTCOMES (net yards/dropback, completion %, INT rate, pass-TD rate, points allowed)
                   over a baseline of the team's own walk-forward pass-defence EPA rating, the
                   opponent's walk-forward pass-offence rating and the opposing passer's offset.
  3. FACE          top/bottom DBs by the carried rating at their last DEV appearance.
  4. GAME          ONE pre-declared indication on the shipped DEV walk-forward harness
                   (bt_nfl_ideas_dev_X.npy, 14 cols, LL 0.622937): add unit-coverage diff.
Output: data/pv_nfl_coverage_validation.json. No odds are read. No season >= 2016 row exists.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
RNG = np.random.default_rng(20260924)
OUT = {}

P = pd.read_parquet("data/pv_nfl_coverage_player_games.parquet")
U = pd.read_parquet("data/pv_nfl_coverage_unit_games.parquet")
assert P.season.max() < TEST_ERA and U.season.max() < TEST_ERA
NM = pd.read_csv("data/nfl_players.csv", usecols=["gsis_id", "display_name", "position"], dtype=str).set_index("gsis_id")

SUMS = ["o_pd", "e_pd", "o_int", "e_int", "o_tc", "e_tc", "o_cpen", "e_cpen", "o_at", "o_def", "e_ds",
        "o_atepa", "e_atepa", "team_pass_epa", "pm_exp", "team_dropbacks", "solo", "att"]


def stats(a):
    """Per-row stats from summed ingredients (defence-positive where it is a value)."""
    o = pd.DataFrame(index=a.index)
    o["ball"] = (a.o_pd + a.o_int) / (a.e_pd + a.e_int)
    o["pd"] = a.o_pd / a.e_pd
    o["int"] = a.o_int / a.e_int
    o["tc"] = a.o_tc / a.e_tc
    o["cpen"] = a.o_cpen / a.e_cpen
    o["ds"] = (a.o_def - a.e_ds) / a.o_at
    o["ds_raw"] = a.o_def / a.o_at
    o["epa_at"] = -(a.o_atepa - a.e_atepa) / a.o_at
    o["pm"] = -(a.team_pass_epa - a.pm_exp) / a.team_dropbacks
    o["r6ball"] = (3 * a.o_int + a.o_pd) / a.games
    o["r6tak"] = a.solo / a.games
    o["ball_raw"] = (a.o_pd + a.o_int) / a.att
    o["at_rate"] = a.o_at / a.att
    return o


PER_TARGET = {"ds", "ds_raw", "epa_at"}
STAT_LIST = ["ball", "pd", "int", "tc", "cpen", "ds", "ds_raw", "epa_at", "pm", "r6ball", "r6tak",
             "ball_raw", "at_rate"]


def corr_ci(x, y, B=1000):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 20:
        return {"r": None, "n": int(len(x))}
    r = float(np.corrcoef(x, y)[0, 1])
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    xb, yb = x[idx], y[idx]
    xb = xb - xb.mean(1, keepdims=True); yb = yb - yb.mean(1, keepdims=True)
    rb = (xb * yb).sum(1) / np.sqrt((xb ** 2).sum(1) * (yb ** 2).sum(1))
    lo, hi = np.percentile(rb, [2.5, 97.5])
    return {"r": round(r, 3), "ci": [round(float(lo), 3), round(float(hi), 3)], "n": int(len(x))}


D = P[(P.season >= DEV_LO) & (P.season <= DEV_HI)].copy()
D["games"] = 1.0
D = D.sort_values(["player_id", "season", "week"])

# ============================================================================ 1. reliability ====
print("1. RELIABILITY (DEV 2006-2015)")
D["half"] = D.groupby(["player_id", "season"]).cumcount() % 2
H = D.groupby(["player_id", "season", "group", "half"])[SUMS + ["games"]].sum()
H0 = H.xs(0, level="half"); H1 = H.xs(1, level="half")
common = H0.index.intersection(H1.index)
H0, H1 = H0.loc[common], H1.loc[common]
S0, S1 = stats(H0), stats(H1)
rel = {}
for grp in ("DB", "LB"):
    g = common.get_level_values("group") == grp
    rel[grp] = {}
    for st in STAT_LIST:
        ok = g & (H0.games.values >= 4) & (H1.games.values >= 4)
        if st in PER_TARGET:
            ok &= (H0.o_at.values >= 10) & (H1.o_at.values >= 10)
        c = corr_ci(S0[st].values[ok], S1[st].values[ok], B=500)
        if c["r"] is not None:
            c["spearman_brown_full_season"] = round(2 * c["r"] / (1 + c["r"]), 3) if c["r"] > -1 else None
        rel[grp][st] = c
    print(f"  split-half {grp}: " + ", ".join(f"{k} {v['r']}(n{v['n']})" for k, v in rel[grp].items()))
OUT["split_half"] = rel

# year over year (full-season stats), stayers vs movers
SEAS = D.groupby(["player_id", "season", "group"])[SUMS + ["games"]].sum()
team_mode = D.groupby(["player_id", "season"]).team.agg(lambda s: s.value_counts().index[0])
SS = stats(SEAS).join(SEAS[["games", "o_at"]])
SS = SS.reset_index()
SS["team"] = [team_mode[(p, s)] for p, s in zip(SS.player_id, SS.season)]
nxt = SS.copy(); nxt["season"] -= 1
Y2 = SS.merge(nxt, on=["player_id", "season", "group"], suffixes=("", "_n"))
yoy = {}
for grp in ("DB", "LB"):
    yoy[grp] = {}
    for st in STAT_LIST:
        base = (Y2.group == grp) & (Y2.games >= 8) & (Y2.games_n >= 8)
        if st in PER_TARGET:
            base &= (Y2.o_at >= 25) & (Y2.o_at_n >= 25)
        res = {}
        for nm_, m in (("all", base), ("stayers", base & (Y2.team == Y2.team_n)),
                       ("movers", base & (Y2.team != Y2.team_n))):
            res[nm_] = corr_ci(Y2.loc[m, st], Y2.loc[m, st + "_n"], B=500)
        yoy[grp][st] = res
    print(f"  YoY {grp}: " + ", ".join(
        f"{k} {v['all']['r']}/{v['stayers']['r']}/{v['movers']['r']}(n{v['all']['n']}/{v['movers']['n']})"
        for k, v in yoy[grp].items()))
OUT["year_over_year"] = yoy

# ================================================================= 2a. player predictive ======
print("2a. PLAYER PREDICTIVE: rating carried into s+1 (state after s) vs s+1 outcomes")
first = D.groupby(["player_id", "season"]).head(1).set_index(["player_id", "season"])
PRED = ["pre_th_ball", "pre_th_int", "pre_dsp", "pre_epap", "pre_cv_eff", "pre_cv", "pre_pm", "pre_r6"]
pp = SS.set_index(["player_id", "season"]).join(first[PRED + ["pre_games", "pre_at_n", "team"]], rsuffix="_first")
pp["prev_team"] = [team_mode.get((p, s - 1), None) for p, s in pp.index]
pp["mover"] = pp.prev_team.notna() & (pp.prev_team != pp.team_first)
# team outcomes in the games he played (s+1): opponent net yards/dropback, comp%, INT rate, points
U["ny_db"] = U.y_pass_yds / U.y_dropbacks
U["comp_pct"] = U.y_comp / U.y_n_att
U["int_rate"] = U.y_ints / U.y_n_att
U["td_rate"] = U.y_pass_td / U.y_n_att
U["epa_db"] = U.y_pass_epa / U.y_dropbacks
tg = D.merge(U[["game_id", "defteam", "ny_db", "comp_pct", "int_rate", "td_rate", "epa_db", "pts_allowed"]],
             left_on=["game_id", "team"], right_on=["game_id", "defteam"], how="left")
tout = tg.groupby(["player_id", "season"])[["ny_db", "comp_pct", "int_rate", "td_rate", "epa_db", "pts_allowed"]].mean()
pp = pp.join(tout)
pv = {}
for grp in ("DB",):
    base = (pp.group == grp) & (pp.games >= 8) & (pp.pre_games >= 8) & (pp.index.get_level_values("season") >= DEV_LO + 1)
    pv[grp] = {}
    outs = {"own_ball": ("ball", 1), "own_int": ("int", 1), "own_ds": ("ds", 1), "own_epa_at": ("epa_at", 1),
            "team_ny_db_allowed": ("ny_db", -1), "team_comp_pct_allowed": ("comp_pct", -1),
            "team_int_rate": ("int_rate", 1), "team_td_rate_allowed": ("td_rate", -1),
            "team_points_allowed": ("pts_allowed", -1), "team_epa_db_allowed": ("epa_db", -1)}
    for on, (col, sgn) in outs.items():
        pv[grp][on] = {}
        for pr in PRED:
            m = base.copy()
            if col in PER_TARGET:
                m &= pp.o_at >= 25
            pv[grp][on][pr] = {"all": corr_ci(pp.loc[m, pr], sgn * pp.loc[m, col]),
                               "movers": corr_ci(pp.loc[m & pp.mover, pr], sgn * pp.loc[m & pp.mover, col])}
        print(f"  {grp} -> {on:<24} " + " ".join(
            f"{k[4:]} {v['all']['r']}|{v['movers']['r']}" for k, v in pv[grp][on].items()))
OUT["player_predictive"] = pv
OUT["player_predictive_note"] = ("r = corr(carried rating, s+1 outcome), outcomes sign-flipped so + = rating "
                                 "predicts BETTER defence; 'movers' = changed team since previous season "
                                 "(new-team outcomes predicted from old-team data). DBs with >=8 games in s+1 "
                                 "and >=8 prior games, s+1 in 2007-2015.")

# ==================================================================== 2b. unit predictive ======
print("2b. UNIT PREDICTIVE (team-game, pre-game unit vs that game's opponent passing outcomes)")
# walk-forward team pass-defence and pass-offence ratings (shipped pass-channel dynamics)
DEC, SDEC, PN = 0.8813581205848026, 0.7130881070009863, 754.0438610349224 / 2
gd = pd.read_parquet("data/pv_nfl_events.parquet", columns=["game_id", "game_date", "season"],
                     filters=[("season", "<", TEST_ERA)]).drop_duplicates("game_id").set_index("game_id")
assert gd.season.max() < TEST_ERA
U["game_date"] = U.game_id.map(gd.game_date)
U = U.sort_values(["game_date", "game_id"]).reset_index(drop=True)
dstate, ostate, lg = {}, {}, [0.0, 0.0]
dr, orr = np.zeros(len(U)), np.zeros(len(U))
cur_season = None
rows_by_date = U.groupby("game_date", sort=True).indices
for date in sorted(rows_by_date):
    idx = rows_by_date[date]
    s = int(U.season.iloc[idx[0]])
    if cur_season is not None and s != cur_season:
        for st in (dstate, ostate):
            for k in st:
                st[k] = [st[k][0] * SDEC, st[k][1] * SDEC]
    cur_season = s
    lgm = lg[0] / lg[1] if lg[1] else 0.0
    for i in idx:
        dt, ot = U.defteam.iat[i], U.offteam.iat[i]
        a = dstate.get(dt, [0.0, 0.0]); b = ostate.get(ot, [0.0, 0.0])
        dr[i] = -((a[0] + PN * lgm) / (a[1] + PN) - lgm)      # defence-positive
        orr[i] = (b[0] + PN * lgm) / (b[1] + PN) - lgm
    for i in idx:
        dt, ot = U.defteam.iat[i], U.offteam.iat[i]
        e, n = U.y_pass_epa.iat[i], U.y_dropbacks.iat[i]
        a = dstate.get(dt, [0.0, 0.0]); dstate[dt] = [a[0] * DEC + e, a[1] * DEC + n]
        b = ostate.get(ot, [0.0, 0.0]); ostate[ot] = [b[0] * DEC + e, b[1] * DEC + n]
        lg = [lg[0] * 0.999 + e, lg[1] * 0.999 + n]
U["def_pass_rating"] = dr
U["opp_pass_rating"] = orr
U["is_home"] = (U.defteam == U.home).astype(float)
U["ny_db"] = U.y_pass_yds / U.y_dropbacks
U["comp_pct"] = U.y_comp / U.y_n_att
U["int_rate"] = U.y_ints / U.y_n_att
U["td_rate"] = U.y_pass_td / U.y_n_att
U["epa_db"] = U.y_pass_epa / U.y_dropbacks
UD = U[(U.season >= DEV_LO) & (U.season <= DEV_HI) & (U.week >= 2) & (U.y_dropbacks > 0)].reset_index(drop=True)
BASE = ["def_pass_rating", "opp_pass_rating", "q_pm", "is_home"]
CANDS = ["dblb_val", "dblb_epap", "dblb_dsp", "dblb_ball", "dblb_int", "dblb_cv", "dblb_pm", "dblb_r6",
         "db_val", "db_epap", "db_ball", "db_int", "db_pm", "all_r6"]
OUTC = {"ny_db": -1, "comp_pct": -1, "int_rate": 1, "td_rate": -1, "pts_allowed": -1, "epa_db": -1}
gids = UD.game_id.values
ug, ginv = np.unique(gids, return_inverse=True)
members = [np.where(ginv == k)[0] for k in range(len(ug))]


def ols_r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    return 1 - r.var() / y.var(), b


def zs(v):
    return (v - v.mean()) / v.std()


unit = {}
Bn = 400
boot_idx = [np.concatenate([members[k] for k in RNG.integers(0, len(ug), len(ug))]) for _ in range(Bn)]
for oc, sgn in OUTC.items():
    y = sgn * UD[oc].values.astype(float)
    ok = np.isfinite(y)
    Xb = np.column_stack([zs(UD[c].values.astype(float)) for c in BASE])
    r2b, _ = ols_r2(Xb[ok], y[ok])
    unit[oc] = {"baseline_r2": round(float(r2b), 5), "n": int(ok.sum()), "cands": {}}
    for c in CANDS:
        x = zs(UD[c].values.astype(float))
        raw = float(np.corrcoef(x[ok], y[ok])[0, 1])
        r2c, b = ols_r2(np.column_stack([Xb, x])[ok], y[ok])
        # partial correlation of the candidate with the outcome given the baseline
        _, bx = ols_r2(Xb[ok], x[ok]); _, by = ols_r2(Xb[ok], y[ok])
        X1 = np.column_stack([np.ones(ok.sum()), Xb[ok]])
        rx = x[ok] - X1 @ bx; ry = y[ok] - X1 @ by
        pr = float(np.corrcoef(rx, ry)[0, 1])
        # cluster bootstrap (by game) of the partial correlation
        rxf = np.full(len(y), np.nan); ryf = np.full(len(y), np.nan); rxf[ok] = rx; ryf[ok] = ry
        pb = []
        for bi in boot_idx:
            a_, b_ = rxf[bi], ryf[bi]
            m = np.isfinite(a_) & np.isfinite(b_)
            pb.append(np.corrcoef(a_[m], b_[m])[0, 1])
        lo, hi = np.percentile(pb, [2.5, 97.5])
        unit[oc]["cands"][c] = {"raw_r": round(raw, 4), "partial_r": round(pr, 4),
                                "partial_ci": [round(float(lo), 4), round(float(hi), 4)],
                                "delta_r2": round(float(r2c - r2b), 5), "std_coef": round(float(b[-1]), 5)}
    # the current team-level rating on its own, for scale
    unit[oc]["def_pass_rating_raw_r"] = round(float(np.corrcoef(Xb[ok, 0], y[ok])[0, 1]), 4)
    best = sorted(unit[oc]["cands"].items(), key=lambda kv: -kv[1]["partial_r"])[:4]
    print(f"  {oc:<11} base R2 {r2b:.4f} (def_pass raw r {unit[oc]['def_pass_rating_raw_r']:+.3f}) | " +
          " ".join(f"{k} raw {v['raw_r']:+.3f} partial {v['partial_r']:+.3f}{v['partial_ci']}" for k, v in best))
OUT["unit_predictive"] = unit
OUT["unit_predictive_note"] = ("team-game rows DEV 2006-2015, week>=2; outcomes sign-flipped so + = better defence; "
                               "baseline = team walk-forward pass-defence EPA rating (shipped pass-channel dynamics) + "
                               "opponent pass-offence rating + opposing passer offset + home; partial_r = candidate's "
                               "correlation with the outcome after both are residualised on the baseline; CI = "
                               "game-cluster bootstrap (400).")
# the full table for the two headline outcomes
for oc in ("ny_db", "pts_allowed"):
    print(f"   all candidates on {oc}: " + " ".join(
        f"{k} {v['partial_r']:+.3f}" for k, v in unit[oc]["cands"].items()))

# ======================================================= 2c. adversarial: attribution or memory? ==
print("2c. ADVERSARIAL: does the unit signal survive the TEAM's own history of the same events?")
# team-game sums of the same credited events, then walk-forward team EWMAs with the PLAYER memory
tgs = P.groupby(["game_id", "team"])[["o_pd", "o_int", "e_pd", "e_int"]].sum()
U["t_ball_o"] = [tgs.o_pd.get((g, t), 0.0) + tgs.o_int.get((g, t), 0.0) for g, t in zip(U.game_id, U.defteam)]
U["t_ball_e"] = [tgs.e_pd.get((g, t), 0.0) + tgs.e_int.get((g, t), 0.0) for g, t in zip(U.game_id, U.defteam)]
HIST = {"slow_def_pass": [], "team_ball_hist": [], "team_comp_hist": [], "team_int_hist": [], "team_ny_hist": []}
st = {}
cur_season = None
vals = {k: np.zeros(len(U)) for k in HIST}
rows_by_date = U.groupby("game_date", sort=True).indices
PD_, PS_ = 0.97, 0.75      # the player engine's memory (per appearance / season boundary)
for date in sorted(rows_by_date):
    idx = rows_by_date[date]
    s_ = int(U.season.iloc[idx[0]])
    if cur_season is not None and s_ != cur_season:
        for k in st:
            st[k] = [v * PS_ for v in st[k]]
    cur_season = s_
    for i in idx:
        t = U.defteam.iat[i]
        a = st.get(t, [0.0] * 10)
        # prior pseudo-counts: 300 dropbacks at league-typical values (constants, not fitted)
        vals["slow_def_pass"][i] = -(a[0] + 300 * 0.0) / (a[1] + 300)
        vals["team_ball_hist"][i] = (a[2] + 20.0) / (a[3] + 20.0)
        vals["team_comp_hist"][i] = -(a[4] + 300 * 0.60) / (a[5] + 300)
        vals["team_int_hist"][i] = (a[6] + 300 * 0.03) / (a[5] + 300)
        vals["team_ny_hist"][i] = -(a[7] + 300 * 6.0) / (a[8] + 300)
    for i in idx:
        t = U.defteam.iat[i]
        a = st.get(t, [0.0] * 10)
        inc = [U.y_pass_epa.iat[i], U.y_dropbacks.iat[i], U.t_ball_o.iat[i], U.t_ball_e.iat[i],
               U.y_comp.iat[i], U.y_n_att.iat[i], U.y_ints.iat[i], U.y_pass_yds.iat[i], U.y_dropbacks.iat[i], 0.0]
        st[t] = [PD_ * x + y for x, y in zip(a, inc)]
for k in HIST:
    U[k] = vals[k]
UD2 = U[(U.season >= DEV_LO) & (U.season <= DEV_HI) & (U.week >= 2) & (U.y_dropbacks > 0)].reset_index(drop=True)
assert len(UD2) == len(UD) and (UD2.game_id.values == UD.game_id.values).all()
BASES = {"B0 shipped-style": BASE,
         "B1 +slow team pass-D": BASE + ["slow_def_pass"],
         "B2 +team history of the same events": BASE + ["slow_def_pass", "team_ball_hist", "team_comp_hist",
                                                        "team_int_hist", "team_ny_hist"]}
adv = {}
for oc in ("comp_pct", "int_rate", "ny_db", "pts_allowed", "epa_db"):
    y = OUTC[oc] * UD2[oc].values.astype(float)
    ok = np.isfinite(y)
    adv[oc] = {}
    for bn, cols in BASES.items():
        Xb = np.column_stack([zs(UD2[c].values.astype(float)) for c in cols])[ok]
        X1 = np.column_stack([np.ones(ok.sum()), Xb])
        by, *_ = np.linalg.lstsq(X1, y[ok], rcond=None); ry = y[ok] - X1 @ by
        adv[oc][bn] = {}
        for c in ("dblb_dsp", "dblb_ball", "dblb_int", "dblb_val", "dblb_pm", "all_r6"):
            x = zs(UD2[c].values.astype(float))[ok]
            bx, *_ = np.linalg.lstsq(X1, x, rcond=None); rx = x - X1 @ bx
            pr = float(np.corrcoef(rx, ry)[0, 1])
            rxf = np.full(len(y), np.nan); ryf = np.full(len(y), np.nan); rxf[ok] = rx; ryf[ok] = ry
            pb = []
            for bi in boot_idx[:200]:
                a_, b_ = rxf[bi], ryf[bi]; m = np.isfinite(a_) & np.isfinite(b_)
                pb.append(np.corrcoef(a_[m], b_[m])[0, 1])
            lo, hi = np.percentile(pb, [2.5, 97.5])
            adv[oc][bn][c] = {"partial_r": round(pr, 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}
        print(f"  {oc:<11} {bn:<38} " + " ".join(f"{c} {v['partial_r']:+.3f}" for c, v in adv[oc][bn].items()))
OUT["unit_adversarial"] = adv
OUT["unit_adversarial_note"] = ("B1 adds a team pass-D EPA rating with the PLAYER engine's memory (0.97/game, 0.75/season); "
                                "B2 also adds the team's own walk-forward history of breakups+INTs (O/E), completion %, "
                                "INT rate and net yards allowed. A unit signal that survives B2 is information carried by "
                                "WHO is on the field (roster composition), not a longer memory of the team's events.")

# ============================================================================ 3. face validity ==
print("3. FACE VALIDITY (DBs, carried rating at last DEV appearance, >=48 DEV games, >=150 attributed targets)")
last = D.sort_values(["player_id", "season", "week"]).groupby("player_id").tail(1)
car = D.groupby("player_id").agg(g=("games", "sum"), at=("o_at", "sum"), first=("season", "min"), last_s=("season", "max"),
                                 ints=("o_int", "sum"), pds=("o_pd", "sum"))
last = last.set_index("player_id").join(car)
last["name"] = last.index.map(NM.display_name); last["pos"] = last.index.map(NM.position)
fv = last[(last.group == "DB") & (last.g >= 48) & (last["at"] >= 150)]
face = {}
for col in ("pre_cv_eff", "pre_epap", "pre_dsp", "pre_th_ball", "pre_pm", "pre_r6"):
    t = fv.sort_values(col, ascending=False)
    cols = ["name", "pos", "team", "first", "last_s", "g", "ints", "pds", col]
    face[col] = {"top": t[cols].head(15).round(4).to_dict("records"), "bottom": t[cols].tail(10).round(4).to_dict("records")}
    print(f"  {col} TOP: " + "; ".join(f"{r['name']}({r['pos']},{r['last_s']})" for r in face[col]["top"][:12]))
    print(f"  {col} BOT: " + "; ".join(f"{r['name']}({r['pos']},{r['last_s']})" for r in face[col]["bottom"]))
ROLE = {"CB": {"CB"}, "S": {"S", "SS", "FS", "SAF"}}
for role, labs in ROLE.items():
    fr = fv[fv.pos.isin(labs)]
    for col in ("pre_cv_eff", "pre_th_ball", "pre_dsp"):
        t = fr.sort_values(col, ascending=False)
        cols = ["name", "pos", "team", "first", "last_s", "g", "ints", "pds", col]
        face[f"{role}|{col}"] = {"n": int(len(fr)), "top": t[cols].head(12).round(4).to_dict("records"),
                                 "bottom": t[cols].tail(8).round(4).to_dict("records")}
        print(f"  [{role} n={len(fr)}] {col} TOP: " + "; ".join(f"{r['name']}({r['last_s']})" for r in face[f"{role}|{col}"]["top"]))
        print(f"  [{role}] {col} BOT: " + "; ".join(f"{r['name']}({r['last_s']})" for r in face[f"{role}|{col}"]["bottom"]))
OUT["face_validity"] = face
OUT["face_n_players"] = int(len(fv))
# correlation between the rating families among these players
OUT["rating_intercorr_DB_career"] = fv[["pre_cv_eff", "pre_epap", "pre_dsp", "pre_th_ball", "pre_pm", "pre_r6"]].corr().round(3).to_dict()

# ================================================================= 4. game-level indication ===
print("4. GAME-LEVEL INDICATION (shipped DEV harness, one pre-declared feature)")
Dj = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
GG = Dj["games"]
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
SE = np.array([g["season"] for g in GG]); assert SE.max() < TEST_ERA
Y = np.array([g["y"] for g in GG], float)
HL, C = 3.0, 100.0


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wf(extra=None, drop=None):
    X = X14 if drop is None else np.delete(X14, drop, axis=1)
    vec, per = [], {}
    for s_ in range(DEV_LO, DEV_HI + 1):
        tr = (SE < s_) & (Y != 0.5); te = SE == s_
        Xs = X
        if extra is not None:
            E_ = extra.reshape(len(Y), -1)
            mu, sd = E_[tr].mean(0), E_[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E_ - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SE[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1]); vec.append(v); per[s_] = float(v.mean())
    v = np.concatenate(vec)
    return float(v.mean()), v, per


b_ll, b_vec, b_per = wf()
assert abs(b_ll - 0.622937) < 5e-4, b_ll
d_ll, d_vec, _ = wf(drop=[1])
print(f"  harness baseline DEV LL {b_ll:.6f} (ledger 0.622937); drop QB col -> {d_ll:.6f} (cost {d_ll - b_ll:+.5f})")
# faster: pivot home/away defence rows
U["side"] = np.where(U.defteam == U.home, "h", "a")
PV = U.pivot_table(index="game_id", columns="side", values=["dblb_val", "dblb_ball", "dblb_pm"], aggfunc="first")
game = {}
feats = {}
for col in ("dblb_val", "dblb_ball", "dblb_pm"):
    f = np.array([(PV.loc[g["gid"], (col, "h")] - PV.loc[g["gid"], (col, "a")]) if g["gid"] in PV.index else 0.0
                  for g in GG], float)
    feats[col] = np.nan_to_num(f)
cov_ok = np.array([g["gid"] in PV.index for g in GG])
game["spine_rows_with_feature"] = int(cov_ok.sum()); game["spine_rows"] = int(len(GG))
game["baseline_dev_ll"] = round(b_ll, 6); game["drop_qb_cost"] = round(d_ll - b_ll, 5)
for nm_, f in (("PRE-DECLARED dblb_val (unit EPA saved/att via attributed targets)", feats["dblb_val"]),
               ("secondary dblb_ball (unit ball-hawk volume)", feats["dblb_ball"]),
               ("comparator dblb_pm (unit on-field plus-minus)", feats["dblb_pm"])):
    ll, vec, per = wf(extra=f)
    dlt = b_vec - vec
    bs = dlt[RNG.integers(0, len(dlt), size=(4000, len(dlt)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    game[nm_] = {"ll": round(ll, 6), "gain": round(b_ll - ll, 6), "ci": [round(float(lo), 6), round(float(hi), 6)],
                 "seasons_pos": pos}
    print(f"  {nm_:<66} LL {ll:.6f} gain {b_ll - ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10")
OUT["game_level_indication"] = game
OUT["game_level_note"] = ("feature = home-defence unit minus away-defence unit (pre-game, walk-forward), z-scored on "
                          "the train fold, added as a 15th column to the shipped 14-col DEV harness; bar +0.00150.")
json.dump(OUT, open("data/pv_nfl_coverage_validation.json", "w"), indent=1, default=str)
print("wrote data/pv_nfl_coverage_validation.json")
