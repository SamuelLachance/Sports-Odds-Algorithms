"""Player-value program (pv), NFL receiving_rushing, STEP 2: the rating walk.

MLB principle -> skill players. Each target and each designed carry is a small
matchup on outcomes the ball-handler influences, measured against the
walk-forward situation baseline of STEP 1 (phase0/pv_nfl_receiving_rushing_xmodels.py).

  TARGET (receiver + his QB vs the pass defense); y_t = yards on the target
  (air + YAC on a completion, 0 otherwise)
    comp  : complete - xcomp                  catch over expected (given depth)
    yac   : yac - xyac   (completions only)   yards after catch over expected
    epat  : epa - xepa                        EPA per target over expected
    xepat : xepa - league                     opportunity quality in EPA units
    xypt  : x_ypt - league                    OPPORTUNITY QUALITY: expected yards
                                              of the targets he earns (depth,
                                              field, down) = xcomp*(air+xyac)
    yptoe : y_t - x_ypt                       execution: yards/target over expected
    ypta  : y_t - league   (P+Q+D)            results, teammate/opponent adjusted
    rawt  : y_t - league   (P only)           naive results (comparison)
  CARRY  (rusher + his blocking unit vs the run defense)
    ryds  : clip(yds,-5,20) - xyds            rush yards over expected
    rsuc  : (epa>0) - xsucc                   success over expected
    epar  : epa - xepa                        EPA per carry over expected
    ypca  : clip(yds,-5,20) - league (P+O+D)  results, line/opponent adjusted
    rawr  : clip(yds,-5,20) - league (P only) naive results (comparison)

Each component is a CONTINUOUS-OUTCOME TRUESKILL (diagonal Gaussian assumed-
density filter): every entity carries (mu, var); an event's residual y is the
sum of the involved entities' mu plus noise R; each entity takes a share of the
surprise proportional to its own variance (K_i = var_i / (sum var + R)).
Entities: the player; for targets also the passer (Q) and the defense (D), for
carries the offense's blocking unit (O) and the defense (D). The player is
credited net of teammates and opponents, the way the MLB PA duel separates
batter from pitcher.

EMPIRICAL BAYES + WALK-FORWARD: a new entity enters at mu=0 with var = V
(method of moments on the WARM-UP seasons only: targets 2006-2008, carries
1999-2005; the player's V is then scaled by a selected factor because the raw
between-player spread also contains QB/line/opponent spread). var drifts by
tau per game played and widens / regresses at each season boundary. Every value
written for a game is the entity state BEFORE that game. Hyper-parameters are
chosen per component by the filter's own one-step-ahead predictive MSE on
2007-2010 events (never on a validation outcome); 2011-2015 is untouched by
selection.

USAGE: target share and carry share per game appeared (EWMA per appearance,
EB toward a small prior). No snap data exist on DEV, so "appeared" = at least
one target or carry.

SHARED-CREDIT PROXY (the 11v11 plus-minus analog on DEV): every offensive
pass/run play of a game credited to ALL skill players who appeared for that
offense, against the defense -- the participation idea with game-level
appearance standing in for the 2016+-only snap lists.

Usage
  python phase0/pv_nfl_receiving_rushing_build.py --select-comp comp,yac   # selection (parallel-safe)
  python phase0/pv_nfl_receiving_rushing_build.py                          # full walk with selected params

Outputs
  data/pv_nfl_receiving_rushing_select_<comp>.json    selection per component
  data/pv_nfl_receiving_rushing_events.parquet        per target/carry residuals
  data/pv_nfl_receiving_rushing_player_games.parquet  per player per game: PRE-GAME
        state of every component + usage + proxy, and (prefixed y_) that game's
        realised outcomes, for validation only
  data/pv_nfl_receiving_rushing_values.parquet / .csv  the same PRE-GAME values WITHOUT any
        same-game outcome column (the feature-safe file) + composites v_e (EPA/target above
        average = xepat + epat) and v_t (yards/target above average = xypt + yptoe)
  data/pv_nfl_receiving_rushing_params.json

LOCKED SPLIT: reads only seasons <= --through (default 2015). Nothing printed
for any season > 2015.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DEV_LAST = 2015

ap = argparse.ArgumentParser()
ap.add_argument("--through", type=int, default=DEV_LAST)
ap.add_argument("--select-comp", default="", help="comma list of components to run the selection grid for")
args = ap.parse_args()
THROUGH = args.through
QUIET = THROUGH > DEV_LAST
t0 = time.time()


def log(*a):
    if not QUIET:
        print(*a, flush=True)


PL = pd.read_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_plays.parquet"))
PL = PL[PL.season <= THROUGH].reset_index(drop=True)
assert PL.season.max() <= THROUGH
PL["passer_id"] = PL["passer_id"].astype(object)
T = PL.kind.to_numpy() == "T"
C = PL.complete_pass.to_numpy() == 1
# yards on the target (air + YAC on a completion, 0 otherwise) and its expectation
PL["y_t"] = np.where(T, np.where(C, PL.air_yards + PL.yac_c, 0.0), np.nan)
PL["x_ypt"] = np.where(T, PL.xcomp * (PL.air_yards + PL.xyac), np.nan)
PL["ypc_c"] = np.where(~T, PL.yds_c, np.nan)


def prev_league_mean(col):
    """walk-forward league mean: mean over the previous (up to 3) seasons; in the first season with
    data (warm-up: 2006 targets / 1999 carries) the expanding mean over that season's EARLIER weeks
    (NaN in its first week -> no residual, no update)."""
    lg = PL.groupby("season")[col].agg(["sum", "count"])
    lg = lg[lg["count"] > 0]
    first = int(lg.index.min())
    prev = {s: float(lg.loc[(lg.index < s) & (lg.index >= s - 3), "sum"].sum()
                     / lg.loc[(lg.index < s) & (lg.index >= s - 3), "count"].sum()) for s in lg.index if s > first}
    out = PL.season.map(prev).to_numpy(float).copy()
    fs = (PL.season == first).to_numpy()
    wk = PL.loc[fs].groupby("week")[col].agg(["sum", "count"]).sort_index()
    cs, cc = wk["sum"].cumsum().shift(1), wk["count"].cumsum().shift(1)
    exp_mean = (cs / cc).to_dict()
    out[fs] = PL.loc[fs, "week"].map(exp_mean).to_numpy(float)
    return out


PL["r_comp"] = np.where(T, PL.complete_pass - PL.xcomp, np.nan)
PL["r_yac"] = np.where(T & C, PL.yac_c - PL.xyac, np.nan)
PL["r_epat"] = np.where(T, PL.epa - PL.xepa, np.nan)
PL["r_xypt"] = PL.x_ypt - prev_league_mean("x_ypt")
PL["r_yptoe"] = PL.y_t - PL.x_ypt
PL["xepa_t"] = np.where(T, PL.xepa, np.nan)
PL["r_xepat"] = PL.xepa_t - prev_league_mean("xepa_t")     # opportunity quality in EPA units
PL["r_ypta"] = PL.y_t - prev_league_mean("y_t")
PL.loc[PL.xcomp.isna(), "r_ypta"] = np.nan  # same event set as the other target components
PL["r_rawt"] = PL["r_ypta"]
PL["r_ryds"] = np.where(~T, PL.yds_c - PL.xyds, np.nan)
PL["r_rsuc"] = np.where(~T, PL.succ - PL.xsucc, np.nan)
PL["r_epar"] = np.where(~T, PL.epa - PL.xepa, np.nan)
PL["r_ypca"] = PL.ypc_c - prev_league_mean("ypc_c")
PL.loc[PL.xyds.isna(), "r_ypca"] = np.nan
PL["r_rawr"] = PL["r_ypca"]

COMPS = {  # component -> (kind, residual col, entity roles)
    "comp": ("T", "r_comp", ("P", "Q", "D")),
    "yac": ("T", "r_yac", ("P", "Q", "D")),
    "epat": ("T", "r_epat", ("P", "Q", "D")),
    "xypt": ("T", "r_xypt", ("P", "Q", "D")),
    "yptoe": ("T", "r_yptoe", ("P", "Q", "D")),
    "xepat": ("T", "r_xepat", ("P", "Q", "D")),
    "ypta": ("T", "r_ypta", ("P", "Q", "D")),
    "rawt": ("T", "r_rawt", ("P",)),
    "ryds": ("R", "r_ryds", ("P", "O", "D")),
    "rsuc": ("R", "r_rsuc", ("P", "O", "D")),
    "epar": ("R", "r_epar", ("P", "O", "D")),
    "ypca": ("R", "r_ypca", ("P", "O", "D")),
    "rawr": ("R", "r_rawr", ("P",)),
}
WARM = {"T": (2006, 2008), "R": (1999, 2005)}


# ---------------------------------------------------------------- EB moments
def moments(comp):
    kind, col, _ = COMPS[comp]
    lo, hi = WARM[kind]
    d = PL[(PL.kind == kind) & PL[col].notna() & PL.season.between(lo, hi)]
    # noise R: within-entity variance, approximated by the total event variance
    R = float(d[col].var())
    out = {"R": R}
    keys = {"P": d.player_id, "Q": d.passer_id, "O": d.posteam, "D": d.defteam}
    for role in COMPS[comp][2]:
        g = d.groupby([keys[role].astype(str), d.season])[col].agg(["mean", "count"])
        g = g[g["count"] >= 20]
        w = g["count"].to_numpy(float)
        m = g["mean"].to_numpy()
        mu = np.average(m, weights=w)
        between = np.average((m - mu) ** 2, weights=w) - np.average(R / w, weights=w)
        out["V" + role] = float(max(between, R / 2000.0))
    return out


MOM = {c: moments(c) for c in COMPS}


# ---------------------------------------------------------------- the filter
class GaussTS:
    """Continuous-outcome TrueSkill (diagonal Gaussian ADF) for one component."""

    def __init__(self, R, V, tau_f, rho, widen_f):
        self.R, self.V, self.tau_f, self.rho, self.widen_f = R, V, tau_f, rho, widen_f
        self.st = {}  # key -> [mu, var, last_gidx, last_season, n]

    def touch(self, key, role, gidx, season):
        s = self.st.get(key)
        V = self.V[role]
        if s is None:
            s = [0.0, V, gidx, season, 0.0]
            self.st[key] = s
            return s
        if s[3] != season:  # season boundary(ies)
            gap = season - s[3]
            s[0] *= self.rho[role] ** gap
            s[1] = min(V, s[1] + V * self.widen_f[role] * gap)
            s[3] = season
        if s[2] != gidx:
            s[1] = min(V, s[1] + V * self.tau_f[role])
            s[2] = gidx
        return s

    @staticmethod
    def update(states, y, R):
        pred = 0.0
        S = R
        for s in states:
            pred += s[0]
            S += s[1]
        e = y - pred
        for s in states:
            k = s[1] / S
            s[0] += k * e
            s[1] -= k * s[1]
            s[4] += 1.0
        return pred


def make_filter(comp, prm):
    m = MOM[comp]
    roles = COMPS[comp][2]
    V = {r: m["V" + r] * (prm.get("vs", 1.0) if r == "P" else 1.0) for r in roles}
    return GaussTS(m["R"], V, prm["tau_f"], prm["rho"], prm["widen_f"])


def default_params():
    return {"vs": 1.0, "tau_f": {"P": 0.0, "Q": 0.0, "O": 0.005, "D": 0.005},
            "rho": {"P": 0.9, "Q": 0.9, "O": 0.5, "D": 0.5},
            "widen_f": {"P": 0.1, "Q": 0.1, "O": 0.5, "D": 0.5}}


KEYS = {"P": PL.player_id.astype(str).to_numpy(),
        "Q": ("Q:" + PL.passer_id.astype(str)).to_numpy(),
        "O": ("O:" + PL.posteam.astype(str)).to_numpy(),
        "D": ("D:" + PL.defteam.astype(str)).to_numpy()}
GIDX = PL.gidx.to_numpy()
SEAS = PL.season.to_numpy().astype(int)
KIND = PL.kind.to_numpy()


def run_comp(comp, prm):
    kind, col, roles = COMPS[comp]
    f = make_filter(comp, prm)
    y = PL[col].to_numpy()
    idx = np.where((KIND == kind) & ~np.isnan(y))[0]
    pred = np.full(len(PL), np.nan)
    adj = np.full(len(PL), np.nan)
    R = f.R
    for i in idx:
        g, s = GIDX[i], SEAS[i]
        sts = [f.touch(KEYS[r][i], r, g, s) for r in roles]
        adj[i] = y[i] - sum(st[0] for st in sts[1:])  # net of PRE-event teammate/opponent states
        pred[i] = f.update(sts, y[i], R)
    return f, pred, adj


# ------------------------------------------------- hyper-parameter selection
if args.select_comp:
    assert not QUIET
    for comp in args.select_comp.split(","):
        kind, col, _ = COMPS[comp]
        rhos = (0.8, 0.9, 1.0) if kind == "T" else (0.5, 0.7, 0.9)
        grid = []
        for vs in (0.35, 0.7, 1.4):
            for tf in (0.0, 0.004):
                for rho in rhos:
                    for wf in (0.0, 0.15):
                        grid.append({"vs": vs, "tau_f": {"P": tf, "Q": tf, "O": 0.005, "D": 0.005},
                                     "rho": {"P": rho, "Q": rho, "O": 0.5, "D": 0.5},
                                     "widen_f": {"P": wf, "Q": wf, "O": 0.5, "D": 0.5}})
        y = PL[col].to_numpy()
        msk = (KIND == kind) & ~np.isnan(y) & (SEAS >= 2007) & (SEAS <= 2010)
        base = float(np.mean(y[msk] ** 2))
        res = []
        for prm in grid:
            _, pred, _ = run_comp(comp, prm)
            res.append(base - float(np.mean((y[msk] - pred[msk]) ** 2)))
        j = int(np.argmax(res))
        out = {"comp": comp, "best": grid[j], "gain_pct": 100 * res[j] / base,
               "grid": [[g["vs"], g["tau_f"]["P"], g["rho"]["P"], g["widen_f"]["P"], round(100 * r / base, 4)]
                        for g, r in zip(grid, res)]}
        json.dump(out, open(os.path.join(DATA, f"pv_nfl_receiving_rushing_select_{comp}.json"), "w"), indent=1)
        b = grid[j]
        log(f"  {comp:>5}: vs {b['vs']} tau {b['tau_f']['P']} rho {b['rho']['P']} widen {b['widen_f']['P']} "
            f"| one-step MSE gain {100*res[j]/base:.3f}%  ({time.time()-t0:.0f}s)")
    raise SystemExit(0)

SEL = {}
for p in glob.glob(os.path.join(DATA, "pv_nfl_receiving_rushing_select_*.json")):
    d = json.load(open(p))
    SEL[d["comp"]] = d
PRM = {c: (SEL[c]["best"] if c in SEL else default_params()) for c in COMPS}
for c, m in MOM.items():
    log(f"  {c:>5}: R {m['R']:.4f} | " + " ".join(f"sd_{k[1]} {np.sqrt(v):.4f}" for k, v in m.items()
                                                  if k.startswith("V"))
        + (f" | selected vs {PRM[c]['vs']} tau {PRM[c]['tau_f']['P']} rho {PRM[c]['rho']['P']} "
           f"widen {PRM[c]['widen_f']['P']}" if c in SEL else " | DEFAULT params"))

# ------------------------------------------------------------ per-event walk
for comp in COMPS:
    _, pred, adj = run_comp(comp, PRM[comp])
    PL["p_" + comp] = pred
    PL["a_" + comp] = adj
PL.to_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_events.parquet"), index=False)
log(f"  per-event walk done ({time.time()-t0:.0f}s)")

# ------------------------------------------------ pre-game player snapshots
filters = {c: make_filter(c, PRM[c]) for c in COMPS}
Y = {c: PL[COMPS[c][1]].to_numpy() for c in COMPS}

US_DECAY = 0.85        # per appearance
US_PRIOR_N = 2.0       # games of prior
US_PRIOR = {"tsh": 0.05, "csh": 0.05}
usage = defaultdict(lambda: {"tsh": 0.0, "tsh_w": 0.0, "csh": 0.0, "csh_w": 0.0, "season": None, "g": 0})

# per player-game counts and realised outcomes (outcomes are validation targets only, never features)
PL["_rec"] = np.where(T, PL.complete_pass, 0.0)
PL["_recyds"] = np.where(T & C, PL.yards_gained, 0.0)
PL["_rectd"] = np.where(T & C, PL.touchdown, 0.0)
PL["_recfd"] = np.where(T & C, PL.first_down, 0.0)
PL["_epat"] = np.where(T, PL.epa, 0.0)
PL["_isT"] = T.astype(float)
PL["_isR"] = (~T).astype(float)
PL["_rushyds"] = np.where(~T, PL.yards_gained, 0.0)
PL["_rushtd"] = np.where(~T, PL.touchdown, 0.0)
PL["_rushfd"] = np.where(~T, PL.first_down, 0.0)
PL["_rsucc"] = np.where(~T, PL.succ, 0.0)
PL["_epar"] = np.where(~T, PL.epa, 0.0)
PL["_air"] = np.where(T, PL.air_yards, 0.0)
PL["_xypt"] = np.where(T, PL.x_ypt, 0.0)
agg = PL.groupby(["gidx", "game_id", "season", "week", "player_id", "posteam"], sort=True).agg(
    y_tgt=("_isT", "sum"), y_rec=("_rec", "sum"), y_recyds=("_recyds", "sum"), y_rectd=("_rectd", "sum"),
    y_recfd=("_recfd", "sum"), y_epat=("_epat", "sum"), y_air=("_air", "sum"), y_xypt=("_xypt", "sum"),
    y_car=("_isR", "sum"), y_rushyds=("_rushyds", "sum"), y_rushtd=("_rushtd", "sum"), y_rushfd=("_rushfd", "sum"),
    y_rsucc=("_rsucc", "sum"), y_epar=("_epar", "sum")).reset_index()
team_tot = agg.groupby(["gidx", "posteam"])[["y_tgt", "y_car"]].sum()
agg = agg.join(team_tot.rename(columns={"y_tgt": "team_tgt", "y_car": "team_car"}), on=["gidx", "posteam"])
AGG_BY_G = {g: d for g, d in agg.groupby("gidx", sort=True)}
IDX_BY_G = {g: d.index.to_numpy() for g, d in PL.groupby("gidx", sort=True)}

# shared-credit proxy inputs: all offensive pass/run plays (incl. sacks / scrambles)
EV = pd.read_parquet(os.path.join(DATA, "pv_nfl_events.parquet"),
                     columns=["game_id", "play_id", "season", "posteam", "defteam", "play_type", "epa"],
                     filters=[("season", "<=", THROUGH)])
assert EV.season.max() <= THROUGH
EV = EV[EV.play_type.isin(["pass", "run"]) & EV.epa.notna()]
gmap = PL.drop_duplicates("game_id").set_index("game_id").gidx
EV = EV[EV.game_id.isin(gmap.index)].copy()
EV["gidx"] = EV.game_id.map(gmap).astype(int)
EV = EV.sort_values(["gidx", "play_id"])
EVG = {g: (d.posteam.to_numpy(), d.defteam.to_numpy(), d.epa.to_numpy(dtype=float)) for g, d in EV.groupby("gidx")}
SH_R = float(EV[EV.season.between(1999, 2005)].epa.astype(float).var())
SH = GaussTS(SH_R, {"P": 0.0015, "D": 0.004}, {"P": 0.0, "D": 0.005}, {"P": 0.9, "D": 0.5},
             {"P": 0.1, "D": 0.5})

CN = list(COMPS)
rows = []
for g in sorted(AGG_BY_G):
    ag = AGG_BY_G[g]
    season = int(ag.season.iat[0])
    pids = ag.player_id.to_numpy(); teams = ag.posteam.to_numpy()
    recs = ag.to_dict("records")
    for rec in recs:
        pid = rec["player_id"]
        rec["team"] = rec.pop("posteam")
        for comp in CN:
            f = filters[comp]
            if pid in f.st:
                s = f.touch(pid, "P", g, season)
                rec["mu_" + comp] = s[0]; rec["sd_" + comp] = float(np.sqrt(s[1])); rec["n_" + comp] = s[4]
            else:
                rec["mu_" + comp] = 0.0; rec["sd_" + comp] = float(np.sqrt(f.V["P"])); rec["n_" + comp] = 0.0
        u = usage[pid]
        for k in ("tsh", "csh"):
            rec["pre_" + k] = (u[k] + US_PRIOR_N * US_PRIOR[k]) / (u[k + "_w"] + US_PRIOR_N)
        rec["pre_games"] = u["g"]
        k_ = "S:" + pid
        if k_ in SH.st:
            s = SH.touch(k_, "P", g, season); rec["mu_shared"] = s[0]; rec["n_shared"] = s[4]
        else:
            rec["mu_shared"] = 0.0; rec["n_shared"] = 0.0
        rows.append(rec)
    # walk this game's events through every component (after the snapshot)
    ix = IDX_BY_G[g]
    for comp in CN:
        f = filters[comp]; y = Y[comp]; kind = COMPS[comp][0]; roles = COMPS[comp][2]; R = f.R
        for i in ix:
            if KIND[i] != kind or y[i] != y[i]:
                continue
            f.update([f.touch(KEYS[r][i], r, g, season) for r in roles], y[i], R)
    # usage update (per appearance)
    for rec in recs:
        u = usage[rec["player_id"]]
        if u["season"] is not None and u["season"] != season:
            for k in ("tsh", "csh"):
                u[k] *= 0.5; u[k + "_w"] *= 0.5
        u["season"] = season
        for k, num, den in (("tsh", rec["y_tgt"], rec["team_tgt"]), ("csh", rec["y_car"], rec["team_car"])):
            u[k] = US_DECAY * u[k] + (num / den if den > 0 else 0.0)
            u[k + "_w"] = US_DECAY * u[k + "_w"] + 1.0
        u["g"] += 1
    # shared-credit proxy: each offensive play credited to every skill player who appeared for that offense
    evg = EVG.get(g)
    if evg is not None:
        byteam = defaultdict(list)
        for p_, t_ in zip(pids, teams):
            byteam[t_].append("S:" + p_)
        for pt, dt, e in zip(*evg):
            ps_ = byteam.get(pt)
            if ps_:
                SH.update([SH.touch(p, "P", g, season) for p in ps_] + [SH.touch("SD:" + dt, "D", g, season)], e, SH.R)

PG = pd.DataFrame(rows)
PG.to_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_player_games.parquet"), index=False)
# feature-safe copy: PRE-GAME walk-forward values only (no same-game outcome column), plus the two
# headline composites: v_e = EPA per target above average (opportunity quality xEPA + execution EPAOE),
# v_t = the same in yards per target
VAL = PG[[c for c in PG.columns if not c.startswith("y_") and c not in ("team_tgt", "team_car")]].copy()
VAL["v_e"] = VAL.mu_xepat + VAL.mu_epat
VAL["v_t"] = VAL.mu_xypt + VAL.mu_yptoe
VAL.to_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_values.parquet"), index=False)
VAL.to_csv(os.path.join(DATA, "pv_nfl_receiving_rushing_values.csv"), index=False, float_format="%.5f")
json.dump({"moments": MOM, "params": PRM, "usage": {"decay": US_DECAY, "prior_n": US_PRIOR_N, "prior": US_PRIOR},
           "shared_proxy": {"R": SH_R, "V": SH.V, "tau_f": SH.tau_f, "rho": SH.rho, "widen_f": SH.widen_f},
           "through": THROUGH},
          open(os.path.join(DATA, "pv_nfl_receiving_rushing_params.json"), "w"), indent=1)
log(f"player-games {len(PG):,}  ({time.time()-t0:.0f}s)")
