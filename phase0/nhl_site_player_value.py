"""NHL player value for the site: a static snapshot of the validated player-value
ratings, as of the end of 2025-26 -> data/nhl_site_pv.json.

WHAT THIS IS. The player-value program (phase0/pv_nhl_*.py) rates every skater on
what he individually controls, each piece walk-forward (strictly earlier games),
opponent-adjusted where there is an opponent, and empirical-Bayes shrunk toward
his position:

  cre   individual creation     5v5 individual expected goals per 60 (ixG/60)
  fin   finishing               goals above expected per 60 at 5v5, from the
                                1v1 shooter-vs-goalie rating fin_mu:
                                ixG/60 x (exp(fin_mu) - 1)
  a1    primary assists         5v5 primary assists per 60
  a2    secondary assists       5v5 secondary assists per 60
  pp    power play              own PP expected goals x finishing, and PP assists
  fo    faceoffs                1v1 faceoff rating (P(win) vs an average taker)
                                valued in goals by zone and strength
  pen   penalties               drawn minus taken, valued in goals
  def   defence                 on-ice xG-against impact (stint RAPM, teammate-
                                and opponent-adjusted), xG/60 prevented at 5v5

and composes them into one value in goals per 60 with the compose weights the
DEV walk-forward froze for every later season (data/pv_nhl_fix_full_compose_
weights.json). Validated at PLAYER level on DEV (year-over-year repeatability and
prediction of future goals); at GAME level it did not clear the bar on the locked
TEST (data/nhl_test_ledger.csv row 11, n.s.) and a forward test on 2026-27 is
pre-registered (data/pv_nhl_forward_prereg_2026_27.json). So on the site it is a
DISPLAY rating: the game model does not use it.

DISPLAY WEIGHT OF DEFENCE (the one display-only change). The frozen compose weights
were fit for GAME prediction; there the on-ice defence weight (ev_def, 1.9008)
absorbs team signal at lineup level. The component's own DEV goal calibration is
about 1.18 (0.85 to 1.83), and at 1.90 defence carried 58% of defencemen's value
variance and correlated negatively with forwards' points/60. So the DISPLAY
composite counts defence at DISPLAY_W_DEF = 1.0 - the structural prior, one goal
per expected goal prevented:

    c_def_display = p_ev_def * (m_ev / m_all) * (DISPLAY_W_DEF / w_fit['ev_def'])
                  = d_xga * m_ev / m_all

applied BEFORE the group centring and the percentiles. Every other component keeps
its frozen weight, and snapshot() still recomputes each state with the FROZEN
weights and asserts it equals the pipeline's v_all60 / v_game first, so the link
to the validated pipeline is kept; only the display composite differs. Same-season
descriptive agreement with even-strength on-ice goal differential per 60 peaks at
w = 1.0 for forwards and defencemen (meta.display_note). Nothing here is a model
metric and no TEST outcome is scored.

  o     offence = cre + fin + a1 + a2 + pp (centred display contributions, goals/60),
        qo its percentile within F / D; defence is c.def (q.def); identity
        v = o + c.def + c.fo + c.pen.
  prov  1 when he had fewer than PROV_GP NHL games on file before the state: the
        value is shown but tagged provisional (young players' values track their
        production far less well than veterans').

SOURCE. The FIXED full run (amendment 1): data/pv_nhl_fix_full_compose_player_
values.parquet and _skater_games.parquet (creation and finishing inputs are the
unaffected data/pv_nhl_full_* files), plus the faceoff / penalty detail of
data/pv_nhl_fix_full_discrete_duels.csv. After the 2026-09 shift-chart repair the
same pinned pipeline was re-run on the repaired shift data into NEW files
(data/pv_nhl_shiftfix_full_*, see data/nhl_shift_repair_2026_09.json); when those
exist they are the source (meta.source_run says which). Every row is a PRE-game
walk-forward state. The snapshot takes each skater's most recent row = his rating
going into his last game on file (for most, his last 2025-26 game, playoffs
included). The last game itself is not folded in - that would need the component
engines run one step past the data, which only the pre-registered pipeline may do.

  * Defence: the on-ice RAPM value exists for regular-season rows only (playoff
    rows carry NaN, which the compose reads as 0 = average). The snapshot carries
    the player's latest regular-season defensive value forward instead of
    zeroing it - it is the latest fit strictly before his last game, so it stays
    walk-forward. Every other component is the last row's own.
  * Values are recomputed from the row's components with the frozen weights via
    the unmodified pv_nhl_compose_core.player_values; for rows whose defensive
    value was not carried, the recomputation must equal the pipeline's own
    v_all60 / v_game (asserted).

DISPLAY CENTRING. The compose centres each rate on a decayed all-player mean of
his position group, and the RAPM defence has its own zero, so the average 2025-26
regular does not sit exactly at 0 (minutes-weighted: forwards about -0.015 goals/60,
defencemen about -0.009; about 0.01 or less per component - meta.reference.centring
holds the exact shifts). For display each component
contribution is re-centred on the minutes-weighted mean of the REFERENCE group -
skaters with >= REF_MIN_GP regular-season games in 2025-26, forwards and
defencemen separately - so 0 = an average regular at his position and the
components still add up to the total. Percentiles are within F / D against the
same reference (faceoffs: against regular faceoff takers only). Rank order is
unchanged by the centring.

UNITS (per player block, merged into site/data/nhl.json players[pid].pv by
phase0/nhl_site_players.py):
  v    goals per 60 of his all-situations ice time, vs an average regular
  g    goals per game at his usual ice time (v x toi / 60)
  p    percentile of v within his position group (1..99)
  o    offence part of v (goals / 60), qo its percentile
  c    per-component contribution to v (goals / 60), sums to v
  q    per-component percentile within his group (fo: None unless he takes draws)
  prov 1 = provisional (fewer than PROV_GP NHL games on file)
  plus the components in their own units (listed in the JSON's meta.units).

TEAMS. teams[code].lu_prev = the summed display value per game (g) of the team's
ACTUAL 2025-26 lineup - the 12 forwards and 6 defencemen with the most 2025-26
regular-season time on ice for that team (toi_s in the skater-games table), each
valued at his snapshot g. nhl_site_players.team_blocks turns it into the
"offseason roster change" of the current lineup (display only).

Market-blind: no odds are read. Nothing here is scored; no model metric is
computed. Local-only builder (pandas/pyarrow); the serve only reads the JSON.

    python phase0/nhl_site_player_value.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def D(p):
    return os.path.join(ROOT, "data", p)


SRC_PV = "pv_nhl_fix_full_compose_player_values.parquet"
SRC_SK = "pv_nhl_fix_full_compose_skater_games.parquet"
SRC_DD = "pv_nhl_fix_full_discrete_duels.csv"
SRC_W = "pv_nhl_fix_full_compose_weights.json"      # pinned by the forward test: read only
OUT = "nhl_site_pv.json"
NAMES = "nhl_player_names.json"
NOSHIFT = "pv_nhl_full_noshift_games.csv"          # games with no shift chart on file
REPAIR_LOG = "nhl_shift_repair_2026_09.json"       # the 2026-09 shift-chart repair
REL_SRC = "pv_nhl_defense_onice_eval_rel.json"     # DEV reliability of the defence fit
REL_KEY = "d_xg_120000_0"                          # single-season xGA fit (lambda 120000)
# Source runs of the pinned pipeline. "fix_full" = the amendment-1 full run (built
# with the 2024-25 / 2025-26 shift-chart gap); "shiftfix" = the same pinned code
# re-run on the repaired shift data, copied to NEW files (the originals are records
# of earlier results and are never overwritten). The newest complete run is used.
RUNS = {
    "shiftfix": {"pv": "pv_nhl_shiftfix_full_compose_player_values.parquet",
                 "sk": "pv_nhl_shiftfix_full_compose_skater_games.parquet",
                 "dd": "pv_nhl_shiftfix_full_discrete_duels.csv"},
    "fix_full": {"pv": SRC_PV, "sk": SRC_SK, "dd": SRC_DD},
}

REF_SEASON = 20252026          # the snapshot season ("as of the end of 2025-26")
REF_MIN_GP = 20                # regular-season games in REF_SEASON to be a reference regular
FO_TAKER = 8.0                 # expected faceoffs per 60 to count as a faceoff taker
MIN_LAST_SEASON = 20242025     # older last games are kept only for names-map players
FIRST_FROZEN = 20182019        # the frozen compose weights apply from here on
DISPLAY_W_DEF = 1.0            # display weight of on-ice defence: 1 goal per xG prevented
PROV_GP = 82                   # fewer NHL games on file than this: value is provisional
LU_F, LU_D = 12, 6             # a lineup: 12 forwards + 6 defencemen
OFFENCE = ("cre", "fin", "a1", "a2", "pp")

# display component -> the compose components it is made of
COMPS = ["cre", "fin", "a1", "a2", "pp", "fo", "pen", "def"]
LABELS = {
    "cre": "Creation: individual 5v5 expected goals per 60 (ixG/60)",
    "fin": "Finishing: goals above expected per 60 at 5v5 (1v1 shooter-vs-goalie rating)",
    "a1": "Primary assists per 60 at 5v5",
    "a2": "Secondary assists per 60 at 5v5",
    "pp": "Power play: own PP expected goals x finishing, plus PP assists",
    "fo": "Faceoffs: 1v1 rating (win probability vs an average taker), valued in goals",
    "pen": "Penalties drawn minus taken, valued in goals",
    "def": "Defence: on-ice xG-against impact (RAPM), xG/60 prevented at 5v5",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r(x, nd=3):
    """JSON-safe rounding: None for missing / non-finite."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    v = round(x, nd)
    return 0.0 if v == 0 else v


# ------------------------------------------------------------ pure helpers --
def percentile(x, ref):
    """Percentile (1..99) of x within the sorted reference values `ref`: the share
    below plus half the ties. None when x is missing or the reference is empty."""
    import bisect
    if x is None or not ref or not math.isfinite(float(x)):
        return None
    lo = bisect.bisect_left(ref, x)
    hi = bisect.bisect_right(ref, x)
    p = 100.0 * (lo + 0.5 * (hi - lo)) / len(ref)
    return int(min(99, max(1, round(p))))


def contributions(p, m_ev, m_pp, m_all, w_duels, fo_g60):
    """Per-component contributions in goals per 60 of ALL-situations ice time.

    p: weighted component rates p_<comp> = w_k * x_k (per 60 of the strength each
    lives in: ev at 5v5, pp on the power play, duels all situations). An ev part
    counts for m_ev / m_all of his minutes, a pp part for m_pp / m_all. The duels
    component (faceoffs + penalties, centred on the group) is split into faceoffs
    (w_duels x fo_g60) and the rest (penalties and the group centring). The
    contributions sum to v_all60 exactly."""
    ev, pp = m_ev / m_all, m_pp / m_all
    fo = w_duels * fo_g60
    return {
        "cre": p["p_ev_shot"] * ev,
        "fin": p["p_ev_fin"] * ev,
        "a1": p["p_ev_a1"] * ev,
        "a2": p["p_ev_a2"] * ev,
        "pp": (p["p_pp_xg"] + p["p_pp_ast"]) * pp,
        "fo": fo,
        "pen": p["p_duels"] - fo,
        "def": p["p_ev_def"] * ev,
    }


def display_def(c_def_fit, w_fit_def, w_display=DISPLAY_W_DEF):
    """The defence contribution at the DISPLAY weight: the pipeline's weighted rate
    p_ev_def = w_fit x d_xga re-weighted to w_display x d_xga (same minutes share)."""
    return c_def_fit * (w_display / w_fit_def)


def shift_gap(path=None, log_path=None):
    """Shift-chart coverage of the data the values were built from, counted from the
    pipeline's own missing-shift list (data/pv_nhl_full_noshift_games.csv): games per
    season with no shift chart, and a sentence for the page. None when the list is
    absent (the counts are then unknown, and nothing is claimed)."""
    import csv
    path = path or D(NOSHIFT)
    try:
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    n, po = {}, {}
    for rw in rows:
        s = str(rw.get("season"))
        n[s] = n.get(s, 0) + 1
        po[s] = po.get(s, 0) + (str(rw.get("gid", ""))[4:6] == "03")
    by = {"20252026": n.get("20252026", 0), "20242025": n.get("20242025", 0)}
    total = len(rows)
    if total == 0:
        rep = {}
        try:
            with open(log_path or D(REPAIR_LOG), encoding="utf-8") as fh:
                rep = json.load(fh)
        except (OSError, json.JSONDecodeError):
            rep = {}
        when = rep.get("date") or rep.get("written")
        src = rep.get("source_summary") or ("the NHL shift-chart API, repopulated upstream, "
                                            "plus the official HTML time-on-ice reports")
        text = ("Shift charts are complete for every 2024-25 and 2025-26 game on file"
                + (f" (repaired {when}: {src})" if when else f" ({src})")
                + "; the values use every game.")
        return {"complete": True, "by_season": by, "games": 0, "repaired": when,
                "text": text}
    def part(s, lbl):
        k, p = n.get(s, 0), po.get(s, 0)
        if p and k - p:
            return f"{k:,} games of {lbl} ({k - p:,} regular-season, {p:,} playoff)"
        return f"{k:,} {'playoff' if p else 'regular-season'} games of {lbl}"
    parts = [part(s, lbl) for s, lbl in (("20252026", "2025-26"), ("20242025", "2024-25"))
             if n.get(s)]
    other = total - by["20252026"] - by["20242025"]
    if other:
        parts.append(f"{other:,} other games")
    text = ("Shift charts are missing for " + " and ".join(parts)
            + ", an NHL API upstream gap (data/pv_nhl_full_noshift_games.csv); the "
            "minutes-based pieces skip those games.")
    return {"complete": False, "by_season": by, "games": total, "repaired": None,
            "text": text}


def def_reliability(path=None):
    """DEV reliability of the single-season on-ice defence fit (year over year), from
    data/pv_nhl_defense_onice_eval_rel.json; None when the file is absent."""
    try:
        with open(path or D(REL_SRC), encoding="utf-8") as fh:
            d = json.load(fh)
        y = d["yoy"][REL_KEY]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return {"r": y.get("r"), "r_F": y.get("r_F"), "r_D": y.get("r_D"),
            "kind": "year-over-year correlation of single-season fits, DEV seasons 2010-18",
            "source": f"data/{REL_SRC} yoy.{REL_KEY}",
            "window": ("the displayed defence is a multi-season on-ice fit: 4 seasons (his "
                       "current one and the 3 before, refit monthly), 730-day half-life, "
                       "adjusted for teammates and opponents")}


def lineup_prev(df, g_by_pid, season=REF_SEASON):
    """code -> the team's actual `season` lineup valued at the snapshot: the LU_F
    forwards and LU_D defencemen with the most regular-season toi_s for that team,
    each at his snapshot g (goals per game; missing = 0). Needs df.toi_s / team."""
    if "toi_s" not in df.columns or "team" not in df.columns:
        return {}
    reg = df[(df.season == season) & (df.gtype == 2)]
    toi = (reg.assign(toi_s=reg.toi_s.fillna(0.0))
           .groupby(["team", "grp", "pid"]).toi_s.sum().reset_index())
    out = {}
    for team, t in toi.groupby("team"):
        ids = {}
        for grp, k in (("F", LU_F), ("D", LU_D)):
            s = t[t.grp == grp].sort_values(["toi_s", "pid"], ascending=[False, True])
            ids[grp] = [str(int(p)) for p in s.pid.head(k)]
        f = sum(g_by_pid.get(p, 0.0) for p in ids["F"])
        d = sum(g_by_pid.get(p, 0.0) for p in ids["D"])
        out[str(team)] = {"lu_prev": r(f + d), "f": r(f), "d": r(d), "season": int(season),
                          "n": len(ids["F"]) + len(ids["D"]), "ids": ids["F"] + ids["D"]}
    return out


def pick_run(runs=None, exists=os.path.exists):
    """The newest complete source run: 'shiftfix' when all its files exist, else
    'fix_full'."""
    runs = runs or RUNS
    for name in ("shiftfix", "fix_full"):
        if all(exists(D(f)) for f in runs[name].values()):
            return name
    return "fix_full"


def group_means(rows, key_w="m_all"):
    """Minutes-weighted mean of every contribution per position group over the
    reference rows: {grp: {comp: mean}}."""
    out = {}
    for g in ("F", "D"):
        rs = [x for x in rows if x["grp"] == g]
        wt = sum(x[key_w] for x in rs)
        out[g] = {k: (sum(x["c"][k] * x[key_w] for x in rs) / wt if wt > 0 else 0.0)
                  for k in COMPS}
    return out


# ------------------------------------------------------------------- build --
def load_frames(run=None):
    import numpy as np
    import pandas as pd
    src = RUNS[run or pick_run()]
    pv = pd.read_parquet(D(src["pv"]))
    sk_cols = ["gid", "pid", "pos", "n_prior_games", "exp_min_ev", "exp_min_pp",
               "ixg_ev_hat", "a1_ev_hat", "a2_ev_hat", "ixg_pp_hat", "a1_pp_hat", "a2_pp_hat",
               "fin_mu", "fin_sd", "fo_g60", "pen_g60", "dd_g60", "d_xga", "toi_s"]
    sk = pd.read_parquet(D(src["sk"]), columns=sk_cols)
    assert len(pv) == len(sk)
    assert (pv.gid.to_numpy() == sk.gid.to_numpy()).all()
    assert (pv.pid.to_numpy() == sk.pid.to_numpy()).all()
    df = pd.concat([pv, sk.drop(columns=["gid", "pid"])], axis=1)
    dd = pd.read_csv(D(src["dd"]), usecols=["gid", "pid", "fo_pavg", "fo_n60", "fo_g60",
                                             "pen_g60", "pen_draw_n60", "pen_take_n60"])
    df = df.merge(dd.rename(columns={"fo_g60": "fo_g60_dd", "pen_g60": "pen_g60_dd"}),
                  on=["gid", "pid"], how="left", validate="one_to_one")
    assert df.fo_pavg.notna().all(), "a panel row has no discrete-duels row"
    assert np.array_equal(df.fo_g60.to_numpy(), df.fo_g60_dd.to_numpy())
    assert np.array_equal(df.pen_g60.to_numpy(), df.pen_g60_dd.to_numpy())
    return df.drop(columns=["fo_g60_dd", "pen_g60_dd"])


def snapshot(df, frozen, keep=None):
    """Each skater's most recent pre-game state (TEST era: the frozen weights apply),
    with the defensive value carried forward over playoff rows and his values
    recomputed with the frozen weights. keep(last) -> boolean mask of the rows to
    keep (applied before the recomputation)."""
    import numpy as np
    import pandas as pd
    import pv_nhl_compose_core as C          # frozen module: imported, never edited

    df = df.sort_values(["pid", "date", "gid"], kind="stable").reset_index(drop=True)
    # latest regular-season defensive value, carried forward (walk-forward: it
    # is a fit strictly before an earlier game, so strictly before the last one)
    d_ff = df.groupby("pid").d_xga.ffill()
    ever = df.d_xga.notna().groupby(df.pid).transform("max").astype(bool)
    reg_n = (df.assign(one=(df.gtype == 2).astype(int))
             .groupby(["pid", "season"]).one.sum())
    last = df.groupby("pid").tail(1).copy()
    last = last[last.season >= FIRST_FROZEN]
    if keep is not None:
        last = last[keep(last)]
    last["d_ff"] = d_ff.loc[last.index].to_numpy()
    last["def_known"] = ever.loc[last.index].to_numpy()
    last["carried"] = last.d_xga.isna().to_numpy() & last.d_ff.notna().to_numpy()
    last["sgp"] = [int(reg_n.get((p, s), 0)) for p, s in zip(last.pid, last.season)]

    x = pd.DataFrame({c: last["x_" + c].to_numpy(float) for c in C.SK_COMPS},
                     index=last.index)
    x["ev_def"] = last.d_ff.fillna(0.0).to_numpy(float)
    m = pd.DataFrame({"ev": last.m_ev.to_numpy(float), "pp": last.m_pp.to_numpy(float),
                      "all": last.m_all.to_numpy(float)}, index=last.index)
    W = {s: {"w": dict(frozen)} for s in sorted(set(last.season))}
    v = C.player_values(last[["season"]], x, m, W)
    # the recomputation IS the pipeline's value wherever nothing was carried
    same = ~last.carried.to_numpy()
    for col in ("v_all60", "v_game"):
        assert np.allclose(v[col].to_numpy()[same], last[col].to_numpy()[same],
                           rtol=0, atol=1e-9), col
    for col in v.columns:
        last["r_" + col] = v[col].to_numpy()
    return last


def build(df=None, weights=None, names=None, built=None, run=None, w_def=DISPLAY_W_DEF,
          noshift_path=None, rel_path=None, repair_log=None):
    """df / weights / names: injected in tests; read from data/ otherwise. run: the
    source run (RUNS; default pick_run()). w_def: the display weight of defence
    (DISPLAY_W_DEF; the frozen fit weight reproduces the fit composite)."""
    from_files = df is None
    wjson = json.load(open(D(SRC_W), encoding="utf-8")) if weights is None else weights
    frozen = wjson["frozen_test_weights"]["w"]
    assert REF_SEASON in wjson["frozen_test_weights"]["applied_to"]
    run = (run or pick_run()) if from_files else run
    df = load_frames(run) if df is None else df
    if names is None:
        try:
            names = json.load(open(D(NAMES), encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            names = {}
    known = set(str(k) for k in names)
    # FIRST: every state recomputed with the FROZEN fit weights must equal the
    # pipeline's own value (asserted inside snapshot) - the link to the validated run
    last = snapshot(df, frozen, keep=lambda L: ((L.season >= MIN_LAST_SEASON)
                                                | L.pid.astype(str).isin(known)))
    w_duels = frozen["duels"]

    rows = []
    for t in last.itertuples(index=False):
        pr = {k: getattr(t, "r_" + k) for k in ("p_ev_shot", "p_ev_fin", "p_ev_a1", "p_ev_a2",
                                                 "p_ev_def", "p_pp_xg", "p_pp_ast", "p_duels")}
        c = contributions(pr, t.m_ev, t.m_pp, t.m_all, w_duels, t.fo_g60)
        assert abs(sum(c.values()) - t.r_v_all60) < 1e-9
        # display only: defence at the display weight, before centring and percentiles
        c["def"] = display_def(c["def"], frozen["ev_def"], w_def)
        e = math.exp(t.fin_mu) if t.fin_mu == t.fin_mu else 1.0
        nat = {"cre": t.ixg_ev_hat, "fin": t.ixg_ev_hat * (e - 1.0), "a1": t.a1_ev_hat,
               "a2": t.a2_ev_hat, "pp": c["pp"], "fo": t.fo_pavg, "pen": t.pen_g60,
               "def": t.d_ff if t.d_ff == t.d_ff else 0.0}
        rows.append({"pid": str(int(t.pid)), "grp": t.grp, "pos": t.pos, "season": int(t.season),
                     "date": str(t.date), "gtype": int(t.gtype), "gp": int(t.n_prior_games),
                     "sgp": t.sgp, "m_all": float(t.m_all), "m_ev": float(t.m_ev),
                     "m_pp": float(t.m_pp), "c": c, "nat": nat, "fm": e,
                     "fo_n60": float(t.fo_n60),
                     "pd": float(t.pen_draw_n60), "pt": float(t.pen_take_n60),
                     "def_known": bool(t.def_known), "carried": bool(t.carried)})

    ref = [x for x in rows if x["season"] == REF_SEASON and x["sgp"] >= REF_MIN_GP]
    mu = group_means(ref)
    for x in rows:
        x["cd"] = {k: x["c"][k] - mu[x["grp"]][k] for k in COMPS}
        x["v"] = sum(x["cd"].values())
        x["o"] = sum(x["cd"][k] for k in OFFENCE)
        x["g"] = x["v"] * x["m_all"] / 60.0
    # reference distributions (sorted), per group
    dist = {}
    for g in ("F", "D"):
        rg = [x for x in ref if x["grp"] == g]
        dist[(g, "v")] = sorted(x["v"] for x in rg)
        dist[(g, "o")] = sorted(x["o"] for x in rg)
        for k in COMPS:
            if k == "fo":
                dist[(g, k)] = sorted(x["nat"]["fo"] for x in rg if x["fo_n60"] >= FO_TAKER)
            else:
                dist[(g, k)] = sorted(x["nat"][k] for x in rg)

    players = {}
    for x in rows:
        g = x["grp"]
        taker = x["fo_n60"] >= FO_TAKER
        q = {k: (percentile(x["nat"][k], dist[(g, k)]) if (k != "fo" or taker) else None)
             for k in COMPS}
        if not x["def_known"]:
            q["def"] = None
        players[x["pid"]] = {
            "grp": g, "s": x["season"], "d": x["date"], "po": 1 if x["gtype"] == 3 else 0,
            "gp": x["gp"], "sgp": x["sgp"], "prov": 1 if x["gp"] < PROV_GP else 0,
            "v": r(x["v"]), "g": r(x["g"]), "p": percentile(x["v"], dist[(g, "v")]),
            "o": r(x["o"]), "qo": percentile(x["o"], dist[(g, "o")]),
            "toi": r(x["m_all"], 1), "t5": r(x["m_ev"], 1), "tpp": r(x["m_pp"], 1),
            "c": {k: r(x["cd"][k]) for k in COMPS},
            "q": q,
            # natural units
            "cre": r(x["nat"]["cre"]), "fin": r(x["nat"]["fin"]), "fm": r(x["fm"]),
            "a1": r(x["nat"]["a1"]), "a2": r(x["nat"]["a2"]),
            "fo": [r(x["nat"]["fo"]), r(x["fo_n60"], 1)] if taker else None,
            "pen": [r(x["nat"]["pen"]), r(x["pd"], 2), r(x["pt"], 2)],
            "def": r(x["nat"]["def"]) if x["def_known"] else None,
        }
    n_ref = {g: len(dist[(g, "v")]) for g in ("F", "D")}
    teams = lineup_prev(df, {x["pid"]: x["g"] for x in rows})
    gap = shift_gap(noshift_path, repair_log)
    srcs = {}
    if from_files:
        files = list(RUNS[run].values()) + [SRC_W]
        srcs = {p: (sha256(D(p)) if os.path.exists(D(p)) else None) for p in files}
    meta = {
        "built": built or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "asof": "end of 2025-26",
        "season": REF_SEASON,
        "through": max(x["date"] for x in rows) if rows else None,
        "model_input": False,
        "label": ("Player value (finishing, creation, faceoffs, penalties, defence) - a "
                  "display rating; the game model does not use it yet (forward test on "
                  "2026-27 pending)"),
        "game_level": ("locked TEST: log loss improved by 0.00062 (0.66418 -> 0.66356), n.s., "
                       "95% CI -0.00066 to +0.00188 (data/nhl_test_ledger.csv row 11); "
                       "forward test pre-registered: data/pv_nhl_forward_prereg_2026_27.json"),
        "state": ("each skater's most recent walk-forward state = his rating going into his "
                  "last game on file (pre-game; that game is not folded in)"),
        "units": {"v": "goals per 60 of all-situations ice time vs an average regular at his "
                       "position", "g": "goals per game at his usual ice time",
                  "p": "percentile within F / D (reference regulars)",
                  "o": "offence: cre + fin + a1 + a2 + pp contributions, goals per 60",
                  "qo": "percentile of o within F / D (reference regulars)",
                  "prov": f"1 = provisional: fewer than {PROV_GP} NHL games on file (gp)",
                  "c": "component contributions to v, goals per 60; they sum to v "
                       "(def at the display weight, meta.display_weights)",
                  "q": "component percentiles within F / D",
                  "cre": "5v5 ixG / 60", "fin": "5v5 goals above expected / 60",
                  "fm": "finishing multiplier exp(fin_mu): goals per expected goal vs average",
                  "a1": "5v5 primary assists / 60", "a2": "5v5 secondary assists / 60",
                  "fo": "[win probability vs an average taker, faceoffs / 60] (takers only)",
                  "pen": "[goals / 60 drawn minus taken, "
                  "penalties drawn / 60, taken / 60]",
                  "def": "5v5 on-ice xG / 60 prevented (RAPM; null = never had a fit)",
                  "toi": "expected minutes per game, all situations", "t5": "5v5",
                  "tpp": "power play", "gp": "NHL games on file before this state (2010-11 on)",
                  "sgp": "regular-season games in his last season", "d": "date of that last game",
                  "s": "its season", "po": "1 = it was a playoff game"},
        "components": {k: LABELS[k] for k in COMPS},
        "reference": {"season": REF_SEASON, "min_gp": REF_MIN_GP, "n": n_ref,
                      "fo_takers": {g: len(dist[(g, "fo")]) for g in ("F", "D")},
                      "fo_taker_min_per60": FO_TAKER,
                      "centring": {g: {k: r(mu[g][k], 5) for k in COMPS} for g in ("F", "D")}},
        # the frozen FIT weights the pipeline applied (and snapshot() re-checks) ...
        "weights": {k: r(v, 4) for k, v in frozen.items()},
        # ... and the one display-only change
        "display_weights": {"ev_def": w_def},
        "display_note": (f"The display counts on-ice defence at {w_def:g} goal per expected goal "
                         f"prevented (the structural prior); the game-prediction fit uses "
                         f"{frozen['ev_def']:.2f}, which at lineup level absorbs team signal. "
                         "Every other component keeps its fit weight, and every value is first "
                         "recomputed with the fit weights and checked against the validated "
                         "pipeline. Same-season on-ice goal differential per 60 agrees best at "
                         "1.0 for forwards and defencemen (descriptive, not a model test)."),
        "def_reliability": def_reliability(rel_path),
        "prov_gp": PROV_GP,
        "defence_carried": sum(1 for x in rows if x["carried"]),
        "source_run": run,
        "sources": srcs,
        "shift_gap": gap["text"] if gap else None,
        "shift_gap_counts": ({"by_season": gap["by_season"], "games": gap["games"],
                              "complete": gap["complete"], "repaired": gap["repaired"]}
                             if gap else None),
    }
    return {"meta": meta, "players": players, "teams": teams}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="NHL player-value site snapshot")
    ap.add_argument("--run", choices=sorted(RUNS), default=None,
                    help="source run of the pinned pipeline (default: newest complete)")
    a = ap.parse_args(argv)
    out = build(run=a.run)
    with open(D(OUT) + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=False)
    os.replace(D(OUT) + ".tmp", D(OUT))
    m = out["meta"]
    print(f"wrote data/{OUT}: {len(out['players'])} skaters (run {m['source_run']}), "
          f"reference {m['reference']['n']}, defence carried over playoffs for "
          f"{m['defence_carried']}, {len(out['teams'])} team lineups; shift gap: {m['shift_gap']}")


if __name__ == "__main__":
    main()
