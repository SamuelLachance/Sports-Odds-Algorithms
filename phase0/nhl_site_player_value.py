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

SOURCE. The FIXED full run (amendment 1): data/pv_nhl_fix_full_compose_player_
values.parquet and _skater_games.parquet (creation and finishing inputs are the
unaffected data/pv_nhl_full_* files), plus the faceoff / penalty detail of
data/pv_nhl_fix_full_discrete_duels.csv. Every row is a PRE-game walk-forward
state. The snapshot takes each skater's most recent row = his rating going into
his last game on file (for most, his last 2025-26 game, playoffs included). The
last game itself is not folded in - that would need the component engines run one
step past the data, which only the pre-registered pipeline may do.

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
  c    per-component contribution to v (goals / 60), sums to v
  q    per-component percentile within his group (fo: None unless he takes draws)
  plus the components in their own units (listed in the JSON's meta.units).

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
SRC_W = "pv_nhl_fix_full_compose_weights.json"
OUT = "nhl_site_pv.json"
NAMES = "nhl_player_names.json"

REF_SEASON = 20252026          # the snapshot season ("as of the end of 2025-26")
REF_MIN_GP = 20                # regular-season games in REF_SEASON to be a reference regular
FO_TAKER = 8.0                 # expected faceoffs per 60 to count as a faceoff taker
MIN_LAST_SEASON = 20242025     # older last games are kept only for names-map players
FIRST_FROZEN = 20182019        # the frozen compose weights apply from here on

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
def load_frames():
    import numpy as np
    import pandas as pd
    pv = pd.read_parquet(D(SRC_PV))
    sk_cols = ["gid", "pid", "pos", "n_prior_games", "exp_min_ev", "exp_min_pp",
               "ixg_ev_hat", "a1_ev_hat", "a2_ev_hat", "ixg_pp_hat", "a1_pp_hat", "a2_pp_hat",
               "fin_mu", "fin_sd", "fo_g60", "pen_g60", "dd_g60", "d_xga"]
    sk = pd.read_parquet(D(SRC_SK), columns=sk_cols)
    assert len(pv) == len(sk)
    assert (pv.gid.to_numpy() == sk.gid.to_numpy()).all()
    assert (pv.pid.to_numpy() == sk.pid.to_numpy()).all()
    df = pd.concat([pv, sk.drop(columns=["gid", "pid"])], axis=1)
    dd = pd.read_csv(D(SRC_DD), usecols=["gid", "pid", "fo_pavg", "fo_n60", "fo_g60",
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


def build(df=None, weights=None, names=None, built=None):
    """df / weights / names: injected in tests; read from data/ otherwise."""
    from_files = df is None
    wjson = json.load(open(D(SRC_W), encoding="utf-8")) if weights is None else weights
    frozen = wjson["frozen_test_weights"]["w"]
    assert REF_SEASON in wjson["frozen_test_weights"]["applied_to"]
    df = load_frames() if df is None else df
    if names is None:
        try:
            names = json.load(open(D(NAMES), encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            names = {}
    known = set(str(k) for k in names)
    last = snapshot(df, frozen, keep=lambda L: ((L.season >= MIN_LAST_SEASON)
                                                | L.pid.astype(str).isin(known)))
    w_duels = frozen["duels"]

    rows = []
    for t in last.itertuples(index=False):
        pr = {k: getattr(t, "r_" + k) for k in ("p_ev_shot", "p_ev_fin", "p_ev_a1", "p_ev_a2",
                                                 "p_ev_def", "p_pp_xg", "p_pp_ast", "p_duels")}
        c = contributions(pr, t.m_ev, t.m_pp, t.m_all, w_duels, t.fo_g60)
        assert abs(sum(c.values()) - t.r_v_all60) < 1e-9
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
        x["g"] = x["v"] * x["m_all"] / 60.0
    # reference distributions (sorted), per group
    dist = {}
    for g in ("F", "D"):
        rg = [x for x in ref if x["grp"] == g]
        dist[(g, "v")] = sorted(x["v"] for x in rg)
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
            "gp": x["gp"], "sgp": x["sgp"],
            "v": r(x["v"]), "g": r(x["g"]), "p": percentile(x["v"], dist[(g, "v")]),
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
                  "c": "component contributions to v, goals per 60; they sum to v",
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
        "weights": {k: r(v, 4) for k, v in frozen.items()},
        "defence_carried": sum(1 for x in rows if x["carried"]),
        "sources": ({p: (sha256(D(p)) if os.path.exists(D(p)) else None)
                     for p in (SRC_PV, SRC_SK, SRC_DD, SRC_W)} if from_files else {}),
        "shift_gap": ("2025-26 shift charts are missing for 505 of the 1,312 regular-season "
                      "games and 20 playoff games (NHL API upstream gap, "
                      "data/pv_nhl_full_noshift_games.csv); minutes-based accumulators skip "
                      "those games"),
    }
    return {"meta": meta, "players": players}


def main():
    out = build()
    with open(D(OUT) + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=False)
    os.replace(D(OUT) + ".tmp", D(OUT))
    m = out["meta"]
    print(f"wrote data/{OUT}: {len(out['players'])} skaters, reference {m['reference']['n']}, "
          f"defence carried over playoffs for {m['defence_carried']}")


if __name__ == "__main__":
    main()
