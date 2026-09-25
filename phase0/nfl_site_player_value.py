"""NFL site: individual player measures from the player-value program. DISPLAY ONLY.

The owner's complaint: the site rated NFL players only through shared-credit play
outcomes (the 11-vs-11 TrueSkill credits all 22 players with each play's EPA). The
player-value program (phase0/pv_nfl_*.py) rated players the MLB way -- on the events
each player individually controls, one-on-one and opponent-adjusted, walk-forward and
empirical-Bayes shrunk -- and validated the measures at PLAYER level on DEV (<= 2015):

  QB      passing composite: opponent-adjusted EPA + success + sack + CPOE per dropback
          (year-over-year r 0.54 vs 0.49 for raw EPA/dropback)
  WR/TE   target value: EPA per target above average = opportunity quality (the
          expected value of the targets he earns) + execution over expectation, QB-
          and defence-adjusted; plus target share (usage)
  RB      rushing value: EPA per designed carry over the situation's expectation,
          blocking-unit- and defence-adjusted (rusher skill is a small, unstable part
          of rushing results -- said on the page)
  DL/LB   pass rush: sacks + QB hits per 100 opponent dropbacks in games he played,
          vs an average opposing line and home scorer (YoY r 0.72); run stops per 100
          designed runs (tackles on runs that failed)
  DB/LB   coverage: passes defensed + INTs over expectation (ball production) and EPA
          saved per target he is credited on (breakups, INTs, tackle after the catch)
  OL      nothing individual: public play-by-play never records which lineman lost a
          rep, so the page says so and shows the UNIT numbers (sacks / pressure / run
          stops allowed vs average, QB + line together) on the team page instead.

At GAME level none of it improved the forecasts (the composed values did not clear
the DEV screen, data/pv_nfl_screen.json), so the game model does not use them. They
are display ratings and the page labels them that way.

Two parts:

  build (LOCAL; needs the pv_nfl_full_* outputs, numba and lightgbm):
      python phase0/pv_nfl_events_pull.py --refresh 2026  (optional: newest plays)
      python phase0/pv_nfl_full_passing.py                (~10 s)
      python phase0/pv_nfl_full_receiving_rushing.py      (LightGBM stage ~6 min)
      python phase0/pv_nfl_full_pass_rush.py              (~20 s)
      python phase0/pv_nfl_full_coverage.py               (~30 s)
      python phase0/nfl_site_player_value.py              -> data/nfl_site_pv.json
    or all five in one go: python phase0/nfl_site_player_value.py --all
    The snapshot is committed; it is small and needs no model input at serve time.

  merge (CI-SAFE; stdlib only): nfl_site_db.py calls merge(payload), which copies the
    snapshot's blocks onto payload players (players[id].pv, only the blocks that
    describe his position family), ranks each measure into a position percentile
    among the payload's players, and adds the team unit numbers (teams[code].pv) and
    payload.pv_meta. It never touches a forecast, a rating or a schedule row. A
    missing or unreadable snapshot leaves the payload without pv (and says so).

Snapshot player blocks (compact keys; every rate is per the unit named):
  qb    v composite EPA/dropback above league | epa opp-adj EPA/db | cpoe CPOE rating
        (percentage points) | sack sack-rate rating (points of dropbacks, + = more
        sacks) | db / dbp / dbc dropbacks this season / last season / career
  rec   v EPA per target above average | xy opportunity (expected yards per target
        above average) | ye execution (yards per target over expected) | ts target
        share (games played, recent-weighted) | t / tp / tc targets
  rush  v EPA per carry over expected | ry rush yards over expected per carry
        (capped -5..20) | cs carry share | c / cp / cc designed carries
  front pr sacks+hits per 100 opp. dropbacks | sk sacks per 100 | st run stops per 100
        designed runs | sh share of an average unit's pressure | g / gp games credited
  cov   ball passes defensed + INTs per 100 opp. attempts | x ball production vs
        position average (1.0 = average) | ept EPA saved per credited target | ar
        credited targets per 100 opp. attempts | g / gp games credited | grp the
        coverage engine's position group (DB / LB / DL: the scale the block is on)

The coverage engine rates a defender within his group (nfl_players.csv position),
while the payload's family can differ (an edge rusher listed OLB on the roster but
DE in nfl_players.csv). A coverage block is shown only when its group is the
family's (COV_GROUP): a DL-scale block would sit at the DL prior (ball production
~1.0x the DL average) and read as "bottom 10%" among linebackers.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = os.path.join(ROOT, "data", "nfl_site_pv.json")
MODEL_USE = ("display rating; the game model does not use it "
             "(it did not improve game predictions)")

# which blocks describe which position family
FAM_BLOCKS = {"QB": ("qb",), "RB": ("rush", "rec"), "WR": ("rec",), "TE": ("rec",),
              "DL": ("front",), "LB": ("front", "cov"), "DB": ("cov",)}
# ranked measures per block (all: higher = better for the player)
RANKED = {"qb": ("v",), "rec": ("v",), "rush": ("v",), "front": ("pr", "st"), "cov": ("ball", "ept")}
# a measure is ranked against the QUALIFIED players of his family: sample this
# season + last season (dropbacks / targets / carries / games credited)
QUALIFY = {"qb": (("db", "dbp"), 150), "rec": (("t", "tp"), 30), "rush": (("c", "cp"), 50),
           "front": (("g", "gp"), 6), "cov": (("g", "gp"), 6)}
MIN_POOL = 8           # fewer qualified peers than this: no percentile
UNIT_KEYS = ("O_sk", "O_pr", "O_st", "D_sk", "D_pr", "D_st")
# a coverage block is shown only on the scale of the family it is ranked in
COV_GROUP = {"LB": "LB", "DB": "DB"}


# ======================================================================= merge (CI)
def load_snapshot(path: str = SNAPSHOT):
    try:
        with open(path, encoding="utf-8") as fh:
            snap = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as ex:
        print(f"[nfl_site_player_value] no player-value snapshot ({type(ex).__name__}): "
              f"players get no pv block", flush=True)
        return None
    if not isinstance(snap, dict) or not isinstance(snap.get("players"), dict):
        print("[nfl_site_player_value] snapshot has no players map: skipped", flush=True)
        return None
    return snap


def qualified(block: str, b: dict) -> bool:
    keys, floor = QUALIFY[block]
    return sum(float(b.get(k) or 0) for k in keys) >= floor


def percentile(v: float, pool: list) -> int:
    """share of the pool below v (ties count half), as a 1-99 percentile (no "0th" /
    "100th": a pool's best and worst are 99th / 1st)."""
    lo = sum(1 for x in pool if x < v)
    eq = sum(1 for x in pool if x == v)
    return max(1, min(99, int(round(100.0 * (lo + 0.5 * eq) / len(pool)))))


_DEFAULT = object()


def merge(payload: dict, snap=_DEFAULT) -> int:
    """Attach display blocks to payload players/teams. Returns players given a pv.
    snap: the snapshot dict; omitted = read data/nfl_site_pv.json; None = no
    snapshot (every stale pv block is removed)."""
    snap = load_snapshot() if snap is _DEFAULT else snap
    players = payload.get("players") or {}
    teams = payload.get("teams") or {}
    for p in players.values():
        p.pop("pv", None)
    for t in teams.values():
        t.pop("pv", None)
    payload.pop("pv_meta", None)
    if snap is None:
        return 0
    sp = snap["players"]
    n = 0
    for pid, p in players.items():
        blocks = FAM_BLOCKS.get(p.get("fam") or "")
        src = sp.get(pid)
        if not blocks or not src:
            continue
        pv = {k: dict(src[k]) for k in blocks if isinstance(src.get(k), dict)}
        cov = pv.get("cov")
        if cov is not None:
            grp = cov.pop("grp", None)
            if grp is not None and grp != COV_GROUP.get(p.get("fam")):
                del pv["cov"]                    # another group's scale: not shown, not ranked
        for k, b in pv.items():
            b["q"] = 1 if qualified(k, b) else 0
        if pv:
            p["pv"] = pv
            n += 1
    # position percentiles: each measure against the qualified players of the family
    pools = {}
    for p in players.values():
        for k, b in (p.get("pv") or {}).items():
            if b["q"]:
                for m in RANKED[k]:
                    if isinstance(b.get(m), (int, float)):
                        pools.setdefault((p["fam"], k, m), []).append(float(b[m]))
    for p in players.values():
        for k, b in (p.get("pv") or {}).items():
            for m in RANKED[k]:
                pool = pools.get((p["fam"], k, m)) or []
                if len(pool) >= MIN_POOL and isinstance(b.get(m), (int, float)):
                    b["p_" + m] = percentile(float(b[m]), pool)
    # team unit numbers (QB + line protection, defensive front), ranked 1 = best
    su = snap.get("teams") or {}
    have = [c for c in teams if isinstance(su.get(c), dict)]
    for c in have:
        teams[c]["pv"] = {k: su[c][k] for k in UNIT_KEYS if k in su[c]}
    for k in UNIT_KEYS:
        vals = sorted((su[c][k], c) for c in have if k in su[c])
        if k.startswith("D_"):
            vals = vals[::-1]                        # defence: more = better
        for i, (_, c) in enumerate(vals):
            teams[c]["pv"].setdefault("rk", {})[k] = i + 1
    payload["pv_meta"] = {
        "through": snap.get("through"), "built": snap.get("built"),
        "model_use": MODEL_USE, "frozen": snap.get("frozen"),
        "league": snap.get("league"),
        "pool": {f"{f}|{k}|{m}": len(v) for (f, k, m), v in sorted(pools.items())},
        "qualify": {k: {"keys": list(v[0]), "min": v[1]} for k, v in QUALIFY.items()},
    }
    return n


# ======================================================================= build (local)
def _r(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if x != x else round(x, d) + 0.0      # + 0.0: no "-0.0" in the json


def build(out: str = SNAPSHOT) -> dict:
    import sys
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "phase0"))
    import pv_nfl_full_common as C

    span = C.events_span()
    last = span["season"]
    D = C.DATA
    qb = pd.read_parquet(os.path.join(D, "pv_nfl_full_passing_current.parquet"))
    rr = pd.read_parquet(os.path.join(D, "pv_nfl_full_rr_current.parquet"))
    fr_ = pd.read_parquet(os.path.join(D, "pv_nfl_full_front_current.parquet"))
    fu = pd.read_parquet(os.path.join(D, "pv_nfl_full_front_units_current.parquet"))
    cv = pd.read_parquet(os.path.join(D, "pv_nfl_full_coverage_current.parquet"))

    # who gets a block: anyone on this season's weekly roster file, or active in this
    # or last season (a free agent signed later still has his numbers)
    import csv
    roster = set()
    rp = os.path.join(D, f"roster_{last}.csv")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as fh:
            roster = {r["gsis_id"] for r in csv.DictReader(fh) if r.get("gsis_id")}

    P = {}

    def put(pid, k, block):
        P.setdefault(pid, {})[k] = {a: b for a, b in block.items() if b is not None}

    for r in qb.itertuples(index=False):
        if r.qb_id not in roster and r.last_season < last - 1:
            continue
        put(r.qb_id, "qb", {"v": _r(r.passing_composite), "epa": _r(r.epa_q),
                            "cpoe": _r(100 * r.cpoe_q, 1), "sack": _r(100 * r.sack_q, 1),
                            "db": int(r.db_cur), "dbp": int(r.db_prev), "dbc": int(r.db_career)})
    for r in rr.itertuples(index=False):
        if r.player_id not in roster and r.last_season < last - 1:
            continue
        if r.tgt_career > 0:
            put(r.player_id, "rec", {"v": _r(r.v_e), "xy": _r(r.mu_xypt, 2), "ye": _r(r.mu_yptoe, 2),
                                     "ts": _r(r.pre_tsh), "t": int(r.tgt_cur), "tp": int(r.tgt_prev),
                                     "tc": int(r.tgt_career)})
        if r.car_career > 0:
            put(r.player_id, "rush", {"v": _r(r.mu_epar), "ry": _r(r.mu_ryds, 2), "cs": _r(r.pre_csh),
                                      "c": int(r.car_cur), "cp": int(r.car_prev), "cc": int(r.car_career)})
    for r in fr_.itertuples(index=False):
        if r.player_id not in roster and r.g_cur + r.g_prev == 0:
            continue
        put(r.player_id, "front", {"pr": _r(100 * r.th_pr * r.L_pr, 2), "sk": _r(100 * r.th_sk * r.L_sk, 2),
                                   "st": _r(100 * r.th_st * r.L_st, 2), "sh": _r(r.th_pr),
                                   "g": int(r.g_cur), "gp": int(r.g_prev)})
    for r in cv.itertuples(index=False):
        if r.player_id not in roster and r.g_cur + r.g_prev == 0:
            continue
        put(r.player_id, "cov", {"ball": _r(100 * r.th_ball * (r.mu_pd + r.mu_int), 2), "x": _r(r.th_ball, 2),
                                 "ept": _r(r.epap), "ar": _r(100 * r.at_rate, 1),
                                 "g": int(r.g_cur), "gp": int(r.g_prev), "grp": str(r.group)})
    T = {}
    for r in fu.itertuples(index=False):
        T[r.team] = {k: _r(getattr(r, k)) for k in UNIT_KEYS}
    L0 = fr_.iloc[0]
    prov = {}
    if os.path.exists(C.PROVENANCE):
        with open(C.PROVENANCE, encoding="utf-8") as fh:
            prov = json.load(fh).get("sources", {})
    snap = {
        "what": "individual player measures from the player-value program (phase0/pv_nfl_*.py): "
                "each player rated on the events he individually controls, opponent-adjusted, "
                "walk-forward, empirical-Bayes shrunk; values as of each player's next game",
        "model_use": MODEL_USE,
        "frozen": "every hyper-parameter frozen at its DEV (<= 2015) value; values only, "
                  "no statistic comparing a value with a later outcome is computed past 2015",
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "through": {"season": last, "week": span["week"]},
        "league": {"pr_per_100_db": _r(100 * L0.L_pr, 2), "sk_per_100_db": _r(100 * L0.L_sk, 2),
                   "st_per_100_runs": _r(100 * L0.L_st, 2)},
        "sources": prov,
        "players": dict(sorted(P.items())),
        "teams": dict(sorted(T.items())),
    }
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, separators=(",", ":"), sort_keys=False)
    os.replace(tmp, out)
    print(f"[nfl_site_player_value] wrote {os.path.relpath(out, ROOT)}: {len(P):,} players, "
          f"{len(T)} teams, through {last} week {span['week']}", flush=True)
    return snap


WRAPPERS = ("pv_nfl_full_passing.py", "pv_nfl_full_receiving_rushing.py",
            "pv_nfl_full_pass_rush.py", "pv_nfl_full_coverage.py")


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    ap = argparse.ArgumentParser(description="build data/nfl_site_pv.json (display only)")
    ap.add_argument("--all", action="store_true",
                    help="first re-run the four pv_nfl_full_* wrappers on the current events table "
                         "(refresh it beforehand with: python phase0/pv_nfl_events_pull.py --refresh <season>)")
    if ap.parse_args().all:
        for w in WRAPPERS:
            subprocess.run([sys.executable, os.path.join(ROOT, "phase0", w)], check=True, cwd=ROOT)
    build()
