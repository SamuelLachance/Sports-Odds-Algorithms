"""MLB TrueSkill-3 sliver screens — DEV-ONLY 5-fold CV of the full 6-feature blend.

Extends phase0/ts2_mlb_eval.py (machinery imported from it). Three screens on the
per-PA batter-vs-pitcher engine, WITHOUT touching mlbwp/trueskill.py:

  S1 — TTT-style boundary re-walk: at each season boundary, re-walk the previous
       ttt_depth seasons' PAs once more through the same update (warm start,
       strictly-past PAs only — leak-safe: the re-walk happens before any game of
       the new season is read). Trials: depth 1 and depth 3, one extra pass each.
  S2 — team-change sigma widening: player team inferred per game from
       data/retro_events/lineups.csv side membership (side 0 = visitor,
       side 1 = home) + the game log's home/vis teams, plus the two starting
       pitchers. On a player's FIRST appearance for a new team vs their stored
       last team: var[p] = min(var[p] + tc_w**2, SIG0**2), once per change.
       Trials tc_w in {0.5, 1.5} (mature sigma ~0.8-0.9, SIG0 = 8.33).
  S3 — Student-t robust damping: damp = 1/(1 + t*t/nu) after t = (mu_w-mu_l)/c;
       the mu deltas AND the variance-reduction factors are scaled by damp.
       Trials nu in {4.0, 9.0}.

(Continuity is structurally N/A in a 1v1 engine — not implemented.)

Protocol (identical to ts2_mlb_eval):
  * walk ALL PAs in strict (date, game_id) order; per-game feature read BEFORE
    that game's PAs are applied; feature = lineup-mean offense vs starter defense
  * blend = logistic on [logit_fip, ts_z, bp_z, pw_z, siera_z, baserun_z]
  * DEV = seasons 2002-2015 only; KFold(5, shuffle=True, random_state=7)
  * metric = pooled out-of-fold DEV log loss; gate: improvement >= 0.0020
  * TEST (2016+) is NEVER touched.

Baseline check: the parameterized TS3 engine with nu=None, tc_w=0, depth=0 must
reproduce a fresh walk of the shipped TrueSkill1v1 bit-identically (hard assert)
and match the shipped ts_feature.csv at its 4dp rounding level.

Diagnostics: team-change event count, share of PAs with damp < 0.5,
TTT mean |mu change| per boundary re-walk.

Writes data/ts3_mlb_screen.json.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

import ts2_mlb_eval as T2  # noqa: E402  (also sets thread env vars)
import trueskill_pa as TP  # noqa: E402
from mlbwp.trueskill import MU0, SIG0, BETA, TAU, TrueSkill1v1, _cdf, _pdf  # noqa: E402

GATE = 0.0020
OUT = "data/ts3_mlb_screen.json"
F_DATE, F_HOME, F_VIS = 0, 6, 3


class TS3(object):
    """Parameterized sliver-variant engine of mlbwp.trueskill.TrueSkill1v1.

    nu        — Student-t damping dof; None => shipped update (bit-identical).
    tc_w      — team-change sigma widening (applied by the walk, not update).
    ttt_depth — boundary re-walk depth in seasons (applied by the walk).

    The nu=None update body is copied verbatim from TrueSkill1v1.update so the
    param baseline is bit-identical to the shipped engine (asserted in main()).
    """

    def __init__(self, nu=None, tc_w=0.0, ttt_depth=0,
                 beta=BETA, tau=TAU, mu0=MU0, sig0=SIG0):
        self.beta, self.tau, self.mu0, self.sig0 = beta, tau, mu0, sig0
        self.nu = None if nu is None else float(nu)
        self.tc_w2 = float(tc_w) ** 2
        self.ttt_depth = int(ttt_depth)
        self.mu = defaultdict(lambda: mu0)
        self.var = defaultdict(lambda: sig0 * sig0)
        self.n = defaultdict(int)
        self.tau2 = tau ** 2                      # ** to match self.tau ** 2 exactly
        self.pa_cnt = 0                           # nu path only
        self.damp_lt = 0                          # nu path only: damp < 0.5
        self.tc_applied = 0

    def update(self, winner, loser):
        vw = self.var[winner] + self.tau2
        vl = self.var[loser] + self.tau2
        c2 = 2 * self.beta ** 2 + vw + vl
        c = math.sqrt(c2)
        t = (self.mu[winner] - self.mu[loser]) / c
        denom = _cdf(t)
        v = _pdf(t) / denom if denom > 1e-12 else -t
        w = min(max(v * (v + t), 0.0), 1.0)
        nu = self.nu
        if nu is None:
            self.mu[winner] += vw / c * v
            self.mu[loser] -= vl / c * v
            self.var[winner] = vw * (1 - vw / c2 * w)
            self.var[loser] = vl * (1 - vl / c2 * w)
        else:
            damp = 1.0 / (1.0 + (t * t) / nu)
            self.pa_cnt += 1
            if damp < 0.5:
                self.damp_lt += 1
            self.mu[winner] += damp * (vw / c) * v
            self.mu[loser] -= damp * (vl / c) * v
            self.var[winner] = vw * (1 - damp * (vw / c2) * w)
            self.var[loser] = vl * (1 - damp * (vl / c2) * w)
        self.n[winner] += 1
        self.n[loser] += 1


def load_vis_teams():
    """game_id -> visiting team from the game logs (home team = game_id[:3])."""
    out = {}
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            out[r[F_HOME] + r[F_DATE] + r[1]] = r[F_VIS]
    return out


def team_change_events(order, lineups, starters, vis_teams):
    """Walk-order team inference: side 0 batters + vis SP -> vis team, side 1
    batters + home SP -> home team. Returns {gid: [players changing team at gid]}
    (a change = stored last team exists and differs), plus counts."""
    last = {}
    events = {}
    n_events = 0
    changers = set()
    for gid in order:
        vis = vis_teams.get(gid)
        if vis is None:
            continue
        home = gid[:3]
        ln = lineups.get(gid)
        sp = starters.get(gid)
        members = []
        if ln:
            members.extend((b, vis) for b in ln[0])
            members.extend((b, home) for b in ln[1])
        if sp:
            if sp[0]:
                members.append((sp[0], home))    # home starting pitcher
            if sp[1]:
                members.append((sp[1], vis))     # visiting starting pitcher
        changed = []
        for p, tm in members:
            lt = last.get(p)
            if lt is not None and lt != tm:
                changed.append(p)
                changers.add(p)
            last[p] = tm
        if changed:
            events[gid] = changed
            n_events += len(changed)
    return events, n_events, len(changers)


def walk3(order, pa_games, lineups, starters, engines, season_pas, season_players,
          tc_events):
    """ts2_mlb_eval.walk plus the S1/S2 hooks. Feature read (mu only) is identical
    to trueskill_pa.build_feature; widening/re-walks touch var/mu before any of
    the new game's/season's PAs or feature reads, so everything stays leak-safe
    (strictly-past information only)."""
    feats = [dict() for _ in engines]
    ttt_engines = [(k, e) for k, e in enumerate(engines)
                   if getattr(e, "ttt_depth", 0) > 0]
    tc_engines = [e for e in engines if getattr(e, "tc_w2", 0.0) > 0.0]
    ttt_log = {k: [] for k, _ in ttt_engines}
    cur_season = None
    for gid in order:
        g = pa_games[gid]
        s = g["season"]
        if cur_season is not None and s != cur_season:
            # ---- S1: boundary re-walk of the previous depth seasons ----
            for k, e in ttt_engines:
                seasons = [ss for ss in range(cur_season - e.ttt_depth + 1,
                                              cur_season + 1) if ss in season_pas]
                involved = set()
                for ss in seasons:
                    involved |= season_players[ss]
                before = {p: e.mu[p] for p in involved}
                n_pas = 0
                for ss in seasons:
                    for pa_list in season_pas[ss]:
                        n_pas += len(pa_list)
                        for b, p, ob in pa_list:
                            if ob:
                                e.update(b, p)
                            else:
                                e.update(p, b)
                mu = e.mu
                mch = (sum(abs(mu[p] - before[p]) for p in involved) / len(involved)
                       if involved else 0.0)
                ttt_log[k].append({"into_season": s, "replayed_seasons": seasons,
                                   "n_players": len(involved), "n_pas": n_pas,
                                   "mean_abs_mu_change": round(mch, 5)})
        cur_season = s
        # ---- S2: team-change sigma widening (var only; once per change) ----
        changed = tc_events.get(gid)
        if changed:
            for e in tc_engines:
                w2, cap = e.tc_w2, e.sig0 * e.sig0
                for p in changed:
                    e.var[p] = min(e.var[p] + w2, cap)
                    e.tc_applied += 1
        # ---- feature read BEFORE this game's PAs (identical to ts2 walk) ----
        ln = lineups.get(gid)
        sp = starters.get(gid)
        if ln and sp and ln[1] and ln[0] and sp[0] and sp[1]:
            for k, ts in enumerate(engines):
                mu = ts.mu
                home_bat = [mu[b] for b in ln[1]]
                away_bat = [mu[b] for b in ln[0]]
                home_off = sum(home_bat) / len(home_bat)
                away_off = sum(away_bat) / len(away_bat)
                feats[k][gid] = (home_off - mu[sp[1]]) - (away_off - mu[sp[0]])
        # ---- apply this game's PAs to every engine ----
        for batter, pitcher, on_base in g["pa"]:
            if on_base:
                for ts in engines:
                    ts.update(batter, pitcher)
            else:
                for ts in engines:
                    ts.update(pitcher, batter)
    return feats, ttt_log


def main():
    t0 = time.time()
    print("building fixed blend inputs (FIP-Elo, siera, bullpen, power, baserun)...",
          flush=True)
    games, rec, sdiff, bpd, pdiff, brdiff = T2.blend_inputs()
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print("loading PA data + lineups + starters + teams...", flush=True)
    pa_games = TP.load_pa_by_game()
    lineups = TP.load_lineups()
    starters = TP.load_starters()
    vis_teams = load_vis_teams()
    order = sorted(pa_games, key=lambda g: (pa_games[g]["date"], g))
    print(f"  {len(pa_games):,} games with PAs  ({time.time() - t0:.0f}s)", flush=True)

    # per-season caches for the S1 re-walks (walk order preserved)
    season_pas = defaultdict(list)      # season -> [game PA list, ...]
    season_players = defaultdict(set)
    for gid in order:
        g = pa_games[gid]
        season_pas[g["season"]].append(g["pa"])
        ps = season_players[g["season"]]
        for b, p, _ in g["pa"]:
            ps.add(b)
            ps.add(p)

    tc_events, n_tc_events, n_tc_players = team_change_events(
        order, lineups, starters, vis_teams)
    print(f"  team-change events: {n_tc_events:,} over {n_tc_players:,} players",
          flush=True)

    engines = [TrueSkill1v1(), TS3()]        # fresh shipped walk + param baseline
    names = ["baseline_fresh", "baseline_param"]
    params = [{}, {"nu": None, "tc_w": 0, "ttt_depth": 0}]
    for d in (1, 3):
        engines.append(TS3(ttt_depth=d))
        names.append(f"s1_ttt_depth{d}")
        params.append({"nu": None, "tc_w": 0, "ttt_depth": d})
    for wch in (0.5, 1.5):
        engines.append(TS3(tc_w=wch))
        names.append(f"s2_teamchange_w{wch:g}")
        params.append({"nu": None, "tc_w": wch, "ttt_depth": 0})
    for nu in (4.0, 9.0):
        engines.append(TS3(nu=nu))
        names.append(f"s3_studentt_nu{nu:g}")
        params.append({"nu": nu, "tc_w": 0, "ttt_depth": 0})

    n_pas = sum(len(pa_games[g]["pa"]) for g in pa_games)
    print(f"walking {n_pas:,} PAs through {len(engines)} engines "
          f"(+ S1 boundary re-walks)...", flush=True)
    feats, ttt_log = walk3(order, pa_games, lineups, starters, engines,
                           season_pas, season_players, tc_events)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # ---- baseline exactness: param engine vs fresh shipped walk + shipped CSV ----
    f_fresh, f_param = feats[0], feats[1]
    assert set(f_fresh) == set(f_param), "baseline feature game sets differ"
    repro_max = max(abs(f_fresh[g] - f_param[g]) for g in f_fresh)
    assert repro_max < 1e-9, f"param baseline does not reproduce shipped engine: {repro_max}"
    shipped = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv"))
               if r[0] != "game_id"}
    common = set(shipped) & set(f_fresh)
    csv_max = max(abs(shipped[g] - f_fresh[g]) for g in common) if common else None
    assert csv_max is not None and csv_max < 1e-4, \
        f"fresh walk does not match shipped csv at 4dp rounding: {csv_max}"
    print(f"baseline repro: max|diff| fresh-vs-param = {repro_max:.2e}  "
          f"vs shipped csv (4dp) = {csv_max:.2e} over {len(common):,}/{len(shipped):,} games",
          flush=True)

    # ---- DEV matrix + CV (identical machinery to ts2) ----
    gids, lf, y, bpz, pwz, siz, brz = T2.dev_matrix(
        games, rec, sdiff, bpd, pdiff, brdiff, f_fresh)
    print(f"DEV blend rows: {len(gids):,} (seasons 2002-2015)", flush=True)

    import numpy as np

    def ts_col(fd):
        v = np.array([fd[g] for g in gids])
        return (v - v.mean()) / v.std()

    base_ll = T2.cv_ll(lf, ts_col(f_fresh), bpz, pwz, siz, brz, y)
    print(f"BASELINE DEV CV LL = {base_ll:.5f}", flush=True)

    results = {}
    for name, fd in zip(names[2:], feats[2:]):
        v_ll = T2.cv_ll(lf, ts_col(fd), bpz, pwz, siz, brz, y)
        results[name] = v_ll
        print(f"  {name:22s} CV LL = {v_ll:.5f}  delta = {base_ll - v_ll:+.5f}",
              flush=True)

    # ---- diagnostics ----
    ttt_diag = {}
    for k, log in ttt_log.items():
        chs = [r["mean_abs_mu_change"] for r in log]
        ttt_diag[names[k]] = {
            "n_boundary_rewalks": len(log),
            "mean_abs_mu_change_avg": round(sum(chs) / len(chs), 5) if chs else None,
            "per_rewalk": log,
        }
    damp_diag = {names[k]: {"pa_count": e.pa_cnt,
                            "share_damp_lt_0.5": round(e.damp_lt / e.pa_cnt, 6)
                            if e.pa_cnt else None}
                 for k, e in enumerate(engines) if e.__class__ is TS3 and e.nu is not None}
    tc_diag = {names[k]: {"widenings_applied": e.tc_applied}
               for k, e in enumerate(engines)
               if e.__class__ is TS3 and e.tc_w2 > 0.0}

    deltas = {n: base_ll - results[n] for n in results}
    best_name = max(deltas, key=deltas.get)
    gate_met = deltas[best_name] >= GATE

    out = {
        "screen": "MLB per-PA TrueSkill-3 slivers: TTT boundary re-walk + "
                  "team-change sigma widening + Student-t damping",
        "protocol": {
            "dev_seasons": "2002-2015", "cv": "KFold(5, shuffle=True, random_state=7)",
            "metric": "pooled out-of-fold DEV log loss of the 6-col blend "
                      "[logit_fip, ts_z, bp_z, pw_z, siera_z, baserun_z]",
            "gate": GATE, "n_dev_games": len(gids), "test_touched": False,
            "note": "continuity screen structurally N/A in a 1v1 engine",
        },
        "baseline": {
            "cv_ll": round(base_ll, 6),
            "repro_max_abs_diff_vs_fresh_walk": repro_max,
            "max_abs_diff_vs_shipped_csv_4dp": csv_max,
            "n_feature_games": len(f_fresh),
        },
        "variants": [
            {"name": n, "params": p, "cv_ll": round(results[n], 6),
             "delta": round(deltas[n], 6)}
            for n, p in zip(names[2:], params[2:])
        ],
        "gate_met": bool(gate_met),
        "best_variant": best_name,
        "best_delta": round(deltas[best_name], 6),
        "diagnostics": {
            "total_pas": n_pas,
            "team_change_events": n_tc_events,
            "team_change_players": n_tc_players,
            "s2_widenings_applied": tc_diag,
            "s3_damp_share": damp_diag,
            "s1_ttt_rewalks": ttt_diag,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}", flush=True)
    print(f"gate (>= +{GATE:.4f}): {'MET' if gate_met else 'NOT MET'} — "
          f"best {best_name} delta {deltas[best_name]:+.5f}", flush=True)


if __name__ == "__main__":
    main()
