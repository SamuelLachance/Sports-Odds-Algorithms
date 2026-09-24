"""Offline checks for the NHL boxel screen (breakthrough program, round 2).

No network and no data files: synthetic payloads and a synthetic game list.

  * the box-score parser maps the api-web schema to the pre-declared columns
    (one row per dressed player, goalie starter flag, TOI in seconds);
  * the screen's projected-lineup pass reads EXACTLY what round 1's lineup pass
    reads for tonight's actual lineup (bit-identical), and its projected read
    uses the team's previous game's dressed list, never tonight's;
  * elo_dual's actual-input probability is round 1's elo_cell, and its projected
    probability collapses onto it when the projected inputs equal the actual ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "phase0"
for p in (str(PHASE0), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bt_nhl_boxel_fetch as F  # noqa: E402
import bt_nhl_boxel_screen as S  # noqa: E402
import bt_nhl_nhl_boxel as R1  # noqa: E402


def test_parser_maps_api_schema():
    d = {"id": 2011020001, "homeTeam": {"abbrev": "BOS"}, "awayTeam": {"abbrev": "PHI"},
         "playerByGameStats": {
             "homeTeam": {
                 "forwards": [{"playerId": 1, "position": "C", "goals": 1, "assists": 2,
                               "sog": 3, "blockedShots": 1, "pim": 2, "toi": "17:05"}],
                 "defense": [{"playerId": 2, "position": "D", "goals": 0, "assists": 0,
                              "sog": 1, "blockedShots": 3, "pim": 0, "toi": "22:30"}],
                 "goalies": [{"playerId": 3, "position": "G", "starter": True,
                              "toi": "59:15", "pim": 0},
                             {"playerId": 5, "position": "G", "starter": False,
                              "toi": "00:00", "pim": 0}]},
             "awayTeam": {"forwards": [], "defense": [],
                          "goalies": [{"playerId": 4, "starter": True, "toi": "60:00"}]}}}
    rows = F.parse(2011020001, d)
    assert rows[0] == [2011020001, "home", "BOS", 1, "C", 0, 0, 1, 2, 3, 1, 2, 1025]
    assert rows[1] == [2011020001, "home", "BOS", 2, "D", 0, 0, 0, 0, 1, 3, 0, 1350]
    starters = {(r[1], r[3]): r[6] for r in rows if r[5] == 1}
    assert starters == {("home", 3): 1, ("home", 5): 0, ("away", 4): 1}
    assert all(len(r) == len(F.COLS) for r in rows)
    assert F.parse(1, {"id": 1}) is None


def _synthetic(n_days=40, seed=3):
    """Four teams, two games a day, 18 skaters a side, two seasons."""
    rng = np.random.default_rng(seed)
    teams = ["AAA", "BBB", "CCC", "DDD"]
    pool = {t: [1000 * (k + 1) + j for j in range(22)] for k, t in enumerate(teams)}
    games, box = [], {}
    gid = 2010020001
    for d in range(n_days):
        season = 20102011 if d < n_days // 2 else 20112012
        date = f"{2010 + d // 20}-{10 + (d % 20) // 10:02d}-{1 + d % 10:02d}"
        order = rng.permutation(teams)
        for h, a in ((order[0], order[1]), (order[2], order[3])):
            g = {"game_id": gid, "season": season, "date": date, "home": h, "away": a,
                 "y": int(rng.random() < 0.55)}
            games.append(g)
            box[gid] = {}
            for side, t in (("home", h), ("away", a)):
                pids = rng.choice(pool[t], size=18, replace=False)
                sk = []
                for p in pids:
                    pc = "D" if p % 3 == 0 else "F"
                    sk.append((int(p), pc, pc, int(rng.poisson(0.15)), int(rng.poisson(0.25)),
                               int(rng.poisson(1.8)), int(rng.poisson(0.7)),
                               int(2 * rng.poisson(0.2)), float(rng.normal(900, 120))))
                box[gid][side] = {"sk": sk, "gstart": None, "goalies": []}
            gid += 1
    return games, box


def test_projected_pass_actual_read_is_round1_bit_identical():
    games, box = _synthetic()
    lp = R1.lineup_pass(games, box, {})
    lpp = S.lineup_pass_proj(games, box, {})
    assert np.array_equal(lpp["Lth"], lp["Lth"])
    assert np.array_equal(lpp["Lta"], lp["Lta"])
    # first appearance of each of the 4 teams has no previous-game roster
    assert lpp["proj_empty_sides"] == 4


def test_projected_read_uses_previous_game_roster_not_tonight():
    games, box = _synthetic()
    lpp = S.lineup_pass_proj(games, box, {})
    # make each team's roster identical game to game: projected == actual once warm
    fixed = {}
    box2 = {}
    for g in games:
        box2[g["game_id"]] = {}
        for side in ("home", "away"):
            t = g[side]
            fixed.setdefault(t, box[g["game_id"]][side])
            box2[g["game_id"]][side] = fixed[t]
    lp2 = S.lineup_pass_proj(games, box2, {})
    seen = set()
    for i, g in enumerate(games):
        for side, a, p in (("home", "Lth", "Lph"), ("away", "Lta", "Lpa")):
            if g[side] in seen:
                assert lp2[p][i] == lp2[a][i]
            seen.add(g[side])
    # changing tonight's roster never moves tonight's projected value
    box3 = {gid: dict(v) for gid, v in box.items()}
    last = games[-1]["game_id"]
    box3[last] = {"home": box[games[0]["game_id"]]["home"],
                  "away": box[games[0]["game_id"]]["away"]}
    lp3 = S.lineup_pass_proj(games, box3, {})
    assert lp3["Lph"][-1] == lpp["Lph"][-1] and lp3["Lpa"][-1] == lpp["Lpa"][-1]


def test_elo_dual_matches_round1_elo_cell():
    games, box = _synthetic()
    n = len(games)
    rng = np.random.default_rng(1)
    Lh, La, Gh, Ga = (rng.normal(size=n) for _ in range(4))
    pa, pp = S.elo_dual(games, 30, 120, Lh, La, Gh, Ga, Lh, La, Gh, Ga)
    assert np.array_equal(pa, R1.elo_cell(games, 30, 120, Lh, La, Gh, Ga))
    assert np.array_equal(pa, pp)
    pa2, pp2 = S.elo_dual(games, 30, 120, Lh, La, Gh, Ga, La, Lh, Ga, Gh)
    assert np.array_equal(pa2, pa) and not np.array_equal(pp2, pa)
