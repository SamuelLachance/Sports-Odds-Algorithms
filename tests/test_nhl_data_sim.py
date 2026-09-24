"""NHL Monte Carlo season projection (phase0/nhl_site_sim.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_site_sim as SIM  # noqa: E402
import nhl_site_teams as ST  # noqa: E402

CFG = {"k": 8, "ha": 30, "regress": 0.3}
BLEND = {"intercept": -0.02695, "coefs": {"elo_logit": 0.70016, "rest": -0.0033,
                                          "b2b_home": -0.2839, "b2b_away": 0.2726,
                                          "xg": 0.37079}}


def _season(n_rounds=4):
    """Every team hosts every other team n_rounds/2 times (balanced)."""
    teams = list(ST.TEAMS)
    games = []
    for r in range(n_rounds):
        for i, h in enumerate(teams):
            a = teams[(i + 1 + r) % len(teams)]
            games.append({"home": h, "away": a, "hrest": 2, "arest": 2, "hb2b": 0, "ab2b": 0})
    return games


def _run(elo, n=4000, seed=1, base=None, rounds=4):
    meta = {t: {"div": m["div"], "conf": m["conf"]} for t, m in ST.team_meta().items()}
    return SIM.simulate(_season(rounds), elo, {}, meta, base or {}, CFG, BLEND, 0.15,
                        n_sims=n, seed=seed, batch=1000)


def test_points_and_probabilities_are_conserved():
    elo = {t: 1500.0 + 5 * i for i, t in enumerate(ST.TEAMS)}
    out, info = _run(elo)
    G = info["games_left"]
    total = sum(v["pts"] for v in out.values())
    # every game hands out 2 points, 3 when it reaches OT/SO
    assert abs(total - G * (2 + SIM.P_OT)) < G * 0.02
    assert abs(sum(v["po"] for v in out.values()) - 1600.0) < 0.5      # 16 playoff spots
    assert abs(sum(v["pres"] for v in out.values()) - 100.0) < 0.5
    for d in ST.DIVISIONS:
        s = sum(out[t]["div"] for t in ST.TEAMS if ST.team_meta()[t]["div"] == d)
        assert abs(s - 100.0) < 0.5
    for v in out.values():
        assert abs(v["w"] + v["l"] + v["otl"] - v["rem"]) < 0.2
        assert v["lo"] <= v["pts"] <= v["hi"]


def test_stronger_team_projects_higher_and_is_deterministic():
    elo = {t: 1500.0 for t in ST.TEAMS}
    elo["COL"] = 1700.0
    elo["SJS"] = 1300.0
    a, _ = _run(elo, n=2000, seed=7, rounds=40)       # 80 games a team
    b, _ = _run(elo, n=2000, seed=7, rounds=40)
    assert a == b
    assert a["COL"]["pts"] > a["DAL"]["pts"] > a["SJS"]["pts"]
    assert a["COL"]["po"] > 90 and a["SJS"]["po"] < 10


def test_played_points_carry_into_the_projection():
    elo = {t: 1500.0 for t in ST.TEAMS}
    base = {"TOR": {"pts": 40, "w": 20, "l": 0, "otl": 0, "rw": 20}}
    out, _ = _run(elo, base=base)
    assert out["TOR"]["pts"] > 40 + 5 and out["TOR"]["w"] >= 20
    assert out["TOR"]["po"] > out["MTL"]["po"]


def test_overtime_losers_take_points():
    elo = {t: 1500.0 for t in ST.TEAMS}
    out, _ = _run(elo, n=2000)
    otl = sum(v["otl"] for v in out.values())
    games = len(_season())
    assert abs(otl / games - SIM.P_OT) < 0.02
