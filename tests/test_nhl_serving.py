"""Guards on the NHL serving math — the path that prices upcoming games in-season.

Two failure modes these catch:
  1. An accidental refit: data/nhl_model.json is FROZEN (one-look protocol).
     Any change to its coefficients must be a deliberate re-freeze that updates
     the pin here in the same commit.
  2. A regression in the final_states() walk that would silently shift every
     upcoming-game probability (wrong warm states -> wrong October prices).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _model():
    p = ROOT / "data" / "nhl_model.json"
    if not p.is_file():
        pytest.skip("nhl_model.json not built")
    return json.loads(p.read_text(encoding="utf-8"))


def test_frozen_model_pin():
    m = _model()
    assert m["test_ll"] == 0.66418, "frozen TEST number changed — deliberate re-freeze?"
    assert m["elo_cfg"] == {"k": 8, "ha": 30, "regress": 0.3}
    b = m["blend"]
    assert set(b["coefs"]) == {"elo_logit", "rest", "b2b_home", "b2b_away", "xg"}
    assert b["coefs"]["elo_logit"] == 0.70016
    assert b["coefs"]["xg"] == 0.37079
    assert "intercept" in b


def test_final_states_walk():
    pytest.importorskip("numpy")
    from nhl_glicko2_eval import load_games
    from nhl_serve import final_states

    m = _model()
    games = load_games()
    R, xg, last = final_states(games, m)

    cur = max(g["season"] for g in games)
    cur_teams = {g[s] for g in games if g["season"] == cur for s in ("home", "away")}
    assert len(cur_teams) == 32
    assert cur_teams <= set(R), "current team missing from the Elo walk"

    vals = [R[t] for t in cur_teams]
    assert 1300 < min(vals) and max(vals) < 1700, "Elo walk left the sane band"
    assert abs(sum(R.values()) / len(R) - 1500) < 5, "Elo drifted off its 1500 anchor"

    # every current team has a last-game date inside the final season
    season_dates = sorted(g["date"] for g in games if g["season"] == cur)
    for t in cur_teams:
        assert season_dates[0] <= last[t] <= season_dates[-1]

    # deterministic: same inputs -> bit-identical states
    R2, xg2, last2 = final_states(games, m)
    assert R == R2 and dict(xg) == dict(xg2) and last == last2


def test_upcoming_slate_served():
    """If an upcoming slate exists, every resolvable game must appear in the
    payload with a real probability (the in-season serving contract)."""
    up_p = ROOT / "data" / "nhl_upcoming.json"
    pay_p = ROOT / "site" / "data" / "nhl.json"
    if not up_p.is_file() or not pay_p.is_file():
        pytest.skip("no upcoming slate (offseason)")
    upcoming = json.loads(up_p.read_text(encoding="utf-8"))
    if not upcoming:
        pytest.skip("empty upcoming slate (offseason)")
    payload = json.loads(pay_p.read_text(encoding="utf-8"))
    served = {g["id"]: g for g in payload["schedule"]}
    known = {t for t in payload["teams"]}
    for u in upcoming:
        if u["home"] not in known or u["away"] not in known:
            continue  # serve refuses unresolved teams by design
        assert u["id"] in served, f"upcoming game {u['id']} missing from payload"
        assert 0.0 < served[u["id"]]["hp"] < 1.0
