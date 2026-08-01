"""The NFL pre-game freeze: the honesty mechanism of the September machinery.

Until now this logic lived inline in phase0/nfl_season_serve.py, was verified
once by a synthetic week-1 dry run, and had NO test. Its failure mode is
invisible and self-flattering: the serve recomputes every game's probability on
every run, and for a PLAYED game those states have already walked through that
game's own result — so a break would silently publish post-hoc numbers as
"our pre-game pick" and inflate the track record.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))

import nfl_ph_freeze as F  # noqa: E402


def _row(w, home, away, ph, hs=None, aw=None, ct=None, pmc=None):
    return {"w": w, "home": home, "away": away, "ph": ph,
            "hs": hs, "as": aw, "ct": ct, "pmc": pmc}


def test_played_game_is_restored_from_the_ledger():
    """The whole point: a played game must publish its PRE-game number."""
    sched = [_row(1, "SEA", "DAL", ph=0.623, hs=24, aw=20, ct=[1, 2], pmc=0.7)]
    ledger = {"1|SEA|DAL": {"ph": 0.611, "ct": [9, 9], "pmc": 0.61}}
    led, frozen, orphans = F.freeze(sched, ledger)
    assert (frozen, orphans) == (1, 0)
    assert sched[0]["ph"] == 0.611, "post-hoc probability leaked into the payload"
    assert sched[0]["ct"] == [9, 9] and sched[0]["pmc"] == 0.61, (
        "ct/pmc must ride with ph — a frozen probability beside a recomputed "
        "contribution breakdown is internally inconsistent")
    assert led["1|SEA|DAL"]["ph"] == 0.611, "a played game must not rewrite its entry"


def test_unplayed_game_refreshes_the_stored_forecast():
    """Before kickoff the stored value tracks the latest forecast, so what is
    eventually frozen is the most-informed pre-game number, not the first one."""
    sched = [_row(2, "KC", "BUF", ph=0.58)]
    ledger = {"2|KC|BUF": {"ph": 0.51, "ct": None, "pmc": None}}
    led, frozen, orphans = F.freeze(sched, ledger)
    assert (frozen, orphans) == (0, 0)
    assert led["2|KC|BUF"]["ph"] == 0.58
    assert sched[0]["ph"] == 0.58, "an unplayed row must keep the fresh forecast"


def test_orphaned_played_game_warns_and_never_pretends():
    """No ledger entry means the pre-game number is GONE. It cannot be repaired,
    so the recomputed value stays — but it must warn, and it must be recorded so
    the next run is not orphaned too. Silence here is the failure mode."""
    sched = [_row(3, "GB", "CHI", ph=0.66, hs=17, aw=13)]
    warnings = []
    led, frozen, orphans = F.freeze(sched, {}, warn=warnings.append)
    assert (frozen, orphans) == (0, 1)
    assert warnings and "no ledger entry" in warnings[0]
    assert "3|GB|CHI" in led, "the orphan must be recorded for the next run"


def test_a_tie_counts_as_played():
    """NFL ties are real. hs/as present -> played, regardless of the scores."""
    sched = [_row(4, "NYG", "WAS", ph=0.70, hs=20, aw=20)]
    led, frozen, _ = F.freeze(sched, {"4|NYG|WAS": {"ph": 0.52}})
    assert frozen == 1 and sched[0]["ph"] == 0.52


def test_partial_score_is_not_played():
    """One score without the other is a data glitch, not a result — freezing on
    it would lock a pre-game number in from a half-written payload."""
    sched = [_row(5, "SF", "LA", ph=0.61, hs=21, aw=None)]
    led, frozen, orphans = F.freeze(sched, {"5|SF|LA": {"ph": 0.55}})
    assert (frozen, orphans) == (0, 0)
    assert sched[0]["ph"] == 0.61 and led["5|SF|LA"]["ph"] == 0.61


def test_bootstrap_seeds_from_a_pre_ledger_payload():
    payload = {"schedule": [
        {"w": 1, "home": "SEA", "away": "DAL", "ph": 0.61, "ct": [1], "pmc": 0.6},
        {"w": 1, "home": "KC", "away": "BUF", "ph": None},      # skipped
    ]}
    led = F.bootstrap_from_payload(payload)
    assert set(led) == {"1|SEA|DAL"} and led["1|SEA|DAL"]["ph"] == 0.61


def test_shipped_ledger_matches_the_shipped_payload():
    """Contract against the real artifacts: every payload row must have an entry,
    so no game can become an orphan at kickoff."""
    import json
    root = Path(__file__).resolve().parents[1]
    lp, pp = root / "data" / "nfl_ph_ledger.json", root / "site" / "data" / "nfl.json"
    if not (lp.is_file() and pp.is_file()):
        import pytest
        pytest.skip("NFL artifacts not built")
    led = json.loads(lp.read_text(encoding="utf-8"))
    sched = json.loads(pp.read_text(encoding="utf-8"))["schedule"]
    missing = [F.ledger_key(r) for r in sched if F.ledger_key(r) not in led]
    assert not missing, f"{len(missing)} payload games have no ledger entry: {missing[:3]}"
