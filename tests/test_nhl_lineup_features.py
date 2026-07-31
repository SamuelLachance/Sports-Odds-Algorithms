"""Synthetic round-trip for the NHL announced-lineup feature path.

data/nhl_lineup_archive.jsonl is written only on game days, so between April and
October there is nothing to run this code against and no way to notice it broke.
That is exactly the situation these tests exist to remove: they build a
synthetic archive covering every payload shape phase0/nhl_lineup_features.py
claims to survive, push it through the real reader and the real feature
builders, and assert the values by hand.

The three properties worth failing a build over:

  * the reader never raises on archive content (a torn append, an unknown key,
    a record with no lineup are all normal), and
  * a missing input yields NaN, never 0 — a zero would silently read as "no
    scratches / usual goalie in net", which is the common case, so it would
    bury every real signal inside false negatives, and
  * side orientation is never guessed. Swapping the home and away scratch lists
    flips the sign of the whole feature, which is the worst failure available
    to this module and the one a screen would not catch.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_lineup_features as lf  # noqa: E402


@pytest.fixture
def archive(tmp_path):
    p = tmp_path / "nhl_lineup_archive.jsonl"
    p.write_text("\n".join(lf.synthetic_archive_lines()), encoding="utf-8")
    return p


@pytest.fixture
def feats(archive):
    snaps, _ = lf.read_archive(str(archive), names={})
    usual = lf.UsualStarter(min_obs=1)
    usual.hist["BOS"] = [900019, 900019, 900019]
    usual.hist["MTL"] = [900119, 900119, 900119]
    return lf.build_features(snaps, rapm=lf._synthetic_rapm(), usual=usual)


def test_module_selftest_passes():
    """The module's own dry run is the contract; keep it green in CI."""
    assert lf.selftest(verbose=False) == 0


def test_reader_survives_a_torn_append_and_unknown_keys(archive):
    snaps, rep = lf.read_archive(str(archive), names={})
    assert rep["records"] == 3
    assert rep["bad_json"] == 1, "the partial final line must be counted, not raised"
    assert rep["no_lineup"] == 1
    assert rep["unknown_lineup_keys"].get("warmupNotes") == 1, (
        "a field the API adds later must be inventoried, not fatal")
    assert len(snaps) == 3


def test_reader_on_a_missing_archive_is_empty_not_an_error(tmp_path):
    snaps, rep = lf.read_archive(str(tmp_path / "nope.jsonl"))
    assert snaps == [] and rep["exists"] is False and rep["records"] == 0
    assert lf.build_features(snaps, rapm={}) == {}


def test_sides_are_resolved_not_guessed(archive):
    snaps, _ = lf.read_archive(str(archive), names={})
    g1 = lf.merge_snapshots(snaps)[2026020001]
    assert g1.dressed["homeTeam"][0] == 900001, "home teamId 6 -> BOS -> homeTeam"
    assert g1.scratches["awayTeam"] == [900150, 900151]


def test_unresolvable_orientation_refuses_to_guess(tmp_path):
    rec = json.loads(lf.synthetic_archive_lines()[1])
    rec["home"], rec["away"] = "CGY", "EDM"        # abbrevs matching neither teamId
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(rec), encoding="utf-8")
    snaps, _ = lf.read_archive(str(p), names={})
    assert len(snaps) == 1
    assert "dressed" not in snaps[0].present
    assert "scratches" in snaps[0].present, (
        "gameInfoScratches are already side-keyed, so they survive an "
        "unresolvable rosterSpots block")
    f = lf.build_features(snaps, rapm=lf._synthetic_rapm())[2026020001]
    assert math.isnan(f["dressed_power_diff"])


def test_scratch_feature_value_and_sign(feats):
    f = feats[2026020001]
    # MTL scratched 0.20 and -0.05 (0.15 lost); BOS scratched 0.40
    assert f["scratch_rapm_diff"] == pytest.approx(0.15 - 0.40)
    assert f["scratch_rapm_diff"] < 0, (
        "away-minus-home convention (row 10): the side that lost the better "
        "player must get the negative sign")
    assert f["scratch_rapm_pos_diff"] == pytest.approx(0.20 - 0.40)
    assert f["scratch_n_diff"] == 1.0
    assert f["n_unrated"] == 0


def test_unrated_scratches_score_zero_and_are_counted(archive):
    snaps, _ = lf.read_archive(str(archive), names={})
    rapm = lf._synthetic_rapm()
    del rapm[900150]                               # a call-up below the RAPM floor
    f = lf.build_features(snaps, rapm=rapm)[2026020001]
    assert f["n_unrated"] == 1
    assert f["scratch_rapm_diff"] == pytest.approx(-0.05 - 0.40)


def test_dressed_power_uses_the_dressed_not_the_scratched(feats):
    f = feats[2026020001]
    assert f["dressed_power_diff"] == pytest.approx((17 * 0.05 + 0.30) - 18 * 0.05)
    assert f["d_count_diff"] == 0.0
    assert f["dressed_rated_share"] == pytest.approx(1.0)


def test_goalie_confirmation_flags_the_backup_start(feats):
    f = feats[2026020001]
    assert f["backup_start_diff"] == 1.0, (
        "MTL started a non-usual goalie and BOS its usual one, so the home side "
        "gains: away-minus-home badness")
    assert f["goalie_source"] == "confirmed"


def test_a_two_goalie_comparison_is_not_a_confirmation(feats):
    """The landing endpoint's goalieComparison is a season-stat widget. Reading
    it as a starter would fabricate a confirmation for every scheduled game."""
    f = feats[2026020002]
    assert math.isnan(f["backup_start_diff"])
    assert f["goalie_source"] == "none"


def test_missing_inputs_are_nan_never_zero(feats):
    f = feats[2026020002]
    for name in ("scratch_rapm_diff", "dressed_power_diff", "backup_start_diff"):
        assert math.isnan(f[name]), f"{name} must be NaN when its input is absent"


def test_usual_starter_needs_evidence_before_it_names_anyone():
    u = lf.UsualStarter(min_obs=3)
    assert u.usual("BOS") is None
    u.observe("BOS", 1); u.observe("BOS", 1)
    assert u.usual("BOS") is None, "two starts is not an established starter"
    u.observe("BOS", 2)
    assert u.usual("BOS") == 1
    for _ in range(10):
        u.observe("BOS", 2)
    assert u.usual("BOS") == 2, "the window must roll off stale starters"


def test_endpoint_helper_produces_a_superset_of_the_archived_shape():
    """phase0/nhl_update.archive_lineups fetches `landing` only, and landing
    carries neither scratches nor rosterSpots in any observable game state. The
    helper must pull them from right-rail / play-by-play while still producing
    something the reader above accepts unchanged."""
    sub = lf.lineup_subset_from_endpoints(
        landing={"matchup": {"goalieComparison": {
            "homeTeam": {"leaders": [{"playerId": 900019}]},
            "awayTeam": {"leaders": [{"playerId": 900118}]}}}},
        right_rail={"gameInfo": {
            "homeTeam": {"scratches": [{"id": 900050}], "headCoach": {"default": "A"}},
            "awayTeam": {"scratches": [{"id": 900150}], "headCoach": {"default": "B"}}}},
        play_by_play={"rosterSpots": [
            {"teamId": 6, "playerId": 900001, "positionCode": "C", "sweaterNumber": 1},
            {"teamId": 8, "playerId": 900101, "positionCode": "C", "sweaterNumber": 2}]})
    assert set(sub) == {"rosterSpots", "gameInfoScratches", "headCoach",
                        "goalieComparison"}
    snap = lf.Snapshot(1, "t", "BOS", "MTL", "PRE", sub)
    lf._parse_lineup(snap, sub, {})
    assert {"dressed", "scratches", "coach"} <= snap.present
    assert snap.scratches["homeTeam"] == [900050]


def test_feature_order_declares_a_single_primary():
    """A reopen that shops among seven columns is not the pre-registered test
    row 10 asked for."""
    assert lf.PRIMARY_FEATURE in lf.FEATURE_ORDER
    assert lf.FEATURE_ORDER[0] == lf.PRIMARY_FEATURE
    assert len(set(lf.FEATURE_ORDER)) == len(lf.FEATURE_ORDER)
