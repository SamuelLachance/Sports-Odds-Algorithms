"""The MLB frozen model — the pin NHL had and MLB did not.

NHL has `test_frozen_model_pin` and NFL re-derives its TEST number at serve time
and asserts on it (`nfl_season_serve.py`: reproduce 0.61947 or stop). MLB had
neither, and it is the league serving live right now.

What that gap allows: `board.accuracy.log_loss` — the score the site publishes
as its headline claim — is read straight from `blend["holdout"]["full_ll"]`.
That value is a STORED LABEL, not something derived from the coefficients beside
it. Editing a coefficient without updating the label leaves the site advertising
a score the served model no longer earns, and nothing anywhere would object.

These tests pin both halves together, so the two cannot drift apart silently.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "mlbwp" / "artifacts"


def _blend():
    return json.loads((ART / "blend.json").read_text(encoding="utf-8"))


def _metrics():
    return json.loads((ART / "metrics.json").read_text(encoding="utf-8"))


def test_shipped_holdout_numbers_are_frozen():
    """The exact scores the site publishes. A deliberate re-freeze updates these
    together with the coefficients below; an accidental one fails here."""
    h = _blend()["holdout"]
    assert h["full_ll"] == 0.67274, "the headline TEST log-loss moved"
    assert h["recbp_ll"] == 0.67411
    assert h["recal_ll"] == 0.67477
    assert h["n"] == 19435, "holdout size changed — the split is supposed to be locked"


def test_richer_tiers_score_better_in_order():
    """A semantic invariant, not a snapshot: each tier adds information, so each
    must beat the one below it and all must beat the plain-Elo baseline. A refit
    that inverted this would be a bug even if every number looked plausible."""
    h, m = _blend()["holdout"], _metrics()
    assert h["full_ll"] < h["recbp_ll"] < h["recal_ll"] < m["model_log_loss"], (
        "tier ordering broken (expected full < recbp < recal < raw FIP-Elo)")
    assert m["model_log_loss"] < m["plain_elo_log_loss"], "core lost to plain Elo"
    assert m["plain_elo_log_loss"] < m["constant_log_loss"], "Elo lost to a coin flip"


def test_serving_coefficients_are_frozen():
    """The coefficients that actually produce every probability on the board.

    Pinned to 10 decimals: a genuine refit moves them far more than that, and
    this is the only thing standing between an accidental artifact edit and a
    site that serves different numbers under the old advertised score.
    """
    full = _blend()["full"]
    assert set(full) == {"b0", "b1", "b2", "b3", "b4", "b5", "b6"}
    expected = {
        "b0": 0.03979815771880227, "b1": 0.7557205250769196,
        "b2": 0.036768029577731516, "b3": 0.04107199473275712,
        "b4": 0.07000176346690114, "b5": -0.09608600992407829,
        "b6": 0.07722413742902438,
    }
    for k, v in expected.items():
        assert abs(full[k] - v) < 1e-10, f"serving coefficient {k} changed"


def test_every_fallback_tier_is_present_and_complete():
    """The board drops to a lower tier when lineups or a starter are missing, so
    a tier that lost its coefficients would fail at serve time on exactly the
    games with the least information — the ones already hardest to get right."""
    b = _blend()
    assert set(b["recal"]) == {"b0", "b1"}
    assert set(b["recbp"]) == {"b0", "b1", "b3", "b5"}
    for name in ("ts", "bp", "pw", "si", "br"):
        assert f"{name}_mu" in b and f"{name}_sd" in b, f"{name} normalisation missing"
        assert b[f"{name}_sd"] > 0, f"{name}_sd must be positive — it is a divisor"


def test_the_site_publishes_the_artifact_number_not_a_copy():
    """board.accuracy.log_loss must equal the artifact, not a literal that was
    right when someone typed it."""
    board_path = ROOT / "site" / "data" / "board.json"
    if not board_path.is_file():
        import pytest
        pytest.skip("board not built")
    acc = json.loads(board_path.read_text(encoding="utf-8"))["accuracy"]
    h, m = _blend()["holdout"], _metrics()
    assert acc["log_loss"] == h["full_ll"]
    assert acc["fallback_log_loss"] == h["recbp_ll"]
    assert acc["elo_log_loss"] == m["plain_elo_log_loss"]
    assert acc["coinflip"] == m["constant_log_loss"]


def test_holdout_note_still_records_the_validation_protocol():
    """The note is the audit trail for what these numbers mean: fit 2000-2015,
    scored 2016-2024, coefficients retrained on all seasons for serving. Losing
    it would leave the scores unfalsifiable."""
    note = _blend()["holdout"]["note"]
    assert "2000-2015" in note and "2016-2024" in note
    assert "retrained" in note
