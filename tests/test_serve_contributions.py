"""The GLASSBOX contribution breakdown — the site's central honesty claim.

Every card says not just what the model thinks but WHY: a per-feature breakdown
in percentage points. If those numbers do not actually reconstruct the served
probability, the explanation is decoration and the page is lying about how the
forecast was produced.

Iteration 65 pinned the blend COEFFICIENTS. Nothing pinned the arithmetic that
turns them into the published explanation, or which tier a game is served at —
and the tier determines which holdout log-loss legitimately applies to it.

These run against the real frozen model, not a fixture: the property must hold
for the artifact actually being served.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# the five EXACT deltas: each is a difference between two tier probabilities
FEATURES = ("bullpen", "siera", "lineup", "power", "baserun")
# the four LINEARISED attributions of the baseline: slope * rating difference.
# these are explanations of recal_prob, not additive steps on top of it.
BASELINE = ("team", "home_field", "home_pitcher", "away_pitcher")


@pytest.fixture(scope="module")
def pred():
    from mlbwp.serve import Predictor
    try:
        return Predictor()
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"model artifacts unavailable: {exc}")


@pytest.fixture(scope="module")
def teams(pred):
    return sorted(pred.fe.R)


@pytest.fixture(scope="module")
def lineups(pred):
    rated = list(pred.tse.mu)
    if len(rated) < 18:
        pytest.skip("not enough rated players")
    return rated[:9], rated[9:18]


def _reconstruct(r):
    c = r["contributions_pp"]
    return r["recal_prob"] + sum(c[k] for k in FEATURES) / 100


# Rounding budget, derived rather than guessed: each contribution is
# round(x * 100, 1), i.e. ±0.05pp = ±0.0005 in probability units, and there are
# five of them; recal_prob and home_win_prob are each round(_, 4) = ±0.00005.
# 5 * 0.0005 + 2 * 0.00005 = 0.0026.
TOL = 5 * 0.0005 + 2 * 0.00005


# ---- the honesty property --------------------------------------------------

def test_contributions_reconstruct_the_served_probability(pred, teams):
    """recal_prob + the five feature deltas must equal what we publish.

    Tolerance is the rounding budget TOL, computed above from how the values are
    rounded rather than picked to make the test pass. Anything larger means the
    breakdown and the number were derived from different things.
    """
    for h, a in zip(teams[:8], teams[8:16]):
        r = pred.predict(h, a, "", "")
        assert abs(_reconstruct(r) - r["home_win_prob"]) < TOL, (
            f"{h} vs {a}: breakdown does not reconstruct the published probability")


def test_it_holds_at_the_lineup_tier_too(pred, teams, lineups):
    """The lineup tier adds three more terms; the identity must survive them."""
    hl, al = lineups
    r = pred.predict(teams[0], teams[1], "", "", home_lineup=hl, away_lineup=al)
    assert r["tier"] == "lineup"
    assert abs(_reconstruct(r) - r["home_win_prob"]) < 2e-4


def test_the_reconstruction_test_has_teeth(pred, teams):
    """A tolerance test proves nothing if every term it checks is zero.

    Drop the largest real contribution and the identity must break by more than
    the rounding budget — otherwise the test above would pass on a model that
    had silently stopped applying that feature. Measured across all 1056 team
    pairs the true residual peaks at 0.0006 against a 0.0026 budget, so the
    headroom is real rather than a tolerance tuned until green.
    """
    r = pred.predict(teams[0], teams[1], "", "")
    c = r["contributions_pp"]
    biggest = max(FEATURES, key=lambda k: abs(c[k]))
    if abs(c[biggest]) < 0.2:
        pytest.skip("no materially non-zero contribution in this matchup")
    without = r["recal_prob"] + sum(c[k] for k in FEATURES if k != biggest) / 100
    assert abs(without - r["home_win_prob"]) > TOL, (
        f"dropping the {biggest} term ({c[biggest]}pp) does not break the "
        f"identity — the tolerance is too loose to detect a missing feature")


def test_every_contribution_key_is_present_and_numeric(pred, teams):
    """The SPA renders each of these; a missing key blanks a row silently."""
    c = pred.predict(teams[0], teams[1], "", "")["contributions_pp"]
    for k in FEATURES + BASELINE:
        assert k in c, f"contribution {k} missing"
        assert isinstance(c[k], (int, float)), f"{k} is not numeric"


# ---- the tier ladder -------------------------------------------------------

def test_tier_is_lineup_exactly_when_the_lineup_edge_resolved(pred, teams, lineups):
    """The tier is not cosmetic: predict_slate advertises the full-tier holdout
    log-loss as the headline and the recbp one as the fallback, so a card's tier
    is what says which number legitimately applies to it."""
    hl, al = lineups
    without = pred.predict(teams[0], teams[1], "", "")
    assert without["tier"] != "lineup" and without["ts_edge"] is None

    with_lu = pred.predict(teams[0], teams[1], "", "", home_lineup=hl, away_lineup=al)
    assert (with_lu["tier"] == "lineup") == (with_lu["ts_edge"] is not None)


def test_a_one_sided_lineup_does_not_reach_the_lineup_tier(pred, teams, lineups):
    """Both sides are required — a lineup edge computed against a missing
    opponent would be an artefact, not information."""
    hl, _ = lineups
    r = pred.predict(teams[0], teams[1], "", "", home_lineup=hl, away_lineup=None)
    assert r["tier"] != "lineup"


def test_lineup_contributions_are_zero_when_no_lineup_was_used(pred, teams):
    r = pred.predict(teams[0], teams[1], "", "")
    c = r["contributions_pp"]
    assert c["lineup"] == 0.0 and c["power"] == 0.0 and c["baserun"] == 0.0


# ---- basic sanity of the served number ------------------------------------

def test_served_probability_is_a_probability(pred, teams):
    for h, a in zip(teams[:10], teams[10:20]):
        p = pred.predict(h, a, "", "")["home_win_prob"]
        assert 0.0 < p < 1.0, f"{h} vs {a} served {p}"


def test_home_advantage_is_real_and_signed_the_right_way(pred, teams):
    """Same team both sides: the only asymmetry left is home field, so the home
    side must be favoured and the home_field contribution positive."""
    t = teams[0]
    r = pred.predict(t, t, "", "")
    assert r["home_win_prob"] > 0.5
    assert r["contributions_pp"]["home_field"] > 0


def test_a_better_home_team_is_never_less_favoured(pred, teams):
    """Monotonicity in the team rating — the one direction that can never be
    right to violate."""
    h, a = teams[0], teams[1]
    base = pred.predict(h, a, "", "")["home_win_prob"]
    original = pred.fe.R[h]
    try:
        pred.fe.R[h] = original + 100
        better = pred.predict(h, a, "", "")["home_win_prob"]
    finally:
        pred.fe.R[h] = original
    assert better > base, "raising the home rating lowered the home win probability"


def test_an_unknown_team_errors_rather_than_guessing(pred, teams):
    """Silently returning a probability for a team we have no rating for would
    put an invented number on the board."""
    r = pred.predict("ZZZ", teams[0], "", "")
    assert "error" in r and "home_win_prob" not in r
    assert "ZZZ" in r["error"]


# ---- what the breakdown does NOT claim ------------------------------------

def test_the_baseline_terms_are_a_linearisation_not_an_exact_split(pred, teams):
    """Documenting a real distinction the site's single breakdown hides: the
    five feature terms are exact probability deltas, while team/home_field/
    pitcher are slope-linearised attributions of the baseline. They explain
    recal_prob approximately and are NOT additive with the exact five — this
    test exists so nobody later 'fixes' the reconstruction by adding them in.
    """
    r = pred.predict(teams[0], teams[1], "", "")
    c = r["contributions_pp"]
    all_nine = r["recal_prob"] + sum(c[k] for k in FEATURES + BASELINE) / 100
    assert abs(all_nine - r["home_win_prob"]) > 2e-4, (
        "baseline terms now appear additive — if that is intentional, the "
        "reconstruction test above must be updated to match")
