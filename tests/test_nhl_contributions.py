"""The NHL contribution breakdown — the third league's answer to the same
honesty question, and the strongest of the three.

NHL was the only league shipping a bare probability while MLB and NFL both
explained every pick. Because the NHL model is a single logistic, it gets an
EXACT decomposition:

    0.5 + (elo + rest + b2b + xg) / 100  ==  the served probability

Stronger than NFL's, which is coefficient * feature * scale — a logit-space
attribution that omits the intercept and the sigmoid's curvature and therefore
does NOT reconstruct its own number (documents/depth_program.md Round 3b). Same
visual on the site, different guarantee, so the guarantee is pinned here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_contributions as C  # noqa: E402

NHL_JSON = ROOT / "site" / "data" / "nhl.json"

# Rounding budget, derived not guessed: four terms at round(x*100, 1) are
# ±0.0005 each in probability units, and hp is round(_, 4) = ±0.00005.
TOL = 4 * 0.0005 + 0.00005

COEFS = {"elo_logit": 0.70016, "rest": 0.05, "b2b_home": -0.08,
         "b2b_away": 0.09, "xg": 0.37079}


def test_the_identity_holds_on_the_raw_maths():
    """No rounding involved — the decomposition telescopes by construction."""
    kw = dict(intercept=0.12, coefs=COEFS, elogit=0.4, rest_diff=1.0,
              b2b_home=1.0, b2b_away=0.0, xg_diff=0.3)
    ct = C.contributions(**kw)
    served = C.served_probability(**kw)
    assert abs(0.5 + sum(ct.values()) / 100 - served) < TOL


def test_it_holds_across_a_wide_sweep():
    """Every term pushed in both directions, including the tails where the
    sigmoid is most non-linear — that is where a naive additive breakdown breaks
    and this one must not."""
    worst = 0.0
    for elogit in (-2.5, -0.7, 0.0, 0.7, 2.5):
        for rest in (-3.0, 0.0, 3.0):
            for hb, ab in ((0, 0), (1, 0), (0, 1)):
                for xg in (-1.2, 0.0, 1.2):
                    kw = dict(intercept=0.12, coefs=COEFS, elogit=elogit,
                              rest_diff=rest, b2b_home=hb, b2b_away=ab,
                              xg_diff=xg)
                    ct = C.contributions(**kw)
                    served = C.served_probability(**kw)
                    worst = max(worst, abs(0.5 + sum(ct.values()) / 100 - served))
    assert worst < TOL, f"worst residual {worst:.6f} exceeds the rounding budget"


def test_a_dead_feature_contributes_exactly_nothing():
    kw = dict(intercept=0.0, coefs=COEFS, elogit=0.5, rest_diff=0.0,
              b2b_home=0.0, b2b_away=0.0, xg_diff=0.0)
    ct = C.contributions(**kw)
    assert ct["rest"] == 0.0 and ct["b2b"] == 0.0 and ct["xg"] == 0.0
    assert ct["elo"] != 0.0, "a non-zero Elo edge must still show"


def test_signs_follow_the_coefficients():
    base = dict(intercept=0.0, coefs=COEFS, elogit=0.0, rest_diff=0.0,
                b2b_home=0.0, b2b_away=0.0, xg_diff=0.0)
    assert C.contributions(**{**base, "rest_diff": 2.0})["rest"] > 0, \
        "more rest for the home side must help it"
    assert C.contributions(**{**base, "b2b_home": 1.0})["b2b"] < 0, \
        "the home team on a back-to-back must be penalised"
    assert C.contributions(**{**base, "b2b_away": 1.0})["b2b"] > 0, \
        "the AWAY team on a back-to-back must help the home side"
    assert C.contributions(**{**base, "xg_diff": 1.0})["xg"] > 0


def test_the_breakdown_has_teeth():
    """A tolerance test proves nothing if the terms are all zero. Drop the
    largest and the identity must break by more than the budget."""
    kw = dict(intercept=0.12, coefs=COEFS, elogit=0.9, rest_diff=2.0,
              b2b_home=1.0, b2b_away=0.0, xg_diff=0.8)
    ct = C.contributions(**kw)
    served = C.served_probability(**kw)
    biggest = max(ct, key=lambda k: abs(ct[k]))
    without = 0.5 + sum(v for k, v in ct.items() if k != biggest) / 100
    assert abs(without - served) > TOL


def test_the_key_order_is_shared_with_the_spa():
    """The SPA renders NHL_CT in this order; generating it from here is what
    stops the two drifting, the same way pred_ledger emits its tier rule."""
    assert C.KEYS == ("elo", "rest", "b2b", "xg")
    assert C.CT_JS == 'const NHL_CT=["elo", "rest", "b2b", "xg"];'


# ---- against the shipped payload -------------------------------------------

def test_every_shipped_game_reconstructs():
    if not NHL_JSON.is_file():
        pytest.skip("nhl.json not built")
    sched = json.loads(NHL_JSON.read_text(encoding="utf-8"))["schedule"]
    assert sched, "empty NHL schedule"
    worst, missing = 0.0, 0
    for g in sched:
        ct = g.get("ct")
        if not ct:
            missing += 1
            continue
        assert set(ct) == set(C.KEYS), f"unexpected keys {sorted(ct)}"
        worst = max(worst, abs(0.5 + sum(ct.values()) / 100 - g["hp"]))
    assert missing == 0, f"{missing} shipped games carry no breakdown"
    assert worst < TOL, f"worst shipped residual {worst:.6f}"


def test_the_spa_guards_its_placeholder():
    """A missed interpolation would ship a page where NHL_CT is undefined and
    every NHL game deep-dive throws. build_site asserts on it; pin that."""
    src = (ROOT / "mlbwp_site" / "build_site.py").read_text(encoding="utf-8")
    assert 'assert "{CT_JS}" in JS' in src
    assert '.replace("{CT_JS}", CT_JS)' in src
