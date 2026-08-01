"""Per-game contribution breakdown for the NHL model.

NHL was the only league shipping a bare probability. MLB and NFL both explain
every pick, and the site is called GLASSBOX, so the asymmetry was a real gap.

The NHL model is a single logistic — z = intercept + c_elo*elogit + c_rest*rest
+ c_b2b_home*h + c_b2b_away*a + c_xg*xg — which makes an EXACT decomposition
available, the kind MLB gets and NFL cannot:

    0.5 + (elo + rest + b2b + xg) / 100  ==  the served probability

Each term is the change in probability from switching that group of coefficients
on, in a fixed order, so the deltas telescope and the identity holds by
construction rather than approximately. NFL's `ct` is coefficient * feature *
scale — a logit-space attribution converted by a local slope, which omits the
intercept and the sigmoid's curvature and therefore does NOT reconstruct its
probability (see documents/depth_program.md Round 3b). Same visual, different
guarantee; this one is the stronger of the two.

Order matters for a non-linear decomposition and is fixed deliberately: Elo
first because it is the model's core, then the schedule terms (rest, then
back-to-back), then the expected-goals rating. Any order telescopes correctly;
this one reads as "team strength, then circumstance, then underlying play".
"""
from __future__ import annotations

import math

# the order the deltas are taken in, and the keys the payload/SPA use
KEYS = ("elo", "rest", "b2b", "xg")


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def contributions(intercept: float, coefs: dict, elogit: float, rest_diff: float,
                  b2b_home: float, b2b_away: float, xg_diff: float) -> dict:
    """Percentage-point contributions that sum with 0.5 to the served probability.

    Returns {elo, rest, b2b, xg} rounded to one decimal place, matching how the
    other two leagues present theirs.
    """
    z0 = intercept + coefs["elo_logit"] * elogit
    z1 = z0 + coefs["rest"] * rest_diff
    z2 = z1 + coefs["b2b_home"] * b2b_home + coefs["b2b_away"] * b2b_away
    z3 = z2 + coefs["xg"] * xg_diff

    p0, p1, p2, p3 = _sig(z0), _sig(z1), _sig(z2), _sig(z3)
    return {
        "elo": round((p0 - 0.5) * 100, 1),
        "rest": round((p1 - p0) * 100, 1),
        "b2b": round((p2 - p1) * 100, 1),
        "xg": round((p3 - p2) * 100, 1),
    }


def served_probability(intercept: float, coefs: dict, elogit: float,
                       rest_diff: float, b2b_home: float, b2b_away: float,
                       xg_diff: float) -> float:
    """The model's probability — the same expression nhl_serve.py evaluates.

    Kept here beside `contributions` so the two can never be computed from
    different formulas; a breakdown that explains a number nobody serves is
    exactly the failure the reconstruction test exists to catch.
    """
    return _sig(intercept
                + coefs["elo_logit"] * elogit
                + coefs["rest"] * rest_diff
                + coefs["b2b_home"] * b2b_home
                + coefs["b2b_away"] * b2b_away
                + coefs["xg"] * xg_diff)


# The SPA needs the same key order; generated here so the two cannot drift, the
# way mlbwp/pred_ledger.py emits its tier rule.
CT_JS = 'const NHL_CT=' + str(list(KEYS)).replace("'", '"') + ';'
