"""Per-game contribution breakdown for the NHL model.

NHL was the only league shipping a bare probability. MLB and NFL both explain
every pick, and the site is called GLASSBOX, so the asymmetry was a real gap.

The NHL model is a single logistic — z = intercept + c_elo*elogit + c_rest*rest
+ c_b2b_home*h + c_b2b_away*a + c_xg*xg — which makes an EXACT decomposition
available, the kind MLB gets and NFL cannot:

    0.5 + (home + elo + rest + b2b + xg) / 100  ==  the served probability

Each term is the change in probability from switching that group of coefficients
on, in a fixed order, so the deltas telescope and the identity holds by
construction rather than approximately. NFL's `ct` is coefficient * feature *
scale — a logit-space attribution converted by a local slope, which omits the
intercept and the sigmoid's curvature and therefore does NOT reconstruct its
probability (see documents/depth_program.md Round 3b). Same visual, different
guarantee; this one is the stronger of the two.

Home ice is its own term (it used to sit inside the Elo bar). Two of the
model's inputs carry a home edge - the Elo logit includes the +ha Elo points
(elo_cfg.ha) and the xG difference includes the +XG_HA goals - and the fitted
intercept applies to every game from the home side's point of view. `home` is
all three together: the model's probability for the home side between two
EQUAL teams on equal rest, minus 50%. `elo` and `xg` are then the rating
differences alone. The caller says how much of `elogit` / `xg_diff` is home
edge (`home_elogit`, `home_xg`); with both left at 0, `home` is the intercept
alone and the Elo/xG bars keep their home edge.

Order matters for a non-linear decomposition and is fixed deliberately: home
ice first (the baseline every game starts from), then Elo, the model's core,
then the schedule terms (rest, then back-to-back), then the expected-goals
rating. Any order telescopes correctly; this one reads as "where the game
starts, team strength, then circumstance, then underlying play".

Rounding: each CUMULATIVE probability is rounded to 0.1 pt and a term is the
difference of two rounded cumulatives, so the bars add up exactly to the
rounded served probability (the residual is the one final rounding, never five
of them stacked).
"""
from __future__ import annotations

import math

# the order the deltas are taken in, and the keys the payload/SPA use
KEYS = ("home", "elo", "rest", "b2b", "xg")


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def home_elogit(ha: float) -> float:
    """The Elo logit of +ha Elo points: the home edge inside an Elo
    probability 1/(1+10^(-(r_home+ha-r_away)/400)), whose logit is exactly
    (r_home+ha-r_away)*ln(10)/400."""
    return ha * math.log(10.0) / 400.0


def contributions(intercept: float, coefs: dict, elogit: float, rest_diff: float,
                  b2b_home: float, b2b_away: float, xg_diff: float,
                  home_elogit: float = 0.0, home_xg: float = 0.0) -> dict:
    """Percentage-point contributions that sum with 0.5 to the served probability.

    `home_elogit` / `home_xg` are the parts of `elogit` / `xg_diff` that are
    home edge (see the module docstring). Returns {home, elo, rest, b2b, xg},
    each at one decimal place, matching how the other two leagues present
    theirs; they sum exactly to the served probability rounded to 0.1 pt.
    """
    ce, cx = coefs["elo_logit"], coefs["xg"]
    zh = intercept + ce * home_elogit + cx * home_xg
    z0 = zh + ce * (elogit - home_elogit)
    z1 = z0 + coefs["rest"] * rest_diff
    z2 = z1 + coefs["b2b_home"] * b2b_home + coefs["b2b_away"] * b2b_away
    z3 = z2 + cx * (xg_diff - home_xg)

    cum = [50.0] + [round(_sig(z) * 100, 1) for z in (zh, z0, z1, z2, z3)]
    return {k: round(cum[i + 1] - cum[i], 1) for i, k in enumerate(KEYS)}


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
