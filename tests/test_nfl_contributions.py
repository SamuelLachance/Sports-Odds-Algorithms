"""The NFL contribution breakdown — the same honesty question as MLB, and a
different answer, which is worth writing down.

Iteration 73 pinned that MLB's breakdown RECONSTRUCTS the served probability:
recal_prob plus the five feature deltas equals what is published. The NFL
breakdown does not, and cannot: `ct` is built as coefficient * feature * scale
(nfl_season_serve.py), i.e. a logit-space attribution converted to points by a
per-game local slope. It omits the intercept and the sigmoid's curvature, so
`ph - sum(ct)` is not constant — measured across the 272-game 2026 payload it
ranges 0.436 to 0.679.

That is not a defect. It is the same construction as MLB's team/home_field/
pitcher terms, which iteration 73 documented as linearised and non-additive.
The two leagues' breakdowns simply mean different things behind the same
visual: MLB's five feature terms are exact probability deltas, every NFL term
is a linearisation.

So the property to guard here is FAITHFULNESS rather than additivity: the
contribution total must order games the same way the model does. Measured
Spearman is 0.9999 — if the breakdown ever decoupled from the forecast (stale
coefficients, a feature-order change, a scale bug), that collapses.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NFL_JSON = ROOT / "site" / "data" / "nfl.json"

# the nine groups nfl_season_serve.py emits (features are pooled into these)
CT_KEYS = {"elo", "qb", "units", "sched", "hfa", "luck", "ts", "abs", "roster"}


def _games():
    if not NFL_JSON.is_file():
        pytest.skip("nfl.json not built")
    sched = json.loads(NFL_JSON.read_text(encoding="utf-8"))["schedule"]
    games = [g for g in sched if g.get("ct") and g.get("ph") is not None]
    if len(games) < 30:
        pytest.skip("too few NFL games with contributions")
    return games


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def test_contributions_order_games_the_way_the_model_does():
    """The faithfulness property. A breakdown that no longer tracks the forecast
    is decoration, and the site presents it as the reason for the pick.

    Threshold 0.999, not 1.0: `ct` values are rounded to one decimal place of a
    percentage point, so near-tied games can swap. Measured on the shipped
    payload the correlation is 0.9999, so the bar has room without being
    vacuous — a decoupled breakdown would fall far below it.
    """
    games = _games()
    totals = [sum(g["ct"].values()) for g in games]
    probs = [g["ph"] for g in games]
    rho = _spearman(totals, probs)
    assert rho > 0.999, (
        f"contribution totals no longer track the served probability (rho={rho:.5f})")


def test_dropping_any_live_factor_breaks_the_bar():
    """Proves 0.999 discriminates rather than passing on anything vaguely
    monotone. Measured: removing elo drops rho to 0.971, roster to 0.929, qb to
    0.968; flipping elo's sign gives 0.832; an unrelated pairing gives -0.05.
    Every realistic corruption lands far below the threshold.
    """
    games = _games()
    probs = [g["ph"] for g in games]
    mean_abs = {k: sum(abs(g["ct"][k]) for g in games) / len(games) for k in CT_KEYS}
    live = [k for k, v in mean_abs.items() if v > 0.5]
    assert live, "no factor is large enough to test sensitivity against"
    for k in live:
        without = [sum(v for j, v in g["ct"].items() if j != k) for g in games]
        assert _spearman(without, probs) < 0.999, (
            f"dropping '{k}' leaves the correlation above the bar — the "
            f"threshold cannot detect a missing factor")


def test_an_unrelated_pairing_is_caught():
    games = _games()
    totals = [sum(g["ct"].values()) for g in games]
    probs = [g["ph"] for g in games]
    shuffled = probs[len(probs) // 2:] + probs[: len(probs) // 2]
    assert _spearman(totals, shuffled) < 0.9


def test_every_game_carries_the_full_key_set():
    """The SPA renders one row per key; a missing key drops a factor from the
    explanation silently."""
    for g in _games():
        assert set(g["ct"]) == CT_KEYS, f"{g['away']}@{g['home']} week {g['w']}"
        for k, v in g["ct"].items():
            assert isinstance(v, (int, float)), f"{k} is not numeric"


def test_the_breakdown_is_not_claimed_to_be_additive():
    """Documents the real difference from MLB so nobody 'fixes' it by asserting
    a reconstruction that cannot hold. If this ever starts passing as additive,
    the construction changed and this module's premise needs revisiting."""
    games = _games()
    residuals = [g["ph"] - sum(g["ct"].values()) / 100 for g in games]
    spread = max(residuals) - min(residuals)
    assert spread > 0.01, (
        "ph - sum(ct) has become near-constant — the NFL breakdown may now be "
        "additive, in which case pin the reconstruction as MLB does")


def test_published_probabilities_are_probabilities():
    for g in _games():
        assert 0.0 < g["ph"] < 1.0
        if g.get("pmc") is not None:
            assert 0.0 < g["pmc"] < 1.0


def test_the_breakdown_is_not_degenerate():
    """No single factor may account for essentially the whole explanation, and
    several must be materially live. A scale or coefficient error typically
    shows up as one term swamping the rest.

    Deliberately NOT asserting which term leads. My first version required elo
    to dominate; measured on the shipped payload `roster` does (4.41 mean pp vs
    elo's 3.18), and that is correct on both counts — the NFL player ratings
    beat team Elo on their own, and this is a PRE-SEASON payload where every
    team's Elo is regressed toward the mean while roster signal is fully
    expressed. Pinning a ranking would have encoded my guess, then broken every
    time the season's information balance shifted.
    """
    games = _games()
    mean_abs = {k: sum(abs(g["ct"][k]) for g in games) / len(games) for k in CT_KEYS}
    total = sum(mean_abs.values())
    assert total > 0, "every contribution is zero"
    assert max(mean_abs.values()) / total < 0.75, (
        f"one factor swamps the breakdown: {mean_abs}")
    live = [k for k, v in mean_abs.items() if v > 0.1]
    assert len(live) >= 4, f"only {len(live)} factors are materially live: {mean_abs}"


def test_contributions_and_probability_agree_on_direction():
    """A game the model likes must not have a net-negative explanation."""
    strong = [g for g in _games() if g["ph"] > 0.60]
    if not strong:
        pytest.skip("no strong home favourites in this payload")
    wrong = [g for g in strong if sum(g["ct"].values()) <= 0]
    assert not wrong, (
        f"{len(wrong)} home favourites have a net-negative breakdown, e.g. "
        f"{wrong[0]['away']}@{wrong[0]['home']}")


def test_the_site_does_not_claim_the_nfl_bars_add_up():
    """The caption is a factual claim and it was wrong.

    It read "one feature group's push on the home win probability vs a 50/50
    game", which invites the reader to add the bars to 50% and expect the
    number shown. Measured on the shipped payload, only 11 of 272 games land
    within 1pp of that identity and errors reach -0.179 — an 18-point miss.

    NHL's breakdown IS exact and its caption says so (iteration 84). Building
    the accurate one next to this is what exposed the overclaim, so this pins
    the distinction: the NFL panel must not assert additivity, and must say the
    bars combine through a logistic.
    """
    src = (ROOT / "mlbwp_site" / "build_site.py").read_text(encoding="utf-8")
    nfl_panel = src.split("Why &mdash; blend contributions", 1)[1].split("</div></div>", 1)[0]
    assert "vs a 50/50 game" not in nfl_panel, "the caption implies additivity again"
    assert "not</b> add up" in nfl_panel or "not add up" in nfl_panel
    assert "logistic" in nfl_panel


def test_only_the_exact_league_advertises_exactness():
    """NHL may claim its bars sum; NFL may not. If NFL's construction ever
    changes to an exact one, move the claim deliberately rather than by
    copy-paste."""
    src = (ROOT / "mlbwp_site" / "build_site.py").read_text(encoding="utf-8")
    nhl_panel = src.split("Why &mdash; model contributions", 1)[1].split("</div></div>", 1)[0]
    assert "exact" in nhl_panel.lower()
    nfl_panel = src.split("Why &mdash; blend contributions", 1)[1].split("</div></div>", 1)[0]
    assert "exact" not in nfl_panel.lower()
