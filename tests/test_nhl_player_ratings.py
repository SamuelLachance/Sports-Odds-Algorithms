"""Face-validity gate on the NHL skater ratings in site/data/nhl.json.

These assert on the SERVED payload, not on the fit, because the bug they guard
was only visible after serving. In July 2026 the shipped ratings had:

  * every defenceman crushed — median rating 36.6 for D against 60.5 for F, a
    23.9-point pure positional offset; Makar 34.6, Josi 34.5, Q. Hughes 37.2,
    all in the worst 20% of the league, ZERO defencemen in the top 25;
  * checking-line forwards above every star — Luostarinen 70.0, Jankowski 69.8,
    Kasper 69.7 against McDavid 63.5 and Draisaitl 57.6;
  * a 1,138-minute player (Tsyplakov, 68.2) inside the top 10.

Root causes were a strength leak (13% of the xG credited to "5v5" stints came
from special-teams shots), an unidentified F-vs-D level in the ridge design, no
deployment context, and no sample-size shrinkage. See phase0/nhl_rapm2.py.

Every check degrades to a skip rather than a failure when the payload or a
named player is absent — players retire, change teams, and drop below the
display floor, and that must not brick the suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "site" / "data" / "nhl.json"

# The payload's `toi` field is 5v5 MINUTES (data/nhl_rapm2_ratings.json calls it
# toi_min); nhl_serve.py drops anyone under 1000 of them from the display.
MIN_TOI_TOP10 = 1500.0

ELITE_F = ("McDavid", "MacKinnon", "Matthews", "Kucherov")
ELITE_D = ("Makar", "Josi", "Quinn Hughes")


def _players():
    if not PAYLOAD.is_file():
        pytest.skip("site/data/nhl.json not built")
    try:
        payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pytest.skip("site/data/nhl.json is not readable JSON")
    ps = payload.get("players") or {}
    if len(ps) < 100:
        pytest.skip(f"only {len(ps)} rated skaters served — nothing to rank")
    return ps


def _ranked(ps):
    """Displayed skaters best-first, plus name -> percentile (1.0 = worst)."""
    order = sorted(ps.values(), key=lambda p: -p["rating"])
    pct = {id(p): (i + 1) / len(order) for i, p in enumerate(order)}
    return order, pct


def _find(ps, needle):
    hits = [p for p in ps.values() if needle.lower() in p["name"].lower()]
    return hits[0] if len(hits) == 1 else None


def test_rating_is_monotone_in_net():
    """The 0-100 rating must be an order-preserving transform of net xG/60 — the
    site sorts by one and colours by the other."""
    ps = _players()
    rows = sorted(ps.values(), key=lambda p: p["net"])
    for a, b in zip(rows, rows[1:]):
        if b["net"] > a["net"]:
            assert b["rating"] >= a["rating"] - 1e-9, (
                f"{b['name']} has more net xG/60 than {a['name']} but a lower rating")
    ratings = [p["rating"] for p in ps.values()]
    assert 1.0 <= min(ratings) and max(ratings) <= 99.0, "rating escaped its 1-99 range"
    assert max(ratings) - min(ratings) > 25.0, "rating spread collapsed — scale is degenerate"


def test_elite_forwards_rank_near_the_top():
    ps = _players()
    _, pct = _ranked(ps)
    checked = []
    for name in ELITE_F:
        p = _find(ps, name)
        if p is None:
            continue                      # traded, retired, or below the display floor
        checked.append((name, pct[id(p)]))
        assert pct[id(p)] <= 0.15, (
            f"{p['name']} ({p['pos']}) is at the {pct[id(p)]*100:.1f}th percentile "
            f"of rated skaters — elite forwards belong in the top 15%")
    if not checked:
        pytest.skip("none of the named elite forwards are in the payload")


def test_elite_defencemen_are_not_buried():
    """The anti-regression guard for the positional artefact.

    Threshold is 40% per player rather than the 25% the forwards get, with the
    group median held to 25%: the metric is on-ice xG differential, and a 35-
    year-old on a bottom-five 5v5 team (Josi, 34th percentile) genuinely does
    not post top-quartile on-ice numbers. Before the fix these three sat at the
    89th, 89th and 79th percentiles, so 40% still fails loudly if it returns.
    """
    ps = _players()
    _, pct = _ranked(ps)
    got = []
    for name in ELITE_D:
        p = _find(ps, name)
        if p is None:
            continue
        got.append(pct[id(p)])
        assert pct[id(p)] <= 0.40, (
            f"{p['name']} (D) is at the {pct[id(p)]*100:.1f}th percentile — "
            f"defencemen are being systematically buried again")
    if not got:
        pytest.skip("none of the named elite defencemen are in the payload")
    assert median(got) <= 0.25, (
        f"median elite-D percentile {median(got)*100:.1f}% — the group should sit "
        f"inside the top quartile even if one name lags")


def test_no_positional_offset_between_forwards_and_defencemen():
    """Ridge cannot identify the F-vs-D level (98.6% of 5v5 sides are 3F+2D), so
    nhl_rapm2.py pins it by centring within position. If that centring is lost
    the two medians separate immediately — they were 23.9 rating points apart."""
    ps = _players()
    fwd = [p["rating"] for p in ps.values() if p["pos"] in ("C", "L", "R")]
    dmn = [p["rating"] for p in ps.values() if p["pos"] == "D"]
    if len(fwd) < 30 or len(dmn) < 20:
        pytest.skip("too few of one position to compare medians")
    gap = median(fwd) - median(dmn)
    # ratings carry sd 15 by construction; 5 points is a third of that, ~5x the
    # gap the corrected build produces (1.0) and ~5x below the broken one (23.9)
    assert abs(gap) <= 5.0, (
        f"median forward rating {median(fwd):.1f} vs defenceman {median(dmn):.1f} "
        f"(gap {gap:+.1f}) — a positional offset is back in the ratings")


def test_top_ten_is_not_small_sample_noise():
    ps = _players()
    order, _ = _ranked(ps)
    thin = [(p["name"], p["toi"]) for p in order[:10] if p["toi"] < MIN_TOI_TOP10]
    assert not thin, (
        f"top-10 skaters under {MIN_TOI_TOP10:.0f} 5v5 minutes: {thin} — "
        f"sample-size shrinkage is not holding")
