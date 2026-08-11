"""The EV badge may never appear on a forecast made before the information exists.

documents/pick_policy.md rule 1 governs picks; an EV badge is a STRONGER claim
than a pick — it asserts the market is mispriced by the EV threshold — so it must clear at
least the same information bar. An EARLY-tier card has no named starter (team
ratings only), and a "value bet" on a game whose pitchers are unknown is a claim
we cannot support.

Today the exposure is latent: the odds feed carries nothing beyond ~2 days out,
so every quoted game is PROJECTED or CONFIRMED. These tests make the gate
structural instead of dependent on a provider's posting window.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from market import edges

# The fixture game must be in the FUTURE. edges.py drops any card whose
# start_utc has passed (it must never badge a game already under way), so a
# hardcoded date turns these tests into a time bomb: they were written on
# 2026-07-31 with "2026-08-01" meaning tomorrow, and began failing on 2026-08-02
# — three green tests that quietly stopped exercising the eligibility path they
# exist to cover. Derive it from the clock instead.
FUTURE = (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat()


def _card(pk, date, **kw):
    c = {"game_pk": pk, "date": date, "home": "BOS", "away": "NYA",
         "home_abbr": "BOS", "away_abbr": "NYA",
         "start_utc": f"{date}T23:10:00Z", "state": "Preview",
         "home_win_prob": 0.75}          # far above any plausible market price
    c.update(kw)
    return c


def _board(cards):
    return {"leagues": [{"code": "mlb", "active": True, "games": cards}]}


def _live(date):
    # a price so long that a 0.75 model prob clears +20% EV comfortably
    return [{"id": "g1", "date": date, "home": "BOS", "away": "NYA",
             "home_dec": 2.10, "away_dec": 1.80, "commence": f"{date}T23:10:00Z",
             "n_books": 8}]


def _run(tmp_path, card, monkeypatch, date=FUTURE):
    monkeypatch.setattr(edges.odds, "fetch_consensus", lambda key=None: _live(date))
    bp = tmp_path / "board.json"
    bp.write_text(json.dumps(_board([card])), encoding="utf-8")
    op = tmp_path / "open.json"
    edges.attach_and_save(board_path=bp, opening_path=op, api_key="x")
    out = json.loads(bp.read_text(encoding="utf-8"))
    return out["leagues"][0]["games"][0]


def test_early_tier_never_gets_a_badge(tmp_path, monkeypatch):
    """No named starter -> no EV badge, however large the modelled edge."""
    g = _run(tmp_path, _card(1, FUTURE, pitcher_known=False,
                             lineup_source=None), monkeypatch)
    assert "value" not in g
    assert "mkt" not in g          # not even quoted: the card is gated out entirely


def test_projected_tier_is_eligible(tmp_path, monkeypatch):
    """A named (announced or rotation-projected) starter clears the bar."""
    g = _run(tmp_path, _card(2, FUTURE, pitcher_known=True,
                             sp_projected=True, lineup_source="projected"),
             monkeypatch)
    assert g.get("mkt"), "a quoted, eligible card must carry the consensus quote"


def test_confirmed_tier_is_eligible(tmp_path, monkeypatch):
    g = _run(tmp_path, _card(3, FUTURE, pitcher_known=True,
                             lineup_source="official"), monkeypatch)
    assert g.get("mkt")


def test_official_lineups_without_pitcher_flag_still_eligible(tmp_path, monkeypatch):
    """lineup_source=='official' implies the starters are known even if the
    pitcher_known flag is missing — the gate must not reject on a missing flag."""
    g = _run(tmp_path, _card(4, FUTURE, lineup_source="official"), monkeypatch)
    assert g.get("mkt")


def test_ev_threshold_has_a_single_source():
    """The badge threshold must exist ONCE.

    It was previously re-typed in edges.py, nfl_edges.py and nhl_edges.py with
    no test at all, so the three leagues could have silently disagreed while the
    site and the logs claimed one figure. Same drift class as the tier rule and
    the lean threshold, but with money attached: the badge asserts the market is
    mispriced, so the leagues disagreeing about WHEN it fires is a correctness
    bug the reader cannot see.
    """
    import re
    from pathlib import Path

    from market import EV_THRESHOLD, edges, nfl_edges, nhl_edges

    # 0.08: where CLV first covers the opening vig — break-even, not profit
    # (documents/bet_logic_verdict_2026_08_10.md; >=20% was the worst
    # winner's-curse cell and is no longer tracked).
    assert edges.EV_THRESHOLD == nfl_edges.EV_THRESHOLD == nhl_edges.EV_THRESHOLD \
        == EV_THRESHOLD == 0.08

    root = Path(__file__).resolve().parents[1]
    # exactly one assignment across the whole market package
    assigns = []
    for f in sorted((root / "market").glob("*.py")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*EV_THRESHOLD\s*=", ln):
                assigns.append(f"{f.name}: {ln.strip()}")
    assert len(assigns) == 1, f"threshold assigned in more than one place: {assigns}"
    assert assigns[0].startswith("__init__.py"), assigns[0]

    # and the operator copy must be derived, never hand-typed — for the CURRENT
    # value and any stale copy of the previous one
    for name in ("refresh.py", "refresh_odds.py"):
        src = (root / name).read_text(encoding="utf-8")
        for pct in (round(EV_THRESHOLD * 100), 20):
            assert f">={pct}%" not in src, (
                f"{name} hard-codes a threshold in its log copy; use EV_PCT so "
                "the message cannot outlive the constant")


def test_the_fixture_game_is_actually_in_the_future():
    """The guard against this file rotting again.

    Three tests here assert that an ELIGIBLE card receives a quote. If the
    fixture date ever slips into the past, edges.py drops the card before the
    eligibility logic runs and all three pass-or-fail on the wrong reason —
    which is exactly how they broke. Pin the precondition explicitly so the
    failure names itself.
    """
    start = datetime.fromisoformat(f"{FUTURE}T23:10:00+00:00")
    assert start > datetime.now(timezone.utc), (
        f"fixture game {FUTURE} is not in the future; the eligibility tests "
        f"would be exercising edges.py's started-game guard instead")
