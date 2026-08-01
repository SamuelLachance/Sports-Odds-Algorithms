"""Structural contracts between the serve payloads and the SPA.

The single-page app reads specific fields from site/data/*.json; a serve-script
refactor that drops or renames one silently blanks a league on the live site
(fetches are fault-tolerant by design). These tests pin the exact field sets the
JS consumes — build fails loudly instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1] / "site" / "data"
TIERS = ("CONFIRMED", "PROJECTED", "EARLY")


def _load(name):
    p = SITE / name
    if not p.is_file():
        pytest.skip(f"{name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def test_board_contract():
    b = _load("board.json")
    assert "leagues" in b and isinstance(b["leagues"], list) and b["leagues"]
    assert {"code", "name"} <= set(b["leagues"][0])
    acc = b.get("accuracy")
    assert acc and {"log_loss", "coinflip"} <= set(acc)      # updAcc MLB branch


def test_board_record_contract():
    """#/record's MLB section reads board.record.mlb — the ONLY place a played
    game's pre-game probability survives (the board itself is forward-only)."""
    b = _load("board.json")
    rec = b.get("record")
    assert rec and "mlb" in rec, "board.json lost the record block"
    m = rec["mlb"]
    # recSource/recRows/recByTier/recScheduled/recSchedTotal
    assert {"rows", "rollup", "by_tier", "since", "pending", "scheduled",
            "n_pending", "n_scheduled", "n_lean"} <= set(m)
    assert isinstance(m["rows"], list) and len(m["rows"]) <= 400   # payload stays lean
    for r in m["rows"]:
        assert {"d", "away", "home", "p", "pick", "lean", "y", "hs", "as",
                "tier"} <= set(r)
        assert 0.0 < r["p"] < 1.0 and r["y"] in (0, 1)
        assert r["pick"] == (r["home"] if r["p"] >= 0.5 else r["away"])
        assert r["tier"] in TIERS
        # rule 2: the call travels WITH the graded row, so a lean can never be
        # silently counted as a pick after the fact
        assert r["lean"] == (1 if max(r["p"], 1 - r["p"]) <= 0.55 else 0)
    # policy rule 1: EARLY games are scheduled, never picks
    for r in m["pending"]:
        assert r["tier"] in ("CONFIRMED", "PROJECTED"), "an EARLY game shipped as a pick"
        assert "lean" in r and "spp" in r      # call + rotation-projected-SP flag
    for r in m["scheduled"]:
        assert r["tier"] == "EARLY"
    assert m["n_pending"] >= len(m["pending"])
    assert m["n_scheduled"] >= len(m["scheduled"])
    assert 0 <= m["n_lean"] <= m["n_pending"]
    # policy rule 3: per-tier rollups over the right subsets, each carrying the
    # picks-only score the tier card reports as its headline
    for t, ru in m["by_tier"].items():
        assert t in TIERS and {"n", "log_loss", "acc", "brier", "n_lean",
                               "pick", "lean"} <= set(ru)
        assert 0.0 <= ru["acc"] <= 1.0 and ru["log_loss"] > 0
        assert (ru["pick"]["n"] if ru["pick"] else 0) + ru["n_lean"] == ru["n"]
        assert (ru["lean"]["n"] if ru["lean"] else 0) == ru["n_lean"]
    if m["by_tier"]:
        assert sum(ru["n"] for ru in m["by_tier"].values()) == m["rollup"]["n"]
    if m["rows"]:
        assert m["rows"] == sorted(m["rows"], key=lambda r: r["d"], reverse=True)
        assert m["since"] and len(m["since"]) == 10
        ru = m["rollup"]
        assert ru and {"n", "log_loss", "acc", "brier"} <= set(ru)
        assert ru["n"] >= len(m["rows"])          # rollup covers every graded pick
        assert 0.0 <= ru["acc"] <= 1.0 and ru["log_loss"] > 0


def test_record_page_reads_played_games_from_nfl_and_nhl():
    """NFL/NHL track-record sections are computed client-side from the payload
    schedules: a played game must keep its frozen pre-game prob AND its score."""
    for name, prob in (("nfl.json", "ph"), ("nhl.json", "hp")):
        played = [g for g in _load(name)["schedule"]
                  if g.get("hs") is not None and g.get("as") is not None]
        for g in played:
            assert g.get(prob) is not None, f"{name}: played game without {prob}"
            assert isinstance(g["hs"], int) and isinstance(g["as"], int)


def test_record_block_grades_only_finished_games(tmp_path):
    """record_block() ships graded rows only, newest first, capped — and its
    rollup is scored over every graded row, not just the shipped ones."""
    import math

    from mlbwp.predict_slate import record_block
    ledger = {
        "1": {"d": "2026-07-30", "home": "LAD", "away": "SEA", "p": 0.6,
              "rec": "2026-07-31T00:42Z", "sp_known": True, "tier": "CONFIRMED",
              "y": 1, "hs": 6, "as": 2},
        "2": {"d": "2026-07-31", "home": "ATH", "away": "BOS", "p": 0.4,
              "rec": "2026-08-01T00:42Z", "sp_known": False, "tier": "PROJECTED",
              "y": 0, "hs": 4, "as": 5},
        "3": {"d": "2026-08-01", "home": "SD", "away": "SF", "p": 0.58,
              "tier": "PROJECTED", "rec": "2026-08-01T10:00Z"},  # pending pick
    }
    p = tmp_path / "led.json"
    p.write_text(json.dumps(ledger), encoding="utf-8")
    blk = record_block(p, cap=1)
    assert len(blk["rows"]) == 1 and blk["rows"][0]["d"] == "2026-07-31"   # newest first
    assert blk["rows"][0]["pick"] == "BOS"                # p < 0.5 -> the away side
    assert blk["n_pending"] == 1 and blk["since"] == "2026-07-31"
    ru = blk["rollup"]
    assert ru["n"] == 2 and ru["acc"] == 1.0              # both picks landed
    assert ru["log_loss"] == pytest.approx(-math.log(0.6), abs=1e-5)
    assert record_block(tmp_path / "missing.json") == {
        "rows": [], "rollup": None, "by_tier": {}, "since": None, "pending": [],
        "scheduled": [], "n_pending": 0, "n_scheduled": 0, "n_lean": 0}


def test_record_block_splits_by_tier_and_excludes_early_from_picks(tmp_path):
    """documents/pick_policy.md rules 1 + 3: EARLY games are SCHEDULED, not
    picks, and the graded rollup is computed per tier over the right subsets."""
    import math

    from mlbwp.predict_slate import record_block
    ledger = {
        # graded: two CONFIRMED (one hit, one miss), one EARLY (hit)
        "1": {"d": "2026-07-01", "home": "LAD", "away": "SEA", "p": 0.8,
              "tier": "CONFIRMED", "rec": "2026-07-01T00:00Z", "y": 1, "hs": 5, "as": 1},
        "2": {"d": "2026-07-02", "home": "NYY", "away": "BOS", "p": 0.8,
              "tier": "CONFIRMED", "rec": "2026-07-02T00:00Z", "y": 0, "hs": 1, "as": 5},
        "3": {"d": "2026-07-03", "home": "SD", "away": "SF", "p": 0.7,
              "tier": "EARLY", "rec": "2026-07-03T00:00Z", "y": 1, "hs": 4, "as": 2},
        # pending: one CONFIRMED pick, one PROJECTED lean, two EARLY (scheduled)
        "4": {"d": "2026-08-01", "home": "TB", "away": "TEX", "p": 0.66,
              "tier": "CONFIRMED", "rec": "2026-08-01T00:00Z"},
        "5": {"d": "2026-08-02", "home": "CHC", "away": "MIL", "p": 0.55,
              "tier": "PROJECTED", "rec": "2026-08-01T00:00Z"},
        "6": {"d": "2026-08-20", "home": "ATL", "away": "NYM", "p": 0.53,
              "tier": "EARLY", "rec": "2026-08-01T00:00Z"},
        "7": {"d": "2026-08-25", "home": "HOU", "away": "LAA", "p": 0.61,
              "tier": "EARLY", "rec": "2026-08-01T00:00Z"},
    }
    p = tmp_path / "led.json"
    p.write_text(json.dumps(ledger), encoding="utf-8")
    blk = record_block(p)

    # per-tier rollups over the right subsets (never pooled)
    bt = blk["by_tier"]
    assert set(bt) == {"CONFIRMED", "EARLY"}          # PROJECTED has nothing graded
    assert bt["CONFIRMED"]["n"] == 2 and bt["CONFIRMED"]["acc"] == 0.5
    assert bt["CONFIRMED"]["log_loss"] == pytest.approx(
        (-math.log(0.8) - math.log(0.2)) / 2, abs=1e-5)
    assert bt["EARLY"]["n"] == 1 and bt["EARLY"]["acc"] == 1.0
    assert blk["rollup"]["n"] == 3                    # overall kept for continuity
    assert sum(o["n"] for o in bt.values()) == blk["rollup"]["n"]

    # rule 1: pending == picks only; EARLY lives in `scheduled`
    assert blk["n_pending"] == 2 and blk["n_scheduled"] == 2
    assert {r["home"] for r in blk["pending"]} == {"TB", "CHC"}
    assert {r["home"] for r in blk["scheduled"]} == {"ATL", "HOU"}
    assert all(r["tier"] == "EARLY" for r in blk["scheduled"])
    # rule 2: the 55/45 band -> the CHC forecast at exactly 0.55 is a lean
    assert blk["n_lean"] == 1


def test_lean_threshold_is_exactly_55_45():
    """Policy rule 2: a forecast INSIDE 55/45 is a lean; outside it is a pick.
    The server counts leans and the SPA labels them off the same constant."""
    from mlbwp.predict_slate import LEAN_MAX
    from mlbwp_site.build_site import JS
    assert LEAN_MAX == 0.55
    # ONE source: the JS constant is generated from LEAN_MAX, and the record
    # page's constant aliases the board's rather than re-typing the number.
    # Three hand-typed copies is three chances for the board and the record
    # page to disagree about what counts as a lean.
    assert f"const POL_LEAN_MAX={LEAN_MAX};" in JS
    assert "const LEAN_MAX=POL_LEAN_MAX;" in JS
    assert "const recIsLean=p=>Math.max(p,1-p)<=LEAN_MAX;" in JS
    assert JS.count("LEAN_MAX=0.55") == 1, "the threshold was re-typed somewhere"

    def lean(p):
        return max(p, 1 - p) <= LEAN_MAX
    assert lean(0.50) and lean(0.55) and lean(0.45)          # inside the band
    assert not lean(0.5501) and not lean(0.4499)             # outside -> picks
    assert not lean(0.80) and not lean(0.20)


def test_by_tier_headline_is_scored_over_picks_only(tmp_path):
    """Policy rules 2 + 3 together: inside a tier, a graded LEAN is not a graded
    PICK. The tier's headline score must exclude leans and report them beside."""
    import math

    from mlbwp.predict_slate import record_block
    ledger = {                       # 2 CONFIRMED picks (1 hit), 2 CONFIRMED leans
        "1": {"d": "2026-07-01", "home": "LAD", "away": "SEA", "p": 0.80,
              "tier": "CONFIRMED", "rec": "2026-07-01T00:00Z", "y": 1},
        "2": {"d": "2026-07-02", "home": "NYY", "away": "BOS", "p": 0.80,
              "tier": "CONFIRMED", "rec": "2026-07-02T00:00Z", "y": 0},
        "3": {"d": "2026-07-03", "home": "SD", "away": "SF", "p": 0.52,
              "tier": "CONFIRMED", "rec": "2026-07-03T00:00Z", "y": 1},
        "4": {"d": "2026-07-04", "home": "TB", "away": "TEX", "p": 0.55,
              "tier": "CONFIRMED", "rec": "2026-07-04T00:00Z", "y": 1},
    }
    p = tmp_path / "led.json"
    p.write_text(json.dumps(ledger), encoding="utf-8")
    c = record_block(p)["by_tier"]["CONFIRMED"]
    assert c["n"] == 4 and c["n_lean"] == 2
    assert c["pick"]["n"] == 2 and c["pick"]["acc"] == 0.5
    assert c["pick"]["log_loss"] == pytest.approx(
        (-math.log(0.8) - math.log(0.2)) / 2, abs=1e-5)
    assert c["lean"]["n"] == 2 and c["lean"]["acc"] == 1.0
    # the pooled tier number is NOT the headline: it would read 75% here
    assert c["acc"] == 0.75 and c["pick"]["acc"] != c["acc"]


def test_board_page_applies_rule_1_not_just_the_record_page():
    """The Board is where picks are actually published. It must classify cards
    with the SAME map the ledger uses and refuse to render an EARLY game as a
    pick — otherwise the 420 -> 42 reduction holds only on #/record."""
    from mlbwp_site.build_site import JS
    assert 'const infoTier = g => g.lineup_source==="official"?"CONFIRMED"' in JS
    assert '(g.pitcher_known?"PROJECTED":"EARLY");' in JS
    assert "const callOf = pp => pp<=POL_LEAN_MAX?\"LEAN\":\"PICK\";" in JS
    assert 'const it=infoTier(g), early=it==="EARLY", call=callOf(g.pick_prob);' in JS
    assert '<span class="pill early"' in JS and ">SCHEDULED<" in JS
    # rule 2 stated on the card, and no tradeable-looking fair price on an EARLY one
    assert "${call} &middot; ${g.pick}" in JS
    assert 'const odds=p=>early?"":`<span class="odds"' in JS

    b = _load("board.json")
    lg = next(l for l in b["leagues"] if l.get("active"))
    def tier(g):
        return ("CONFIRMED" if g.get("lineup_source") == "official"
                else "PROJECTED" if g.get("pitcher_known") else "EARLY")
    picks = [g for g in lg["games"] if tier(g) != "EARLY"]
    early = [g for g in lg["games"] if tier(g) == "EARLY"]
    # the board's pick/scheduled split and the ledger's pending/scheduled split
    # The board's pick/scheduled split and the ledger's pending/scheduled split
    # are two views of ONE rule. They are NOT the same population, though: the
    # ledger keeps a game pending until the FINALS file catches up, while the
    # board drops it once played. (Observed 2026-08-01: 36 pending vs 31 eligible
    # cards — the 5 extras were all previous-day games, played and awaiting
    # grading. An earlier version of this test asserted pending <= board picks
    # and failed the moment real games were played; that invariant only holds
    # before first pitch.)
    #
    # The invariant that DOES hold, and is the one rule 1 actually claims: every
    # pending row still resident on the board is non-EARLY, and every ledger row
    # that left the board is in the past.
    import datetime as _dt
    from mlbwp.pred_ledger import entry_tier
    m = b["record"]["mlb"]
    by_pk = {str(g["game_pk"]): g for g in lg["games"]}
    led_path = Path(__file__).resolve().parents[1] / "data" / "mlb_pred_ledger.json"
    if led_path.is_file():
        led = json.loads(led_path.read_text(encoding="utf-8"))
        today = _dt.date.today().isoformat()
        resident = off_board = 0
        for pk, e in led.items():
            if e.get("y") is not None or entry_tier(e) == "EARLY":
                continue
            if pk in by_pk:
                resident += 1
                assert tier(by_pk[pk]) != "EARLY", (
                    f"game {pk} is published as a pending PICK while its board card "
                    "is EARLY — rule 1 is broken")
            else:
                off_board += 1
                assert (e.get("d") or "9999") <= today, (
                    f"game {pk} is pending but absent from the board with a FUTURE "
                    f"date {e.get('d')} — it should still be on the board")
        assert resident + off_board == m["n_pending"]
    assert m["n_scheduled"] <= len(early)
    assert len(picks) < len(lg["games"]), "every card is a pick — rule 1 is not applied"


def test_site_header_never_shows_a_pooled_realized_accuracy():
    """Policy rule 3: the persistent header renders on every page, so a pooled
    CONFIRMED+PROJECTED+EARLY accuracy there is the single worst place for one.
    It stays in the payload for inspection, but nothing renders it."""
    from mlbwp_site.build_site import JS, SHELL
    assert '<span class="acc" id="acc">' in SHELL          # header slot, every page
    for expr in ("realized_2026.log_loss", "realized_2026.acc", "realized_2026.n",
                 "% picks"):
        assert expr not in JS, f"pooled realized accuracy is back in the header: {expr}"
    html = SITE.parent / "index.html"
    if html.is_file():
        body = html.read_text(encoding="utf-8")
        assert "realized_2026.log_loss" not in body and "% picks" not in body


def test_tiers_follow_the_information_column_not_the_league():
    """A tier chip means what the MODEL has, so it means the same thing on every
    league's page. NHL ships no roster/lineup/goalie input -> EARLY. NFL is the
    live model inside a week and a team-strength simulation beyond -> EARLY."""
    from mlbwp_site.build_site import JS
    n = _load("nhl.json")
    feats = " ".join(n["model_card"]["features"]).lower()
    for word in ("roster", "lineup", "goalie"):
        assert word not in feats, f"NHL model gained a {word} input — retier it"
    assert 'y:g.y!=null?g.y:(g.hs>g.as?1:0),hs:g.hs,as:g.as,tier:"EARLY"' in JS
    assert 'sp:1,tier:"EARLY",href:"#/game/nhl-"+g.id' in JS
    assert 'sp:1,tier:nflProb(g).near?"PROJECTED":"EARLY",' in JS


def test_record_block_tier_falls_back_for_pre_stamp_entries(tmp_path):
    """Entries graded before the tier stamp existed are immutable, so readers
    infer their tier from sp_known instead of back-filling the ledger."""
    from mlbwp.predict_slate import record_block
    ledger = {
        "1": {"d": "2026-07-30", "home": "LAD", "away": "SEA", "p": 0.6,
              "sp_known": True, "rec": "2026-07-30T00:00Z", "y": 1, "hs": 5, "as": 1},
        "2": {"d": "2026-07-30", "home": "NYY", "away": "BOS", "p": 0.6,
              "sp_known": False, "rec": "2026-07-30T00:00Z", "y": 1, "hs": 5, "as": 1},
    }
    p = tmp_path / "led.json"
    p.write_text(json.dumps(ledger), encoding="utf-8")
    blk = record_block(p)
    assert {r["home"]: r["tier"] for r in blk["rows"]} == {
        "LAD": "PROJECTED", "NYY": "EARLY"}
    assert set(blk["by_tier"]) == {"PROJECTED", "EARLY"}


def test_nfl_contract():
    n = _load("nfl.json")
    mc = n["model_card"]
    assert {"test_log_loss", "close_log_loss"} <= set(mc)    # updAcc NFL branch
    sched = n["schedule"]
    assert sched, "empty NFL schedule"
    g = sched[0]
    assert {"w", "d", "home", "away", "ph"} <= set(g)        # nflPage/nflProb
    assert n["teams"], "empty NFL teams"
    t = next(iter(n["teams"].values()))
    assert {"elo", "rank"} <= set(t)


def test_nhl_contract():
    n = _load("nhl.json")
    mc = n["model_card"]
    assert {"test_ll", "baseline_elo_test"} <= set(mc)       # updAcc NHL branch
    sched = n["schedule"]
    assert sched, "empty NHL schedule"
    g = sched[0]
    assert {"id", "d", "home", "away", "hp"} <= set(g)       # nhlCard/nhlGamePage
    assert 0.0 < g["hp"] < 1.0
    assert len(n["teams"]) == 32
    t = next(iter(n["teams"].values()))
    assert {"elo", "xg", "pts", "rank", "w", "l", "otl"} <= set(t)  # nhlTeams/standings
    assert n["players"], "empty NHL players"
    p = next(iter(n["players"].values()))
    assert {"name", "pos", "team", "off", "def", "net", "rating", "toi"} <= set(p)


def test_nhl_probs_sane():
    """Model probabilities must be probabilities, and not degenerate."""
    n = _load("nhl.json")
    hps = [g["hp"] for g in n["schedule"]]
    assert all(0.0 < h < 1.0 for h in hps)
    assert max(hps) - min(hps) > 0.15, "prediction spread collapsed"


def test_record_route_is_wired_into_the_spa():
    """A payload block nobody routes to is invisible: pin the nav entry, the
    #/record route and the view function together."""
    from mlbwp_site.build_site import JS, SHELL
    assert 'href="#/record" data-v="record"' in SHELL             # nav entry
    assert 'if(v==="record"){setNav("record");return recordPage();}' in JS
    for fn in ("function recordPage(", "function recSection(", "function recRows(",
               "function recScore(", "function recPeriods(", "function recTierGrid(",
               "function recByTier(", "function recScheduled(", "function recSchedTotal("):
        assert fn in JS, fn
    # EARLY games render collapsed-by-default (no `open` attribute) and labelled
    assert '<details class="sched">' in JS
    assert '<details class="sched" open' not in JS
    assert "Scheduled games &mdash; not picks" in JS
    html = (Path(__file__).resolve().parents[1] / "site" / "index.html")
    if html.is_file():
        body = html.read_text(encoding="utf-8")
        assert "function recordPage(" in body and "#/record" in body


def test_nhl_edges_noop_without_payload(tmp_path):
    """The edge layer must degrade to a clean no-op when inputs are missing —
    never raise inside the CI odds refresh."""
    from market import nhl_edges
    assert nhl_edges.attach_and_save(payload_path=tmp_path / "missing.json",
                                     opening_path=tmp_path / "open.json") == 0


def test_nhl_edges_noop_when_all_played(tmp_path):
    """All games completed (offseason) -> 0 edges, no odds API call needed."""
    from market import nhl_edges
    payload = {"schedule": [{"d": "2026-04-01", "home": "COL", "away": "DAL",
                             "hp": 0.6, "hs": 3, "as": 2}]}
    p = tmp_path / "nhl.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert nhl_edges.attach_and_save(payload_path=p,
                                     opening_path=tmp_path / "open.json") == 0
    out = json.loads(p.read_text(encoding="utf-8"))
    assert "value_updated" in out                            # touched, cleanly


def test_tier_rule_has_a_single_source():
    """The tier rule must exist ONCE. It is stamped server-side by
    pred_ledger.info_tier and rendered client-side by the SPA's infoTier; two
    hand-maintained copies would drift silently, and a drift here is dishonest
    rather than merely buggy — the site would label a card one tier while the
    ledger stamped it another.

    build_site.py therefore substitutes the JS emitted by pred_ledger instead of
    retyping it. This pins that wiring: the rule text must NOT appear literally
    in build_site.py's source, and the built page must contain exactly the
    generated text.
    """
    from pathlib import Path

    from mlbwp.pred_ledger import TIER_JS

    src = (Path(__file__).resolve().parents[1] / "mlbwp_site" / "build_site.py"
           ).read_text(encoding="utf-8")
    rule = 'g.lineup_source==="official"?"CONFIRMED"'
    assert rule not in src, (
        "the tier rule was retyped into build_site.py — it must come from "
        "mlbwp.pred_ledger.TIER_JS so the two copies cannot drift")
    assert "{TIER_JS}" in src and "JS.replace(" in src

    built = Path(__file__).resolve().parents[1] / "site" / "index.html"
    if built.is_file():
        html = built.read_text(encoding="utf-8")
        assert "{TIER_JS}" not in html, "placeholder leaked into the built page"
        assert TIER_JS.split("\n")[0] in html, "generated rule missing from the page"
