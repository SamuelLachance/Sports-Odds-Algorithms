"""NFL/NHL edge layer: freshness guard, start-time guard, disclosure, and the edge
ledger's settlement from the committed results files with evidence-only voids.

Audit 2026-09-24. The NFL payload sat at its 2026-07-30 build for eight weeks while
odds.yml badged it (up to +106% EV) and the ledger recorded 21 bets off it; six
played week-1 bets were shown as "voided on postponed/cancelled games" because the
payload had not been re-served; NHL rows were censored out of the CLV gate because
the ledger had no puck-drop time. Fixtures only: no network, no API key, no clock
dependence (every call passes an explicit `now`), and no real payload is touched.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from market import edge_ledger as EL
from market import nfl_edges, nhl_edges, odds, sched_edges

UTC = timezone.utc
NOW = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)        # Thursday, 12:00 ET
FRESH = "2026-09-29T14:00:00Z"                        # Tuesday build, ~2 days old


# ------------------------------------------------------------------ fixtures

def _nfl(d, t, away, home, ph=0.70, hs=None, as_=None, **kw):
    g = {"w": 4, "d": d, "t": t, "home": home, "away": away, "neutral": 0,
         "ph": ph, "pmc": ph, "hs": hs, "as": as_}
    g.update(kw)
    return g


def _nhl(gid, start, away, home, hp=0.70, hs=None, as_=None, **kw):
    g = {"id": gid, "d": start[:10], "home": home, "away": away, "hp": hp,
         "hs": hs, "as": as_, "y": None, "ot": None, "playoff": 0,
         "start_utc": start, "tier": "EARLY"}
    g.update(kw)
    return g


def _quote(g, home_dec=1.80, away_dec=2.10):
    """One consensus game from the (faked) odds feed. At p_home 0.70 a 1.80 home
    price is +26% EV — far over the threshold."""
    return {"id": f"ev-{g['d']}-{g['home']}", "commence": "x", "date": g["d"],
            "home": g["home"], "away": g["away"], "home_dec": home_dec,
            "away_dec": away_dec, "home_best": home_dec, "away_best": away_dec,
            "n_books": 8}


def _feed(monkeypatch, quotes):
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return list(quotes)
    monkeypatch.setattr(odds, "fetch_consensus", fake)
    return calls


def _run(tmp_path, monkeypatch, mod, payload, quotes, now=NOW):
    p = tmp_path / "payload.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    calls = _feed(monkeypatch, quotes)
    n = mod.attach_and_save(payload_path=p, opening_path=tmp_path / "open.json",
                            api_key="unused", now=now)
    return n, json.loads(p.read_text(encoding="utf-8")), calls


def _by(out, home):
    return next(g for g in out["schedule"] if g["home"] == home)


# ------------------------------------------------------------------ edge layer: NFL

def test_fresh_nfl_build_badges_and_discloses(tmp_path, monkeypatch):
    g = _nfl("2026-10-04", "13:00", "KC", "MIA")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": FRESH, "schedule": [g]}, [_quote(g)])
    assert n == 1
    v = _by(out, "MIA")["value"]
    assert v["team"] == "MIA" and v["available"] is True
    assert v["tier"] == "PROJECTED" and v["model_ts"] == FRESH
    assert "build" in v["note"]                      # the disclosed deficit rides along
    st = out["value_status"]
    assert st["state"] == "live" and st["n_badges"] == 1 and st["n_blocked"] == 0
    assert st["model_ts"] == FRESH and "value_blocked" not in out


def test_stale_nfl_build_holds_every_badge_but_keeps_the_price_feed(tmp_path, monkeypatch):
    """The July-30 incident: a build 11 days old may not price a badge. `mkt` still
    flows — the ledger's closing line for bets already recorded depends on it."""
    g = _nfl("2026-10-04", "13:00", "KC", "MIA")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": "2026-09-20T12:00:00Z", "schedule": [g]}, [_quote(g)])
    assert n == 0
    row = _by(out, "MIA")
    assert "value" not in row and row["mkt"]["home_dec"] == 1.80
    assert "days ago" in row["value_blocked"]
    assert out["value_status"]["state"] == "suppressed"
    assert out["value_blocked"] == out["value_status"]["reason"]
    assert out["value_status"]["n_blocked"] == 1


def test_unstamped_payload_gets_no_badge(tmp_path, monkeypatch):
    g = _nfl("2026-10-04", "13:00", "KC", "MIA")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges, {"schedule": [g]}, [_quote(g)])
    assert n == 0 and "timestamp" in out["value_blocked"]
    assert _by(out, "MIA")["mkt"]                     # quoted all the same


def test_served_at_wins_over_generated(tmp_path, monkeypatch):
    g = _nfl("2026-10-04", "13:00", "KC", "MIA")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": "2026-07-30T23:57:40Z", "served_at": FRESH,
                      "schedule": [g]}, [_quote(g)])
    assert n == 1 and out["value_status"]["model_ts"] == FRESH


def test_forecast_made_more_than_7_days_out_is_early_not_a_badge(tmp_path, monkeypatch):
    """Pick policy: NFL is PROJECTED only inside 7 days of kickoff. The number a
    badge prices must have been PRODUCED inside that window, not merely be shown in
    it: a Sunday build forecasting the Monday eight days later is EARLY."""
    near = _nfl("2026-10-01", "20:15", "PIT", "CLE")
    far = _nfl("2026-10-05", "20:15", "NYJ", "BUF")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": "2026-09-27T12:00:00Z", "schedule": [near, far]},
                     [_quote(near), _quote(far)])
    assert n == 1
    assert _by(out, "CLE")["value"]["tier"] == "PROJECTED"
    assert "8 days before the game" in _by(out, "BUF")["value_blocked"]
    assert out["value_status"]["state"] == "live"


def test_unabsorbed_result_blocks_only_that_teams_next_game(tmp_path, monkeypatch):
    """Monday night was played but the payload has no score: CHI's next forecast has
    not absorbed it. A GLOBAL rule would also kill KC@MIA; the rule is per team."""
    mnf = _nfl("2026-09-28", "20:15", "PHI", "CHI")             # kicked off, unscored
    chi = _nfl("2026-10-04", "13:00", "CHI", "DET")
    mia = _nfl("2026-10-04", "13:00", "KC", "MIA")
    done = _nfl("2026-09-27", "13:00", "ARI", "SF", hs=20, as_=17)
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": FRESH, "schedule": [done, mnf, chi, mia]},
                     [_quote(chi), _quote(mia)])
    assert n == 1 and _by(out, "MIA")["value"]
    assert _by(out, "DET")["value_blocked"] == "CHI's 2026-09-28 result is not in the model yet"
    assert _by(out, "DET")["mkt"]


def test_a_week_old_cancelled_game_does_not_freeze_a_season(tmp_path, monkeypatch):
    """nflverse keeps a cancelled game unscored forever; past the look-back it no
    longer blocks (the build-age rule covers a genuinely stale payload)."""
    cancelled = _nfl("2026-09-06", "13:00", "BUF", "CIN")
    nxt = _nfl("2026-10-04", "13:00", "BUF", "NE")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": FRESH, "schedule": [cancelled, nxt]}, [_quote(nxt)])
    assert n == 1 and _by(out, "NE")["value"]


def test_stale_fields_from_an_earlier_run_are_cleared(tmp_path, monkeypatch):
    g = _nfl("2026-10-04", "13:00", "KC", "MIA", value={"team": "KC"},
             value_blocked="old", mkt={"home_dec": 9})
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": FRESH, "value_blocked": "old", "schedule": [g]}, [])
    row = _by(out, "MIA")
    assert n == 0 and not ({"value", "value_blocked", "mkt"} & set(row))
    assert "value_blocked" not in out and out["value_status"]["n_quoted"] == 0


def test_no_window_games_is_idle_and_never_fetches(tmp_path, monkeypatch):
    g = _nfl("2026-09-27", "13:00", "ARI", "SF", hs=20, as_=17)
    n, out, calls = _run(tmp_path, monkeypatch, nfl_edges,
                         {"generated": "2026-07-30T23:57:40Z", "schedule": [g]}, [])
    assert n == 0 and calls == [] and out["value_status"]["state"] == "idle"
    assert "value_blocked" not in out and "value_updated" in out


def test_the_edge_layer_never_moves_a_model_probability(tmp_path, monkeypatch):
    """Market-blind + played games keep their pre-game number: ph/pmc/hs untouched."""
    sched = [_nfl("2026-09-27", "13:00", "ARI", "SF", ph=0.611, hs=20, as_=17),
             _nfl("2026-10-04", "13:00", "KC", "MIA", ph=0.683)]
    before = [(g["ph"], g["pmc"], g["hs"]) for g in sched]
    _, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"generated": FRESH, "schedule": sched}, [_quote(sched[1])])
    assert [(g["ph"], g["pmc"], g["hs"]) for g in out["schedule"]] == before


# ------------------------------------------------------------------ edge layer: NHL

def test_nhl_started_game_is_never_quoted_or_badged(tmp_path, monkeypatch):
    """start_utc is served now: a game past puck drop is out of the window even if a
    feed were to return it (fetch_consensus drops them; this is the second lock)."""
    live = _nhl(1, (NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), "FLA", "CAR")
    later = _nhl(2, (NOW + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), "MTL", "TOR",
                 tier="PROJECTED")              # a goalie-confirmed row: badge-eligible
    n, out, _ = _run(tmp_path, monkeypatch, nhl_edges,
                     {"served_at": "2026-10-01T12:00:00Z", "schedule": [live, later]},
                     [_quote(live), _quote(later)])
    assert n == 1
    assert not ({"mkt", "value"} & set(_by(out, "CAR")))
    assert _by(out, "TOR")["value"]["team"] == "TOR"


def test_nhl_early_forecast_is_quoted_but_never_badged(tmp_path, monkeypatch):
    """Rule 5 (GATE_EARLY): every shipped NHL forecast is EARLY — team ratings only,
    no goalie input — and an EARLY forecast is not a pick, so it carries no badge.
    The price still flows (`mkt`, the ledger's close) and the row says why."""
    assert sched_edges.GATE_EARLY is True
    g = _nhl(3, "2026-10-02T23:00:00Z", "NYI", "BOS")
    n, out, _ = _run(tmp_path, monkeypatch, nhl_edges,
                     {"served_at": "2026-10-01T12:00:00Z", "schedule": [g]}, [_quote(g)])
    row = _by(out, "BOS")
    assert n == 0 and "value" not in row and row["mkt"]["home_dec"] == 1.80
    assert row["value_blocked"].startswith("EARLY tier") and "goalie" in row["value_blocked"]
    st = out["value_status"]
    assert (st["state"], st["n_quoted"], st["n_blocked"], st["n_early"], st["n_badges"]) \
        == ("live", 1, 1, 1, 0)
    assert st["gate_early"] is True


def test_gate_off_restores_the_disclosed_early_badge(tmp_path, monkeypatch):
    """The owner can flip GATE_EARLY back to the 2026-07-31 disclosure policy. The
    badge then names its goalie deficit and no longer calls itself 'not a pick' —
    the ledger stakes it in units."""
    monkeypatch.setattr(sched_edges, "GATE_EARLY", False)
    g = _nhl(3, "2026-10-02T23:00:00Z", "NYI", "BOS")
    n, out, _ = _run(tmp_path, monkeypatch, nhl_edges,
                     {"served_at": "2026-10-01T12:00:00Z", "schedule": [g]}, [_quote(g)])
    v = _by(out, "BOS")["value"]
    assert n == 1 and v["tier"] == "EARLY" and "goalie" in v["note"].lower()
    assert "not a pick" not in v["note"].lower()
    assert out["value_status"]["gate_early"] is False


def test_nfl_row_the_serve_stamped_early_is_not_badged(tmp_path, monkeypatch):
    """A row the NFL serve stamped EARLY (nfl_ph_freeze.tier_for) is held by rule 5
    even when the build is fresh and the game sits inside the window."""
    g = _nfl("2026-10-04", "13:00", "KC", "MIA", tier="EARLY")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"served_at": FRESH, "schedule": [g]}, [_quote(g)])
    assert n == 0 and _by(out, "MIA")["value_blocked"].startswith("EARLY tier")
    assert out["value_status"]["n_early"] == 1


# ------------------------------------------------------------------ rule 4: absorbed

SUN_BUILD = "2026-10-04T14:00:00Z"                   # Sunday 10:00 ET, before kickoffs


def test_a_final_attached_after_the_build_is_not_in_the_model(tmp_path, monkeypatch):
    """phase0/nfl_results_attach.py (refresh.py, every 4 h) scores the payload WITHOUT
    re-running the model or moving served_at. DAL's Sunday final, attached after a
    Sunday-morning build, is not in the Thursday forecast: that badge must wait."""
    sun = _nfl("2026-10-04", "13:00", "DAL", "NYG", hs=17, as_=24)     # attached later
    thu = _nfl("2026-10-08", "20:15", "DAL", "PHI")
    other = _nfl("2026-10-08", "20:15", "KC", "MIA")
    n, out, _ = _run(tmp_path, monkeypatch, nfl_edges,
                     {"served_at": SUN_BUILD, "schedule": [sun, thu, other]},
                     [_quote(thu), _quote(other)],
                     now=datetime(2026, 10, 5, 16, 0, tzinfo=UTC))
    assert n == 1 and _by(out, "MIA")["value"]
    assert _by(out, "PHI")["value_blocked"] == "DAL's 2026-10-04 result is not in the model yet"


def test_ledger_refuses_a_badge_whose_team_result_was_attached_after_the_build(tmp_path):
    """The same hole on the ledger side: a payload carrying a badge for DAL's Thursday
    game (old edge layer, re-serve carry-forward) is not recorded."""
    lp = tmp_path / "ledger.json"
    sun = _nfl("2026-10-04", "13:00", "DAL", "NYG", hs=17, as_=24)
    thu = _nfl("2026-10-08", "20:15", "DAL", "PHI", value=_badge())
    p = _write(tmp_path, "nfl.json", {"served_at": SUN_BUILD, "schedule": [sun, thu]})
    st = EL.update("nfl", p, lp, now=datetime(2026, 10, 5, 16, 0, tzinfo=UTC))
    assert (st["recorded"], st["blocked"]) == (0, 1) and _rows(lp) == {}


def test_a_game_in_progress_at_build_time_is_not_absorbed():
    """Built 1 h after kickoff: the final came later, whoever wrote the score."""
    built = sched_edges.parse_ts("2026-10-04T18:00:00Z")
    g = _nfl("2026-10-04", "13:00", "DAL", "NYG", hs=17, as_=24)      # 17:00Z kickoff
    assert sched_edges.absorbed(g, built) is False
    assert sched_edges.absorbed(g, built + timedelta(hours=3)) is True  # final by then
    assert sched_edges.absorbed({**g, "hs": None}, built + timedelta(days=2)) is False


def test_the_payloads_walked_through_date_overrides_a_score():
    """A final dated after the model's own walked-through date (NFL power_asof, NHL
    ratings_through) is not in the model, whatever the build time says."""
    built = sched_edges.parse_ts("2026-10-06T14:00:00Z")
    g = _nfl("2026-10-05", "20:15", "NYJ", "BUF", hs=20, as_=10)
    assert sched_edges.absorbed(g, built) is True
    thr = sched_edges.through_date({"power_asof": "2026-10-04"})
    assert sched_edges.absorbed(g, built, thr) is False
    sched = [g, _nfl("2026-10-11", "13:00", "BUF", "MIA")]
    now = datetime(2026, 10, 7, 12, tzinfo=UTC)
    assert sched_edges.unabsorbed(sched, now, built, thr) == {"BUF": "2026-10-05",
                                                              "NYJ": "2026-10-05"}
    assert sched_edges.unabsorbed(sched, now, built, None) == {}
    # never the attach-moved fields
    assert sched_edges.through_date({"results_through": {"d": "2026-10-04"}}) is None


def test_nhl_reserve_after_the_final_absorbs_it():
    """NHL re-serves every 4 h: a serve stamped after the final has walked it."""
    g = _nhl(4, "2026-10-01T23:00:00Z", "BOS", "NYR", hs=3, as_=2)
    assert sched_edges.absorbed(g, sched_edges.parse_ts("2026-10-02T04:00:00Z")) is True
    assert sched_edges.absorbed(g, sched_edges.parse_ts("2026-10-02T01:00:00Z")) is False


def test_nhl_build_older_than_36h_holds_badges(tmp_path, monkeypatch):
    g = _nhl(3, "2026-10-02T23:00:00Z", "NYI", "BOS")
    n, out, _ = _run(tmp_path, monkeypatch, nhl_edges,
                     {"served_at": "2026-09-29T23:00:00Z", "schedule": [g]}, [_quote(g)])
    assert n == 0 and out["value_status"]["state"] == "suppressed"
    assert _by(out, "BOS")["mkt"]


def test_nhl_back_to_back_waits_for_last_nights_result(tmp_path, monkeypatch):
    last = _nhl(4, "2026-10-01T00:00:00Z", "BOS", "NYR")        # played, not ingested
    b2b = _nhl(5, "2026-10-01T23:00:00Z", "BOS", "PHI")
    n, out, _ = _run(tmp_path, monkeypatch, nhl_edges,
                     {"served_at": "2026-09-30T20:00:00Z", "schedule": [last, b2b]},
                     [_quote(b2b)])
    assert n == 0 and "BOS's 2026-10-01 result" in _by(out, "PHI")["value_blocked"]


def test_nhl_docstring_no_longer_claims_a_20pct_gate():
    assert "20%" not in (nhl_edges.__doc__ or "")
    assert "20%" not in (nfl_edges.__doc__ or "")


# ------------------------------------------------------------------ ledger: fixtures

NFL_HDR = "game_id,season,game_type,week,gameday,gametime,away_team,away_score,home_team,home_score\n"
NHL_HDR = ("game_id,date,season,type,away,home,away_goals,home_goals,home_win,"
           "last_period,neutral,win_goalie\n")


def _csv(tmp_path, name, header, lines):
    p = tmp_path / name
    p.write_text(header + "".join(ln + "\n" for ln in lines), encoding="utf-8")
    return p


def _badge(side="away", cur=3.2):
    return {"side": side, "team": None, "ev_open": 0.25, "ev_cur": 0.05,
            "open_dec": 3.3, "cur_dec": cur, "available": True, "books": 9}


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _rows(lp):
    return json.loads(lp.read_text(encoding="utf-8"))["rows"]


def _record_nfl(tmp_path, d="2026-09-13", t="13:00", away="NE", home="SEA",
                generated="2026-09-08T14:00:00Z", when=None):
    lp = tmp_path / "ledger.json"
    g = _nfl(d, t, away, home, ph=0.611, value=_badge())
    p = _write(tmp_path, "nfl.json", {"generated": generated, "schedule": [g]})
    when = when or datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    st = EL.update("nfl", p, lp, now=when)
    return lp, p, g, st


# ------------------------------------------------------------------ ledger: settlement

def test_nfl_settles_from_the_committed_results_file(tmp_path):
    """0 of 12 played NFL bets were settled on 2026-09-24 because only the payload
    was read; data/nfl_games.csv already had every final."""
    lp, p, g, st = _record_nfl(tmp_path)
    assert st["recorded"] == 1
    row = _rows(lp)["nfl|2026-09-13|NE@SEA"]
    assert row["model_ts"] == "2026-09-08T14:00:00Z" and row["tier"] == "PROJECTED"
    g.pop("value")                                    # payload never re-served: no hs
    _write(tmp_path, "nfl.json", {"generated": "2026-09-08T14:00:00Z", "schedule": [g]})
    res = _csv(tmp_path, "nfl_games.csv", NFL_HDR,
               ["2026_02_NE_SEA,2026,REG,2,2026-09-13,13:00,NE,24,SEA,17"])
    later = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
    # a caller's own payload never pulls in the repo's results silently
    assert EL.update("nfl", p, lp, now=later)["settled"] == 0
    st = EL.update("nfl", p, lp, now=later, league_results_path=res)
    assert st["settled"] == 1
    row = _rows(lp)["nfl|2026-09-13|NE@SEA"]
    assert row["y"] == 1 and row["roi"] == pytest.approx(2.2)   # away NE won at 3.2


def test_nhl_settles_from_the_committed_results_file_by_game_id(tmp_path):
    lp = tmp_path / "ledger.json"
    g = _nhl(2026020123, "2026-11-02T00:00:00Z", "DAL", "COL", hp=0.55,
             value=_badge("home", 2.2), tier="PROJECTED")
    p = _write(tmp_path, "nhl.json", {"served_at": "2026-11-01T12:00:00Z", "schedule": [g]})
    assert EL.update("nhl", p, lp, now=datetime(2026, 11, 1, 20, tzinfo=UTC))["recorded"] == 1
    row = _rows(lp)["nhl|2026020123"]
    assert row["tier"] == "PROJECTED" and row["model_ts"] == "2026-11-01T12:00:00Z"
    g.pop("value")
    _write(tmp_path, "nhl.json", {"served_at": "2026-11-01T12:00:00Z", "schedule": [g]})
    res = _csv(tmp_path, "nhl_games.csv", NHL_HDR,
               ["2026020123,2026-11-01,20262027,2,DAL,COL,1,4,1,REG,0,1"])
    st = EL.update("nhl", p, lp, now=datetime(2026, 11, 2, 12, tzinfo=UTC),
                   league_results_path=res)
    row = _rows(lp)["nhl|2026020123"]
    assert st["settled"] == 1 and row["y"] == 1 and row["roi"] == pytest.approx(1.2)


def test_nfl_tie_is_a_push_not_a_loss(tmp_path):
    lp, p, g, _ = _record_nfl(tmp_path)
    g.pop("value")
    g["hs"], g["as"] = 20, 20
    _write(tmp_path, "nfl.json", {"generated": "2026-09-08T14:00:00Z", "schedule": [g]})
    EL.update("nfl", p, lp, now=datetime(2026, 9, 14, 12, tzinfo=UTC))
    row = _rows(lp)["nfl|2026-09-13|NE@SEA"]
    assert row["y"] is None and "tie" in row["void_reason"]
    blk = EL.site_block(lp, now=datetime(2026, 9, 14, 12, tzinfo=UTC))
    assert blk["leagues"]["nfl"]["n_void"] == 1 and blk["pending"] == []
    assert blk["leagues"]["nfl"]["settled"] is None      # never staked
    # the reason ships, so the page never calls a push "postponed/cancelled"
    assert len(blk["voids"]) == 1 and "tie" in blk["voids"][0]["reason"]


# ------------------------------------------------------------------ ledger: void vs overdue

def test_unsettled_is_overdue_not_void_without_evidence(tmp_path):
    """The mislabel: six played week-1 bets were shown as 'voided on postponed/
    cancelled games' only because their results had not been read in."""
    lp, p, g, _ = _record_nfl(tmp_path)
    g.pop("value")
    _write(tmp_path, "nfl.json", {"generated": "2026-09-08T14:00:00Z", "schedule": [g]})
    # the results file was last fetched BEFORE the game: silence, not evidence
    res = _csv(tmp_path, "nfl_games.csv", NFL_HDR,
               ["2026_01_KC_BAL,2026,REG,1,2026-09-10,20:20,KC,20,BAL,23",
                "2026_02_NE_SEA,2026,REG,2,2026-09-13,13:00,NE,,SEA,"])
    now = datetime(2026, 9, 24, 12, tzinfo=UTC)
    EL.update("nfl", p, lp, now=now, league_results_path=res)
    row = _rows(lp)["nfl|2026-09-13|NE@SEA"]
    assert row["y"] is None and "void_reason" not in row
    blk = EL.site_block(lp, now=now)
    lg = blk["leagues"]["nfl"]
    assert (lg["n_void"], lg["n_open"], lg["n_awaiting"], lg["n_overdue"]) == (0, 0, 1, 1)
    assert [b["status"] for b in blk["pending"]] == ["overdue"]
    rep = EL.report(lp, now=now + timedelta(days=7))["leagues"]["nfl"]
    assert rep["n_void"] == 0 and rep["n_stale_unsettled"] == 1   # 'late', not void


def test_game_gone_from_the_league_schedule_is_void(tmp_path):
    lp, p, g, _ = _record_nfl(tmp_path)
    g.pop("value")
    _write(tmp_path, "nfl.json", {"generated": "2026-09-08T14:00:00Z", "schedule": []})
    res = _csv(tmp_path, "nfl_games.csv", NFL_HDR,
               ["2026_01_KC_BAL,2026,REG,1,2026-09-10,20:20,KC,20,BAL,23",
                "2026_02_NE_SEA,2026,REG,2,2026-09-15,20:15,NE,,SEA,",   # moved to Tue
                "2026_05_NE_BUF,2026,REG,5,2026-10-04,13:00,NE,,BUF,"])
    now = datetime(2026, 9, 14, 12, tzinfo=UTC)
    EL.update("nfl", p, lp, now=now, league_results_path=res)
    row = _rows(lp)["nfl|2026-09-13|NE@SEA"]
    assert "not on the league schedule" in row["void_reason"]
    blk = EL.site_block(lp, now=now)
    assert blk["leagues"]["nfl"]["n_void"] == 1 and blk["pending"] == []


def test_nhl_void_needs_results_read_past_the_date(tmp_path):
    lp = tmp_path / "ledger.json"
    g = _nhl(99, "2026-10-10T23:00:00Z", "DAL", "COL", value=_badge("home", 2.2),
             tier="PROJECTED")
    p = _write(tmp_path, "nhl.json", {"served_at": "2026-10-10T12:00:00Z", "schedule": [g]})
    EL.update("nhl", p, lp, now=datetime(2026, 10, 10, 18, tzinfo=UTC))
    _write(tmp_path, "nhl.json", {"served_at": "2026-10-10T12:00:00Z", "schedule": []})
    thin = _csv(tmp_path, "thin.csv", NHL_HDR,
                ["98,2026-10-11,20262027,2,BOS,NYR,2,3,1,REG,0,1"])
    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    EL.update("nhl", p, lp, now=now, league_results_path=thin)
    assert "void_reason" not in _rows(lp)["nhl|99"]          # results end at d+1
    assert EL.site_block(lp, now=now)["leagues"]["nhl"]["n_overdue"] == 1
    wide = _csv(tmp_path, "wide.csv", NHL_HDR,
                ["98,2026-10-11,20262027,2,BOS,NYR,2,3,1,REG,0,1",
                 "97,2026-10-19,20262027,2,BOS,NYR,2,3,1,REG,0,1"])
    EL.update("nhl", p, lp, now=now, league_results_path=wide)
    assert "postponed or cancelled" in _rows(lp)["nhl|99"]["void_reason"]
    assert EL.site_block(lp, now=now)["leagues"]["nhl"]["n_void"] == 1


def test_a_rescheduled_game_back_on_the_slate_is_not_void(tmp_path):
    lp = tmp_path / "ledger.json"
    g = _nhl(77, "2026-10-10T23:00:00Z", "DAL", "COL", value=_badge("home", 2.2),
             tier="PROJECTED")
    p = _write(tmp_path, "nhl.json", {"served_at": "2026-10-10T12:00:00Z", "schedule": [g]})
    EL.update("nhl", p, lp, now=datetime(2026, 10, 10, 18, tzinfo=UTC))
    obj = json.loads(lp.read_text(encoding="utf-8"))
    obj["rows"]["nhl|77"]["void_reason"] = "stale evidence"
    lp.write_text(json.dumps(obj), encoding="utf-8")
    g2 = _nhl(77, "2026-10-25T23:00:00Z", "DAL", "COL")          # made-up date
    _write(tmp_path, "nhl.json", {"served_at": "2026-10-20T12:00:00Z", "schedule": [g2]})
    EL.update("nhl", p, lp, now=datetime(2026, 10, 20, 18, tzinfo=UTC))
    assert "void_reason" not in _rows(lp)["nhl|77"]


# ------------------------------------------------------------------ ledger: freshness

def test_ledger_refuses_a_badge_priced_from_a_stale_build(tmp_path):
    """Whatever wrote the payload (a re-serve carrying old badges forward, an older
    edge layer), the ledger does not accrue a row from a stale build."""
    lp, _, _, st = _record_nfl(tmp_path, generated="2026-07-30T23:57:40Z")
    assert st["recorded"] == 0 and st["blocked"] == 1 and _rows(lp) == {}


def test_ledger_refuses_a_badge_for_a_team_with_an_unabsorbed_result(tmp_path):
    lp = tmp_path / "ledger.json"
    prev = _nfl("2026-09-10", "20:20", "SEA", "KC")              # played, unscored
    g = _nfl("2026-09-13", "13:00", "NE", "SEA", value=_badge())
    p = _write(tmp_path, "nfl.json", {"generated": "2026-09-08T14:00:00Z",
                                      "schedule": [prev, g]})
    st = EL.update("nfl", p, lp, now=datetime(2026, 9, 13, 12, tzinfo=UTC))
    assert st["recorded"] == 0 and st["blocked"] == 1


def test_nhl_row_is_not_censored_after_puck_drop(tmp_path):
    """Without start_utc an NHL row stayed 'pregame' until its result landed; every
    cycle after puck drop found no price (the feed drops started games) while the
    feed was live for other games, so the row was censored out of the CLV gate."""
    lp = tmp_path / "ledger.json"
    ours = _nhl(10, "2026-10-10T23:00:00Z", "DAL", "COL", value=_badge("home", 2.2),
                mkt={"home_dec": 2.2, "away_dec": 1.7}, tier="PROJECTED")
    other = _nhl(11, "2026-10-11T23:00:00Z", "BOS", "NYR",
                 mkt={"home_dec": 1.9, "away_dec": 1.95})
    p = _write(tmp_path, "nhl.json", {"served_at": "2026-10-10T12:00:00Z",
                                      "schedule": [ours, other]})
    EL.update("nhl", p, lp, now=datetime(2026, 10, 10, 18, tzinfo=UTC))
    assert _rows(lp)["nhl|10"]["start_utc"] == "2026-10-10T23:00:00Z"
    ours.pop("value"); ours.pop("mkt")                           # started: no quote
    _write(tmp_path, "nhl.json", {"served_at": "2026-10-10T12:00:00Z",
                                  "schedule": [ours, other]})
    st = EL.update("nhl", p, lp, now=datetime(2026, 10, 11, 0, 30, tzinfo=UTC))
    row = _rows(lp)["nhl|10"]
    assert st["closed"] == 1 and row["n_missed"] == 0 and row["clv_censored"] is False


def test_legacy_rows_from_the_july_build_are_flagged_and_out_of_the_gate(tmp_path):
    """The 21 NFL rows recorded 09-06..09-24 were priced from the 2026-07-30 build.
    They cannot be corrected (first record wins) — they are flagged: kept in the
    units (the badge was published), excluded from the CLV promotion gate."""
    lp = tmp_path / "ledger.json"
    base = {"league": "nfl", "home": "LAC", "away": "ARI", "side": "away",
            "team": "ARI", "p_model": 0.239, "dec_at_record": 5.15,
            "imp_at_record": 1 / 5.15, "clv_pts": 0.01, "n_obs": 52,
            "clv_censored": False, "dec_close": 4.8}
    rows = {
        "nfl|2026-09-13|ARI@LAC": {**base, "key": "nfl|2026-09-13|ARI@LAC",
                                   "d": "2026-09-13", "ts_record": "2026-09-06T05:41:30Z",
                                   "y": 1, "roi": 4.15},
        "nfl|2026-10-11|ARI@SEA": {**base, "key": "nfl|2026-10-11|ARI@SEA",
                                   "home": "SEA", "d": "2026-10-11",
                                   "ts_record": "2026-10-06T05:00:00Z",
                                   "model_ts": "2026-10-06T02:00:00Z",
                                   "y": 0, "roi": -1.0},
    }
    lp.write_text(json.dumps({"v": 1, "updated": "x", "rows": rows}), encoding="utf-8")
    assert "37 days after" in EL.stale_model_reason(rows["nfl|2026-09-13|ARI@LAC"])
    assert EL.stale_model_reason(rows["nfl|2026-10-11|ARI@SEA"]) is None
    rep = EL.report(lp, now=datetime(2026, 10, 12, tzinfo=UTC))["leagues"]["nfl"]
    assert rep["n_stale_model"] == 1 and rep["n_clv_graded"] == 1
    assert rep["units_staked"] == 2.0                     # both still in the units
    blk = EL.site_block(lp, now=datetime(2026, 10, 12, tzinfo=UTC))["leagues"]["nfl"]
    assert blk["stale_model"] == {"n": 1, "n_settled": 1, "net": 4.15,
                                  "builds": ["2026-07-30"]}
    assert blk["settled"]["n"] == 2


def test_nhl_legacy_rows_are_attributed_to_the_weekly_serve():
    r = {"league": "nhl", "d": "2026-09-29", "ts_record": "2026-09-22T04:38:38Z"}
    assert EL._row_model_ts(r) == sched_edges.parse_ts("2026-09-21T16:12:17Z")
    assert "8 days before the game" in EL.stale_model_reason(r)
    assert EL.stale_model_reason({**r, "league": "mlb"}) is None     # MLB gated upstream


def test_site_block_ships_each_leagues_badge_policy(tmp_path):
    lp = tmp_path / "ledger.json"
    lp.write_text(json.dumps({"v": 1, "rows": {}}), encoding="utf-8")
    blk = EL.site_block(lp, now=NOW)["leagues"]
    assert blk["nhl"]["policy"]["tier"] == "EARLY"
    assert blk["nhl"]["policy"]["gate"] == "enforced"          # GATE_EARLY on
    assert "goalie" in blk["nhl"]["policy"]["deficit"]
    assert blk["nfl"]["policy"]["gate"] == "enforced"
    assert blk["mlb"]["policy"]["gate"] == "enforced"
    assert all(blk[lg]["by_tier"] == {} for lg in ("mlb", "nfl", "nhl"))


def test_ledger_never_records_an_nhl_early_badge(tmp_path):
    """Second lock: a stamped, fresh NHL payload still carrying an EARLY badge (the
    NHL serve carries `value` forward across re-serves) is not staked."""
    lp = tmp_path / "ledger.json"
    g = _nhl(12, "2026-10-02T23:00:00Z", "NYI", "BOS", value=_badge("home", 2.2))
    p = _write(tmp_path, "nhl.json", {"served_at": "2026-10-01T12:00:00Z", "schedule": [g]})
    st = EL.update("nhl", p, lp, now=NOW)
    assert (st["recorded"], st["blocked"]) == (0, 1) and _rows(lp) == {}


# ------------------------------------------------------------------ ledger: tiers

def _legacy_nfl(key_d="2026-10-01", home="CLE", away="PIT", **kw):
    """A row as the pre-stamp ledger holds it: no tier, no model_ts, priced from the
    2026-07-30 build (LEGACY_MODEL_BUILDS)."""
    r = {"league": "nfl", "key": f"nfl|{key_d}|{away}@{home}", "d": key_d,
         "home": home, "away": away, "side": "away", "team": away,
         "p_model": 0.4, "p_home": 0.6, "dec_at_record": 2.9, "imp_at_record": 1 / 2.9,
         "ev_at_record": 0.16, "ts_record": "2026-09-24T05:05:14Z",
         "start_utc": "2026-10-02T00:15:00Z", "n_obs": 3, "clv_censored": False,
         "dec_close": None, "clv_pts": None, "y": None, "roi": None}
    r.update(kw)
    return r


def test_legacy_nfl_bets_priced_two_months_out_are_early_not_projected(tmp_path):
    """The 9 open legacy NFL bets were priced from the July build 56-63 days before
    kickoff: EARLY by the pick policy's tier line, whatever the league default."""
    r = _legacy_nfl()
    assert "tier" not in r and EL.row_tier(r) == "EARLY"
    lp = tmp_path / "ledger.json"
    lp.write_text(json.dumps({"v": 1, "rows": {r["key"]: r}}), encoding="utf-8")
    blk = EL.site_block(lp, now=datetime(2026, 9, 24, 18, tzinfo=UTC))
    assert [(p["tier"], p["stale_model"]) for p in blk["pending"]] == [("EARLY", True)]
    # a row whose own stamp says PROJECTED but whose build was 8+ days out: EARLY
    assert EL.row_tier({**r, "tier": "PROJECTED"}) == "EARLY"
    # stamped inside the window: the row's tier stands
    fresh = {**r, "tier": "PROJECTED", "model_ts": "2026-09-29T14:00:00Z"}
    assert EL.row_tier(fresh) == "PROJECTED"
    assert EL.row_tier({"league": "nhl", "d": "2026-10-02",
                        "model_ts": "2026-10-01T12:00:00Z"}) == "EARLY"
    assert EL.row_tier({"league": "mlb", "tier": "lineup"}) is None
    assert EL.row_tier({"league": "mlb", "tier": "CONFIRMED"}) == "CONFIRMED"


def test_forecast_tier_needs_evidence_for_anything_above_early(tmp_path):
    built = sched_edges.parse_ts("2026-09-29T14:00:00Z")
    g = _nfl("2026-10-04", "13:00", "KC", "MIA")
    assert sched_edges.forecast_tier(g, built, "PROJECTED") == "PROJECTED"
    assert sched_edges.forecast_tier({**g, "d": "2026-10-12"}, built, "PROJECTED") == "EARLY"
    assert sched_edges.forecast_tier(g, None, "PROJECTED") == "EARLY"        # undatable
    assert sched_edges.forecast_tier({**g, "tier": "PROJECTED"}, None) == "PROJECTED"
    # the ledger on an unstamped payload: an untiered row is EARLY, so not staked
    lp = tmp_path / "ledger.json"
    p = _write(tmp_path, "nfl.json", {"schedule": [{**g, "value": _badge()}]})
    st = EL.update("nfl", p, lp, now=NOW)
    assert (st["recorded"], st["blocked"]) == (0, 1)


def test_units_are_split_by_tier_never_only_pooled(tmp_path):
    lp = tmp_path / "ledger.json"
    early_w = _legacy_nfl("2026-09-13", "LAC", "ARI", ts_record="2026-09-06T05:41:30Z",
                          dec_at_record=5.15, y=1, roi=4.15)
    early_l = _legacy_nfl("2026-09-20", "BAL", "NO", ts_record="2026-09-13T06:18:57Z",
                          y=0, roi=-1.0)
    proj_w = _legacy_nfl("2026-10-11", "SEA", "ARI", ts_record="2026-10-06T05:00:00Z",
                         model_ts="2026-10-06T02:00:00Z", tier="PROJECTED",
                         dec_at_record=2.0, y=1, roi=1.0)
    rows = {r["key"]: r for r in (early_w, early_l, proj_w)}
    lp.write_text(json.dumps({"v": 1, "rows": rows}), encoding="utf-8")
    nfl = EL.site_block(lp, now=datetime(2026, 10, 12, tzinfo=UTC))["leagues"]["nfl"]
    assert nfl["by_tier"] == {
        "PROJECTED": {"n": 1, "w": 1, "l": 0, "net": 1.0, "roi": 1.0},
        "EARLY": {"n": 2, "w": 1, "l": 1, "net": 3.15, "roi": 1.575},
    }
    assert nfl["settled"]["n"] == 3 and nfl["settled"]["net"] == pytest.approx(4.15)


def test_mlb_rows_carry_the_information_tier_not_the_blend_tier():
    """board.json cards carry tier = "lineup"/"bullpen" (the BLEND tier). The ledger
    stamps the pick policy's information tier (mlbwp/pred_ledger.info_tier)."""
    card = {"game_pk": 1, "date": "2026-09-24", "home": "A", "away": "B",
            "home_win_prob": 0.6, "tier": "lineup", "state": "Preview",
            "start_utc": "2026-09-24T23:00:00Z"}
    now = datetime(2026, 9, 24, 12, tzinfo=UTC)

    def tier_of(c):
        return EL._mlb_games({"leagues": [{"code": "mlb", "games": [c]}]}, now)[0]["tier"]
    assert tier_of({**card, "lineup_source": "official", "pitcher_known": True}) == "CONFIRMED"
    assert tier_of({**card, "lineup_source": "projected", "pitcher_known": True}) == "PROJECTED"
    assert tier_of({**card, "pitcher_known": False}) == "EARLY"


# ------------------------------------------------------------------ ledger: provenance

def test_an_unattributed_nfl_row_is_kept_out_of_the_gate():
    """No model_ts and outside every legacy window: recorded by the still-deployed
    old code (or from an unstamped payload). Nobody can date its build, so it stays
    in the units but not in the CLV gate."""
    r = _legacy_nfl(ts_record="2026-09-26T05:00:00Z")          # after the legacy window
    assert EL._row_model_ts(r) is None
    assert "model build unknown" in EL.stale_model_reason(r)
    assert EL.row_tier(r) == "EARLY"
    assert EL.stale_model_reason({**r, "model_ts": "2026-09-25T12:00:00Z"}) is None
    assert EL.stale_model_reason({**r, "league": "mlb"}) is None


def test_update_all_with_own_payloads_reads_no_repo_results(tmp_path, monkeypatch):
    """A caller passing its own payloads gets no repo results silently mixed in."""
    lp, p, g, _ = _record_nfl(tmp_path)
    g.pop("value")
    _write(tmp_path, "nfl.json", {"generated": "2026-09-08T14:00:00Z", "schedule": [g]})
    res = _csv(tmp_path, "nfl_games.csv", NFL_HDR,
               ["2026_02_NE_SEA,2026,REG,2,2026-09-13,13:00,NE,24,SEA,17"])
    monkeypatch.setattr(EL, "NFL_GAMES", res)                   # the "repo" file
    missing = tmp_path / "none.json"
    later = datetime(2026, 9, 15, 12, tzinfo=UTC)
    own = {"mlb": missing, "nfl": p, "nhl": missing}
    out = {s["league"]: s for s in EL.update_all(lp, now=later, payloads=own)}
    assert out["nfl"]["settled"] == 0
    out = {s["league"]: s for s in EL.update_all(lp, now=later, payloads=own,
                                                 league_results={"nfl": res})}
    assert out["nfl"]["settled"] == 1
