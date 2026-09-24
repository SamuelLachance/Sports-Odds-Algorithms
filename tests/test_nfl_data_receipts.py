"""NFL receipts: a played game is graded only with the receipt of the number
that was actually published before kickoff.

Pins the fixes from the nfl-data review (2026-09-24):
  - the freeze locks a game at kickoff, not at the final (nflverse fills
    scores hours later, and a Sunday-slate run used to rewrite in-progress
    games with a post-kickoff PROJECTED stamp);
  - PROJECTED means within 7 x 24 h of kickoff (start_utc), not a date window;
  - the CI results step never copies a ledger receipt onto a different number,
    and validates BEFORE it writes;
  - the weekly chain stages the ledger and swaps it in with the payload, only
    when no pre-game entry moved;
  - refresh.yml re-attaches on top of a local serve instead of reverting it.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nfl_ph_freeze as F  # noqa: E402
import nfl_results_attach as A  # noqa: E402
import nfl_weekly as W  # noqa: E402

NFL_JSON = ROOT / "site" / "data" / "nfl.json"
LEDGER = ROOT / "data" / "nfl_ph_ledger.json"

TNF = {"w": 3, "d": "2026-09-24", "t": "20:15", "start_utc": "2026-09-25T00:15:00Z",
       "home": "GB", "away": "ATL"}


def _row(**kw):
    r = dict(TNF, ph=0.70, pmc=0.70, ct={"elo": 1.0}, hs=None)
    r["as"] = None
    r.update(kw)
    return r


# ------------------------------------------------------------ the tier clock
def test_projected_is_seven_times_24_hours_before_kickoff():
    served = "2026-09-24T15:08:56Z"                   # Thu 11:08 ET
    wk4_tnf = "2026-10-02T00:15:00Z"                  # Thu 20:15 ET, 7 d 9 h later
    assert F.tier_for("2026-10-01", served, wk4_tnf) == "EARLY"
    assert F.tier_for("2026-10-01", "2026-09-25T00:15:00Z", wk4_tnf) == "PROJECTED"  # exactly 7 d
    assert F.tier_for("2026-09-27", served, "2026-09-27T17:00:00Z") == "PROJECTED"
    # no start_utc: the legacy date rule still answers
    assert F.tier_for("2026-10-01", served) == "PROJECTED"


def test_the_ledger_stamp_uses_kickoff_time():
    row = _row(w=4, d="2026-10-01", start_utc="2026-10-02T00:15:00Z", home="CLE", away="PIT")
    led, _, _ = F.freeze([row], {}, now_utc="2026-09-24T15:08:56Z")
    assert led["4|CLE|PIT"]["tier"] == "EARLY" and row["tier"] == "EARLY"


# ------------------------------------------------------------ kickoff lock
def test_a_game_in_progress_keeps_its_pre_game_entry():
    """Kicked off, no final yet: the ledger value is restored and never rewritten."""
    ent = {"ph": 0.683, "pmc": 0.683, "ct": {"elo": 2.0}, "t": "2026-09-24T15:08:56Z",
           "tier": "PROJECTED", "qb": {"h": {"id": "then"}}}
    ledger = {"3|GB|ATL": dict(ent)}
    row = _row(qb={"h": {"id": "now"}})
    led, frozen, orphans = F.freeze([row], ledger, now_utc="2026-09-25T01:30:00Z")
    assert (frozen, orphans) == (1, 0)
    assert led["3|GB|ATL"] == ent, "a kicked-off game's entry was rewritten"
    assert row["ph"] == 0.683 and row["pmc"] == 0.683 and row["ct"] == {"elo": 2.0}
    assert row["tier"] == "PROJECTED" and row["frozen_at"] == "2026-09-24T15:08:56Z"
    assert row["qb"]["h"]["id"] == "then"


def test_before_kickoff_the_entry_still_refreshes():
    ledger = {"3|GB|ATL": {"ph": 0.6, "t": "2026-09-23T12:00:00Z", "tier": "PROJECTED"}}
    row = _row()
    F.freeze([row], ledger, now_utc="2026-09-24T23:00:00Z")
    assert ledger["3|GB|ATL"]["ph"] == 0.70 and ledger["3|GB|ATL"]["t"] == "2026-09-24T23:00:00Z"


def test_a_kicked_off_game_without_entry_is_a_replay():
    msgs = []
    row = _row()
    led, _, orphans = F.freeze([row], {}, warn=msgs.append, now_utc="2026-09-25T01:00:00Z")
    assert orphans == 1 and msgs
    assert led["3|GB|ATL"]["replay"] is True and led["3|GB|ATL"]["tier"] == "EARLY"
    assert row["replay"] is True and row["tier"] == "EARLY" and "frozen_at" not in row


def test_an_entry_stamped_after_kickoff_is_never_graded_as_a_pick():
    msgs = []
    row = _row(hs=24, **{"as": 20})
    ledger = {"3|GB|ATL": {"ph": 0.70, "t": "2026-09-25T01:00:00Z", "tier": "PROJECTED"}}
    F.freeze([row], ledger, warn=msgs.append, now_utc="2026-09-26T12:00:00Z")
    assert row["tier"] == "EARLY" and row["replay"] is True and "frozen_at" not in row
    assert any("after kickoff" in m for m in msgs)


def test_a_replay_entry_restores_without_a_freeze_time():
    row = _row(hs=24, **{"as": 20})
    F.freeze([row], {"3|GB|ATL": {"ph": 0.70, "replay": True, "tier": "EARLY",
                                  "t": "2026-09-26T00:00:00Z"}},
             now_utc="2026-09-27T00:00:00Z")
    assert row["replay"] is True and row["tier"] == "EARLY" and "frozen_at" not in row


# ------------------------------------------------- CI results step receipts
def _spine_rows():
    return [{"season": "2026", "game_type": "REG", "game_id": "2026_03_ATL_GB",
             "week": "3", "gameday": "2026-09-24", "home_team": "GB",
             "away_team": "ATL", "home_score": "24", "away_score": "20",
             "home_qb_id": "", "away_qb_id": "", "home_qb_name": "", "away_qb_name": ""}]


def test_a_receipt_for_another_number_is_refused():
    """The reviewer's reproduction: payload ph 0.66 (EARLY publication) + ledger
    ph 0.70 (PROJECTED, 20:00Z). The row must not become 'ph 0.66, PROJECTED'."""
    row = _row(ph=0.66, pmc=0.66, tier="EARLY")
    ledger = {"3|GB|ATL": {"ph": 0.70, "pmc": 0.70, "t": "2026-09-24T20:00:00Z",
                           "tier": "PROJECTED"}}
    msgs = []
    cnt = A.attach({"season": 2026, "schedule": [row]}, _spine_rows(), ledger,
                   warn=msgs.append)
    assert cnt["mismatch"] == 1 and msgs
    assert row["ph"] == 0.66 and (row["hs"], row["as"]) == (24, 20)
    assert row["tier"] == "EARLY" and "frozen_at" not in row
    assert row["receipt_mismatch"] is True


def test_a_matching_receipt_is_copied():
    row = _row(ph=0.683, pmc=0.683, tier="PROJECTED")
    ledger = {"3|GB|ATL": {"ph": 0.683, "pmc": 0.683, "t": "2026-09-24T15:08:56Z",
                           "tier": "PROJECTED"}}
    A.attach({"season": 2026, "schedule": [row]}, _spine_rows(), ledger, warn=lambda m: None)
    assert row["tier"] == "PROJECTED" and row["frozen_at"] == "2026-09-24T15:08:56Z"
    assert "receipt_mismatch" not in row


def test_validate_rejects_a_freeze_time_after_kickoff():
    row = _row(hs=24, tier="PROJECTED", frozen_at="2026-09-25T01:00:00Z", **{"as": 20})
    ledger = {"3|GB|ATL": {"ph": 0.70, "pmc": 0.70}}
    errs = A.validate({"season": 2026, "schedule": [row]}, _spine_rows(), ledger)
    assert any("after kickoff" in e for e in errs)
    row2 = dict(row)
    row2.pop("frozen_at")
    errs = A.validate({"season": 2026, "schedule": [row2]}, _spine_rows(), ledger)
    assert any("no pre-game freeze time" in e for e in errs)


def _write_spine(path):
    rows = _spine_rows()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_run_validates_before_writing_and_refuses_a_mismatch(tmp_path, monkeypatch):
    spine, led, pay = tmp_path / "games.csv", tmp_path / "led.json", tmp_path / "nfl.json"
    _write_spine(spine)
    led.write_text(json.dumps({"3|GB|ATL": {"ph": 0.70, "pmc": 0.70,
                                             "t": "2026-09-24T20:00:00Z",
                                             "tier": "PROJECTED"}}), encoding="utf-8")
    pay.write_text(json.dumps({"season": 2026, "status": "season",
                               "schedule": [_row(ph=0.66, pmc=0.66, tier="EARLY")]}),
                   encoding="utf-8")
    before = pay.read_text(encoding="utf-8")
    monkeypatch.setattr(A, "GAMES_CSV", str(spine))
    monkeypatch.setenv("NFL_LEDGER", str(led))
    monkeypatch.chdir(ROOT)
    assert A.run(str(pay), fetch=False) == 1
    assert pay.read_text(encoding="utf-8") == before, "a refused attach was written"


def test_run_writes_a_clean_attach(tmp_path, monkeypatch):
    spine, led, pay = tmp_path / "games.csv", tmp_path / "led.json", tmp_path / "nfl.json"
    _write_spine(spine)
    led.write_text(json.dumps({"3|GB|ATL": {"ph": 0.683, "pmc": 0.683,
                                             "t": "2026-09-24T15:08:56Z",
                                             "tier": "PROJECTED"}}), encoding="utf-8")
    pay.write_text(json.dumps({"season": 2026, "status": "season",
                               "schedule": [_row(ph=0.683, pmc=0.683, tier="PROJECTED")]}),
                   encoding="utf-8")
    monkeypatch.setattr(A, "GAMES_CSV", str(spine))
    monkeypatch.setenv("NFL_LEDGER", str(led))
    monkeypatch.chdir(ROOT)
    A.run(str(pay), fetch=False)          # 1: a 1-row schedule is not 272 rows
    row = json.loads(pay.read_text(encoding="utf-8"))["schedule"][0]
    assert (row["hs"], row["as"]) == (24, 20) and row["ph"] == 0.683
    assert row["frozen_at"] == "2026-09-24T15:08:56Z" and row["tier"] == "PROJECTED"


# ------------------------------------------------ staged ledger, swapped together
def test_ledger_swap_refuses_a_moved_pre_game_entry():
    played = dict(TNF, hs=24, **{"as": 20})
    later = {"w": 4, "home": "CLE", "away": "PIT", "start_utc": "2026-10-02T00:15:00Z",
             "hs": None, "as": None}
    live = {"3|GB|ATL": {"ph": 0.683}, "4|CLE|PIT": {"ph": 0.43}}
    payload = {"served_at": "2026-09-26T12:00:00Z", "schedule": [played, later]}
    assert W.ledger_swap_errors(live, dict(live, **{"4|CLE|PIT": {"ph": 0.45}}), payload) == []
    errs = W.ledger_swap_errors(live, dict(live, **{"3|GB|ATL": {"ph": 0.70}}), payload)
    assert errs and "changed" in errs[0]
    errs = W.ledger_swap_errors(live, {"3|GB|ATL": {"ph": 0.683}}, payload)
    assert errs and "vanish" in errs[0]
    # kicked off at serve time but not yet scored: locked too
    running = dict(TNF, hs=None, **{"as": None})
    errs = W.ledger_swap_errors({"3|GB|ATL": {"ph": 0.683}}, {"3|GB|ATL": {"ph": 0.70}},
                                {"served_at": "2026-09-25T01:00:00Z", "schedule": [running]})
    assert errs


def test_a_rebuild_may_never_change_a_played_games_published_number():
    live = {"schedule": [dict(TNF, ph=0.66, pmc=0.66, hs=24, **{"as": 20}),
                         {"w": 4, "home": "CLE", "away": "PIT", "ph": 0.43, "hs": None,
                          "as": None}]}
    same = {"schedule": [dict(TNF, ph=0.66, pmc=0.66, hs=24, **{"as": 20}),
                         {"w": 4, "home": "CLE", "away": "PIT", "ph": 0.47}]}
    assert W.played_number_errors(live, same) == []      # unplayed rows may move
    flipped = {"schedule": [dict(TNF, ph=0.70, pmc=0.70, hs=24, **{"as": 20})]}
    errs = W.played_number_errors(live, flipped)
    assert errs and "published ph/pmc" in errs[0]
    assert W.played_number_errors(None, flipped) == []


def test_the_swap_gate_refuses_a_flipped_played_number(tmp_path, monkeypatch):
    live_row = dict(TNF, ph=0.66, pmc=0.66, hs=24, **{"as": 20})

    def fake(step, tmo, args=(), env=None):
        Path(env["NFL_PAYLOAD"]).write_text(json.dumps(
            {"schedule": [dict(live_row, ph=0.70, pmc=0.70)]}), encoding="utf-8")
        return True
    live, stage = tmp_path / "nfl.json", tmp_path / "stage.json"
    live.write_text(json.dumps({"schedule": [live_row]}), encoding="utf-8")
    before = live.read_text(encoding="utf-8")
    monkeypatch.setattr(W, "run_step", fake)
    fails = W.build_payload(str(stage), str(live), str(tmp_path / "led.json"),
                            str(tmp_path / "led_stage.json"))
    assert fails and "published ph/pmc" in fails[0]
    assert live.read_text(encoding="utf-8") == before


def _chain(tmp_path, monkeypatch, fake):
    live, stage = tmp_path / "nfl.json", tmp_path / "stage.json"
    led, led_stage = tmp_path / "led.json", tmp_path / "led_stage.json"
    live.write_text('{"old": true}', encoding="utf-8")
    led.write_text('{"1|A|B": {"ph": 0.5}}', encoding="utf-8")
    monkeypatch.setattr(W, "run_step", fake)
    fails = W.build_payload(str(stage), str(live), str(led), str(led_stage))
    return fails, live, led, led_stage


def test_the_serve_writes_the_staged_ledger_and_both_swap_together(tmp_path, monkeypatch):
    seen = []

    def fake(step, tmo, args=(), env=None):
        seen.append(env.get("NFL_LEDGER"))
        Path(env["NFL_PAYLOAD"]).write_text('{"new": true}', encoding="utf-8")
        if "season_serve" in step:
            led = json.loads(Path(env["NFL_LEDGER"]).read_text(encoding="utf-8"))
            led["9|C|D"] = {"ph": 0.6}
            Path(env["NFL_LEDGER"]).write_text(json.dumps(led), encoding="utf-8")
        return True
    fails, live, led, led_stage = _chain(tmp_path, monkeypatch, fake)
    assert fails == []
    assert set(seen) == {str(led_stage)}, "every step must see the staged ledger"
    assert json.loads(live.read_text(encoding="utf-8")) == {"new": True}
    assert set(json.loads(led.read_text(encoding="utf-8"))) == {"1|A|B", "9|C|D"}
    assert not led_stage.exists()


def test_a_failed_gate_leaves_the_live_ledger_untouched(tmp_path, monkeypatch):
    def fake(step, tmo, args=(), env=None):
        Path(env["NFL_PAYLOAD"]).write_text('{"new": true}', encoding="utf-8")
        if "season_serve" in step:
            Path(env["NFL_LEDGER"]).write_text('{"1|A|B": {"ph": 0.9}}', encoding="utf-8")
        return "nfl_results_attach" not in step
    fails, live, led, _ = _chain(tmp_path, monkeypatch, fake)
    assert fails
    assert live.read_text(encoding="utf-8") == '{"old": true}'
    assert led.read_text(encoding="utf-8") == '{"1|A|B": {"ph": 0.5}}', \
        "the serve's unpublished ledger leaked onto the live file"


def test_a_lost_ledger_entry_blocks_the_swap(tmp_path, monkeypatch):
    def fake(step, tmo, args=(), env=None):
        Path(env["NFL_PAYLOAD"]).write_text('{"new": true}', encoding="utf-8")
        if "season_serve" in step:
            Path(env["NFL_LEDGER"]).write_text("{}", encoding="utf-8")
        return True
    fails, live, led, _ = _chain(tmp_path, monkeypatch, fake)
    assert fails and "vanish" in fails[0]
    assert live.read_text(encoding="utf-8") == '{"old": true}'


# ---------------------------------------------------------------- wiring
def test_wiring_serve_ledger_refresh_and_workflow():
    serve = (ROOT / "phase0" / "nfl_season_serve.py").read_text(encoding="utf-8")
    assert 'os.environ.get("NFL_LEDGER")' in serve
    assert "refusing to serve" in serve, "the serve must never bootstrap over played games"
    ref = (ROOT / "refresh.py").read_text(encoding="utf-8")
    assert "TimeoutExpired" in ref, "a stalled NFL step must not abort MLB's refresh"
    yml = (ROOT / ".github" / "workflows" / "refresh.yml").read_text(encoding="utf-8")
    assert "git checkout origin/master -- site/data/nfl.json data/nfl_ph_ledger.json" in yml
    assert yml.index("nfl_results_attach.py --no-fetch") < yml.index("git pull --rebase")


# ------------------------------------------------------ the shipped payload
def _shipped():
    if not (NFL_JSON.is_file() and LEDGER.is_file()):
        pytest.skip("NFL artifacts not built")
    return (json.loads(NFL_JSON.read_text(encoding="utf-8")),
            json.loads(LEDGER.read_text(encoding="utf-8")))


def test_shipped_receipts_predate_kickoff():
    n, _ = _shipped()
    for g in n["schedule"]:
        if g.get("hs") is None or g.get("replay") or not g.get("start_utc"):
            continue
        assert g.get("frozen_at"), F.ledger_key(g)
        assert F.utc(g["frozen_at"]) <= F.utc(g["start_utc"]), F.ledger_key(g)
        assert not g.get("receipt_mismatch"), F.ledger_key(g)


def test_shipped_unplayed_tiers_follow_the_kickoff_clock():
    n, led = _shipped()
    if "served_at" not in n:
        pytest.skip("payload predates served_at")
    for g in n["schedule"]:
        if g.get("hs") is not None or not g.get("start_utc"):
            continue
        ent = led[F.ledger_key(g)]
        if F.utc(ent.get("t")) is None or F.utc(ent["t"]) > F.utc(g["start_utc"]):
            continue
        assert g["tier"] == F.tier_for(g["d"], ent["t"], g["start_utc"]), F.ledger_key(g)


def test_shipped_player_tiers_are_never_null():
    n, _ = _shipped()
    rated = [p for p in n["players"].values() if p.get("rating")]
    if not any(p["rating"].get("tier") == "–" for p in rated):
        pytest.skip("payload predates the served no-grade dash")
    assert all(p["rating"].get("tier") is not None for p in rated)


def test_shipped_q_receipts_name_only_active_players_that_moved_the_number():
    n, _ = _shipped()
    P = n["players"]
    for g in n["schedule"]:
        if not g.get("q") or g.get("hs") is not None:
            continue
        assert round(g["ct"].get("avail", 0.0), 1) != 0.0, F.ledger_key(g)
        for side in ("h", "a"):
            for e in g["q"][side]:
                assert P.get(e["id"], {}).get("status") == "ACT", (F.ledger_key(g), e["name"])
