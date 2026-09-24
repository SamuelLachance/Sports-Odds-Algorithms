"""The NFL payload package: complete, current, truthful and self-updating.

Pins the contracts downstream readers (the SPA, the record page, the edge
ledger) rely on:

  - tiers: every forecast carries the information tier it was published in,
    and a played game keeps the tier and freeze time of its PRE-game number
    (July pre-season values are EARLY, never pooled into PROJECTED);
  - results: finals attach every refresh cycle without touching ph/pmc/ct;
  - the chain publishes atomically: a failed step or failed validation leaves
    the live payload untouched and exits non-zero;
  - standings apply the NFL division tiebreakers, not the alphabet;
  - the exact breakdown `cx` adds up to the served probability.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nfl_payload as NP  # noqa: E402
import nfl_ph_freeze as F  # noqa: E402
import nfl_results_attach as A  # noqa: E402
import nfl_weekly as W  # noqa: E402

NFL_JSON = ROOT / "site" / "data" / "nfl.json"
LEDGER = ROOT / "data" / "nfl_ph_ledger.json"


# ------------------------------------------------------------------ tiers
def test_tier_is_projected_only_inside_the_week():
    served = "2026-09-24T15:00:00Z"                       # Thursday, US Eastern
    assert F.tier_for("2026-09-24", served) == "PROJECTED"
    assert F.tier_for("2026-10-01", served) == "PROJECTED"    # exactly 7 days
    assert F.tier_for("2026-10-02", served) == "EARLY"
    assert F.tier_for("2026-12-27", served) == "EARLY"
    # 02:00Z on the 25th is still the 24th in New York
    assert F.tier_for("2026-10-01", "2026-09-25T02:00:00Z") == "PROJECTED"
    assert F.tier_for("2026-10-02", "2026-09-25T02:00:00Z") == "EARLY"


def test_freeze_stamps_unplayed_rows_with_time_and_tier():
    now = "2026-09-24T15:00:00Z"
    sched = [{"w": 3, "d": "2026-09-27", "home": "KC", "away": "BUF", "ph": 0.55,
              "hs": None, "as": None, "qb": {"h": {"id": "q1"}, "a": {"id": "q2"}}},
             {"w": 9, "d": "2026-11-01", "home": "NE", "away": "NYJ", "ph": 0.6,
              "hs": None, "as": None}]
    led, _, _ = F.freeze(sched, {}, now_utc=now)
    assert led["3|KC|BUF"]["t"] == now and led["3|KC|BUF"]["tier"] == "PROJECTED"
    assert led["3|KC|BUF"]["qb"]["h"]["id"] == "q1", "the forecast's QBs freeze with it"
    assert led["9|NE|NYJ"]["tier"] == "EARLY"
    assert [s["tier"] for s in sched] == ["PROJECTED", "EARLY"]


def test_played_game_restores_its_publication_tier_and_time():
    sched = [{"w": 3, "d": "2026-09-27", "home": "KC", "away": "BUF", "ph": 0.70,
              "hs": 20, "as": 17, "qb": {"h": {"id": "today"}}}]
    ledger = {"3|KC|BUF": {"ph": 0.55, "ct": {"elo": 1.0}, "pmc": 0.56,
                           "t": "2026-09-23T12:00:00Z", "tier": "PROJECTED",
                           "qb": {"h": {"id": "then"}}}}
    F.freeze(sched, ledger, now_utc="2026-09-29T12:00:00Z")
    s = sched[0]
    assert s["ph"] == 0.55 and s["pmc"] == 0.56
    assert s["tier"] == "PROJECTED" and s["frozen_at"] == "2026-09-23T12:00:00Z"
    assert s["qb"]["h"]["id"] == "then", "a played game shows the QBs its number used"
    assert ledger["3|KC|BUF"]["t"] == "2026-09-23T12:00:00Z", "played entry rewritten"


def test_pre_stamp_ledger_entries_are_early_july_publications():
    """Weeks 1-2 were last written by the July 30 pre-season serve. Grading them
    as PROJECTED picks would pool 40-day-old numbers into the live-model rate."""
    sched = [{"w": 1, "d": "2026-09-09", "home": "SEA", "away": "NE", "ph": 0.9,
              "hs": 13, "as": 10, "qb": {"h": {"id": "x"}}}]
    F.freeze(sched, {"1|SEA|NE": {"ph": 0.611, "ct": None, "pmc": 0.611}},
             now_utc="2026-09-24T15:00:00Z")
    s = sched[0]
    assert s["ph"] == 0.611 and s["tier"] == "EARLY"
    assert s["frozen_at"] == F.LEGACY_T
    assert "qb" not in s, "no QB receipt was stored: never pair July ph with today's QBs"


def test_orphan_is_flagged_replay_and_never_a_pick():
    sched = [{"w": 3, "d": "2026-09-27", "home": "GB", "away": "CHI", "ph": 0.66,
              "hs": 17, "as": 13}]
    led, _, orphans = F.freeze(sched, {}, warn=lambda m: None,
                               now_utc="2026-09-29T00:00:00Z")
    assert orphans == 1 and sched[0]["replay"] is True
    assert sched[0]["tier"] == "EARLY" and led["3|GB|CHI"]["replay"] is True


def test_tracking_since_uses_the_oldest_receipt():
    assert F.tracking_since({"a": {"ph": 1}, "b": {"t": "2026-09-24T00:00:00Z"}}) == "2026-07-30"
    assert F.tracking_since({"b": {"t": "2026-09-24T00:00:00Z"}}) == "2026-09-24"


# ---------------------------------------------------------- results attach
def _spine(rows):
    base = {"season": "2026", "game_type": "REG", "home_qb_id": "", "away_qb_id": "",
            "home_qb_name": "", "away_qb_name": ""}
    return [dict(base, **r) for r in rows]


def _payload(rows):
    return {"season": 2026, "status": "season", "schedule": rows}


def test_attach_adds_scores_and_receipt_but_never_the_number():
    spine = _spine([{"game_id": "2026_03_BUF_KC", "week": "3", "gameday": "2026-09-27",
                     "home_team": "KC", "away_team": "BUF", "home_score": "20",
                     "away_score": "17", "home_qb_id": "k", "home_qb_name": "Mahomes"}])
    row = {"w": 3, "d": "2026-09-27", "home": "KC", "away": "BUF", "ph": 0.55,
           "pmc": 0.57, "ct": {"elo": 2.0}, "hs": None, "as": None, "tier": "PROJECTED"}
    ledger = {"3|KC|BUF": {"ph": 0.55, "pmc": 0.57, "t": "2026-09-23T12:00:00Z",
                           "tier": "PROJECTED"}}
    pay = _payload([row])
    cnt = A.attach(pay, spine, ledger, warn=lambda m: None)
    assert cnt["new"] == 1
    assert (row["hs"], row["as"]) == (20, 17)
    assert row["ph"] == 0.55 and row["pmc"] == 0.57 and row["ct"] == {"elo": 2.0}
    assert row["frozen_at"] == "2026-09-23T12:00:00Z" and row["tier"] == "PROJECTED"
    assert row["qb_start"]["h"]["name"] == "Mahomes" and row["id"] == "2026_03_BUF_KC"


def test_attach_reports_a_ledger_mismatch_instead_of_repairing_it():
    spine = _spine([{"game_id": "g", "week": "3", "gameday": "2026-09-27",
                     "home_team": "KC", "away_team": "BUF", "home_score": "20",
                     "away_score": "17"}])
    row = {"w": 3, "d": "2026-09-27", "home": "KC", "away": "BUF", "ph": 0.61,
           "hs": None, "as": None}
    msgs = []
    cnt = A.attach(_payload([row]), spine, {"3|KC|BUF": {"ph": 0.55}}, warn=msgs.append)
    assert cnt["mismatch"] == 1 and row["ph"] == 0.61 and msgs


def test_attach_corrects_a_changed_final_and_maps_franchise_codes():
    spine = _spine([{"game_id": "g", "week": "5", "gameday": "2026-10-11",
                     "home_team": "LAR", "away_team": "WSH", "home_score": "24",
                     "away_score": "21"}])
    row = {"w": 5, "d": "2026-10-11", "home": "LA", "away": "WAS", "ph": 0.5,
           "hs": 21, "as": 21, "tier": "EARLY"}
    cnt = A.attach(_payload([row]), spine, {}, warn=lambda m: None)
    assert cnt["corrected"] == 1 and (row["hs"], row["as"]) == (24, 21)


def test_validate_catches_what_must_block_a_publish():
    spine = _spine([{"game_id": "g", "week": "1", "gameday": "2026-09-10",
                     "home_team": "KC", "away_team": "BUF", "home_score": "20",
                     "away_score": "17"}])
    unattached = {"w": 1, "d": "2026-09-10", "home": "KC", "away": "BUF", "ph": 0.5,
                  "hs": None, "as": None}
    errs = A.validate(_payload([unattached]), spine, {})
    assert any("final not attached" in e for e in errs)
    assert any("272" in e for e in errs)
    drifted = dict(unattached, hs=20, **{"as": 17}, tier="EARLY", ph=0.7)
    errs = A.validate(_payload([drifted]), spine, {"1|KC|BUF": {"ph": 0.5}})
    assert any("ledger" in e for e in errs)
    assert A.validate({"season": 2026, "schedule": []}, spine, {}) == ["no schedule"]


def test_run_refuses_a_gutted_payload_and_does_not_write(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)              # run() chdirs to the project
    p = tmp_path / "nfl.json"
    p.write_text(json.dumps({"status": "preseason", "season": 2026}), encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    assert A.run(str(p), fetch=False) == 2
    assert p.read_text(encoding="utf-8") == before


# ------------------------------------------------------ atomic weekly chain
def test_a_failed_step_leaves_the_live_payload_untouched(tmp_path, monkeypatch):
    live, stage = tmp_path / "nfl.json", tmp_path / "stage.json"
    live.write_text('{"good": true}', encoding="utf-8")
    calls = []

    def fake(step, tmo, args=(), env=None):
        calls.append(step)
        Path(env["NFL_PAYLOAD"]).write_text('{"half": true}', encoding="utf-8")
        return "nfl_lineups" not in step
    monkeypatch.setattr(W, "run_step", fake)
    fails = W.build_payload(str(stage), str(live), str(tmp_path / "led.json"),
                            str(tmp_path / "led_stage.json"))
    assert fails and "nfl_lineups" in fails[0]
    assert live.read_text(encoding="utf-8") == '{"good": true}'
    assert "phase0/nfl_season_serve.py" not in calls, "the chain must stop at the failure"


def test_a_failed_validation_leaves_the_live_payload_untouched(tmp_path, monkeypatch):
    live, stage = tmp_path / "nfl.json", tmp_path / "stage.json"
    live.write_text('{"good": true}', encoding="utf-8")

    def fake(step, tmo, args=(), env=None):
        Path(env["NFL_PAYLOAD"]).write_text('{"built": true}', encoding="utf-8")
        return "nfl_results_attach" not in step
    monkeypatch.setattr(W, "run_step", fake)
    assert W.build_payload(str(stage), str(live), str(tmp_path / "led.json"),
                           str(tmp_path / "led_stage.json"))
    assert live.read_text(encoding="utf-8") == '{"good": true}'


def test_a_validated_build_is_swapped_in(tmp_path, monkeypatch):
    live, stage = tmp_path / "nfl.json", tmp_path / "stage.json"
    live.write_text('{"old": true}', encoding="utf-8")
    seen = []

    def fake(step, tmo, args=(), env=None):
        seen.append((step, tuple(args), env.get("NFL_PAYLOAD")))
        Path(env["NFL_PAYLOAD"]).write_text('{"new": true}', encoding="utf-8")
        return True
    monkeypatch.setattr(W, "run_step", fake)
    assert W.build_payload(str(stage), str(live), str(tmp_path / "led.json"),
                           str(tmp_path / "led_stage.json")) == []
    assert json.loads(live.read_text(encoding="utf-8")) == {"new": True}
    assert not stage.exists()
    assert all(env == str(stage) for _, _, env in seen), "every step must build the stage"
    last = seen[-1]
    assert last[0] == "phase0/nfl_results_attach.py" and "--full" in last[1]


def test_weekly_exits_nonzero_when_the_required_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # main() chdirs; monkeypatch restores cwd
    monkeypatch.setattr(W, "PROJECT", tmp_path)
    monkeypatch.setattr(W, "fetch", lambda p, u, r: "failed" if r else "skip")
    monkeypatch.setattr(W, "REDUCERS", [("absent.csv", "x.py")])
    assert W.main() == 1


def test_weekly_fetch_only_run_is_green_when_fetches_are(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(W, "PROJECT", tmp_path)
    monkeypatch.setattr(W, "fetch", lambda p, u, r: "ok")
    monkeypatch.setattr(W, "REDUCERS", [("absent.csv", "x.py")])
    assert W.main() == 0


# ---------------------------------------------------------------- standings
def _g(d, w, h, a, hs, as_):
    return {"d": d, "w": w, "home": h, "away": a, "hs": hs, "as": as_}


def test_head_to_head_breaks_a_two_team_tie_before_the_alphabet():
    games = [_g("2026-09-10", 1, "BUF", "MIA", 10, 20),     # MIA beats BUF
             _g("2026-09-17", 2, "BUF", "NYJ", 30, 0),
             _g("2026-09-17", 2, "MIA", "NE", 0, 30)]
    st = NP.standings(games)
    assert (st["BUF"]["w"], st["MIA"]["w"]) == (1, 1)
    assert st["MIA"]["div_rank"] < st["BUF"]["div_rank"], "head-to-head decides, not 'B' < 'M'"


def test_record_strings_and_streaks():
    games = [_g("2026-09-10", 1, "KC", "LV", 20, 20), _g("2026-09-17", 2, "DEN", "KC", 10, 24)]
    kc = NP.standings(games)["KC"]
    assert (kc["w"], kc["l"], kc["t"], kc["gp"]) == (1, 0, 1, 2)
    assert kc["home"] == "0-0-1" and kc["away"] == "1-0" and kc["div_rec"] == "1-0-1"
    assert kc["streak"] == "W1" and kc["pct"] == 0.75


def test_real_2025_nfc_south_goes_to_carolina():
    """8-9 three ways; CAR hosted the wild-card game, so CAR won the division.
    The old stable sort put ATL first because 'A' < 'C'."""
    path = ROOT / "data" / "nfl_games.csv"
    if not path.is_file():
        pytest.skip("games spine not present")
    st = NP.standings(NP.final_rows(NP.read_games(str(path)), 2025))
    assert st["CAR"]["div_rank"] == 1
    assert {st[t]["w"] for t in ("ATL", "CAR", "TB")} == {8}


def test_eastern_time_conversion_with_and_without_tzdata():
    assert NP.et_to_utc("2026-09-24", "20:15") == "2026-09-25T00:15:00Z"
    assert NP.et_to_utc("2026-12-06", "13:00") == "2026-12-06T18:00:00Z"
    from datetime import date
    assert NP._us_eastern_offset(date(2026, 9, 24), 20) == -4
    assert NP._us_eastern_offset(date(2026, 11, 8), 13) == -5
    assert NP._us_eastern_offset(date(2026, 11, 1), 1) == -4     # before 02:00
    assert NP._us_eastern_offset(date(2026, 11, 1), 13) == -5


# ------------------------------------------------------- exact breakdown
def test_exact_breakdown_adds_up_to_the_probability():
    ic = 0.155
    for ph, ct in ((0.611, {"elo": 4.1, "qb": 1.5, "units": -0.1, "roster": 0.9,
                             "ts": 0.3, "luck": 0.3, "avail": 0.0}),
                   (0.436, {"elo": 2.4, "qb": -6.5, "units": -0.4, "roster": -3.4,
                            "ts": -2.4, "luck": 0.1})):
        p = ph
        scale = p * (1 - p) * 100
        # make ct consistent with ph through the logistic, as the serve does
        z = math.log(p / (1 - p)) - ic - sum(v / scale for v in ct.values())
        ct = dict(ct, hfa=round(z * scale, 1))
        cx = NP.exact_contrib(ph, ct, ic)
        assert cx is not None
        assert abs(50 + sum(cx.values()) - 100 * ph) < 1e-9
        assert cx["base"] == round((1 / (1 + math.exp(-ic)) - 0.5) * 100, 1)
        assert list(cx) == ["base"] + list(NP.CX_ORDER)


def test_exact_breakdown_refuses_inconsistent_inputs():
    assert NP.exact_contrib(0.9, {"elo": -20.0}, 0.155) is None
    assert NP.exact_contrib(None, {"elo": 1.0}, 0.155) is None


# -------------------------------------------------- wiring (CI and chain)
def test_refresh_runs_the_results_step_before_bets_settle():
    src = (ROOT / "refresh.py").read_text(encoding="utf-8")
    assert "phase0/nfl_results_attach.py" in src
    assert src.index("phase0/nfl_results_attach.py") < src.index("edge_ledger.attach_bets()")
    yml = (ROOT / ".github" / "workflows" / "refresh.yml").read_text(encoding="utf-8")
    assert "site/data/nfl.json" in yml and "data/nfl_games.csv" in yml
    wk = (ROOT / ".github" / "workflows" / "nfl-weekly.yml").read_text(encoding="utf-8")
    assert "if: always()" in wk


def test_results_step_is_standard_library_only():
    import ast
    for name in ("nfl_results_attach.py", "nfl_payload.py", "nfl_ph_freeze.py",
                 "nfl_weekly.py", "nfl_season_guards.py"):
        tree = ast.parse((ROOT / "phase0" / name).read_text(encoding="utf-8"))
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module}
        assert not mods & {"numpy", "sklearn", "pandas", "scipy"}, (name, mods)


def test_chain_scripts_write_the_staged_payload_not_the_live_file():
    for name in ("nfl_site_data.py", "nfl_site_db.py", "nfl_trueskill_players.py",
                 "nfl_lineups.py", "nfl_season_serve.py"):
        src = (ROOT / "phase0" / name).read_text(encoding="utf-8")
        assert "payload_path()" in src, name
        assert 'open("site/data/nfl.json", "w")' not in src, name


# ---------------------------------------------- the shipped payload itself
def _shipped():
    if not (NFL_JSON.is_file() and LEDGER.is_file()):
        pytest.skip("NFL artifacts not built")
    return (json.loads(NFL_JSON.read_text(encoding="utf-8")),
            json.loads(LEDGER.read_text(encoding="utf-8")))


def test_shipped_played_games_equal_the_ledger():
    n, led = _shipped()
    played = [g for g in n["schedule"] if g.get("hs") is not None]
    for g in played:
        ent = led[F.ledger_key(g)]
        assert g["ph"] == ent["ph"], F.ledger_key(g)
        if ent.get("pmc") is not None:
            assert g["pmc"] == ent["pmc"], F.ledger_key(g)


def test_shipped_rows_carry_tiers_and_played_rows_receipts():
    n, _ = _shipped()
    if not any("tier" in g for g in n["schedule"]):
        pytest.skip("payload predates the tier stamp")
    for g in n["schedule"]:
        assert g.get("tier") in ("PROJECTED", "EARLY"), F.ledger_key(g)
        if g.get("hs") is not None and not g.get("replay"):
            assert g.get("frozen_at"), F.ledger_key(g)
            # a July pre-season number can never be graded as a live-model pick
            if g["frozen_at"] < "2026-09-01":
                assert g["tier"] == "EARLY"


def test_shipped_payload_says_which_season_everything_is():
    n, _ = _shipped()
    if "standings_season" not in n:
        pytest.skip("payload predates the season fields")
    assert isinstance(n["standings_season"], int)
    assert isinstance(n["stats_season"], int)
    assert n["stats_prev_season"] == n["stats_season"] - 1
    for t in n["teams"].values():
        assert "prev" in t and t["prev"]["season"] == n["standings_season"] - 1


def test_shipped_rosters_keep_injured_players_and_cover_the_lineups():
    n, _ = _shipped()
    if not any("status" in p for p in n["players"].values()):
        pytest.skip("payload predates roster statuses")
    P = n["players"]
    for code, t in n["teams"].items():
        for pid in t["roster"] + t.get("reserve", []) + t.get("practice", []):
            assert P[pid]["team"] == code
        for side in ("off", "def"):
            for e in (t.get("lineup") or {}).get(side, []):
                assert e["id"] in P, f"{code} starter {e['name']} has no player page"
    reserve = [p for p in P.values() if p["status"] in ("IR", "IR-R", "PUP", "RES")]
    assert len(reserve) > 50, "injured players vanished from the payload again"
    assert all(p["status"] in n["status_labels"] for p in P.values())
