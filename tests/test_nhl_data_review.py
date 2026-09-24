"""Review fixes for the nhl-data package.

  * the display fetch is bounded in wall-clock time (a hung NHL API must not
    push nhl_update past refresh.py's 900 s subprocess timeout);
  * RAPM ratings carry the shift-data coverage they were fit on, unrated
    reasons name it, and a CI checkout never guesses a fresh build;
  * the team GLASSBOX rating counts every lineup slot;
  * last season (a TEST season) is emitted verbatim from the published
    numbers, never recomputed;
  * playoff finals stay in the schedule; every ledger key is refreshed;
  * season lines name their ice-time unit.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_names as N  # noqa: E402
import nhl_site_fetch as FX  # noqa: E402
import nhl_site_players as SP  # noqa: E402
import nhl_site_prev as PREV  # noqa: E402
import nhl_update as U  # noqa: E402
from nhl_site_schedule import ledger_path, playoff_finals  # noqa: E402

PAYLOAD = ROOT / "site" / "data" / "nhl.json"
LEDGER = ROOT / "data" / "nhl_hp_ledger.json"


class Counter:
    def __init__(self, fn=None):
        self.n = 0
        self.fn = fn

    def __call__(self, url):
        self.n += 1
        if self.fn:
            return self.fn(url)
        raise OSError("timed out")


# ------------------------------------------------------------ time budget --
def test_roster_loop_stops_after_three_failures_in_a_row(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"1": {"name": "A", "team": "TOR", "on_roster": True}}))
    get = Counter()
    ok, n = N.refresh(0, teams=tuple(f"T{i}" for i in range(32)), get_fn=get,
                      path=str(path), sleep=0)
    assert get.n == N.MAX_CONSEC_FAIL == 3 and ok == 0
    assert json.loads(path.read_text())["1"]["on_roster"] is True   # nothing written


def test_roster_loop_respects_the_deadline(tmp_path):
    path = tmp_path / "names.json"
    get = Counter(lambda url: {"forwards": [{"id": 1, "firstName": "A", "lastName": "B",
                                             "positionCode": "C"}]})
    ok, n = N.refresh(0, teams=("TOR", "MTL"), get_fn=get, path=str(path), sleep=0,
                      deadline=time.monotonic() - 1)
    assert get.n == 0 and ok == 0


def test_a_slow_api_is_cut_off_by_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(FX, "schedule_path", lambda s: str(tmp_path / f"sch_{s}.json"))

    def slow(url):
        time.sleep(0.05)
        return {"games": []}
    get = Counter(slow)
    FX.fetch_schedule(20262027, force=True, teams=tuple(f"T{i}" for i in range(32)),
                      get_fn=get, sleep=0, deadline=time.monotonic() + 0.12)
    assert 1 <= get.n < 32


def test_schedule_loop_stops_after_three_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(FX, "schedule_path", lambda s: str(tmp_path / f"sch_{s}.json"))
    get = Counter()
    msg = FX.fetch_schedule(20262027, force=True, teams=tuple(f"T{i}" for i in range(32)),
                            get_fn=get, sleep=0)
    assert get.n == 3 and "previous cache kept" in msg


def test_a_partial_club_stats_answer_never_replaces_a_complete_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(FX, "stats_path", lambda s: str(tmp_path / f"st_{s}.json"))
    old = {"fetched_at": "2000-01-01T00:00:00Z", "season": 1, "src": "rest",
           "skaters": {"1": {"gp": 82}}, "goalies": {}}
    FX.save(FX.stats_path(1), old)

    def fake(url):
        if "stats/rest" in url or "/club-stats/MTL/" in url:
            raise OSError("down")
        return {"skaters": [{"playerId": 2, "gamesPlayed": 1}], "goalies": []}
    monkeypatch.setattr(FX, "TEAMS", ("TOR", "MTL"))
    msg = FX.fetch_stats(1, 3, force=True, get_fn=fake)
    assert "previous cache kept" in msg
    assert FX.load(FX.stats_path(1))["skaters"] == {"1": {"gp": 82}}


def test_a_spent_budget_skips_every_step_without_a_request(monkeypatch, capsys):
    def no_network(*a, **k):
        raise AssertionError("a request was made after the budget was spent")
    monkeypatch.setattr(FX.urllib.request, "urlopen", no_network)
    assert FX.main(force=True, budget_s=0) == 0
    out = capsys.readouterr().out
    if "no season derivable" not in out:
        assert out.count("time budget spent") == 6


def test_lineup_archive_stops_at_the_deadline_and_after_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "LINEUP_ARCHIVE", str(tmp_path / "arch.jsonl"))
    monkeypatch.setattr(U.time, "sleep", lambda s: None)
    games = [{"id": i, "d": "2026-10-10", "home": "TOR", "away": "MTL"} for i in range(8)]
    get = Counter()
    monkeypatch.setattr(U, "get", get)
    assert U.archive_lineups(games, "2026-10-10", deadline=time.monotonic() - 1) == 0
    assert get.n == 0
    assert U.archive_lineups(games, "2026-10-10") == 0
    assert get.n == U.MAX_CONSEC_FAIL             # 3 landing failures, then stop


def test_worst_case_update_run_fits_inside_refresh_timeout():
    """refresh.py kills nhl_update at 900 s. Each budget is checked BEFORE a
    request, so a stage can overshoot by its largest unchecked burst."""
    assert U.SCHEDULE_BUDGET_S + U.REQ_TIMEOUT < U.ARCHIVE_BUDGET_S
    assert U.ARCHIVE_BUDGET_S + 3 * U.REQ_TIMEOUT < U.UPDATE_BUDGET_S
    assert FX.BUDGET_S <= U.UPDATE_BUDGET_S
    assert U.UPDATE_BUDGET_S + 3 * FX.REQ_TIMEOUT < 900 * 0.8
    assert FX.REQ_TIMEOUT <= 15 and N.REQ_TIMEOUT <= 15


# ------------------------------------------------------------ RAPM meta --
def _files(tmp_path):
    spine = tmp_path / "games.csv"
    spine.write_text("game_id,season,type\n"
                     "2024020001,20242025,2\n2024020002,20242025,2\n"
                     "2025020001,20252026,2\n2025020002,20252026,2\n"
                     "2025020003,20252026,2\n2025020004,20252026,2\n"
                     "2025030111,20252026,3\n")
    shifts = tmp_path / "shifts.csv"
    shifts.write_text("game_id,player_id,team,period,start_s,end_s\n"
                      "2024020001,1,TOR,1,0,40\n2024020002,1,TOR,1,0,40\n"
                      "2025020001,1,TOR,1,0,40\n2025020001,2,TOR,1,0,40\n"
                      "2025030111,1,TOR,1,0,40\n")
    return spine, shifts


def test_shift_coverage_counts_regular_season_games_with_shifts(tmp_path):
    spine, shifts = _files(tmp_path)
    cov = SP.shift_coverage([20242025, 20252026], str(shifts), str(spine))
    assert cov == {"20242025": {"with_shifts": 2, "games": 2},
                   "20252026": {"with_shifts": 1, "games": 4}}
    assert SP.shift_coverage([20252026], str(tmp_path / "absent.csv"), str(spine)) is None


def test_ci_checkout_without_a_matching_sidecar_never_claims_a_fresh_fit(tmp_path):
    r, m = tmp_path / "r.json", tmp_path / "m.json"
    r.write_text("{}")
    warned = []
    info = SP.rapm_meta([20212022, 20222023, 20232024, 20242025, 20252026, 20262027],
                        str(r), str(m), cur_season=20262027,
                        shifts_path=str(tmp_path / "absent.csv"), warn=warned.append)
    assert info["window"] == "2022-23 to 2025-26"        # seasons BEFORE the served one
    assert info["built"] is None and info["inferred"] is True and info["coverage"] is None
    assert warned and not m.exists()                     # a guess is never persisted


def test_a_committed_sidecar_is_used_verbatim_in_ci(tmp_path):
    r, m = tmp_path / "r.json", tmp_path / "m.json"
    r.write_bytes(b'{"1": {"rating": 50}}\n')
    m.write_text(json.dumps({"sha256": SP.ratings_sha(str(r)),
                             "fit_seasons": [20222023, 20232024, 20242025, 20252026],
                             "built": "2026-07-31",
                             "coverage": {"20252026": {"with_shifts": 807, "games": 1312}}}))
    info = SP.rapm_meta([20232024, 20242025, 20252026, 20262027], str(r), str(m),
                        cur_season=20262027, shifts_path=str(tmp_path / "absent.csv"))
    assert info["window"] == "2022-23 to 2025-26" and info["built"] == "2026-07-31"
    assert info["coverage"] == {"2025-26": {"with_shifts": 807, "games": 1312, "share": 0.615}}
    assert not info["inferred"]
    r.write_bytes(b'{"1": {"rating": 50}}\r\n')          # CRLF checkout: same file
    assert SP.rapm_meta([20262027], str(r), str(m), cur_season=20262027,
                        shifts_path=str(tmp_path / "absent.csv")) == info


def test_local_refit_records_window_mtime_and_coverage(tmp_path):
    spine, shifts = _files(tmp_path)
    r, m = tmp_path / "r.json", tmp_path / "m.json"
    r.write_text("{}")
    info = SP.rapm_meta([20242025, 20252026], str(r), str(m), cur_season=20262027,
                        shifts_path=str(shifts), spine_path=str(spine), warn=lambda *_: None)
    assert info["window"] == "2024-25 to 2025-26"
    assert info["built"] == datetime.fromtimestamp(r.stat().st_mtime,
                                                   timezone.utc).date().isoformat()
    assert info["coverage"]["2025-26"] == {"with_shifts": 1, "games": 4, "share": 0.25}
    assert json.loads(m.read_text())["coverage"]["20252026"]["games"] == 4


def test_the_committed_sidecar_describes_the_committed_ratings():
    ratings, meta = ROOT / SP.RAPM, ROOT / SP.RAPM_META
    if not ratings.is_file():
        pytest.skip("no ratings")
    assert meta.is_file(), "data/nhl_site_rapm_meta.json must be committed with the ratings"
    m = json.loads(meta.read_text(encoding="utf-8"))
    assert m["sha256"] == SP.ratings_sha(str(ratings))
    assert m["built"] and len(m["fit_seasons"]) == 4 and m.get("coverage")


# ------------------------------------------------------- unrated reasons --
COV = {"2024-25": {"with_shifts": 1255, "games": 1312, "share": 0.957},
       "2025-26": {"with_shifts": 807, "games": 1312, "share": 0.615},
       "2023-24": {"with_shifts": 1312, "games": 1312, "share": 1.0}}


def test_caveat_names_only_incomplete_seasons_the_player_played():
    assert SP.coverage_caveat(COV, {2025}) == "2025-26 shift data covers 807 of 1,312 games"
    assert SP.coverage_caveat(COV, {2023}) is None
    assert SP.coverage_caveat(COV).startswith("2025-26 shift data covers 807 of 1,312 games; "
                                              "2024-25")
    assert SP.coverage_caveat(None) is None


def test_a_full_season_rookie_reads_as_a_data_gap_not_a_small_sample():
    names = {"9": {"name": "Rookie", "pos": "R", "team": "MTL", "on_roster": True},
             "8": {"name": "Callup", "pos": "C", "team": "MTL", "on_roster": True},
             "7": {"name": "Prospect", "pos": "C", "team": "MTL", "on_roster": True}}
    rapm = {"9": {"off": 0.1, "def": 0.0, "net": 0.1, "toi_min": 754.7, "rel": 0.4,
                  "rating": 55.0}}
    stats = {"cur": None, "prev": {"skaters": {"9": {"gp": 82, "toi": 930},
                                               "8": {"gp": 5}}, "goalies": {}}}
    ps = SP.build_players(names, rapm, stats, {}, {}, {}, 2026, date(2026, 9, 24),
                          "2022-23 to 2025-26", None, COV)
    assert ps["9"]["nr"] == ("under 1,000 5v5 minutes in the shift data on file "
                             "(755; 2025-26 shift data covers 807 of 1,312 games)")
    assert ps["8"]["nr"] == ("no 5v5 minutes in the shift data on file "
                             "(2022-23 to 2025-26; 2025-26 shift data covers 807 of 1,312 games)")
    assert ps["7"]["nr"] == "no 5v5 minutes in the shift data on file (2022-23 to 2025-26)"
    assert all("so far" not in (p["nr"] or "") for p in ps.values())
    assert ps["9"]["stats"]["prev"]["toi_pg"] == 930 and "toi" not in ps["9"]["stats"]["prev"]


# --------------------------------------------------------------- units --
def test_season_lines_name_their_ice_time_unit():
    assert SP.norm_line({"gp": 3, "toi": 1100}, "F") == {"gp": 3, "toi_pg": 1100}
    assert SP.norm_line({"gp": 3, "toi": 10800}, "G") == {"gp": 3, "toi_s": 10800}
    assert SP.norm_line({"gp": 3, "toi_pg": 1}, "F") == {"gp": 3, "toi_pg": 1}
    assert SP.norm_line(None, "F") is None
    sk = {"grp": "F", "stats": {"cur": {"gp": 10, "toi_pg": 1000}}, "toi": 5.0}
    gl = {"grp": "G", "stats": {"cur": {"gp": 2, "toi_s": 7000}}}
    assert SP.usage(sk, "cur") == 10 * 1000 + 10 and SP.usage(gl, "cur") == 7000


# ----------------------------------------------------------- team rating --
def test_team_rating_counts_every_lineup_slot():
    ps = {}
    for i in range(12):
        ps[f"f{i}"] = {"id": f"f{i}", "name": f"F{i}", "team": "VAN", "grp": "F",
                       "rating": 70.0 if i < 6 else None, "net": 0.1 if i < 6 else None,
                       "toi": 2000.0, "stats": {"cur": {"gp": 1, "toi_pg": 1000 - i}}}
    for i in range(6):
        ps[f"d{i}"] = {"id": f"d{i}", "name": f"D{i}", "team": "VAN", "grp": "D",
                       "rating": None, "net": None, "toi": 100.0,
                       "stats": {"cur": {"gp": 1, "toi_pg": 900 - i}}}
    rapm = {f"f{i}": {"rating": 40.0} for i in range(6, 9)}   # shrunk, under the floor
    tb = SP.team_blocks(ps, ["VAN"], rapm)["VAN"]
    # 6 rated at 70, 3 from the RAPM file at 40, 3 + 6 D on the prior 50
    assert tb["gb_f"] == round((6 * 70 + 3 * 40 + 3 * 50) / 12, 1)
    assert tb["gb_d"] == 50.0 and tb["n_f"] == 6 and tb["n_d"] == 0
    assert tb["cov"] == round(6 / 18, 3) and tb["fill"] == {"rapm": 3, "prior": 9}
    assert tb["glassbox"] == round((6 * 70 + 3 * 40 + 9 * 50) / 18, 1)


# ------------------------------------------------------------ last season --
def test_test_window_is_2018_19_through_2025_26():
    assert PREV.is_test(20182019) and PREV.is_test(20252026)
    assert not PREV.is_test(20172018) and not PREV.is_test(20262027)


def test_from_published_copies_rows_and_metrics_and_computes_nothing():
    pub = {"cur_season": 20252026, "as_of": "2026-04-16",
           "model_card": {"cur_season_ll": 0.123, "cur_season_acc": 0.9},
           "schedule": [
               {"id": 2, "d": "2025-10-08", "home": "A", "away": "B", "hp": 0.4, "hs": 1,
                "as": 2, "last": "OT", "playoff": 0},
               {"id": 1, "d": "2025-10-07", "home": "B", "away": "A", "hp": 0.6, "hs": 3,
                "as": 2, "last": "REG", "playoff": 0},
               {"id": 3, "d": "2026-04-20", "home": "A", "away": "B", "hp": 0.5, "hs": 1,
                "as": 0, "playoff": 1}]}
    blk = PREV.from_published(pub, "test")
    assert blk["ll"] == 0.123 and blk["acc"] == 0.9     # copied (deliberately "wrong")
    assert blk["n"] == 2 and [r[0] for r in blk["rows"]] == [1, 2]
    assert blk["kind"] == "replay" and "not a live record" in blk["note"]
    ro = PREV.rows_only(20252026, blk["rows"], "2026-04-16")
    assert ro["ll"] is None and ro["acc"] is None and ro["n"] == 2


def test_served_test_season_is_the_frozen_file_verbatim():
    if not PAYLOAD.is_file():
        pytest.skip("nhl.json not built")
    ps = json.loads(PAYLOAD.read_text(encoding="utf-8")).get("prev_season")
    if not ps or not PREV.is_test(ps["season"]):
        pytest.skip("last season is not a TEST season")
    frozen = PREV.load(ps["season"], lambda s: str(ROOT / PREV.path(s)))
    assert frozen is not None, "a TEST season must be served from its frozen file"
    assert ps == frozen


# -------------------------------------------------- schedule and ledger --
def test_playoff_finals_are_kept_as_played_playoff_rows():
    raw = {2026030111: {"type": "3", "season": "20262027", "date": "2027-04-20",
                        "home": "TOR", "away": "MTL", "home_goals": "3", "away_goals": "2",
                        "last_period": "OT"},
           2026020001: {"type": "2", "season": "20262027", "date": "2026-10-07",
                        "home": "TOR", "away": "MTL", "home_goals": "1", "away_goals": "0",
                        "last_period": "REG"},
           2025030111: {"type": "3", "season": "20252026", "date": "2026-04-20",
                        "home": "TOR", "away": "MTL", "home_goals": "1", "away_goals": "0",
                        "last_period": "REG"},
           2026030112: {"type": "3", "season": "20262027", "date": "2027-04-22",
                        "home": "TOR", "away": "XXX", "home_goals": "1", "away_goals": "0",
                        "last_period": "REG"}}
    rows = playoff_finals(raw, 20262027, {2026030111: "2027-04-20T23:00:00Z"},
                          {"TOR": 1, "MTL": 1})
    assert rows == [{"id": 2026030111, "d": "2027-04-20", "home": "TOR", "away": "MTL",
                     "playoff": 1, "t": "2027-04-20T23:00:00Z", "hs": 3, "as": 2,
                     "last": "OT"}]


def test_every_row_with_a_ledger_entry_goes_through_the_freeze():
    led = {"5": {"hp": 0.5}}
    assert ledger_path({"id": 5, "hs": None, "near": 0}, led)        # far, but ledgered
    assert ledger_path({"id": 6, "hs": None, "near": 1}, led)
    assert ledger_path({"id": 7, "hs": 2, "near": 0}, led)
    assert not ledger_path({"id": 8, "hs": None, "near": 0}, led)


def test_payload_hp_equals_ledger_hp_for_every_ledger_key():
    if not (PAYLOAD.is_file() and LEDGER.is_file()):
        pytest.skip("not built")
    n = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    rows = {str(g["id"]): g for g in n["schedule"]}
    for k, ent in led.items():
        if k in rows and int(k[:4]) == n["cur_season"] // 10000:
            assert rows[k]["hp"] == ent["hp"], k


def test_payload_players_and_teams_carry_the_new_contract():
    if not PAYLOAD.is_file():
        pytest.skip("not built")
    n = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    assert "coverage" in n["rapm"] and "caveat" in n["rapm"]
    for p in n["players"].values():
        assert "so far" not in (p["nr"] or "") or p["grp"] == "G"
        for k in ("cur", "prev"):
            ln = p["stats"].get(k)
            if ln:
                assert "toi" not in ln, (p["name"], k)
    for t in n["teams"].values():
        assert {"cov", "fill", "n_f", "n_d"} <= set(t)
