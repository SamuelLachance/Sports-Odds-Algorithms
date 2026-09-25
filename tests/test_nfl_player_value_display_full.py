"""The pv_nfl_full_* wrappers extend the NFL player-value components past DEV with
FROZEN DEV hyper-parameters. Pinned here:

  - run on DEV seasons, each wrapper reproduces the DEV component output it extends
    (so the extension is the same code with the same settings, not a re-fit):
      passing  data/pv_nfl_passing_values.csv            (to the csv's 6 digits)
      front    data/pv_nfl_pass_rush_and_front_player_games.parquet   (exactly)
      coverage data/pv_nfl_coverage_player_games.parquet (exactly)
      rec/rush data/pv_nfl_receiving_rushing_events.parquet (exactly)
  - the frozen settings are read from the DEV outputs (through = 2015);
  - the current-state tables hold values and sample counts only (no realised yards,
    EPA, sacks or success), and the snapshot's recorded source hashes match the
    component files on disk (built from unmodified code).

Skipped where the local pv inputs are absent (they are gitignored; CI has none).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "phase0"))

EVENTS = DATA / "pv_nfl_events.parquet"
needs_events = pytest.mark.skipif(not EVENTS.is_file(), reason="pv events table absent (local-only input)")


@pytest.fixture(autouse=True)
def _no_provenance(monkeypatch):
    import pv_nfl_full_common as C
    monkeypatch.setattr(C, "RECORD", False)


def _need(*names):
    miss = [n for n in names if not (DATA / n).is_file()]
    if miss:
        pytest.skip(f"absent: {miss}")


@needs_events
def test_passing_reproduces_dev(monkeypatch):
    pytest.importorskip("numba")
    _need("pv_nfl_passing_values.csv", "pv_nfl_passing_hyper.json")
    monkeypatch.chdir(ROOT)
    import pv_nfl_full_common as C
    import pv_nfl_full_passing as W
    B = C.import_component("pv_nfl_passing_build")
    d = W.verify_dev(B, C.read_json("pv_nfl_passing_hyper.json"))
    assert max(d.values()) < 2e-5, d


@needs_events
def test_front_reproduces_dev():
    _need("pv_nfl_pass_rush_and_front_credits.parquet", "pv_nfl_pass_rush_and_front_player_games.parquet")
    import pv_nfl_full_pass_rush as W
    d = W.verify_dev()
    assert max(d.values()) == 0.0, d


@needs_events
def test_coverage_reproduces_dev(monkeypatch):
    _need("pv_nfl_coverage_player_games.parquet", "pv_nfl_coverage_priors.json")
    monkeypatch.chdir(ROOT)
    import pv_nfl_full_common as C
    import pv_nfl_full_coverage as W
    d = W.verify_dev(C.import_component("pv_nfl_coverage"))
    assert max(d.values()) == 0.0, d


def test_receiving_rushing_walk_reproduces_dev():
    _need("pv_nfl_receiving_rushing_plays.parquet", "pv_nfl_receiving_rushing_events.parquet",
          "pv_nfl_receiving_rushing_params.json")
    import pv_nfl_full_receiving_rushing as W
    d = W.stage_walk(2026, verify_dev=True)
    assert max(d.values()) == 0.0, d


def test_frozen_settings_come_from_dev_runs():
    _need("pv_nfl_receiving_rushing_params.json", "pv_nfl_passing_hyper.json", "pv_nfl_coverage_priors.json")
    rr = json.loads((DATA / "pv_nfl_receiving_rushing_params.json").read_text())
    assert rr["through"] == 2015
    hy = json.loads((DATA / "pv_nfl_passing_hyper.json").read_text())
    w = hy["composite_weights_by_season"]["2015"]
    assert w["train"] == "2007-2014"                    # the last DEV fold, frozen
    pri = json.loads((DATA / "pv_nfl_coverage_priors.json").read_text())
    assert "warm-up seasons 1999-2005" in pri["note"]


def test_ast_extract_is_verbatim():
    import pv_nfl_full_common as C
    src, _ = C.read_source("pv_nfl_receiving_rushing_build")
    code = C.ast_extract(src, ["GaussTS", "prev_league_mean"])
    assert code.startswith("def prev_league_mean") or code.startswith("class GaussTS")
    for block in code.split("\n\n\n"):
        assert block.strip() in src
    with pytest.raises(AssertionError):
        C.ast_extract(src, ["no_such_function"])


CURRENT = {
    "pv_nfl_full_passing_current.parquet": {"qb_id", "team", "game_date", "epa_q", "epa_sd", "succ_q", "succ_sd",
                                            "sack_q", "sack_sd", "cpoe_q", "cpoe_sd", "passing_composite",
                                            "db_cur", "db_prev", "db_career", "last_season"},
    "pv_nfl_full_rr_current.parquet": None,           # checked by prefix below
    "pv_nfl_full_front_current.parquet": {"player_id", "team", "bucket", "th_pr", "th_sk", "th_ht", "th_st", "th_tf",
                                          "neff_pr", "neff_st", "n_games", "L_pr", "L_sk", "L_ht", "L_st", "L_tf",
                                          "g_cur", "db_cur", "run_cur", "g_prev", "db_prev", "run_prev",
                                          "g_career", "name"},
    "pv_nfl_full_coverage_current.parquet": {"player_id", "group", "team", "th_pd", "th_int", "th_ball", "th_tc",
                                             "th_cpen", "cv", "epap", "dsp", "at_rate", "cv_eff", "at_n", "games",
                                             "mu_pd", "mu_int", "g_cur", "att_cur", "g_prev", "att_prev"},
}


@pytest.mark.parametrize("name", sorted(CURRENT))
def test_current_tables_hold_values_and_counts_only(name):
    pd = pytest.importorskip("pandas")
    _need(name)
    cols = set(pd.read_parquet(DATA / name).columns)
    if CURRENT[name] is not None:
        assert cols == CURRENT[name], cols ^ CURRENT[name]
    else:
        ok = ("player_id", "team", "last_season", "mu_", "sd_", "n_", "pre_", "games", "tgt_", "car_", "v_e", "v_t")
        assert all(c.startswith(ok) for c in cols), [c for c in cols if not c.startswith(ok)]
    for bad in ("y_", "epa_sum", "yards", "sacks", "success", "pts"):
        assert not any(c.startswith(bad) for c in cols), bad


def test_source_hash_ignores_line_endings(tmp_path):
    """Provenance identifies the committed code on any checkout: a CRLF working copy
    (Windows, core.autocrlf=true) and an LF one (Linux / cloud) hash the same, equal
    to sha256 of the LF content; a real edit still changes the hash."""
    import hashlib
    import pv_nfl_full_common as C
    body = "x = 1\ny = 2\n"
    lf, crlf, edit = tmp_path / "lf.py", tmp_path / "crlf.py", tmp_path / "edit.py"
    lf.write_bytes(body.encode())
    crlf.write_bytes(body.replace("\n", "\r\n").encode())
    edit.write_bytes(body.replace("2", "3").encode())
    h = C.sha256_file(str(lf))
    assert h == C.sha256_file(str(crlf)) == hashlib.sha256(body.encode()).hexdigest()
    assert C.sha256_file(str(edit)) != h


def test_snapshot_built_from_unmodified_component_sources():
    snap = DATA / "nfl_site_pv.json"
    if not snap.is_file():
        pytest.skip("data/nfl_site_pv.json not built")
    import pv_nfl_full_common as C
    src = json.loads(snap.read_text(encoding="utf-8"))["sources"]
    expect = {"pv_nfl_passing_build.py", "pv_nfl_receiving_rushing_build.py", "pv_nfl_receiving_rushing_xmodels.py",
              "pv_nfl_pass_rush_and_front.py", "pv_nfl_pass_rush_and_front_credits.py", "pv_nfl_coverage.py"}
    assert expect <= set(src)
    for f in expect:
        assert C.sha256_file(C.src_path(f)) == src[f], f"{f} changed since the snapshot was built"
