"""NFL individual player measures (player-value program) on the site: the chain's
merge (phase0/nfl_site_player_value.merge), the committed snapshot
(data/nfl_site_pv.json) and the hook in phase0/nfl_site_db.py.

What is pinned:
  - the merge is DISPLAY ONLY: it adds players[].pv / teams[].pv / pv_meta and
    touches nothing else (no rating, forecast, schedule or ledger field);
  - it is CI-safe: the module imports only the standard library at import time
    (nfl-weekly.yml runs the chain with numpy/pandas/pyarrow/scipy/sklearn only,
    and the pv inputs - numba, lightgbm, the 100 MB events table - are local);
  - a player gets only the blocks that describe his position family; offensive
    linemen and specialists get none;
  - percentiles rank each measure against the QUALIFIED players of the same
    family, and too small a pool gives no percentile;
  - team unit numbers are ranked 1 = best (fewer allowed on offence, more on defence);
  - the snapshot carries values and sample counts only: no key that could hold an
    outcome metric, and it says the game model does not use it;
  - a missing snapshot leaves no stale pv behind.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))
import nfl_site_player_value as PVS  # noqa: E402

SNAP = ROOT / "data" / "nfl_site_pv.json"
USE = "display rating; the game model does not use it (it did not improve game predictions)"


def _snap(n_qb=10):
    players = {}
    for i in range(n_qb):
        players[f"QB{i}"] = {"qb": {"v": round(0.01 * i, 3), "epa": 0.0, "cpoe": 0.0, "sack": 0.0,
                                    "db": 100, "dbp": 400, "dbc": 900}}
    players["QBsmall"] = {"qb": {"v": 0.5, "db": 5, "dbp": 10, "dbc": 15}}
    players["RB1"] = {"rush": {"v": 0.01, "c": 100, "cp": 100}, "rec": {"v": 0.02, "t": 10, "tp": 10},
                      "qb": {"v": 0.9, "db": 1, "dbp": 0}}
    players["OL1"] = {"front": {"pr": 1.0, "st": 1.0, "g": 17, "gp": 17}}
    players["DL1"] = {"front": {"pr": 5.0, "st": 3.0, "g": 17, "gp": 17}, "cov": {"ball": 1.0, "ept": 0.1, "g": 2}}
    # linebackers: coverage blocks on the LB scale, plus an edge rusher whose coverage
    # block is on the DL scale (the coverage engine's group comes from nfl_players.csv)
    for i in range(PVS.MIN_POOL):
        players[f"LB{i}"] = {"front": {"pr": 1.0 + i, "st": 5.0, "g": 17, "gp": 17},
                             "cov": {"ball": 0.8 + 0.1 * i, "x": 1.0, "ept": 0.01 * i, "g": 17, "gp": 17, "grp": "LB"}}
    players["EDGE1"] = {"front": {"pr": 9.0, "st": 4.0, "g": 17, "gp": 17},
                        "cov": {"ball": 0.28, "x": 1.0, "ept": 0.0, "g": 17, "gp": 17, "grp": "DL"}}
    players["S1"] = {"cov": {"ball": 2.0, "x": 1.2, "ept": 0.05, "g": 17, "gp": 17, "grp": "LB"}}
    teams = {"AAA": {k: 1.0 for k in PVS.UNIT_KEYS}, "BBB": {k: 1.0 for k in PVS.UNIT_KEYS}}
    teams["AAA"].update(O_sk=0.8, D_pr=1.3)
    teams["BBB"].update(O_sk=1.2, D_pr=0.7)
    return {"players": players, "teams": teams, "through": {"season": 2026, "week": 2},
            "built": "x", "frozen": "y", "league": {"pr_per_100_db": 15.9}}


def _payload(n_qb=10):
    P = {f"QB{i}": {"id": f"QB{i}", "fam": "QB", "rating": {"r": 50}} for i in range(n_qb)}
    P["QBsmall"] = {"id": "QBsmall", "fam": "QB", "rating": None}
    P["RB1"] = {"id": "RB1", "fam": "RB"}
    P["OL1"] = {"id": "OL1", "fam": "OL"}
    P["DL1"] = {"id": "DL1", "fam": "DL"}
    P["K1"] = {"id": "K1", "fam": "K"}
    for i in range(PVS.MIN_POOL):
        P[f"LB{i}"] = {"id": f"LB{i}", "fam": "LB"}
    P["EDGE1"] = {"id": "EDGE1", "fam": "LB"}
    P["S1"] = {"id": "S1", "fam": "DB"}
    return {"players": P, "teams": {"AAA": {"code": "AAA", "elo": 1500}, "BBB": {"code": "BBB"}},
            "schedule": [{"w": 1, "ph": 0.61}], "model_card": {"x": 1}}


def test_merge_adds_only_display_fields():
    pay = _payload()
    before = copy.deepcopy(pay)
    n = PVS.merge(pay, _snap())
    assert n == 13 + PVS.MIN_POOL + 1               # 11 QBs + RB + DL + LBs + edge; OL / K get nothing,
    #                                                 nor S1 (his only block is on another group's scale)
    for pid, p in pay["players"].items():
        q = {k: v for k, v in p.items() if k != "pv"}
        assert q == before["players"][pid], pid     # nothing but pv changed on a player
    for c, t in pay["teams"].items():
        assert {k: v for k, v in t.items() if k != "pv"} == before["teams"][c]
    assert pay["schedule"] == before["schedule"] and pay["model_card"] == before["model_card"]
    assert set(pay) - set(before) == {"pv_meta"}
    assert pay["pv_meta"]["model_use"] == USE == PVS.MODEL_USE


def test_blocks_follow_the_position_family():
    pay = _payload()
    PVS.merge(pay, _snap())
    P = pay["players"]
    assert set(P["RB1"]["pv"]) == {"rush", "rec"}        # the RB's stray qb block is not shown
    assert set(P["DL1"]["pv"]) == {"front"}              # a lineman's coverage block is not shown
    assert "pv" not in P["OL1"] and "pv" not in P["K1"]  # no individual blocking / specialist number


def test_coverage_block_only_on_its_own_group_scale():
    """An LB-family edge rusher whose coverage block is on the DL scale (ball
    production at the DL prior) gets no coverage block: it is neither shown nor
    ranked, and it does not enter the linebackers' pools. Same for a DB-family
    player rated in the LB group. The group key itself is not copied to the payload."""
    pay = _payload()
    PVS.merge(pay, _snap())
    P = pay["players"]
    assert set(P["EDGE1"]["pv"]) == {"front"}             # pass rush kept, DL-scale coverage dropped
    assert "pv" not in P["S1"]                            # only block was on another scale
    lb = P["LB0"]["pv"]["cov"]
    assert "grp" not in lb and lb["q"] == 1
    assert pay["pv_meta"]["pool"]["LB|cov|ball"] == PVS.MIN_POOL   # the edge rusher is not in the pool
    assert pay["pv_meta"]["pool"]["LB|front|pr"] == PVS.MIN_POOL + 1
    assert P["LB0"]["pv"]["cov"]["p_ball"] == PVS.percentile(0.8, [0.8 + 0.1 * i for i in range(PVS.MIN_POOL)])
    # a block without a recorded group (an older snapshot) is kept as before
    snap = _snap()
    del snap["players"]["EDGE1"]["cov"]["grp"]
    pay2 = _payload()
    PVS.merge(pay2, snap)
    assert "cov" in pay2["players"]["EDGE1"]["pv"]


def test_percentiles_rank_against_qualified_peers():
    pay = _payload()
    PVS.merge(pay, _snap())
    P = pay["players"]
    pool = [0.01 * i for i in range(10)]
    assert P["QB9"]["pv"]["qb"]["p_v"] == PVS.percentile(0.09, pool) == 95
    assert P["QB0"]["pv"]["qb"]["p_v"] == 5
    small = P["QBsmall"]["pv"]["qb"]
    assert small["q"] == 0 and small["p_v"] == 99        # ranked against the qualified pool (capped), flagged
    assert PVS.percentile(-1.0, pool) == 1 and PVS.percentile(1.0, pool) == 99
    assert P["QB3"]["pv"]["qb"]["q"] == 1
    assert pay["pv_meta"]["pool"]["QB|qb|v"] == 10
    # a pool below MIN_POOL: no percentile at all
    pay2 = _payload(n_qb=PVS.MIN_POOL - 1)
    PVS.merge(pay2, _snap(n_qb=PVS.MIN_POOL - 1))
    assert "p_v" not in pay2["players"]["QB0"]["pv"]["qb"]


def test_team_units_rank_one_is_best():
    pay = _payload()
    PVS.merge(pay, _snap())
    a, b = pay["teams"]["AAA"]["pv"], pay["teams"]["BBB"]["pv"]
    assert a["rk"]["O_sk"] == 1 and b["rk"]["O_sk"] == 2  # fewer sacks allowed = better
    assert a["rk"]["D_pr"] == 1 and b["rk"]["D_pr"] == 2  # more pressure = better
    assert a["O_sk"] == 0.8


def test_merge_is_idempotent_and_clears_without_snapshot(capsys):
    pay = _payload()
    PVS.merge(pay, _snap())
    once = json.dumps(pay, sort_keys=True)
    PVS.merge(pay, _snap())
    assert json.dumps(pay, sort_keys=True) == once
    missing = PVS.load_snapshot(str(ROOT / "data" / "no_such_snapshot.json"))
    assert missing is None and "no player-value snapshot" in capsys.readouterr().out
    PVS.merge(pay, missing)
    assert "pv_meta" not in pay
    assert not any("pv" in p for p in pay["players"].values())
    assert not any("pv" in t for t in pay["teams"].values())


def test_module_is_ci_safe():
    """Top-level imports are stdlib only; heavy imports live inside build()."""
    tree = ast.parse((ROOT / "phase0" / "nfl_site_player_value.py").read_text(encoding="utf-8"))
    mods = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "json", "os", "datetime"}, mods
    # and it imports in a bare interpreter with numpy/pandas unavailable
    code = ("import sys; sys.modules['numpy']=None; sys.modules['pandas']=None; sys.modules['pyarrow']=None;"
            "sys.path.insert(0,'phase0'); import nfl_site_player_value as m; print(m.merge({'players':{},'teams':{}}, None))")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith("0"), r.stderr


def test_nfl_site_db_calls_the_merge_non_fatally():
    src = (ROOT / "phase0" / "nfl_site_db.py").read_text(encoding="utf-8")
    i = src.index("import nfl_site_player_value as PVS")
    hook = src[src.rindex("try:", 0, i):src.index("NP.dump_atomic(payload, PAYLOAD", i)]
    assert "PVS.merge(payload)" in hook and "except Exception" in hook
    assert src.index("PVS.merge(payload)") < src.index("NP.dump_atomic(payload, PAYLOAD")


@pytest.mark.skipif(not SNAP.is_file(), reason="data/nfl_site_pv.json not built")
def test_snapshot_is_values_only_and_labelled():
    s = json.loads(SNAP.read_text(encoding="utf-8"))
    assert s["model_use"] == USE
    assert "frozen" in s and "DEV" in s["frozen"]
    assert set(s["through"]) == {"season", "week"}
    allowed = {"qb": {"v", "epa", "cpoe", "sack", "db", "dbp", "dbc"},
               "rec": {"v", "xy", "ye", "ts", "t", "tp", "tc"},
               "rush": {"v", "ry", "cs", "c", "cp", "cc"},
               "front": {"pr", "sk", "st", "sh", "g", "gp"},
               "cov": {"ball", "x", "ept", "ar", "g", "gp", "grp"}}
    assert len(s["players"]) > 1000
    for pid, blocks in s["players"].items():
        assert set(blocks) <= set(allowed), pid
        for k, b in blocks.items():
            assert set(b) <= allowed[k], (pid, k, set(b) - allowed[k])
            num = {m: v for m, v in b.items() if m != "grp"}
            assert all(isinstance(v, (int, float)) and v == v for v in num.values()), (pid, k)
        if "cov" in blocks:                                  # the scale the block is on
            assert blocks["cov"].get("grp") in ("DB", "LB", "DL"), pid
    assert len(s["teams"]) == 32
    for c, t in s["teams"].items():
        assert set(t) == set(PVS.UNIT_KEYS) and all(0.3 < v < 3.0 for v in t.values()), c
    # no metric-like key anywhere (a value is never scored against a later outcome)
    blob = json.dumps({k: v for k, v in s.items() if k not in ("players", "teams", "sources")}).lower()
    for bad in ("log_loss", "brier", "yoy", "\"r\":", "auc", "accuracy", "ll\":"):
        assert bad not in blob, bad


@pytest.mark.skipif(not SNAP.is_file(), reason="data/nfl_site_pv.json not built")
def test_nfl_site_db_hook_end_to_end(tmp_path):
    """Run the real chain step on a copy of the live payload (NFL_PAYLOAD): the copy
    gains pv blocks, the live site/data/nfl.json is untouched."""
    live = ROOT / "site" / "data" / "nfl.json"
    if not live.is_file():
        pytest.skip("site/data/nfl.json not built")
    for need in ("data/nfl_games.csv", "data/nfl_players.csv", "data/roster_2026.csv"):
        if not (ROOT / need).is_file():
            pytest.skip(f"{need} absent")
    before = live.read_bytes()
    stage = tmp_path / "nfl.json"
    stage.write_bytes(before)
    env = dict(os.environ, NFL_PAYLOAD=str(stage), PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-X", "utf8", "phase0/nfl_site_db.py"], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "player-value display:" in r.stdout and "SKIPPED" not in r.stdout
    out = json.loads(stage.read_text(encoding="utf-8"))
    assert out["pv_meta"]["model_use"] == USE
    assert sum(1 for p in out["players"].values() if p.get("pv")) > 1000
    assert not any(p.get("pv") for p in out["players"].values() if p["fam"] in ("OL", "K", "P", "LS"))
    assert all(t.get("pv") for t in out["teams"].values())
    assert live.read_bytes() == before


@pytest.mark.skipif(not SNAP.is_file(), reason="data/nfl_site_pv.json not built")
def test_live_merge_shows_coverage_only_on_the_family_scale():
    """On the shipped payload: every linebacker / defensive back shown a coverage
    block was rated in his own coverage group (the edge rushers the coverage engine
    rates as DL - Garrett, Ossai, Fowler... - show pass rush only), so the LB pools
    hold LB-scale values only."""
    live = ROOT / "site" / "data" / "nfl.json"
    if not live.is_file():
        pytest.skip("site/data/nfl.json not built")
    snap = PVS.load_snapshot(str(SNAP))
    pay = json.loads(live.read_text(encoding="utf-8"))
    if not pay.get("players"):
        pytest.skip("payload has no players")
    PVS.merge(pay, snap)
    shown = dropped = 0
    for pid, p in pay["players"].items():
        src = snap["players"].get(pid, {}).get("cov")
        if p.get("fam") not in PVS.COV_GROUP or src is None:
            continue
        cov = (p.get("pv") or {}).get("cov")
        if src.get("grp") == PVS.COV_GROUP[p["fam"]]:
            assert cov is not None and "grp" not in cov, pid
            shown += 1
        else:
            assert cov is None, (pid, p.get("name"), src.get("grp"))
            dropped += 1
    assert shown > 500 and dropped < shown
