"""NHL player value on the site - the data side (package nhl-player-value-display).

phase0/nhl_site_player_value.py turns the validated player-value program (the
FIXED full run, data/pv_nhl_fix_full_*) into a static snapshot, as of the end of
2025-26, data/nhl_site_pv.json; phase0/nhl_site_players.py merges it into
site/data/nhl.json players[pid].pv and teams[code].pv_lu. Pinned here:

  * the builder's pure pieces: percentiles, the component split that adds up to
    the total, the reference-group centring;
  * the committed snapshot: every skater's components add up to his value, the
    percentiles are in range and computed within F / D, faceoff percentiles only
    for faceoff takers, the label says it is not a model input, no market field;
  * the merge: skaters (never goalies) get their block, a missing or corrupt
    snapshot degrades to no value (CI keeps serving), the lineup value sums the
    12 F + 6 D with the most ice time and ranks the teams (payload only: the pages
    show no lineup rank);
  * the forward test stays valid: no file pinned by
    data/pv_nhl_forward_prereg_2026_27.json was modified (nhl_serve.py included -
    the merge lives in nhl_site_players.py for that reason).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_site_player_value as PV  # noqa: E402
import nhl_site_players as SP  # noqa: E402

SNAP = ROOT / "data" / "nhl_site_pv.json"
PREREG = ROOT / "data" / "pv_nhl_forward_prereg_2026_27.json"
MARKET = {"odds", "dec", "mkt", "value", "book", "books", "ev", "vig", "line", "price"}


# ------------------------------------------------------------ pure helpers --
def test_percentile_counts_half_the_ties_and_stays_in_1_99():
    ref = sorted([0.0, 1.0, 2.0, 3.0])
    assert PV.percentile(1.5, ref) == 50
    assert PV.percentile(1.0, ref) == 38            # 1 below + half of 1 tie = 1.5 / 4
    assert PV.percentile(-9.0, ref) == 1 and PV.percentile(9.0, ref) == 99
    assert PV.percentile(None, ref) is None and PV.percentile(1.0, []) is None
    assert PV.percentile(float("nan"), ref) is None


def test_contributions_add_up_to_the_value_per_60():
    p = {"p_ev_shot": 0.2, "p_ev_fin": 0.05, "p_ev_a1": 0.1, "p_ev_a2": 0.02,
         "p_ev_def": -0.04, "p_pp_xg": 0.6, "p_pp_ast": 0.01, "p_duels": 0.03}
    m_ev, m_pp, m_all, w, fo = 14.0, 3.0, 19.0, 0.6, 0.02
    c = PV.contributions(p, m_ev, m_pp, m_all, w, fo)
    assert set(c) == set(PV.COMPS)
    v_game = (sum(p[k] for k in ("p_ev_shot", "p_ev_fin", "p_ev_a1", "p_ev_a2", "p_ev_def")) * m_ev
              + (p["p_pp_xg"] + p["p_pp_ast"]) * m_pp + p["p_duels"] * m_all) / 60.0
    assert sum(c.values()) == pytest.approx(v_game / m_all * 60.0, abs=1e-12)
    assert c["fo"] == pytest.approx(w * fo) and c["pen"] == pytest.approx(0.03 - w * fo)
    assert c["pp"] == pytest.approx(0.61 * 3.0 / 19.0)


def test_group_means_are_minutes_weighted_within_the_group():
    rows = [{"grp": "F", "m_all": 10.0, "c": {k: 1.0 for k in PV.COMPS}},
            {"grp": "F", "m_all": 30.0, "c": {k: 0.0 for k in PV.COMPS}},
            {"grp": "D", "m_all": 20.0, "c": {k: 2.0 for k in PV.COMPS}}]
    mu = PV.group_means(rows)
    assert mu["F"]["cre"] == pytest.approx(0.25) and mu["D"]["def"] == pytest.approx(2.0)


# ------------------------------------------------------- committed snapshot --
@pytest.fixture(scope="module")
def snap():
    if not SNAP.is_file():
        pytest.skip("data/nhl_site_pv.json not built")
    return json.loads(SNAP.read_text(encoding="utf-8"))


def test_snapshot_meta_says_display_only(snap):
    m = snap["meta"]
    assert m["model_input"] is False
    assert "the game model does not use it yet (forward test on 2026-27 pending)" in m["label"]
    assert m["season"] == 20252026 and m["asof"] == "end of 2025-26"
    assert m["reference"]["n"]["F"] > 300 and m["reference"]["n"]["D"] > 150
    assert set(m["components"]) == set(PV.COMPS)
    # the frozen compose weights the pipeline applied to every TEST season
    frozen = json.loads((ROOT / "data" / PV.SRC_W).read_text())["frozen_test_weights"]["w"] \
        if (ROOT / "data" / PV.SRC_W).is_file() else None
    if frozen:
        assert m["weights"] == {k: round(v, 4) for k, v in frozen.items()}
    # the one display-only change: defence at 1 goal per xG prevented (fit: 1.9008)
    assert m["display_weights"] == {"ev_def": 1.0}


def test_every_skater_block_is_consistent(snap):
    P = snap["players"]
    assert len(P) > 900
    for pid, v in P.items():
        assert v["grp"] in ("F", "D"), pid
        assert set(v["c"]) == set(PV.COMPS) and set(v["q"]) == set(PV.COMPS), pid
        # components add up to the total (3-decimal rounding on 8 parts)
        assert abs(sum(v["c"].values()) - v["v"]) <= 0.006, (pid, v["v"], v["c"])
        # per game = per 60 x usual minutes / 60
        assert abs(v["g"] - v["v"] * v["toi"] / 60.0) <= 0.003, pid
        assert v["p"] is None or 1 <= v["p"] <= 99
        for k, q in v["q"].items():
            assert q is None or 1 <= q <= 99, (pid, k, q)
        # faceoffs are rated only for takers; defence only once he has an on-ice fit
        assert (v["fo"] is None) == (v["q"]["fo"] is None), pid
        assert (v["def"] is None) == (v["q"]["def"] is None), pid
        assert v["s"] >= 20182019 and v["gp"] >= 0 and v["toi"] > 0


def test_percentiles_are_within_position_and_track_the_value(snap):
    """Within F (and D), a higher value never carries a lower percentile, and the
    reference regulars spread over the whole scale."""
    for g in ("F", "D"):
        rows = sorted((v for v in snap["players"].values() if v["grp"] == g and v["p"] is not None),
                      key=lambda v: (v["v"], v["p"]))      # ties: v is rounded to 3 dp
        ps = [v["p"] for v in rows]
        assert all(a <= b for a, b in zip(ps, ps[1:])), g
        assert min(ps) <= 5 and max(ps) >= 95
        reg = [v["p"] for v in rows if v["s"] == 20252026 and v["sgp"] >= 20]
        assert 40 <= sorted(reg)[len(reg) // 2] <= 60, g          # median regular ~ 50th


def test_known_stars_rank_high(snap):
    """Face validity on the snapshot (names from the roster map when present)."""
    names_p = ROOT / "data" / "nhl_player_names.json"
    if not names_p.is_file():
        pytest.skip("names map absent")
    names = json.loads(names_p.read_text(encoding="utf-8"))
    by = {(names.get(pid) or {}).get("name"): v for pid, v in snap["players"].items()}
    for who in ("Connor McDavid", "Nathan MacKinnon", "Cale Makar"):
        if who in by:
            assert by[who]["p"] >= 95, (who, by[who]["p"])


def test_no_market_fields_in_the_snapshot(snap):
    keys = set()
    for v in snap["players"].values():
        keys |= set(v) | set(v["c"]) | set(v["q"])
    assert not {k for k in keys if MARKET & set(k.lower().split("_"))}


# -------------------------------------------------------------------- merge --
def _names():
    return {"1": {"name": "Star C", "pos": "C", "team": "EDM", "on_roster": True},
            "2": {"name": "Rookie W", "pos": "L", "team": "EDM", "on_roster": True},
            "3": {"name": "Top D", "pos": "D", "team": "EDM", "on_roster": True},
            "4": {"name": "Starter G", "pos": "G", "team": "EDM", "on_roster": True},
            "5": {"name": "Other C", "pos": "C", "team": "TOR", "on_roster": True}}


def _pv(v, g, toi=20.0, grp="F"):
    return {"grp": grp, "s": 20252026, "v": v, "g": g, "p": 80, "toi": toi,
            "c": {k: (v if k == "cre" else 0.0) for k in PV.COMPS},
            "q": {k: 50 for k in PV.COMPS}}


def _build(pv):
    return SP.build_players(_names(), {}, {"cur": None, "prev": None}, {}, {}, {}, 2026,
                            date(2026, 9, 24), None, None, pv=pv)


def test_skaters_get_their_block_goalies_never():
    pv = {"1": _pv(0.5, 0.167), "3": _pv(0.2, 0.08, 24.0, "D"), "4": _pv(9.9, 9.9)}
    ps = _build(pv)
    assert ps["1"]["pv"]["v"] == 0.5 and ps["3"]["pv"]["grp"] == "D"
    assert ps["2"]["pv"] is None                    # no NHL game on file
    assert ps["4"]["pv"] is None                    # goalies keep GSAx only
    ps["1"]["pv"]["v"] = 0.0                        # a copy, not the snapshot's dict
    assert pv["1"]["v"] == 0.5


def test_lineup_value_sums_the_lineup_and_ranks_teams():
    ps = _build({"1": _pv(0.5, 0.167), "3": _pv(0.2, 0.08, 24.0, "D"), "5": _pv(-0.1, -0.03)})
    tb = SP.team_blocks(ps, ["EDM", "TOR", "MTL"])
    lu = tb["EDM"]["pv_lu"]
    assert lu["tot"] == pytest.approx(0.247) and lu["f"] == pytest.approx(0.167)
    assert lu["d"] == pytest.approx(0.08)
    assert lu["n"] == 2 and lu["slots"] == 3        # the rookie counts as average
    assert lu["k"]["cre"] == pytest.approx(round(0.5 * 20 / 60 + 0.2 * 24 / 60, 3))
    assert lu["top"] == ["1", "3"] and lu["rk"] == 1
    assert tb["TOR"]["pv_lu"]["rk"] == 2
    assert tb["MTL"]["pv_lu"] is None               # nobody on the roster


def test_a_missing_or_corrupt_snapshot_degrades_to_no_value(tmp_path):
    assert SP.load_pv(str(tmp_path / "absent.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert SP.load_pv(str(bad)) == {}
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps({"players": {"1": {"v": None}, "2": {"v": 0.1}}}))
    assert list(SP.load_pv(str(odd))) == ["2"]
    ps = _build({})
    assert all(p["pv"] is None for p in ps.values())
    tb = SP.team_blocks(ps, ["EDM"])["EDM"]
    assert tb["pv_lu"]["n"] == 0 and "rk" not in tb["pv_lu"]


def test_a_missing_snapshot_is_logged_not_silent(tmp_path):
    """CI only reads the snapshot (its parquet inputs are local-only), so a serve
    without it must say so in the log instead of quietly dropping the feature."""
    for path, why in ((tmp_path / "absent.json", "is missing"),
                      (tmp_path / "bad.json", "is unreadable"),
                      (tmp_path / "empty.json", "has no valued skater")):
        if path.name == "bad.json":
            path.write_text("{not json")
        elif path.name == "empty.json":
            path.write_text(json.dumps({"players": {"1": {"v": None}}}))
        said = []
        assert SP.load_pv(str(path), warn=said.append) == {}
        assert len(said) == 1 and why in said[0] and "WARNING" in said[0], said
        assert "data/nhl_site_pv.json" in said[0]
    said = []
    assert SP.load_pv(str(SNAP), warn=said.append) and said == []   # the real one: quiet


def test_the_snapshot_states_the_game_level_result_as_an_improvement():
    """Ledger row 11's delta is baseline minus new: +0.00062 is a (n.s.) gain."""
    gl = json.loads(SNAP.read_text(encoding="utf-8"))["meta"]["game_level"]
    assert "improved by 0.00062" in gl and "0.66418 -> 0.66356" in gl and "n.s." in gl
    assert "+0.00062 log loss" not in gl


def test_served_payload_carries_the_values():
    p = ROOT / "site" / "data" / "nhl.json"
    if not p.is_file() or not SNAP.is_file():
        pytest.skip("payload or snapshot not built")
    n = json.loads(p.read_text(encoding="utf-8"))
    snap = json.loads(SNAP.read_text(encoding="utf-8"))["players"]
    sk = [q for q in n["players"].values() if q["grp"] != "G"]
    have = [q for q in sk if q.get("pv")]
    assert len(have) >= 0.7 * len(sk)          # rookies without an NHL game have none
    for q in have:
        assert q["pv"] == snap[q["id"]]
    assert all(not q.get("pv") for q in n["players"].values() if q["grp"] == "G")
    assert sum(1 for t in n["teams"].values() if (t.get("pv_lu") or {}).get("n")) == len(n["teams"])


# ---------------------------------------------------- forward-test hygiene --
def _matches(path: Path, want: str) -> bool:
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return want in {hashlib.sha256(b).hexdigest() for b in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def test_no_file_pinned_by_the_forward_test_was_modified():
    if not PREREG.is_file():
        pytest.skip("forward-test pre-registration absent")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    for rel, want in pre["frozen_code_sha256"].items():
        f = ROOT / rel.replace("\\", "/")
        assert f.is_file(), rel
        assert _matches(f, want), f"{rel} changed: the 2026-27 forward test would be void"
    # this package's own files are not pinned (the merge hook lives outside the pins)
    pinned = {r.replace("\\", "/") for r in pre["frozen_code_sha256"]}
    for own in ("phase0/nhl_site_player_value.py", "phase0/nhl_site_players.py"):
        assert own not in pinned


# ------------------------------------------------ builder on a synthetic run --
W9 = {"ev_shot": 1.1, "ev_fin": 0.6, "ev_a1": 1.2, "ev_a2": 0.5, "ev_def": 1.9,
      "pp_xg": 0.9, "pp_ast": 0.07, "duels": 0.6, "goalie": 0.9}


def _frame():
    """Four skaters as the fixed full run stores them: pre-game rows, values from
    the frozen weights, playoff rows with no on-ice defensive value (NaN)."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    import pv_nhl_compose_core as C
    rows = []

    def row(pid, grp, gid, date, season, gtype, d_xga, shot, n):
        rows.append({"gid": gid, "pid": pid, "date": date, "season": season, "gtype": gtype,
                     "team": "EDM", "side": "H", "grp": grp, "pos": "C" if grp == "F" else "D",
                     "m_ev": 14.0, "m_pp": 2.0, "m_all": 18.0,
                     "x_ev_shot": shot, "x_ev_fin": 0.01, "x_ev_a1": 0.02, "x_ev_a2": 0.0,
                     "x_ev_def": 0.0 if d_xga != d_xga else d_xga, "x_pp_xg": 0.1,
                     "x_pp_ast": 0.0, "x_duels": 0.03,
                     "n_prior_games": n, "ixg_ev_hat": 0.7, "a1_ev_hat": 0.5, "a2_ev_hat": 0.3,
                     "fin_mu": 0.05, "fin_sd": 0.1, "fo_g60": 0.01 if grp == "F" else 0.0,
                     "pen_g60": 0.02, "d_xga": d_xga, "fo_pavg": 0.53,
                     "fo_n60": 25.0 if grp == "F" else 0.0,
                     "pen_draw_n60": 0.8, "pen_take_n60": 0.5})
    row(1, "F", 2025020001, "2025-10-08", 20252026, 2, 0.10, 0.20, 400)
    row(1, "F", 2025020100, "2026-04-10", 20252026, 2, 0.20, 0.25, 481)
    row(1, "F", 2025030111, "2026-04-20", 20252026, 3, float("nan"), 0.30, 482)
    row(2, "D", 2025020002, "2025-10-08", 20252026, 2, -0.05, -0.10, 90)
    row(3, "F", 2019020001, "2019-10-05", 20192020, 2, 0.00, 0.00, 10)
    row(4, "F", 2016020001, "2016-10-12", 20162017, 2, 0.00, 0.00, 10)
    df = pd.DataFrame(rows)
    x = pd.DataFrame({c: df["x_" + c] for c in C.SK_COMPS})
    m = pd.DataFrame({"ev": df.m_ev, "pp": df.m_pp, "all": df.m_all})
    v = C.player_values(df[["season"]], x, m, {s: {"w": W9} for s in set(df.season)})
    for c in v.columns:
        df[c] = v[c].to_numpy()
    assert np.isfinite(df.v_all60).all()
    return df


def _synthetic_build(monkeypatch, names):
    monkeypatch.setattr(PV, "REF_MIN_GP", 1)
    weights = {"frozen_test_weights": {"w": W9, "applied_to": [20182019, 20252026]}}
    return PV.build(df=_frame(), weights=weights, names=names, built="2026-09-24")


def test_builder_takes_the_latest_state_and_carries_defence_over_playoffs(monkeypatch):
    out = _synthetic_build(monkeypatch, names={})
    P = out["players"]
    # DEV-era last games never appear; an old last game only for a names-map player
    assert set(P) == {"1", "2"}
    one = P["1"]
    assert one["d"] == "2026-04-20" and one["po"] == 1 and one["gp"] == 482
    assert one["sgp"] == 2                               # regular-season games that season
    assert one["def"] == 0.2                             # carried from his last regular-season row
    assert out["meta"]["defence_carried"] == 1
    for v in P.values():
        assert abs(sum(v["c"].values()) - v["v"]) <= 0.006
    assert out["meta"]["model_input"] is False and out["meta"]["sources"] == {}


def test_builder_keeps_older_players_who_are_on_the_names_map(monkeypatch):
    out = _synthetic_build(monkeypatch, names={"3": {"name": "Returning F"}, "4": {}})
    assert set(out["players"]) == {"1", "2", "3"}        # 4's last game is DEV-era
    assert out["players"]["3"]["s"] == 20192020
