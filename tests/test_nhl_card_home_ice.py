"""NHL round-2 fixes (package nhl-fix): a model card whose numbers reconcile,
home ice as its own contribution bar, playoff games on the board, and the edge
layer's fields carried across a re-serve.

  * model card - the gain vs Elo and "Elo alone" come from ONE harness (the
    one TEST look, data/nhl_features_report.json 'all'), so on the card
    Elo alone - test log loss == the gain, exactly as printed. The tuned Elo's
    raw probabilities (another harness, one more game) ship labelled, never as
    the baseline. Read-only: nothing is scored on TEST here.
  * home ice - `ct.home` is the model's edge for the home side between two
    equal teams (the +ha Elo points, the +XG_HA goals and the intercept); the
    five bars still add exactly to the served probability.
  * playoffs - a playoff slate renders on the board (tagged), so the board is
    not empty April-June.
  * carry-forward - a re-serve keeps the edge layer's badges, price feed,
    hold reasons and status line, but drops a pre-gate badge with no tier
    stamp on an EARLY row (never shown, refused by the edge ledger).
"""
from __future__ import annotations

import copy
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_contributions as C  # noqa: E402
import nhl_serve as S  # noqa: E402

DATA = ROOT / "site" / "data"
NHL_JSON = DATA / "nhl.json"
MODEL = ROOT / "data" / "nhl_model.json"
FEAT = ROOT / "data" / "nhl_features_report.json"

COEFS = {"elo_logit": 0.70016, "rest": -0.0033, "b2b_home": -0.2839,
         "b2b_away": 0.2726, "xg": 0.37079}
INTERCEPT = -0.02695


def _sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def _served():
    if not NHL_JSON.is_file():
        pytest.skip("site/data/nhl.json not built")
    return json.loads(NHL_JSON.read_text(encoding="utf-8"))


# ---- model card: one harness ------------------------------------------------

def test_card_reconciles_on_the_published_report():
    if not (MODEL.is_file() and FEAT.is_file()):
        pytest.skip("model / features report missing")
    model = json.loads(MODEL.read_text())
    rep = json.loads(FEAT.read_text())
    card = S.test_card(model, rep)
    a = rep["all"]
    assert card["test_ll"] == a["with"] and card["test_delta_vs_elo"] == a["delta"]
    assert card["baseline_elo_test"] == a["base"], "the baseline must be the same-harness one"
    assert abs(card["baseline_elo_test"] - card["test_ll"] - card["test_delta_vs_elo"]) < 1e-9
    assert card["test_ci"] == a["ci"] and card["test_n"] == a["n"]
    # the other harness's number is shipped under its own name, never as the baseline
    assert card["elo_raw_test"] == model["baseline_elo_test"] != card["baseline_elo_test"]
    assert card["test_seasons"] == "2018-19 to 2025-26"


def test_card_without_the_report_still_reconciles():
    model = {"test_ll": 0.66418, "test_delta_vs_elo": 0.00535, "baseline_elo_test": 0.66918,
             "test_ci": [0.00353, 0.00721]}
    card = S.test_card(model)
    assert card["baseline_elo_test"] == 0.66953 and card["test_n"] is None
    assert card["test_ci"] == [0.00353, 0.00721]
    # a report from another run (different `with`) is not mixed in
    card = S.test_card(model, {"all": {"with": 0.6, "base": 0.7, "n": 5, "ci": [0, 1]}})
    assert card["baseline_elo_test"] == 0.66953 and card["test_n"] is None


def test_card_refuses_numbers_that_do_not_add_up():
    model = {"test_ll": 0.66418, "test_delta_vs_elo": 0.00535}
    with pytest.raises(ValueError):
        S.test_card(model, {"all": {"with": 0.66418, "base": 0.66918, "n": 1}})


def test_served_card_adds_up():
    mc = _served()["model_card"]
    assert abs(mc["baseline_elo_test"] - mc["test_ll"] - mc["test_delta_vs_elo"]) < 1e-9
    assert mc["test_n"] and mc["test_ci"] and mc["test_seasons"] and mc["baseline_def"]
    assert mc["home_ice"] == {"elo": 30, "xg": 0.15}


def test_train_count_is_the_warm_xg_covered_history_through_as_of():
    games = [{"season": 20102011, "date": "2010-11-01"},   # before the warm start
             {"season": 20112012, "date": "2011-11-01"},
             {"season": 20112012, "date": "2011-11-02"},   # no xG
             {"season": 20252026, "date": "2026-04-16"},
             {"season": 20262027, "date": "2026-10-10"}]   # after the model's as_of
    F = {"xg_diff": np.array([0.1, 0.1, np.nan, 0.2, 0.3])}
    assert S.train_count(games, F, "2026-04-16") == 2
    assert S.train_count(games, F, None) == 3


# ---- home ice, its own bar ----------------------------------------------------

def test_home_elogit_is_the_elo_logit_of_the_home_points():
    p = 1.0 / (1.0 + 10 ** (-30 / 400.0))
    assert abs(C.home_elogit(30) - math.log(p / (1 - p))) < 1e-12


def test_equal_teams_leave_only_home_ice():
    he = C.home_elogit(30)
    ct = C.contributions(INTERCEPT, COEFS, he, 0.0, 0.0, 0.0, 0.15,
                         home_elogit=he, home_xg=0.15)
    want = round(_sig(INTERCEPT + COEFS["elo_logit"] * he + COEFS["xg"] * 0.15) * 100 - 50, 1)
    assert ct == {"home": want, "elo": 0.0, "rest": 0.0, "b2b": 0.0, "xg": 0.0}
    assert 3.0 < ct["home"] < 4.5          # about the 54.1% home win rate


def test_the_bars_add_exactly_to_the_rounded_probability():
    he = C.home_elogit(30)
    for d_elo in (-2.0, -0.4, 0.0, 0.3, 1.9):
        for rest in (-4.0, 0.0, 2.0):
            for hb, ab in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for d_xg in (-1.1, 0.0, 0.8):
                    kw = dict(intercept=INTERCEPT, coefs=COEFS, elogit=he + d_elo,
                              rest_diff=rest, b2b_home=hb, b2b_away=ab, xg_diff=0.15 + d_xg)
                    ct = C.contributions(**kw, home_elogit=he, home_xg=0.15)
                    assert list(ct) == list(C.KEYS)
                    p = C.served_probability(**kw)
                    assert abs(50 + sum(ct.values()) - round(p * 100, 1)) < 1e-9, ct


def test_every_served_row_carries_home_ice_and_it_is_one_number():
    n = _served()
    if not MODEL.is_file():
        pytest.skip("model missing")
    b = json.loads(MODEL.read_text())["blend"]
    c = b["coefs"]
    want = round(_sig(b["intercept"] + c["elo_logit"] * C.home_elogit(30) + c["xg"] * 0.15)
                 * 100 - 50, 1)
    rows = [g for g in n["schedule"] if g["hs"] is None]
    assert rows
    assert all(set(g["ct"]) == set(C.KEYS) for g in rows)
    assert {g["ct"]["home"] for g in rows} == {want}


# ---- the edge layer's fields across a re-serve ---------------------------------

def test_carry_edge_keeps_the_edge_run_and_drops_pre_gate_badges():
    val = {"side": "home", "team": "CAR", "ev_cur": 0.2, "available": True}
    old = {"schedule": [
        {"id": 1, "value": dict(val), "mkt": {"home_dec": 1.8}},                 # pre-gate
        {"id": 2, "value": dict(val, tier="EARLY", note="n"), "mkt": {"home_dec": 1.9}},
        {"id": 3, "mkt": {"home_dec": 2.0}, "value_blocked": "EARLY tier: x"},
        {"id": 4, "value": dict(val, tier="EARLY")},                             # now played
        {"id": 5, "value": dict(val)},                                           # PROJECTED row
    ]}
    sched = [{"id": 1, "hs": None, "tier": "EARLY"}, {"id": 2, "hs": None, "tier": "EARLY"},
             {"id": 3, "hs": None, "tier": "EARLY"}, {"id": 4, "hs": 3, "tier": "EARLY"},
             {"id": 5, "hs": None, "tier": "PROJECTED"}]
    dropped = S.carry_edge(sched, old)
    r = {s["id"]: s for s in sched}
    assert dropped == 1
    assert "value" not in r[1] and r[1]["mkt"] == {"home_dec": 1.8}     # price feed still carried
    assert r[2]["value"]["tier"] == "EARLY" and r[2]["mkt"]["home_dec"] == 1.9
    assert r[3]["value_blocked"] == "EARLY tier: x"
    assert "value" not in r[4]                                          # played rows never
    assert r[5]["value"] == val                                         # not an EARLY row


def test_serve_carries_the_status_line_not_the_stale_build_block():
    src = (ROOT / "phase0" / "nhl_serve.py").read_text(encoding="utf-8")
    assert 'payload["value_status"] = old["value_status"]' in src
    assert 'payload["value_blocked"]' not in src


# ---- the pages -------------------------------------------------------------

STUB = r"""
const el=()=>({onclick:null,oninput:null,innerHTML:"",textContent:"",title:"",className:"",style:{},dataset:{},
  classList:{toggle(){},add(){},remove(){},contains(){return false;}},lastElementChild:null,
  cloneNode(){return el();},setAttribute(){},getAttribute(){return null;},removeAttribute(){},
  querySelector(){return null;},querySelectorAll(){return [];},closest(){return null;}});
const __els={};
globalThis.document={documentElement:el(),hidden:false,title:"",querySelector:s=>__els[s]||(__els[s]=el()),
  querySelectorAll:()=>[],getElementById:()=>el(),addEventListener(){}};
globalThis.window=globalThis; globalThis.addEventListener=()=>{};
const __store={}; globalThis.localStorage={getItem:k=>__store[k]??null,setItem:(k,v)=>{__store[k]=String(v);}};
globalThis.location={hash:"",pathname:"/",search:""};
globalThis.history={state:null,replaceState(){},pushState(){}};
globalThis.MutationObserver=class{observe(){}};
globalThis.fetch=()=>new Promise(()=>{});
globalThis.scrollTo=()=>{};
"""

BODY = r"""
;(function(){
  const fs=require("fs"), A=JSON.parse(process.env.NHL_FIX_ARGS);
  const J=f=>JSON.parse(fs.readFileSync(f,"utf8"));
  state.board=J(A.data+"/board.json"); state.db=J(A.data+"/db.json");
  try{state.nfl=J(A.data+"/nfl.json");}catch(e){state.nfl=null;}
  state.nhl=J(A.payload); siteLeaguesTidy(); state.league="nhl";
  const view=()=>document.querySelector("#view").innerHTML, out={};
  state.nhlRange="week"; nhlPage(); out.board=view();
  out.card=nhlModelCard(state.nhl.model_card);
  nhlGamePage(String(A.gid)); out.game=view();
  if(A.blocked!=null){nhlGamePage(String(A.blocked)); out.blocked=view();}
  state.nhlTeamSort="elo"; nhlTeams(); out.teams=view();
  console.log(JSON.stringify(out));
})();
"""


def _render(tmp_path, n: dict, gid, blocked=None) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    for f in ("board.json", "db.json"):
        if not (DATA / f).is_file():
            pytest.skip(f"site/data/{f} not built")
    from mlbwp_site.build_site import JS
    p = tmp_path / "nhl.json"
    p.write_text(json.dumps(n), encoding="utf-8")
    script = tmp_path / "nhl_fix.js"
    script.write_text(STUB + JS.replace("\nboot();", "\n") + BODY, encoding="utf-8")
    env = dict(__import__("os").environ, NHL_FIX_ARGS=json.dumps(
        {"data": str(DATA), "payload": str(p), "gid": gid, "blocked": blocked}))
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=240,
                       env=env, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_model_card_prints_a_subtraction_that_holds(tmp_path):
    n = _served()
    out = _render(tmp_path, n, n["schedule"][0]["id"])
    card = out["card"]
    m = re.search(r"Elo alone (\d\.\d{5}) &minus; model (\d\.\d{5}) = gain (\d\.\d{5})", card)
    assert m, card[:600]
    base, ll, gain = (float(x) for x in m.groups())
    assert round(base - ll, 5) == gain
    mc = n["model_card"]
    assert f"{mc['test_ci'][0]:.5f}&ndash;{mc['test_ci'][1]:.5f}" in card
    assert f"{mc['test_n']:,}" in card and mc["test_seasons"] in card
    assert "different reference" in card and f"{mc['elo_raw_test']:.5f}" in card
    assert "undefined" not in card and "NaN" not in card and "null" not in card


def test_game_page_and_cards_show_home_ice_on_its_own(tmp_path):
    n = _served()
    g = next(x for x in n["schedule"] if x["hs"] is None)
    out = _render(tmp_path, n, g["id"])
    assert ">Home ice<" in out["game"] and "Home ice is " in out["game"]
    assert "Team Elo includes" not in out["game"]
    assert re.search(r'<span class="[pn]">home [+&]', out["board"])
    # the teams index shows Elo only as the results half of team strength
    assert re.search(r"Results: Elo <b>\d+</b> \(#\d+\)", out["teams"])


def test_a_playoff_slate_fills_the_board(tmp_path):
    """April-June: every regular-season game is played and the next games are
    playoff games. They are on the board, tagged, not filtered out."""
    from nhl_update import et_today
    n = copy.deepcopy(_served())
    today = et_today()
    S0 = n["schedule"]
    done = []
    for g in S0[6:10]:
        r = copy.deepcopy(g)
        r.update(d=(today - timedelta(days=2)).isoformat(), hs=3, y=1, last="REG",
                 tier="EARLY", frozen_at="2027-04-10T20:00:00Z")
        r["as"] = 1
        r.pop("value", None)
        done.append(r)
    po = []
    for i, g in enumerate(S0[:6]):
        r = copy.deepcopy(g)
        d = (today + timedelta(days=1 + i // 2)).isoformat()
        r.update(id=2026030111 + i, playoff=1, d=d, start_utc=d + "T23:00:00Z", near=1)
        r.pop("value", None)
        po.append(r)
    n["schedule"] = done + po
    n.update(phase="playoffs", proj={}, proj_info=None)
    out = _render(tmp_path, n, po[0]["id"])
    for r in po:
        assert f'href="#/game/nhl-{r["id"]}"' in out["board"], r["id"]
    assert '<span class="chip po">Playoffs</span>' in out["board"]
    assert '<span class="lbadge proj">Playoff</span>' in out["board"]
    assert "No NHL games in this window" not in out["board"]


def test_a_held_badge_says_why_and_a_suppressed_run_says_so(tmp_path):
    n = copy.deepcopy(_served())
    g = next(x for x in n["schedule"] if x["hs"] is None)
    g.pop("value", None)
    g["value_blocked"] = "EARLY tier: team ratings only, no goalie or lineup input; shown as a lean, not a badge"
    n["value_status"] = {"state": "suppressed", "reason": "the model build is 40h old", "gate_early": True,
                         "n_early": 0}
    out = _render(tmp_path, n, g["id"], blocked=g["id"])
    assert "<b>no EDGE badge</b>: EARLY tier: team ratings only" in out["blocked"]
    assert "No EDGE badges this cycle</b>: the model build is 40h old." in out["board"]


# ---- a game whose horn has gone but whose result is not official -----------

def test_an_over_game_stays_on_the_slate_until_it_is_official():
    import nhl_update as U
    g = lambda gid, st, d: {"id": gid, "gameType": 2, "gameState": st, "season": 20262027,  # noqa: E731
                            "homeTeam": {"abbrev": "TOR", "score": 3}, "awayTeam": {"abbrev": "MTL", "score": 2},
                            "startTimeUTC": d + "T23:00:00Z"}
    data = {"gameWeek": [{"date": "2026-10-01", "games": [g(1, "OVER", "2026-10-01"), g(2, "LIVE", "2026-10-01")]}]}
    new, up = U.collect_week(data, set(), "2026-10-02")      # the ET date has already turned
    assert new == [] and {u["id"]: u["state"] for u in up} == {1: "OVER", 2: "LIVE"}
    # the serve holds a started row at its pre-game entry, OVER included
    assert S.started({"state": "OVER"}, "2026-10-01T00:00:00Z")
