"""NHL player value on the pages (package nhl-player-value-display), rendered for
real under Node - the same harness as tests/test_nhl_pages_frontend.py (the fully
substituted site script, a minimal DOM stub, the real site/data payloads).

Pinned, on the served payload, on an in-season copy and on a payload WITHOUT the
snapshot (the serve's degrade path):

  * every player page, every team page, the teams index and the players index in
    every position / sort (value, per game, each component header, on-ice xG) /
    minimum renders with nothing thrown and no undefined / NaN / null;
  * a valued skater's page leads with player value - the percentile badge in the
    header, the eight components, the display-only label, the help block - and
    keeps on-ice xG impact (RAPM) below it; a skater without a value says why;
    a goalie keeps GSAx and says honestly why saving is not shown as an upgrade;
  * the players index ranks by value by default, and every sortable column
    really sorts; the team page carries the lineup-value summary and a Value
    column; the label 'the game model does not use it yet (forward test on
    2026-27 pending)' is on the player, players and team pages;
  * without the snapshot every page falls back to the RAPM GlassBox, cleanly.
Skipped when Node is absent.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mlbwp_site.build_site import JS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"
_spec = importlib.util.spec_from_file_location("nhl_pages_frontend",
                                               ROOT / "tests" / "test_nhl_pages_frontend.py")
FE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FE)

LABEL = "the game model does not use it yet (forward test on 2026-27 pending)"
COMPS = ["Creation", "Finishing", "Primary assists", "Secondary assists", "Power play",
         "Faceoffs", "Penalties", "Defence"]

BODY = r"""
;(function(){
  const fs=require("fs"), A=JSON.parse(process.env.NHL_PV_ARGS);
  const view=()=>document.querySelector("#view").innerHTML;
  const J=f=>JSON.parse(fs.readFileSync(f,"utf8"));
  state.board=J(A.data+"/board.json"); state.db=J(A.data+"/db.json");
  try{state.nfl=J(A.data+"/nfl.json");}catch(e){state.nfl=null;}
  state.nhl=J(A.payload); siteLeaguesTidy(); state.league="nhl";
  const out={n:0,throws:[],bad:[],r:{}};
  const BAD=/undefined|NaN|\bnull\b|\[object/;
  const T=(k,f)=>{out.n++; try{f(); const h=view();
      const m=BAD.exec(h.replace(/data-[a-z]+="[^"]*"/g,""));
      if(m) out.bad.push([k,h.slice(Math.max(0,m.index-120),m.index+40)]);
      return h;}
    catch(e){out.throws.push([k,String(e&&e.stack||e).split("\n").slice(0,2).join(" | ")]); return "";}};
  const n=state.nhl, R=out.r, COMPS=JSON.parse(process.env.NHL_PV_COMPS);
  const ids=h=>[...h.matchAll(/location\.hash='#\/nhl\/player\/(\d+)'/g)].map(m=>m[1]);
  // a fresh visit: the ladder opens on its own default, whatever core's legacy nhlSort was
  R.fresh=ids(T("players:fresh",()=>nhlPlayers()));
  R.players={};
  for(const [id,p] of Object.entries(n.players)){const h=T("player:"+id,()=>nhlPlayerPage(id));
    const head=(/<div class="phead">([\s\S]*?)<\/div>/.exec(h)||[])[1]||"";
    R.players[id]={label:h.includes(A.label), pvPanel:/<h3>Player value /.test(h),
      rapm:h.includes("On-ice xG impact (RAPM)"), help:h.includes("How player value works"),
      goalieNote:h.includes("did not prove more repeatable"), gsax:/GSAx/.test(h),
      comps:COMPS.filter(x=>h.includes("<b>"+x+"</b>")).length,
      headPct:/<span class="v">\d+(st|nd|rd|th)<\/span>/.test(head), headNR:/>NR</.test(head),
      noValue:/No player value: no NHL game on file|Player value is not in this build/.test(h),
      snapNote:/not updated during the season/.test(h),
      promise:/rookie gets one|after his first NHL games|gets one after/.test(h),
      total:h.includes("Total value"), netxg:h.includes("Net xG/60")};}
  const sorts=["v","p","g","r","toi","net","off","def","gsax","gp","c_cre","c_fin","c_a1","c_a2","c_pp","c_fo","c_pen","c_def"];
  for(const pos of ["all","F","C","L","R","D","G"]) for(const mn of ["0","20","60","nr","2000"]) for(const s of sorts){
    state.nhlPos=pos; state.nhlMin=mn; state.nhlSort=s; T(`players:${pos}:${mn}:${s}`,()=>nhlPlayers());}
  state.nhlPos="all"; state.nhlMin="0"; state.nhlSort=null; R.ladder=T("ladder",()=>nhlPlayers());
  R.order={def:ids(R.ladder)}; R.lit={};
  for(const s of ["v","p","g","c_cre","c_a1","c_a2","c_pp","c_fin","c_fo","c_pen","c_def","net"]){state.nhlSort=s;
    const h=T("ord:"+s,()=>nhlPlayers()); R.order[s]=ids(h);
    R.lit[s]=[...h.matchAll(/<th data-nsort="([^"]+)" class="nhl-srt on"/g)].map(m=>m[1]);}
  state.nhlSort=null; state.nhlMin="nr"; R.nr=T("players:nr",()=>nhlPlayers()); state.nhlMin="0";
  R.teams={};
  for(const c of Object.keys(n.teams)){const h=T("team:"+c,()=>nhlTeamPage(c));
    R.teams[c]={lu:h.includes("<h3>Lineup value"), label:h.includes(A.label), valCol:/<th[^>]*>Value<\/th>/.test(h),
      gbHead:h.includes(">Lineup GlassBox<"), help:h.includes("How player value works")};}
  R.tindex={};
  for(const s of ["elo","gb","pv","proj"]){state.nhlTeamSort=s; R.tindex[s]=T("teams:"+s,()=>nhlTeams());}
  R.game=T("game:first",()=>nhlGamePage(String(n.schedule[0].id)));
  console.log(JSON.stringify(out));
})();
"""


def _run(tmp_path: Path, payload: Path) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    for f in ("board.json", "db.json"):
        if not (DATA / f).is_file():
            pytest.skip(f"site/data/{f} not built")
    script = tmp_path / "nhl_pv_pages.js"
    script.write_text(FE.STUB + JS.replace("\nboot();", "\n") + BODY, encoding="utf-8")
    env = dict(os.environ,
               NHL_PV_ARGS=json.dumps({"data": str(DATA), "payload": str(payload), "label": LABEL}),
               NHL_PV_COMPS=json.dumps(COMPS))
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=600,
                       env=env, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _write(d: Path, n: dict) -> Path:
    p = d / "nhl.json"
    p.write_text(json.dumps(n), encoding="utf-8")
    return p


def _without_snapshot(n: dict) -> dict:
    n = copy.deepcopy(n)
    for p in n["players"].values():
        p["pv"] = None
    for t in n["teams"].values():
        t.pop("pv_lu", None)
    return n


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    n = FE._served()
    if not any(p.get("pv") for p in n["players"].values()):
        pytest.skip("the served payload carries no player value (snapshot not merged)")
    return n, _run(tmp_path_factory.mktemp("pv_served"), DATA / "nhl.json")


@pytest.fixture(scope="module")
def season(tmp_path_factory):
    n = FE._in_season(FE._served())
    d = tmp_path_factory.mktemp("pv_season")
    return n, _run(d, _write(d, n))


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    n = _without_snapshot(FE._served())
    d = tmp_path_factory.mktemp("pv_bare")
    return n, _run(d, _write(d, n))


def _clean(out):
    assert out["n"] > 1500
    assert out["throws"] == [], out["throws"][:5]
    assert out["bad"] == [], out["bad"][:5]


def test_every_page_renders_with_and_without_the_snapshot(served, season, bare):
    for _, out in (served, season, bare):
        _clean(out)


def test_valued_skaters_lead_with_player_value(served):
    n, out = served
    pl = out["r"]["players"]
    valued = [pid for pid, p in n["players"].items() if p["grp"] != "G" and p.get("pv")]
    assert len(valued) > 600
    for pid in valued:
        r = pl[pid]
        assert r["pvPanel"] and r["label"] and r["help"] and r["total"], pid
        assert r["comps"] == len(COMPS), (pid, r["comps"])
        assert r["headPct"] and not r["headNR"], pid          # header badge = value percentile
        assert r["rapm"], pid                                  # RAPM kept, relabelled
        if n["players"][pid]["rating"] is not None:
            assert r["netxg"], pid


def test_skaters_without_a_value_say_why(served):
    """The snapshot is static (end of 2025-26, built locally, only read by CI), so
    a debut player keeps no value all season: the page says exactly that and never
    promises one after his first games."""
    n, out = served
    missing = 0
    for pid, p in n["players"].items():
        r = out["r"]["players"][pid]
        assert not r["promise"], pid
        if p["grp"] != "G" and not p.get("pv"):
            missing += 1
            assert r["noValue"] and r["snapNote"] and r["rapm"] and not r["headPct"], pid
    assert missing > 0
    assert "not updated during the season" in out["r"]["nr"]


def test_goalies_keep_gsax_and_say_why(served):
    n, out = served
    for pid, p in n["players"].items():
        if p["grp"] == "G":
            r = out["r"]["players"][pid]
            assert r["gsax"] and r["goalieNote"] and not r["pvPanel"], pid


def _pv(n, pid):
    return n["players"][pid]["pv"]


def test_players_index_ranks_by_value_and_every_column_sorts(served):
    n, out = served
    order = out["r"]["order"]
    lad = out["r"]["ladder"]
    assert LABEL in lad and "How player value works" in lad
    for h in (">Value<", ">/60<", ">/gm<", ">Cre<", ">Fin<", ">A1<", ">A2<", ">PP<", ">FO<",
              ">Pen<", ">Def<", ">On-ice xG<"):
        assert h in lad, h
    n_valued = sum(1 for p in n["players"].values() if p["grp"] != "G" and p.get("pv"))
    assert order["def"] == order["v"] and len(order["v"]) == n_valued       # value is the default
    assert out["r"]["fresh"] == order["v"]                                    # also on a fresh visit

    def nonincreasing(ids, f):
        xs = [f(i) for i in ids]
        xs = [x if x is not None else -1e9 for x in xs]
        return all(a >= b - 1e-12 for a, b in zip(xs, xs[1:]))

    assert nonincreasing(order["v"], lambda i: _pv(n, i)["v"])
    assert nonincreasing(order["p"], lambda i: _pv(n, i)["p"])
    assert nonincreasing(order["g"], lambda i: _pv(n, i)["g"])
    for k in ("cre", "fin", "fo", "pen", "def"):
        assert nonincreasing(order["c_" + k], lambda i, k=k: _pv(n, i)["c"][k]), k
    assert nonincreasing(order["net"], lambda i: n["players"][i]["net"])
    # the players without a value are listed, with the reason
    assert "no NHL game on file through" in out["r"]["nr"]


def test_team_pages_carry_the_lineup_value(served):
    n, out = served
    for c, r in out["r"]["teams"].items():
        assert r["lu"] and r["label"] and r["valCol"] and r["help"], c
    assert ">Lineup value <" in out["r"]["tindex"]["pv"] or "Lineup value <b>" in out["r"]["tindex"]["pv"]
    assert "Key skaters <span class=\"sub\">player value" in out["r"]["game"]
    assert "lineup value" in out["r"]["game"]


def test_in_season_pages_keep_the_value(season):
    n, out = season
    valued = [pid for pid, p in n["players"].items() if p["grp"] != "G" and p.get("pv")]
    assert valued and all(out["r"]["players"][pid]["pvPanel"] for pid in valued)


def test_without_the_snapshot_the_pages_fall_back_to_rapm(bare):
    n, out = bare
    r = out["r"]
    for pid, p in n["players"].items():
        if p["grp"] != "G":
            assert not r["players"][pid]["pvPanel"] or r["players"][pid]["noValue"], pid
            assert not r["players"][pid]["headPct"], pid
    lad = r["ladder"]
    assert LABEL not in lad and ">GlassBox<" in lad and "How player value works" not in lad
    for c, t in r["teams"].items():
        assert not t["lu"] and t["gbHead"] == (n["teams"][c].get("glassbox") is not None), c
    assert "Lineup value <b>" not in r["tindex"]["elo"]


def test_each_sort_lights_exactly_its_own_header(served):
    """'Value' sorts by percentile, '/60' by value per 60: one lit header per sort."""
    _, out = served
    for k, lit in out["r"]["lit"].items():
        assert lit == [k], (k, lit)


def test_help_text_states_the_validated_claims_exactly(served):
    t = " ".join(served[1]["r"]["ladder"].split())
    # ledger row 11: delta = baseline - new, so +0.00062 is a small, n.s. improvement
    assert "improved log loss by only 0.00062 on the locked test seasons" in t
    assert "is not significant (95% CI &minus;0.00066 to +0.00188)" in t
    assert "+0.00062 log loss" not in t
    # the 6/6 per-season count includes the goalie's saving rating; skaters alone: pooled
    assert ("Summed over a lineup and added to the goalie's saving rating, the value predicted game "
            "goal differential better than the xG-only RAPM in each of six seasons (2012-18)") in t
    assert "the skaters' value alone also beat RAPM over 2012-18 as a whole" in t
    assert "It is not updated during the season" in t


def _num(t: str) -> float:
    return float(t.replace("&minus;", "-"))


def test_value_histogram_bins_are_labelled_truthfully(served):
    """No catch-all edge bin: every valued skater sits inside the labelled range,
    and the bins partition it with the counts adding up."""
    n, out = served
    lad = out["r"]["ladder"]
    bins = [(_num(a), _num(b), int(c)) for a, b, c in
            re.findall(r'title="([^" ]+) to ([^" ]+) goals/60: (\d+)"', lad)]
    assert len(bins) >= 13
    vals = [p["pv"]["v"] for p in n["players"].values() if p["grp"] != "G" and p.get("pv")]
    assert sum(c for *_, c in bins) == len(vals)
    for (_, hi0, _), (lo1, _, _) in zip(bins, bins[1:]):
        assert hi0 == pytest.approx(lo1)
    top = bins[-1][1]
    assert bins[0][0] <= min(vals) and max(vals) <= top
    for lo, hi, c in bins:
        inside = sum(1 for v in vals if lo - 1e-9 <= v < hi - 1e-9 or (hi == top and v >= hi - 1e-9))
        assert inside == c, (lo, hi, c, inside)


def test_value_histogram_widens_to_outliers_under_node(tmp_path):
    """Synthetic values beyond the default -0.5..+0.8 get their own honest bins."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = tmp_path / "vhist.js"
    script.write_text(FE.STUB + JS.replace("\nboot();", "\n")
                      + '\n;console.log(JSON.stringify(nhlVHist([-0.646,-0.2,0,0.1,0.95,null,NaN])));\n',
                      encoding="utf-8")
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=120,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    h = json.loads(r.stdout.strip().splitlines()[-1])
    bins = [(_num(a), _num(b), int(c)) for a, b, c in
            re.findall(r'title="([^" ]+) to ([^" ]+) goals/60: (\d+)"', h)]
    assert bins[0][0] == pytest.approx(-0.7) and bins[-1][1] == pytest.approx(1.0)
    assert len(bins) == 17 and sum(c for *_, c in bins) == 5
    assert bins[0][2] == 1 and bins[-1][2] == 1          # -0.646 in [-0.7,-0.6), 0.95 in [0.9,1.0)
    assert "<span>&minus;0.7</span>" in h and "<span>+1.0 goals/60</span>" in h


def test_phone_width_hides_the_component_percentile_bar():
    """At 375px the component table must fit its 315px wrapper: the percentile
    number stays, its bar goes."""
    css = (ROOT / "mlbwp_site" / "css" / "nhl.css").read_text(encoding="utf-8")
    m = re.search(r"@media\(max-width:560px\)\{(.*?)\n\}", css, re.S)
    assert m and ".nhl-pvt .nhl-pq{display:none}" in m.group(1)
