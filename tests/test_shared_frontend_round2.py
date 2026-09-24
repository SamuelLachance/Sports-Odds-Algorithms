"""Shared chrome, round 2: one meaning for the header's accuracy line, NFL dates
and tiers on the US Eastern clock for every visitor, a boot that draws the
league it opens on without waiting for the others, the Season page's rail, and
links nested in clickable rows that no longer leave a phantom history entry.

Like tests/test_shared_frontend.py, the real, fully substituted page script
runs under Node against a minimal DOM stub. Skipped when Node is absent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from mlbwp_site.build_site import CSS, JS

HARNESS = r"""
const el=(extra)=>Object.assign({onclick:null,innerHTML:"",textContent:"",title:"",className:"",style:{},dataset:{},
  classList:{toggle(){},add(){},remove(){},contains(){return false;}},
  setAttribute(k,v){this["@"+k]=v;},getAttribute(k){return this["@"+k]??null;},removeAttribute(k){delete this["@"+k];},
  querySelector(){return null;},querySelectorAll(){return [];},closest(){return null;},
  cloneNode(){return {lastElementChild:null,textContent:""};}},extra||{});
const store={}, els={}, listeners=[];
globalThis.document={documentElement:el(),hidden:false,body:{},querySelector:s=>els[s]||(els[s]=el()),querySelectorAll:()=>[],
  getElementById:id=>els["#"+id]||(els["#"+id]=el()),addEventListener(t,f,c){listeners.push([t,f,c]);},
  createElement:()=>el()};
const view=()=>document.querySelector("#view").innerHTML;
globalThis.window=globalThis; globalThis.addEventListener=()=>{};
globalThis.localStorage={getItem:k=>store[k]??null,setItem:(k,v)=>{store[k]=String(v);}};
globalThis.location={hash:"",pathname:"/",search:""};
globalThis.history={state:null,replaceState(s,t,u){location.hash=u.slice(u.indexOf("#"));},pushState(s,t,u){location.hash=u.slice(u.indexOf("#"));}};
globalThis.MutationObserver=class{observe(){}};
globalThis.scrollTo=()=>{};
globalThis.setInterval=()=>0;                   // boot's pollers must not keep Node alive
// fetch: each site payload is a deferred the test resolves by hand; the live
// feeds (MLB Stats API, ESPN) fail fast and are ignored
const pend={};
const deferred=()=>{let res; const p=new Promise(r=>{res=r;}); return {p,res};};
globalThis.fetch=(url)=>{
  const m=/\/data\/(\w+)\.json/.exec(String(url));
  if(!m) return Promise.reject(new Error("offline"));
  const d=pend[m[1]]||(pend[m[1]]=deferred());
  return d.p.then(v=>v===null?{ok:false,status:404,json:async()=>null}:{ok:true,status:200,json:async()=>v});
};
const give=(k,v)=>{(pend[k]||(pend[k]=deferred())).res(v);};
const tick=()=>new Promise(r=>setTimeout(r,0));
__JS__
;(async function(){
  const out={};
  const T=async(k,f)=>{try{out[k]=await f();}catch(e){out[k]="THREW "+(e&&e.stack||e);}};
  const today=nflToday();
  // page renderers are other packages' code: stub them to markers so these
  // tests pin the chrome, not the pages
  nflPage=()=>{$("#view").innerHTML="NFL-BOARD";};
  nhlPage=()=>{$("#view").innerHTML="NHL-BOARD";};
  nflSeason=()=>{$("#view").innerHTML="NFL-SEASON";};
  nflTeamPage=c=>{$("#view").innerHTML="NFL-TEAM "+c;};
  const BOARD={generated:today,leagues:[{code:"mlb",name:"MLB",active:true,games:[]},{code:"nhl",name:"NHL",active:false},
      {code:"nba",name:"NBA",active:false}],
    accuracy:{log_loss:0.67274,elo_log_loss:0.67861,coinflip:0.69315},record:{}};
  const NFL={status:"season",season:2026,teams:{BUF:{name:"Buffalo Bills"}},players:{},
    schedule:[{w:3,d:today,home:"BUF",away:"MIA",ph:0.6,pmc:0.55}],
    model_card:{test_log_loss:0.61961,close_log_loss:0.60913,holdout:"2016-2025, n=2,761, scored once",
      ratings_model:{test_ll:0.63211,elo_ll:0.63824},serve:{tier_window_days:7}}};
  const NHL={status:"season",cur_season:20262027,teams:{COL:{name:"Colorado Avalanche"}},players:{},
    schedule:[{id:2026020001,d:today,home:"COL",away:"DAL",hp:0.55}],
    model_card:{test_ll:0.66418,test_delta_vs_elo:0.00535,baseline_elo_test:0.66918,test_ci:[0.00353,0.00721]}};

  // ---------- boot: the MLB board draws before the NFL/NHL files land ----------
  location.hash="#/mlb";
  const bootP=boot();
  give("board",BOARD); give("db",{teams:{},players:{}});
  await tick(); await tick(); await tick();
  await T("mlb_first",()=>({booted:state.booted,loading:{...state.loading},view:view().includes("No games in this window"),
    rail:siteRail("mlb")}));
  // open the NHL while its file is still on its way: a loading line, not "unavailable"
  location.hash="#/nhl"; route(true);
  await T("nhl_wait",()=>({view:view(),waiting:state.waitingFor,league:state.league}));
  give("nhl",NHL); await tick(); await tick();
  await T("nhl_in",()=>({view:view(),waiting:state.waitingFor,loading:state.loading.nhl,acc:siteAccLine("nhl")}));
  give("nfl",NFL); await tick(); await tick(); await bootP;
  await T("nfl_in",()=>({loading:state.loading.nfl,view:view(),acc:siteAccLine("nfl"),mlb:siteAccLine("mlb")}));

  // ---------- header line: one comparator, labelled, in every league ----------
  await T("acc_html",()=>SITE_LGS.map(l=>{state.league=l; updAcc(); return document.querySelector("#acc").innerHTML;}));
  await T("acc_titles",()=>SITE_LGS.map(l=>siteAccLine(l).title));
  // the builder ships team Elo on the SAME 2016-2025, n=2,761 holdout
  await T("acc_nfl_elo",()=>{NFL.model_card.elo_log_loss=0.63984; const a=siteAccLine("nfl");
    state.league="nfl"; updAcc(); const h=document.querySelector("#acc").innerHTML;
    delete NFL.model_card.elo_log_loss; return {a,h};});

  // ---------- the league the first paint needs, from the URL alone ----------
  state.league="mlb";
  await T("boot_league",()=>["#/nhl","#/game/nhl-2026020001","#/game/3_ATL_GB","#/season","#/mlb/standings",
    "#/nfl/team/BUF","#/team/BUF","#/player/8474600","#/nfl/player/00-0034869",""].map(h=>siteBootLeague(siteParse(h))));

  // ---------- NFL clock: kickoff minus NOW against 7 x 24 h ----------
  const iso=ms=>new Date(ms).toISOString().replace(/\.\d+Z$/,"Z");
  const H=3600e3, now=Date.now();
  // the row dates below disagree with start_utc on purpose: the clock, not
  // the calendar day, decides whenever the row has a kickoff time
  await T("near_clock",()=>[
    nflProb({d:addDays(today,7),start_utc:iso(now+7*24*H+H),ph:0.6,pmc:0.55}).hp,        // 7 d 1 h out: the simulation
    nflProb({d:addDays(today,8),start_utc:iso(now+7*24*H-H),ph:0.6,pmc:0.55}).hp,        // 6 d 23 h out: the live model
    nflProb({d:addDays(today,-1),start_utc:iso(now-26*H),ph:0.6,pmc:0.55,hs:20,as:10}).hp, // played: always the frozen ph
    nflProb({d:addDays(today,30),ph:0.6,pmc:0.55}).hp,                                    // no start_utc: date rule
    // a PROJECTED stamp is never shown with the simulation's number
    nflProb({d:addDays(today,6),start_utc:iso(now+6*24*H),frozen_at:iso(now-H),ph:0.6,pmc:0.55}).near]);
  await T("et_dates",()=>[siteEtDate(Date.parse("2026-09-25T03:30:00Z")),siteEtDate(Date.parse("2026-09-25T04:30:00Z")),
    siteEtDate(Date.parse("2026-03-08T06:59:00Z")),siteEtDate(Date.parse("2026-11-01T04:30:00Z")),
    /^\d{4}-\d{2}-\d{2}$/.test(nflToday()), siteNowEtMin()>=0&&siteNowEtMin()<1440]);
  await T("tier_served_date",()=>[
    siteTier("nfl",{d:"2026-10-01",frozen_at:"2026-09-25T03:30:00Z"}),   // served 23:30 ET on 09-24: 7 days before the game
    siteTier("nfl",{d:"2026-10-02",frozen_at:"2026-09-25T03:30:00Z"})]);

  // ---------- Season page: the rail the board has, never two ----------
  const V=document.querySelector("#view");
  V.insertAdjacentHTML=function(pos,h){this.innerHTML=h+this.innerHTML;};
  V.querySelector=function(s){return s===".rail"&&this.innerHTML.includes('class="rail"')?{}:null;};
  state.league="nfl";
  await T("season_rail",()=>{siteDispatch("nfl","season"); const a=view();
    nflSeason=()=>{$("#view").innerHTML=`<div class="controls"><div class="rail">OWN</div></div>NFL-SEASON`;};
    siteDispatch("nfl","season"); return [a, view()];});

  // ---------- team names come from the payloads ----------
  await T("team_names",()=>[siteTeamName("nhl","COL"),siteTeamName("nfl","BUF"),siteTeamName("nhl","TBL")]);

  // ---------- a link inside a clickable row: the row stands down once ----------
  await T("nested_link",async()=>{
    const [,f,cap]=listeners.find(x=>x[0]==="click");
    let rowRan=0; const row={onclick:()=>{rowRan++;},parentElement:document.body};
    const a={parentElement:row}; const e={target:{closest:s=>s==="a[href]"?a:null}};
    f(e); const during=row.onclick; await new Promise(r=>setTimeout(r,5));
    const after=typeof row.onclick; row.onclick();
    // a click that is not on a link leaves the row alone
    const row2={onclick:()=>{},parentElement:document.body}; f({target:{closest:()=>null}});
    return {cap:cap===true,during,after,rowRan,row2:typeof row2.onclick};});
  console.log(JSON.stringify(out));
})().catch(e=>{console.log(JSON.stringify({fatal:String(e&&e.stack||e)}));});
"""


def _run(tmp_path_factory, tz=None):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    p = tmp_path_factory.mktemp("js") / "harness_r2.js"
    p.write_text(HARNESS.replace("__JS__", JS.replace("\nboot();", "\n")), encoding="utf-8")
    env = dict(os.environ, TZ=tz) if tz else None
    r = subprocess.run([node, str(p)], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "fatal" not in out, out.get("fatal")
    return out


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    return _run(tmp_path_factory)


def test_first_paint_does_not_wait_for_other_leagues(page):
    m = page["mlb_first"]
    assert m["booted"] and m["view"], m
    assert m["loading"] == {"mlb": False, "nfl": True, "nhl": True}
    # still-loading leagues say so on the rail; never the "unavailable" dash
    assert 'class="lg loading" data-lg="nfl"' in m["rail"] and 'class="lg loading" data-lg="nhl"' in m["rail"]
    assert "unavailable" not in m["rail"]


def test_a_league_opened_while_loading_draws_when_it_lands(page):
    w = page["nhl_wait"]
    assert "Loading NHL data" in w["view"] and "could not be loaded" not in w["view"]
    assert w["waiting"] == "nhl" and w["league"] == "nhl"
    i = page["nhl_in"]
    assert i["view"] == "NHL-BOARD" and i["waiting"] is None and i["loading"] is False
    assert page["nfl_in"]["loading"] is False


def test_header_compares_every_league_with_team_elo(page):
    mlb, nfl, nhl = page["acc_html"]
    for lg, h in zip(("MLB", "NFL", "NHL"), (mlb, nfl, nhl)):
        assert f"{lg} log loss" in h, h
    assert "team Elo" in mlb and "team Elo" in nhl
    assert "closing" not in nfl.lower() and "coin flip" not in mlb      # no market line, no mixed baselines
    assert "<b>0.673</b>" in mlb and "team Elo 0.679" in mlb
    # NHL: the baseline the delta was measured against (0.66418 + 0.00535),
    # not the other report's 0.66918
    a = page["nhl_in"]["acc"]
    assert abs(a["cmp"]["v"] - 0.66953) < 1e-9 and "team Elo 0.670" in nhl
    assert "beats bare Elo by 0.0054 (95% CI 0.0035 to 0.0072)" in a["title"]
    assert "2016-2025, n=2,761" in page["nfl_in"]["acc"]["title"]


def test_nfl_header_never_pairs_holdouts(page):
    """ratings_model.elo_ll (0.638) is team Elo on TEST 2022-2025, n=1,139: it
    must not sit next to the 2016-2025, n=2,761 log loss. Without a same-holdout
    model_card.elo_log_loss the line is the labelled coin flip."""
    nfl = page["acc_html"][1]
    assert "<b>0.620</b>" in nfl and "coin flip 0.693" in nfl
    assert "0.638" not in nfl and "team Elo" not in nfl
    fb = page["nfl_in"]["acc"]
    assert fb["cmp"]["k"] == "coin flip" and abs(fb["cmp"]["v"] - 0.693147) < 1e-5
    assert "same games" in fb["title"] and "team-Elo model scored" not in fb["title"]
    # once the builder ships team Elo on the same holdout (data/nfl_elo_base.json)
    e = page["acc_nfl_elo"]
    assert e["a"]["cmp"]["k"] == "team Elo" and abs(e["a"]["cmp"]["v"] - 0.63984) < 1e-9
    assert "<b>0.620</b>" in e["h"] and "team Elo 0.640" in e["h"]
    assert "(2016-2025, n=2,761, scored once)" in e["a"]["title"]
    assert "a plain team-Elo model scored on the same games" in e["a"]["title"]


def test_closing_line_pointer_only_where_the_board_prints_it(page):
    mlb, nfl, nhl = page["acc_titles"]
    assert "closing-line benchmark is shown on the board" in nfl
    for t in (mlb, nhl):
        assert "closing" not in t.lower() and "Market-free: the odds are never an input." in t


def test_boot_league_comes_from_the_url(page):
    assert page["boot_league"] == ["nhl", "nhl", "nfl", "nfl", "mlb", "nfl", None, None, None, "mlb"]


def test_nfl_number_follows_the_kickoff_clock(page):
    sim, live, played, date_rule, projected_near = page["near_clock"]
    assert (sim, live, played, date_rule) == (0.55, 0.6, 0.6, 0.55)
    assert projected_near is True


def test_eastern_dates(page):
    assert page["et_dates"] == ["2026-09-24", "2026-09-25", "2026-03-08", "2026-11-01", True, True]
    assert page["tier_served_date"] == ["PROJECTED", "EARLY"]


@pytest.mark.parametrize("tz", ["Europe/Paris", "Asia/Tokyo", "Pacific/Kiritimati", "America/Los_Angeles"])
def test_dates_and_tiers_do_not_depend_on_the_visitor(tmp_path_factory, page, tz):
    got = _run(tmp_path_factory, tz)
    for k in ("et_dates", "tier_served_date", "near_clock", "boot_league"):
        assert got[k] == page[k], (tz, k)


def test_season_page_gets_one_rail(page):
    ours, theirs = page["season_rail"]
    assert ours.startswith('<div class="controls"><div class="rail">') and ours.endswith("NFL-SEASON")
    assert 'data-lg="mlb"' in ours and 'class="lg on" data-lg="nfl"' in ours
    assert theirs.count('class="rail"') == 1 and "OWN" in theirs


def test_team_names_from_payloads(page):
    assert page["team_names"] == ["Colorado Avalanche", "Buffalo Bills", "Tampa Bay"]


def test_nested_link_does_not_fire_its_row(page):
    n = page["nested_link"]
    assert n["cap"] is True                      # capture phase: before the row's own handler
    assert n["during"] is None                   # the row stood down for this click
    assert n["after"] == "function" and n["rowRan"] == 1   # and is back for the next one
    assert n["row2"] == "function"


def test_nav_overflow_fade_is_measured_not_assumed():
    """The phone nav fade was a fixed mask under 560px (even when everything
    fit) and absent at 560-700px (where NFL's Season item pushes Players off)."""
    assert "nav.main.ovf{" in CSS and "nav.main.ovl{" in CSS
    phone = CSS.split("@media(max-width:560px)", 1)[1]
    assert "mask-image" not in phone.split("}}", 1)[0]
    assert "function siteNavFit(" in JS and 'addEventListener("resize"' in JS


def test_mlb_team_links_are_pinned_to_mlb():
    """SEA is the Mariners, the Seahawks and the Kraken: an MLB page must not
    take its team links' league from state.league."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "mlbwp_site" / "js" / "mlb.js").read_text(encoding="utf-8")
    calls = re.findall(r"teamLink\(([^()]*)\)", src)
    assert calls, "mlb.js no longer calls teamLink"
    for args in calls:
        assert args.rstrip().endswith('"mlb"'), args
