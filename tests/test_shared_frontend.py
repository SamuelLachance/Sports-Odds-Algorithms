"""The shared chrome (document head, league-qualified URLs, rail, freshness) and
the track record's tier/replay/overdue handling.

The head is checked on the built SHELL. The behaviour is checked by running the
real, fully substituted page script under Node with a minimal DOM stub and
calling its helpers directly - so a regression in the league router or in the
record's honesty rules fails here, not on a phone. Skipped when Node is absent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from mlbwp_site.build_site import JS, SHELL


def test_shell_is_a_real_document():
    """Without the viewport meta every mobile @media rule is dead on phones
    (they lay the page out at 980px); without a doctype it renders in quirks."""
    assert SHELL.lstrip().lower().startswith("<!doctype html>")
    assert '<html lang="en">' in SHELL
    assert '<meta charset="utf-8">' in SHELL
    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in SHELL
    assert "<title>" in SHELL and "</title>" in SHELL
    assert SHELL.index("<head>") < SHELL.index("<style>") < SHELL.index("</head>") < SHELL.index("<body>")
    assert SHELL.rstrip().endswith("</html>")
    assert 'id="ft-fresh"' in SHELL and "ESPN" in SHELL          # data age + live-score credit


def test_page_title_is_ascii_in_source():
    """build() entity-encodes non-ASCII; document.title does not decode
    entities, so the title separator must be plain ASCII in the JS."""
    line = next(ln for ln in JS.splitlines() if "document.title=" in ln)
    assert all(ord(ch) < 128 for ch in line.split("//")[0])


HARNESS = r"""
const el=()=>({onclick:null,innerHTML:"",textContent:"",title:"",className:"",style:{},dataset:{},
  classList:{toggle(){},add(){},remove(){},contains(){return false;}},
  setAttribute(){},getAttribute(){return null;},removeAttribute(){},
  querySelector(){return null;},querySelectorAll(){return [];},closest(){return null;}});
const store={}, els={};
// one element per selector, so a test can read back what a renderer wrote to #view
globalThis.document={documentElement:el(),hidden:false,querySelector:s=>els[s]||(els[s]=el()),querySelectorAll:()=>[],
  getElementById:()=>el(),addEventListener(){}};
const view=()=>document.querySelector("#view").innerHTML;
globalThis.window=globalThis; globalThis.addEventListener=()=>{};
globalThis.localStorage={getItem:k=>store[k]??null,setItem:(k,v)=>{store[k]=String(v);}};
globalThis.location={hash:"",pathname:"/",search:""};
globalThis.history={state:null,replaceState(){},pushState(){}};
globalThis.MutationObserver=class{observe(){}};
globalThis.fetch=()=>new Promise(()=>{});      // boot() never proceeds: the tests own `state`
globalThis.scrollTo=()=>{};
__JS__
;(function(){
  const out={};
  const T=(k,f)=>{try{out[k]=f();}catch(e){out[k]="THREW "+e.message;}};
  const today=nflToday(), plus=n=>addDays(today,n);
  T("parse_prefixed",()=>siteParse("#/nfl/team/BUF"));
  T("parse_legacy",()=>siteParse("#/team/BUF"));
  T("parse_board",()=>siteParse("#/nhl"));
  T("href",()=>[siteHref("nhl","player","8474600"),siteHref("mlb",""),siteHref("nfl","record")]);
  T("id_leagues",()=>[siteLeagueOfId("game","nhl-2026020001"),siteLeagueOfId("game","3_ATL_GB"),
    siteLeagueOfId("game","776543"),siteLeagueOfId("pos","QB"),siteLeagueOfId("pos","hitters"),
    siteLeagueOfId("season",undefined),siteLeagueOfId("player","00-0039851"),siteLeagueOfId("team","SEA")]);
  T("team_link",()=>teamLink("BUF","Bills","nfl"));
  T("clock",()=>[siteClock("20:15"),siteClock("13:00"),siteClock("00:05")]);
  // NFL tiers: the stamp wins; a lock time dates it; a played row with neither is EARLY
  T("nfl_tiers",()=>[
    siteTier("nfl",{d:plus(-3),hs:20,as:10}),
    siteTier("nfl",{d:plus(-3),hs:20,as:10,tier:"PROJECTED"}),
    siteTier("nfl",{d:"2026-09-13",hs:1,as:0,frozen_at:"2026-09-10T15:00:00Z"}),
    siteTier("nfl",{d:"2026-09-13",hs:1,as:0,frozen_at:"2026-07-30T23:57:40Z"}),
    siteTier("nfl",{d:plus(3)}), siteTier("nfl",{d:plus(30)}),
    siteTier("nfl",{d:plus(3),tier:"EARLY"}),
    siteTier("nhl",{d:plus(1)})]);
  T("nflprob",()=>{const far={d:plus(30),ph:0.6,pmc:0.55}, near={d:plus(2),ph:0.6,pmc:0.55};
    return [nflProb(far).hp,nflProb(near).hp,nflProb(far).tier,nflProb(near).tier];});
  T("pill",()=>[sitePill("EARLY",0.7,"SEA"),sitePill("PROJECTED",0.53,"SEA"),sitePill("PROJECTED",0.7,"SEA")]);
  // a URL id is whitelisted before any page sees it; what reaches #view is text
  const XSS="%3Cimg%20src%3Dx%20onerror%3D%22window.__xss%3D1%22%3E";
  T("parse_ids",()=>[siteParse("#/nfl/player/"+XSS).arg, siteParse("#/team/"+XSS).arg,
    siteParse("#/nfl/player/<img src=x>").arg, siteParse("#/nfl/player/%E0%A4%A").arg,
    siteParse("#/nfl/player/00-0034869").arg, siteParse("#/game/3_ATL_GB").arg,
    siteParse("#/nhl/game/nhl-2026020001").arg, siteParse("#/mlb/pos/hitters").arg,
    siteParse("#/mlb/player/660271").arg, siteParse("#/nfl/<b>x</b>/1").v]);
  T("notfound_xss",()=>{siteNotFound("nfl","player",decodeURIComponent(XSS)); return view();});
  T("adddays",()=>[addDays("2026-09-24",0),addDays("2026-09-24",7),addDays("2026-03-08",1),addDays("2026-11-01",-1)]);
  // the freeze's exact clock (phase0/nfl_ph_freeze.tier_for) when start_utc exists
  T("tier_clock",()=>[
    siteTier("nfl",{d:"2026-10-01",start_utc:"2026-10-02T00:15:00Z",frozen_at:"2026-09-24T15:08:00Z"}),
    siteTier("nfl",{d:"2026-10-01",start_utc:"2026-10-02T00:15:00Z",frozen_at:"2026-09-25T15:00:00Z"}),
    siteTier("nfl",{d:"2026-10-01",frozen_at:"2026-09-24T15:08:00Z"})]);
  T("draw_error",()=>{const kb=state.board, kf=state.failed;
    state.board=null; state.failed={mlb:true};
    siteDrawError("nfl",new TypeError("Cannot read properties of null (reading 'leagues')")); const a=view();
    siteDrawError("nfl",new TypeError("g.hp is undefined")); const b=view();
    state.failed={}; siteDrawError("nfl",new TypeError("Cannot read properties of null (reading 'leagues')")); const c=view();
    state.board=kb; state.failed=kf;
    return [a.includes("MLB board file failed to load"), b.includes("current NFL data"), c.includes("current NFL data")];});
  // record: replay rows and NHL playoff rows stay out of every rate; played
  // unscored games are overdue, never pending picks
  state.nfl={status:"season",season:2026,teams:{BUF:{name:"Bills"}},players:{},schedule:[
    {w:1,d:"2026-09-13",home:"BUF",away:"MIA",ph:0.7,pmc:0.68,hs:30,as:10},
    {w:1,d:"2026-09-14",home:"SEA",away:"NE",ph:0.6,pmc:0.6,hs:10,as:20,tier:"PROJECTED",frozen_at:"2026-09-10T00:00:00Z"},
    {w:1,d:"2026-09-14",home:"KC",away:"DEN",ph:0.6,pmc:0.6,hs:10,as:20,replay:true},
    {w:3,d:plus(-2),home:"GB",away:"ATL",ph:0.66,pmc:0.7,hs:null,as:null},
    {w:3,d:plus(2),home:"LAC",away:"LV",ph:0.62,pmc:0.6,hs:null,as:null,tier:"PROJECTED"},
    {w:9,d:plus(40),home:"DAL",away:"NYG",ph:0.52,pmc:0.45,hs:null,as:null}]};
  T("nfl_rows",()=>recRows("nfl").map(r=>[r.home,r.tier,!!r.replay]));
  T("nfl_rated",()=>recRated(recRows("nfl")).length);
  T("nfl_picks",()=>recPicks("nfl").length);
  T("nfl_pending",()=>recPending("nfl").map(r=>r.home));
  T("nfl_overdue",()=>recOverdue("nfl").map(r=>r.home));
  T("nfl_sched",()=>recScheduled("nfl").map(r=>[r.home,r.pick,r.p]));
  state.nhl={status:"season",cur_season:20262027,teams:{BOS:{}},players:{},schedule:[
      {id:2026020001,d:"2026-10-01",home:"BOS",away:"MTL",hp:0.6,hs:3,as:2,y:1,last:"OT",tier:"EARLY",frozen_at:"2026-10-01T12:00:00Z"},
      {id:2026020002,d:"2026-10-02",home:"TOR",away:"MTL",hp:0.6,hs:1,as:2,y:0,tier:"EARLY",replay:true},
      {id:2026030111,d:"2027-04-20",home:"BOS",away:"TOR",hp:0.55,hs:4,as:1,y:1,tier:"EARLY"}],
    prev_season:{season:20252026,label:"2025-26",kind:"replay",rows_cols:["id","d","home","away","hp","hs","as","last"],
      rows:[[2025020001,"2025-10-07","FLA","CHI",0.72,3,2,"REG"]]}};
  T("nhl_rows",()=>recRows("nhl").map(r=>[r.home,r.tier,!!r.replay,!!r.po,r.ot]));
  T("nhl_rated",()=>recRated(recRows("nhl")).length);
  T("nhl_prev",()=>{const P=recPrevNhl(); return [P.rows[0].home,P.rows[0].hp,P.rows[0].last];});
  T("nhl_year",()=>nhlYear());
  T("nhl_badge",()=>recBadge("nhl"));
  // rail: one NHL button once nhl.json is in; the served leagues first
  state.board={generated:today,leagues:[{code:"mlb",name:"MLB",active:true,games:[],n_games:0},
    {code:"nhl",name:"NHL",active:false},{code:"nba",name:"NBA",active:false},
    {code:"nfl",name:"NFL",active:false},{code:"soccer",name:"Soccer",active:false}],
    record:{bets:{leagues:{nfl:{n_open:1,n_void:0,settled:null,by_month:[]}},
      pending:[{league:"nfl",d:plus(2),away:"LV",home:"WAS",team:"WAS",side:"home",dec:2.1,ev:0.1}]}}};
  siteLeaguesTidy();
  T("leagues",()=>state.board.leagues.map(l=>l.code));
  T("rail_nhl_buttons",()=>(siteRail("mlb").match(/data-lg="nhl"/g)||[]).length+(siteRail("mlb").match(/>NHL<\/span><span class="soon"/g)||[]).length);
  T("rail_order",()=>[...siteRail("mlb").matchAll(/<span>(\w+)<\/span>/g)].map(m=>m[1]));
  T("units_was",()=>{const h=recUnitsPanel("nfl"); return [h.includes(">WAS<"),h.includes("WSH")];});
  T("fresh_nfl",()=>{const f=siteFresh("nfl"); return [f.pending,f.late];});
  // a code another league owns: ask, in words that read right, never echo raw
  T("team_route",()=>{siteTeamRoute("nhl","BUF"); const a=view();
    siteTeamRoute("nfl",decodeURIComponent(XSS)); return [a, view()];});
  // units: an old-shape block (no by_tier, no statuses, no void list) is shown
  // as a plain sum and says what it cannot tell
  state.board.record.bets={leagues:{mlb:{n_open:0,n_void:1,
      settled:{n:5,w:3,l:2,staked:5,net:0.5,roi:0.1,avg_dec:1.9},by_month:[]}},pending:[]};
  T("units_old",()=>recUnitsPanel("mlb"));
  // units: the market/edge_ledger.site_block contract
  const ago=h=>new Date(Date.now()-h*3600e3).toISOString();
  state.board.record.bets={updated:"2026-09-24T10:00:00Z",stake:1,
    leagues:{nfl:{n_open:1,n_awaiting:4,n_overdue:1,n_void:2,
        settled:{n:3,w:2,l:1,staked:3,net:4.15,roi:1.3833,avg_dec:3.35},
        by_tier:{PROJECTED:{n:1,w:1,l:0,net:1.0,roi:1.0},EARLY:{n:2,w:1,l:1,net:3.15,roi:1.575}},
        by_month:[],curve:[],stale_model:{n:2,n_settled:1,net:-1.0,builds:["2026-07-30"]},
        policy:{tier:"PROJECTED",gate:"enforced",deficit:"priced from one weekly model build"}},
      nhl:{n_open:0,n_awaiting:0,n_overdue:0,n_void:0,settled:null,by_tier:{},by_month:[],curve:[],
        stale_model:null,policy:{tier:"EARLY",gate:"enforced",deficit:"team ratings only"}}},
    pending:[
      {league:"nfl",d:plus(-9),away:"NO",home:"BAL",team:"NO",side:"away",dec:3.95,ev:0.25,status:"overdue",tier:"EARLY",stale_model:true},
      {league:"nfl",d:plus(-1),away:"IND",home:"KC",team:"IND",side:"away",dec:3.15,ev:0.09,status:"awaiting",start_utc:ago(30),tier:"EARLY",stale_model:false},
      {league:"nfl",d:today,away:"ATL",home:"CHI",team:"ATL",side:"away",dec:3.4,ev:0.18,status:"awaiting",start_utc:ago(1),tier:"PROJECTED",stale_model:false},
      {league:"nfl",d:"2026-09-13",away:"MIA",home:"BUF",team:"BUF",side:"home",dec:1.5,ev:0.05,status:"awaiting",tier:"PROJECTED",stale_model:false},
      {league:"nfl",d:plus(3),away:"CIN",home:"PIT",team:"PIT",side:"home",dec:2.12,ev:0.26,status:"open",tier:"PROJECTED",stale_model:false}],
    voids:[{league:"nfl",d:"2026-09-13",away:"A",home:"B",team:"A",reason:"tie: a two-way moneyline pushes (stake returned)"},
      {league:"nfl",d:"2026-09-14",away:"C",home:"D",team:"C",reason:"not on the league schedule on this date (moved or cancelled)"},
      {league:"mlb",d:"2026-09-01",away:"E",home:"F",team:"E",reason:"<i>mlb only</i>"}]};
  T("units_new",()=>recUnitsPanel("nfl"));
  T("units_nhl_gated",()=>recUnitsPanel("nhl"));
  // rail badge on #/record: graded PICKS, never leans, never EARLY, never the
  // capped row count; MLB from the server's uncapped by_tier block
  const bt=(n,p,l)=>({n,acc:0.6,log_loss:0.66,brier:0.23,pick:{n:p,acc:0.62,log_loss:0.65,brier:0.22},
    lean:{n:l,acc:0.5,log_loss:0.69,brier:0.25}});
  state.board.record.mlb={rows:[{d:today,away:"NYM",home:"TEX",p:0.52,pick:"TEX",y:0,hs:2,as:7,sp:1,tier:"CONFIRMED"},
      {d:today,away:"SEA",home:"HOU",p:0.66,pick:"HOU",y:1,hs:5,as:1,sp:1,tier:"PROJECTED"}],
    rollup:{n:739,acc:0.58,log_loss:0.663,brier:0.236},
    by_tier:{CONFIRMED:bt(413,268,145),PROJECTED:bt(297,208,89),EARLY:bt(29,18,11)}};
  T("badge_mlb",()=>recBadge("mlb"));
  T("history_mlb",()=>{const h=recSection("mlb"); const m=/Full history[\s\S]*?<\/summary>/.exec(h); return m?m[0]:"";});
  state.nfl.schedule.push({w:2,d:"2026-09-20",home:"NYJ",away:"NE",ph:0.53,pmc:0.5,hs:21,as:17,tier:"PROJECTED"});
  T("badge_nfl",()=>recBadge("nfl"));
  console.log(JSON.stringify(out));
})();
"""


def _run(tmp_path_factory, tz=None):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    p = tmp_path_factory.mktemp("js") / "harness.js"
    p.write_text(HARNESS.replace("__JS__", JS.replace("\nboot();", "\n")), encoding="utf-8")
    env = dict(os.environ, TZ=tz) if tz else None
    r = subprocess.run([node, str(p)], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    return _run(tmp_path_factory)


def test_urls_carry_the_league(page):
    assert page["parse_prefixed"] == {"lg": "nfl", "v": "team", "arg": "BUF"}
    assert page["parse_legacy"]["lg"] is None and page["parse_legacy"]["v"] == "team"
    assert page["parse_board"] == {"lg": "nhl", "v": ""}
    assert page["href"] == ["#/nhl/player/8474600", "#/mlb", "#/nfl/record"]
    # ids decide their own league; a bare team code does not
    assert page["id_leagues"] == ["nhl", "nfl", "mlb", "nfl", "mlb", "nfl", "nfl", None]
    assert 'href="#/nfl/team/BUF"' in page["team_link"]


def test_nfl_tier_is_the_stamp_never_the_date_of_a_played_game(page):
    early, stamped, dated_near, dated_far, near, far, stale, nhl = page["nfl_tiers"]
    assert early == "EARLY"            # played, no stamp: conservative, never pooled as PROJECTED
    assert stamped == "PROJECTED"
    assert (dated_near, dated_far) == ("PROJECTED", "EARLY")
    assert (near, far) == ("PROJECTED", "EARLY")
    assert stale == "EARLY"            # a stale serve is not promoted by the calendar
    assert nhl == "EARLY"


def test_nfl_prob_shows_the_frozen_number_semantics(page):
    far_hp, near_hp, far_t, near_t = page["nflprob"]
    assert (far_hp, near_hp) == (0.55, 0.6)      # sim beyond a week, live model inside it
    assert (far_t, near_t) == ("EARLY", "PROJECTED")


def test_pill_follows_policy(page):
    early, lean, pick = page["pill"]
    assert "SCHEDULED" in early and "%" not in early          # rule 1: no call on EARLY
    assert "LEAN" in lean and "leanp" in lean                  # rule 2: inside 55/45
    assert "PICK" in pick and "strong" in pick


def test_record_reads_row_tiers_and_keeps_replays_out(page):
    assert page["nfl_rows"] == [["SEA", "PROJECTED", False], ["KC", "EARLY", True],
                                ["BUF", "EARLY", False]]
    assert page["nfl_rated"] == 2            # the replay is listed, never rated
    assert page["nfl_picks"] == 1            # only the stamped PROJECTED game is a graded pick
    assert page["nfl_pending"] == ["LAC"]    # played-but-unscored ATL@GB is not "awaiting the result"
    assert page["nfl_overdue"] == ["GB"]
    # the scheduled (EARLY) list shows the board's number - the season sim - not ph
    assert page["nfl_sched"] == [["DAL", "NYG", 0.45]]


def test_nhl_record_rules(page):
    rows = page["nhl_rows"]
    assert ["BOS", "EARLY", False, True, None] in rows       # playoff id (type 03) flagged
    assert ["BOS", "EARLY", False, False, "OT"] in rows
    assert page["nhl_rated"] == 1                            # replay + playoff kept out
    assert page["nhl_prev"] == ["FLA", 0.72, "REG"]          # columnar prev_season decoded
    assert page["nhl_year"] == "2026&ndash;27"               # never the bare "2027"
    b = page["nhl_badge"]
    assert b["cls"] == "early" and "0 graded picks" in b["title"]


def test_one_nhl_button_and_units_codes(page):
    assert page["leagues"] == ["mlb", "nfl", "nba", "soccer"]  # placeholder dropped, served first
    assert page["rail_nhl_buttons"] == 1
    assert page["rail_order"][:3] == ["MLB", "NFL", "NHL"]
    has_was, has_wsh = page["units_was"]
    assert has_was and not has_wsh                             # MLB's Retrosheet map stays MLB's


def test_freshness_flags_played_games_without_results(page):
    pending, late = page["fresh_nfl"]
    assert pending == 1 and late == 0      # ATL@GB two days ago: pending, not yet "late" for NFL


def test_url_ids_are_whitelisted_and_never_markup(page):
    """#/nfl/player/<img onerror=...> ran script via siteNotFound's innerHTML."""
    ids = page["parse_ids"]
    assert ids[:4] == ["?", "?", "?", "?"]          # markup, raw or encoded, and a malformed escape
    assert ids[4:9] == ["00-0034869", "3_ATL_GB", "nhl-2026020001", "hitters", "660271"]
    assert ids[9] == ""                              # an unknown view opens the board
    nf = page["notfound_xss"]
    assert "<img" not in nf and "&lt;img" in nf
    asked, raw = page["team_route"]
    assert "BUF is not an NHL team" in asked and 'href="#/nfl/team/BUF"' in asked and "Bills" in asked
    assert "<img" not in raw


@pytest.mark.parametrize("tz", ["Europe/Paris", "Pacific/Kiritimati", "America/Los_Angeles"])
def test_add_days_is_the_same_calendar_day_in_every_timezone(tmp_path_factory, tz):
    """A local-midnight Date read back through toISOString() was a day early east of UTC."""
    got = _run(tmp_path_factory, tz)["adddays"]
    assert got == ["2026-09-24", "2026-10-01", "2026-03-09", "2026-10-31"]


def test_nfl_tier_fallback_uses_the_freeze_clock(page):
    seven_days_nine_hours, inside, date_rule = page["tier_clock"]
    assert seven_days_nine_hours == "EARLY"          # TNF 20:15 ET locked the Thursday before at 11:08 ET
    assert inside == "PROJECTED"
    assert date_rule == "PROJECTED"                  # no start_utc: game day <= serve day + 7


def test_error_boundary_names_the_failed_board(page):
    board_gone, other_error, board_loaded = page["draw_error"]
    assert board_gone and other_error and board_loaded


def test_units_follow_the_edge_ledger_contract(page):
    h = page["units_new"]
    # by tier first, the all-badges sum as a secondary line under it
    assert h.index("t-PROJECTED") < h.index("All badges") and h.index("t-EARLY") < h.index("All badges")
    assert "+3.15u" in h and "+4.15u" in h and "allrow" in h
    # the ledger's own status per bet, refined only for a played/started awaiting bet
    for chip in (">OVERDUE<", ">AWAITING RESULT<", ">IN PROGRESS<", ">FINAL<", ">OPEN<", ">STALE MODEL<"):
        assert chip in h, chip
    assert "4 bets are on started games awaiting settlement, 1 of them overdue" in h
    assert "pipeline gap, not a void" in h
    # voids: evidence, with the ledger's reasons - never "after 7 days"
    assert "voided on evidence" in h and "after 7 days" not in h
    assert "tie: a two-way moneyline pushes" in h and "not on the league schedule" in h
    assert "mlb only" not in h                       # another league's void is not listed here
    assert "stale model build" in h and "2026-07-30" in h and "CLV promotion gate" in h
    assert "priced from one weekly model build" in h
    nhl = page["units_nhl_gated"]
    assert "no badge" in nhl and "disclosed rather than gated" not in nhl


def test_units_old_shape_is_a_labelled_sum(page):
    h = page["units_old"]
    assert "does not split units by information tier" in h and "All badges" in h
    assert "does not list the void reasons" in h and "after 7 days" not in h


def test_record_badge_counts_graded_picks_only(page):
    mlb = page["badge_mlb"]
    assert mlb["txt"] == 268 + 208                   # uncapped, CONFIRMED + PROJECTED picks, no leans
    assert "234 leans" in mlb["title"] and "29 EARLY-tier forecasts" in mlb["title"]
    assert "739 graded" in page["history_mlb"] and "476 picks" in page["history_mlb"]
    nfl = page["badge_nfl"]
    assert nfl["txt"] == 1 and "1 lean " in nfl["title"]   # the PROJECTED 53% game is a lean
