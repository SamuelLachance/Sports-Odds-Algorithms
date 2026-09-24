"""The NHL pages (mlbwp_site/js/nhl.js), rendered for real.

The fully substituted page script runs under Node with a minimal DOM stub, the
real site/data payloads loaded into `state`, and every NHL page is drawn: the
board in every range, every game, every team, every player, the players ladder
in every position / sort / minimum, the standings in every season and view, the
teams index - once on the served payload and once on an in-season copy (played
games with OT/SO, replays, a playoff game, current records and stat lines).

Pinned: nothing throws, no page prints undefined / NaN / null, and the pick
policy holds - every NHL forecast is EARLY, so no card or game page shows a
pick pill, a PICK/LEAN call, fair odds or PICK HIT/MISS; the EDGE bar keeps its
goalie disclosure and a stale (tier-less) badge never shows on an EARLY row; the
contribution equation holds as printed; the two sides add to 100; a replay
prints no log loss; a postponed game says so; the board order is a consistent
comparator; the back-link trail pops on Back; the phone-width guards are in
place; skater xG/60 keeps two decimals; a team page lists its whole roster;
standings are never a wall of 0-0-0. Skipped when Node is absent.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mlbwp_site.build_site import JS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"

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
  const fs=require("fs"), A=JSON.parse(process.env.NHL_PAGES_ARGS);
  const view=()=>document.querySelector("#view").innerHTML;
  const J=f=>JSON.parse(fs.readFileSync(f,"utf8"));
  state.board=J(A.data+"/board.json"); state.db=J(A.data+"/db.json");
  try{state.nfl=J(A.data+"/nfl.json");}catch(e){state.nfl=null;}
  state.nhl=J(A.payload); siteLeaguesTidy(); state.league="nhl";
  const out={n:0,throws:[],bad:[],pol:{}};
  const BAD=/undefined|NaN|\bnull\b|\[object/;
  const T=(k,f)=>{out.n++; try{f(); const h=view();
      const m=BAD.exec(h.replace(/data-[a-z]+="[^"]*"/g,""));
      if(m) out.bad.push([k,h.slice(Math.max(0,m.index-120),m.index+40)]);
      return h;}
    catch(e){out.throws.push([k,String(e&&e.stack||e).split("\n").slice(0,2).join(" | ")]); return "";}};
  const n=state.nhl, P=out.pol;
  P.board={};
  for(const r of ["today","tomorrow","week","month","results","year"]){state.nhlRange=r; P.board[r]=T("board:"+r,()=>nhlPage());}
  P.games=[];
  for(const g of n.schedule){const h=T("game:"+g.id,()=>nhlGamePage(String(g.id)));
    const ctSum=Object.values(g.ct||{}).reduce((s,v)=>s+(+v||0),0);
    P.games.push({id:g.id,hp:g.hp,played:g.hs!=null&&g.as!=null,value:!!(g.value&&g.value.available),
      badge:nhlBadge(g), valbar:/class="valbar"/.test(h), off:nhlOff(g),
      offShown:/(Postponed|Cancelled) by the NHL/.test(h), replay:!!g.replay, logLoss:/log loss/.test(h),
      fair:/Fair odds|class="odds"/.test(h), pickPill:/PICK &middot;|LEAN &middot;/.test(h),
      notPick:/not a pick/.test(h), goalieConf:/no goalie conf\./.test(h)&&/Priced before starting goalies are confirmed/.test(h),
      sum:(/50% (\+|&minus;)\s*(\d+\.\d) = (\d+\.\d)%, [A-Z]{2,3}'s home win probability/.exec(h)||[]).slice(1),
      bars:(/the bars add to (\+|&minus;)?(\d+\.\d)\)/.exec(h)||[]).slice(1), ctSum:Math.round(ctSum*10)/10,
      pct:(/<div class="barlab"><span>[A-Z]{2,3} (\d+)%<\/span><span>(\d+)% [A-Z]{2,3}<\/span><\/div>/.exec(h)||[]).slice(1).map(Number),
      pickHit:/PICK (HIT|MISS)/.test(h)});}
  // the board comparator is a consistent order: cmp(a,b) = -cmp(b,a) on every same-day pair
  const S=n.schedule, sg=x=>x>0?1:(x<0?-1:0); P.cmpBad=0;
  for(let i=0;i<S.length;i++) for(let j=i+1;j<S.length&&S[j].d===S[i].d;j++)
    if(sg(nhlCmp(S[i],S[j]))!==-sg(nhlCmp(S[j],S[i]))) P.cmpBad++;
  // back-link trail: a browser Back pops, a new page pushes, never a ping-pong
  nhlTrail.length=0; const tr=[];
  const step=(o,h)=>{nhlTrailStep(o,h); tr.push(nhlTrail.slice());};
  step("#/nhl","#/game/nhl-1"); step("#/game/nhl-1","#/nhl/team/MTL"); step("#/nhl/team/MTL","#/game/nhl-1");
  step("#/game/nhl-1","#/nhl"); P.trail=tr; nhlTrail.length=0;
  T("game:missing",()=>nhlGamePage("123"));
  P.teams={};
  for(const c of Object.keys(n.teams)){const h=T("team:"+c,()=>nhlTeamPage(c));
    P.teams[c]={rows:(h.match(/location\.hash='#\/nhl\/player\//g)||[]).length,roster:(n.teams[c].roster||[]).length};}
  T("team:missing",()=>nhlTeamPage("ZZZ"));
  P.players={n:0,badNR:[],badG:[],badSk:[]};
  for(const [id,p] of Object.entries(n.players)){const h=T("player:"+id,()=>nhlPlayerPage(id)); P.players.n++;
    if(p.rating==null&&!/Not rated: /.test(h)) P.players.badNR.push(id);
    if(p.grp==="G"&&p.rating!=null&&!/GSAx\/60/.test(h)) P.players.badG.push(id);
    if(p.grp!=="G"&&p.rating!=null&&!/Net xG\/60/.test(h)) P.players.badSk.push(id);}
  for(const pos of ["all","F","C","L","R","D","G"]) for(const mn of ["0","2000","3500","50","150","nr"])
    for(const s of ["r","off","def","toi","gsax","gp","net"]){
      state.nhlPos=pos; state.nhlMin=mn; state.nhlSort=s; T(`players:${pos}:${mn}:${s}`,()=>nhlPlayers());}
  state.nhlPos="all"; state.nhlMin="0"; state.nhlSort="r"; P.ladder=T("ladder",()=>nhlPlayers());
  P.std={};
  for(const S of ["cur","proj","prev"]) for(const V of ["div","wc","league"]){
    state.nhlStd=S; state.nhlStdView=V; P.std[S+":"+V]=T(`std:${S}:${V}`,()=>nhlStandings()); P.std[S+":"+V+":shown"]=state.nhlStd;}
  for(const s of ["elo","gb","proj"]){state.nhlTeamSort=s; T("teams:"+s,()=>nhlTeams());}
  P.team0=T("team:first",()=>nhlTeamPage(Object.keys(n.teams)[0]));
  P.offCards=n.schedule.filter(g=>nhlOff(g)).map(g=>nhlCard(g));
  P.valCards=n.schedule.filter(g=>g.value).map(g=>[nhlCard(g),nhlBadge(g)]);
  state.league="mlb";
  for(const h of ["#/nhl","#/nhl/standings","#/nhl/teams","#/nhl/players","#/nhl/team/COL",
      "#/nhl/player/"+Object.keys(n.players)[0],"#/game/nhl-"+n.schedule[0].id,"#/nhl/player/99999999"])
    T("route:"+h,()=>{location.hash=h; route(true);});
  console.log(JSON.stringify(out));
})();
"""


def _sweep(tmp_path, payload: Path) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    for f in ("board.json", "db.json"):
        if not (DATA / f).is_file():
            pytest.skip(f"site/data/{f} not built")
    script = tmp_path / "nhl_pages.js"
    script.write_text(STUB + JS.replace("\nboot();", "\n") + BODY, encoding="utf-8")
    env = dict(__import__("os").environ,
               NHL_PAGES_ARGS=json.dumps({"data": str(DATA), "payload": str(payload)}))
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=240,
                       env=env, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _served() -> dict:
    p = DATA / "nhl.json"
    if not p.is_file():
        pytest.skip("site/data/nhl.json not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _in_season(n: dict) -> dict:
    """The served payload moved a few weeks into the season: finals with OT/SO,
    a replayed row, a playoff final, current records and current stat lines."""
    n = copy.deepcopy(n)
    S = n["schedule"]
    played = S[:40]
    for i, g in enumerate(played):
        h, a = (i * 7) % 6, (i * 5) % 5
        if h == a:
            h += 1
        g.update(hs=h, **{"as": a}, y=int(h > a), last=("REG", "REG", "OT", "SO")[i % 4],
                 tier="EARLY")
        g.pop("value", None)
        if i % 9 == 0:
            g["replay"] = True
        else:
            g["frozen_at"] = "2026-09-29T20:00:00Z"
    po = copy.deepcopy(S[3])
    po.update(id=2026030111, playoff=1, d="2027-04-20", hs=4, last="OT", y=1)
    po["as"] = 2
    S.append(po)
    # value layer: an EARLY badge the edge layer disclosed on purpose (GATE_EARLY
    # off stamps value.tier), and a stale pre-gate badge with no tier
    val = {"side": "home", "team": S[40]["home"], "ev_open": 0.05, "ev_cur": 0.05,
           "open_dec": 1.9, "cur_dec": 1.9, "available": True, "books": 8}
    S[40]["value"] = dict(val, tier="EARLY", model_ts="2026-09-29T12:00:00Z")
    S[41]["value"] = dict(val, team=S[41]["home"])
    S[41].pop("tier", None)
    # a postponed game (nhl_serve stamps ppd:1 and the NHL API state)
    S[42].update(ppd=1, state="PPD")
    for c, t in n["teams"].items():
        gs = [g for g in played if c in (g["home"], g["away"])]
        w = l = o = gf = ga = 0
        for g in gs:
            me, op = (g["hs"], g["as"]) if g["home"] == c else (g["as"], g["hs"])
            gf, ga = gf + me, ga + op
            if me > op:
                w += 1
            elif g["last"] != "REG":
                o += 1
            else:
                l += 1
        t.update(gp=len(gs), w=w, l=l, otl=o, gf=gf, ga=ga, pts=2 * w + o, gd=gf - ga,
                 pts_pct=round((2 * w + o) / (2 * len(gs)), 3) if gs else None,
                 rw=w, row=w, l10=f"{w}-{l}-{o}", streak="W1", home="1-0-0", away="0-1-0",
                 div_rank=None, conf_rank=None, league_rank=None, wc_rank=None)
    for i, p in enumerate(n["players"].values()):
        if i % 3 == 0 and p["stats"].get("prev"):
            p["stats"]["cur"] = dict(p["stats"]["prev"])
            p["stats"]["main"] = "cur"
    n["model_card"].update(cur_season_acc=0.55, cur_season_n=36, cur_season_ll=0.68)
    n["phase"] = "regular"
    return n


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    n = _served()
    return n, _sweep(tmp_path_factory.mktemp("nhl_served"), DATA / "nhl.json")


@pytest.fixture(scope="module")
def season(tmp_path_factory):
    n = _in_season(_served())
    d = tmp_path_factory.mktemp("nhl_season")
    p = d / "nhl.json"
    p.write_text(json.dumps(n), encoding="utf-8")
    return n, _sweep(d, p)


def _clean(out):
    assert out["n"] > 1000
    assert out["throws"] == [], out["throws"][:5]
    assert out["bad"] == [], out["bad"][:5]


def test_every_nhl_page_renders_on_the_served_payload(served):
    _clean(served[1])


def test_every_nhl_page_renders_in_season(season):
    _clean(season[1])


def test_early_forecasts_are_never_picks(served, season):
    for n, out in (served, season):
        early = all(g.get("tier", "EARLY") == "EARLY" for g in n["schedule"])
        assert early, "an NHL row is no longer EARLY - revisit these assertions"
        for g in out["pol"]["games"]:
            assert not g["fair"], f"fair odds on EARLY game {g['id']}"
            assert not g["pickPill"], f"pick pill on EARLY game {g['id']}"
            assert not g["pickHit"], f"PICK HIT/MISS on EARLY game {g['id']}"
            assert g["notPick"], f"game {g['id']} does not say it is not a pick"
        for r, h in out["pol"]["board"].items():
            assert 'class="odds"' not in h, f"fair odds on the {r} board"
            assert "PICK &middot;" not in h and "LEAN &middot;" not in h, r
            assert "PICK HIT" not in h and "PICK MISS" not in h, r


def test_board_opens_on_scheduled_leans_and_grades_played_leans(served, season):
    wk = served[1]["pol"]["board"]["week"]
    assert "SCHEDULED" in wk and "Not a pick" in wk and "Track record by tier" in wk
    res = season[1]["pol"]["board"]["results"]
    assert "Final " in res and re.search(r"lean &#1000[37];|>replay<", res)


def test_edge_bar_keeps_the_goalie_disclosure(served, season):
    for n, out in (served, season):
        games = out["pol"]["games"]
        for g in games:
            if g["badge"]:
                assert g["valbar"] and g["goalieConf"], f"EDGE bar without goalie disclosure on {g['id']}"
            else:
                assert not g["valbar"], f"EDGE bar on {g['id']}, which the edge layer never cleared"
        for h, on in out["pol"]["valCards"]:
            assert ("no goalie conf." in h) == on and ('class="valbar"' in h) == on


def test_stale_badges_never_show_on_early_rows(served, season):
    """A badge on an EARLY row shows only when the edge layer disclosed it
    (value.tier == "EARLY"); a pre-gate badge (no tier, priced on an older
    forecast) never does - not on the game page, not on the board."""
    for n, out in (served, season):
        rows = {g["id"]: g for g in n["schedule"]}
        for g in out["pol"]["games"]:
            v = rows[g["id"]].get("value") or {}
            want = bool(v.get("available")) and (rows[g["id"]].get("tier", "EARLY") != "EARLY"
                                                 or v.get("tier") == "EARLY")
            assert g["badge"] == want, g["id"]
    # the in-season copy carries one disclosed and one stale badge
    sg = {g["id"]: g for g in season[1]["pol"]["games"]}
    S = season[0]["schedule"]
    assert sg[S[40]["id"]]["badge"] and sg[S[40]["id"]]["valbar"]
    assert sg[S[41]["id"]]["value"] and not sg[S[41]["id"]]["valbar"]


def test_contribution_copy_is_a_true_equation(served, season):
    """'50% + margin = home probability' holds exactly as printed; the bars are
    rounded to 0.1 pt and, where their sum differs, the page says what it is."""
    for n, out in (served, season):
        for g in out["pol"]["games"]:
            assert len(g["sum"]) == 3, g["id"]
            sign, margin, shown = g["sum"]
            assert abs(float(shown) - 100 * g["hp"]) <= 0.051, (g["id"], shown, g["hp"])
            assert f"{50 + (1 if sign == '+' else -1) * float(margin):.1f}" == shown, (g["id"], g["sum"])
            m10 = round((1 if sign == "+" else -1) * float(margin) * 10)
            s10 = round(g["ctSum"] * 10)
            if g["bars"]:                       # printed only when the rounded bars miss
                bsign, bval = g["bars"]
                assert round((-1 if bsign == "&minus;" else 1) * float(bval) * 10) == s10 != m10, g["id"]
                assert abs(s10 - m10) <= 3, (g["id"], s10, m10)
            else:
                assert s10 == m10, (g["id"], g["ctSum"], g["sum"])


def test_two_sides_add_to_100(served, season):
    for n, out in (served, season):
        for g in out["pol"]["games"]:
            assert len(g["pct"]) == 2 and sum(g["pct"]) == 100, (g["id"], g["pct"])


def test_replay_games_print_no_log_loss(season):
    reps = [g for g in season[1]["pol"]["games"] if g["replay"] and g["played"]]
    assert reps
    for g in reps:
        assert not g["logLoss"], g["id"]
    rated = [g for g in season[1]["pol"]["games"] if g["played"] and not g["replay"]]
    assert rated and all(g["logLoss"] for g in rated)


def test_postponed_games_say_so(season):
    offs = [g for g in season[1]["pol"]["games"] if g["off"]]
    assert offs
    for g in offs:
        assert g["offShown"], g["id"]
    for h in season[1]["pol"]["offCards"]:
        assert "Postponed &mdash; new date to be announced. Not a pick." in h
        assert "Current lean" not in h and 'class="lv"' not in h


def test_board_order_and_back_trail(served):
    pol = served[1]["pol"]
    assert pol["cmpBad"] == 0
    assert pol["trail"] == [["#/nhl", "#/game/nhl-1"], ["#/nhl", "#/game/nhl-1", "#/nhl/team/MTL"],
                            ["#/nhl", "#/game/nhl-1"], ["#/nhl"]]


def test_phone_width_guards(served):
    """At 375px a grid item keeps its table's natural width and a chip row that
    cannot wrap pushes the page sideways; the NHL containers carry the classes
    nhl.css uses to stop both."""
    pol = served[1]["pol"]
    css = (ROOT / "mlbwp_site" / "css" / "nhl.css").read_text(encoding="utf-8")
    assert ".nhl-cols>*{min-width:0}" in css and ".nhl-chips .filters{flex-wrap:wrap}" in css
    assert 'class="cols2 nhl-cols"' in pol["board"]["week"]
    assert 'class="cols2 nhl-cols"' in pol["team0"]
    assert 'class="controls nhl-chips"' in pol["ladder"]


def test_team_pages_list_the_whole_roster(served):
    for c, t in served[1]["pol"]["teams"].items():
        # every rostered player has a row; the key-skater strip is not on this page
        assert t["rows"] >= t["roster"], (c, t)


def test_skater_xg_keeps_two_decimals(served):
    n, out = served
    lad = out["pol"]["ladder"]
    top = max((p for p in n["players"].values() if p["grp"] != "G" and p["rating"] is not None),
              key=lambda p: p["rating"])
    want = ("+" if top["net"] > 0 else "&minus;") + f"{abs(top['net']):.2f}"
    assert want in lad, want
    # one decimal collapsed 558 skaters into ten strings; never back
    assert not re.search(r'class="num (pos|neg)">(\+|&minus;)0\.\d<', lad)


def test_player_pages_cover_goalies_and_unrated(served):
    """Every rostered player has a page: rated goalies on GSAx, rated skaters on
    xG/60, and an unrated player says why in words (580 of these threw before)."""
    n, out = served
    pl = out["pol"]["players"]
    assert pl["n"] == len(n["players"])
    assert pl["badNR"] == [] and pl["badG"] == [] and pl["badSk"] == []


def test_standings_never_a_wall_of_zero_records(served):
    n, out = served
    if not any((t.get("gp") or 0) > 0 for t in n["teams"].values()):
        std = out["pol"]["std"]
        assert std["cur:div:shown"] in ("proj", "prev")         # the empty table is not offered
        for k in ("proj:div", "prev:div", "prev:wc"):
            assert "0-0-0" not in std[k], k
        assert "regular season opens" in std["proj:div"]
        assert "Wild card" in std["prev:wc"] and "nhl-cut" in std["prev:wc"]
