"""The NFL pages, rendered: the real page script runs under Node against the
shipped payloads (site/data/*.json) and against a small synthetic schedule, and
the tests read back what each renderer wrote into #view.

What is pinned is the honesty contract of documents/pick_policy.md as it shows
on the NFL surfaces (board cards, game page, season table, team schedule):
  - an EARLY forecast is a SCHEDULED game - no PICK/LEAN pill, no fair price,
    and a played EARLY game is graded as a lean, never HIT/MISS;
  - a PROJECTED forecast inside 55/45 is a LEAN, outside it a PICK;
  - a played game shows the pre-game number (ph), which equals the pre-game
    ledger's value;
  - the exact breakdown (cx) adds up to the number it explains, and only it is
    drawn as "exact"; the slope-scaled ct fallback says it does not add up;
  - labels come from the payload's seasons, never a hard-coded year;
  - links are league-qualified (#/nfl/...), so shared URLs open the NFL.
Skipped when Node or the payload is absent.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mlbwp_site.build_site import JS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"

HARNESS = r"""
const el=()=>({onclick:null,oninput:null,innerHTML:"",textContent:"",title:"",className:"",style:{},dataset:{},value:"",
  classList:{toggle(){},add(){},remove(){},contains(){return false;}},
  setAttribute(){},getAttribute(){return null;},removeAttribute(){},
  querySelector(){return null;},querySelectorAll(){return [];},closest(){return null;}});
const store={}, els={};
globalThis.document={documentElement:el(),hidden:false,querySelector:s=>els[s]||(els[s]=el()),querySelectorAll:()=>[],
  getElementById:()=>el(),addEventListener(){}};
const view=()=>document.querySelector("#view").innerHTML;
globalThis.window=globalThis; globalThis.addEventListener=()=>{};
globalThis.localStorage={getItem:k=>store[k]??null,setItem:(k,v)=>{store[k]=String(v);}};
globalThis.location={hash:"",pathname:"/",search:""};
globalThis.history={state:null,replaceState(){},pushState(){}};
globalThis.MutationObserver=class{observe(){}};
globalThis.fetch=()=>new Promise(()=>{});
globalThis.scrollTo=()=>{};
__JS__
;(function(){
  const fs=require("fs"), D=__DATA__;
  const rd=f=>{try{return JSON.parse(fs.readFileSync(D+"/"+f,"utf8"));}catch(e){return null;}};
  const out={}, txt=h=>h.replace(/<[^>]+>/g," ");
  const R=(k,f)=>{try{f(); out[k]=view();}catch(e){out[k]="THREW "+(e&&e.stack||e);}};
  state.board=rd("board.json"); state.db=rd("db.json"); state.nhl=rd("nhl.json");
  if(state.board) siteLeaguesTidy();
  const today=nflToday(), plus=n=>addDays(today,n);
  // ---------- synthetic schedule: every tier / call / result combination ----------
  const T=c=>({code:c,abbr:c,name:c+" Team",nick:c,div:"AFC East",elo:1500,rank:1,w:1,l:0,t:0,gp:1,pct:1,pf:20,pa:10,
    pt_diff:10,home:"1-0",away:"0-0",div_rec:"0-0",streak:"W1",luck:0.3,div_rank:1,glassbox:55,gb_off:55,gb_def:55,
    roster:[],reserve:[],practice:[],inj:[],inj_week:3,proj:{w:9,sd:2,div:40,po:50}});
  const cx={base:3.9,elo:4.0,qb:2.0,units:0.5,roster:0.4,ts:0.2,hfa:0,sched:0,luck:0,abs:0,avail:0};
  state.nfl={status:"season",season:2026,stats_season:2026,stats_prev_season:2025,standings_season:2026,
    standings_through:{w:2,d:plus(-3),n:4},divisions:["AFC East"],players:{},proj:{},
    model_card:{test_log_loss:0.619,close_log_loss:0.609,calibration:[],serve:{sims:20000,cx_order:["base","elo","qb","units","roster","ts","hfa","sched","luck","abs","avail"]},ratings:{plays:1,seasons:"2016-2025",through_season:2025}},
    teams:{AAA:T("AAA"),BBB:T("BBB"),CCC:T("CCC"),DDD:T("DDD")},
    schedule:[
      {w:1,d:plus(-10),t:"13:00",home:"AAA",away:"BBB",ph:0.61,pmc:0.61,hs:20,as:10,tier:"EARLY",frozen_at:"2026-07-30T23:57:40Z",cx},
      {w:2,d:plus(-3),t:"13:00",home:"CCC",away:"DDD",ph:0.66,pmc:0.62,hs:10,as:20,tier:"PROJECTED",frozen_at:plus(-5)+"T12:00:00Z",cx:{...cx,elo:10.2,qb:1.9}},
      {w:2,d:plus(-3),t:"16:25",home:"AAA",away:"DDD",ph:0.53,pmc:0.53,hs:24,as:17,tier:"PROJECTED",frozen_at:plus(-5)+"T12:00:00Z"},
      {w:2,d:plus(-3),t:"20:20",home:"BBB",away:"CCC",ph:0.60,pmc:0.60,hs:27,as:3,replay:true,tier:"EARLY"},
      {w:3,d:plus(2),t:"13:00",home:"AAA",away:"CCC",ph:0.53,pmc:0.52,hs:null,as:null,tier:"PROJECTED",ct:{elo:2}},
      {w:3,d:plus(2),t:"16:05",home:"BBB",away:"DDD",ph:0.70,pmc:0.68,hs:null,as:null,tier:"PROJECTED",cx:{...cx,elo:8.1,qb:4.1,roster:3.9}},
      {w:3,d:plus(-1),t:"20:15",home:"CCC",away:"BBB",ph:0.70,pmc:0.66,hs:null,as:null,tier:"PROJECTED",frozen_at:plus(-3)+"T12:00:00Z",
        value:{available:true,team:"CCC",ev_cur:0.05,cur_dec:1.8}},
      {w:4,d:plus(6),t:"13:00",home:"DDD",away:"CCC",ph:0.60,pmc:0.55,hs:null,as:null,tier:"EARLY"},
      {w:9,d:plus(40),t:"20:20",home:"DDD",away:"AAA",ph:0.58,pmc:0.45,hs:null,as:null,tier:"EARLY",cx:{...cx,elo:1.5,qb:1.3,units:1.3,roster:0}}]};
  state.league="nfl"; state.nflRange="year";
  R("syn_board",()=>nflPage());
  const S=state.nfl.schedule, key=g=>g.w+"_"+g.away+"_"+g.home;
  S.forEach(g=>R("syn_game_"+key(g),()=>nflGamePage(key(g))));
  R("syn_season2",()=>nflSeason("2")); R("syn_season9",()=>nflSeason("9")); R("syn_season3",()=>nflSeason("3"));
  R("syn_team",()=>nflTeamPage("AAA")); R("syn_team_bbb",()=>nflTeamPage("BBB"));
  // ---------- the shipped payload ----------
  const n=rd("nfl.json");
  if(n&&n.schedule&&n.schedule.length){
    state.nfl=n; const before=JSON.stringify(n.schedule);
    out.real=true;
    const bad=[];
    const chk=(k,h)=>{if(/^THREW/.test(h)||/\b(undefined|NaN|null)\b/.test(txt(h))) bad.push(k+": "+String(h).slice(0,300));};
    ["today","week","month","year"].forEach(r=>{state.nflRange=r; state.nflWk=null; R("board_"+r,()=>nflPage()); chk("board_"+r,out["board_"+r]);});
    R("season",()=>nflSeason(undefined)); chk("season",out.season);
    [...new Set(n.schedule.map(g=>g.w))].forEach(w=>{R("sw",()=>nflSeason(String(w))); chk("season_"+w,out.sw);});
    R("standings",()=>nflStandings()); chk("standings",out.standings);
    R("teams",()=>nflTeams()); chk("teams",out.teams);
    R("players",()=>nflPlayers()); chk("players",out.players);
    ["QB","RB","WR","TE","OL","DL","LB","DB"].forEach(k=>{state.posMin=0; R("pos",()=>nflPosPage(k)); chk("pos_"+k,out.pos);});
    Object.keys(n.teams).forEach(c=>{R("tp",()=>nflTeamPage(c)); chk("team_"+c,out.tp);});
    Object.keys(n.players).forEach(id=>{R("pp",()=>nflPlayerPage(id)); chk("player_"+id,out.pp);});
    out.games={};
    n.schedule.forEach(g=>{const k=key(g); R("gp",()=>nflGamePage(k)); chk("game_"+k,out.gp);
      const h=out.gp, c=nflCall(g);
      out.games[k]={tier:c.t,call:c.call,done:c.done,hp:c.hp,ph:g.ph,overdue:!c.done&&g.d<nflToday(),
        big:(/<div class="big">(\d+)</.exec(h)||[])[1], fair:/Fair odds/.test(h), exact:/exact contributions/.test(h),
        sched:/Scheduled &mdash; current lean/.test(h), hitmiss:/chip (hit|miss)"/.test(h), lean:/chip lean"/.test(h)};});
    state.nflRange="year"; R("board_all",()=>nflPage());
    out.cards={};
    out.board_all.split('<a class="gc').slice(1).forEach(a=>{const m=/href="#\/game\/([^"]+)"/.exec(a); if(!m) return;
      out.cards[m[1]]={odds:/class="odds"/.test(a), sched:/SCHEDULED/.test(a), pick:/>PICK &middot;/.test(a), lean:/>LEAN &middot;/.test(a),
        hitmiss:/chip (hit|miss)"/.test(a)};});
    out.bad=bad; out.mutated=JSON.stringify(n.schedule)!==before;
  }
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    p = tmp_path_factory.mktemp("js") / "nfl_pages.js"
    src = HARNESS.replace("__DATA__", json.dumps(str(DATA).replace("\\", "/")))
    p.write_text(src.replace("__JS__", JS.replace("\nboot();", "\n")), encoding="utf-8")
    r = subprocess.run([node, str(p)], capture_output=True, text=True, timeout=600,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _txt(h: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))


# ------------------------------------------------------------- synthetic
def test_synthetic_pages_render(page):
    for k, v in page.items():
        if k.startswith("syn_"):
            assert not v.startswith("THREW"), (k, v[:500])


def test_early_card_is_scheduled_with_no_price(page):
    board = page["syn_board"]
    cards = {m.group(1): a for a in board.split('<a class="gc')[1:]
             for m in [re.search(r'href="#/game/([^"]+)"', a)] if m}
    far = cards["9_AAA_DDD"]
    assert "SCHEDULED" in far and 'class="odds"' not in far
    assert ">PICK &middot;" not in far and ">LEAN &middot;" not in far
    assert "early" in far.split(">", 1)[0]                   # dashed EARLY card
    assert "Not a pick" in far and "AAA 55%" in far           # the lean is the season sim (pmc)
    # PROJECTED: PICK outside 55/45, LEAN inside, fair price on both
    assert ">PICK &middot; BBB 70%" in cards["3_DDD_BBB"] and 'class="odds"' in cards["3_DDD_BBB"]
    assert ">LEAN &middot; AAA 53%" in cards["3_CCC_AAA"]


def test_played_games_are_graded_by_tier(page):
    board = page["syn_board"]
    cards = {m.group(1): a for a in board.split('<a class="gc')[1:]
             for m in [re.search(r'href="#/game/([^"]+)"', a)] if m}
    early = cards["1_BBB_AAA"]                                 # EARLY final: a lean, never HIT
    assert 'chip lean"' in early and 'chip hit"' not in early and 'class="odds"' not in early
    assert 'chip miss"' in cards["2_DDD_CCC"]                  # PROJECTED pick that missed
    assert 'chip lean"' in cards["2_DDD_AAA"]                  # PROJECTED inside 55/45
    assert "REPLAY" in cards["2_CCC_BBB"]
    # the played card shows the pre-game number and the score, no price
    assert ">61%<" in early and "nsc" in early


def test_game_page_tier_presentation(page):
    far = page["syn_game_9_AAA_DDD"]
    assert "Scheduled &mdash; current lean" in far and "Fair odds" not in far
    assert "Lean: AAA 55%" in _txt(far)
    # bars explain the live model's number, and the page says the lean is the sim
    assert "exact contributions" in far and "of the live model's 58%" in far
    near = page["syn_game_3_DDD_BBB"]
    assert "Fair odds" in near and "Pick: BBB" in near and "Our projection" in near
    lean = page["syn_game_3_CCC_AAA"]
    assert "Lean: AAA" in lean and "inside 55/45" in lean
    # no cx on the row: the ct fallback, which says it does not add up
    assert "blend contributions" in lean and "not</b> add up" in lean
    fin = page["syn_game_2_DDD_CCC"]
    assert 'class="livepanel"' in fin and "Final" in fin and "MISS" in fin
    assert "log loss" in fin and "Pre-game number locked" in fin
    assert "Lineup panels show today's depth chart" in fin     # no current lineup on a played game


def test_season_and_team_tables(page):
    s2 = page["syn_season2"]
    assert "CCC 10&ndash;20 DDD" not in s2                     # away-first score
    assert "DDD 20&ndash;10 CCC" in s2 and 'chip miss"' in s2
    s9 = page["syn_season9"]
    assert "EARLY" in s9 and "lean</span> <span class=\"ab\">AAA" in s9
    team = page["syn_team"]
    assert "bye" in page["syn_team_bbb"]                        # BBB has no week-9 game
    assert "#/nfl/team/CCC" in team and "#/nfl/team/DDD" in team      # opponents, league-qualified
    assert "W 20-17" not in team and "W 24-17" in team                  # AAA's result, from AAA's side


def _cards(board: str) -> dict:
    return {m.group(1): a for a in board.split('<a class="gc')[1:]
            for m in [re.search(r'href="#/game/([^"]+)"', a)] if m}


def test_overdue_game_keeps_its_call_but_no_price(page):
    """Played, result not in the build yet: the published call stays, the
    fair price and the EDGE badge go."""
    card = _cards(page["syn_board"])["3_BBB_CCC"]
    assert ">PICK &middot; CCC 70%" in card and "awaiting result" in card
    assert 'class="odds"' not in card and 'class="valbar"' not in card
    gp = page["syn_game_3_BBB_CCC"]
    assert "Fair odds" not in gp and 'class="valbar"' not in gp
    assert "Pre-game forecast" in gp and "result not in this build yet" in gp
    assert "graded as a pick" in gp and "it becomes a pick" not in gp


def test_early_badge_names_the_number_shown(page):
    cards = _cards(page["syn_board"])
    near = cards["4_CCC_DDD"]                  # inside a calendar week: the live ph
    assert "Early read" in near and "Season sim" not in near and "DDD 60%" in near
    assert "SCHEDULED" in near and 'class="odds"' not in near
    far = cards["9_AAA_DDD"]                   # beyond: the season simulation
    assert "Season sim" in far and "Early read" not in far


def test_counts_keep_leans_out_of_picks(page):
    board = page["syn_board"]
    # unplayed: PICK 3_DDD_BBB, PICK 3_BBB_CCC (overdue), LEAN 3_CCC_AAA, EARLY x2
    assert "2 picks &middot; 1 lean &middot; 2 scheduled" in board
    assert "1 pick &middot; 1 lean" in board                          # the day with one of each
    # the model paragraph sits under the season outlook, after the cards
    assert board.index("feature blend around an 11-vs-11") > board.rindex('<a class="gc')
    s3 = _txt(page["syn_season3"])
    assert "2 picks · 1 lean" in s3.replace("&middot;", "·")


# ------------------------------------------------------------- shipped payload
def _real(page):
    if not page.get("real"):
        pytest.skip("site/data/nfl.json not built")
    return page


def test_every_nfl_page_renders_clean(page):
    p = _real(page)
    assert not p["bad"], p["bad"][:5]
    assert not p["mutated"], "rendering changed the payload"


def test_shipped_cards_follow_the_policy(page):
    p = _real(page)
    lean_max = 0.55
    for k, g in p["games"].items():
        card = p["cards"].get(k)
        assert card is not None, k
        if g["done"]:
            assert not card["odds"], k
            if g["tier"] == "EARLY":
                assert not card["hitmiss"] and not g["hitmiss"], k
            continue
        if g["overdue"]:                          # played, result not in yet: no price
            assert not card["odds"] and not g["fair"], k
            continue
        if g["tier"] == "EARLY":
            assert card["sched"] and not card["odds"] and not (card["pick"] or card["lean"]), k
            assert g["sched"] and not g["fair"], k
        else:
            pp = max(g["hp"], 1 - g["hp"])
            assert card["lean"] == (pp <= lean_max) and card["pick"] == (pp > lean_max), k
            assert card["odds"] and g["fair"], k


def test_played_game_shows_the_ledger_number(page):
    """Rule 3: a played game's page shows its pre-game ph, and that ph is the
    pre-game ledger's value."""
    p = _real(page)
    ledger_path = ROOT / "data" / "nfl_ph_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
    for k, g in p["games"].items():
        if not g["done"]:
            continue
        assert g["hp"] == g["ph"], k
        assert int(g["big"]) == math.floor(g["ph"] * 100 + 0.5), k      # JS Math.round
        w, away, home = k.split("_")
        ent = ledger.get(f"{w}|{home}|{away}")
        if ledger:
            assert ent is not None and ent["ph"] == g["ph"], k


def test_shipped_breakdown_is_exact():
    n = json.loads((DATA / "nfl.json").read_text(encoding="utf-8")) if (DATA / "nfl.json").is_file() else None
    if not n or not n.get("schedule"):
        pytest.skip("site/data/nfl.json not built")
    rows = [g for g in n["schedule"] if g.get("cx")]
    assert rows, "no exact breakdown served"
    for g in rows:
        assert abs(50 + sum(g["cx"].values()) - 100 * g["ph"]) < 0.06, g["id"]


def test_labels_come_from_the_payload():
    src = (ROOT / "mlbwp_site" / "js" / "nfl.js").read_text(encoding="utf-8")
    for lit in ("2025 season", "2025 record", "2025 final", "in 2025", "kicks off September",
                "344,801", "Elo 26"):
        assert lit not in src, f"hard-coded label is back: {lit}"
    assert all(ord(ch) < 128 for ch in src), "non-ASCII in nfl.js: the build entity-encodes it inside <script>"


def test_nfl_links_are_league_qualified():
    src = (ROOT / "mlbwp_site" / "js" / "nfl.js").read_text(encoding="utf-8")
    assert "location.hash='#/team/" not in src and "location.hash='#/player/" not in src
    assert 'href="#/team/' not in src and 'href="#/player/' not in src


def test_result_block_and_status_chip_do_not_share_a_class():
    """`.nfl-res` is the reserve / practice-squad status chip only; the played
    game's result block is `.nfl-result`, so its flex/margin rules never reach
    an inline chip."""
    css = (ROOT / "mlbwp_site" / "css" / "nfl.css").read_text(encoding="utf-8")
    rest = css.replace(".chip.nfl-res{", "")
    assert not re.search(r"\.nfl-res(?![\w-])", rest), "a bare .nfl-res rule would restyle the status chip"
    assert ".nfl-result{" in css
    src = (ROOT / "mlbwp_site" / "js" / "nfl.js").read_text(encoding="utf-8")
    assert 'class="nfl-res"' not in src and 'class="nfl-result"' in src


def test_one_team_label_and_honest_copy():
    src = (ROOT / "mlbwp_site" / "js" / "nfl.js").read_text(encoding="utf-8")
    assert "t.abbr" not in src, "teams are labelled by their code everywhere (LA, not LAR on some pages)"
    assert "it means 75%" not in src, "the calibration table does not support that claim"
    assert '"current depth chart":' not in src    # the lineup panel already prints 'depth chart <date>'
