"""NHL ratings on the pages (package nhl-ratings), rendered for real under Node -
the harness of tests/test_nhl_pages_frontend.py (the fully substituted site
script, a minimal DOM stub, the real site/data payloads).

Every page is drawn - teams index in every sort, every team page, every game
page, standings in every season and view, the players index in every position /
sort / minimum, every player page - on the served payload and on a copy WITHOUT
the player-value snapshot (the serve's degrade path). Pinned:

  P1  nothing throws; no undefined / NaN / null / [object;
  P2  no lineup GlassBox, no "by player RAPM", no skater or team 0-100 GlassBox
      badge or column, no pooled "Rank of" on a skater page, no "0-100 on this
      on-ice xG rating"; nhl.js has no reader of the lineup GlassBox fields;
  P3  teams index: the default order is the team-strength rank; each card's chip
      is win% (1 dp) and #rank from the Python formula; the sort chips are
      exactly Team strength / Projected points / Results (Elo) / Shot quality (xG);
  P4  team page: the header is the strength and "#k of 32"; Results (Elo) and
      Shot quality tiles; "Skater lineup value" only inside the display-only
      Roster panel, with no rank; the offseason line when the snapshot carries
      the 2025-26 lineup; key skaters ordered by value per game;
  P5  game page: the model block is Team strength, Results (Elo), Shot quality -
      no lineup value, no GlassBox; the neutral-edge + home + rest line adds up
      to the displayed model % and agrees with the served decomposition (ct);
  P6  skater page: header percentile = pv.p; Offence (qo) and Defence (q.def);
      provisional tag iff prov; the defence-reliability sentence; the RAPM panel
      titled "On-ice 5v5 xG impact". Goalie page: "Goalie rating (GSAx)" and the
      latest-season GSAx;
  P7  players index: value is the default sort; Offence and Defence columns; no
      skater GlassBox header; goalies under "Goalie rating";
  P8  standings: the projected table has a Strength column and no Lineup column.
Skipped when Node is absent.
"""
from __future__ import annotations

import copy
import html as H
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mlbwp_site.build_site import JS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"
NHL_JS = ROOT / "mlbwp_site" / "js" / "nhl.js"
_spec = importlib.util.spec_from_file_location("nhl_pages_frontend",
                                               ROOT / "tests" / "test_nhl_pages_frontend.py")
FE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FE)
_spec2 = importlib.util.spec_from_file_location("nhl_team_strength",
                                                ROOT / "tests" / "test_nhl_team_strength.py")
TS = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(TS)

BODY = r"""
;(function(){
  const fs=require("fs"), A=JSON.parse(process.env.NHL_RT_ARGS);
  const view=()=>document.querySelector("#view").innerHTML;
  const J=f=>JSON.parse(fs.readFileSync(f,"utf8"));
  state.board=J(A.data+"/board.json"); state.db=J(A.data+"/db.json");
  try{state.nfl=J(A.data+"/nfl.json");}catch(e){state.nfl=null;}
  state.nhl=J(A.payload); siteLeaguesTidy(); state.league="nhl";
  const out={n:0,throws:[],bad:[],forbid:[],r:{}};
  const BAD=/undefined|NaN|\bnull\b|\[object/;
  const FORBID=[/lineup GlassBox/i,/by player RAPM/,/GlassBox/i,/0-100 on this on-ice xG rating/];
  const T=(k,f,extra)=>{out.n++; try{f(); const h=view();
      const m=BAD.exec(h.replace(/data-[a-z]+="[^"]*"/g,""));
      if(m) out.bad.push([k,h.slice(Math.max(0,m.index-120),m.index+40)]);
      for(const re of FORBID.concat(extra||[])){const x=re.exec(h); if(x) out.forbid.push([k,String(re),h.slice(Math.max(0,x.index-80),x.index+60)]);}
      return h;}
    catch(e){out.throws.push([k,String(e&&e.stack||e).split("\n").slice(0,2).join(" | ")]); return "";}};
  const n=state.nhl, R=out.r;
  // teams index: every sort, the default order and each card's chip
  R.tindex={};
  for(const s of [null,"str","proj","elo","xg","gb","pv"]){state.nhlTeamSort=s; const h=T("teams:"+s,()=>nhlTeams());
    R.tindex[String(s)]={codes:[...h.matchAll(/class="tcard" onclick="location\.hash='#\/nhl\/team\/([A-Z]{3})'"/g)].map(m=>m[1]),
      chips:[...h.matchAll(/<span class="w">([0-9.]+)%<\/span><span class="rk">#(\d+)<\/span>/g)].map(m=>[+m[1],+m[2]]),
      sorts:[...h.matchAll(/data-tsort="([a-z]+)">([^<]+)<\/button>/g)].map(m=>[m[1],m[2]]),
      lit:[...h.matchAll(/class="filt on" data-tsort="([a-z]+)"/g)].map(m=>m[1])};}
  // team pages
  R.teams={};
  for(const c of Object.keys(n.teams)){const h=T("team:"+c,()=>nhlTeamPage(c));
    const hd=(h.split('<div class="tstats">')[0]||"");
    const i0=h.indexOf('class="panel nhl-roster"');
    let i1=-1; for(const mk of ['<div class="subh">Forwards</div>','<div class="subh">Defence</div>','<div class="subh">Goaltending</div>','Season columns:']){
      const j=h.indexOf(mk,i0); if(i0>=0&&j>=0&&(i1<0||j<i1)) i1=j;}
    const rp=i0>=0?h.slice(i0,i1>0?i1:undefined):"";
    const ks=(/<table class="nhl-keysk">([\s\S]*?)<\/table>/.exec(h)||[])[1]||"";
    R.teams[c]={strWin:(/<div class="nhl-strbig [a-z]+"><span class="num">([0-9.]+)%<\/span> <span class="rk">#(\d+) of (\d+)<\/span>/.exec(hd)||[]).slice(1),
      tElo:h.includes('<div class="k">Results (Elo)</div>'), tXg:h.includes('<div class="k">Shot quality, recent form (xG)</div>'),
      luAll:(h.match(/Skater lineup value/g)||[]).length, luPanel:(rp.match(/Skater lineup value/g)||[]).length,
      rpHead:(/<h3>([\s\S]*?)<\/h3>/.exec(rp)||[])[1]||"", rpRank:/#\d/.test(rp.replace(/#\/nhl\//g,"").replace(/&#\d+;/g,"")),
      chg:/Offseason roster change/.test(rp), key:[...ks.matchAll(/data-pid="(\d+)"/g)].map(m=>m[1])};}
  // game pages
  R.games={};
  for(const g of n.schedule){const h=T("game:"+g.id,()=>nhlGamePage(String(g.id)));
    const mb=(/model &mdash; team strength and its two halves<\/td><\/tr>([\s\S]*?)<\/table>/.exec(h)||[])[1]||"";
    const e=/Neutral-ice edge <b>(\S+) pts<\/b>[\s\S]*?home ice <b>(\S+)<\/b> \+ rest\/back-to-back <b>(\S+)<\/b> = model <b>([0-9.]+)%<\/b>/.exec(h);
    R.games[g.id]={model:!!mb, rows:[...mb.matchAll(/<td class="lbl">([^<]+)<\/td>/g)].map(m=>m[1]),
      lineupInModel:/lineup value|GlassBox/i.test(mb), edge:e?e.slice(1):null};}
  // standings: every season and view
  R.std={};
  for(const S of ["cur","proj","prev"]) for(const V of ["div","wc","league"]){
    state.nhlStd=S; state.nhlStdView=V; const h=T(`std:${S}:${V}`,()=>nhlStandings());
    R.std[S+":"+V]={shown:state.nhlStd, head:(/<thead>([\s\S]*?)<\/thead>/.exec(h)||[])[1]||""};}
  // players index: every position / sort / minimum
  const ids=h=>[...h.matchAll(/location\.hash='#\/nhl\/player\/(\d+)'/g)].map(m=>m[1]);
  state.nhlPos="all"; state.nhlMin="0"; state.nhlSort=null; const lad=T("ladder",()=>nhlPlayers());
  R.ladder={order:ids(lad), off:/data-nsort="qo"[^>]*>Offence<\/th>/.test(lad), def:/data-nsort="qd"[^>]*>Defence<\/th>/.test(lad),
    lit:[...lad.matchAll(/<th data-nsort="([^"]+)" class="nhl-srt on/g)].map(m=>m[1]),
    chipOn:[...lad.matchAll(/class="filt on" data-nsort="([^"]+)"/g)].map(m=>m[1])};
  state.nhlPos="G"; state.nhlSort=null; const gl=T("ladder:G",()=>nhlPlayers());
  R.goalieHead=/<th[^>]*>Goalie rating<\/th>/.test(gl);
  const sorts=["v","p","g","o","qo","qd","r","toi","net","off","def","gsax","gp","c_cre","c_def"];
  for(const pos of ["all","F","C","L","R","D","G"]) for(const mn of ["0","20","60","50","2000","nr"]) for(const s of sorts){
    state.nhlPos=pos; state.nhlMin=mn; state.nhlSort=s; T(`players:${pos}:${mn}:${s}`,()=>nhlPlayers());}
  // player pages
  R.players={};
  for(const [id,p] of Object.entries(n.players)){const h=T("player:"+id,()=>nhlPlayerPage(id),p.grp==="G"?[]:[/Rank of/]);
    const head=(/<div class="phead">([\s\S]*?)<\/div>/.exec(h)||[])[1]||"";
    const od=(/<div class="nhl-od">([\s\S]*?)<\/div><\/div>/.exec(h)||[])[0]||"";
    const odq=[...od.matchAll(/<span class="lab">([^<]+)<\/span>\s*(?:<span class="nhl-odn">([^<]+)<\/span>|<span class="nhl-odb[^"]*"><i[^>]*><\/i><\/span>\s*<span class="num nhl-odq">(\d+)(?:st|nd|rd|th)<\/span>)/g)]
      .map(m=>[m[1],m[2]||null,m[3]!=null?+m[3]:null]);
    R.players[id]={headPct:((/<span class="v">(\d+)(?:st|nd|rd|th)<\/span>/.exec(head))||[])[1]||null,
      od:odq, prov:(/Provisional: ([\d,]+) NHL game/.exec(h)||[])[1]||null,
      rel:/single-season reliability is only about/.test(h), rapmTitle:/<h3>On-ice 5v5 xG impact \(RAPM/.test(h),
      gRating:h.includes("Goalie rating (GSAx)"),
      gsaxLine:[...h.matchAll(/(\d{4})&ndash;(\d{2}): <b>([^<]+)<\/b> GSAx/g)].map(m=>[m[1],m[3]])};}
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
    script = tmp_path / "nhl_ratings_pages.js"
    script.write_text(FE.STUB + JS.replace("\nboot();", "\n") + BODY, encoding="utf-8")
    env = dict(os.environ, NHL_RT_ARGS=json.dumps({"data": str(DATA), "payload": str(payload)}))
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=900,
                       env=env, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _without_snapshot(n: dict) -> dict:
    n = copy.deepcopy(n)
    for p in n["players"].values():
        p["pv"] = None
    for t in n["teams"].values():
        t.pop("pv_lu", None)
    (n.get("rapm") or {}).pop("pv", None)
    return n


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    n = FE._served()
    if not any(p.get("pv") for p in n["players"].values()):
        pytest.skip("the served payload carries no player value")
    return n, _run(tmp_path_factory.mktemp("rt_served"), DATA / "nhl.json")


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    n = _without_snapshot(FE._served())
    d = tmp_path_factory.mktemp("rt_bare")
    p = d / "nhl.json"
    p.write_text(json.dumps(n), encoding="utf-8")
    return n, _run(d, p)


def _strength(n):
    m = json.loads((ROOT / "data" / "nhl_model.json").read_text(encoding="utf-8"))
    c = m["blend"]["coefs"]
    return TS.strength(n["teams"], c["elo_logit"], c["xg"])


# ------------------------------------------------------------ P1 / P2 ----
def test_p1_every_page_renders_cleanly(served, bare):
    for n, out in (served, bare):
        assert out["n"] > 3000
        assert out["throws"] == [], out["throws"][:5]
        assert out["bad"] == [], out["bad"][:5]


def test_p2_no_glassbox_no_rapm_rank_anywhere(served, bare):
    for _, out in (served, bare):
        assert out["forbid"] == [], out["forbid"][:5]
    src = NHL_JS.read_text(encoding="utf-8")
    assert not re.search(r"glassbox", src, re.I)
    for f in ("gb_rank", "gb_f", "gb_d"):
        assert f not in src, f


# ---------------------------------------------------------------- P3 ----
def test_p3_teams_index_is_ordered_and_badged_by_team_strength(served, bare):
    for n, out in (served, bare):
        st = _strength(n)
        want = sorted(st, key=lambda c: st[c][2])
        ti = out["r"]["tindex"]
        for key in ("null", "str", "gb", "pv"):          # default and stale sorts: strength
            assert ti[key]["codes"] == want, key
            assert ti[key]["lit"] == ["str"], key
        for code, (w, rk) in zip(ti["null"]["codes"], ti["null"]["chips"]):
            assert w == round(st[code][1], 1) and rk == st[code][2], code
        assert len(ti["null"]["chips"]) == len(want) == 32
        assert ti["null"]["sorts"] == [["str", "Team strength"], ["proj", "Projected points"],
                                       ["elo", "Results (Elo)"], ["xg", "Shot quality (xG)"]]
        by_elo = sorted(n["teams"], key=lambda c: (-n["teams"][c]["elo"], st[c][2]))
        assert ti["elo"]["codes"] == by_elo


# ---------------------------------------------------------------- P4 ----
def test_p4_team_page_leads_with_strength_and_keeps_lineup_value_in_the_roster_panel(served):
    n, out = served
    st = _strength(n)
    for code, r in out["r"]["teams"].items():
        z, w, rk = st[code]
        assert r["strWin"] == [f"{w:.1f}", str(rk), "32"], (code, r["strWin"])
        assert r["tElo"] and r["tXg"], code
        t = n["teams"][code]
        lu = t.get("pv_lu") or {}
        if lu.get("n"):
            assert r["luAll"] >= 1 and r["luAll"] == r["luPanel"], code   # only in the panel
            assert "display only" in r["rpHead"], code
            assert not r["rpRank"], code                                   # no lineup rank
            assert r["chg"] == (lu.get("chg") is not None), code
        g = [n["players"][i]["pv"]["g"] for i in r["key"] if (n["players"][i].get("pv") or {}).get("g") is not None]
        assert g == sorted(g, reverse=True) and len(g) >= 3, code
        assert r["key"] == t["top"], code
    assert all(r["chg"] for r in out["r"]["teams"].values())     # the snapshot carries lu_prev


# ---------------------------------------------------------------- P5 ----
def _num(s: str) -> float:
    return float(H.unescape(s).replace("−", "-"))


def test_p5_game_page_model_block_and_the_edge_line(served):
    n, out = served
    st = _strength(n)
    rows = {g["id"]: g for g in n["schedule"]}
    checked = 0
    for gid, r in out["r"]["games"].items():
        g = rows[int(gid)]
        assert r["model"], gid
        assert r["rows"][:3] == ["Team strength", "Results (Elo)", "Shot quality (xG)"], (gid, r["rows"])
        assert not r["lineupInModel"], gid
        if g.get("hs") is not None:
            assert r["edge"] is None, gid                 # a frozen pre-game number: no rebuild
            continue
        assert r["edge"], gid
        x, h, s, y = (_num(v) for v in r["edge"])
        assert abs(50 + x + h + s - y) <= 0.051, (gid, r["edge"])          # adds up as printed
        assert abs(y - 100 * g["hp"]) <= 0.051, (gid, y, g["hp"])          # the displayed model %
        assert abs(y - (50 + sum(g["ct"].values()))) <= 0.1 + 1e-9, gid     # the served decomposition
        dz = st[g["home"]][0] - st[g["away"]][0]
        assert abs(x - 100 * (1 / (1 + math.exp(-dz)) - 0.5)) <= 0.051, gid
        checked += 1
    assert checked > 1000


# ---------------------------------------------------------------- P6 ----
def test_p6_skater_and_goalie_pages(served):
    n, out = served
    nval = 0
    for pid, p in n["players"].items():
        r = out["r"]["players"][pid]
        v = p.get("pv")
        if p["grp"] == "G":
            assert r["gRating"], pid
            for k in ("cur", "prev"):
                ln = (p.get("stats") or {}).get(k)
                if ln and ln.get("gsax") is not None and ln.get("gp"):
                    assert r["gsaxLine"], pid
            continue
        assert r["rapmTitle"], pid
        if not v:
            assert r["headPct"] is None and r["prov"] is None, pid
            continue
        nval += 1
        assert int(r["headPct"]) == v["p"], pid
        od = {lab: (none, q) for lab, none, q in r["od"]}
        assert od["Offence"][1] == v["qo"], pid
        dq = (v.get("q") or {}).get("def")
        if dq is None:
            assert od["Defence, on-ice 5v5"][0] == "no on-ice estimate", pid
        else:
            assert od["Defence, on-ice 5v5"][1] == dq, pid
        assert (r["prov"] is not None) == bool(v["prov"]), pid
        if v["prov"]:
            assert int(r["prov"].replace(",", "")) == v["gp"], pid
        assert r["rel"], pid
    assert nval > 600


# ---------------------------------------------------------------- P7 ----
def test_p7_players_index(served, bare):
    n, out = served
    L = out["r"]["ladder"]
    vals = {pid: p["pv"]["v"] for pid, p in n["players"].items() if p["grp"] != "G" and p.get("pv")}
    assert sorted(L["order"]) == sorted(vals)                              # every valued skater
    assert [vals[i] for i in L["order"]] == sorted(vals.values(), reverse=True)   # by value
    assert L["off"] and L["def"] and L["lit"] == ["v"] and L["chipOn"] == ["v"]
    for _, o in (served, bare):
        assert o["r"]["goalieHead"]


# ---------------------------------------------------------------- P8 ----
def test_p8_standings_show_strength_not_lineup(served, bare):
    for _, out in (served, bare):
        std = out["r"]["std"]
        for k, r in std.items():
            if r["shown"] == "proj":
                assert ">Strength<" in r["head"] and ">Lineup<" not in r["head"], k
        assert any(r["shown"] == "proj" for r in std.values())
