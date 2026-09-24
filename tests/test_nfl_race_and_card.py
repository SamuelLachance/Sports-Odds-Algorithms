"""NFL round 2: the race view, the snap-map fix, the model card's provenance.

  - conference seeding follows the NFL wild-card tiebreakers: on every
    seven-club season on file (2020-2025) the computed seeds reproduce the real
    playoff bracket (the same 14 clubs, 1 seed with the bye, 2v7 / 3v6 / 4v5);
  - clinch marks are sufficient conditions, so a mark is never wrong: replayed
    week by week over 2012-2025, no 'z' club missed its division, no 'x' club
    missed the playoffs, no 'e' club made them;
  - snap rows join to players by pfr id, then by (name, team) from the weekly
    roster: a rookie whose pfr id nfl_players.csv lacks keeps his usage;
  - the shipped payload: lineups name each player once per side, half sacks
    survive, the model card's measured-once numbers carry their provenance;
  - the pages: the playoff picture, byes, the week bars, tackles split into
    combined / solo / assisted, projected wins with their spread.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nfl_payload as NP  # noqa: E402

GAMES = ROOT / "data" / "nfl_games.csv"
NFL_JSON = ROOT / "site" / "data" / "nfl.json"


def _rows():
    if not GAMES.is_file():
        pytest.skip("data/nfl_games.csv absent")
    return NP.read_games(str(GAMES))


def _payload():
    if not NFL_JSON.is_file():
        pytest.skip("site/data/nfl.json not built")
    n = json.loads(NFL_JSON.read_text(encoding="utf-8"))
    if not n.get("schedule"):
        pytest.skip("site/data/nfl.json has no schedule")
    return n


# ------------------------------------------------------------ seeding
def test_seeding_reproduces_every_seven_club_bracket():
    rows = _rows()
    for season in range(2020, 2026):
        cur = NP.final_rows(rows, season)
        if not cur:
            continue
        seeds = NP.conference_seeding(cur)
        post = [r for r in rows if r["season"] == str(season) and r["game_type"] in ("WC", "DIV")]
        wc = [r for r in rows if r["season"] == str(season) and r["game_type"] == "WC"]
        for conf, order in seeds.items():
            s = {t: i for i, t in enumerate(order, 1)}
            real = {NP.FR.get(r[k], r[k]) for r in post for k in ("home_team", "away_team")
                    if NP.CONF[NP.FR.get(r[k], r[k])] == conf}
            assert set(order[:7]) == real, (season, conf)
            for r in wc:
                h, a = NP.FR.get(r["home_team"], r["home_team"]), NP.FR.get(r["away_team"], r["away_team"])
                if NP.CONF[h] == conf:
                    assert s[h] + s[a] == 9 and s[h] < s[a], (season, conf, h, a)


def test_wild_card_sweep_rule():
    """Three clubs from three divisions, equal records: one beat both others."""
    g = lambda w, h, a, hs, as_: {"d": f"2026-09-{10 + w:02d}", "w": w, "home": h, "away": a,  # noqa: E731
                                  "hs": hs, "as": as_, "type": "REG"}
    games = [g(1, "BUF", "BAL", 20, 10), g(2, "BUF", "HOU", 20, 10),
             g(3, "BAL", "DEN", 20, 10), g(4, "HOU", "DEN", 20, 10), g(5, "DEN", "BUF", 20, 10)]
    book = NP._Book(games)
    assert NP._best_wc(book, ["BAL", "BUF", "HOU"]) == "BUF"
    # two clubs that met: head-to-head decides before the alphabet
    assert NP._best_wc(book, ["BAL", "BUF"]) == "BUF"
    assert NP._best_wc(book, ["DEN", "BUF"]) == "DEN"


def test_seeding_is_not_published_before_a_game_is_played(tmp_path):
    payload = {"teams": {t: {"code": t, "div": NP.TEAM_DIV[t]} for t in NP.TEAMS}}
    rows = [{"season": "2026", "game_type": "REG", "week": "1", "gameday": "2026-09-13",
             "home_team": "BUF", "away_team": "MIA", "home_score": "", "away_score": ""}]
    rows += [{"season": "2025", "game_type": "REG", "week": "1", "gameday": "2025-09-07",
              "home_team": "BUF", "away_team": "MIA", "home_score": "20", "away_score": "10"}]
    NP.apply_standings(payload, rows, 2026)
    assert all("conf_seed" not in t and "clinch" not in t for t in payload["teams"].values())


# ------------------------------------------------------------ clinch marks
def test_clinch_marks_are_never_wrong():
    rows = _rows()
    n_marks = 0
    for season in range(2012, 2026):
        allf = NP.final_rows(rows, season)
        if not allf:
            continue
        real = {NP.FR.get(r[k], r[k]) for r in rows if r["season"] == str(season)
                and r["game_type"] in ("WC", "DIV") for k in ("home_team", "away_team")}
        book = NP._Book(allf)
        winners = {NP.rank_division(book, ts)[0] for ts in NP.DIVS.values()}
        for wk in range(1, max(g["w"] for g in allf) + 1):
            cur = [g for g in allf if g["w"] <= wk]
            rem = defaultdict(int)
            for g in allf:
                if g["w"] > wk:
                    rem[g["home"]] += 1
                    rem[g["away"]] += 1
            for t, f in NP.clinch_flags(cur, rem).items():
                n_marks += 1
                assert not (f == "z" and t not in winners), (season, wk, t, f)
                if season >= 2020:        # the x rule is for the seven-club format
                    assert not (f == "x" and t not in real), (season, wk, t, f)
                assert not (f == "e" and t in real), (season, wk, t, f)
    assert n_marks > 500                  # the rule does fire, late in seasons


def test_no_marks_without_the_whole_schedule():
    games = [{"d": "2026-09-13", "w": 1, "home": "BUF", "away": "MIA", "hs": 30, "as": 0, "type": "REG"}]
    assert NP.clinch_flags(games, {}) == {}                   # remaining unknown: nothing marked


def test_a_club_that_cannot_be_caught_clinches_its_division():
    """BUF 12-0 (worst case 12-5, .706); every rival 0-6 (best case 11-6, .647)."""
    east = set(NP.DIVS["AFC East"])
    others = [t for t in NP.TEAMS if t not in east]
    g = lambda w, h, a: {"d": f"2026-{9 + w // 5:02d}-{1 + 5 * (w % 5):02d}", "w": w,  # noqa: E731
                         "home": h, "away": a, "hs": 20, "as": 10, "type": "REG"}
    games = [g(w, "BUF", others[w - 1]) for w in range(1, 13)]
    for i, rival in enumerate(("MIA", "NE", "NYJ")):
        games += [g(w, others[12 + (i * 6 + w) % 16], rival) for w in range(1, 7)]
    rem = {t: 17 - sum(1 for x in games if t in (x["home"], x["away"])) for t in NP.TEAMS}
    assert min(rem.values()) >= 0
    fl = NP.clinch_flags(games, rem)
    assert fl.get("BUF") == "z"
    assert all(fl.get(t) not in ("z", "x") for t in ("MIA", "NE", "NYJ"))
    # one more rival win keeps the race open: 12-5 would tie BUF's worst case
    games[-1] = dict(games[-1], hs=10, **{"as": 20})
    fl = NP.clinch_flags(games, rem)
    assert fl.get("BUF") != "z"


# ------------------------------------------------------------ name join
def test_name_key():
    assert NP.name_key("Michael Penix Jr.") == NP.name_key("michael penix") == "michael penix"
    assert NP.name_key("T.J. Watt") == "tj watt"
    assert NP.name_key("D'Andre Swift") == "dandre swift"
    assert NP.name_key("Amon-Ra St. Brown") == "amon ra st brown"


def test_snap_rows_join_by_roster_pfr_then_name_and_team():
    """Every current-season snap player maps to a gsis id (a handful who left
    every roster may not), and the site builder reads the roster before the
    snap tables."""
    src = (ROOT / "phase0" / "nfl_site_db.py").read_text(encoding="utf-8")
    assert src.index("roster_rows = ") < src.index("snap_cur, weeks_seen = snap_usage(")
    assert "by_name_team.get((name_key(" in src
    n = _payload()
    snap = ROOT / "data" / f"snap_{n.get('stats_season')}.csv"
    if not snap.is_file():
        pytest.skip("no current-season snap table")
    played = defaultdict(set)
    names = {}
    for r in csv.DictReader(open(snap, encoding="utf-8")):
        if r.get("game_type", "REG") == "REG" and (float(r["offense_pct"] or 0) + float(r["defense_pct"] or 0)) > 0.5:
            played[(NP.name_key(r["player"]), NP.FR.get(r["team"], r["team"]))].add(r["week"])
            names[(NP.name_key(r["player"]), NP.FR.get(r["team"], r["team"]))] = r["player"]
    by = {(NP.name_key(p["name"]), p["team"]): p for p in n["players"].values()}
    lost = [names[k] for k, wks in played.items() if k in by and by[k]["snap_g"] == 0]
    assert not lost, f"starters on the payload with 0 snap games: {lost[:10]}"


# ------------------------------------------------------------ shipped payload
def test_lineups_name_each_player_once_per_side():
    n = _payload()
    for t, tm in n["teams"].items():
        for side in ("off", "def"):
            ids = [e["id"] for e in (tm.get("lineup") or {}).get(side, [])]
            assert len(ids) == len(set(ids)), (t, side)


def test_half_sacks_survive():
    n = _payload()
    frac = [p for p in n["players"].values() if (p.get("stats") or {}).get("sk", 0) % 1]
    assert frac, "no fractional sack total shipped (3.5 truncated to 3?)"
    assert all(abs(p["stats"]["sk"] * 2 - round(p["stats"]["sk"] * 2)) < 1e-9 for p in frac)


def test_seeds_and_marks_ship_with_the_standings():
    n = _payload()
    thr = n.get("standings_through")
    if not thr or n.get("standings_season") != n.get("season"):
        pytest.skip("no current-season final yet")
    for conf in ("AFC", "NFC"):
        seeds = sorted(t["conf_seed"] for t in n["teams"].values() if t["div"].startswith(conf))
        assert seeds == list(range(1, 17)), conf
    assert all(t.get("clinch") in (None, "z", "x", "e") for t in n["teams"].values())


def test_model_card_numbers_carry_their_provenance():
    n = _payload()
    mc = n["model_card"]
    ms = mc.get("measured") or {}
    assert ms.get("asof") and ms.get("model_test_ll") and ms.get("ledger_row")
    for k in ("calibration", "acc_home", "acc_elo", "acc_close", "close_log_loss"):
        assert k in ms.get("keys", []) and k in mc, k
    # the headline is the serve's measurement, never the July literal
    assert mc["test_log_loss"] != ms["model_test_ll"]
    assert mc["serve"]["test_repro_ll"] == mc["test_log_loss"]
    ro = mc.get("ratings_only") or {}
    assert ro.get("test_ll") and ro.get("elo_ll") and ro["test_ll"] < ro["elo_ll"]
    src = (ROOT / "phase0" / "nfl_site_data.py").read_text(encoding="utf-8")
    assert '"test_log_loss": 0.61947' not in src and '"n_tests": 47' not in src


# ------------------------------------------------------------ pages
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
  const fs=require("fs"), D=__DATA__, out={};
  const rd=f=>{try{return JSON.parse(fs.readFileSync(D+"/"+f,"utf8"));}catch(e){return null;}};
  const R=(k,f)=>{try{f(); out[k]=view();}catch(e){out[k]="THREW "+(e&&e.stack||e);}};
  state.board=rd("board.json"); state.db=rd("db.json"); state.nhl=rd("nhl.json");
  if(state.board) siteLeaguesTidy();
  const n=rd("nfl.json"); if(!n||!n.schedule||!n.schedule.length){console.log("{}");return;}
  state.nfl=n; state.league="nfl";
  state.nflStand="div"; R("st_div",()=>nflStandings());
  state.nflStand="po"; R("st_po",()=>nflStandings());
  state.nflRange="month"; state.nflWk=null; R("board_4w",()=>nflPage());
  state.nflRange="week"; R("board_wk",()=>nflPage());
  const byeW=[...new Set(n.schedule.map(g=>g.w))].find(w=>nflByes(w).length);
  out.byeW=byeW; out.byes=byeW!=null?nflByes(byeW):[];
  if(byeW!=null) R("season_bye",()=>nflSeason(String(byeW)));
  const ids=Object.keys(n.players);
  const def=ids.find(id=>{const p=n.players[id]; return p.fam==="LB"&&p.stats&&p.stats.g&&(p.stats.tak||0)>0&&(p.stats.ast||0)>0;});
  out.def=def; if(def) R("player_def",()=>nflPlayerPage(def));
  if(def) R("team_def",()=>nflTeamPage(n.players[def].team));
  R("players",()=>nflPlayers());
  // a lineup with a rookie who has no rating
  const rk=Object.entries(n.teams).find(([c,t])=>(t.lineup&&t.lineup.off.concat(t.lineup.def)||[]).some(e=>e.r==null&&n.players[e.id]&&n.players[e.id].exp===0));
  out.rookieTeam=rk?rk[0]:null; if(rk) R("team_rookie",()=>nflTeamPage(rk[0]));
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    _payload()
    from mlbwp_site.build_site import JS
    p = tmp_path_factory.mktemp("js") / "nfl_race.js"
    src = HARNESS.replace("__DATA__", json.dumps(str(ROOT / "site" / "data").replace("\\", "/")))
    p.write_text(src.replace("__JS__", JS.replace("\nboot();", "\n")), encoding="utf-8")
    r = subprocess.run([node, str(p)], capture_output=True, text=True, timeout=600, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _txt(h: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))


def test_pages_render(pages):
    for k, v in pages.items():
        if isinstance(v, str):
            assert not v.startswith("THREW"), (k, v[:600])
            assert not re.search(r"\b(undefined|NaN|null)\b", _txt(v)), (k, _txt(v)[:400])


def test_playoff_picture(pages):
    n = _payload()
    if not (n.get("standings_through") and n.get("standings_season") == n.get("season")):
        pytest.skip("no current-season final yet")
    po = pages["st_po"]
    assert "Playoff picture" in po and 'class="nfl-cut"' in po
    assert po.count('class="nfl-cut"') == 2                  # one cut line per conference
    assert "wild card" in po and "Conf" in po
    div = pages["st_div"]
    assert "<th>Conf</th>" in div and "Seed" in div and "L10" not in div


def test_byes_and_week_bars(pages):
    if pages.get("byeW") is None:
        pytest.skip("no bye week in the schedule")
    s = pages["season_bye"]
    assert "Bye:" in s and all(f"#/nfl/team/{t}" in s for t in pages["byes"])
    assert 'class="rail"' in s                                 # the Season page has the league rail
    b = pages["board_4w"]
    assert b.count('class="nfl-wkbar"') >= 2 and "Weeks " in b


def test_tackles_are_labelled(pages):
    if not pages.get("def"):
        pytest.skip("no defender with tackles")
    n = _payload()
    s = n["players"][pages["def"]]["stats"]
    pp = _txt(pages["player_def"])
    assert f"Tkl {s['tak'] + s['ast']}" in pp and f"Solo {s['tak']}" in pp and f"Ast {s['ast']}" in pp
    assert "<th>Tkl</th><th>Solo</th><th>Ast</th>" in pages["team_def"]


def test_projection_spread_and_position_values(pages):
    n = _payload()
    if n.get("proj"):
        assert "nfl-sd" in pages["board_wk"] and "nfl-sd" in pages["st_div"]
    if n.get("pos_values"):
        assert "What a starter is worth" in pages["players"]


def test_lineup_labels_rookies(pages):
    if not pages.get("rookieTeam"):
        pytest.skip("no unrated rookie in a lineup")
    assert ">rookie<" in pages["team_rookie"]
