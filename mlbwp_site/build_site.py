"""Generate the GLASSBOX predictions board — a data-first, multi-league page.

Reads site/data/board.json and emits a single self-contained index.html: a
league rail, date-range filters, and dense prediction rows. Rendering only —
no market data, no model logic here.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SITE = PROJECT / "site"


def load():
    board = json.loads((SITE / "data" / "board.json").read_text(encoding="utf-8"))
    ratings = json.loads((PROJECT / "mlbwp" / "artifacts" / "ratings.json").read_text(encoding="utf-8"))
    return board, ratings


CSS = """
:root{
  --paper:#f5f4f1; --raised:#fff; --sunk:#efede8; --ink:#191b20; --muted:#71757d;
  --faint:#a2a5ab; --line:#e2e0d9; --line-2:#d2cfc6;
  --accent:#c8102e; --accent-2:#c8102e14;
  --fav:#1f6f43; --favbg:#1f6f4316; --away:#3f6c86; --home:#b23b2e;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --serif:Georgia,'Times New Roman',serif;
  --mono:ui-monospace,'SF Mono','Cascadia Code',Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#101216; --raised:#181b21; --sunk:#0c0e12; --ink:#e9e7e1; --muted:#8b9098;
  --faint:#5c616b; --line:#242830; --line-2:#333844;
  --accent:#ff4d5e; --accent-2:#ff4d5e1c; --fav:#54c185; --favbg:#54c1851c;
  --away:#63a6c9; --home:#e28066;
}}
:root[data-theme="light"]{--paper:#f5f4f1;--raised:#fff;--sunk:#efede8;--ink:#191b20;
  --muted:#71757d;--faint:#a2a5ab;--line:#e2e0d9;--line-2:#d2cfc6;--accent:#c8102e;
  --accent-2:#c8102e14;--fav:#1f6f43;--favbg:#1f6f4316;--away:#3f6c86;--home:#b23b2e;}
:root[data-theme="dark"]{--paper:#101216;--raised:#181b21;--sunk:#0c0e12;--ink:#e9e7e1;
  --muted:#8b9098;--faint:#5c616b;--line:#242830;--line-2:#333844;--accent:#ff4d5e;
  --accent-2:#ff4d5e1c;--fav:#54c185;--favbg:#54c1851c;--away:#63a6c9;--home:#e28066;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.4;-webkit-font-smoothing:antialiased}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:1000px;margin:0 auto;padding:0 16px}

/* top bar */
header{position:sticky;top:0;z-index:30;background:color-mix(in srgb,var(--paper) 90%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
header .wrap{display:flex;align-items:center;gap:14px;height:54px}
.brand{font-family:var(--serif);font-weight:700;font-size:20px;letter-spacing:.01em}
.brand .b{color:var(--accent)}
.tag{font-size:11px;color:var(--muted);border-left:1px solid var(--line-2);padding-left:12px}
.spacer{flex:1}
.stat{font-family:var(--mono);font-size:11.5px;color:var(--muted);text-align:right;line-height:1.3}
.stat b{color:var(--ink)}
.tog{border:1px solid var(--line-2);background:var(--raised);color:var(--muted);border-radius:7px;
  width:32px;height:30px;cursor:pointer;font-size:14px;flex:none}
.tog:hover{color:var(--ink);border-color:var(--accent)}

/* league rail */
.rail{display:flex;gap:6px;overflow-x:auto;padding:12px 0 4px;scrollbar-width:none}
.rail::-webkit-scrollbar{display:none}
.lg{flex:none;border:1px solid var(--line-2);background:var(--raised);border-radius:9px;
  padding:7px 13px;cursor:pointer;font-weight:600;font-size:13px;color:var(--muted);
  display:flex;align-items:center;gap:7px}
.lg .n{font-family:var(--mono);font-size:11px;color:var(--faint);font-weight:500}
.lg[aria-selected="true"]{border-color:var(--accent);color:var(--ink);background:var(--accent-2)}
.lg[disabled]{opacity:.5;cursor:not-allowed}
.lg .soon{font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}

/* date filters */
.filters{display:flex;gap:4px;padding:12px 0;border-bottom:1px solid var(--line);
  position:sticky;top:54px;background:var(--paper);z-index:20}
.filt{border:none;background:none;color:var(--muted);font-weight:600;font-size:13px;
  padding:6px 12px;border-radius:8px;cursor:pointer}
.filt[aria-selected="true"]{background:var(--ink);color:var(--paper)}
.filt:hover:not([aria-selected="true"]){background:var(--sunk);color:var(--ink)}
.note{font-size:11.5px;color:var(--muted);margin-left:auto;align-self:center}

/* board */
main{padding:8px 0 40px}
.daygroup{margin-top:18px}
.dayhead{display:flex;align-items:baseline;gap:10px;padding:6px 2px;position:sticky;top:107px;
  background:var(--paper);z-index:10}
.dayhead .d{font-family:var(--serif);font-size:15px;font-weight:700}
.dayhead .c{font-family:var(--mono);font-size:11px;color:var(--faint)}
.dayhead .proj{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);
  margin-left:auto;border:1px solid var(--line-2);border-radius:5px;padding:1px 6px}

.row{display:grid;grid-template-columns:52px 1fr 150px 92px 26px;align-items:center;gap:10px;
  padding:9px 8px;border-bottom:1px solid var(--line);cursor:pointer}
.row:hover{background:var(--raised)}
.time{font-family:var(--mono);font-size:11.5px;color:var(--muted);text-align:center}
.time .st{display:block;font-size:9px;color:var(--accent);font-weight:700;letter-spacing:.04em}

.match{min-width:0}
.side{display:flex;align-items:center;gap:8px;padding:1px 0}
.side .ab{font-family:var(--mono);font-weight:700;font-size:14px;width:34px;letter-spacing:.02em}
.side .pit{font-size:11.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.side.win .ab{color:var(--fav)}
.side.win .pit{color:var(--ink)}
.side .tbd{color:var(--faint);font-style:italic}

.prob{display:flex;flex-direction:column;gap:3px}
.pbar{height:7px;border-radius:5px;background:var(--away);position:relative;overflow:hidden}
.pbar .h{position:absolute;right:0;top:0;bottom:0;background:var(--home)}
.pbar .mid{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--ink);opacity:.3}
.pends{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--muted)}

.pick{text-align:right}
.pick .w{font-family:var(--mono);font-weight:700;font-size:14px}
.pick .pc{font-family:var(--mono);font-size:11px;color:var(--muted);display:block}
.pick.strong .w{color:var(--fav)}
.pick.lean .w{color:var(--ink)}
.pick.toss .w{color:var(--muted)}
.exp{color:var(--faint);text-align:center;font-size:15px;transition:transform .15s}
.row[aria-expanded="true"] .exp{transform:rotate(90deg);color:var(--accent)}

.detail{grid-column:1/-1;padding:4px 8px 12px;display:none;gap:8px;flex-wrap:wrap;align-items:center}
.row[aria-expanded="true"]+.detail{display:flex}
.chip{font-family:var(--mono);font-size:11px;padding:2px 9px;border-radius:6px;border:1px solid var(--line-2);
  background:var(--sunk);font-variant-numeric:tabular-nums}
.chip b{font-family:var(--sans);color:var(--muted);font-weight:600}
.chip .p{color:var(--fav)}.chip .n{color:var(--home)}
.detail .lead{font-size:11.5px;color:var(--muted)}

.empty{padding:50px 0;text-align:center;color:var(--muted)}

footer{border-top:1px solid var(--line);padding:18px 0 40px;color:var(--faint);font-size:11px;line-height:1.6}
footer b{color:var(--muted)}
footer a{color:var(--muted)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}

@media (max-width:640px){
  .row{grid-template-columns:44px 1fr 78px 22px;gap:8px}
  .prob{display:none}
  .tag{display:none}
}
"""

JS = r"""
const B = JSON.parse(document.getElementById("board-data").textContent);
const root = document.documentElement;
document.getElementById("tog").onclick = () => {
  const cur = root.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme:dark)").matches ? "dark":"light");
  root.setAttribute("data-theme", cur==="dark"?"light":"dark");
};

const GEN = B.generated;
const addDays = (iso,n)=>{const d=new Date(iso+"T00:00:00");d.setDate(d.getDate()+n);
  return d.toISOString().slice(0,10);};
const RANGES = {
  today:  [GEN, GEN],
  tomorrow:[addDays(GEN,1), addDays(GEN,1)],
  week:   [GEN, addDays(GEN,6)],
  month:  [GEN, addDays(GEN,60)],
};
let active = {league:"mlb", range:"today"};

const LOC = "en-US";  // keep dates in English to match the UI, whatever the OS locale
const fmtDay = iso => {
  const d = new Date(iso+"T12:00:00");
  const wd = d.toLocaleDateString(LOC,{weekday:"long"});
  const md = d.toLocaleDateString(LOC,{month:"short",day:"numeric"});
  if (iso===GEN) return "Today · "+md;
  if (iso===addDays(GEN,1)) return "Tomorrow · "+md;
  return wd+" · "+md;
};
const fmtTime = utc => new Date(utc).toLocaleTimeString(LOC,{hour:"numeric",minute:"2-digit"});
const pct = x => Math.round(x*100);
const tier = p => p>=0.62?"strong":(p>=0.555?"lean":"toss");
const sgn = v => (v>=0?"+":"")+v.toFixed(1);

function leagueRail(){
  const el = document.getElementById("rail");
  el.innerHTML = B.leagues.map(l=>{
    if(!l.active) return `<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`;
    const sel = l.code===active.league;
    return `<button class="lg" data-lg="${l.code}" aria-selected="${sel}">
      <span>${l.name}</span><span class="n">${l.n_games}</span></button>`;
  }).join("");
  el.querySelectorAll("[data-lg]").forEach(b=>b.onclick=()=>{active.league=b.dataset.lg;render();});
}

function row(g){
  const hp=g.home_win_prob, homeWin=hp>=0.5;
  const st = g.state==="Live" ? `<span class="st">LIVE</span>` : "";
  const away = `<div class="side ${homeWin?"":"win"}">
    <span class="ab">${g.away_abbr}</span>
    <span class="pit ${g.away_sp==="TBD"?"tbd":""}">${g.away_sp}</span></div>`;
  const home = `<div class="side ${homeWin?"win":""}">
    <span class="ab">${g.home_abbr}</span>
    <span class="pit ${g.home_sp==="TBD"?"tbd":""}">${g.home_sp}</span></div>`;
  const bar = `<div class="prob"><div class="pbar"><div class="h" style="width:${pct(hp)}%"></div>
      <div class="mid"></div></div>
    <div class="pends"><span>${g.away_abbr} ${pct(1-hp)}</span><span>${pct(hp)} ${g.home_abbr}</span></div></div>`;
  const e = g.edge||{};
  const detail = `<div class="detail">
    <span class="lead">Why ${g.pick}:</span>
    ${chip("Team",e.team)}${chip("Home",e.home_field)}
    ${chip(g.home_abbr+" SP",e.home_pitcher)}${chip(g.away_abbr+" SP",e.away_pitcher)}
    ${g.pitcher_known?"":'<span class="lead">Starters not set — team-only forecast.</span>'}
  </div>`;
  return `<div class="row" tabindex="0" aria-expanded="false">
    <div class="time">${fmtTime(g.start_utc)}${st}</div>
    <div class="match">${away}${home}</div>
    ${bar}
    <div class="pick ${tier(g.pick_prob)}"><span class="w">${g.pick}</span>
      <span class="pc">${pct(g.pick_prob)}%</span></div>
    <div class="exp">›</div>
  </div>${detail}`;
}
function chip(label,v){v=v||0;return `<span class="chip"><b>${label}</b> <span class="${v>=0?"p":"n"}">${sgn(v)}</span></span>`;}

function render(){
  document.querySelectorAll(".filt").forEach(f=>f.setAttribute("aria-selected", f.dataset.r===active.range));
  leagueRail();
  const lg = B.leagues.find(l=>l.code===active.league);
  const [lo,hi] = RANGES[active.range];
  const games = (lg.games||[]).filter(g=>g.date>=lo && g.date<=hi);
  const board = document.getElementById("board");
  if(!games.length){board.innerHTML=`<div class="empty">No games in this window.</div>`;return;}
  const days=[...new Set(games.map(g=>g.date))].sort();
  board.innerHTML = days.map(d=>{
    const gs=games.filter(g=>g.date===d);
    const anyTbd=gs.some(g=>!g.pitcher_known);
    return `<section class="daygroup"><div class="dayhead">
      <span class="d">${fmtDay(d)}</span><span class="c">${gs.length} games</span>
      ${anyTbd?'<span class="proj">projected · starters tbd</span>':''}
    </div>${gs.map(row).join("")}</section>`;
  }).join("");
  board.querySelectorAll(".row").forEach(r=>{
    const t=()=>r.setAttribute("aria-expanded", r.getAttribute("aria-expanded")!=="true");
    r.onclick=t; r.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();t();}};
  });
}

document.querySelectorAll(".filt").forEach(f=>f.onclick=()=>{active.range=f.dataset.r;render();});
render();
"""


def build():
    board, ratings = load()
    acc = board["accuracy"]
    data_json = json.dumps(board)

    body = f"""<style>{CSS}</style>
<script id="board-data" type="application/json">{data_json}</script>

<header><div class="wrap">
  <span class="brand">GLASS<span class="b">BOX</span></span>
  <span class="tag">market-blind predictions</span>
  <span class="spacer"></span>
  <span class="stat"><b>{acc['log_loss']:.3f}</b> log loss<br>coin flip {acc['coinflip']:.3f}</span>
  <button class="tog" id="tog" aria-label="Toggle theme">&#9681;</button>
</div></header>

<div class="wrap">
  <nav class="rail" id="rail"></nav>
  <div class="filters">
    <button class="filt" data-r="today" aria-selected="true">Today</button>
    <button class="filt" data-r="tomorrow">Tomorrow</button>
    <button class="filt" data-r="week">This week</button>
    <button class="filt" data-r="month">Month</button>
    <span class="note">tap a game for the why</span>
  </div>
  <main id="board"></main>
  <footer>
    <b>Research only &mdash; not betting advice.</b> A market-blind statistical model
    (team Elo + FIP starting-pitcher rating); it never sees the odds. Team form current
    through {board['current_through']}; pitcher ratings through {ratings['trained_through_season']}.
    Data: <b>Retrosheet</b> (obtained free of charge and copyrighted by Retrosheet,
    <a href="https://www.retrosheet.org">www.retrosheet.org</a>) and the MLB Stats API
    (individual, non-commercial use).
  </footer>
</div>

<script>{JS}</script>"""

    body_ascii = body.encode("ascii", "xmlcharrefreplace").decode("ascii")
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(body_ascii, encoding="utf-8")
    n = board["leagues"][0]["n_games"]
    print(f"wrote {SITE/'index.html'} ({len(body_ascii):,} bytes, {n} MLB games)")


if __name__ == "__main__":
    build()
