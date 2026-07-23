"""Generate the GLASSBOX SPA shell (site/index.html).

The page fetches ./data/board.json and ./data/db.json at runtime and renders four
views through a hash router: the predictions board, a per-game deep dive, the
standings, and the model's rankings. Rendering only; no model logic, no market
data. The data files are produced by mlbwp.predict_slate and mlbwp.db.
"""

from __future__ import annotations

from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"

CSS = r"""
:root{
  --bg:#0d0f13; --panel:#151820; --panel-2:#1b1f28; --sunk:#0a0c10; --ink:#e9e7e1;
  --muted:#8b929c; --faint:#5a616c; --line:#232833; --line-2:#333a47;
  --accent:#ff4d5e; --accent-2:#ff4d5e18; --fav:#4ec98a; --favbg:#4ec98a16;
  --away:#5aa0c9; --home:#e07a5f; --warn:#e0a44e;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --serif:Georgia,'Times New Roman',serif;
  --mono:ui-monospace,'SF Mono','Cascadia Code',Menlo,Consolas,monospace;
  --w:1360px;
}
/* Dark is the default; light applies only when the viewer explicitly toggles it
   (data-theme="light"), which the toggle persists in localStorage. */
:root[data-theme="light"]{
  --bg:#f4f3ef; --panel:#fff; --panel-2:#faf9f6; --sunk:#eeece6; --ink:#1a1c22;
  --muted:#6b7079; --faint:#a3a7ae; --line:#e4e1d9; --line-2:#d3cfc5;
  --accent:#c8102e; --accent-2:#c8102e12; --fav:#1c7a48; --favbg:#1c7a4812;
  --away:#356b8a; --home:#b34a35; --warn:#a9741e;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;
  line-height:1.4;-webkit-font-smoothing:antialiased}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:var(--w);margin:0 auto;padding:0 18px}
a{color:inherit;text-decoration:none}

/* nav */
header{position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--bg) 92%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
header .wrap{display:flex;align-items:center;gap:20px;height:56px}
.brand{font-family:var(--serif);font-weight:700;font-size:21px;flex:none}
.brand .b{color:var(--accent)}
nav.main{display:flex;gap:2px;min-width:0;overflow-x:auto;scrollbar-width:none}
nav.main::-webkit-scrollbar{display:none}
nav.main a{flex:none}
nav.main a{font-weight:600;font-size:13.5px;color:var(--muted);padding:7px 13px;border-radius:8px}
nav.main a:hover{color:var(--ink);background:var(--panel)}
nav.main a.on{color:var(--ink);background:var(--accent-2);box-shadow:inset 0 0 0 1px var(--line-2)}
.grow{flex:1}
.acc{font-family:var(--mono);font-size:11.5px;color:var(--muted);text-align:right;line-height:1.25}
.acc b{color:var(--ink)}
.tog{border:1px solid var(--line-2);background:var(--panel);color:var(--muted);border-radius:8px;
  width:34px;height:32px;cursor:pointer;font-size:14px;flex:none}
.tog:hover{color:var(--ink);border-color:var(--accent)}

main{padding:16px 0 60px;min-height:70vh}
.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700}
h1.pt{font-family:var(--serif);font-size:26px;margin:2px 0 14px}

/* controls row */
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:14px}
.rail{display:flex;gap:6px;flex-wrap:wrap}
.lg{border:1px solid var(--line-2);background:var(--panel);border-radius:9px;padding:7px 13px;
  cursor:pointer;font-weight:600;font-size:13px;color:var(--muted);display:flex;align-items:center;gap:7px}
.lg .n{font-family:var(--mono);font-size:11px;color:var(--faint)}
.lg.on{border-color:var(--accent);color:var(--ink);background:var(--accent-2)}
.lg[disabled]{opacity:.45;cursor:not-allowed}
.lg .soon{font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
.filters{display:flex;gap:3px;margin-left:auto;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:3px}
.filt{border:none;background:none;color:var(--muted);font-weight:600;font-size:13px;padding:6px 13px;
  border-radius:7px;cursor:pointer}
.filt.on{background:var(--ink);color:var(--bg)}
.filt:hover:not(.on){color:var(--ink)}

/* BOARD: big responsive grid */
.day{margin:18px 0 8px;display:flex;align-items:baseline;gap:10px}
.day .d{font-family:var(--serif);font-size:16px;font-weight:700}
.day .c{font-family:var(--mono);font-size:11px;color:var(--faint)}
.day .proj{margin-left:auto;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);
  border:1px solid var(--line-2);border-radius:5px;padding:1px 7px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:10px}
.grid>*{min-width:0}
.gc{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 13px;
  cursor:pointer;display:flex;flex-direction:column;gap:9px;transition:border-color .12s,transform .12s}
.gc:hover{border-color:var(--line-2);transform:translateY(-1px)}
.gc .top{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.gc .top .live{color:var(--accent);font-weight:700;letter-spacing:.04em}
.gc .top .pill{margin-left:auto;font-family:var(--sans);font-weight:700;font-size:12px;padding:3px 10px;
  border-radius:20px;background:var(--sunk);border:1px solid var(--line-2)}
.gc .top .pill.strong{color:var(--fav);border-color:var(--fav);background:var(--favbg)}
.gc .top .lbadge{font-family:var(--sans);font-weight:700;font-size:10px;letter-spacing:.03em;
  text-transform:uppercase;padding:2px 8px;border-radius:6px;border:1px solid transparent;
  white-space:nowrap;line-height:1.55}
.gc .top .lbadge.off{color:var(--fav);border-color:var(--fav);background:var(--favbg)}
.gc .top .lbadge.proj{color:var(--warn);border-color:rgba(224,164,78,.5);background:rgba(224,164,78,.14)}
.gc .top .lbadge.tbd{color:var(--faint);border-color:var(--line-2);background:var(--sunk)}
.gc .pj{font-family:var(--mono);font-size:8.5px;text-transform:uppercase;letter-spacing:.04em;
  color:var(--warn);border:1px solid rgba(224,164,78,.45);border-radius:4px;padding:0 3px;vertical-align:1px}
.gc.hasval{border-color:#e8b23a;box-shadow:0 0 0 1px rgba(232,178,58,.35),0 6px 18px rgba(232,178,58,.14)}
.gc .valbar{display:flex;align-items:center;gap:6px;margin-bottom:3px;padding:5px 9px;border-radius:8px;
  background:linear-gradient(90deg,rgba(232,178,58,.20),rgba(232,178,58,.06));border:1px solid rgba(232,178,58,.5);
  font-family:var(--sans);font-size:12px;font-weight:700;color:var(--ink)}
.gc .valbar .vt{font-family:var(--mono);font-size:9px;letter-spacing:.09em;background:#e8b23a;color:#181200;
  padding:1px 6px;border-radius:4px}
.gc .valbar b{color:#d99a1e}
.gc .valbar .vodds{color:var(--muted);font-weight:600;font-size:11px;font-family:var(--mono)}
.gc .valbar .vlive{margin-left:auto;color:var(--fav);font-size:10px;font-weight:700;letter-spacing:.03em}
.side{display:grid;grid-template-columns:38px 1fr auto;align-items:center;gap:9px;padding:2px 0}
.side .ab{font-family:var(--mono);font-weight:700;font-size:15px}
.side .who{min-width:0}
.side .who .sp{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.side .who .sp .era{color:var(--faint);font-family:var(--mono)}
.side .pc{font-family:var(--mono);font-size:14px;font-weight:600;text-align:right}
.side.win .ab,.side.win .pc{color:var(--fav)}
.side .tbd{color:var(--faint);font-style:italic}
.pbar{height:6px;border-radius:4px;background:var(--away);position:relative;overflow:hidden;margin-top:1px}
.pbar .h{position:absolute;right:0;top:0;bottom:0;background:var(--home)}
.pbar .mid{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--ink);opacity:.35}
.gc .edge{display:flex;gap:6px;flex-wrap:wrap;font-family:var(--mono);font-size:10.5px;color:var(--muted)}
.gc .edge .k{color:var(--faint)}
.gc .edge .p{color:var(--fav)}.gc .edge .n{color:var(--home)}
.lv.live{color:var(--accent);font-weight:700}
.lv.live .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);
  margin-right:5px;vertical-align:middle;animation:pulse 1.4s ease-in-out infinite}
.lv.final{color:var(--ink);font-weight:600}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* tables (standings, rankings) */
.twrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:700;
  border-bottom:1px solid var(--line-2);background:var(--panel)}
tbody tr{border-bottom:1px solid var(--line);cursor:pointer}
tbody tr:hover{background:var(--panel-2)}
tbody tr:last-child{border-bottom:none}
td .num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.tm{display:flex;align-items:center;gap:8px}
.tm .ab{font-family:var(--mono);font-weight:700;width:34px}
.pos{color:var(--fav)}.neg{color:var(--home)}.warnc{color:var(--warn)}
.subh{font-family:var(--serif);font-size:16px;font-weight:700;margin:22px 0 8px}

/* GAME PAGE */
.back{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-weight:600;font-size:13px;margin-bottom:10px}
.back:hover{color:var(--accent)}
.gh{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.gh .mt{font-family:var(--serif);font-size:24px;font-weight:700}
.gh .meta{font-family:var(--mono);font-size:12px;color:var(--muted)}
.gh .live{color:var(--accent);font-weight:700}
.cols{display:grid;grid-template-columns:1.15fr .85fr;gap:14px;align-items:start}
@media(max-width:860px){.cols{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px 16px}
.panel h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 12px;font-weight:700}
.panel h3 .sub{text-transform:none;letter-spacing:.01em;font-weight:400}
.nocase{text-transform:none!important;letter-spacing:0!important}
.proj{display:flex;align-items:center;gap:16px;margin-bottom:6px}
.proj .big{font-family:var(--mono);font-size:46px;font-weight:600;line-height:1}
.proj .big .u{font-size:19px;color:var(--muted)}
.proj .pk{font-size:13px;color:var(--muted)}
.proj .pk b{font-family:var(--mono);font-size:18px;color:var(--ink);display:block}
.proj .pk .cf{color:var(--fav);font-weight:700}
.bigbar{height:12px;border-radius:7px;background:var(--away);position:relative;overflow:hidden;margin:10px 0 4px}
.bigbar .h{position:absolute;right:0;top:0;bottom:0;background:var(--home)}
.bigbar .mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:var(--ink);opacity:.4}
.barlab{display:flex;justify-content:space-between;font-family:var(--mono);font-size:12px;color:var(--muted)}
.why{margin-top:14px;display:flex;flex-direction:column;gap:7px}
.why .r{display:grid;grid-template-columns:96px 1fr 52px;align-items:center;gap:9px;font-size:12.5px}
.why .r .lab{color:var(--muted)}
.why .r .tr{height:16px;background:var(--sunk);border-radius:4px;position:relative;overflow:hidden}
.why .r .tr .f{position:absolute;top:0;bottom:0;left:50%;background:var(--fav)}
.why .r .tr .f.neg{background:var(--home)}
.why .r .v{font-family:var(--mono);text-align:right;font-weight:600}
.signals{display:flex;flex-direction:column;gap:8px}
.sig{display:flex;gap:9px;align-items:flex-start;font-size:12.5px;padding:8px 10px;border-radius:8px;
  background:var(--sunk);border-left:3px solid var(--line-2)}
.sig.warn{border-left-color:var(--warn)}.sig.good{border-left-color:var(--fav)}
.sig .ic{font-family:var(--mono);font-weight:700;color:var(--muted)}
.mup{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center}
.mup .col{display:flex;flex-direction:column;gap:6px}
.mup .col.a{text-align:left}.mup .col.h{text-align:right}
.mup .nm{font-weight:700;font-size:14px}
.mup .sub{font-size:11.5px;color:var(--muted)}
.mup .vs{font-family:var(--mono);color:var(--faint);font-size:12px}
.statline{display:flex;gap:10px;flex-wrap:wrap;font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.mup .col.h .statline{justify-content:flex-end}
.statline b{color:var(--ink)}
.cmp{width:100%;font-size:13px}
.cmp td{padding:6px 8px}
.cmp .lbl{text-align:center;color:var(--muted);font-size:11px;letter-spacing:.04em;text-transform:uppercase}
.cmp .a{text-align:left}.cmp .h{text-align:right}
.cmp .win{color:var(--fav);font-weight:700}

.livepanel{background:var(--sunk);border:1px solid var(--line-2);border-radius:10px;padding:11px 15px;
  margin-bottom:12px;display:none;align-items:center;gap:14px}
.livepanel .sc{font-family:var(--mono);font-size:22px;font-weight:700}
.livepanel .st{font-size:12px;color:var(--muted)}
.livepanel.live{border-color:var(--accent)}
.livepanel.live .st{color:var(--accent);font-weight:700}
.livepanel .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);
  margin-right:5px;animation:pulse 1.4s ease-in-out infinite}
.updated{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-left:8px}
.updated .dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--fav);
  margin-right:4px;animation:pulse 2s ease-in-out infinite}
/* rating badge (GlassBox 0-100) */
.gb{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-weight:700}
.gb .v{font-size:15px}
.gb .meter{width:46px;height:6px;border-radius:4px;background:var(--sunk);overflow:hidden;position:relative}
.gb .meter i{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
.gb.hi .v{color:var(--fav)} .gb.mid .v{color:var(--ink)} .gb.lo .v{color:var(--home)}
.gb.hi .meter i{background:var(--fav)} .gb.mid .meter i{background:var(--away)} .gb.lo .meter i{background:var(--home)}

/* teams directory + team page */
.tgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.tcard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;
  cursor:pointer;display:flex;flex-direction:column;gap:10px;transition:border-color .12s,transform .12s}
.tcard:hover{border-color:var(--line-2);transform:translateY(-1px)}
.tcard .h{display:flex;align-items:center;gap:10px}
.tcard .code{font-family:var(--mono);font-weight:700;font-size:20px}
.tcard .nm{font-size:13px;color:var(--muted);flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tcard .stat{display:flex;gap:14px;font-size:12px;color:var(--muted)}
.tcard .stat b{color:var(--ink);font-family:var(--mono)}
.psearch{width:100%;max-width:420px;padding:9px 14px;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--ink);font:inherit;font-size:14px;margin:2px 0 6px;outline:none}
.psearch:focus{border-color:var(--accent)}
.thead{display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:14px}
.thead .code{font-family:var(--serif);font-size:34px;font-weight:700}
.thead .sub{color:var(--muted);font-size:13px}
.tstats{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:16px}
.tstats .b{background:var(--panel);padding:10px 12px}
.tstats .k{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.tstats .v{font-family:var(--mono);font-size:18px;font-weight:600;margin-top:3px}
.player-link{cursor:pointer}
.player-link:hover{color:var(--accent)}
a.tl{color:inherit;border-bottom:1px dotted var(--line-2)} a.tl:hover{color:var(--accent);border-color:var(--accent)}

/* player page */
.phead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.phead .nm{font-family:var(--serif);font-size:28px;font-weight:700}
.pgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;align-items:start}
.statgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:10px;overflow:hidden}
.statgrid .b{background:var(--panel);padding:9px 10px;text-align:center}
.statgrid .k{font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.statgrid .v{font-family:var(--mono);font-size:16px;font-weight:600;margin-top:2px}

.loading,.empty{padding:60px 0;text-align:center;color:var(--muted)}
footer{border-top:1px solid var(--line);padding:20px 0 46px;color:var(--faint);font-size:11px;line-height:1.6}
footer b{color:var(--muted)} footer a{color:var(--muted);text-decoration:underline}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto}}
@media(max-width:860px){.updated{display:none}}
@media(max-width:700px){.acc{display:none}header .wrap{gap:10px;padding:0 12px}
  .brand{font-size:18px}nav.main a{padding:6px 8px;font-size:12.5px}}
@media(max-width:560px){.grid{grid-template-columns:1fr}nav.main a{padding:6px 6px;font-size:12px}}
"""

JS = r"""
const $ = s => document.querySelector(s);
const state = {board:null, db:null, nfl:null,
  league:(localStorage.getItem("league")||"mlb"), range:"today", nflRange:"year", live:{}, updated:null};
function setLeague(lg){state.league=lg; try{localStorage.setItem("league",lg);}catch(e){}
  const ns=document.getElementById("navseason");
  if(ns) ns.style.display=(lg==="nfl"&&state.nfl&&state.nfl.schedule)?"":"none";
  updAcc();}
const SAPI = "https://statsapi.mlb.com/api/v1";
const root = document.documentElement;
$("#tog").onclick = () => {
  const next = (root.getAttribute("data-theme")||"dark")==="dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  try{ localStorage.setItem("theme", next); }catch(e){}
};

const pctI = x => Math.round(x*100);
const sgn = v => (v>=0?"+":"")+(+v).toFixed(1);
const tier = p => p>=0.62?"strong":(p>=0.555?"lean":"toss");
const norm = s => (s||"").toLowerCase().normalize("NFKD").replace(/[^a-z0-9 ]+/g," ").replace(/\s+/g," ").trim();
const LOC="en-US", TZ="America/New_York";   // all clock/day display is US Eastern
const addDays=(iso,n)=>{const d=new Date(iso+"T00:00:00");d.setDate(d.getDate()+n);return d.toISOString().slice(0,10);};
const fmtTime=u=>new Date(u).toLocaleTimeString(LOC,{hour:"numeric",minute:"2-digit",timeZone:TZ})+" ET";
const fmtDay=(iso,gen)=>{const d=new Date(iso+"T12:00:00Z");   // noon UTC => same ET day for any viewer
  const md=d.toLocaleDateString(LOC,{month:"short",day:"numeric",timeZone:TZ});
  if(iso===gen)return"Today · "+md; if(iso===addDays(gen,1))return"Tomorrow · "+md;
  return d.toLocaleDateString(LOC,{weekday:"long",timeZone:TZ})+" · "+md;};

/* ---------- LIVE (client-side, polls the MLB Stats API directly) ---------- */
async function pollLive(force){
  if(!force && document.hidden) return;   // skip recurring polls in a hidden tab; always run the first
  try{
    const t=new Date(), dd=n=>{const x=new Date(t);x.setDate(x.getDate()+n);return x.toISOString().slice(0,10);};
    const r=await fetch(`${SAPI}/schedule?sportId=1&startDate=${dd(-1)}&endDate=${dd(1)}&hydrate=linescore`);
    const d=await r.json(); const m={};
    for(const day of d.dates||[]) for(const g of day.games||[]){
      const ls=g.linescore||{};
      m[g.game_pk||g.gamePk]={state:g.status.abstractGameState,
        as:g.teams.away.score, hs:g.teams.home.score,
        inning:ls.currentInningOrdinal, half:ls.inningState, top:ls.isTopInning};
    }
    state.live=m; state.updated=new Date();
    applyLive();
  }catch(e){/* StatsAPI hiccup — keep the last good state, try again next tick */}
}
function liveBadge(pk){
  const l=state.live[pk]; if(!l) return null;
  if(l.state==="Live") return {cls:"live",
    html:`<span class="dot"></span>${l.as}-${l.hs} &middot; ${(l.half||"").slice(0,3)} ${l.inning||""}`};
  if(l.state==="Final") return {cls:"final", html:`Final ${l.as}-${l.hs}`, fin:l};
  return null;
}
function applyLive(){
  document.querySelectorAll(".lv[data-pk]").forEach(el=>{
    const b=liveBadge(el.dataset.pk);
    if(b){el.className="lv "+b.cls; el.innerHTML=b.html;}
    else{el.className="lv"; el.textContent=el.dataset.def;}
  });
  const gp=document.querySelector("#livepanel[data-pk]");
  if(gp){const l=state.live[gp.dataset.pk];
    if(l&&(l.state==="Live"||l.state==="Final")){
      gp.style.display="flex"; gp.className="livepanel "+(l.state==="Live"?"live":"");
      gp.innerHTML=`<span class="sc">${gp.dataset.away} ${l.as} &ndash; ${l.hs} ${gp.dataset.home}</span>
        <span class="st">${l.state==="Live"?`<span class="dot"></span>${l.half||""} ${l.inning||""}`:"Final"}</span>`;
    } else gp.style.display="none";
  }
  const u=$("#updated");
  if(u&&state.updated) u.innerHTML=`<span class="dot"></span>live &middot; updated ${state.updated.toLocaleTimeString(LOC,{hour:"numeric",minute:"2-digit",second:"2-digit"})}`;
}

const nflToday=()=>new Date().toLocaleDateString("en-CA",{timeZone:TZ});
function nflProb(g){const near=g.d<=addDays(nflToday(),7);
  return {near, hp:(!near&&g.pmc!=null)?g.pmc:g.ph};}

const ESPN_NFL="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard";
async function pollNfl(){
  if(document.hidden||state.league!=="nfl"||!state.nfl||!state.nfl.schedule) return;
  const today=nflToday();
  if(!state.nfl.schedule.some(g=>g.d===today)) return;      // no NFL slate today
  try{
    const r=await fetch(ESPN_NFL); const data=await r.json();
    const MAP={LAR:"LA",WSH:"WAS"};
    (data.events||[]).forEach(ev=>{
      const c=ev.competitions&&ev.competitions[0]; if(!c) return;
      const H=(c.competitors||[]).find(x=>x.homeAway==="home");
      const A=(c.competitors||[]).find(x=>x.homeAway==="away");
      if(!H||!A) return;
      const h=MAP[H.team.abbreviation]||H.team.abbreviation;
      const a=MAP[A.team.abbreviation]||A.team.abbreviation;
      const el=document.querySelector(`[data-ng="${a}_${h}"]`); if(!el) return;
      const st=ev.status||{}, tp=st.type||{};
      if(tp.state==="in")
        el.innerHTML=`<span class="dot"></span>${A.score}-${H.score} &middot; Q${st.period||""} ${st.displayClock||""}`;
      else if(tp.state==="post") el.textContent=`FINAL ${A.score}-${H.score}`;
    });
  }catch(e){}
}

function updAcc(){
  const el=$("#acc"); if(!el) return;
  if(state.league==="nfl"&&state.nfl){
    const mc=state.nfl.model_card;
    el.innerHTML=`<b>${mc.test_log_loss.toFixed(3)}</b> log loss (NFL)<br>closing line ${mc.close_log_loss.toFixed(3)}`;
  }else if(state.board&&state.board.accuracy){
    const a=state.board.accuracy;
    el.innerHTML=`<b>${a.log_loss.toFixed(3)}</b> log loss<br>coin flip ${a.coinflip.toFixed(3)}`;
  }
}

async function boot(){
  try{
    const bust = "?t=" + Math.floor(Date.now()/60000);   // fresh each minute; beats stale caches
    const [b,db,nfl] = await Promise.all([
      fetch("./data/board.json"+bust, {cache:"no-cache"}).then(r=>r.json()),
      fetch("./data/db.json"+bust, {cache:"no-cache"}).then(r=>r.json()),
      fetch("./data/nfl.json"+bust, {cache:"no-cache"}).then(r=>r.ok?r.json():null).catch(()=>null),
    ]);
    state.board=b; state.db=db; state.nfl=nfl;
    updAcc();
    route();
    pollLive(true); pollNfl();      // first poll always runs, even if the tab loads hidden
    setInterval(()=>{pollLive();pollNfl();}, 25000);   // both APIs cache ~20s
    document.addEventListener("visibilitychange", ()=>{ if(!document.hidden) pollLive(true); });
  }catch(e){ $("#view").innerHTML=`<div class="empty">Could not load data. ${e}</div>`; }
}
window.addEventListener("hashchange", route);

function setNav(v){document.querySelectorAll("nav.main a").forEach(a=>a.classList.toggle("on",a.dataset.v===v));}

function route(){
  const h=location.hash.replace(/^#\//,"");
  const [v,arg]=h.split("/");
  window.scrollTo(0,0);
  const ns=document.getElementById("navseason");
  if(ns) ns.style.display=(state.league==="nfl"&&state.nfl&&state.nfl.schedule)?"":"none";
  if(v==="season"&&state.nfl&&state.nfl.schedule){state.league="nfl";if(ns)ns.style.display="";updAcc();setNav("season");return nflSeason(arg);}
  if(v==="game"&&arg) return /_/.test(arg)&&state.nfl?nflGamePage(arg):gamePage(arg);
  if(v==="team"&&arg){setNav("teams");return teamPage(arg);}
  if(v==="player"&&arg){setNav("teams");return playerPage(arg);}
  if(v==="players"){setNav("players");return playersPage();}
  if(v==="teams"){setNav("teams");return teamsPage();}
  if(v==="standings"){setNav("standings");return standings();}
  setNav("board"); board();
}

/* ---------- shared helpers ---------- */
const gbTier = v => v>=58?"hi":(v>=44?"mid":"lo");
function gb(v){ if(v==null) return '<span class="gb lo"><span class="v">NR</span></span>';
  return `<span class="gb ${gbTier(v)}"><span class="v">${v.toFixed(0)}</span>
    <span class="meter"><i style="width:${Math.max(3,Math.min(100,v))}%"></i></span></span>`;}
const teamLink = (code,label) => `<a class="tl" href="#/team/${code}">${label||code}</a>`;
const playerLinkByName = name => { const p=findPlayerByName(name);
  return p?`<span class="player-link" onclick="location.hash='#/player/${p.id}'">${name}</span>`:name; };
function findPlayerByName(name){ const n=norm(name);
  return Object.values(state.db.players||{}).find(p=>norm(p.name)===n); }

/* ---------- BOARD ---------- */
const pitEra = name=>{const p=findPlayerByName(name);
  return p&&p.pit&&p.pit.era!=null?`<span class="era">${p.pit.era}</span>`:"";};
function gcard(g){
  const hp=g.home_win_prob, homeWin=hp>=0.5, e=g.edge||{};
  const spTag=(nm,proj)=>nm==="TBD"?`<span class="tbd">TBD</span>`
    :`${nm} ${pitEra(nm)}${proj?' <span class="pj">proj</span>':''}`;
  const spA=spTag(g.away_sp,g.away_sp_proj), spH=spTag(g.home_sp,g.home_sp_proj);
  const val=g.value;
  const valbar = (val&&val.available)
    ? `<div class="valbar"><span class="vt">EDGE</span> ${val.team} <b>+${Math.round(val.ev_cur*100)}% EV</b>
        <span class="vodds">@ ${val.cur_dec}</span><span class="vlive">● live</span></div>` : "";
  return `<a class="gc${val&&val.available?' hasval':''}" href="#/game/${g.game_pk}" data-card="${g.game_pk}"
      data-pick="${g.pick}" data-home="${g.home_abbr}" data-away="${g.away_abbr}">
    ${valbar}
    <div class="top"><span class="lv" data-pk="${g.game_pk}" data-def="${fmtTime(g.start_utc)}">${fmtTime(g.start_utc)}</span>
      ${g.lineup_source==="official"?'<span class="lbadge off">Lineups in</span>'
        :g.sp_projected?'<span class="lbadge proj">Projected</span>'
        :g.lineup_source==="projected"?'<span class="lbadge proj">Proj lineup</span>'
        :(g.pitcher_known?"":'<span class="lbadge tbd">Starters TBD</span>')}
      <span class="pill ${tier(g.pick_prob)==="strong"?"strong":""}">${g.pick} ${pctI(g.pick_prob)}%</span></div>
    <div class="side ${homeWin?"":"win"}"><span class="ab">${g.away_abbr}</span>
      <span class="who"><span class="sp">${spA}</span></span><span class="pc">${pctI(1-hp)}%</span></div>
    <div class="side ${homeWin?"win":""}"><span class="ab">${g.home_abbr}</span>
      <span class="who"><span class="sp">${spH}</span></span><span class="pc">${pctI(hp)}%</span></div>
    <div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
    <div class="edge"><span class="k">edge</span>
      <span class="${e.home_pitcher>=0?"p":"n"}">SP ${sgn((e.home_pitcher||0)-(e.away_pitcher||0)>=0?Math.max(e.home_pitcher||0,0):(e.away_pitcher||0))}</span>
      <span class="${e.team>=0?"p":"n"}">team ${sgn(e.team||0)}</span>
      ${e.bullpen!=null&&Math.abs(e.bullpen)>=0.3?`<span class="${e.bullpen>=0?"p":"n"}">pen ${sgn(e.bullpen)}</span>`:""}
      ${e.power!=null&&Math.abs(e.power)>=0.3?`<span class="${e.power>=0?"p":"n"}">pow ${sgn(e.power)}</span>`:""}
      ${e.baserun!=null&&Math.abs(e.baserun)>=0.3?`<span class="${e.baserun>=0?"p":"n"}">run ${sgn(e.baserun)}</span>`:""}</div>
  </a>`;
}
function board(){
  if(state.league==="nfl"&&state.nfl) return nflPage();
  const b=state.board, lg=b.leagues.find(l=>l.code===state.league), gen=b.generated;
  const R={today:[gen,gen],tomorrow:[addDays(gen,1),addDays(gen,1)],week:[gen,addDays(gen,6)],month:[gen,addDays(gen,60)]}[state.range];
  const games=(lg.games||[]).filter(g=>g.date>=R[0]&&g.date<=R[1]);
  const rail=b.leagues.map(l=>{
    if(l.active)
      return `<button class="lg ${l.code===state.league?"on":""}" data-lg="${l.code}"><span>${l.name}</span><span class="n">${l.n_games}</span></button>`;
    if(l.code==="nfl"&&state.nfl)
      return `<button class="lg ${state.league==="nfl"?"on":""}" data-lg="nfl"><span>NFL</span><span class="n">2026</span></button>`;
    return `<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`;}).join("");
  const filts=[["today","Today"],["tomorrow","Tomorrow"],["week","Week"],["month","Month"]]
    .map(([k,t])=>`<button class="filt ${k===state.range?"on":""}" data-r="${k}">${t}</button>`).join("");
  let body;
  if(!games.length){body=`<div class="empty">No games in this window.</div>`;}
  else{const days=[...new Set(games.map(g=>g.date))].sort();
    body=days.map(d=>{const gs=games.filter(g=>g.date===d),allTbd=gs.every(g=>!g.pitcher_known);
      return `<div class="day"><span class="d">${fmtDay(d,gen)}</span><span class="c">${gs.length} games</span>
        ${allTbd?'<span class="proj">projected · starters tbd</span>':""}</div>
        <div class="grid">${gs.map(gcard).join("")}</div>`;}).join("");}
  $("#view").innerHTML=`<div class="controls"><div class="rail">${rail}</div><div class="filters">${filts}</div></div>${body}`;
  $("#view").querySelectorAll("[data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);board();});
  $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.range=x.dataset.r;board();});
  applyLive();
}

/* ---------- NFL (preseason) ---------- */
function nflPage(){
  const n=state.nfl, mc=n.model_card, b=state.board;
  const rail=b.leagues.map(l=>{
    if(l.active)
      return `<button class="lg" data-lg="${l.code}"><span>${l.name}</span><span class="n">${l.n_games}</span></button>`;
    if(l.code==="nfl")
      return `<button class="lg on" data-lg="nfl"><span>NFL</span><span class="n">2026</span></button>`;
    return `<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`;}).join("");
  const pow=n.power.map(t=>`<tr onclick="location.hash='#/team/${t.code}'">
    <td><span class="num">${t.rank}</span></td>
    <td class="a"><span class="ab" style="color:var(--accent)">${t.code}</span> <span class="sub">${t.name}</span></td>
    <td><span class="num">${t.elo.toFixed(0)}</span></td>
    <td><span class="num ${t.off_pass>=0?"pos":"neg"}">${t.off_pass>=0?"+":""}${t.off_pass.toFixed(1)}</span></td>
    <td><span class="num ${t.off_run>=0?"pos":"neg"}">${t.off_run>=0?"+":""}${t.off_run.toFixed(1)}</span></td>
    <td><span class="num ${t.def_pass>=0?"pos":"neg"}">${t.def_pass>=0?"+":""}${t.def_pass.toFixed(1)}</span></td>
    <td><span class="num ${t.def_run>=0?"pos":"neg"}">${t.def_run>=0?"+":""}${t.def_run.toFixed(1)}</span></td></tr>`).join("");
  const bd=pos=>n.boards[pos].map((p,i)=>`<tr ${p.id?`onclick="location.hash='#/player/${p.id}'"`:""}><td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.player}</span></td><td><span class="num">${p.z>=0?"+":""}${p.z.toFixed(2)}</span></td>
    <td><span class="num">${p.n.toLocaleString()}</span></td></tr>`).join("");
  const btbl=(t,pos)=>`<div class="panel"><h3>${t}</h3><div class="twrap"><table>
    <thead><tr><th></th><th>Player</th><th>Value z</th><th>Plays</th></tr></thead>
    <tbody>${bd(pos)}</tbody></table></div></div>`;
  const mvp=n.mvp.map((m,i)=>`<tr ${m.id?`onclick="location.hash='#/player/${m.id}'"`:""}><td><span class="num">${i+1}</span></td><td class="a"><span class="player-link">${m.player}</span></td>
    <td><span class="num pos">+${m.pts.toFixed(2)}</span></td><td><span class="num">${m.n_abs}</span></td></tr>`).join("");
  const cal=mc.calibration.map(c=>`<tr><td class="a">${c.bucket}</td>
    <td><span class="num">${c.hit.toFixed(1)}%</span></td><td><span class="num">${c.n}</span></td></tr>`).join("");
  const sch=n.schedule||[], proj=n.proj||{};
  const today=new Date().toLocaleDateString("en-CA",{timeZone:TZ});
  const NR={today:[today,today],week:[today,addDays(today,6)],month:[today,addDays(today,29)],year:["2000-01-01","2099-01-01"]}[state.nflRange];
  const games=sch.filter(g=>g.d>=NR[0]&&g.d<=NR[1]);
  const filts=[["today","Today"],["week","Week"],["month","Month"],["year","Year"]]
    .map(([k,t])=>`<button class="filt ${k===state.nflRange?"on":""}" data-r="${k}">${t}</button>`).join("");
  const qbOf=c=>n.teams&&n.teams[c]&&n.teams[c].qb1?n.teams[c].qb1.name:"";
  const ncard=g=>{
    const pr=nflProb(g), hp=pr.hp, homeWin=hp>=0.5, done=g.hs!=null;
    const pick=homeWin?g.home:g.away, pp=Math.max(hp,1-hp);
    const val=g.value;
    const valbar=(val&&val.available)
      ? `<div class="valbar"><span class="vt">EDGE</span> ${val.team} <b>+${Math.round(val.ev_cur*100)}% EV</b>
          <span class="vodds">@ ${val.cur_dec}</span><span class="vlive">● live</span></div>` : "";
    let badge=pr.near
      ?`<span class="lbadge off">Model &middot; Wk ${g.w}</span><span class="lbadge proj">Proj lineup</span>`
      :`<span class="lbadge proj">Sim &middot; Wk ${g.w}</span>`;
    if(done){const winner=g.hs>g.as?g.home:(g.hs<g.as?g.away:null);
      badge=winner==null?`<span class="lbadge proj">TIE</span>`
        :(winner===pick?`<span class="lbadge off">HIT</span>`:`<span class="lbadge tbd">MISS</span>`);}
    return `<a class="gc${val&&val.available?' hasval':''}" href="#/game/${g.w}_${g.away}_${g.home}">
      ${valbar}
      <div class="top"><span class="lv" data-ng="${g.away}_${g.home}">${done?`FINAL ${g.as}-${g.hs}`:`${g.d.slice(5)} &middot; ${g.t?g.t.slice(0,5):""} ET`}</span>
        ${badge}
        <span class="pill ${tier(pp)==="strong"?"strong":""}">${pick} ${pctI(pp)}%</span></div>
      <div class="side ${homeWin?"":"win"}"><span class="ab">${g.away}</span>
        <span class="who"><span class="sp">${qbOf(g.away)}</span></span><span class="pc">${pctI(1-hp)}%</span></div>
      <div class="side ${homeWin?"win":""}"><span class="ab">${g.home}</span>
        <span class="who"><span class="sp">${qbOf(g.home)}</span></span><span class="pc">${pctI(hp)}%</span></div>
      <div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
    </a>`;};
  let cardsBody;
  if(!games.length){cardsBody=`<div class="empty">No NFL games in this window &mdash; the 2026 season kicks off September&nbsp;9. Pick <b>Year</b> to see every game.</div>`;}
  else{const days=[...new Set(games.map(g=>g.d))].sort();
    cardsBody=days.map(d=>{const gs=games.filter(g=>g.d===d);
      return `<div class="day"><span class="d">${fmtDay(d,today)}</span><span class="c">${gs.length} games</span></div>
        <div class="grid">${gs.map(ncard).join("")}</div>`;}).join("");}
  const pTop=Object.entries(proj).sort((x,y)=>y[1].w-x[1].w).slice(0,8).map(([c,p],i)=>
    `<tr onclick="location.hash='#/standings'"><td><span class="num">${i+1}</span></td><td class="a">${c}</td>
      <td><span class="num">${p.w.toFixed(1)}-${(17-p.w).toFixed(1)}</span></td>
      <td><span class="num">${p.po.toFixed(0)}%</span></td></tr>`).join("");
  if(!games.length){
    $("#view").innerHTML=`<div class="controls"><div class="rail">${rail}</div><div class="filters">${filts}</div></div>${cardsBody}`;
    $("#view").querySelectorAll("[data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);board();});
    $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.nflRange=x.dataset.r;board();});
    return;
  }
  $("#view").innerHTML=`<div class="controls"><div class="rail">${rail}</div><div class="filters">${filts}</div></div>
    ${cardsBody}
    <div class="eyebrow" style="margin-top:22px">NFL &middot; ${n.season} season</div>
    <h1 class="pt">NFL Board <span class="sub" style="font-weight:400">all 272 games predicted &middot; season simulated on every build</span></h1>
    <div class="sub" style="margin-bottom:10px">Every row is clickable &mdash; teams, players, weeks. <a href="#/season" style="color:var(--accent)">Season</a> has every weekly pick; <a href="#/standings" style="color:var(--accent)">Standings</a> has the full projections; <a href="#/teams" style="color:var(--accent)">Teams</a> has roster ratings.</div>
    <div class="sub" style="margin-bottom:14px">Market-blind model &middot; <b>${(mc.accuracy).toFixed(1)}%</b> accurate on a locked
      ${mc.holdout} holdout &middot; log loss <b>${mc.test_log_loss.toFixed(3)}</b> (closing line ${mc.close_log_loss.toFixed(3)})
      &middot; ${mc.n_features} features &middot; ${mc.n_tests} documented tests &middot; ${mc.training}.</div>
    ${sch.length?`<div class="grid" style="margin-bottom:4px">
      <div class="panel"><h3>Projected standings &middot; <a href="#/standings" style="color:var(--accent)">all 32 &rarr;</a></h3>
        <div class="twrap"><table><thead><tr><th></th><th>Team</th><th>Proj W-L</th><th>Playoffs</th></tr></thead><tbody>${pTop}</tbody></table></div>
        <div class="sub" style="margin-top:8px">20,000 season simulations from the per-game model probabilities. QB1s from live depth charts; team states roll forward as games are played.</div></div>
    </div>`:""}
    <div class="subh">2026 preseason power ratings</div>
    <div class="sub" style="margin-bottom:8px">Elo regressed to the season prior. Unit values = EPA/play vs league, x100 (+ = good), from the pass/run split ratings.</div>
    <div class="twrap"><table><thead><tr><th></th><th>Team</th><th>Elo</th>
      <th>Pass off</th><th>Run off</th><th>Pass def</th><th>Run def</th></tr></thead><tbody>${pow}</tbody></table></div>
    <div class="grid" style="margin-top:16px">${btbl("Top QBs","QB")}${btbl("Top RBs","RB")}${btbl("Top WRs","WR")}${btbl("Top TEs","TE")}</div>
    <div class="grid" style="margin-top:16px">
      <div class="panel"><h3>MVP impact (with-vs-without, pts/game)</h3><div class="twrap"><table>
        <thead><tr><th></th><th>Player</th><th>Cost when out</th><th>Absences</th></tr></thead>
        <tbody>${mvp}</tbody></table></div>
        <div class="sub" style="margin-top:8px">Measured team drop-off when the player misses games, opponent-adjusted. Elite QBs land inside the market's 4.5&ndash;7.0 point range.</div></div>
      <div class="panel"><h3>Calibration (locked holdout)</h3><div class="twrap"><table>
        <thead><tr><th>Model says</th><th>Actually wins</th><th>Games</th></tr></thead>
        <tbody>${cal}</tbody></table></div>
        <div class="sub" style="margin-top:8px">When this model says 75%, it means 75%. Accuracy ladder: home-always ${mc.acc_home}% &middot; Elo ${mc.acc_elo}% &middot; <b>GlassBox ${mc.accuracy}%</b> &middot; closing line ${mc.acc_close}%.</div></div>
    </div>`;
  $("#view").querySelectorAll("[data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);board();});
  $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.nflRange=x.dataset.r;board();});
}

function nflSeason(wk){
  const n=state.nfl, sch=n.schedule||[];
  if(!sch.length){$("#view").innerHTML=`<div class="empty">No schedule loaded.</div>`;return;}
  const weeks=[...new Set(sch.map(g=>g.w))].sort((a,b)=>a-b);
  let W=parseInt(wk)||0;
  if(!weeks.includes(W)){const up=sch.find(g=>g.hs==null);W=up?up.w:weeks[0];}
  const chips=weeks.map(w=>`<button class="filt ${w===W?"on":""}" data-w="${w}">W${w}</button>`).join("");
  const games=sch.filter(g=>g.w===W).slice().sort((a,b)=>a.d<b.d?-1:(a.d>b.d?1:(a.t<b.t?-1:1)));
  const row=g=>{
    const hp=nflProb(g).hp;
    const pick=hp>=0.5?g.home:g.away, pp=Math.round(100*Math.max(hp,1-hp)), done=g.hs!=null;
    let res=`<td class="sub">&mdash;</td>`;
    if(done){const winner=g.hs>g.as?g.home:(g.hs<g.as?g.away:null);
      const hit=winner==null?null:winner===pick;
      res=`<td><span class="num">${g.as}-${g.hs}</span> ${hit==null?'<span class="sub">tie</span>':hit?'<span class="num pos">&#10003;</span>':'<span class="num neg">&#10007;</span>'}</td>`;}
    return `<tr onclick="location.hash='#/game/${g.w}_${g.away}_${g.home}'"><td class="sub">${g.d.slice(5)} ${g.t?g.t.slice(0,5):""}</td>
      <td class="a">${teamLink(g.away,g.away)} <span class="sub">at</span> ${teamLink(g.home,g.home)}${g.neutral?' <span class="sub">(neutral)</span>':""}</td>
      <td><span class="pill ${pp>=65?"strong":""}">${pick} ${pp}%</span></td>
      <td><div class="pbar" style="min-width:110px;margin:0"><div class="h" style="width:${Math.round(nflProb(g).hp*100)}%"></div><div class="mid"></div></div></td>
      ${res}</tr>`;};
  $("#view").innerHTML=`<div class="eyebrow">NFL &middot; ${n.season} season</div>
    <h1 class="pt">Season <span class="sub" style="font-weight:400">every game, predicted &middot; market-blind</span></h1>
    <div class="sub" style="margin-bottom:12px">Games inside <b>one week</b> use the live model; anything further out uses the <b>20,000-run season simulation</b> (team strength evolves inside every sim). Probability bar shows the <b>home</b> side. Refreshes on every data build.</div>
    <div class="filters" style="flex-wrap:wrap;margin-bottom:12px">${chips}</div>
    <div class="subh">Week ${W} &middot; ${games.length} games</div>
    <div class="twrap"><table><thead><tr><th>Kickoff (ET)</th><th>Game</th><th>Pick</th><th>Home win prob</th><th>Result</th></tr></thead>
      <tbody>${games.map(row).join("")}</tbody></table></div>`;
  $("#view").querySelectorAll("[data-w]").forEach(x=>x.onclick=()=>{location.hash="#/season/"+x.dataset.w;});
}

function nflStandings(){
  const n=state.nfl, T=Object.values(n.teams);
  const divs={}; T.forEach(t=>{(divs[t.div]=divs[t.div]||[]).push(t);});
  const order=n.divisions;
  let html=`<div class="eyebrow">Database &middot; NFL</div><h1 class="pt">Standings <span class="sub" style="font-weight:400">2026 projected &middot; 2025 final below</span></h1>
    <div class="sub" style="margin-bottom:14px">Projection = 20,000 season simulations of the market-blind model, refreshed on every data build. Click any team.</div>`;
  const proj=n.proj||{};
  if(Object.keys(proj).length){
    html+=order.filter(d=>divs[d]).map(d=>{
      const rows=divs[d].slice().sort((a,b)=>((proj[b.code]||{}).w||0)-((proj[a.code]||{}).w||0)).map(t=>{const p=proj[t.code]||{};
        return `<tr onclick="location.hash='#/team/${t.code}'">
        <td class="a"><span class="ab" style="color:var(--accent)">${t.abbr}</span> <span class="sub">${t.name}</span></td>
        <td><span class="num">${p.w!=null?p.w.toFixed(1)+"-"+(17-p.w).toFixed(1):"-"}</span></td>
        <td><span class="num">${p.div!=null?p.div.toFixed(0)+"%":"-"}</span></td>
        <td><span class="num">${p.po!=null?p.po.toFixed(0)+"%":"-"}</span></td>
        <td><span class="num">${t.elo?t.elo.toFixed(0):"-"}</span></td>
        <td><span class="num">#${t.rank||"-"}</span></td></tr>`;}).join("");
      return `<div class="subh">${d} &middot; projection</div><div class="twrap"><table>
        <thead><tr><th>Team</th><th>Proj W-L</th><th>Win div</th><th>Playoffs</th><th>Elo</th><th>Power</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }).join("");
    html+=`<div class="subh" style="margin-top:20px">2025 final standings</div>`;
  }
  html+=order.filter(d=>divs[d]).map(d=>{
    const rows=divs[d].sort((a,b)=>a.div_rank-b.div_rank).map(t=>`
      <tr onclick="location.hash='#/team/${t.code}'">
        <td class="a"><span class="ab" style="color:var(--accent)">${t.abbr}</span> <span class="sub">${t.name}</span></td>
        <td><span class="num">${t.w}-${t.l}${t.t?"-"+t.t:""}</span></td>
        <td><span class="num">${t.pct.toFixed(3).slice(1)}</span></td>
        <td><span class="num ${t.pt_diff>=0?"pos":"neg"}">${t.pt_diff>=0?"+":""}${t.pt_diff}</span></td>
        <td><span class="num">${t.l10}</span></td><td><span class="num">${t.streak}</span></td>
        <td><span class="num">${t.home}</span></td><td><span class="num">${t.away}</span></td>
        <td><span class="num">${t.elo?t.elo.toFixed(0):"-"}</span></td>
        <td><span class="num">#${t.rank||"-"}</span></td></tr>`).join("");
    return `<div class="subh">${d}</div><div class="twrap"><table>
      <thead><tr><th>Team</th><th>W-L</th><th>Pct</th><th>Pt diff</th><th>L10</th><th>Strk</th>
      <th>Home</th><th>Away</th><th>Elo 26</th><th>Power</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }).join("");
  $("#view").innerHTML=html;
}

function nflTeams(){
  const n=state.nfl, T=Object.values(n.teams).sort((a,b)=>(b.glassbox||0)-(a.glassbox||0));
  const card=t=>`<div class="tcard" onclick="location.hash='#/team/${t.code}'">
    <div class="h"><span class="code" style="color:var(--accent)">${t.abbr}</span><span class="nm">${t.name}</span>${gb(t.glassbox)}</div>
    <div class="stat"><span>Proj 26 <b>${t.proj?t.proj.w.toFixed(1)+"W":"-"}</b></span>
      <span>PO <b>${t.proj?t.proj.po.toFixed(0)+"%":"-"}</b></span>
      <span>Elo <b>${t.elo?t.elo.toFixed(0):"-"}</b></span><span>Power <b>#${t.rank||"-"}</b></span></div>
    <div class="stat"><span>Off <b>${t.gb_off!=null?t.gb_off.toFixed(0):"-"}</b></span>
      <span>Def <b>${t.gb_def!=null?t.gb_def.toFixed(0):"-"}</b></span>
      <span>QB <b>${t.qb1?t.qb1.name:"-"}</b></span></div>
  </div>`;
  $("#view").innerHTML=`<div class="eyebrow">Database &middot; NFL</div><h1 class="pt">Teams</h1>
    <div class="sub" style="margin-bottom:14px">The <b>GlassBox rating</b> (0-100) is the roster's snap-weighted average <b>per-play TrueSkill</b> rating &mdash; every snap is an 11-vs-11 match, all 22 players Bayesian-updated on the outcome (344,801 plays, 2016-2025). 50 = league average at the position. Off = offensive personnel, Def = defensive. QB1 from live depth charts. Click a team.</div>
    <div class="tgrid">${T.map(card).join("")}</div>`;
}

function nflTeamPage(code){
  const n=state.nfl, t=n.teams[code];
  if(!t){$("#view").innerHTML=`<div class="empty">Team not found. <a href="#/teams">All teams</a></div>`;return;}
  const P=t.roster.map(id=>n.players[id]).filter(Boolean);
  const OFFP=["QB","RB","FB","WR","TE"], OLP=["T","G","C","OT","OG","OL","LT","RT","LG","RG"];
  const grp=p=>OFFP.includes(p.pos)?"off":(OLP.includes(p.pos)?"ol":"def");
  const statStr=p=>{const s=p.stats||{};
    if(p.pos==="QB")return `${s.pass_yds||0} yds &middot; ${s.pass_td||0} TD &middot; ${s.ints||0} INT`;
    if(["RB","FB"].includes(p.pos))return `${s.rush_yds||0} rush &middot; ${s.rec||0} rec &middot; ${(s.rush_td||0)+(s.rec_td||0)} TD`;
    if(["WR","TE"].includes(p.pos))return `${s.rec||0} rec &middot; ${s.rec_yds||0} yds &middot; ${s.rec_td||0} TD`;
    return `${s.tak||0} tkl &middot; ${s.sk||0} sk &middot; ${s.dint||0} INT &middot; ${s.pd||0} PD`;};
  const prow=p=>`<tr onclick="location.hash='#/player/${p.id}'">
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos}</span></td>
    <td>${p.rating?gb(p.rating.r):gb(null)}</td>
    <td><span class="num">${Math.round((p.snap_share||0)*100)}%</span></td>
    <td class="sub">${statStr(p)}</td></tr>`;
  const tbl=(title,rows)=>rows.length?`<div class="subh">${title}</div><div class="twrap"><table>
    <thead><tr><th>Player</th><th>GlassBox</th><th>Snap %</th><th>2025</th></tr></thead>
    <tbody>${rows.map(prow).join("")}</tbody></table></div>`:"";
  const off=P.filter(p=>grp(p)==="off"), ol=P.filter(p=>grp(p)==="ol"), def=P.filter(p=>grp(p)==="def");
  const p26=t.proj, q=t.qb1;
  const sch=(n.schedule||[]).filter(g=>g.home===code||g.away===code);
  const srow2=g=>{const homeG=g.home===code, opp=homeG?g.away:g.home;
    const hp=nflProb(g).hp, pw=homeG?hp:1-hp;
    const pick=hp>=0.5?g.home:g.away;
    return `<tr onclick="location.hash='#/game/${g.w}_${g.away}_${g.home}'"><td><span class="num">W${g.w}</span></td><td class="sub">${g.d.slice(5)}</td>
      <td class="a">${homeG?"vs":"at"} ${teamLink(opp,opp)}${g.neutral?' <span class="sub">(n)</span>':""}</td>
      <td><span class="num ${pw>=0.5?"pos":"neg"}">${Math.round(pw*100)}%</span></td>
      <td><span class="pill ${Math.max(hp,1-hp)>=0.65?"strong":""}">${pick}</span></td></tr>`;};
  const schTbl=sch.length?`<div class="subh">2026 schedule &middot; model picks</div><div class="twrap"><table>
    <thead><tr><th>Wk</th><th>Date</th><th>Opponent</th><th>Win prob</th><th>Pick</th></tr></thead>
    <tbody>${sch.map(srow2).join("")}</tbody></table></div>`:"";
  const projLine=p26?`<div class="sub" style="margin-bottom:14px">Projected 2026: <b>${p26.w.toFixed(1)}-${(17-p26.w).toFixed(1)}</b>
    &middot; win division <b>${p26.div.toFixed(0)}%</b> &middot; make playoffs <b>${p26.po.toFixed(0)}%</b>
    ${q&&q.id&&n.players[q.id]?`&middot; QB1 <span class="player-link" onclick="location.hash='#/player/${q.id}'">${q.name}</span>`:(q?`&middot; QB1 ${q.name}`:"")}</div>`:"";
  $("#view").innerHTML=`<a class="back" href="#/teams">&lsaquo; Teams</a>
    <div class="eyebrow">${t.div}</div><h1 class="pt">${t.name}</h1>
    <div class="sub" style="margin-bottom:14px">${t.w}-${t.l}${t.t?"-"+t.t:""} in 2025 &middot; ${t.pf} PF / ${t.pa} PA
      &middot; Elo <b>${t.elo?t.elo.toFixed(0):"-"}</b> (power #${t.rank})
      &middot; units: pass off ${t.off_pass>=0?"+":""}${t.off_pass} / run off ${t.off_run>=0?"+":""}${t.off_run}
      / pass def ${t.def_pass>=0?"+":""}${t.def_pass} / run def ${t.def_run>=0?"+":""}${t.def_run}</div>
    ${projLine}${t.lineup?nflLineupPanel(code,t,n).replace('class="panel"','class="panel" style="margin-bottom:14px"'):""}${schTbl}${tbl("Offense",off)}${tbl("Offensive line",ol)}${tbl("Defense",def)}`;
}

function nflPlayerPage(id){
  const n=state.nfl, p=n.players[id];
  if(!p){$("#view").innerHTML=`<div class="empty">Player not found.</div>`;return;}
  const t=n.teams[p.team], s=p.stats||{};
  const srow=(k,v)=>v?`<tr><td class="a">${k}</td><td><span class="num">${v.toLocaleString()}</span></td></tr>`:"";
  let srows="";
  if(p.pos==="QB") srows=srow("Pass yards",s.pass_yds)+srow("Pass TD",s.pass_td)+srow("INT",s.ints)+srow("Rush yards",s.rush_yds)+srow("Rush TD",s.rush_td);
  else if(["RB","FB"].includes(p.pos)) srows=srow("Rush yards",s.rush_yds)+srow("Rush TD",s.rush_td)+srow("Receptions",s.rec)+srow("Rec yards",s.rec_yds)+srow("Rec TD",s.rec_td);
  else if(["WR","TE"].includes(p.pos)) srows=srow("Targets",s.tgt)+srow("Receptions",s.rec)+srow("Rec yards",s.rec_yds)+srow("Rec TD",s.rec_td);
  else srows=srow("Solo tackles",s.tak)+srow("Sacks",s.sk)+srow("INT",s.dint)+srow("Passes defended",s.pd);
  $("#view").innerHTML=`<a class="back" href="#/team/${p.team}">&lsaquo; ${t?t.name:p.team}</a>
    <div class="eyebrow">${p.pos} &middot; ${t?t.name:p.team}</div><h1 class="pt">${p.name}</h1>
    <div class="grid" style="margin-top:12px">
      <div class="panel"><h3>GlassBox rating <span class="sub" style="font-weight:400">per-play TrueSkill</span></h3>
        <div style="display:flex;align-items:center;gap:14px;margin:6px 0">${p.rating?gb(p.rating.r):gb(null)}
          ${p.rating?`<span class="sub">tier <b>${p.rating.tier}</b> &middot; ${p.rating.n_eff.toLocaleString()} plays rated${p.rating.mu!=null?` &middot; TrueSkill &mu; ${p.rating.mu.toFixed(1)} &plusmn; ${p.rating.sigma.toFixed(1)}`:""}</span>`:`<span class="sub">not enough sample</span>`}</div>
        ${p.board?`<div class="sub">Per-play value: z ${p.board.z>=0?"+":""}${p.board.z.toFixed(2)}
          (conservative ${p.board.cons>=0?"+":""}${p.board.cons.toFixed(2)}) over ${p.board.n.toLocaleString()} plays, opponent-adjusted.</div>`
          :`<div class="sub">No per-play board entry.</div>`}
        <div class="sub" style="margin-top:6px">Snap share ${Math.round((p.snap_share||0)*100)}%</div></div>
      <div class="panel"><h3>2025 season</h3><div class="twrap"><table><tbody>${srows||`<tr><td class="sub">No stats recorded.</td></tr>`}</tbody></table></div></div>
    </div>`;
}

/* ---------- PLAYERS TAB ---------- */
function playersPage(){
  if(state.league==="nfl"&&state.nfl) return nflPlayers();
  const db=state.db, P=Object.values(db.players||{});
  const bats=P.filter(p=>p.role==="batter"&&p.ts100!=null).sort((a,b)=>b.ts100-a.ts100);
  const pits=P.filter(p=>p.role==="pitcher"&&p.ts100!=null).sort((a,b)=>b.ts100-a.ts100);
  const brow=(p,i)=>`<tr onclick="location.hash='#/player/${p.id}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
    <td>${gb(p.ts100)}</td>
    <td><span class="num">${p.ts_mu!=null?p.ts_mu.toFixed(1):"-"}</span></td>
    <td class="sub">${p.bat?`${p.bat.ops!=null?p.bat.ops:"-"} OPS &middot; ${p.bat.hr||0} HR &middot; ${p.bat.sb||0} SB`:"&mdash;"}</td></tr>`;
  const prow=(p,i)=>`<tr onclick="location.hash='#/player/${p.id}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
    <td>${gb(p.ts100)}</td>
    <td><span class="num">${p.ts_mu!=null?p.ts_mu.toFixed(1):"-"}</span></td>
    <td class="sub">${p.pit?`${p.pit.era!=null?p.pit.era:"-"} ERA &middot; ${p.pit.whip!=null?p.pit.whip:"-"} WHIP &middot; ${p.pit.so||0} K`:"&mdash;"}</td></tr>`;
  const tbl=(title,rows,statHdr)=>`<div class="panel"><h3>${title}</h3><div class="twrap"><table>
    <thead><tr><th></th><th>Player</th><th>GlassBox</th><th class="nocase">&mu;</th><th>${statHdr}</th></tr></thead>
    <tbody>${rows}</tbody></table></div></div>`;
  $("#view").innerHTML=`<div class="eyebrow">Database &middot; MLB</div><h1 class="pt">Players</h1>
    <div class="sub" style="margin-bottom:10px">Every plate appearance is a <b>TrueSkill duel</b>: batter vs pitcher, both Bayesian-updated
      on the outcome. The 0-100 <b>GlassBox rating</b> comes straight from that ladder (50 = league average); &mu; is the raw skill estimate.
      ${bats.length+pits.length} rated players. Click anyone.</div>
    <input class="psearch" id="psearch" placeholder="Search ${P.length} players&hellip;" autocomplete="off">
    <div id="psres"></div>
    <div class="grid" style="margin-top:8px">
      ${tbl("Top hitters",bats.slice(0,15).map(brow).join(""),"2026 season")}
      ${tbl("Top pitchers",pits.slice(0,15).map(prow).join(""),"2026 season")}
    </div>`;
  const res=$("#psres");
  $("#psearch").oninput=e=>{
    const q=norm(e.target.value);
    if(q.length<2){res.innerHTML="";return;}
    const hits=P.filter(p=>norm(p.name).includes(q)).slice(0,20);
    res.innerHTML=hits.length?`<div class="twrap" style="margin-bottom:6px"><table><tbody>
      ${hits.map(p=>`<tr onclick="location.hash='#/player/${p.id}'">
        <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
        <td>${gb(p.ts100)}</td>
        <td class="sub">${p.role==="pitcher"?(p.pit?`${p.pit.era!=null?p.pit.era:"-"} ERA`:""):(p.bat?`${p.bat.ops!=null?p.bat.ops:"-"} OPS`:"")}</td></tr>`).join("")}
      </tbody></table></div>`:`<div class="sub" style="margin:6px 0 10px">No players match.</div>`;
  };
}

function nflPlayers(){
  const n=state.nfl, P=Object.values(n.players||{}).filter(p=>p.rating);
  const GROUPS=["QB","RB","WR","TE","OL","DL","LB","DB"];
  const GN={QB:"Quarterbacks",RB:"Running backs",WR:"Receivers",TE:"Tight ends",
            OL:"Offensive line",DL:"Defensive line",LB:"Linebackers",DB:"Secondary"};
  const byg={}; GROUPS.forEach(g=>byg[g]=[]);
  P.forEach(p=>{if(byg[p.rating.bucket]) byg[p.rating.bucket].push(p);});
  GROUPS.forEach(g=>byg[g].sort((a,b)=>b.rating.r-a.rating.r));
  const row=(p,i)=>`<tr onclick="location.hash='#/player/${p.id}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.team}</span></td>
    <td>${gb(p.rating.r)}</td>
    <td><span class="num">${p.rating.mu.toFixed(1)}&thinsp;&plusmn;&thinsp;${p.rating.sigma.toFixed(1)}</span></td>
    <td><span class="num">${p.rating.n_eff.toLocaleString()}</span></td></tr>`;
  const panel=g=>`<div class="panel"><h3>${GN[g]}</h3><div class="twrap"><table>
    <thead><tr><th></th><th>Player</th><th>Rating</th><th class="nocase">&mu;&thinsp;&plusmn;&thinsp;&sigma;</th><th>Plays</th></tr></thead>
    <tbody>${byg[g].slice(0,10).map(row).join("")}</tbody></table></div></div>`;
  const proven=P.filter(p=>p.rating.n_eff>=1000).sort((a,b)=>a.rating.sigma-b.rating.sigma).slice(0,8);
  const wild=P.filter(p=>p.rating.n_eff>=300).sort((a,b)=>b.rating.sigma-a.rating.sigma).slice(0,8);
  const mini=(p,i)=>`<tr onclick="location.hash='#/player/${p.id}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.rating.bucket} &middot; ${p.team}</span></td>
    <td>${gb(p.rating.r)}</td>
    <td><span class="num">&sigma; ${p.rating.sigma.toFixed(2)}</span></td></tr>`;
  $("#view").innerHTML=`<div class="eyebrow">Database &middot; NFL</div><h1 class="pt">Players</h1>
    <div class="sub" style="margin-bottom:10px">Every snap is an <b>11-vs-11 TrueSkill match</b> &mdash; all 22 players on the field are
      Bayesian-updated on whether the offense beat the defense (<b>344,801 plays</b>, 2016-2025, garbage time down-weighted).
      Ratings are conservative (&mu;&thinsp;&minus;&thinsp;3&sigma;) and scored within position group, so thin samples can't fake it.
      ${P.length} rated players. Click anyone.</div>
    <input class="psearch" id="psearch" placeholder="Search ${P.length} rated players&hellip;" autocomplete="off">
    <div id="psres"></div>
    <div class="grid" style="margin-top:8px">${GROUPS.map(panel).join("")}</div>
    <div class="grid" style="margin-top:16px">
      <div class="panel"><h3>Bankable <span class="sub" style="font-weight:400">lowest &sigma;, 1,000+ plays &mdash; the engine is surest about these ratings</span></h3>
        <div class="twrap"><table><tbody>${proven.map(mini).join("")}</tbody></table></div></div>
      <div class="panel"><h3>Wild cards <span class="sub" style="font-weight:400">highest &sigma;, 300+ plays &mdash; ratings that could move fastest in 2026</span></h3>
        <div class="twrap"><table><tbody>${wild.map(mini).join("")}</tbody></table></div></div>
    </div>`;
  const res=$("#psres");
  $("#psearch").oninput=e=>{
    const q=norm(e.target.value);
    if(q.length<2){res.innerHTML="";return;}
    const hits=P.filter(p=>norm(p.name).includes(q)).slice(0,20);
    res.innerHTML=hits.length?`<div class="twrap" style="margin-bottom:6px"><table><tbody>
      ${hits.map(p=>`<tr onclick="location.hash='#/player/${p.id}'">
        <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.rating.bucket} &middot; ${p.team}</span></td>
        <td>${gb(p.rating.r)}</td>
        <td><span class="num">${p.rating.mu.toFixed(1)}&thinsp;&plusmn;&thinsp;${p.rating.sigma.toFixed(1)}</span></td>
        <td><span class="num">${p.rating.n_eff.toLocaleString()}</span></td></tr>`).join("")}
      </tbody></table></div>`:`<div class="sub" style="margin:6px 0 10px">No rated players match.</div>`;
  };
}

/* ---------- NFL GAME PAGE ---------- */
function nflGamePage(key){
  setNav("");
  const n=state.nfl;
  if(!n||!n.schedule){$("#view").innerHTML=`<div class="empty">Game not found. <a href="#/">Back to board</a></div>`;return;}
  const parts=key.split("_"), w=parseInt(parts[0]), away=parts[1], home=parts[2];
  const g=n.schedule.find(x=>x.w===w&&x.away===away&&x.home===home);
  if(!g){$("#view").innerHTML=`<div class="empty">Game not found. <a href="#/">Back to board</a></div>`;return;}
  const A=n.teams[away], H=n.teams[home];
  const pr=nflProb(g), hp=pr.hp, homeWin=hp>=0.5, done=g.hs!=null;
  const pick=homeWin?home:away, conf=Math.max(hp,1-hp);
  const ct=g.ct||{};
  const LBL={elo:"Team strength (Elo)", qb:"Quarterback", units:"Pass/run units (EPA)",
             roster:"Roster quality", hfa:"Home field (team)", sched:"Rest &amp; kickoff spot",
             ts:"Unit TrueSkill", luck:"Fumble luck", abs:"Injuries / absences"};
  const ORDER=["elo","qb","units","roster","hfa","sched","ts","luck","abs"];
  const whyRows=ORDER.filter(k=>ct[k]!=null&&(Math.abs(ct[k])>=0.05||["elo","qb","units"].includes(k)))
    .sort((x,y)=>Math.abs(ct[y])-Math.abs(ct[x])).map(k=>pRow(LBL[k],ct[k])).join("");
  const qcell=t=>{const q=t&&t.qb1; if(!q)return "-";
    return q.id&&n.players[q.id]?`<span class="player-link" onclick="location.hash='#/player/${q.id}'">${q.name}</span>`:q.name;};
  const qrat=t=>{const q=t&&t.qb1, p=q&&q.id?n.players[q.id]:null;
    return p&&p.rating?`${gb(p.rating.r)} <div class="sub" style="margin-top:4px">TrueSkill &mu; ${p.rating.mu.toFixed(1)} &plusmn; ${p.rating.sigma.toFixed(1)} &middot; ${p.rating.n_eff.toLocaleString()} plays</div>`:gb(null);};
  const f0=x=>x==null?"-":Math.round(+x), s1=x=>x==null?"-":((+x>=0?"+":"")+(+x).toFixed(1));
  const teamCmp=(A&&H)?`<table class="cmp">
    ${cmpRow("2025 record",`${A.w}-${A.l}${A.t?"-"+A.t:""}`,`${H.w}-${H.l}${H.t?"-"+H.t:""}`,x=>x)}
    ${cmpRow("our Elo",A.elo,H.elo,f0,"hi")}
    ${cmpRow("power rank","#"+A.rank,"#"+H.rank,x=>x)}
    ${cmpRow("proj 2026 wins",A.proj?A.proj.w:null,H.proj?H.proj.w:null,x=>x==null?"-":x.toFixed(1),"hi")}
    ${cmpRow("playoff odds",A.proj?A.proj.po:null,H.proj?H.proj.po:null,x=>x==null?"-":x.toFixed(0)+"%","hi")}
    ${cmpRow("roster GlassBox",A.glassbox,H.glassbox,x=>x==null?"-":x.toFixed(0),"hi")}
    ${cmpRow("off / def GB",`${A.gb_off??"-"} / ${A.gb_def??"-"}`,`${H.gb_off??"-"} / ${H.gb_def??"-"}`,x=>x)}
    ${cmpRow("pass off EPA",A.off_pass,H.off_pass,s1,"hi")}
    ${cmpRow("run off EPA",A.off_run,H.off_run,s1,"hi")}
    ${cmpRow("pass def EPA",A.def_pass,H.def_pass,s1,"hi")}
    ${cmpRow("run def EPA",A.def_run,H.def_run,s1,"hi")}
    ${cmpRow("rest days",g.arest,g.hrest,x=>x==null?"-":x,"hi")}
  </table>`:`<div class="sub">team data unavailable</div>`;
  const val=g.value;
  const valbar=(val&&val.available)
    ? `<div class="valbar" style="margin-bottom:10px"><span class="vt">EDGE</span> ${val.team} <b>+${Math.round(val.ev_cur*100)}% EV</b>
        <span class="vodds">@ ${val.cur_dec}</span><span class="vlive">● live</span></div>` : "";
  let result="";
  if(done){const winner=g.hs>g.as?home:(g.hs<g.as?away:null);
    result=`<div class="sub" style="margin:6px 0"><b>FINAL ${g.as}-${g.hs}</b> ${winner==null?"&middot; tie":winner===pick?'&middot; pick <span class="num pos">HIT &#10003;</span>':'&middot; pick <span class="num neg">MISS &#10007;</span>'}</div>`;}
  $("#view").innerHTML=`<a class="back" href="#/">&lsaquo; Board</a>
    <div class="eyebrow">Week ${g.w} &middot; ${g.d} &middot; ${g.t?g.t.slice(0,5):""} ET${g.neutral?" &middot; neutral site":""}</div>
    <h1 class="pt">${A?A.name:away} <span class="sub" style="font-weight:400">at</span> ${H?H.name:home}</h1>
    <div class="grid" style="margin-top:12px">
      <div class="panel"><h3>The pick</h3>
        ${valbar}${result}
        <div style="margin:8px 0 2px"><span class="pill ${conf>=0.62?"strong":""}" style="font-size:15px;padding:6px 12px">${pick} ${pctI(conf)}%</span>
          <span class="sub" style="margin-left:8px">${pr.near?"live model":"season simulation"}</span></div>
        <div class="side ${homeWin?"":"win"}" style="margin-top:10px"><span class="ab">${teamLink(away)}</span>
          <span class="who"><span class="sp">${qcell(A)}</span></span><span class="pc">${pctI(1-hp)}%</span></div>
        <div class="side ${homeWin?"win":""}"><span class="ab">${teamLink(home)}</span>
          <span class="who"><span class="sp">${qcell(H)}</span></span><span class="pc">${pctI(hp)}%</span></div>
        <div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
        <div class="sub" style="margin-top:10px">Model ${pctI(g.ph)}% home &middot; simulation ${pctI(g.pmc!=null?g.pmc:g.ph)}% home.
          Games inside a week serve the model; further out, the 20,000-run sim (team strength evolves inside every sim).</div></div>
      <div class="panel why"><h3>Why &mdash; blend contributions <span class="sub" style="font-weight:400">home prob points</span></h3>
        ${whyRows||`<div class="sub">no contribution data</div>`}
        <div class="sub" style="margin-top:10px">Each bar = one feature group's push on the home win probability vs a 50/50 game,
          from the market-blind blend's own coefficients. Pass/run/EPA are grouped (their split is collinear; the sum is what the model believes).</div></div>
    </div>
    <div class="grid" style="margin-top:16px">
      <div class="panel"><h3>Head to head</h3>${teamCmp}</div>
      <div class="panel"><h3>Quarterbacks <span class="sub" style="font-weight:400">live depth charts</span></h3>
        <table class="cmp">
          ${cmpRow("QB1",qcell(A),qcell(H),x=>x)}
        </table>
        <div style="display:flex;gap:24px;margin-top:10px">
          <div style="flex:1"><div class="sub">${away}</div>${qrat(A)}</div>
          <div style="flex:1;text-align:right"><div class="sub">${home}</div><div style="display:flex;justify-content:flex-end">${qrat(H)}</div></div>
        </div>
        <div class="sub" style="margin-top:10px">QB ratings are the per-play 11v11 TrueSkill (conservative, position-normalized).</div></div>
    </div>
    ${(A&&A.lineup)||(H&&H.lineup)?`<div class="grid" style="margin-top:16px">${nflLineupPanel(away,A,n)}${nflLineupPanel(home,H,n)}</div>`:""}`;
}

function nflLineupPanel(code,t,n){
  const L=t&&t.lineup; if(!L) return "";
  const lrow=e=>`<tr ${e.id&&n.players[e.id]?`onclick="location.hash='#/player/${e.id}'"`:""}>
    <td><span class="sub" style="font-size:11px;letter-spacing:.04em">${e.slot}</span></td>
    <td class="a"><span class="${e.id&&n.players[e.id]?"player-link":""}">${e.name}</span>${e.src==="usage"?' <span class="sub" title="promoted: real 2025 usage + rating beat the listed starter">&uarr; usage</span>':""}</td>
    <td>${gb(e.r)}</td>
    <td><span class="num">${Math.round((e.share||0)*100)}%</span></td></tr>`;
  const tbl=(side,title)=>L[side]&&L[side].length?`<div class="subh">${title}</div><div class="twrap"><table>
    <thead><tr><th></th><th>Player</th><th>Rating</th><th>Snap %</th></tr></thead>
    <tbody>${L[side].map(lrow).join("")}</tbody></table></div>`:"";
  return `<div class="panel"><h3>${code} projected lineup <span class="sub" style="font-weight:400">live depth charts &middot; ${L.dt||""}</span></h3>
    ${tbl("off","Offense")}${tbl("def","Defense")}
    <div class="sub" style="margin-top:8px">Depth-chart starters, cuts and injury-outs removed, promoted (&uarr;) when real usage and rating say the chart is stale. Refreshes every build.</div></div>`;
}

/* ---------- GAME PAGE ---------- */
function findGame(pk){for(const l of state.board.leagues){const g=(l.games||[]).find(x=>String(x.game_pk)===String(pk));if(g)return g;}return null;}
function pRow(label,v){const w=Math.min(Math.abs(v)*3.0,50);const neg=v<0;
  return `<div class="r"><span class="lab">${label}</span>
    <div class="tr"><div class="f ${neg?"neg":""}" style="${neg?"right:50%":"left:50%"};width:${w}%"></div></div>
    <span class="v ${v>=0?"pos":"neg"}">${sgn(v)}</span></div>`;}
function statline(p){if(!p||!p.pit||p.pit.era==null)return`<span class="sub">no season line</span>`;
  const s=p.pit;
  return `<div class="statline"><span>ERA <b>${s.era}</b></span><span>WHIP <b>${s.whip}</b></span>
    <span>K/9 <b>${s.k9}</b></span><span>BB/9 <b>${s.bb9}</b></span><span>HR/9 <b>${s.hr9}</b></span>
    <span>${s.w}-${s.l}, ${s.gs} GS</span></div>`;}
function cmpRow(label,a,h,fmt,better){const f=fmt||(x=>x);
  let aw="",hw=""; if(better!=null){if(better==="hi"){a>h?aw="win":h>a?hw="win":0;}else{a<h?aw="win":h<a?hw="win":0;}}
  return `<tr><td class="a ${aw}"><span class="num">${f(a)}</span></td><td class="lbl">${label}</td><td class="h ${hw}"><span class="num">${f(h)}</span></td></tr>`;}
function gamePage(pk){
  setNav("");
  const g=findGame(pk); if(!g){$("#view").innerHTML=`<div class="empty">Game not found. <a href="#/">Back to board</a></div>`;return;}
  const db=state.db, hp=g.home_win_prob, homeWin=hp>=0.5, e=g.edge||{};
  const A=db.teams[g.away], H=db.teams[g.home];
  const pa=findPlayerByName(g.away_sp), ph=findPlayerByName(g.home_sp);
  const conf=Math.max(hp,1-hp);
  // signals
  const sig=[];
  if(H&&Math.abs(H.luck)>=0.04) sig.push([H.luck>0?"warn":"good",H.abbr+(H.luck>0?" is overperforming its run differential ("+sgn(H.luck*100)+" pts) — regression risk":" is underperforming its run differential ("+sgn(H.luck*100)+" pts) — due to bounce back")]);
  if(A&&Math.abs(A.luck)>=0.04) sig.push([A.luck>0?"warn":"good",A.abbr+(A.luck>0?" is overperforming ("+sgn(A.luck*100)+" pts) — regression risk":" is underperforming ("+sgn(A.luck*100)+" pts)")]);
  if(H&&A&&H.elo_rank<A.elo_rank&&H.record_rank>A.record_rank) sig.push(["good","We rate "+H.abbr+" higher than its record (our #"+H.elo_rank+" vs #"+H.record_rank+" by W-L)"]);
  if(pa&&ph&&pa.ts100!=null&&ph.ts100!=null){const d=ph.ts100-pa.ts100;
    if(Math.abs(d)>=6) sig.push([d>=0?"good":"warn","Pitching edge: "+(d>=0?H.abbr:A.abbr)+" ("+(d>=0?g.home_sp:g.away_sp)+", GlassBox "+(d>=0?ph.ts100:pa.ts100)+" vs "+(d>=0?pa.ts100:ph.ts100)+")"]);}
  const teamCmp = (A&&H)?`<table class="cmp">
    ${cmpRow("record",A.w+"-"+A.l,H.w+"-"+H.l,x=>x)}
    ${cmpRow("win%",A.pct,H.pct,x=>x.toFixed(3).slice(1),"hi")}
    ${cmpRow("run diff",A.run_diff,H.run_diff,x=>(x>=0?"+":"")+x,"hi")}
    ${cmpRow("runs/gm",A.rs_g,H.rs_g,x=>x,"hi")}
    ${cmpRow("allowed/gm",A.ra_g,H.ra_g,x=>x,"lo")}
    ${cmpRow("last 10",A.l10,H.l10,x=>x)}
    ${cmpRow("home / away",A.away,H.home,x=>x)}
    ${cmpRow("streak",A.streak,H.streak,x=>x)}
    ${cmpRow("our Elo",A.elo,H.elo,x=>Math.round(x),"hi")}
    ${cmpRow("Elo rank","#"+A.elo_rank,"#"+H.elo_rank,x=>x)}
  </table>`:`<div class="sub">team data unavailable</div>`;
  $("#view").innerHTML=`
    <a class="back" href="#/">&lsaquo; Board</a>
    <div class="gh"><span class="mt">${teamLink(g.away,g.away_abbr)} <span style="color:var(--faint)">@</span> ${teamLink(g.home,g.home_abbr)}</span>
      <span class="meta">${fmtDay(g.date,state.board.generated)} &middot; ${fmtTime(g.start_utc)}</span></div>
    <div class="livepanel" id="livepanel" data-pk="${g.game_pk}" data-home="${g.home_abbr}" data-away="${g.away_abbr}" data-pick="${g.pick}"></div>
    <div class="cols">
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="panel"><h3>Our projection</h3>
          <div class="proj"><div class="big">${pctI(hp)}<span class="u">%</span></div>
            <div class="pk">${g.home_abbr} win probability<b>Pick: ${g.pick} <span class="cf">${pctI(conf)}%</span></b>
            ${g.pitcher_known?"":'<span class="sub">starters not set — team-only</span>'}</div></div>
          <div class="bigbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
          <div class="barlab"><span>${g.away_abbr} ${pctI(1-hp)}%</span><span>${pctI(hp)}% ${g.home_abbr}</span></div>
          <div class="why">
            ${pRow("Team rating",e.team||0)}${pRow("Home field",e.home_field||0)}
            ${pRow(g.home_abbr+" starter",e.home_pitcher||0)}${pRow(g.away_abbr+" starter",e.away_pitcher||0)}
            ${e.bullpen!=null?pRow("Bullpen quality",e.bullpen||0):""}
            ${e.siera!=null?pRow("Starter SIERA (K/BB/GB)",e.siera||0):""}
            ${g.tier==="lineup"?pRow("Lineup on-base (TrueSkill)",e.lineup||0):""}
            ${g.tier==="lineup"&&e.power!=null?pRow("Lineup power (ISO)",e.power||0):""}
            ${g.tier==="lineup"&&e.baserun!=null?pRow("Lineup baserunning",e.baserun||0):""}
          </div>
          <div class="sub" style="margin-top:8px;color:var(--faint)">
            ${g.lineup_source==="official"
              ? "Official lineups confirmed &mdash; full model adds the actual lineup's on-base (TrueSkill), power (ISO) and baserunning."
              : g.lineup_source==="projected"
              ? ("Projected lineup &mdash; the injury-aware likely nine (most-frequent recent starters still on the active roster, IL'd players excluded)"
                 +(g.sp_projected?", plus a near-term rotation-projected starter (today/tomorrow only, where rest is reliable)":"")
                 +". The full model runs on it and upgrades automatically when the official lineup posts.")
              : "Pre-lineup estimate; sharpens when the starters and lineups are posted."}
            ${g.home_bp_fip!=null?"Bullpen = season-to-date reliever FIP ("+g.away_abbr+" "+g.away_bp_fip+" vs "+g.home_abbr+" "+g.home_bp_fip+", lower is better). ":""}
            ${g.tier==="lineup"&&e.power!=null?"Power = lineup isolated-power (extra bases per PA) vs league; on-base and power are separated because a walk and a homer are the same to the on-base rating. ":""}
            ${g.tier==="lineup"&&e.baserun!=null?"Baserunning = the lineup's stolen bases, extra bases taken, outs on the bases and double plays, run-valued vs an average runner &mdash; the third leg of offense after on-base and power. ":""}
            Contributions in probability points around a 50% base. Market-blind.</div>
        </div>
        <div class="panel"><h3>Starting pitchers</h3>
          <div class="mup">
            <div class="col a"><div class="nm">${pa?playerLinkByName(g.away_sp):g.away_sp}</div>
              <div class="sub">${teamLink(g.away,g.away_abbr)}${pa&&pa.ts100!=null?" &middot; GlassBox "+pa.ts100:""}</div>
              ${statline(pa)}</div>
            <div class="vs">vs</div>
            <div class="col h"><div class="nm">${ph?playerLinkByName(g.home_sp):g.home_sp}</div>
              <div class="sub">${teamLink(g.home,g.home_abbr)}${ph&&ph.ts100!=null?" &middot; GlassBox "+ph.ts100:""}</div>
              ${statline(ph)}</div>
          </div>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="panel"><h3>${g.away_abbr} vs ${g.home_abbr}</h3>${teamCmp}</div>
        <div class="panel"><h3>Signals</h3><div class="signals">
          ${sig.length?sig.map(([c,t])=>`<div class="sig ${c}"><span class="ic">!</span><span>${t}</span></div>`).join(""):'<div class="sub">No standout signals.</div>'}
        </div></div>
      </div>
    </div>`;
  applyLive();
}

/* ---------- STANDINGS ---------- */
function standings(){
  if(state.league==="nfl"&&state.nfl) return nflStandings();
  const db=state.db, T=Object.values(db.teams);
  const divs={}; T.forEach(t=>{(divs[t.div]=divs[t.div]||[]).push(t);});
  const order=["AL East","AL Central","AL West","NL East","NL Central","NL West"];
  let html=`<div class="eyebrow">Database</div><h1 class="pt">Standings</h1>
    <div class="sub" style="margin-bottom:14px"><b>GlassBox</b> is the roster's 0-100 TrueSkill rating (50 = league average). <b>Luck</b> = win% minus pythagorean (large + = regression risk). <b>Elo</b> is our market-blind rating. Click any team.</div>`;
  html+=order.filter(d=>divs[d]).map(d=>{
    const rows=divs[d].sort((a,b)=>a.div_rank-b.div_rank).map(t=>`
      <tr onclick="location.hash='#/team/${t.code}'">
        <td class="a"><span class="ab" style="color:var(--accent)">${t.abbr}</span></td>
        <td><span class="num">${t.w}-${t.l}</span></td><td><span class="num">${t.pct.toFixed(3).slice(1)}</span></td>
        <td><span class="num">${t.gb}</span></td>
        <td><span class="num ${t.run_diff>=0?"pos":"neg"}">${t.run_diff>=0?"+":""}${t.run_diff}</span></td>
        <td><span class="num">${t.l10}</span></td><td><span class="num">${t.streak}</span></td>
        <td><span class="num">${t.home}</span></td><td><span class="num">${t.away}</span></td>
        <td><span class="num ${Math.abs(t.luck)>=0.04?(t.luck>0?"warnc":"pos"):""}">${sgn(t.luck*100)}</span></td>
        <td><span class="num">${Math.round(t.elo)}</span></td>
        <td>${gb(t.ts100)}</td>
      </tr>`).join("");
    return `<div class="subh">${d}</div><div class="twrap"><table>
      <thead><tr><th>Team</th><th>W-L</th><th>Pct</th><th>GB</th><th>Diff</th><th>L10</th><th>Strk</th>
      <th>Home</th><th>Away</th><th>Luck</th><th>Elo</th><th>GlassBox</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }).join("");
  $("#view").innerHTML=html;
}

/* ---------- TEAMS DIRECTORY ---------- */
function teamsPage(){
  if(state.league==="nfl"&&state.nfl) return nflTeams();
  const db=state.db;
  const T=Object.values(db.teams).sort((a,b)=>(b.ts100||0)-(a.ts100||0));
  const card=t=>`<div class="tcard" onclick="location.hash='#/team/${t.code}'">
    <div class="h"><span class="code">${t.abbr}</span><span class="nm">${t.name}</span>${gb(t.ts100)}</div>
    <div class="stat"><span>Rec <b>${t.w}-${t.l}</b></span><span>Diff <b>${t.run_diff>=0?"+":""}${t.run_diff}</b></span>
      <span>L10 <b>${t.l10}</b></span><span>Elo <b>${Math.round(t.elo)}</b></span></div>
    <div class="stat"><span>Off <b>${t.ts_off??"-"}</b></span><span>Pitch <b>${t.ts_pit??"-"}</b></span>
      <span>${t.div} #${t.div_rank}</span></div>
  </div>`;
  $("#view").innerHTML=`<div class="eyebrow">Database</div><h1 class="pt">Teams</h1>
    <div class="sub" style="margin-bottom:14px">The <b>GlassBox rating</b> (0-100) is the roster's average per-plate-appearance TrueSkill rating; 50 is a league-average player. Click a team.</div>
    <div class="tgrid">${T.map(card).join("")}</div>`;
}

/* ---------- TEAM PAGE ---------- */
function teamPage(code){
  if(state.league==="nfl"&&state.nfl) return nflTeamPage(code);
  const db=state.db, t=db.teams[code];
  if(!t){$("#view").innerHTML=`<div class="empty">Team not found. <a href="#/teams">All teams</a></div>`;return;}
  const roster=(t.roster||[]).map(id=>db.players[String(id)]).filter(Boolean);
  const bat=roster.filter(p=>p.role==="batter");
  const pit=roster.filter(p=>p.role==="pitcher");
  const cell=(v,cls)=>`<td><span class="num ${cls||""}">${v==null?"-":v}</span></td>`;
  const prow=p=>`<tr onclick="location.hash='#/player/${p.id}'">
    <td class="a"><span class="num" style="color:var(--faint)">${p.pos}</span>
      <span style="margin-left:8px;font-weight:600">${p.name}</span></td>
    ${p.role==="batter"
      ? cell(p.bat&&p.bat.avg,"")+cell(p.bat&&p.bat.hr)+cell(p.bat&&p.bat.rbi)+cell(p.bat&&p.bat.ops)
      : cell(p.pit&&p.pit.era)+cell(p.pit&&p.pit.so)+cell(p.pit&&p.pit.whip)+cell(p.pit&&(p.pit.gs>0?p.pit.gs+" GS":(p.pit.sv||0)+" SV"))}
    <td>${gb(p.ts100)}</td></tr>`;
  const tbl=(title,rows,cols)=>`<div class="subh">${title}</div><div class="twrap"><table>
    <thead><tr><th>Player</th>${cols.map(c=>`<th>${c}</th>`).join("")}<th>GlassBox</th></tr></thead>
    <tbody>${rows.map(prow).join("")}</tbody></table></div>`;
  $("#view").innerHTML=`<a class="back" href="#/teams">&lsaquo; Teams</a>
    <div class="thead"><span class="code">${t.abbr}</span>
      <div><div style="font-size:15px;font-weight:600">${t.name}</div>
        <div class="sub">${t.div} &middot; #${t.div_rank} &middot; ${t.w}-${t.l} (${t.pct.toFixed(3).slice(1)})</div></div>
      <div style="margin-left:auto;text-align:right"><div class="sub">GlassBox rating</div>${gb(t.ts100)}
        <div class="sub" style="margin-top:4px">off ${t.ts_off??"-"} &middot; pitch ${t.ts_pit??"-"} &middot; #${t.ts_rank||"-"} of 30</div></div>
    </div>
    <div class="tstats">
      <div class="b"><div class="k">Run diff</div><div class="v ${t.run_diff>=0?"pos":"neg"}">${t.run_diff>=0?"+":""}${t.run_diff}</div></div>
      <div class="b"><div class="k">Runs/gm</div><div class="v">${t.rs_g}</div></div>
      <div class="b"><div class="k">Allowed/gm</div><div class="v">${t.ra_g}</div></div>
      <div class="b"><div class="k">Last 10</div><div class="v">${t.l10}</div></div>
      <div class="b"><div class="k">Streak</div><div class="v">${t.streak}</div></div>
      <div class="b"><div class="k">Home</div><div class="v">${t.home}</div></div>
      <div class="b"><div class="k">Away</div><div class="v">${t.away}</div></div>
      <div class="b"><div class="k">Our Elo</div><div class="v">${Math.round(t.elo)} <span class="sub">#${t.elo_rank}</span></div></div>
      <div class="b"><div class="k">Pythag luck</div><div class="v ${Math.abs(t.luck)>=0.04?"warnc":""}">${sgn(t.luck*100)}</div></div>
    </div>
    ${tbl("Lineup / hitters",bat.sort((a,b)=>(b.ts100||0)-(a.ts100||0)),["AVG","HR","RBI","OPS"])}
    ${tbl("Pitching",pit.sort((a,b)=>(b.ts100||0)-(a.ts100||0)),["ERA","SO","WHIP","Role"])}`;
}

/* ---------- PLAYER PAGE ---------- */
function playerPage(id){
  if(state.league==="nfl"&&state.nfl&&state.nfl.players[id]) return nflPlayerPage(id);
  const db=state.db, p=db.players[String(id)];
  if(!p){$("#view").innerHTML=`<div class="empty">Player not found. <a href="#/teams">Teams</a></div>`;return;}
  const t=db.teams[p.team];
  const box=(k,v)=>`<div class="b"><div class="k">${k}</div><div class="v">${v==null?"-":v}</div></div>`;
  let season="", career="";
  if(p.role==="batter"){
    const b=p.bat||{}, c=p.career||{};
    season=[["PA",b.pa],["AVG",b.avg],["OBP",b.obp],["SLG",b.slg],["OPS",b.ops],["HR",b.hr],["RBI",b.rbi],["SB",b.sb]].map(([k,v])=>box(k,v)).join("");
    career=[["G",c.g],["H",c.h],["HR",c.hr],["RBI",c.rbi],["AVG",c.avg],["OPS",c.ops],["SB",c.sb]].map(([k,v])=>box(k,v)).join("");
  }else{
    const b=p.pit||{}, c=p.career||{};
    season=[["GS",b.gs],["IP",b.ip],["ERA",b.era],["WHIP",b.whip],["SO",b.so],["K/9",b.k9],["BB/9",b.bb9],["SV",b.sv]].map(([k,v])=>box(k,v)).join("");
    career=[["W-L",(c.w!=null?c.w+"-"+c.l:null)],["ERA",c.era],["IP",c.ip],["SO",c.so],["WHIP",c.whip],["GS",c.gs],["SV",c.sv]].map(([k,v])=>box(k,v)).join("");
  }
  $("#view").innerHTML=`<a class="back" href="#/team/${p.team}">&lsaquo; ${t?t.name:p.team}</a>
    <div class="phead"><span class="nm">${p.name}</span>
      <span class="sub">${teamLink(p.team,p.team)} &middot; ${p.pos}${p.num?" &middot; #"+p.num:""}</span>
      <span style="margin-left:auto">${gb(p.ts100)}</span></div>
    <div class="sub" style="margin-bottom:16px">GlassBox rating (0-100) &mdash; this player's per-plate-appearance TrueSkill rating vs league-average ${p.role}s. Market-blind.</div>
    <div class="pgrid">
      <div class="panel"><h3>2026 season</h3><div class="statgrid">${season}</div></div>
      <div class="panel"><h3>Career</h3><div class="statgrid">${career}</div></div>
    </div>`;
}

boot();
"""

SHELL = f"""<style>{CSS}</style>
<script>try{{var t=localStorage.getItem('theme');document.documentElement.setAttribute('data-theme',t==='light'?'light':'dark');}}catch(e){{document.documentElement.setAttribute('data-theme','dark');}}</script>
<header><div class="wrap">
  <a class="brand" href="#/">GLASS<span class="b">BOX</span></a>
  <nav class="main">
    <a href="#/" data-v="board">Board</a>
    <a href="#/season" data-v="season" id="navseason" style="display:none">Season</a>
    <a href="#/standings" data-v="standings">Standings</a>
    <a href="#/teams" data-v="teams">Teams</a>
    <a href="#/players" data-v="players">Players</a>
  </nav>
  <span class="grow"></span>
  <span class="updated" id="updated"></span>
  <span class="acc" id="acc"></span>
  <button class="tog" id="tog" aria-label="Toggle theme">&#9681;</button>
</div></header>
<main><div class="wrap" id="view"><div class="loading">Loading predictions&hellip;</div></div></main>
<footer><div class="wrap">
  <b>Research only &mdash; not betting advice.</b> A market-blind model (team Elo + xFIP &amp;
  SIERA starting-pitcher ratings + season-to-date bullpen FIP + per-plate-appearance TrueSkill
  on-base ratings + lineup isolated-power + lineup baserunning) that never sees the odds. Data: <b>Retrosheet</b> (free of charge,
  copyrighted by Retrosheet, <a href="https://www.retrosheet.org">retrosheet.org</a>) and the
  MLB Stats API (individual, non-commercial use); NFL data from <b>nflverse</b> (community-maintained).
</div></footer>
<script>{JS}</script>"""


def build():
    body = SHELL.encode("ascii", "xmlcharrefreplace").decode("ascii")
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(body, encoding="utf-8")
    print(f"wrote {SITE/'index.html'} ({len(body):,} bytes)")


if __name__ == "__main__":
    build()
