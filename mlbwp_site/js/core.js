/* core: state, polling, boot, router, shared helpers, MLB board. */

const $ = s => document.querySelector(s);
const state = {board:null, db:null, nfl:null, nhl:null,
  league:(localStorage.getItem("league")||"mlb"), range:"today", nflRange:"year", nhlRange:"year",
  posKey:null, posSort:"r", posMin:0, nhlSort:"net", live:{}, updated:null};
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
const amOdds=p=>{p=Math.min(Math.max(p,0.01),0.99);
  return p>=0.5?"−"+Math.round(100*p/(1-p)):"+"+Math.round(100*(1-p)/p);};
const sgn = v => (v>=0?"+":"")+(+v).toFixed(1);
const tier = p => p>=0.62?"strong":(p>=0.555?"lean":"toss");
/* documents/pick_policy.md, applied on the BOARD as well as on #/record.
   infoTier mirrors mlbwp/pred_ledger.info_tier exactly, off the same card
   fields, so the board and the ledger can never disagree about what a
   forecast is. Rule 1: EARLY is shown as a scheduled game with a labelled
   lean, never as a pick. Rule 2: outside EARLY, a forecast inside 55/45 is a
   LEAN, not a PICK - the call is stated on the card, not implied by size. */
{LEAN_JS}
{TIER_JS}
{CT_JS}
const callOf = pp => pp<=POL_LEAN_MAX?"LEAN":"PICK";
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

const ESPN_NHL="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard";
async function pollNhl(){
  if(document.hidden||state.league!=="nhl"||!state.nhl||!state.nhl.schedule) return;
  const today=nflToday();
  if(!state.nhl.schedule.some(g=>g.d===today&&g.hs==null)) return;   // no live NHL slate today
  try{
    const r=await fetch(ESPN_NHL); const data=await r.json();
    const MAP={LA:"LAK",NJ:"NJD",SJ:"SJS",TB:"TBL",UTAH:"UTA"};    // ESPN -> NHL-API codes
    (data.events||[]).forEach(ev=>{
      const c=ev.competitions&&ev.competitions[0]; if(!c) return;
      const H=(c.competitors||[]).find(x=>x.homeAway==="home");
      const A=(c.competitors||[]).find(x=>x.homeAway==="away");
      if(!H||!A) return;
      const h=MAP[H.team.abbreviation]||H.team.abbreviation;
      const a=MAP[A.team.abbreviation]||A.team.abbreviation;
      const el=document.querySelector(`[data-ng="nhl_${a}_${h}"]`); if(!el) return;
      const st=ev.status||{}, tp=st.type||{};
      if(tp.state==="in")
        el.innerHTML=`<span class="dot"></span>${A.score}-${H.score} &middot; P${st.period||""} ${st.displayClock||""}`;
      else if(tp.state==="post") el.textContent=`FINAL ${A.score}-${H.score}`;
    });
  }catch(e){}
}

function updAcc(){
  const el=$("#acc"); if(!el) return;
  if(state.league==="nfl"&&state.nfl){
    const mc=state.nfl.model_card;
    el.innerHTML=`<b>${mc.test_log_loss.toFixed(3)}</b> log loss (NFL)<br>closing line ${mc.close_log_loss.toFixed(3)}`;
  }else if(state.league==="nhl"&&state.nhl){
    const mc=state.nhl.model_card;
    el.innerHTML=`<b>${mc.test_ll.toFixed(3)}</b> log loss (NHL)<br>base Elo ${mc.baseline_elo_test.toFixed(3)}`;
  }else if(state.board&&state.board.accuracy){
    /* Policy rule 3: realized accuracy is NEVER shown as one pooled headline.
       board.accuracy.realized_2026 mixes CONFIRMED, PROJECTED and EARLY rows
       and would sit in the persistent header on every page, so it is not
       rendered here - #/record reports it per tier. The header keeps the
       locked holdout number, which is a single, well-defined thing. */
    const a=state.board.accuracy;
    el.innerHTML=`<b>${a.log_loss.toFixed(3)}</b> log loss<br>coin flip ${a.coinflip.toFixed(3)}`;
  }
}

async function boot(){
  try{
    const bust = "?t=" + Math.floor(Date.now()/60000);   // fresh each minute; beats stale caches
    const [b,db,nfl,nhl] = await Promise.all([
      fetch("./data/board.json"+bust, {cache:"no-cache"}).then(r=>r.json()),
      fetch("./data/db.json"+bust, {cache:"no-cache"}).then(r=>r.json()),
      fetch("./data/nfl.json"+bust, {cache:"no-cache"}).then(r=>r.ok?r.json():null).catch(()=>null),
      fetch("./data/nhl.json"+bust, {cache:"no-cache"}).then(r=>r.ok?r.json():null).catch(()=>null),
    ]);
    state.board=b; state.db=db; state.nfl=nfl; state.nhl=nhl;
    updAcc();
    route();
    pollLive(true); pollNfl(); pollNhl();   // first poll always runs, even if the tab loads hidden
    setInterval(()=>{pollLive();pollNfl();pollNhl();}, 25000);   // all APIs cache ~20s
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
  if(v==="game"&&arg){
    if(arg.slice(0,4)==="nhl-"&&state.nhl) return nhlGamePage(arg.slice(4));
    return /_/.test(arg)&&state.nfl?nflGamePage(arg):gamePage(arg);}
  if(v==="record"){setNav("record");return recordPage();}
  if(v==="team"&&arg){setNav("teams");return teamPage(arg);}
  if(v==="player"&&arg){setNav("teams");return playerPage(arg);}
  if(v==="pos"&&arg){setNav("players");return posPage(arg);}
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
  const it=infoTier(g), early=it==="EARLY", call=callOf(g.pick_prob);
  const spTag=(nm,proj)=>nm==="TBD"?`<span class="tbd">TBD</span>`
    :`${nm} ${pitEra(nm)}${proj?' <span class="pj">proj</span>':''}`;
  const spA=spTag(g.away_sp,g.away_sp_proj), spH=spTag(g.home_sp,g.home_sp_proj);
  const val=g.value;
  const valbar = (val&&val.available)
    ? `<div class="valbar"><span class="vt">EDGE</span> ${val.team} <b>+${Math.round(val.ev_cur*100)}% EV</b>
        <span class="vodds">@ ${val.cur_dec}</span><span class="vlive">● live</span></div>` : "";
  /* Rule 1: an EARLY card is a SCHEDULED game. No pick pill and no fair price
     - a no-vig price on a team-ratings-only number invites reading it as
     tradeable. The lean is still shown, labelled as pre-information. */
  const pill=early
    ? `<span class="pill early" title="no starter announced - not published as a pick">SCHEDULED</span>`
    : `<span class="pill ${call==="PICK"&&tier(g.pick_prob)==="strong"?"strong":""}${call==="LEAN"?" leanp":""}"
        >${call} &middot; ${g.pick} ${pctI(g.pick_prob)}%</span>`;
  const odds=p=>early?"":`<span class="odds" title="fair American odds (no vig) from the model">${amOdds(p)}</span>`;
  return `<a class="gc${val&&val.available?' hasval':''}${early?' early':''}" href="#/game/${g.game_pk}" data-card="${g.game_pk}"
      data-pick="${g.pick}" data-home="${g.home_abbr}" data-away="${g.away_abbr}">
    ${valbar}
    <div class="top"><span class="lv" data-pk="${g.game_pk}" data-def="${fmtTime(g.start_utc)}">${fmtTime(g.start_utc)}</span>
      ${g.lineup_source==="official"?'<span class="lbadge off">Lineups in</span>'
        :g.sp_projected?'<span class="lbadge proj">Projected</span>'
        :g.lineup_source==="projected"?'<span class="lbadge proj">Proj lineup</span>'
        :(g.pitcher_known?"":'<span class="lbadge tbd">Starters TBD</span>')}
      ${pill}</div>
    <div class="side ${homeWin?"":"win"}"><span class="ab">${g.away_abbr}</span>
      <span class="who"><span class="sp">${spA}</span></span>${odds(1-hp)}<span class="pc">${pctI(1-hp)}%</span></div>
    <div class="side ${homeWin?"win":""}"><span class="ab">${g.home_abbr}</span>
      <span class="who"><span class="sp">${spH}</span></span>${odds(hp)}<span class="pc">${pctI(hp)}%</span></div>
    <div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
    ${early?`<div class="earlyn">Not a pick &mdash; no starter announced, team ratings only.
      Current lean <b>${g.pick} ${pctI(g.pick_prob)}%</b>; it becomes a pick when the starters are named.</div>`:""}
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
  if(state.league==="nhl"&&state.nhl) return nhlPage();
  const b=state.board, lg=b.leagues.find(l=>l.code===state.league), gen=b.generated;
  const R={today:[gen,gen],tomorrow:[addDays(gen,1),addDays(gen,1)],week:[gen,addDays(gen,6)],month:[gen,addDays(gen,60)]}[state.range];
  const games=(lg.games||[]).filter(g=>g.date>=R[0]&&g.date<=R[1]);
  // Rule 1 accounting, on the board itself: the league badge counts PICKS
  // (CONFIRMED + PROJECTED), not every scheduled game on the 30-day horizon.
  const nPick=(lg.games||[]).filter(g=>infoTier(g)!=="EARLY").length;
  const rail=b.leagues.map(l=>{
    if(l.active)
      return `<button class="lg ${l.code===state.league?"on":""}" data-lg="${l.code}" title="${nPick} picks &middot; ${l.n_games} games on the board"><span>${l.name}</span><span class="n">${nPick}</span></button>`;
    if(l.code==="nfl"&&state.nfl)
      return `<button class="lg ${state.league==="nfl"?"on":""}" data-lg="nfl"><span>NFL</span><span class="n">2026</span></button>`;
    return `<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`;}).join("")+(state.nhl?nhlRailBtn(state.league==="nhl"):"");
  const filts=[["today","Today"],["tomorrow","Tomorrow"],["week","Week"],["month","Month"]]
    .map(([k,t])=>`<button class="filt ${k===state.range?"on":""}" data-r="${k}">${t}</button>`).join("");
  // Live value bets, pinned ABOVE the date filter: the range defaults to today,
  // and a badge that fires on tomorrow's slate must never be invisible for it.
  const vEdges=(lg.games||[]).filter(g=>g.value&&g.value.available)
    .sort((a,b)=>(b.value.ev_cur||0)-(a.value.ev_cur||0));
  const vstrip=vEdges.length?`<div class="valstrip"><span class="vh">VALUE &middot; ${vEdges.length} live</span>
    ${vEdges.map(g=>`<a class="vs" href="#/game/${g.game_pk}">
      <b>${g.value.team}</b><span class="ev">+${Math.round(g.value.ev_cur*100)}% EV</span>
      <span class="vodds">@ ${g.value.cur_dec}</span>
      <span class="when">${g.away_abbr}@${g.home_abbr} &middot; ${fmtDay(g.date,gen)}</span></a>`).join("")}
    <span class="vnote">Model vs opening consensus, still live at the current price &mdash; a disagreement
      indicator under live CLV evaluation, not a profit promise. <a href="#/record">Units &amp; record &rsaquo;</a></span>
  </div>`:"";
  const legend=`<div class="polnote">Picks are published only once the starters are named
    (<span class="chip t-CONFIRMED">CONFIRMED</span> lineups posted,
    <span class="chip t-PROJECTED">PROJECTED</span> starters named). Games with no starter announced are
    <b>scheduled, not picks</b> &mdash; the model has team ratings only, so its lean is shown and labelled
    pre-information. Outside 55/45 a forecast is a <b>PICK</b>; inside it is a <b>LEAN</b>.
    <a href="#/record">Track record by tier &rsaquo;</a></div>`;
  let body;
  if(!games.length){body=`<div class="empty">No games in this window.</div>`;}
  else{const days=[...new Set(games.map(g=>g.date))].sort();
    body=days.map(d=>{const gs=games.filter(g=>g.date===d);
      const np=gs.filter(g=>infoTier(g)!=="EARLY").length;
      return `<div class="day"><span class="d">${fmtDay(d,gen)}</span>
        <span class="c">${gs.length} game${gs.length===1?"":"s"} &middot; ${np} pick${np===1?"":"s"}</span>
        ${np?"":'<span class="proj">scheduled · starters tbd</span>'}</div>
        <div class="grid">${gs.map(gcard).join("")}</div>`;}).join("");}
  $("#view").innerHTML=`<div class="controls"><div class="rail">${rail}</div><div class="filters">${filts}</div></div>${vstrip}${legend}${body}`;
  $("#view").querySelectorAll("[data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);board();});
  $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.range=x.dataset.r;board();});
  applyLive();
}

/* ---------- NHL ---------- */
const NHL_CITY={ANA:"Anaheim",BOS:"Boston",BUF:"Buffalo",CGY:"Calgary",CAR:"Carolina",CHI:"Chicago",COL:"Colorado",CBJ:"Columbus",DAL:"Dallas",DET:"Detroit",EDM:"Edmonton",FLA:"Florida",LAK:"Los Angeles",MIN:"Minnesota",MTL:"Montreal",NSH:"Nashville",NJD:"New Jersey",NYI:"NY Islanders",NYR:"NY Rangers",OTT:"Ottawa",PHI:"Philadelphia",PIT:"Pittsburgh",SJS:"San Jose",SEA:"Seattle",STL:"St. Louis",TBL:"Tampa Bay",TOR:"Toronto",VAN:"Vancouver",VGK:"Vegas",WSH:"Washington",WPG:"Winnipeg",UTA:"Utah",ARI:"Arizona",ATL:"Atlanta",PHX:"Phoenix"};
const nhlCity=c=>NHL_CITY[c]||c;
const nhlCS=()=>String((state.nhl&&state.nhl.cur_season)||20252026);
const nhlYear=()=>nhlCS().slice(4);                       // "2026"
const nhlSeasLbl=()=>nhlCS().slice(0,4)+"&ndash;"+nhlCS().slice(6);   // "2025–26"
