/* core: state, polling, boot, router, shared helpers, MLB board. */

const $ = s => document.querySelector(s);
/* The three leagues the site serves. Anything read from storage or a URL is
   validated against this list: a stale or hand-typed value must never pick a
   renderer that does not exist. */
const SITE_LGS=["mlb","nfl","nhl"];
const SITE_NAME={mlb:"MLB",nfl:"NFL",nhl:"NHL"};
const siteStoredLeague=()=>{let v=null;try{v=localStorage.getItem("league");}catch(e){}
  return SITE_LGS.indexOf(v)>=0?v:"mlb";};
const state = {board:null, db:null, nfl:null, nhl:null, failed:{}, loading:{}, waitingFor:null, booted:false,
  league:siteStoredLeague(), range:"today", nflRange:"year", nhlRange:"year",
  posKey:null, posSort:"r", posMin:0, nhlSort:"net", live:{}, updated:null,
  liveNfl:{}, liveNhl:{}, liveAt:{}};
function setLeague(lg){
  if(SITE_LGS.indexOf(lg)<0) lg="mlb";
  const changed=state.league!==lg;
  state.league=lg; try{localStorage.setItem("league",lg);}catch(e){}
  updAcc(); siteNavHrefs(); siteFreshness();
  if(changed) sitePollNow();   // the new league's live scores, not the next 25 s tick
}
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
// Calendar arithmetic on "YYYY-MM-DD" at noon UTC: a local-midnight Date read back
// through toISOString() returned the previous day for every viewer east of UTC.
const addDays=(iso,n)=>{const d=new Date(iso+"T12:00:00Z");d.setUTCDate(d.getUTCDate()+n);return d.toISOString().slice(0,10);};
/* Every "which day is it" on the site is the US Eastern calendar day, whatever
   the visitor's clock says: a game's tier, the board windows and the live
   look-back must not depend on where the reader is. Built from the parts, not
   from a locale's date string ("en-CA" is YYYY-MM-DD only by convention). */
const SITE_ET_FMT=new Intl.DateTimeFormat("en-US",{timeZone:TZ,year:"numeric",month:"2-digit",day:"2-digit",
  hour:"2-digit",minute:"2-digit",hourCycle:"h23"});
function siteEtParts(ms){const o={};
  SITE_ET_FMT.formatToParts(new Date(ms==null?Date.now():ms)).forEach(p=>{o[p.type]=p.value;});
  return o;}
const siteEtDate=ms=>{const o=siteEtParts(ms); return `${o.year}-${o.month}-${o.day}`;};
/* Anything from a URL, a payload or a feed that lands in innerHTML as text. */
const siteEsc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmtTime=u=>new Date(u).toLocaleTimeString(LOC,{hour:"numeric",minute:"2-digit",timeZone:TZ})+" ET";
const fmtDay=(iso,gen)=>{const d=new Date(iso+"T12:00:00Z");   // noon UTC => same ET day for any viewer
  const md=d.toLocaleDateString(LOC,{month:"short",day:"numeric",timeZone:TZ});
  if(iso===gen)return"Today · "+md; if(iso===addDays(gen,1))return"Tomorrow · "+md;
  return d.toLocaleDateString(LOC,{weekday:"long",timeZone:TZ})+" · "+md;};

/* ---------- LIVE (client-side, polls the MLB Stats API directly) ---------- */
async function pollLive(force){
  if(!force && document.hidden) return;   // skip recurring polls in a hidden tab; always run the first
  // MLB's feed only drives MLB cards, the MLB game page and the MLB header
  // stamp; polling it under an NFL/NHL page made the header claim "live" data.
  if(state.league!=="mlb") return;
  try{
    // yesterday..tomorrow on the US Eastern calendar (MLB's official game dates)
    const today=nflToday(), dd=n=>addDays(today,n);
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
  siteFreshness();
}

const nflToday=()=>siteEtDate();
/* The policy's "inside a week of kickoff": phase0/nfl_ph_freeze.TIER_WINDOW_DAYS
   x 24 h, read from the serve's model card when it ships the number. */
function siteNflWindowDays(){
  const s=state.nfl&&state.nfl.model_card&&state.nfl.model_card.serve, d=s&&+s.tier_window_days;
  return d>0?d:7;}
/* Is the game inside that window NOW? Kickoff (start_utc) minus the current
   instant - the same clock the freeze stamps tiers with, and an absolute one,
   so it reads the same for every visitor. A row without start_utc falls back to
   the US Eastern date rule. A played or started game is always inside. */
function nflNear(g){
  const st=g&&g.start_utc?Date.parse(g.start_utc):NaN;
  if(Number.isFinite(st)) return st-Date.now()<=siteNflWindowDays()*864e5;
  return !!(g&&g.d)&&g.d<=addDays(nflToday(),siteNflWindowDays());
}
/* Which number an NFL game shows, and in which information tier: inside the
   window (and for played games) the live model's ph, further out the season
   simulation's pmc. `tier` is the policy tier (siteTier): the stamp the serve
   wrote when the number was locked, so a stale forecast is not promoted to
   PROJECTED just because its date came near. A PROJECTED stamp is always
   `near` (it was served inside the window, and now is later), so a PROJECTED
   card never shows the simulation's number. */
function nflProb(g){const near=nflNear(g);
  return {near, hp:(!near&&g.pmc!=null)?g.pmc:g.ph, tier:siteTier("nfl",g)};}

/* ---------- shared: information tier for any league's row ----------
   documents/pick_policy.md. MLB: infoTier off the card fields. NHL: the
   shipped model has no lineup or goalie input, so EARLY unless the serve
   stamped otherwise. NFL: the serve's stamp (phase0/nfl_ph_freeze.py); a row
   with a lock time but no stamp is dated the same way the serve dates it (the
   game inside a week of the lock = PROJECTED); a PLAYED row with neither is
   EARLY - the conservative stamp, never pooled into the PROJECTED rate. */
const SITE_TIERS=["CONFIRMED","PROJECTED","EARLY"];
function siteTier(lg,g){
  if(!g) return "EARLY";
  if(lg==="mlb") return infoTier(g);
  if(SITE_TIERS.indexOf(g.tier)>=0) return g.tier;
  if(lg==="nhl") return "EARLY";
  const at=g.frozen_at||g.rec||null;
  // phase0/nfl_ph_freeze.tier_for: kickoff minus lock time against 7 x 24 h
  // when both clocks exist, else the US Eastern date rule
  const st=at&&g.start_utc?Date.parse(g.start_utc):NaN, sv=at?Date.parse(at):NaN, W=siteNflWindowDays();
  if(Number.isFinite(st)&&Number.isFinite(sv)) return st-sv<=W*864e5?"PROJECTED":"EARLY";
  if(at&&g.d&&Number.isFinite(sv)) return g.d<=addDays(siteEtDate(sv),W)?"PROJECTED":"EARLY";
  if(g.hs!=null&&g.as!=null) return "EARLY";
  return nflNear(g)?"PROJECTED":"EARLY";
}
/* The MLB-standard pick pill for any league (gcard's rule): EARLY is a
   SCHEDULED game with no call; otherwise PICK/LEAN on the 55/45 band, green
   only for a strong PICK. `pp` is the pick side's probability. */
function sitePill(tierName,pp,pick,extraStyle){
  const st=extraStyle?` style="${extraStyle}"`:"";
  if(tierName==="EARLY")
    return `<span class="pill early"${st} title="team ratings only - not published as a pick">SCHEDULED</span>`;
  const call=callOf(pp);
  return `<span class="pill ${call==="PICK"&&tier(pp)==="strong"?"strong":""}${call==="LEAN"?" leanp":""}"${st}>${call} &middot; ${pick} ${pctI(pp)}%</span>`;
}
/* "20:15" (US Eastern wall clock, as the NFL payload carries it) -> "8:15 PM ET". */
function siteClock(t){ if(!t) return "";
  const m=/^(\d{1,2}):(\d{2})/.exec(t); if(!m) return t;
  const h=+m[1], ap=h>=12?"PM":"AM", h12=((h+11)%12)+1;
  return `${h12}:${m[2]} ${ap} ET`;}

/* ---------- LIVE: NFL / NHL (ESPN public scoreboard, display only) ----------
   Scores are cached per league in state.liveNfl / state.liveNhl, keyed
   "AWAY_HOME|YYYY-MM-DD" (US Eastern game day), and re-applied after every
   render of #view (see boot's observer), so a filter click or a league switch
   never blanks them. ANY element carrying data-ng="AWAY_HOME" (NFL) or
   data-ng="nhl_AWAY_HOME" (NHL) is updated: a .livepanel gets the MLB game-page
   score panel, anything else gets the .lv live/final treatment. Grading never
   reads this - the record grades from the payload's official result. */
const ESPN_NFL="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard";
const ESPN_NHL="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard";
const SITE_ESPN={nfl:{url:ESPN_NFL,map:{LAR:"LA",WSH:"WAS"},per:"Q"},
                 nhl:{url:ESPN_NHL,map:{LA:"LAK",NJ:"NJD",SJ:"SJS",TB:"TBL",UTAH:"UTA"},per:"P"}};
const siteLiveOf=lg=>lg==="nfl"?state.liveNfl:state.liveNhl;
function siteEspnIngest(lg,data){
  const C=SITE_ESPN[lg], out=siteLiveOf(lg);
  (data&&data.events||[]).forEach(ev=>{
    const c=ev.competitions&&ev.competitions[0]; if(!c) return;
    const H=(c.competitors||[]).find(x=>x.homeAway==="home");
    const A=(c.competitors||[]).find(x=>x.homeAway==="away");
    if(!H||!A||!H.team||!A.team) return;
    const h=C.map[H.team.abbreviation]||H.team.abbreviation;
    const a=C.map[A.team.abbreviation]||A.team.abbreviation;
    const st=ev.status||{}, tp=st.type||{};
    const t=ev.date?Date.parse(ev.date):NaN, d=Number.isFinite(t)?siteEtDate(t):"";
    out[`${a}_${h}|${d}`]={st:tp.state, as:A.score, hs:H.score, d, a, h,
      det:tp.state==="in"?(tp.shortDetail||`${C.per}${st.period||""} ${st.displayClock||""}`)
         :tp.state==="post"?(tp.shortDetail||"Final").replace(/^FINAL/i,"Final"):""};
  });
  state.liveAt[lg]=new Date();
}
// minutes since midnight, US Eastern (h23: midnight is 0, never "24")
const siteNowEtMin=()=>{const o=siteEtParts(); return (+o.hour%24)*60+(+o.minute);};
/* Rows the live layer should resolve: unscored games that have started (or
   finished) inside the look-back window and are not already final in the cache. */
function siteLiveNeeded(lg){
  const n=state[lg]; if(!n||!n.schedule) return [];
  const today=nflToday(), back=addDays(today,lg==="nfl"?-8:-1), now=siteNowEtMin(), cache=siteLiveOf(lg);
  return n.schedule.filter(g=>{
    if(g.hs!=null||!g.d||g.d<back||g.d>today) return false;
    if(g.d===today){
      if(g.start_utc){if(Date.parse(g.start_utc)-15*60000>Date.now()) return false;}
      else if(g.t){const m=/^(\d{1,2}):(\d{2})/.exec(g.t); if(m&&(+m[1])*60+(+m[2])-15>now) return false;}
    }
    const c=cache[`${g.away}_${g.home}|${g.d}`];
    return !(c&&c.st==="post");
  });
}
/* One request per NFL week / NHL day that still has a started, unscored game.
   A slate with a game on today polls every tick; a past slate (a final the
   payload has not picked up) is re-asked at most every 10 minutes. */
const siteLiveTry={};
function siteLiveDue(key,live){
  if(!live&&Date.now()-(siteLiveTry[key]||0)<10*60e3) return false;
  siteLiveTry[key]=Date.now(); return true;
}
async function pollNfl(force){
  if((!force&&document.hidden)||state.league!=="nfl"||!siteOk("nfl")) return;
  const need=siteLiveNeeded("nfl"), today=nflToday();
  const weeks=[...new Set(need.map(g=>g.w))].filter(w=>w!=null)
    .filter(w=>siteLiveDue("nfl|"+w,need.some(g=>g.w===w&&g.d===today)));
  if(!weeks.length) return;                  // nothing started and unscored: no request
  for(const w of weeks){
    try{const r=await fetch(`${ESPN_NFL}?seasontype=2&week=${w}`); if(r.ok) siteEspnIngest("nfl",await r.json());}
    catch(e){/* keep the last good state */}
  }
  siteApplyLive(); siteFreshness();
}
async function pollNhl(force){
  if((!force&&document.hidden)||state.league!=="nhl"||!siteOk("nhl")) return;
  const today=nflToday();
  const days=[...new Set(siteLiveNeeded("nhl").map(g=>g.d))].filter(d=>siteLiveDue("nhl|"+d,d===today));
  if(!days.length) return;
  for(const d of days){
    try{const r=await fetch(`${ESPN_NHL}?dates=${d.replace(/-/g,"")}`); if(r.ok) siteEspnIngest("nhl",await r.json());}
    catch(e){}
  }
  siteApplyLive(); siteFreshness();
}
function sitePollNow(){
  if(state.league==="mlb") pollLive(true);
  else if(state.league==="nfl") pollNfl(true);
  else pollNhl(true);
}
/* The schedule row a live-score element belongs to, for its game day: from
   data-d when the page sets one, else from the game link it sits in (cards),
   else from the game page's own route (#livepanel). */
function siteGameRow(lg,arg){
  const n=state[lg]; if(!n||!n.schedule||!arg) return null;
  if(lg==="nhl"){const id=String(arg).replace(/^nhl-/,""); return n.schedule.find(g=>String(g.id)===id)||null;}
  const p=String(arg).split("_"); if(p.length<3) return null;
  return n.schedule.find(g=>String(g.w)===p[0]&&g.away===p[1]&&g.home===p[2])||null;
}
function siteElDate(lg,el){
  if(el.dataset.d) return el.dataset.d;
  const a=el.closest("a[href]"), m=a&&/#\/(?:(?:nfl|nhl)\/)?game\/([^/?#]+)/.exec(a.getAttribute("href")||"");
  let row=m?siteGameRow(lg,m[1]):null;
  if(!row&&el.id==="livepanel"){const P=siteParse(location.hash); if(P.v==="game") row=siteGameRow(lg,P.arg);}
  return row?row.d:null;
}
function siteApplyLive(){
  ["nfl","nhl"].forEach(lg=>{
    const cache=siteLiveOf(lg), keys=Object.keys(cache); if(!keys.length) return;
    document.querySelectorAll("[data-ng]").forEach(el=>{
      let key=el.dataset.ng||"";
      if(lg==="nhl"){if(key.slice(0,4)!=="nhl_") return; key=key.slice(4);}
      else if(key.slice(0,4)==="nhl_") return;
      const d=siteElDate(lg,el);
      let hit=d?cache[key+"|"+d]:null;
      if(!hit&&!d){const today=nflToday();
        hit=cache[key+"|"+today]||keys.filter(k=>k.split("|")[0]===key).map(k=>cache[k]).sort((x,y)=>x.d<y.d?1:-1)[0];}
      if(!hit||(hit.st!=="in"&&hit.st!=="post")) return;
      const live=hit.st==="in";
      el.title="Live score from ESPN (display only) - the record grades from the official result once the site's data has it";
      if(el.classList.contains("livepanel")){
        const A=el.dataset.away||hit.a, H=el.dataset.home||hit.h;
        el.style.display="flex"; el.classList.toggle("live",live);
        el.innerHTML=`<span class="sc">${A} ${hit.as} &ndash; ${hit.hs} ${H}</span>
          <span class="st">${live?`<span class="dot"></span>${hit.det}`:`${hit.det} <span class="sub">(unofficial)</span>`}</span>`;
      }else{
        el.classList.toggle("live",live); el.classList.toggle("final",!live);
        el.innerHTML=live?`<span class="dot"></span>${hit.as}-${hit.hs} &middot; ${hit.det}`:`${hit.det} ${hit.as}-${hit.hs}`;
      }
    });
  });
}

/* The header's accuracy line: ONE meaning in every league. The model's locked
   test-set log loss against the same market-free comparator, a plain TEAM Elo
   scored on the same held-out games. It used to read "coin flip" on MLB, the
   market's "closing line" on NFL and a "base Elo" from a different harness on
   NHL, so the same slot changed meaning with the league. The closing line stays
   on the NFL board's accuracy ladder, where it is labelled as the market.
   The comparator must be scored on the model's OWN holdout: NFL's
   ratings_model.elo_ll (0.638) is team Elo on TEST 2022-2025 only, n=1,139,
   so pairing it with the 2016-2025, n=2,761 log loss was false. Only a
   model_card.elo_log_loss (same holdout) is used; without it the line falls
   back to the labelled coin flip.
   Policy rule 3: realized accuracy is NEVER a pooled headline -
   board.accuracy.realized_2026 mixes CONFIRMED, PROJECTED and EARLY rows, so it
   is not rendered here; #/record reports it per tier. */
const siteNum=v=>typeof v==="number"&&Number.isFinite(v)?v:null;
function siteAccLine(lg){
  let ll=null, elo=null, span="", extra="", close=null;
  if(lg==="mlb"){const a=state.board&&state.board.accuracy; if(!a) return null;
    ll=siteNum(a.log_loss); elo=siteNum(a.elo_log_loss);
    span=a.holdout?String(a.holdout):"";}
  else if(lg==="nfl"){const mc=state.nfl&&state.nfl.model_card; if(!mc) return null;
    ll=siteNum(mc.test_log_loss);
    elo=siteNum(mc.elo_log_loss);      // same holdout only; never ratings_model.elo_ll (TEST 2022-2025)
    close=siteNum(mc.close_log_loss);  // the NFL board's ladder prints it, labelled as the market
    span=mc.holdout?String(mc.holdout):"";}
  else if(lg==="nhl"){const mc=state.nhl&&state.nhl.model_card; if(!mc) return null;
    ll=siteNum(mc.test_ll);
    // test_delta_vs_elo (and its CI) is measured against the bare-Elo baseline
    // of the SAME harness, so ll+delta IS that baseline. Payloads before
    // 2026-09-24 shipped a baseline_elo_test from another report (0.66918
    // against a 0.66953 delta base), which made the card's numbers not add up.
    const dl=siteNum(mc.test_delta_vs_elo);
    elo=ll!=null&&dl!=null?ll+dl:siteNum(mc.baseline_elo_test);
    span=mc.test_seasons?String(mc.test_seasons)+(mc.test_n?", n="+Number(mc.test_n).toLocaleString(LOC):""):"";
    const ci=Array.isArray(mc.test_ci)&&mc.test_ci.length===2?mc.test_ci.map(siteNum):null;
    const r4=x=>(Math.sign(x)*Math.round(Math.abs(x)*1e4+1e-9)/1e4).toFixed(4);   // 0.00535 -> "0.0054", as the card rounds it
    if(dl!=null&&ci&&ci[0]!=null&&ci[1]!=null) extra=` The feature model ${dl>=0?"beats":"trails"} bare Elo by`
      +` ${r4(Math.abs(dl))} (95% CI ${r4(ci[0])} to ${r4(ci[1])}).`;
    if(mc.baseline_def) extra+=` Elo baseline: ${String(mc.baseline_def)}.`;}
  if(ll==null) return null;
  const cmp=elo!=null?{k:"team Elo",v:elo}:{k:"coin flip",v:Math.log(2)};
  return {ll, cmp, title:`${SITE_NAME[lg]} model log loss on its locked test set${span?` (${span})`:""}, lower is better,`
    +` compared with ${elo!=null?"a plain team-Elo model scored on the same games":"a coin flip (this build has no team-Elo score on these same games)"}.`
    +`${extra}${elo!=null?` Coin flip = ${Math.log(2).toFixed(3)}.`:""} Market-free: the odds are never an input.`
    // only a board that actually prints the closing line may point to it
    +(close!=null?` The closing-line benchmark is shown on the board, labelled as the market.`:"")};
}
function updAcc(){
  const el=$("#acc"); if(!el) return;
  const a=siteAccLine(state.league);
  if(!a){el.innerHTML=""; el.removeAttribute("title"); return;}
  el.innerHTML=`<b>${a.ll.toFixed(3)}</b> ${SITE_NAME[state.league]} log loss<br>${a.cmp.k} ${a.cmp.v.toFixed(3)}`;
  el.setAttribute("title",a.title);
}

/* ---------- FRESHNESS: how old is the data on screen, per league ----------
   The header used to print MLB's live-poll clock on every page, which made an
   eight-week-old NFL payload read "live". Each league now reports its OWN age:
   MLB the live poll (and its board build date), NFL/NHL the time their payload
   was served, what results it holds, and an amber flag when it is stale or a
   played game is still waiting on its result. */
const SITE_STALE_H={mlb:36,nfl:8*24,nhl:36};   // hours before a payload counts as stale in-season
const siteEt=(ms,opt)=>new Date(ms).toLocaleString(LOC,Object.assign({timeZone:TZ},opt));
function siteAgo(ms){const m=Math.max(0,Math.round(ms/60000));
  if(m<60) return m+"m"; const h=Math.round(m/60); if(h<48) return h+"h"; return Math.round(h/24)+"d";}
function siteFresh(lg){
  const today=nflToday();
  if(lg==="mlb"){const b=state.board; if(!b) return null;
    const stale=!!(b.generated&&b.generated<addDays(today,-1));
    return {lg, at:null, built:b.generated, through:b.current_through, stale, pending:0,
      title:`MLB board built ${b.generated||"?"}; results through ${b.current_through||"?"}. Live scores from the MLB Stats API.`};}
  const n=state[lg]; if(!n) return null;
  const iso=lg==="nfl"?(n.served_at||(n.model_card&&n.model_card.serve&&n.model_card.serve.served_at)||n.generated):n.served_at;
  const at=iso?Date.parse(iso):null;
  const sch=n.schedule||[], fin=sch.filter(g=>g.hs!=null);
  const through=fin.length?fin.reduce((m,g)=>g.d>m?g.d:m,""):null;
  const pend=sch.filter(g=>g.hs==null&&g.d&&g.d<today);
  // NFL results land with the weekly chain (Sunday -> Tuesday is normal);
  // NHL is re-served every 4 h, so a day-old unscored game is already late.
  const late=pend.filter(g=>g.d<addDays(today,lg==="nfl"?-2:-1)).length;
  const inSeason=n.status==="season";
  const old=!!(at&&inSeason&&Date.now()-at>SITE_STALE_H[lg]*3600e3);
  const wk=lg==="nfl"&&fin.length?Math.max(...fin.map(g=>g.w||0)):null;
  const title=`${SITE_NAME[lg]} data served ${at?siteEt(at,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})+" ET":"(no build time in this payload)"}`
    +`; results through ${through?(wk?"Wk "+wk+" ("+through+")":through):"none yet"}`
    +(pend.length?`; ${pend.length} played game${pend.length===1?"":"s"} not yet in the data`:"")
    +`. Live scores from ESPN (display only).`;
  return {lg, at, through, wk, pending:pend.length, late, stale:old||late>0, old, title};
}
function siteFreshness(){
  const u=$("#updated"); if(!u) return;
  const lg=state.league, f=siteFresh(lg);
  if(!f){u.innerHTML=""; u.className="updated"; u.removeAttribute("title"); return;}
  u.className="updated"+(f.stale?" stale":"");
  u.title=f.title;
  if(lg==="mlb"){
    const t=state.updated?state.updated.toLocaleTimeString(LOC,{hour:"numeric",minute:"2-digit",second:"2-digit",timeZone:TZ}):null;
    u.innerHTML=t?`<span class="dot"></span><span class="l">live &middot; updated </span>${t} ET${f.stale?` <span class="l">&middot; board ${f.built}</span>`:""}`
      :`<span class="dot"></span><span class="l">MLB board </span>${f.built||""}`;
    return;
  }
  const liveOn=state.liveAt[lg]&&Date.now()-state.liveAt[lg]<90e3;
  const when=f.at?siteEt(f.at,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})+" ET":"build time unknown";
  const flag=f.late?`results pending`:(f.old?`${siteAgo(Date.now()-f.at)} old`:"");
  u.innerHTML=`<span class="dot"></span><span class="l">${liveOn?"live scores &middot; ":""}${SITE_NAME[lg]} data &middot; </span>`
    +`<span class="l">${when}</span><span class="s">${f.at?siteAgo(Date.now()-f.at):""}</span>`
    +(flag?` <b class="fl">${flag}</b>`:"");
}
/* Footer: the model lines that come from the payloads, and per-league data age. */
function siteFooter(){
  const n=state.nfl, el=$("#ft-nfl");
  if(el&&n&&n.model_card){const r=n.model_card.ratings||{}, sv=n.model_card.serve||{};
    el.innerHTML=` (${r.plays?r.plays.toLocaleString(LOC)+" plays":"per-snap"}${r.seasons?", "+r.seasons:""}); inside a
      week of kickoff the live model runs on depth-chart lineups, further out a ${sv.sims?sv.sims.toLocaleString(LOC)+"-run ":""}season simulation`;}
  const fr=$("#ft-fresh"); if(!fr) return;
  fr.innerHTML=SITE_LGS.map(lg=>{const f=siteFresh(lg); if(!f) return "";
    const when=lg==="mlb"?`board ${f.built||"?"}`:(f.at?siteEt(f.at,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})+" ET":"?");
    return `<b>${SITE_NAME[lg]}</b> ${when}${f.stale?" (stale)":""}`;}).filter(Boolean).join(" &middot; ");
}

/* ---------- payload health ----------
   A payload that failed to load, or landed half-built (the NFL chain writes
   its stages in place), is refused rather than drawn: a page of empty tables
   and "0 rated players" is worse than saying the data is being rebuilt. */
function siteOk(lg){
  if(lg==="mlb") return !!(state.board&&state.board.leagues);
  const n=state[lg]; if(!n||!n.teams||!Object.keys(n.teams).length) return false;
  if(n.status==="season"&&!(Array.isArray(n.schedule)&&n.schedule.length)) return false;
  return true;
}
async function siteFetch(f){
  const r=await fetch("./data/"+f+"?t="+Math.floor(Date.now()/60000),{cache:"no-cache"});
  if(!r.ok) throw new Error(f+" HTTP "+r.status);
  return r.json();
}
/* ---------- BOOT: each league draws as soon as ITS data is in ----------
   The first paint used to wait for all four payloads (about 3.3 MB): an MLB
   reader downloaded the NFL and NHL files, an NHL reader MLB's player database,
   before anything was drawn. Now the page draws the league it opens on once
   that league's files are in (plus the small board.json, which carries the
   rails' placeholders and the record's units); the other leagues fill in
   behind it. Until then their rail buttons say "loading", and opening one
   shows a loading line that turns into the page when the data lands - never
   "unavailable" for a file that is still on its way. A legacy link naming a
   team or a player with no league waits for everything: resolving it needs
   every league's ids. */
const SITE_LG_FILES={mlb:["board","db"],nfl:["nfl"],nhl:["nhl"]};
const SITE_FILES=["board","db","nfl","nhl"];
const siteFileDone={};
function siteSetFile(k,v){ if(k==="board") state.board=v; else if(k==="db") state.db=v; else state[k]=v; }
const siteLgOfFile=k=>(k==="board"||k==="db")?"mlb":k;
function siteLgSettled(lg){
  state.loading[lg]=false;
  state.failed[lg]=lg==="mlb"?!(state.board&&state.db):!state[lg];
  // In season the boards open on the coming week in date order, as MLB opens
  // on today; "Year" (every game, and NHL newest-first) is the offseason view.
  if(lg==="nfl"&&state.nfl&&state.nfl.status==="season"&&state.nflRange==="year") state.nflRange="week";
  if(lg==="nhl"&&state.nhl&&state.nhl.status==="season"&&state.nhlRange==="year") state.nhlRange="week";
}
function siteApplyFile(k,v){
  siteSetFile(k,v); siteFileDone[k]=true;
  const lg=siteLgOfFile(k);
  if(!SITE_LG_FILES[lg].every(f=>siteFileDone[f])) return;
  siteLgSettled(lg);
  if(state.booted) siteLeagueIn(lg);
}
/* A league's data landed after the first paint: refresh the chrome, draw the
   page that was waiting for it, else update its rail button in place (a full
   redraw would reset the reader's scroll and inputs). */
function siteLeagueIn(lg){
  siteLeaguesTidy(); updAcc(); siteNavHrefs(); siteFreshness(); siteFooter();
  if(state.waitingFor===lg){state.waitingFor=null; route(true);}
  else siteRailPatch(lg);
  if(lg===state.league) sitePollNow();
}
/* The league the first paint draws, from the URL alone; null = wait for all. */
function siteBootLeague(P){
  if(P.v==="player"&&P.arg) return null;              // player ids resolve across leagues
  if(P.v==="team"&&P.arg&&!P.lg) return null;          // a bare team code may be any league's
  return siteLeagueOfId(P.v,P.arg)||P.lg||state.league;
}
async function siteReload(lg){
  state.loading[lg]=true;
  const fs=SITE_LG_FILES[lg];
  const got=await Promise.allSettled(fs.map(f=>siteFetch(f+".json")));
  // a retry never replaces a good file with a failed fetch
  fs.forEach((f,i)=>{if(got[i].status==="fulfilled") siteSetFile(f,got[i].value);});
  siteLgSettled(lg);
  siteLeaguesTidy(); updAcc(); siteNavHrefs(); siteFreshness(); siteFooter(); route(true);
}
function siteUnavailable(lg,v){
  const rec=v==="record";
  const wire=()=>rec
    ?$("#view").querySelectorAll(".rail [data-lg]").forEach(x=>x.onclick=()=>{location.hash=siteHref(x.dataset.lg,"record");})
    :siteWireRail();
  const rail=`<div class="controls"><div class="rail">${siteRail(lg,rec&&typeof recBadge==="function"?recBadge:undefined)}</div></div>`;
  if(state.loading[lg]){           // still on its way: say so, and draw the page when it lands
    state.waitingFor=lg;
    $("#view").innerHTML=`${rail}<div class="loading">Loading ${SITE_NAME[lg]} data&hellip;</div>`;
    wire(); return;
  }
  const f=state.failed[lg];
  $("#view").innerHTML=`${rail}
    <div class="empty">${SITE_NAME[lg]} data ${f?"could not be loaded":"is being rebuilt &mdash; this build is incomplete"}.<br>
      <span class="sub">Nothing is drawn rather than a half-built page.</span>
      <div style="margin-top:14px"><button class="filt on" id="siteretry">Retry</button>
      ${lg!=="mlb"&&siteOk("mlb")?` &nbsp;<a class="tl" href="${siteHref("mlb",v&&SITE_LIST.indexOf(v)>=0?v:"")}">View MLB instead</a>`:""}</div></div>`;
  wire();
  const b=$("#siteretry"); if(b) b.onclick=()=>{b.disabled=true; b.textContent="Loading..."; siteReload(lg);};
}

async function boot(){
  SITE_LGS.forEach(l=>{state.loading[l]=true;});
  const first=siteBootLeague(siteParse(location.hash));
  const got={};
  SITE_FILES.forEach(k=>{got[k]=siteFetch(k+".json").then(v=>v,()=>null).then(v=>siteApplyFile(k,v));});
  const need=first?["board",...SITE_LG_FILES[first].filter(f=>f!=="board")]:SITE_FILES;
  await Promise.all(need.map(k=>got[k]));
  state.booted=true;
  siteLeaguesTidy(); updAcc(); siteNavHrefs();
  route(true);
  siteFreshness(); siteFooter();
  sitePollNow();   // first poll always runs, even if the tab loads hidden
  setInterval(()=>{pollLive();pollNfl();pollNhl();siteFreshness();}, 25000);   // all APIs cache ~20s
  document.addEventListener("visibilitychange", ()=>{ if(!document.hidden) sitePollNow(); });
  // Re-apply cached NFL/NHL scores after every render of the view.
  try{new MutationObserver(()=>siteApplyLive()).observe($("#view"),{childList:true});}catch(e){}
  await Promise.all(SITE_FILES.map(k=>got[k]));
  if(!siteOk("mlb")&&!siteOk("nfl")&&!siteOk("nhl")){
    $("#view").innerHTML=`<div class="empty">The prediction data could not be loaded.<br>
      <span class="sub">A network hiccup or a deploy in progress &mdash; nothing is wrong with your browser.</span>
      <div style="margin-top:14px"><button class="filt on" onclick="location.reload()">Retry</button></div></div>`;
  }
}
/* board.json lists every league the MLB builder knows, with NHL/NFL as
   inactive placeholders. The live NHL button is drawn from nhl.json, so the
   placeholder is dropped once that payload is in - one NHL button, not two -
   and the served leagues are listed first. */
function siteLeaguesTidy(){
  const b=state.board; if(!b||!Array.isArray(b.leagues)) return;
  const rank=c=>SITE_LGS.indexOf(c)>=0?SITE_LGS.indexOf(c):SITE_LGS.length;
  b.leagues=b.leagues.filter(l=>!(l.code==="nhl"&&!l.active&&state.nhl))
    .map((l,i)=>[l,i]).sort((x,y)=>(rank(x[0].code)-rank(y[0].code))||(x[1]-y[1])).map(x=>x[0]);
}
window.addEventListener("hashchange", ()=>route(true));
window.addEventListener("resize", ()=>siteNavFit(true));
{const nv=document.querySelector("nav.main");
  if(nv&&nv.addEventListener) nv.addEventListener("scroll",()=>siteNavFit(),{passive:true});}
/* A link inside a clickable row or card (a team link in a schedule row): the
   click ran the row's own navigation first and then the link's, leaving a
   history entry the reader never opened, so Back went to a page they had not
   seen. The link wins; the row's handler stands down for this one click. */
document.addEventListener("click",e=>{
  const t=e.target, a=t&&t.closest?t.closest("a[href]"):null; if(!a) return;
  for(let p=a.parentElement;p&&p!==document.body;p=p.parentElement){
    if(typeof p.onclick==="function"){const h=p.onclick, el=p; el.onclick=null; setTimeout(()=>{el.onclick=h;},0);}
  }
},true);

/* ---------- ROUTER ----------
   URLs carry the league: #/nfl/team/BUF, #/nhl/player/8474600, #/mlb/standings,
   #/nhl/record, #/nfl (board). Game routes were already unambiguous by id shape
   (numeric game_pk = MLB, "w_AWAY_HOME" = NFL, "nhl-<id>" = NHL) and #/season is
   NFL-only; those keep their form and set the league from the id. Unprefixed
   legacy links still work: an entity id resolves its own league where it can
   (player ids never collide; a team code resolves against the current league,
   else the one league that has it), and the address bar is then rewritten to
   the league-qualified form, so what a reader copies or bookmarks is exact. */
const SITE_LIST=["standings","teams","players","record"];
const SITE_NFL_POS=["QB","RB","WR","TE","OL","DL","LB","DB"];
let siteRouting=false;
/* Every real id is [A-Za-z0-9_.-]: team codes, gsis 00-0034869, numeric MLB/NHL
   ids, 3_ATL_GB, nhl-2026020001, QB/hitters. Anything else is replaced by "?"
   (an honest not-found page), so a crafted link can never carry markup into a
   page - the renderers print ids as text. An unknown view opens the board. */
const SITE_VIEWS=["","season","game","record","team","player","pos","players","teams","standings"];
const SITE_ID_RE=/^[A-Za-z0-9_.-]{1,40}$/;
function siteParse(hash){
  const parts=String(hash||"").replace(/^#\/?/,"").split("/").filter(x=>x!=="");
  const lg=parts.length&&SITE_LGS.indexOf(parts[0])>=0?parts.shift():null;
  let v=parts[0]||"", arg=parts[1];
  if(SITE_VIEWS.indexOf(v)<0){v=""; arg=undefined;}
  try{if(arg!=null) arg=decodeURIComponent(arg);}catch(e){arg="?";}
  if(arg!=null&&!SITE_ID_RE.test(arg)) arg="?";
  return {lg, v, arg};
}
const siteHref=(lg,v,arg)=>"#/"+lg+(v?"/"+v:"")+(arg!=null&&arg!==""?"/"+encodeURIComponent(arg):"");
/* League-aware team link. Pass `lg` when the page is not the current league's. */
const teamLink = (code,label,lg) => `<a class="tl" href="${siteHref(lg||state.league,"team",code)}">${label||code}</a>`;
function siteHasTeam(lg,code){
  if(lg==="mlb") return !!(state.db&&state.db.teams&&state.db.teams[code]);
  const n=state[lg]; return !!(n&&n.teams&&n.teams[code]);
}
function siteHasPlayer(lg,id){
  if(lg==="mlb") return !!(state.db&&state.db.players&&state.db.players[id]);
  const n=state[lg]; return !!(n&&n.players&&n.players[id]);
}
/* The league an entity id decides on its own, or null when it does not. */
function siteLeagueOfId(v,arg){
  if(!arg) return v==="season"?"nfl":null;
  if(v==="season") return "nfl";
  if(v==="game") return arg.slice(0,4)==="nhl-"?"nhl":(/_/.test(arg)?"nfl":(/^\d+$/.test(arg)?"mlb":null));
  if(v==="player"){const own=SITE_LGS.filter(l=>siteHasPlayer(l,arg));
    if(own.length===1) return own[0];
    if(/^\d\d-\d/.test(arg)) return "nfl";           // gsis ids
    return null;}
  if(v==="pos") return (arg==="hitters"||arg==="pitchers")?"mlb":(SITE_NFL_POS.indexOf(arg)>=0?"nfl":null);
  return null;
}
function siteSetHash(h,push){
  if(location.hash===h) return;
  try{history[push?"pushState":"replaceState"](history.state,"",location.pathname+location.search+h);}
  catch(e){ if(push) location.hash=h; }
}
function route(fromUrl){
  const P=siteParse(location.hash);
  let v=P.v, arg=P.arg, lg;
  window.scrollTo(0,0);
  state.waitingFor=null;          // a page still waiting on a payload is replaced by this route
  const own=siteLeagueOfId(v,arg);
  if(fromUrl===true){
    // the URL decides: a decisive id first, then an explicit prefix, then a
    // team code's own league, then the reader's current league
    if(v==="team"&&arg&&!P.lg&&!siteHasTeam(state.league,arg)){
      const c=SITE_LGS.filter(l=>siteHasTeam(l,arg)); lg=c.length===1?c[0]:state.league;
    }else lg=(v==="team"?null:own)||P.lg||state.league;
  }else{
    // called by a page after a league switch: the page's league wins, and an
    // entity of another league falls back to that league's matching list
    lg=state.league;
    const was=own||P.lg||lg;
    if(was!==lg&&["team","player","pos","game","season"].indexOf(v)>=0){
      v=v==="team"?"teams":(v==="player"||v==="pos")?"players":""; arg=undefined;}
  }
  if(SITE_LGS.indexOf(lg)<0) lg="mlb";
  if(lg!==state.league) setLeague(lg); else siteNavHrefs();
  siteRouting=true;
  try{ siteDispatch(lg,v,arg); }
  catch(e){
    try{console.error(e);}catch(_){}
    siteDrawError(lg,e);
  }
  finally{ siteRouting=false; }
  if(location.hash&&v!=="game"&&v!=="season") siteSetHash(siteHref(lg,v,arg),false);
  siteTitle(lg,v);
}
function siteDispatch(lg,v,arg){
  const nav=()=>setNav(v===""?"board":(SITE_LIST.indexOf(v)>=0||v==="season"?v:""));
  if(state.loading[lg]){nav();return siteUnavailable(lg,v);}      // on its way: wait, then draw
  const needs=lg==="mlb"?(v===""?!!state.board:!!(state.board&&state.db)):siteOk(lg);
  if(!needs&&v!=="record"){nav();return siteUnavailable(lg,v);}
  if(v==="season"){setNav("season");nflSeason(arg);return siteSeasonRail();}
  if(v==="game"&&arg){setNav("");
    if(lg==="nhl") return nhlGamePage(arg.replace(/^nhl-/,""));
    if(lg==="nfl") return nflGamePage(arg);
    return gamePage(arg);}
  if(v==="record"){setNav("record");return recordPage();}
  if(v==="team"&&arg){setNav("teams");return siteTeamRoute(lg,arg);}
  if(v==="player"&&arg){setNav("teams");
    if(lg!=="mlb"&&!siteHasPlayer(lg,arg)) return siteNotFound(lg,"player",arg);
    return playerPage(arg);}
  if(v==="pos"&&arg){setNav("players");return posPage(arg);}
  if(v==="players"){setNav("players");return playersPage();}
  if(v==="teams"){setNav("teams");return teamsPage();}
  if(v==="standings"){setNav("standings");return standings();}
  setNav("board"); board();
}
/* The error boundary's message names the real cause. A league page whose own
   rail reads board.json throws when only board.json failed to load: that is
   the MLB file's failure, not this league's data. */
function siteDrawError(lg,e){
  const boardGone=lg!=="mlb"&&state.failed.mlb&&!state.board
    &&/board|leagues/.test(String(e&&e.message||""));
  const why=boardGone
    ?"The MLB board file failed to load, so this page's league rail could not be drawn."
    :`This page could not be drawn from the current ${SITE_NAME[lg]} data.`;
  $("#view").innerHTML=`<div class="empty">${why}
    <a class="tl" href="${siteHref(lg,"")}">Back to the ${SITE_NAME[lg]} board</a></div>`;
}
function siteNotFound(lg,what,id){
  $("#view").innerHTML=`<div class="empty">No ${SITE_NAME[lg]} ${what} <b>${siteEsc(id)}</b> in this build.
    <a class="tl" href="${siteHref(lg,what==="team"?"teams":"players")}">All ${SITE_NAME[lg]} ${what==="team"?"teams":"players"}</a></div>`;
}
/* A legacy #/team/CODE whose code exists in several leagues but not the
   current one asks instead of guessing. */
function siteTeamRoute(lg,code){
  if(siteHasTeam(lg,code)) return teamPage(code);
  const c=SITE_LGS.filter(l=>l!==lg&&siteHasTeam(l,code));
  if(!c.length) return siteNotFound(lg,"team",code);
  const nm=l=>siteEsc(siteTeamName(l,code));
  const C=siteEsc(code);   // a team code is only printed once it exists in a league, but never raw
  $("#view").innerHTML=`<div class="empty">${C} is not an ${SITE_NAME[lg]} team. Did you mean
    ${c.map(l=>`<a class="tl" href="${siteHref(l,"team",code)}">${SITE_NAME[l]} ${C} &middot; ${nm(l)}</a>`).join(" or ")}?</div>`;
}
/* A team's full name, from its league's payload ("Colorado Avalanche", "Buffalo
   Bills", "Tampa Bay Rays"); the NHL city map only for a payload without names. */
function siteTeamName(lg,code){
  const T=lg==="mlb"?(state.db&&state.db.teams):(state[lg]&&state[lg].teams), t=T&&T[code];
  if(t&&t.name) return String(t.name);
  return lg==="nhl"?nhlCity(code):String(code);
}
/* The Season page (nfl.js) is a top-level NFL view like the board, so it gets
   the league rail the board has - drawn here unless the page drew its own. A
   click on another league opens that league's board; NFL is where you are. */
function siteSeasonRail(){
  const V=$("#view"); if(!V||!V.insertAdjacentHTML||V.querySelector(".rail")) return;
  V.insertAdjacentHTML("afterbegin",`<div class="controls"><div class="rail">${siteRail("nfl")}</div></div>`);
  V.querySelectorAll(".rail [data-lg]").forEach(x=>x.onclick=()=>{
    if(x.dataset.lg==="nfl"&&siteOk("nfl")) return;
    setLeague(x.dataset.lg); board();});
}
function setNav(v){document.querySelectorAll("nav.main a").forEach(a=>a.classList.toggle("on",a.dataset.v===v));
  siteNavFit(true);}
/* Nav links follow the current league so middle-click / copy-link are exact. */
function siteNavHrefs(){
  const lg=state.league;
  document.querySelectorAll("nav.main a[data-v]").forEach(a=>{const v=a.dataset.v;
    if(v!=="season") a.setAttribute("href",siteHref(lg,v==="board"?"":v));});
  const br=document.querySelector("header .brand"); if(br) br.setAttribute("href",siteHref(lg,""));
  // the Season view is NFL-only
  const ns=document.getElementById("navseason");
  if(ns) ns.style.display=(lg==="nfl"&&siteOk("nfl")&&state.nfl.schedule&&state.nfl.schedule.length)?"":"none";
  const fr=document.getElementById("ft-rec"); if(fr&&fr.setAttribute) fr.setAttribute("href",siteHref(lg,"record"));
  siteNavFit();
}
/* The nav scrolls sideways when it does not fit (phones, and 560-700px windows,
   where NFL's extra Season item pushes Players off the edge). A fade on each
   clipped edge says there is more, and the current section is scrolled into
   view, so the highlighted item is never the one hidden. */
function siteNavFit(reveal){
  const nav=document.querySelector("nav.main");
  if(!nav||typeof nav.scrollWidth!=="number"||!nav.classList||!nav.getBoundingClientRect) return;
  const over=nav.scrollWidth>nav.clientWidth+1;
  if(reveal&&over){const a=nav.querySelector("a.on");
    if(a&&a.offsetParent!==null){const r=a.getBoundingClientRect(), n=nav.getBoundingClientRect();
      if(r.left<n.left||r.right>n.right) nav.scrollLeft+=(r.left-n.left)-(n.width-r.width)/2;}}
  nav.classList.toggle("ovf",over&&nav.scrollLeft+nav.clientWidth<nav.scrollWidth-2);
  nav.classList.toggle("ovl",over&&nav.scrollLeft>2);
}
function siteTitle(lg,v){
  const L=SITE_NAME[lg]||"";
  const V={"":"Board",record:"Track record",standings:"Standings",teams:"Teams",players:"Players",season:"Season"}[v];
  let t;
  if(V!=null) t=`${L} ${V}`;
  else{
    // an entity page: the team's name from the data, else the page's own
    // heading without its trailing subtitle ("Roman Josi", not "Roman Josi D ...")
    let x="";
    const P=siteParse(location.hash);
    if(v==="team"&&P.arg&&siteHasTeam(lg,P.arg)) x=siteTeamName(lg,P.arg);
    else{const h=document.querySelector("#view h1, #view .gh .mt, #view .phead .nm, #view .thead .code");
      if(h){const c=h.cloneNode(true), last=c.lastElementChild;
        if(last&&last.classList.contains("sub")&&!(last.nextSibling&&last.nextSibling.textContent.trim())) last.remove();
        x=c.textContent.replace(/\s+/g," ").trim().slice(0,70);}}
    t=x?`${x} | ${L}`:L;}
  document.title=(t?t+" | ":"")+"GLASSBOX";   // ASCII only: the build entity-encodes non-ASCII, and document.title would print it raw
}

/* ---------- shared: the league rail ----------
   One rail for every page: the served leagues first (MLB, NFL, NHL), then the
   placeholders. The badge is the league's SEASON label on every page; the
   pick count lives in the tooltip (Rule 1 accounting: PICKS are CONFIRMED +
   PROJECTED forecasts, never every scheduled game). A league whose payload is
   missing or half-built stays clickable and explains itself. `badge(lg)` may
   return {txt,title,cls} to override (the record page counts graded picks). */
function siteSeasonLbl(lg){
  if(lg==="mlb"){const g=state.board&&state.board.generated; return g?g.slice(0,4):"";}
  if(lg==="nfl") return state.nfl&&state.nfl.season?String(state.nfl.season):"";
  return state.nhl?nhlSeasLbl():"";
}
function siteRailCounts(lg){
  const today=nflToday();
  if(lg==="mlb"){const l=state.board&&(state.board.leagues||[]).find(x=>x.code==="mlb"), gs=(l&&l.games)||[];
    return {picks:gs.filter(g=>infoTier(g)!=="EARLY").length, games:gs.length, what:"games on the board"};}
  const n=state[lg], up=((n&&n.schedule)||[]).filter(g=>g.hs==null&&g.d>=today&&g.d<=addDays(today,6));
  return {picks:up.filter(g=>siteTier(lg,g)!=="EARLY").length, games:up.length, what:"games in the next 7 days"};
}
function siteRailBadge(lg){
  const c=siteRailCounts(lg), s=siteSeasonLbl(lg);
  return {txt:siteSeasonLbl(lg), title:`${s} season · ${c.picks} pick${c.picks===1?"":"s"} · ${c.games} ${c.what}`
    +(lg==="nhl"?" (every NHL forecast is EARLY: team ratings only, not a pick)":"")};
}
/* One rail button's class, tooltip and content. A league whose payload is still
   on its way says "loading" (never "unavailable" for a file not yet arrived). */
function siteRailBtn(lg,active,badge){
  const on=lg===active?" on":"", N=SITE_NAME[lg];
  if(!siteOk(lg)){
    if(state.loading[lg]) return {cls:"lg loading"+on, title:`${N} data is loading`,
      inner:`<span>${N}</span><span class="n">&hellip;</span>`};
    return {cls:"lg down"+on, title:`${N} data is unavailable - click to retry`,
      inner:`<span>${N}</span><span class="n">&mdash;</span>`};}
  const x=(badge||siteRailBadge)(lg)||{};
  return {cls:"lg"+on, title:x.title||"",
    inner:`<span>${N}</span>${x.txt!=null&&x.txt!==""?`<span class="n${x.cls?" "+x.cls:""}">${x.txt}</span>`:""}`};
}
let siteRailBadgeFn=null;          // the badge rule of the rail on screen (the record page counts graded picks)
function siteRail(active,badge){
  const b=state.board;
  siteRailBadgeFn=badge||null;
  const btn=lg=>{const x=siteRailBtn(lg,active,badge);
    return `<button class="${x.cls}" data-lg="${lg}" title="${x.title}">${x.inner}</button>`;};
  const soon=((b&&b.leagues)||[]).filter(l=>SITE_LGS.indexOf(l.code)<0)
    .map(l=>`<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`);
  return SITE_LGS.map(btn).join("")+soon.join("");
}
/* A league's data arrived after the page was drawn: update its rail button in
   place - the element, and the click handler the page wired onto it, stay. */
function siteRailPatch(lg){
  const V=$("#view"); if(!V||!V.querySelectorAll) return;
  V.querySelectorAll(`.rail [data-lg="${lg}"]`).forEach(el=>{
    const x=siteRailBtn(lg,el.classList.contains("on")?lg:null,siteRailBadgeFn);
    // the tooltip is markup-escaped text (entities), as siteRail writes it:
    // parse it the same way instead of assigning the raw string
    const tmp=document.createElement("div"); tmp.innerHTML=`<i title="${x.title}"></i>`;
    el.className=x.cls; el.title=tmp.firstChild.title; el.innerHTML=x.inner;});
}
/* Board-style rail wiring: switch league, redraw the board (board() keeps the
   URL and nav in step when it is called off-route). */
function siteWireRail(){
  $("#view").querySelectorAll(".rail [data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);board();});
}
/* The MLB VALUE strip, for any league: items are {team, ev, dec, href, when};
   `note` is appended after the standard disagreement caveat. */
function siteValStrip(items,note){
  if(!items.length) return "";
  return `<div class="valstrip"><span class="vh">VALUE &middot; ${items.length} live</span>
    ${items.map(x=>`<a class="vs" href="${x.href}">
      <b>${x.team}</b><span class="ev">+${Math.round(x.ev*100)}% EV</span>
      <span class="vodds">@ ${x.dec}</span>
      <span class="when">${x.when}</span></a>`).join("")}
    <span class="vnote">Model vs opening consensus, still live at the current price &mdash; a disagreement
      indicator under live CLV evaluation, not a profit promise.${note?" "+note:""} <a href="${siteHref(state.league,"record")}">Units &amp; record &rsaquo;</a></span>
  </div>`;
}

/* ---------- shared helpers ---------- */
const gbTier = v => v>=58?"hi":(v>=44?"mid":"lo");
function gb(v){ if(v==null) return '<span class="gb lo"><span class="v">NR</span></span>';
  return `<span class="gb ${gbTier(v)}"><span class="v">${v.toFixed(0)}</span>
    <span class="meter"><i style="width:${Math.max(3,Math.min(100,v))}%"></i></span></span>`;}
const playerLinkByName = name => { const p=findPlayerByName(name);
  return p?`<span class="player-link" onclick="location.hash='${siteHref("mlb","player",p.id)}'">${name}</span>`:name; };
function findPlayerByName(name){ const n=norm(name);
  return Object.values((state.db&&state.db.players)||{}).find(p=>norm(p.name)===n); }

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
  /* Called by a rail click outside the router (the NFL/NHL pages' own rails
     call setLeague()+board()): keep the URL and the nav in step. On a list page
     (standings, teams, players, record) the switch keeps the view and changes
     the league; anywhere else it opens the new league's board. */
  if(!siteRouting){
    const cur=siteParse(location.hash);
    if(SITE_LIST.indexOf(cur.v)>=0){location.hash=siteHref(state.league,cur.v);return;}
    siteSetHash(siteHref(state.league,""),cur.v!=="");
    setNav("board"); siteTitle(state.league,"");
    state.waitingFor=null;
  }
  if(state.loading[state.league]) return siteUnavailable(state.league,"");   // draws itself when the data lands
  if(state.league!=="mlb"&&!siteOk(state.league)) return siteUnavailable(state.league,"");
  if(state.league==="nfl") return nflPage();
  if(state.league==="nhl") return nhlPage();
  if(!siteOk("mlb")) return siteUnavailable("mlb","");
  const b=state.board, lg=b.leagues.find(l=>l.code==="mlb")||{games:[]}, gen=b.generated;
  const R={today:[gen,gen],tomorrow:[addDays(gen,1),addDays(gen,1)],week:[gen,addDays(gen,6)],month:[gen,addDays(gen,60)]}[state.range];
  const games=(lg.games||[]).filter(g=>g.date>=R[0]&&g.date<=R[1]);
  const rail=siteRail("mlb");
  const filts=[["today","Today"],["tomorrow","Tomorrow"],["week","Week"],["month","Month"]]
    .map(([k,t])=>`<button class="filt ${k===state.range?"on":""}" data-r="${k}">${t}</button>`).join("");
  // Live value bets, pinned ABOVE the date filter: the range defaults to today,
  // and a badge that fires on tomorrow's slate must never be invisible for it.
  const vEdges=(lg.games||[]).filter(g=>g.value&&g.value.available)
    .sort((a,b)=>(b.value.ev_cur||0)-(a.value.ev_cur||0));
  const vstrip=siteValStrip(vEdges.map(g=>({team:g.value.team,ev:g.value.ev_cur,dec:g.value.cur_dec,
    href:`#/game/${g.game_pk}`,when:`${g.away_abbr}@${g.home_abbr} &middot; ${fmtDay(g.date,gen)}`})));
  const legend=`<div class="polnote">Picks are published only once the starters are named
    (<span class="chip t-CONFIRMED">CONFIRMED</span> lineups posted,
    <span class="chip t-PROJECTED">PROJECTED</span> starters named). Games with no starter announced are
    <b>scheduled, not picks</b> &mdash; the model has team ratings only, so its lean is shown and labelled
    pre-information. Outside 55/45 a forecast is a <b>PICK</b>; inside it is a <b>LEAN</b>.
    <a href="${siteHref("mlb","record")}">Track record by tier &rsaquo;</a></div>`;
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
  siteWireRail();
  $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.range=x.dataset.r;board();});
  applyLive();
}

/* ---------- NHL ---------- */
const NHL_CITY={ANA:"Anaheim",BOS:"Boston",BUF:"Buffalo",CGY:"Calgary",CAR:"Carolina",CHI:"Chicago",COL:"Colorado",CBJ:"Columbus",DAL:"Dallas",DET:"Detroit",EDM:"Edmonton",FLA:"Florida",LAK:"Los Angeles",MIN:"Minnesota",MTL:"Montreal",NSH:"Nashville",NJD:"New Jersey",NYI:"NY Islanders",NYR:"NY Rangers",OTT:"Ottawa",PHI:"Philadelphia",PIT:"Pittsburgh",SJS:"San Jose",SEA:"Seattle",STL:"St. Louis",TBL:"Tampa Bay",TOR:"Toronto",VAN:"Vancouver",VGK:"Vegas",WSH:"Washington",WPG:"Winnipeg",UTA:"Utah",ARI:"Arizona",ATL:"Atlanta",PHX:"Phoenix"};
const nhlCity=c=>NHL_CITY[c]||c;
// The payload's season id; before one loads, the season a September-or-later
// date belongs to (an NHL season is named for the autumn it opens in).
const nhlCS=()=>{if(state.nhl&&state.nhl.cur_season) return String(state.nhl.cur_season);
  const t=nflToday(), y=+t.slice(0,4); return (+t.slice(5,7)>=9)?`${y}${y+1}`:`${y-1}${y}`;};
const nhlSeasLbl=()=>nhlCS().slice(0,4)+"&ndash;"+nhlCS().slice(6);   // "2026–27"
// The rail badge: the season the payload serves, "2026–27" - never the bare
// closing year ("2027" read as a different season next to "NFL 2026").
const nhlYear=()=>nhlSeasLbl();
