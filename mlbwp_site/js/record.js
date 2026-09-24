/* Track record: every league's graded history, tiers, units. */
const REC_ROWS=300;                 // rows rendered; rollups always use every graded pick
const REC_LG=["mlb","nfl","nhl"];
const REC_NAME={mlb:"MLB",nfl:"NFL",nhl:"NHL"};
const LL_COIN=Math.log(2);          // 0.6931 - the coin-flip log loss
/* documents/pick_policy.md. Information tiers are the honest unit: a forecast
   made 30 days out with team ratings only is a different product from one made
   with both lineups posted, and the two are never pooled into a headline.

   The tier is the one STAMPED when the graded number was locked, read off the
   row (siteTier in core.js) - never assumed from the league:
     MLB - stamped server-side (mlbwp/pred_ledger.py).
     NFL - stamped by the pre-game ledger (phase0/nfl_ph_freeze.py): PROJECTED
       when the live model served it inside a week of the game, EARLY when it
       was served further out (the season-simulation era). A played row with no
       stamp is EARLY - the conservative reading, never pooled into PROJECTED.
     NHL - site/data/nhl.json model_card features are Elo, rest, back-to-back and
       an xG TEAM rating. No roster, no lineup, no goalie: team ratings only, i.e.
       EARLY, stamped per row by phase0/nhl_hp_freeze.py. It becomes PROJECTED
       when a lineup/goalie input actually ships in the model.
   A played row whose pre-game number was never archived keeps a walk-forward
   REPLAY (row.replay): it is listed, labelled, and kept out of every rate. */
const REC_TIERS=["CONFIRMED","PROJECTED","EARLY"];
const REC_TIER_NOTE={CONFIRMED:"official lineup card posted",
  PROJECTED:"starters named or rotation-projected, lineup projected",
  EARLY:"no starter - team ratings only"};
const REC_MIN_N=30;                 // below this a tier's rate is not reported as a number
const LEAN_MAX=POL_LEAN_MAX;        // one source: see mlbwp/predict_slate.LEAN_MAX
const recIsLean=p=>Math.max(p,1-p)<=LEAN_MAX;
/* One fallback for a missing server tier, shared by every reader (a stale
   cached board.json must not render EARLY in one table and PROJECTED in
   another). Mirrors mlbwp/pred_ledger.entry_tier. */
const recTierOf=r=>REC_TIERS.indexOf(r.tier)>=0?r.tier:(r.sp?"PROJECTED":"EARLY");
const recLeague=()=>REC_LG.indexOf(state.league)>=0?state.league:"mlb";
const recMlb=()=>(state.board&&state.board.record&&state.board.record.mlb)||null;
const recMonth=k=>/^\d{4}-\d{2}$/.test(k)
  ?new Date(k+"-01T12:00:00Z").toLocaleDateString(LOC,{month:"short",year:"numeric",timeZone:TZ}):k;
const recNflHref=g=>"#/game/"+g.w+"_"+g.away+"_"+g.home;

function recRows(lg){
  if(lg==="mlb"){const m=recMlb(); if(!m) return [];
    return (m.rows||[]).map(r=>({d:r.d,away:r.away,home:r.home,p:r.p,pick:r.pick,y:r.y,
      hs:r.hs,as:r.as,sp:r.sp,tier:recTierOf(r),per:(r.d||"").slice(0,7),href:null}));}
  if(lg==="nfl"){const n=state.nfl; if(!n||!n.schedule) return [];
    // ph is the graded number: inside a week of kickoff it is what the board
    // shows, so it is the last number published before every game.
    return n.schedule.filter(g=>g.hs!=null&&g.as!=null&&g.ph!=null)
      .map(g=>({d:g.d,away:g.away,home:g.home,p:g.ph,pick:g.ph>=0.5?g.home:g.away,
        y:g.hs>g.as?1:(g.hs<g.as?0:null),hs:g.hs,as:g.as,tier:siteTier("nfl",g),
        replay:!!g.replay,at:g.frozen_at||null,per:"Wk "+g.w,href:recNflHref(g)}))
      .sort((a,b)=>a.d<b.d?1:(a.d>b.d?-1:0));}
  if(lg==="nhl"){const n=state.nhl; if(!n||!n.schedule) return [];
    return recNhlRows(n.schedule,true);}
  return [];
}
function recNhlRows(sched,links){
  return sched.filter(g=>g.hs!=null&&g.as!=null&&g.hp!=null)
    .map(g=>({d:g.d,away:g.away,home:g.home,p:g.hp,pick:g.hp>=0.5?g.home:g.away,
      y:g.y!=null?g.y:(g.hs>g.as?1:0),hs:g.hs,as:g.as,tier:siteTier("nhl",g),
      replay:!!g.replay,at:g.frozen_at||null,
      po:!!g.playoff||String(g.id||"").slice(4,6)==="03",     // NHL ids: type 03 = playoffs
      ot:g.last&&g.last!=="REG"?g.last:null,
      per:(g.d||"").slice(0,7),href:links?"#/game/nhl-"+g.id:null}))
    .sort((a,b)=>a.d<b.d?1:(a.d>b.d?-1:0));
}
/* The rows that enter any rate: never a replay (its number was not published
   before the game) and, for the NHL, never a playoff game - the model card
   scores the regular season only, and the two must agree. */
const recRated=rows=>rows.filter(r=>!r.replay&&!r.po);
const recPicks=lg=>recRated(recRows(lg)).filter(r=>r.tier!=="EARLY");   // graded PICKS only
/* Per-tier scores. MLB ships them from the server (computed over EVERY graded
   row, not just the shipped page); NFL/NHL are scored client-side. */
/* Policy rule 2 inside rule 3: a graded LEAN is not a graded PICK, so each tier
   carries BOTH - `pick` (the tier's headline rate, leans excluded) and `lean`
   (reported beside it, never folded into it). */
const recSrv=o=>o?{n:o.n,correct:Math.round(o.acc*o.n),acc:o.acc,ll:o.log_loss,
  brier:o.brier,ties:0}:null;
function recByTier(lg,rowsOverride){
  const out={};
  if(lg==="mlb"&&!rowsOverride){const m=recMlb(), bt=(m&&m.by_tier)||{};
    REC_TIERS.forEach(t=>{const o=bt[t]; if(o) out[t]={...recSrv(o),
      pick:recSrv(o.pick),lean:recSrv(o.lean)};});
    return out;}
  const rows=rowsOverride||recRated(recRows(lg));
  REC_TIERS.forEach(t=>{const sub=rows.filter(r=>r.tier===t), s=recScore(sub);
    if(s) out[t]={...s,pick:recScore(sub.filter(r=>!recIsLean(r.p))),
      lean:recScore(sub.filter(r=>recIsLean(r.p)))};});
  return out;
}
function recScore(rows){            // log loss / accuracy / Brier over graded rows
  const g=rows.filter(r=>r.y!=null&&r.p!=null);
  if(!g.length) return null;
  let ll=0,br=0,c=0;
  g.forEach(r=>{const p=Math.min(Math.max(r.p,1e-9),1-1e-9);
    ll+=-(r.y*Math.log(p)+(1-r.y)*Math.log(1-p));
    br+=(r.p-r.y)*(r.p-r.y);
    if((r.p>=0.5)===(r.y===1))c++;});
  return {n:g.length,correct:c,acc:c/g.length,ll:ll/g.length,brier:br/g.length,
    ties:rows.length-g.length};
}
function recPeriods(rows){          // oldest-first buckets with a running cumulative
  const asc=rows.slice().reverse(), keys=[], m={};
  asc.forEach(r=>{const k=r.per||"?";
    if(!m[k]){m[k]={n:0,c:0,ll:0};keys.push(k);}
    if(r.y==null) return;
    const p=Math.min(Math.max(r.p,1e-9),1-1e-9), o=m[k];
    o.n++; o.ll+=-(r.y*Math.log(p)+(1-r.y)*Math.log(1-p));
    if((r.p>=0.5)===(r.y===1))o.c++;});
  let cn=0,cc=0,cll=0;
  return keys.map(k=>{const o=m[k]; cn+=o.n; cc+=o.c; cll+=o.ll;
    return {k,n:o.n,c:o.c,acc:o.n?o.c/o.n:null,ll:o.n?o.ll/o.n:null,
      cacc:cn?cc/cn:null,cll:cn?cll/cn:null,cn};});
}
/* True pending count: the MLB list is capped by the payload, so the headline
   must come from the server's own total, not from the shipped rows. Policy
   rule 1 - this counts PICKS (CONFIRMED + PROJECTED); EARLY games are counted
   separately by recSchedTotal and never presented as picks. */
function recPendTotal(lg){
  const m=lg==="mlb"?recMlb():null;
  return m&&m.n_pending!=null?m.n_pending:recPending(lg).length;
}
function recSchedTotal(lg){
  const m=lg==="mlb"?recMlb():null;
  return m&&m.n_scheduled!=null?m.n_scheduled:recScheduled(lg).length;
}
/* Every ungraded forecast, tier-stamped, with the number the BOARD shows
   (nflProb: the season simulation beyond a week, the live model inside it).
   A game dated before today with no score has been PLAYED: it is `overdue`
   (result not in the data yet), never a pending pick "awaiting the result". */
function recUngraded(lg){
  const today=nflToday();
  if(lg==="nfl"){const n=state.nfl; if(!n||!n.schedule) return [];
    return n.schedule.filter(g=>g.hs==null&&g.ph!=null)
      .map(g=>{const pr=nflProb(g), p=pr.hp;
        return {d:g.d,away:g.away,home:g.home,p,pick:p>=0.5?g.home:g.away,
          sp:1,tier:pr.tier,overdue:g.d<today,per:"Wk "+g.w,href:recNflHref(g)};})
      .sort((a,b)=>a.d<b.d?-1:(a.d>b.d?1:0));}
  if(lg==="nhl"){const n=state.nhl; if(!n||!n.schedule) return [];
    return n.schedule.filter(g=>g.hs==null&&g.hp!=null)
      .map(g=>({d:g.d,away:g.away,home:g.home,p:g.hp,pick:g.hp>=0.5?g.home:g.away,
        sp:1,tier:siteTier("nhl",g),overdue:g.d<today,po:!!g.playoff,href:"#/game/nhl-"+g.id}))
      .sort((a,b)=>a.d<b.d?-1:(a.d>b.d?1:0));}
  return [];
}
function recPending(lg){            // locked-in PICKS whose games are not played yet
  if(lg==="mlb"){const m=recMlb(); if(!m) return [];
    return (m.pending||[]).map(r=>({d:r.d,away:r.away,home:r.home,p:r.p,pick:r.pick,
      sp:r.sp,spp:r.spp,tier:recTierOf(r),href:null}));}
  return recUngraded(lg).filter(r=>!r.overdue&&r.tier!=="EARLY");
}
/* EARLY games: scheduled, NOT picks. Shown collapsed and labelled
   pre-information. */
function recScheduled(lg){
  if(lg!=="mlb") return recUngraded(lg).filter(r=>!r.overdue&&r.tier==="EARLY");
  const m=recMlb(); if(!m) return [];
  return (m.scheduled||[]).map(r=>({d:r.d,away:r.away,home:r.home,p:r.p,pick:r.pick,
    sp:r.sp,spp:r.spp,tier:"EARLY",href:null}));
}
/* Played, not yet graded: the result has not reached the payload. */
const recOverdue=lg=>lg==="mlb"?[]:recUngraded(lg).filter(r=>r.overdue);
/* The lean count must come from the server's uncapped total when it ships one:
   counting leans in the capped `pending` array and subtracting from the
   uncapped n_pending makes the two halves stop summing to the stated total. */
function recLeanTotal(lg){
  const m=lg==="mlb"?recMlb():null;
  return m&&m.n_lean!=null?m.n_lean:recPending(lg).filter(r=>recIsLean(r.p)).length;
}
/* Cumulative-accuracy curve: the picture the tables cannot draw. Plots running
   accuracy against the 50% line so drift is visible at a glance. Pure inline
   SVG - no library, themes via currentColor/CSS vars. It pools every tier and
   both calls, so its unit is a graded FORECAST, never a "pick". */
function recChart(rows){
  const asc=rows.slice().reverse().filter(r=>r.y!=null&&r.p!=null);
  if(asc.length<8) return "";
  const W=680,H=140,PL=34,PR=8,PT=10,PB=18, iw=W-PL-PR, ih=H-PT-PB;
  let c=0; const pts=asc.map((r,i)=>{ if((r.p>=0.5)===(r.y===1))c++;
    return {x:i+1,acc:c/(i+1)};});
  const lo=Math.max(0.30,Math.min(0.45,Math.min.apply(null,pts.slice(Math.floor(pts.length/4)).map(o=>o.acc))-0.05));
  const hi=Math.min(0.90,Math.max(0.65,Math.max.apply(null,pts.slice(Math.floor(pts.length/4)).map(o=>o.acc))+0.05));
  const X=i=>PL+(i-1)/Math.max(1,pts.length-1)*iw, Y=v=>PT+(1-(v-lo)/(hi-lo))*ih;
  const d=pts.map((o,i)=>(i?"L":"M")+X(o.x).toFixed(1)+" "+Y(o.acc).toFixed(1)).join(" ");
  const area=d+` L ${X(pts.length).toFixed(1)} ${Y(lo).toFixed(1)} L ${X(1).toFixed(1)} ${Y(lo).toFixed(1)} Z`;
  const last=pts[pts.length-1], y50=Y(0.5);
  const ticks=[lo,(lo+hi)/2,hi].map(v=>`<text x="${PL-6}" y="${(Y(v)+3).toFixed(1)}" text-anchor="end"
      font-size="9" fill="var(--muted)">${(v*100).toFixed(0)}%</text>
    <line x1="${PL}" y1="${Y(v).toFixed(1)}" x2="${W-PR}" y2="${Y(v).toFixed(1)}"
      stroke="var(--line-2)" stroke-width="1" opacity=".5"/>`).join("");
  return `<div class="panel" style="margin-top:16px"><h3>Accuracy over time
      <span class="sub">running cumulative, oldest &rarr; newest &middot; all tiers pooled</span></h3>
    <div class="twrap"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}"
        preserveAspectRatio="none" role="img" aria-label="Cumulative accuracy of every graded forecast">
      ${ticks}
      ${(y50>PT&&y50<H-PB)?`<line x1="${PL}" y1="${y50.toFixed(1)}" x2="${W-PR}" y2="${y50.toFixed(1)}"
        stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3" opacity=".8"/>
        <text x="${W-PR}" y="${(y50-4).toFixed(1)}" text-anchor="end" font-size="9"
          fill="var(--muted)">coin flip</text>`:""}
      <path d="${area}" fill="var(--accent)" opacity=".10"/>
      <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="${X(last.x).toFixed(1)}" cy="${Y(last.acc).toFixed(1)}" r="3.5" fill="var(--accent)"/>
      <text x="${PL}" y="${H-5}" font-size="9" fill="var(--muted)">forecast 1</text>
      <text x="${W-PR}" y="${H-5}" text-anchor="end" font-size="9" fill="var(--muted)">forecast ${pts.length}</text>
    </svg></div></div>`;
}
/* ---------- UNITS (#/record) ----------
   Units track the EDGE-BADGE BETS and nothing else (decision 2026-08-12):
   a flat 1u on the badge side at the price RECORDED when the badge fired
   (market/edge_ledger.py — the ledger rows are the bet slips), settled on the
   result. Model picks without a badge carry NO stake: they are scored on
   accuracy/log loss in the tables below, never in units. The block ships as
   board.record.bets, computed server-side over every ledger row. */
const fmtU=v=>{const a=Math.abs(v).toFixed(2); return (+a===0?"":(v>0?"+":"&minus;"))+a+"u";};   // no "-0.00u"
/* MLB ledger rows carry Retrosheet team codes; show the everyday abbrs. The
   map is MLB's only: NFL "WAS" and NHL "ANA" are already the codes those
   leagues use everywhere else on the site. */
const RETRO2ABBR={SLN:"STL",CHA:"CWS",NYN:"NYM",NYA:"NYY",LAN:"LAD",SDN:"SD",
  SFN:"SF",CHN:"CHC",KCA:"KC",ANA:"LAA",TBA:"TB",WAS:"WSH"};
const abbrOf=c=>RETRO2ABBR[c]||c;
const recBets=()=>((state.board||{}).record||{}).bets||null;
const recUnits=lg=>{const b=recBets();return b&&b.leagues&&b.leagues[lg]?b.leagues[lg]:null;};
/* The game a bet slip was written on, from the league's own data (for a week
   label, a link and whether it has been played). MLB's board drops played
   games, so an MLB slip only resolves while its game is still upcoming. */
function recBetGame(p){
  if(p.league==="nfl"||p.league==="nhl"){const n=state[p.league];
    const g=n&&(n.schedule||[]).find(x=>x.d===p.d&&x.away===p.away&&x.home===p.home);
    return g?{g,href:p.league==="nfl"?recNflHref(g):"#/game/nhl-"+g.id,w:p.league==="nfl"?g.w:null,played:g.hs!=null}:null;}
  const l=state.board&&(state.board.leagues||[]).find(x=>x.code==="mlb");
  const g=l&&(l.games||[]).find(x=>x.date===p.d&&x.away===p.away&&x.home===p.home);
  return g?{g,href:"#/game/"+g.game_pk,w:null,played:false}:null;
}
function recUnitsChart(u){
  const pts=(u.curve||[]).map((c,i)=>({x:i+1,v:c[1],d:c[0]}));
  if(pts.length<8) return "";
  const W=680,H=140,PL=40,PR=8,PT=10,PB=18, iw=W-PL-PR, ih=H-PT-PB;
  const vs=pts.map(o=>o.v);
  let lo=Math.min(0,Math.min.apply(null,vs)), hi=Math.max(0,Math.max.apply(null,vs));
  const pad=Math.max(0.5,(hi-lo)*0.08); lo-=pad; hi+=pad;
  const X=i=>PL+(i-1)/Math.max(1,pts.length-1)*iw, Y=v=>PT+(1-(v-lo)/(hi-lo))*ih;
  const d=pts.map((o,i)=>(i?"L":"M")+X(o.x).toFixed(1)+" "+Y(o.v).toFixed(1)).join(" ");
  const last=pts[pts.length-1], y0=Y(0);
  const area=d+` L ${X(pts.length).toFixed(1)} ${y0.toFixed(1)} L ${X(1).toFixed(1)} ${y0.toFixed(1)} Z`;
  const ticks=[lo+pad,(lo+hi)/2,hi-pad].map(v=>`<text x="${PL-6}" y="${(Y(v)+3).toFixed(1)}" text-anchor="end"
      font-size="9" fill="var(--muted)">${(v>0?"+":v<0?"&minus;":"")+Math.abs(v).toFixed(1)}u</text>
    <line x1="${PL}" y1="${Y(v).toFixed(1)}" x2="${W-PR}" y2="${Y(v).toFixed(1)}"
      stroke="var(--line-2)" stroke-width="1" opacity=".5"/>`).join("");
  return `<div class="twrap" style="margin-top:10px"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}"
      preserveAspectRatio="none" role="img" aria-label="Cumulative units, picks only">
    ${ticks}
    <line x1="${PL}" y1="${y0.toFixed(1)}" x2="${W-PR}" y2="${y0.toFixed(1)}"
      stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3" opacity=".8"/>
    <text x="${W-PR}" y="${(y0-4).toFixed(1)}" text-anchor="end" font-size="9" fill="var(--muted)">break even</text>
    <path d="${area}" fill="var(--accent)" opacity=".10"/>
    <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${X(last.x).toFixed(1)}" cy="${Y(last.v).toFixed(1)}" r="3.5" fill="var(--accent)"/>
    <text x="${PL}" y="${H-5}" font-size="9" fill="var(--muted)">bet 1${u.priced_since?" &middot; "+u.priced_since:""}</text>
    <text x="${W-PR}" y="${H-5}" text-anchor="end" font-size="9" fill="var(--muted)">bet ${pts.length} &middot; ${fmtU(last.v)}</text>
  </svg></div>`;
}
/* NFL units roll up by week, like the NFL history: per-bet nets come from the
   settled curve (cumulative units, one point per settled bet, in date order). */
function recUnitsByWeek(u){
  const wOf={}; ((state.nfl&&state.nfl.schedule)||[]).forEach(g=>{if(!(g.d in wOf)) wOf[g.d]=g.w;});
  const m={}, keys=[]; let prev=0;
  (u.curve||[]).forEach(([d,cum])=>{const k=wOf[d]!=null?"Wk "+wOf[d]:recMonth(String(d).slice(0,7));
    if(!m[k]){m[k]={k,n:0,net:0};keys.push(k);} m[k].n++; m[k].net+=cum-prev; prev=cum;});
  let c=0; return keys.map(k=>{c+=m[k].net; return {k,n:m[k].n,net:m[k].net,cum:c};});
}
/* A bet slip's status, as the LEDGER states it (market/edge_ledger.bet_status):
   open = not started; awaiting = started, result not read in yet; overdue =
   still no result 7+ days after the game (the result was never ingested - a
   pipeline gap, not a void). The client only refines "awaiting" (FINAL once the
   league's own data has the score, IN PROGRESS while the game is on) and falls
   back to its own reading for a payload that predates the status field. */
const REC_STALE_DAYS=7;            // market/edge_ledger.STALE_UNSETTLED_DAYS
function recBetStatus(p,G){
  const today=nflToday();
  const inProg=()=>{const st=p.start_utc?Date.parse(p.start_utc):NaN;
    return Number.isFinite(st)?(Date.now()>=st&&Date.now()-st<5*3600e3):p.d===today;};
  const FIN=`<span class="chip pend" title="played - settles on the next odds cycle">FINAL</span>`;
  const AWT=`<span class="chip ovd" title="played - the result is not in the data yet">AWAITING RESULT</span>`;
  const OPN=`<span class="chip" title="not played yet">OPEN</span>`;
  if(p.status==="overdue") return `<span class="chip ovd" title="no result ingested ${REC_STALE_DAYS}+ days after the game - a pipeline gap, not a void; it settles when the final is read in">OVERDUE</span>`;
  if(p.status==="awaiting") return G&&G.played?FIN
    :(inProg()?`<span class="chip pend" title="started - not final yet">IN PROGRESS</span>`:AWT);
  if(p.status==="open") return OPN;
  return G&&G.played?FIN:(p.d<today?AWT:OPN);
}
const REC_UNIT_TIERS=[...REC_TIERS,"UNKNOWN"];
const recUnitTierChip=t=>REC_TIERS.indexOf(t)>=0?recTierChip(t)
  :`<span class="chip t-UNKNOWN" title="recorded before bets carried an information-tier stamp">UNKNOWN</span>`;
const fmtRoi=r=>{const v=Math.abs(r*100).toFixed(1); return (+v===0?"":(r>0?"+":"&minus;"))+v+"%";};   // no "-0.0%"
function recUnitsPanel(lg){
  const head=`<h3>Units <span class="sub">EDGE bets only &mdash; flat 1u per badge at the recorded price</span></h3>`;
  const u=recUnits(lg), all=recBets(), pol=(u&&u.policy)||null;
  // documents/pick_policy.md: what a badge of this league is priced without is
  // disclosed. With the EARLY gate on (policy.gate "enforced") every NHL
  // forecast is EARLY, so no new NHL badge is staked until a goalie input ships.
  const gated=!!(pol&&pol.gate==="enforced");
  const nhlNote=lg!=="nhl"?"":(gated
    ?`<div class="polnote warn" style="margin:0 0 10px">Every NHL forecast is EARLY (team ratings only, no goalie or
      lineup input), and an EARLY forecast carries <b>no badge</b>: no new NHL bet is staked until a goalie input ships.
      Any NHL bet below was recorded before that gate, priced <b>before starting goalies were confirmed</b> &mdash; it
      measures the badge, not a pick.</div>`
    :`<div class="polnote warn" style="margin:0 0 10px">NHL badges are priced <b>before starting
      goalies are confirmed</b> &mdash; our own measurement says goalie news is the one thing the market knows and this
      model does not. Every NHL forecast is EARLY (team ratings only), so these units measure the <b>badge</b>, not a
      pick: NHL badges are disclosed rather than gated until a goalie input ships.</div>`);
  const polLine=lg==="nfl"&&pol&&pol.deficit?`<div class="sub" style="margin:0 0 10px">${/^priced /.test(pol.deficit)
      ?"Every NFL badge is ":"What every NFL badge is priced without: "}${siteEsc(pol.deficit)}.${gated
      ?" A forecast more than a week from kickoff is EARLY and carries no badge.":""}</div>`:"";
  const open=all?(all.pending||[]).filter(p=>p.league===lg):[];
  const voids=all&&Array.isArray(all.voids)?all.voids.filter(v=>v.league===lg):null;
  const nVoid=(u&&u.n_void)||0, nOver=(u&&u.n_overdue)||0;
  const nAwait=u&&u.n_awaiting!=null?u.n_awaiting
    :open.filter(p=>p.status==="awaiting"||p.status==="overdue").length;
  if(!u||(!u.settled&&!open.length&&!nVoid&&!nOver))
    return `<div class="panel" style="margin-top:16px">${head}${nhlNote}${polLine}
      <div class="empty" style="padding:18px 0">No ${REC_NAME[lg]} EDGE bet yet. A unit is staked only when a game carries the value
        badge (model vs opening consensus above the EV threshold) &mdash; picks without a badge are scored on
        accuracy below and are never staked.</div></div>`;
  const s=u.settled, bt=u.by_tier&&typeof u.by_tier==="object"?u.by_tier:null;
  /* Policy rule 3: units are split by information tier first; the all-badges
     sum is a secondary line under the split, never the headline. A payload
     that predates the split shows the sum alone and says so. */
  const tRow=(chip,o,cls)=>{const thinT=o.n<REC_MIN_N;
    return `<tr${cls?` class="${cls}"`:""}><td class="a">${chip}</td>
    <td><span class="num">${o.n}</span></td>
    <td><span class="num">${o.w}&ndash;${o.l}</span></td>
    <td><span class="num ${o.net>=0?"pos":"neg"}">${fmtU(o.net)}</span></td>
    <td><span class="num ${o.roi>=0?"pos":"neg"}"${thinT?` title="n=${o.n}: too few settled bets for the ROI to mean much"`:""}>${fmtRoi(o.roi)}</span>${thinT?` <span class="sub" style="font-size:10px">n&lt;${REC_MIN_N}</span>`:""}</td></tr>`;};
  const tierKeys=bt?REC_UNIT_TIERS.concat(Object.keys(bt).filter(t=>REC_UNIT_TIERS.indexOf(t)<0)).filter(t=>bt[t]&&bt[t].n):[];
  const tierRows=tierKeys.map(t=>tRow(recUnitTierChip(t),bt[t])).join("");
  const allRow=s?tRow(`<span class="sub">All badges</span>`,s,"allrow"):"";
  const unitsTbl=s?`<div class="twrap" style="margin-top:10px"><table>
      <thead><tr><th class="a">${bt?"Tier":""}</th><th>Bets</th><th>W&ndash;L</th><th>Units</th><th>ROI</th></tr></thead>
      <tbody>${tierRows}${allRow}</tbody></table></div>
    <div class="sub" style="margin-top:6px">${bt
      ?`Split by the information tier of the forecast each badge was priced from; <i>All badges</i> is their sum, not a headline.`
      :`This build does not split units by information tier yet &mdash; the line above is every badge summed.`}
      Staked ${s.staked.toFixed(0)}u &middot; average price ${s.avg_dec.toFixed(2)}.</div>`:"";
  const ab=c=>lg==="mlb"?abbrOf(c):c;
  const hasTier=open.some(p=>p.tier);
  const obody=open.map(p=>{const G=recBetGame(p);
    return `<tr class="recrow" ${G?`onclick="location.hash='${G.href}'"`:""}>
    <td class="a"><span class="sub">${G&&G.w!=null?"Wk "+G.w+" &middot; ":""}${(p.d||"").slice(5)}</span></td>
    <td class="a">${ab(p.away)} <span class="sub">@</span> ${ab(p.home)}</td>
    <td class="a"><span class="ab" style="color:var(--accent)">${ab(p.team)}</span></td>
    ${hasTier?`<td class="a">${p.tier?recUnitTierChip(p.tier):""}</td>`:""}
    <td><span class="num">${p.dec!=null?(+p.dec).toFixed(2):"&mdash;"}</span></td>
    <td><span class="num">${p.ev!=null?"+"+(p.ev*100).toFixed(1)+"%":"&mdash;"}</span></td>
    <td>${recBetStatus(p,G)}${p.stale_model?` <span class="chip stale" title="priced from a model build the freshness rules would not badge from - kept in the units, out of the CLV promotion gate">STALE MODEL</span>`:""}</td></tr>`;}).join("");
  const openTbl=open.length?`<div class="subh" style="margin-top:12px">Open bets <span class="sub"
      style="font-weight:400;font-family:var(--sans)">1u each, locked at the recorded price</span></div>
    <div class="twrap"><table>
      <thead><tr><th class="a">Date</th><th class="a">Matchup</th><th class="a">Bet</th>${hasTier?`<th class="a">Tier</th>`:""}<th>Price</th><th>EV @ record</th><th>Status</th></tr></thead>
      <tbody>${obody}</tbody></table></div>`:"";
  const byWeek=lg==="nfl";
  const rolls=byWeek?recUnitsByWeek(u):(u.by_month||[]);
  const months=rolls.slice().reverse().map(o=>`<tr>
    <td class="a">${byWeek?o.k:recMonth(o.k)}</td><td><span class="num">${o.n}</span></td>
    <td><span class="num ${o.net>=0?"pos":"neg"}">${fmtU(o.net)}</span></td>
    <td><span class="num ${o.cum>=0?"pos":"neg"}">${fmtU(o.cum)}</span></td></tr>`).join("");
  const monthTbl=months?`<details class="sched" style="margin-top:10px"><summary>By ${byWeek?"week":"month"}</summary><div class="body">
    <div class="twrap"><table>
      <thead><tr><th class="a">${byWeek?"Week":"Month"}</th><th>Bets</th><th>Units</th><th>Cum units</th></tr></thead>
      <tbody>${months}</tbody></table></div></div></details>`:"";
  const thin=s&&s.n<30?`<div class="sub" style="margin-top:8px"><b>n=${s.n} settled &mdash; not yet meaningful.</b>
    Units swing hard at this sample.</div>`:"";
  const pl=(k,a,b)=>k===1?a:b;
  // why each void happened, from the ledger's own evidence (grouped, counted)
  let voidTxt="";
  if(nVoid){
    const rc={}; (voids||[]).forEach(v=>{const r=String(v.reason||"reason not recorded"); rc[r]=(rc[r]||0)+1;});
    const reasons=Object.keys(rc).map(r=>siteEsc(r)+(rc[r]>1?` &times;${rc[r]}`:"")).join("; ");
    // the ledger voids on positive evidence and lists why; an older block
    // carries a bare count, which is reported as just that
    voidTxt=voids
      ?` ${nVoid} ${pl(nVoid,"bet was","bets were")} <b>voided on evidence</b>, stake returned and counted nowhere
        (a void is not a loss).${reasons?` Why: ${reasons}.`:""}`
      :` ${nVoid} ${pl(nVoid,"bet was","bets were")} <b>voided</b>, stake returned and counted nowhere (a void is
        not a loss); this build does not list the void reasons.`;
  }
  const ovdTxt=`no result ingested ${REC_STALE_DAYS}+ days after the game &mdash; a pipeline gap, not a void`;
  const awaitTxt=nAwait?` ${nAwait} ${pl(nAwait,"bet is","bets are")} on started games awaiting settlement${nOver
      ?`, ${nOver} of them overdue (${ovdTxt})`:""}.`
    :(nOver?` ${nOver} ${pl(nOver,"bet is","bets are")} overdue (${ovdTxt}).`:"");
  const sm=u.stale_model&&u.stale_model.n>0?u.stale_model:null;
  const staleTxt=sm?` <span class="warnc">${sm.n} ${pl(sm.n,"bet was","bets were")} priced from a stale model build${
      Array.isArray(sm.builds)&&sm.builds.length?` (${sm.builds.map(siteEsc).join(", ")})`:""}</span>: they were published badges, so
      they ${s?"stay in the units above":"will count in the units when they settle"}${sm.n_settled?` (${sm.n_settled} settled, ${fmtU(sm.net||0)})`:""}, but they are excluded
      from the CLV promotion gate.`:"";
  return `<div class="panel" style="margin-top:16px">${head}${nhlNote}${polLine}
    ${s?unitsTbl
      :`<div class="empty" style="padding:18px 0">Tracking starts here &mdash; ${open.length} open bet${open.length===1?"":"s"}, none settled yet.</div>`}
    ${recUnitsChart(u)}
    ${thin}
    ${openTbl}
    ${monthTbl}
    <div class="sub" style="margin-top:10px">Every unit is a bet the <b>EDGE badge</b> fired and the site marked
      takeable: 1u on the badge side at the consensus price <b>recorded at that moment</b> (never re-priced); a win
      banks price&nbsp;&minus;&nbsp;1, a loss costs 1. Picks without a badge are never staked. This is a live
      evaluation of the badge, not a profit promise.${awaitTxt}${voidTxt}${staleTxt}</div></div>`;
}
function recSource(lg){
  const m=recMlb();
  if(lg==="mlb") return `Every pick is archived <b>before first pitch</b> and graded from the final score &mdash;
    nothing back-filled. Tracking began <b>${(m&&m.since)||"&mdash;"}</b>; what gets graded is the last, most-informed
    pre-game forecast, and the tier stamp records which it was.`;
  if(lg==="nfl"){const since=state.nfl&&state.nfl.tracking_since;
    return `Graded numbers are the last pre-game probabilities the site published, locked in the NFL pre-game ledger
    and restored on every rebuild, then scored against the final.${since?` Tracking began <b>${since}</b>.`:""} Each row
    keeps the tier it was locked in: <b>PROJECTED</b> when the live model served it inside a week of kickoff,
    <b>EARLY</b> when it was served further out (season-simulation era) &mdash; scheduled, not a pick. A played game
    with no archived pre-game number is marked <span class="chip replay">REPLAY</span> and kept out of every rate.`;}
  return `Graded numbers are the pre-game probabilities locked in the NHL pre-game ledger at every serve, before puck
    drop, and restored on every rebuild. The shipped model has no roster, lineup or goalie input, so every NHL forecast
    is <b>EARLY</b> &mdash; published for transparency, not counted as picks. A played game with no locked number keeps
    its walk-forward replay, is marked <span class="chip replay">REPLAY</span> and is kept out of every rate; playoff
    games are listed but scored apart, as the model card scores the regular season only.`;
}
/* One line per league: how old the data behind this page is, and whether a
   played game is still waiting on its result. */
function recFreshLine(lg){
  const f=siteFresh(lg), b=recBets(); if(!f) return "";
  const when=f.at?siteEt(f.at,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})+" ET":null;
  const parts=lg==="mlb"
    ?[`MLB board built <b>${f.built||"?"}</b>`,`graded through <b>${f.through||"?"}</b>`]
    :[`${REC_NAME[lg]} data served <b>${when||"(no build time in this payload)"}</b>`,
      `results through <b>${f.through?(f.wk?"Wk "+f.wk+" &middot; "+f.through:f.through):"none yet"}</b>`,
      ...(f.pending?[`<b>${f.pending}</b> played game${f.pending===1?"":"s"} not yet in the data`]:[])];
  if(b&&b.updated) parts.push(`units ledger ${siteEt(Date.parse(b.updated),{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})} ET`);
  return `<div class="${f.stale?"polnote warn":"sub"}" style="margin:8px 0 0;font-size:12px">Data as of: ${parts.join(" &middot; ")}${
    f.stale?(f.late?" &mdash; results are late; the pending lists below say which games.":" &mdash; this data is older than it should be in season."):""}</div>`;
}
/* Policy rule 3: accuracy and log loss are reported PER TIER and never pooled
   into a headline. A tier with too few graded rows shows a dash (with the
   reason in its tooltip) instead of printing a number nobody should read. */
function recTierGrid(lg,rowsOverride,appliesOverride){
  const bt=recByTier(lg,rowsOverride);
  // Only tiers the league can actually produce: MLB all three, NFL live-model
  // + simulation, NHL team ratings only.
  const applies=appliesOverride||(lg==="mlb"?REC_TIERS:(lg==="nfl"?["PROJECTED","EARLY"]:["EARLY"]));
  const note=t=>(lg==="nfl"&&t==="PROJECTED")
    ?"live model: ratings + depth charts, no inactives feed"
    :(lg!=="mlb"&&t==="EARLY")
    ?(lg==="nhl"?"team ratings only - no roster, lineup or goalie input"
                :"season simulation from team strength only")
    :REC_TIER_NOTE[t];
  const trs=applies.map(t=>{
    const s=bt[t], early=t==="EARLY";
    // Rule 1: an EARLY tier has no picks in it, so its row scores every graded
    // forecast; the headline of the other tiers is PICKS only, leans beside.
    const head=early?s:(s&&s.pick), ln=early?null:(s&&s.lean);
    const dash=r=>`<span class="sub" title="${r}">&mdash;</span>`;
    const scored=head&&head.n>=REC_MIN_N;
    const d=scored?head.ll-LL_COIN:0;
    return `<tr><td class="a"><span class="chip t-${t}" title="${note(t)}">${t}</span></td>
      <td><span class="num">${head?head.n:0}</span>${early?` <span class="sub" style="font-size:10px">not picks</span>`:""}</td>
      <td>${scored?`<span class="num ${head.acc>=0.5?"pos":"neg"}">${(head.acc*100).toFixed(1)}%</span>
        <span class="sub">${head.correct}/${head.n}</span>`:dash(head&&head.n?`only ${head.n} graded — needs ${REC_MIN_N} before a rate is worth reading`:"nothing graded at this tier yet")}</td>
      <td>${scored?`<span class="num">${head.ll.toFixed(4)}</span>
        <span class="${d<=0?"pos":"neg"}" style="font-size:11px">${d<=0?"&minus;":"+"}${Math.abs(d).toFixed(4)}</span>`:dash("scored once the tier clears "+REC_MIN_N)}</td>
      <td class="a"><span class="sub">${ln?`${ln.n} lean${ln.n===1?"":"s"} beside${ln.n>=REC_MIN_N?` (${(ln.acc*100).toFixed(1)}%)`:""}`:""}</span></td></tr>`;
  }).join("");
  return `<div class="twrap"><table>
    <thead><tr><th class="a">Tier</th><th>Graded</th><th>Accuracy</th><th>LL vs coin</th><th></th></tr></thead>
    <tbody>${trs}</tbody></table></div>`;
}
// confidence bar: how far the pick sits from a coin flip, so the eye can scan
// conviction down the column instead of reading every percentage
const recConf=p=>{const c=Math.max(p,1-p);
  return `<span class="cbar"><i style="width:${((c-0.5)*200).toFixed(0)}%"></i></span>
    <span class="num">${pctI(c)}%</span>`;};
const recTierChip=t=>`<span class="chip t-${t}">${t}</span>`;
/* Rule 1 then rule 2: an EARLY forecast carries no call at all (it was never a
   pick); otherwise inside 55/45 the model is not claiming much - say so. */
const recCallChip=r=>r.tier==="EARLY"?`<span class="chip t-EARLY" title="published before the information existed - not a pick">PRE-INFO</span>`
  :(recIsLean(r.p)?`<span class="chip lean">LEAN</span>`:`<span class="chip pick">PICK</span>`);
const recFinal=r=>`<span class="num">${r.as}&ndash;${r.hs}</span>${r.ot?` <span class="sub">${r.ot}</span>`:""}`;
function recHistBody(shown){
  const outcome=r=>{
    if(r.y==null) return `<td>${recFinal(r)}</td><td><span class="chip">TIE</span></td>`;
    const hit=(r.p>=0.5)===(r.y===1);
    return `<td>${recFinal(r)}</td>
      <td><span class="chip ${hit?"hit":"miss"}">${hit?"HIT":"MISS"}</span></td>`;};
  return shown.map(r=>`<tr class="recrow" ${r.href?`onclick="location.hash='${r.href}'"`:""}>
    <td class="a"><span class="sub"${r.at?` title="pre-game number locked ${String(r.at).replace("T"," ").slice(0,16)} UTC"`:""}>${(r.d||"").slice(5)}</span></td>
    <td class="a">${r.away} <span class="sub">@</span> ${r.home}</td>
    <td class="a">${recTierChip(r.tier)}${r.replay?` <span class="chip replay" title="no pre-game number was archived: a walk-forward replay, excluded from every rate">REPLAY</span>`:""}${r.po?` <span class="chip po" title="playoff game - scored apart from the regular season">PLAYOFF</span>`:""}</td>
    <td class="a">${recCallChip(r)}</td>
    <td class="a"><span class="ab" style="color:var(--accent)">${r.pick}</span></td>
    <td class="cf">${recConf(r.p)}</td>
    ${outcome(r)}</tr>`).join("");
}
const REC_HIST_HEAD=`<thead><tr><th class="a">Date</th><th class="a">Matchup</th><th class="a">Tier</th><th class="a">Call</th><th class="a">Pick</th><th>Confidence</th><th>Final</th><th>Result</th></tr></thead>`;
function recPeriodTable(lg,rated){
  const per=recPeriods(rated).reverse().map(o=>`<tr>
    <td class="a">${recMonth(o.k)}</td><td><span class="num">${o.n}</span></td>
    <td><span class="num">${o.c}</span></td>
    <td><span class="num">${o.acc==null?"&mdash;":(o.acc*100).toFixed(1)+"%"}</span></td>
    <td><span class="num">${o.ll==null?"&mdash;":o.ll.toFixed(4)}</span></td>
    <td><span class="num">${o.cacc==null?"&mdash;":(o.cacc*100).toFixed(1)+"%"}</span></td>
    <td><span class="num">${o.cll==null?"&mdash;":o.cll.toFixed(4)}</span></td></tr>`).join("");
  return `<div class="twrap"><table>
    <thead><tr><th class="a">${lg==="nfl"?"Week":"Month"}</th><th>Graded</th><th>Correct</th><th>Acc</th><th>LL</th><th>Cum acc</th><th>Cum LL</th></tr></thead>
    <tbody>${per}</tbody></table></div>`;
}
/* ---------- the NHL's previous season ----------
   nhl.json carries the served season only, so at the rollover the NHL record
   went from a full season to nothing. Until the new season grades its first
   game, the previous one is shown - in its own collapsed block, labelled for
   what it is, and never pooled with the live record. Source: the prev_season
   block phase0/nhl_serve.py ships in nhl.json - {season, label, kind
   ("replay" | "ledger"), note, rows} with rows either objects or arrays
   named by rows_cols. */
function recPrevNhl(){
  const P=state.nhl&&state.nhl.prev_season; if(!P||!Array.isArray(P.rows)) return null;
  const cols=P.rows_cols;
  const rows=P.rows.map(r=>Array.isArray(r)&&Array.isArray(cols)?Object.fromEntries(cols.map((c,i)=>[c,r[i]])):r)
    .filter(r=>r&&typeof r==="object");
  return {...P,rows};
}
function recPrevBlock(lg){
  if(lg!=="nhl"||recRated(recRows("nhl")).length) return "";
  const P=recPrevNhl(); if(!P) return "";
  // a previous season is never a pick record here: team ratings only (EARLY)
  const rows=recNhlRows(P.rows,false).map(r=>Object.assign(r,{tier:"EARLY"}));
  if(!rows.length) return "";
  const rated=rows.filter(r=>!r.po&&!r.replay);
  const sid=String(P.season||""), lbl=P.label||(sid.slice(0,4)+"&ndash;"+sid.slice(6));
  const isReplay=(P.kind||P.source)!=="ledger";
  const shown=rows.slice(0,REC_ROWS);
  return `<details class="sched" style="margin-top:16px"><summary>${lbl} season &mdash; ${isReplay?"back-tested replay":"final record"}
      <span class="sub">${rows.length} games &middot; ${isReplay?"not a live record &middot; ":""}never pooled with ${nhlSeasLbl()}</span></summary>
    <div class="body">
      <div class="sub" style="margin-bottom:10px">${P.note||(isReplay
        ?`The ${lbl} numbers are the NHL model's walk-forward replay of that season: each game predicted from its
          pre-game state, but computed after the fact, not published before puck drop. It is shown because
          ${nhlSeasLbl()} has no graded game yet &mdash; read it as a back-test, not as a track record.`
        :`Last season's locked pre-game forecasts, graded.`)}</div>
      ${recTierGrid("nhl",rated,["EARLY"])}
      <div class="subh" style="margin-top:14px">By month</div>
      ${recPeriodTable("nhl",rated)}
      <details class="sched" style="margin-top:10px"><summary>Every ${lbl} game
          <span class="sub">${shown.length<rows.length?`most recent ${shown.length} of ${rows.length}`:`all ${rows.length}`}</span></summary>
        <div class="body"><div class="twrap"><table>${REC_HIST_HEAD}<tbody>${recHistBody(shown)}</tbody></table></div></div></details>
    </div></details>`;
}
function recSection(lg){
  if(lg!=="mlb"&&!siteOk(lg))
    return `<div class="empty" style="margin-top:12px;padding:28px 0">${REC_NAME[lg]} data ${state.failed[lg]?"could not be loaded":"is being rebuilt &mdash; this build is incomplete"},
      so its record cannot be drawn right now.</div>${recUnitsPanel(lg)}`;
  const all=recRows(lg), rows=lg==="mlb"?all:recRated(all), srv=lg==="mlb"?(recMlb()||{}).rollup:null;
  const s=srv?{n:srv.n,correct:Math.round(srv.acc*srv.n),acc:srv.acc,ll:srv.log_loss,
               brier:srv.brier,ties:0}:recScore(rows);
  const nExcl=all.length-rows.length, nRep=all.filter(r=>r.replay).length, nPo=all.filter(r=>r.po&&!r.replay).length;

  // The long per-league tier explanation lives behind a click; the numbers don't.
  const tiers=`<div class="panel" style="margin-top:16px"><h3>Accuracy by tier
      <span class="sub">never pooled &mdash; different information, different product</span></h3>
    ${recTierGrid(lg)}
    ${nExcl?`<div class="sub" style="margin-top:8px">${[nRep?`${nRep} replay${nRep===1?"":"s"} (no archived pre-game number)`:"",
      nPo?`${nPo} playoff game${nPo===1?"":"s"}`:""].filter(Boolean).join(" and ")} listed in the history but kept out of these rates.</div>`:""}
    <details class="sched" style="margin-top:8px"><summary>What the tiers mean</summary>
      <div class="body"><div class="sub">Tiers describe what the model actually had when the graded forecast was
      made. ${lg==="mlb"
        ?`<b>CONFIRMED</b> the official lineup card is posted, <b>PROJECTED</b> both starters known with a projected
          lineup, <b>EARLY</b> team ratings only. The tier is the state at the <i>last pre-game refresh</i> (every
          4h), so CONFIRMED is a start-time-selected subset of lineups-posted games &mdash; do not read the tiers
          as a controlled comparison.`
        :lg==="nfl"
        ?`<b>PROJECTED</b> is the live model (ratings, depth charts, lineup power) served inside a week of kickoff;
          a number served further out is the season-simulation era &mdash; <b>EARLY</b>, not a pick. Each graded game
          keeps the tier stamped when its number was locked. No confirmed-inactives feed, so no CONFIRMED tier.`
        :`The shipped NHL model has no roster, lineup or goalie input, so every forecast is <b>EARLY</b> &mdash;
          team ratings only. A PROJECTED tier arrives when a lineup or goalie feature actually ships in the model.`}
      </div></div></details></div>`;

  // PENDING = picks only (CONFIRMED + PROJECTED), not yet played.
  const pend=recPending(lg), pshown=pend.slice(0,REC_ROWS);
  const nLean=recLeanTotal(lg), nPick=recPendTotal(lg)-nLean;
  const pbody=pshown.map(r=>`<tr class="recrow" ${r.href?`onclick="location.hash='${r.href}'"`:""}>
    <td class="a"><span class="sub">${r.per&&lg==="nfl"?r.per+" &middot; ":""}${(r.d||"").slice(5)}</span></td>
    <td class="a">${r.away} <span class="sub">@</span> ${r.home}</td>
    <td class="a">${recTierChip(r.tier)}${r.spp?' <span class="sub" style="font-size:10px" title="no probable announced yet — the starter is this model\'s own rotation projection">SP proj</span>':""}</td>
    <td class="a"><span class="ab" style="color:var(--accent)">${r.pick}</span></td>
    <td class="cf">${recConf(r.p)}</td>
    <td><span class="num">${amOdds(Math.max(r.p,1-r.p))}</span></td>
    <td>${recCallChip(r)}</td></tr>`).join("");
  const nextUp=lg==="mlb"?null:recUngraded(lg).find(r=>!r.overdue);
  const emptyWhy=lg==="nhl"
    ?` &mdash; the NHL model has no lineup or goalie input, so it publishes no picks: every forecast is EARLY and
      listed under <i>Scheduled games</i> below.`
    :lg==="nfl"
    ?(!state.nfl||!state.nfl.schedule||!state.nfl.schedule.length?` &mdash; the NFL schedule is missing from this build.`
      :!nextUp?` &mdash; the ${state.nfl.season||""} regular season is complete.`
      :` &mdash; no NFL forecast is PROJECTED right now (served inside a week of kickoff). Next kickoff:
        ${fmtDay(nextUp.d,nflToday())}.`)
    :".";
  const pendPanel=`<div class="panel"><h3>Pending picks
      <span class="sub">${pend.length?`${pshown.length<recPendTotal(lg)?`next ${pshown.length} of ${recPendTotal(lg)}`:`all ${pend.length}`} &middot; locked in, awaiting the result`:"none right now"}</span></h3>
    ${pend.length?`<div class="twrap"><table>
        <thead><tr><th class="a">Date</th><th class="a">Matchup</th><th class="a">Tier</th><th class="a">Pick</th><th>Confidence</th><th>Fair odds</th><th>Call</th></tr></thead>
        <tbody>${pbody}</tbody></table></div>
      <div class="sub" style="margin-top:8px">${nPick} pick${nPick===1?"":"s"} + ${nLean}
        lean${nLean===1?"":"s"} (inside 55/45), written down <b>before</b> the games are played &mdash;
        every one grades HIT or MISS.</div>`
    :`<div class="empty" style="padding:18px 0">No picks waiting on a result${emptyWhy}</div>`}</div>`;

  // PLAYED, NOT YET GRADED: the result has not reached the data.
  const ovd=recOverdue(lg);
  const ovdPanel=ovd.length?`<div class="panel" style="margin-top:16px"><h3>Played &mdash; result not in the data yet
      <span class="sub">${ovd.length} game${ovd.length===1?"":"s"} &middot; graded as published once the final lands</span></h3>
    <div class="twrap"><table>
      <thead><tr><th class="a">Date</th><th class="a">Matchup</th><th class="a">Tier</th><th class="a">Side</th><th>Confidence</th><th>Status</th></tr></thead>
      <tbody>${ovd.map(r=>`<tr class="recrow" ${r.href?`onclick="location.hash='${r.href}'"`:""}>
        <td class="a"><span class="sub">${r.per?r.per+" &middot; ":""}${(r.d||"").slice(5)}</span></td>
        <td class="a">${r.away} <span class="sub">@</span> ${r.home}</td>
        <td class="a">${recTierChip(r.tier)}</td>
        <td class="a"><span class="ab">${r.pick}</span></td>
        <td class="cf">${recConf(r.p)}</td>
        <td><span class="chip ovd">AWAITING RESULT</span></td></tr>`).join("")}</tbody></table></div>
    <div class="sub" style="margin-top:8px">These games have been played; the site's ${REC_NAME[lg]} data has not picked
      up the final yet (${lg==="nfl"?"NFL results arrive with the weekly data run":"NHL data is re-served every 4 hours"}).
      Their pre-game numbers are already locked and are graded exactly as published.</div></div>`:"";

  // SCHEDULED = EARLY tier. Collapsed, labelled pre-information, not counted.
  const sch=recScheduled(lg), sshown=sch.slice(0,REC_ROWS);
  const sbody=sshown.map(r=>`<tr class="recrow" ${r.href?`onclick="location.hash='${r.href}'"`:""}>
    <td class="a"><span class="sub">${r.per&&lg==="nfl"?r.per+" &middot; ":""}${(r.d||"").slice(5)}</span></td>
    <td class="a">${r.away} <span class="sub">@</span> ${r.home}</td>
    <td class="a"><span class="ab">${r.pick}</span></td>
    <td class="cf">${recConf(r.p)}</td>
    <td><span class="chip t-EARLY">PRE-INFO</span></td></tr>`).join("");
  const schPanel=sch.length?`<details class="sched"><summary>Scheduled games &mdash; not picks
      <span class="sub">${recSchedTotal(lg)} EARLY-tier games on the board</span></summary>
    <div class="body">
      <div class="sub" style="margin-bottom:10px">${lg==="mlb"
        ?`No starter announced yet &mdash; team ratings only.`
        :lg==="nfl"
        ?`Not served by the live model inside a week of kickoff &mdash; the season simulation off team strength
          only, the same number the board shows.`
        :`The shipped NHL model has no roster, lineup or goalie input &mdash; team ratings only.`}
        The lean is what the model currently thinks, published for transparency and <b>excluded from the pick
        count</b>.</div>
      <div class="twrap"><table>
        <thead><tr><th class="a">Date</th><th class="a">Matchup</th><th class="a">Lean</th><th>Confidence</th><th>Information</th></tr></thead>
        <tbody>${sbody}</tbody></table></div>
      ${sshown.length<recSchedTotal(lg)?`<div class="sub" style="margin-top:8px">Showing the next ${sshown.length}
        of ${recSchedTotal(lg)}.</div>`:""}
    </div></details>`:"";

  if(!s){                     // nothing graded yet - the pending picks are still receipts
    return `<div class="empty" style="margin-top:12px;padding:28px 0">No ${REC_NAME[lg]} forecasts have been graded yet${lg==="nhl"?` in ${nhlSeasLbl()}`:""}.<br>
        <span class="sub">${recSource(lg)}</span></div>
      ${recUnitsPanel(lg)}
      ${tiers}
      <div style="margin-top:16px">${pendPanel}</div>
      ${ovdPanel}
      ${schPanel}
      ${recPrevBlock(lg)}`;
  }
  const warn=s.n<30?`<div class="sub" style="margin-top:12px"><b>n=${s.n} &mdash; not yet meaningful.</b>
    Treat this page as a receipt that the forecasts were made, not as evidence.</div>`:"";
  const shown=all.slice(0,REC_ROWS);
  const cnt=recCounts(lg), nGPick=cnt.picks, nTot=srv?srv.n:all.length;
  // MLB ships only the most recent graded rows: the listing, the curve and the
  // period table below are drawn from those; the tier table uses every row.
  const capped=lg==="mlb"&&all.length<nTot;
  const oldest=capped&&all.length?all[all.length-1].d:null;
  const capNote=capped?`<div class="sub" style="margin-top:8px">The listing, the curve and this table are drawn from
      the most recent ${all.length} graded forecasts the page ships${oldest?` (from ${new Date(oldest+"T12:00:00Z")
      .toLocaleDateString(LOC,{month:"short",day:"numeric",timeZone:TZ})})`:""}, so the oldest month is partial.
      The tier table above is computed over all ${nTot}.</div>`:"";
  // One glance: units P&L, then per-tier accuracy, then the pending receipts.
  // Everything else (full graded listing, period rollups, accuracy curve,
  // methodology) is real but secondary — it lives behind one click.
  return `${warn}
    ${recUnitsPanel(lg)}
    ${tiers}
    <div style="margin-top:16px">${pendPanel}</div>
    ${ovdPanel}
    ${schPanel}
    <details class="sched" style="margin-top:16px"><summary>Full history &mdash; every graded forecast
        <span class="sub">${nTot} graded &middot; ${nGPick} pick${nGPick===1?"":"s"} &middot; ${recPendTotal(lg)} pending</span></summary>
      <div class="body">
        <div class="sub" style="margin-bottom:10px">${recSource(lg)}</div>
        ${recChart(rows)}
        <div class="twrap" style="margin-top:10px"><table>
          ${REC_HIST_HEAD}
          <tbody>${recHistBody(shown)}</tbody></table></div>
        <div class="sub" style="margin-top:8px">${shown.length<nTot?`Most recent ${shown.length} of ${nTot}. `:""}This
          listing pools every tier and both calls &mdash; a <span class="chip lean">LEAN</span> or
          <span class="chip t-EARLY">EARLY</span> row is never inside any pick rate on this page.</div>
        <div class="subh" style="margin-top:14px">By ${lg==="nfl"?"week":"month"}</div>
        ${recPeriodTable(lg,rows)}
        ${capNote}
      </div></details>`;
}
/* Graded counts by what each forecast WAS (policy rules 1 and 2): picks are
   CONFIRMED/PROJECTED forecasts outside 55/45, leans the ones inside it, EARLY
   forecasts neither. MLB's come from the server's uncapped by_tier block - the
   payload ships only the most recent rows, so counting them undercounts. */
function recCounts(lg){
  if(lg==="mlb"){const m=recMlb()||{}, bt=m.by_tier||{}, PT=["CONFIRMED","PROJECTED"];
    const sum=k=>PT.reduce((a,t)=>a+(((bt[t]||{})[k]||{}).n||0),0);
    const rows=recRows("mlb");
    return {graded:m.rollup&&m.rollup.n!=null?m.rollup.n:rows.length, picks:sum("pick"), leans:sum("lean"),
      early:(bt.EARLY&&bt.EARLY.n)||0, shipped:rows.length};}
  const rated=recRated(recRows(lg)), early=rated.filter(r=>r.tier==="EARLY").length;
  const picks=rated.filter(r=>r.tier!=="EARLY"&&!recIsLean(r.p)).length;
  return {graded:rated.length, picks, leans:rated.length-early-picks, early, shipped:rated.length};
}
/* Rail badge on this page: GRADED PICKS - never a lean, never an EARLY
   forecast. A league with graded forecasts and no graded pick shows its
   forecast count, styled EARLY, so the badge is informative without ever
   counting a pre-information lean as a pick. */
function recBadge(lg){
  const c=recCounts(lg), pl=(k,w)=>`${k} ${w}${k===1?"":"s"}`;
  const parts=`${pl(c.picks,"graded pick")} · ${pl(c.leans,"lean")} (inside 55/45) · ${pl(c.early,"EARLY-tier forecast")} (not picks)`;
  if(lg==="nhl"||(!c.picks&&c.graded)) return {txt:c.graded,cls:"early",
    title:`${parts} - the badge counts graded forecasts, none of them a pick`
      +(lg==="nhl"?"; every NHL forecast is EARLY: team ratings only":"")};
  return {txt:c.picks,title:parts};
}
function recordPage(){
  const lg=recLeague();
  $("#view").innerHTML=`<div class="controls"><div class="rail">${siteRail(lg,recBadge)}</div></div>
    <div class="eyebrow">Track record &middot; ${REC_NAME[lg]}</div>
    <h1 class="pt">Every pick, graded <span class="sub">locked pre-game, settled on the final</span></h1>
    <div class="sub" style="margin-bottom:4px">Market-blind model, graded before/after &mdash; nothing back-filled.
      Accuracy is split by <b>information tier</b>, never pooled into one headline.</div>
    ${recFreshLine(lg)}
    ${recSection(lg)}`;
  // the league lives in the URL: a rail click is a navigation, so Back works
  $("#view").querySelectorAll(".rail [data-lg]").forEach(x=>x.onclick=()=>{location.hash=siteHref(x.dataset.lg,"record");});
}

boot();
