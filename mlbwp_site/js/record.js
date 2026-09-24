/* Track record: every league's graded history, tiers, units. */
const REC_ROWS=300;                 // rows rendered; rollups always use every graded pick
const REC_LG=["mlb","nfl","nhl"];
const REC_NAME={mlb:"MLB",nfl:"NFL",nhl:"NHL"};
const LL_COIN=Math.log(2);          // 0.6931 - the coin-flip log loss
/* documents/pick_policy.md. Information tiers are the honest unit: a forecast
   made 30 days out with team ratings only is a different product from one made
   with both lineups posted, and the two are never pooled into a headline.

   The tier is decided by the policy's "what the model actually has" column, per
   league - NOT by which league it is. MLB stamps it server-side
   (mlbwp/pred_ledger.py). For the other two:
     NHL - site/data/nhl.json model_card features are Elo, rest, back-to-back and
       an xG TEAM rating. No roster, no lineup, no goalie. That is team ratings
       only, i.e. EARLY, and the same chip therefore means the same thing on
       every league's page. It becomes PROJECTED when a lineup/goalie input
       actually ships in the model, not when the archive merely exists.
     NFL - the payload carries live depth charts and a lineup-power rating, so a
       forecast served by the live model (inside a week, nflProb().near) is
       PROJECTED. Outside that window the number is the 20k-run season
       simulation off team strength alone - EARLY, exactly like a 30-days-out
       MLB game. A Week 18 game in September is not a pick. */
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

function recRows(lg){
  if(lg==="mlb"){const m=recMlb(); if(!m) return [];
    return (m.rows||[]).map(r=>({d:r.d,away:r.away,home:r.home,p:r.p,pick:r.pick,y:r.y,
      hs:r.hs,as:r.as,sp:r.sp,tier:recTierOf(r),per:(r.d||"").slice(0,7),href:null}));}
  if(lg==="nfl"){const n=state.nfl; if(!n||!n.schedule) return [];
    return n.schedule.filter(g=>g.hs!=null&&g.as!=null&&g.ph!=null)
      .map(g=>({d:g.d,away:g.away,home:g.home,p:g.ph,pick:g.ph>=0.5?g.home:g.away,
        y:g.hs>g.as?1:(g.hs<g.as?0:null),hs:g.hs,as:g.as,tier:"PROJECTED",per:"Wk "+g.w,
        href:"#/game/"+g.w+"_"+g.away+"_"+g.home}))
      .sort((a,b)=>a.d<b.d?1:(a.d>b.d?-1:0));}
  if(lg==="nhl"){const n=state.nhl; if(!n||!n.schedule) return [];
    return n.schedule.filter(g=>g.hs!=null&&g.as!=null&&g.hp!=null)
      .map(g=>({d:g.d,away:g.away,home:g.home,p:g.hp,pick:g.hp>=0.5?g.home:g.away,
        y:g.y!=null?g.y:(g.hs>g.as?1:0),hs:g.hs,as:g.as,tier:"EARLY",
        per:(g.d||"").slice(0,7),href:"#/game/nhl-"+g.id}))
      .sort((a,b)=>a.d<b.d?1:(a.d>b.d?-1:0));}
  return [];
}
const recPicks=lg=>recRows(lg).filter(r=>r.tier!=="EARLY");   // graded PICKS only
/* Per-tier scores. MLB ships them from the server (computed over EVERY graded
   row, not just the shipped page); NFL/NHL are scored client-side. */
/* Policy rule 2 inside rule 3: a graded LEAN is not a graded PICK, so each tier
   carries BOTH - `pick` (the tier's headline rate, leans excluded) and `lean`
   (reported beside it, never folded into it). */
const recSrv=o=>o?{n:o.n,correct:Math.round(o.acc*o.n),acc:o.acc,ll:o.log_loss,
  brier:o.brier,ties:0}:null;
function recByTier(lg){
  const out={};
  if(lg==="mlb"){const m=recMlb(), bt=(m&&m.by_tier)||{};
    REC_TIERS.forEach(t=>{const o=bt[t]; if(o) out[t]={...recSrv(o),
      pick:recSrv(o.pick),lean:recSrv(o.lean)};});
    return out;}
  const rows=recRows(lg);
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
/* Every ungraded forecast, tier-stamped. recPending / recScheduled split it on
   rule 1: PICKS are CONFIRMED + PROJECTED, everything EARLY is a scheduled
   game. MLB arrives already split (and capped) from the server. */
function recUngraded(lg){
  if(lg==="nfl"){const n=state.nfl; if(!n||!n.schedule) return [];
    return n.schedule.filter(g=>g.hs==null&&g.ph!=null)
      .map(g=>({d:g.d,away:g.away,home:g.home,p:g.ph,pick:g.ph>=0.5?g.home:g.away,
        sp:1,tier:nflProb(g).near?"PROJECTED":"EARLY",
        href:"#/game/"+g.w+"_"+g.away+"_"+g.home}))
      .sort((a,b)=>a.d<b.d?-1:(a.d>b.d?1:0));}
  if(lg==="nhl"){const n=state.nhl; if(!n||!n.schedule) return [];
    return n.schedule.filter(g=>g.hs==null&&g.hp!=null)
      .map(g=>({d:g.d,away:g.away,home:g.home,p:g.hp,pick:g.hp>=0.5?g.home:g.away,
        sp:1,tier:"EARLY",href:"#/game/nhl-"+g.id}))
      .sort((a,b)=>a.d<b.d?-1:(a.d>b.d?1:0));}
  return [];
}
function recPending(lg){            // locked-in PICKS whose games are not graded yet
  if(lg==="mlb"){const m=recMlb(); if(!m) return [];
    return (m.pending||[]).map(r=>({d:r.d,away:r.away,home:r.home,p:r.p,pick:r.pick,
      sp:r.sp,spp:r.spp,tier:recTierOf(r),href:null}));}
  return recUngraded(lg).filter(r=>r.tier!=="EARLY");
}
/* EARLY games: scheduled, NOT picks. Shown collapsed and labelled
   pre-information. */
function recScheduled(lg){
  if(lg!=="mlb") return recUngraded(lg).filter(r=>r.tier==="EARLY");
  const m=recMlb(); if(!m) return [];
  return (m.scheduled||[]).map(r=>({d:r.d,away:r.away,home:r.home,p:r.p,pick:r.pick,
    sp:r.sp,spp:r.spp,tier:"EARLY",href:null}));
}
/* The lean count must come from the server's uncapped total when it ships one:
   counting leans in the capped `pending` array and subtracting from the
   uncapped n_pending makes the two halves stop summing to the stated total. */
function recLeanTotal(lg){
  const m=lg==="mlb"?recMlb():null;
  return m&&m.n_lean!=null?m.n_lean:recPending(lg).filter(r=>recIsLean(r.p)).length;
}
/* Cumulative-accuracy curve: the picture the tables cannot draw. Plots running
   pick accuracy against the 50% line so drift is visible at a glance. Pure
   inline SVG - no library, themes via currentColor/CSS vars. */
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
        preserveAspectRatio="none" role="img" aria-label="Cumulative pick accuracy">
      ${ticks}
      ${(y50>PT&&y50<H-PB)?`<line x1="${PL}" y1="${y50.toFixed(1)}" x2="${W-PR}" y2="${y50.toFixed(1)}"
        stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3" opacity=".8"/>
        <text x="${W-PR}" y="${(y50-4).toFixed(1)}" text-anchor="end" font-size="9"
          fill="var(--muted)">coin flip</text>`:""}
      <path d="${area}" fill="var(--accent)" opacity=".10"/>
      <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="${X(last.x).toFixed(1)}" cy="${Y(last.acc).toFixed(1)}" r="3.5" fill="var(--accent)"/>
      <text x="${PL}" y="${H-5}" font-size="9" fill="var(--muted)">pick 1</text>
      <text x="${W-PR}" y="${H-5}" text-anchor="end" font-size="9" fill="var(--muted)">pick ${pts.length}</text>
    </svg></div></div>`;
}
/* ---------- UNITS (#/record) ----------
   Units track the EDGE-BADGE BETS and nothing else (decision 2026-08-12):
   a flat 1u on the badge side at the price RECORDED when the badge fired
   (market/edge_ledger.py — the ledger rows are the bet slips), settled on the
   result. Model picks without a badge carry NO stake: they are scored on
   accuracy/log loss in the tables below, never in units. The block ships as
   board.record.bets, computed server-side over every ledger row. */
const fmtU=v=>(v>0?"+":(v<0?"&minus;":""))+Math.abs(v).toFixed(2)+"u";
/* MLB ledger rows carry Retrosheet team codes; show the everyday abbrs. */
const RETRO2ABBR={SLN:"STL",CHA:"CWS",NYN:"NYM",NYA:"NYY",LAN:"LAD",SDN:"SD",
  SFN:"SF",CHN:"CHC",KCA:"KC",ANA:"LAA",TBA:"TB",WAS:"WSH"};
const abbrOf=c=>RETRO2ABBR[c]||c;
const recBets=()=>((state.board||{}).record||{}).bets||null;
const recUnits=lg=>{const b=recBets();return b&&b.leagues&&b.leagues[lg]?b.leagues[lg]:null;};
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
    <text x="${PL}" y="${H-5}" font-size="9" fill="var(--muted)">bet 1 &middot; ${u.priced_since||""}</text>
    <text x="${W-PR}" y="${H-5}" text-anchor="end" font-size="9" fill="var(--muted)">bet ${pts.length} &middot; ${fmtU(last.v)}</text>
  </svg></div>`;
}
function recUnitsPanel(lg){
  const head=`<h3>Units <span class="sub">EDGE bets only &mdash; flat 1u per badge at the recorded price</span></h3>`;
  const u=recUnits(lg), all=recBets();
  const open=all?(all.pending||[]).filter(p=>p.league===lg):[];
  if(!u||(!u.settled&&!open.length))
    return `<div class="panel" style="margin-top:16px">${head}
      <div class="empty">No ${REC_NAME[lg]} EDGE bet yet. A unit is staked only when a game carries the value
        badge (model vs opening consensus above the EV threshold) &mdash; picks without a badge are scored on
        accuracy below and are never staked.</div></div>`;
  const s=u.settled;
  const box=(k,v,cls)=>`<div class="b"><div class="k">${k}</div><div class="v${cls?" "+cls:""}">${v}</div></div>`;
  const tiles=s?[
    box("Record",`${s.w}&ndash;${s.l}`),
    box("Units net",fmtU(s.net),s.net>=0?"pos":"neg"),
    box("ROI",(s.roi>=0?"+":"&minus;")+Math.abs(s.roi*100).toFixed(1)+"%",s.roi>=0?"pos":"neg"),
    box("Staked",s.staked.toFixed(0)+"u"),
    box("Avg price",s.avg_dec.toFixed(2)),
  ].join(""):"";
  const obody=open.map(p=>`<tr>
    <td class="a"><span class="sub">${(p.d||"").slice(5)}</span></td>
    <td class="a">${abbrOf(p.away)} <span class="sub">@</span> ${abbrOf(p.home)}</td>
    <td class="a"><span class="ab" style="color:var(--accent)">${p.team}</span></td>
    <td><span class="num">${p.dec!=null?p.dec.toFixed(2):"&mdash;"}</span></td>
    <td><span class="num">${p.ev!=null?"+"+(p.ev*100).toFixed(1)+"%":"&mdash;"}</span></td></tr>`).join("");
  const openTbl=open.length?`<div class="subh" style="margin-top:12px">Open bets <span class="sub"
      style="font-weight:400;font-family:var(--sans)">1u each, locked at the recorded price</span></div>
    <div class="twrap"><table>
      <thead><tr><th>Date</th><th>Matchup</th><th>Bet</th><th>Price</th><th>EV @ record</th></tr></thead>
      <tbody>${obody}</tbody></table></div>`:"";
  const months=(u.by_month||[]).slice().reverse().map(o=>`<tr>
    <td class="a">${recMonth(o.k)}</td><td><span class="num">${o.n}</span></td>
    <td><span class="num ${o.net>=0?"pos":"neg"}">${fmtU(o.net)}</span></td>
    <td><span class="num ${o.cum>=0?"pos":"neg"}">${fmtU(o.cum)}</span></td></tr>`).join("");
  const monthTbl=months?`<details class="sched" style="margin-top:10px"><summary>By month</summary><div class="body">
    <div class="twrap"><table>
      <thead><tr><th>Month</th><th>Bets</th><th>Units</th><th>Cum units</th></tr></thead>
      <tbody>${months}</tbody></table></div></div></details>`:"";
  const thin=s&&s.n<30?`<div class="sub" style="margin-top:8px"><b>n=${s.n} settled &mdash; not yet meaningful.</b>
    Units swing hard at this sample.</div>`:"";
  return `<div class="panel" style="margin-top:16px">${head}
    ${s?`<div class="statgrid" style="margin-top:10px;grid-template-columns:repeat(auto-fit,minmax(110px,1fr))">${tiles}</div>`
      :`<div class="empty">Tracking starts here &mdash; ${open.length} open bet${open.length===1?"":"s"}, none settled yet.</div>`}
    ${recUnitsChart(u)}
    ${thin}
    ${openTbl}
    ${monthTbl}
    <div class="sub" style="margin-top:10px">Every unit is a bet the <b>EDGE badge</b> fired and the site marked
      takeable: 1u on the badge side at the consensus price <b>recorded at that moment</b> (never re-priced); a win
      banks price&nbsp;&minus;&nbsp;1, a loss costs 1. Picks without a badge are never staked. This is a live
      evaluation of the badge, not a profit promise.${u.n_void?` ${u.n_void} bet${u.n_void===1?"":"s"} on
      postponed/cancelled games ${u.n_void===1?"was":"were"} <b>voided</b> &mdash; stake returned, counted nowhere.`:""}</div></div>`;
}
function recSource(lg){
  const m=recMlb();
  if(lg==="mlb") return `Every pick is archived <b>before first pitch</b> and graded from the final score &mdash;
    nothing back-filled. Tracking began <b>${(m&&m.since)||"&mdash;"}</b>; what gets graded is the last, most-informed
    pre-game forecast, and the tier stamp records which it was.`;
  if(lg==="nfl") return `Picks are the frozen pre-game probabilities in the NFL payload, scored against the final.
    Only games inside a week of kickoff are published as picks; the rest is the season simulation (scheduled, not picks).`;
  return `Picks are the frozen pre-game probabilities in the NHL payload (leak-safe replay). The shipped model has no
    roster, lineup or goalie input, so every NHL forecast is EARLY &mdash; published for transparency, not counted as picks.`;
}
/* Policy rule 3: accuracy and log loss are reported PER TIER and never pooled
   into a headline. A tier with too few graded rows shows a dash (with the
   reason in its tooltip) instead of printing a number nobody should read. */
function recTierGrid(lg){
  const bt=recByTier(lg);
  // Only tiers the league can actually produce: MLB all three, NFL live-model
  // + simulation, NHL team ratings only.
  const applies=lg==="mlb"?REC_TIERS:(lg==="nfl"?["PROJECTED","EARLY"]:["EARLY"]);
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
    <thead><tr><th>Tier</th><th>Graded</th><th>Accuracy</th><th>LL vs coin</th><th></th></tr></thead>
    <tbody>${trs}</tbody></table></div>`;
}
function recSection(lg){
  const rows=recRows(lg), srv=lg==="mlb"?(recMlb()||{}).rollup:null;
  const s=srv?{n:srv.n,correct:Math.round(srv.acc*srv.n),acc:srv.acc,ll:srv.log_loss,
               brier:srv.brier,ties:0}:recScore(rows);
  // confidence bar: how far the pick sits from a coin flip, so the eye can scan
  // conviction down the column instead of reading every percentage
  const conf=p=>{const c=Math.max(p,1-p);
    return `<span class="cbar"><i style="width:${((c-0.5)*200).toFixed(0)}%"></i></span>
      <span class="num">${pctI(c)}%</span>`;};
  const tierChip=t=>`<span class="chip t-${t}">${t}</span>`;
  // Policy rule 2: inside 55/45 the model is not claiming much - say so.
  const callChip=p=>recIsLean(p)?`<span class="chip lean">LEAN</span>`:`<span class="chip pick">PICK</span>`;

  // The long per-league tier explanation lives behind a click; the numbers don't.
  const tiers=`<div class="panel" style="margin-top:16px"><h3>Accuracy by tier
      <span class="sub">never pooled &mdash; different information, different product</span></h3>
    ${recTierGrid(lg)}
    <details class="sched" style="margin-top:8px"><summary>What the tiers mean</summary>
      <div class="body"><div class="sub">Tiers describe what the model actually had when the graded forecast was
      made. ${lg==="mlb"
        ?`<b>CONFIRMED</b> the official lineup card is posted, <b>PROJECTED</b> both starters known with a projected
          lineup, <b>EARLY</b> team ratings only. The tier is the state at the <i>last pre-game refresh</i> (every
          4h), so CONFIRMED is a start-time-selected subset of lineups-posted games &mdash; do not read the tiers
          as a controlled comparison.`
        :lg==="nfl"
        ?`<b>PROJECTED</b> is the live model (ratings, depth charts, lineup power) inside a week of kickoff; beyond
          that the number is the season simulation off team strength alone &mdash; <b>EARLY</b>, not a pick. No
          confirmed-inactives feed, so no CONFIRMED tier.`
        :`The shipped NHL model has no roster, lineup or goalie input, so every forecast is <b>EARLY</b> &mdash;
          team ratings only. A PROJECTED tier arrives when a lineup or goalie feature actually ships in the model.`}
      </div></div></details></div>`;

  // PENDING = picks only (CONFIRMED + PROJECTED). EARLY rides in its own block.
  const pend=recPending(lg), pshown=pend.slice(0,REC_ROWS);
  const nLean=recLeanTotal(lg), nPick=recPendTotal(lg)-nLean;
  const pbody=pshown.map(r=>`<tr class="recrow" ${r.href?`onclick="location.hash='${r.href}'"`:""}>
    <td class="a"><span class="sub">${(r.d||"").slice(5)}</span></td>
    <td class="a">${r.away} <span class="sub">@</span> ${r.home}</td>
    <td class="a">${tierChip(r.tier)}${r.spp?' <span class="sub" style="font-size:10px" title="no probable announced yet — the starter is this model\'s own rotation projection">SP proj</span>':""}</td>
    <td class="a"><span class="ab" style="color:var(--accent)">${r.pick}</span></td>
    <td class="cf">${conf(r.p)}</td>
    <td><span class="num">${amOdds(Math.max(r.p,1-r.p))}</span></td>
    <td>${callChip(r.p)}</td></tr>`).join("");
  const pendPanel=`<div class="panel"><h3>Pending picks
      <span class="sub">${pend.length?`${pshown.length<recPendTotal(lg)?`next ${pshown.length} of ${recPendTotal(lg)}`:`all ${pend.length}`} &middot; locked in, awaiting the result`:"none right now"}</span></h3>
    ${pend.length?`<div class="twrap"><table>
        <thead><tr><th>Date</th><th>Matchup</th><th>Tier</th><th>Pick</th><th>Confidence</th><th>Fair odds</th><th>Call</th></tr></thead>
        <tbody>${pbody}</tbody></table></div>
      <div class="sub" style="margin-top:8px">${nPick} pick${nPick===1?"":"s"} + ${nLean}
        lean${nLean===1?"":"s"} (inside 55/45), written down <b>before</b> the games are played &mdash;
        every one grades HIT or MISS.</div>`
    :`<div class="empty">No picks waiting on a result${lg==="nfl"?" &mdash; the 2026 season has not started.":"."}</div>`}</div>`;

  // SCHEDULED = EARLY tier. Collapsed, labelled pre-information, not counted.
  const sch=recScheduled(lg), sshown=sch.slice(0,REC_ROWS);
  const sbody=sshown.map(r=>`<tr class="recrow">
    <td class="a"><span class="sub">${(r.d||"").slice(5)}</span></td>
    <td class="a">${r.away} <span class="sub">@</span> ${r.home}</td>
    <td class="a"><span class="ab">${r.pick}</span></td>
    <td class="cf">${conf(r.p)}</td>
    <td><span class="chip t-EARLY">PRE-INFO</span></td></tr>`).join("");
  const schPanel=sch.length?`<details class="sched"><summary>Scheduled games &mdash; not picks
      <span class="sub">${recSchedTotal(lg)} EARLY-tier games on the board</span></summary>
    <div class="body">
      <div class="sub" style="margin-bottom:10px">${lg==="mlb"
        ?`No starter announced yet &mdash; team ratings only.`
        :lg==="nfl"
        ?`More than a week out &mdash; season simulation off team strength only.`
        :`The shipped NHL model has no roster, lineup or goalie input &mdash; team ratings only.`}
        The lean is what the model currently thinks, published for transparency and <b>excluded from the pick
        count</b>.</div>
      <div class="twrap"><table>
        <thead><tr><th>Date</th><th>Matchup</th><th>Lean</th><th>Confidence</th><th>Information</th></tr></thead>
        <tbody>${sbody}</tbody></table></div>
      ${sshown.length<recSchedTotal(lg)?`<div class="sub" style="margin-top:8px">Showing the next ${sshown.length}
        of ${recSchedTotal(lg)}.</div>`:""}
    </div></details>`:"";

  if(!s){                     // nothing graded yet - the pending picks are still receipts
    return `<div class="empty" style="margin-top:12px">No ${REC_NAME[lg]} forecasts have been graded yet.<br>
        <span class="sub">${recSource(lg)}</span></div>
      ${recUnitsPanel(lg)}
      ${tiers}
      <div style="margin-top:16px">${pendPanel}</div>
      ${schPanel}`;
  }
  const warn=s.n<30?`<div class="sub" style="margin-top:12px"><b>n=${s.n} &mdash; not yet meaningful.</b>
    Treat this page as a receipt that the picks were made, not as evidence.</div>`:"";
  const per=recPeriods(rows).reverse().map(o=>`<tr>
    <td class="a">${recMonth(o.k)}</td><td><span class="num">${o.n}</span></td>
    <td><span class="num">${o.c}</span></td>
    <td><span class="num">${o.acc==null?"&mdash;":(o.acc*100).toFixed(1)+"%"}</span></td>
    <td><span class="num">${o.ll==null?"&mdash;":o.ll.toFixed(4)}</span></td>
    <td><span class="num">${o.cacc==null?"&mdash;":(o.cacc*100).toFixed(1)+"%"}</span></td>
    <td><span class="num">${o.cll==null?"&mdash;":o.cll.toFixed(4)}</span></td></tr>`).join("");
  const shown=rows.slice(0,REC_ROWS);
  const outcome=r=>{
    if(r.y==null) return `<td><span class="num">${r.as}&ndash;${r.hs}</span></td><td><span class="chip">TIE</span></td>`;
    const hit=(r.p>=0.5)===(r.y===1);
    return `<td><span class="num">${r.as}&ndash;${r.hs}</span></td>
      <td><span class="chip ${hit?"hit":"miss"}">${hit?"HIT":"MISS"}</span></td>`;};
  const body=shown.map(r=>`<tr class="recrow" ${r.href?`onclick="location.hash='${r.href}'"`:""}>
    <td class="a"><span class="sub">${(r.d||"").slice(5)}</span></td>
    <td class="a">${r.away} <span class="sub">@</span> ${r.home}</td>
    <td class="a">${tierChip(r.tier)}</td>
    <td class="a">${callChip(r.p)}</td>
    <td class="a"><span class="ab" style="color:var(--accent)">${r.pick}</span></td>
    <td class="cf">${conf(r.p)}</td>
    ${outcome(r)}</tr>`).join("");
  const nGPick=rows.filter(r=>r.tier!=="EARLY"&&!recIsLean(r.p)).length;
  // One glance: units P&L, then per-tier accuracy, then the pending receipts.
  // Everything else (full graded listing, monthly rollups, accuracy curve,
  // methodology) is real but secondary — it lives behind one click.
  return `${warn}
    ${recUnitsPanel(lg)}
    ${tiers}
    <div style="margin-top:16px">${pendPanel}</div>
    ${schPanel}
    <details class="sched" style="margin-top:16px"><summary>Full history &mdash; every graded forecast
        <span class="sub">${rows.length} graded &middot; ${nGPick} picks &middot; ${recPendTotal(lg)} pending</span></summary>
      <div class="body">
        <div class="sub" style="margin-bottom:10px">${recSource(lg)}</div>
        ${recChart(rows)}
        <div class="twrap" style="margin-top:10px"><table>
          <thead><tr><th>Date</th><th>Matchup</th><th>Tier</th><th>Call</th><th>Pick</th><th>Confidence</th><th>Final</th><th>Result</th></tr></thead>
          <tbody>${body}</tbody></table></div>
        <div class="sub" style="margin-top:8px">${shown.length<rows.length?`Most recent ${shown.length} of ${rows.length}. `:""}This
          listing pools every tier and both calls &mdash; a <span class="chip lean">LEAN</span> or
          <span class="chip t-EARLY">EARLY</span> row is never inside any pick rate on this page.</div>
        <div class="subh" style="margin-top:14px">By ${lg==="nfl"?"week":"month"}</div>
        <div class="twrap"><table>
          <thead><tr><th>${lg==="nfl"?"Week":"Month"}</th><th>Picks</th><th>Hits</th><th>Acc</th><th>LL</th><th>Cum acc</th><th>Cum LL</th></tr></thead>
          <tbody>${per}</tbody></table></div>
      </div></details>`;
}
function recordPage(){
  const lg=recLeague(), b=state.board;
  const rail=b.leagues.map(l=>{
    if(l.code==="nhl"&&state.nhl) return "";        // the live NHL button is appended below
    // Rule 1: the badge on a page titled "Every pick, graded" counts graded
    // PICKS. EARLY rows are graded forecasts but were never published as picks.
    if(l.code==="mlb"||(l.code==="nfl"&&state.nfl))
      return `<button class="lg ${lg===l.code?"on":""}" data-lg="${l.code}" title="graded picks (EARLY-tier forecasts excluded)"><span>${l.name}</span><span class="n">${recPicks(l.code).length}</span></button>`;
    return `<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`;}).join("")
    +(state.nhl?`<button class="lg ${lg==="nhl"?"on":""}" data-lg="nhl" title="graded picks (EARLY-tier forecasts excluded)"><span>NHL</span><span class="n">${recPicks("nhl").length}</span></button>`:"");
  $("#view").innerHTML=`<div class="controls"><div class="rail">${rail}</div></div>
    <div class="eyebrow">Track record &middot; ${REC_NAME[lg]}</div>
    <h1 class="pt">Every pick, graded <span class="sub" style="font-weight:400">locked pre-game, settled on the final</span></h1>
    <div class="sub" style="margin-bottom:4px">Market-blind model, graded before/after &mdash; nothing back-filled.
      Accuracy is split by <b>information tier</b>, never pooled into one headline.</div>
    ${recSection(lg)}`;
  $("#view").querySelectorAll("[data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);recordPage();});
}

boot();

