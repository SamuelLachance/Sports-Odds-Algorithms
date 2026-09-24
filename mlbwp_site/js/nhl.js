/* NHL: board, cards, game page, teams, team page, players, player page, standings.

   documents/pick_policy.md. The shipped NHL model runs on team ratings only -
   Elo, an expected-goals team rating, rest and back-to-backs - with no goalie,
   lineup or roster input, so every NHL forecast is EARLY tier. An EARLY
   forecast is a SCHEDULED game with a labelled lean: no pick pill, no fair
   odds, and a played game grades the lean, never a pick. The tier is read off
   the row (siteTier), so a row the serve ever stamps PROJECTED gets MLB's
   PICK/LEAN treatment with no change here. Player and lineup ratings (RAPM for
   skaters, GSAx for goalies) are display metrics and never model inputs. */

if(state.nhlPos==null) Object.assign(state,{nhlPos:"all",nhlMin:0,nhlStd:null,nhlStdView:"div",
  nhlTeamSort:"elo",nhlResN:14});

/* ---------- small helpers ---------- */
const NHL_POSL={C:"C",L:"LW",R:"RW",D:"D",G:"G"};
const NHL_CLINCH={x:"clinched a playoff spot",y:"clinched the division",z:"clinched the conference",
  p:"Presidents' Trophy",e:"eliminated"};
const NHL_POSF={C:"Centre",L:"Left wing",R:"Right wing",D:"Defence",G:"Goalie"};
const nhlPosL=p=>NHL_POSL[p&&p.pos]||(p&&p.pos)||"";
const nhlTeam=c=>(state.nhl&&state.nhl.teams&&state.nhl.teams[c])||null;
const nhlName=c=>{const t=nhlTeam(c); return siteEsc((t&&t.name)||nhlCity(c));};
const nhlNick=c=>{const t=nhlTeam(c); return siteEsc((t&&t.nick)||nhlCity(c));};
const nhlH=(v,arg)=>siteHref("nhl",v,arg);
const nhlGameHref=g=>"#/game/nhl-"+g.id;
const nhlDone=g=>g.hs!=null&&g.as!=null;
const nhlOT=g=>g.last&&g.last!=="REG"?` (${g.last})`:"";
// links inside clickable rows must not also fire the row's own navigation
const nhlTL=(c,label)=>teamLink(c,label||c,"nhl").replace("<a ",'<a onclick="event.stopPropagation()" ');
const nhlPL=p=>`<a class="player-link" href="${nhlH("player",p.id)}" onclick="event.stopPropagation()">${siteEsc(p.name)}</a>`;
const nhlRow=href=>`onclick="location.hash='${href}'"`;
/* Signed, at the precision the quantity needs, with a true minus sign and no
   "-0.00": skater xG/60 values mostly sit inside +-0.3, where one decimal
   collapsed 558 skaters into ten strings. */
function nhlSg(v,d){ if(v==null||!isFinite(v)) return "&mdash;"; d=d==null?2:d;
  const a=Math.abs(v).toFixed(d); return +a===0?a:(v>0?"+":"&minus;")+a;}
function nhlSgN(v,d,faint){ if(v==null||!isFinite(v)) return `<span class="num sub">&mdash;</span>`;
  d=d==null?2:d; const z=+Math.abs(v).toFixed(d)===0;
  return `<span class="num ${faint||z?"sub":(v>0?"pos":"neg")}">${nhlSg(v,d)}</span>`;}
const nhlPct3=v=>v==null?"&mdash;":(v>=1?"1.000":(+v).toFixed(3).replace(/^0/,""));
const nhlSv=v=>v==null?"-":(+v).toFixed(3).replace(/^0/,"");
const nhlMMSS=s=>{if(s==null||!isFinite(s)) return "-"; s=Math.round(s); return Math.floor(s/60)+":"+String(s%60).padStart(2,"0");};
const nhlInt=v=>v==null?"-":Math.round(v).toLocaleString(LOC);
const nhlPM=v=>v==null?"-":(v>0?"+"+v:(v<0?"&minus;"+Math.abs(v):"0"));
const nhlShortDay=d=>new Date(d+"T12:00:00Z").toLocaleDateString(LOC,{weekday:"short",month:"short",day:"numeric",timeZone:TZ});
const nhlWhen=g=>fmtDay(g.d,nflToday())+(g.start_utc?" &middot; "+fmtTime(g.start_utc):"");
const nhlEtStamp=iso=>{const t=iso?Date.parse(iso):NaN;
  return Number.isFinite(t)?siteEt(t,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})+" ET":null;};
const nhlDateOnly=d=>d?new Date(d+"T12:00:00Z").toLocaleDateString(LOC,{month:"short",day:"numeric",year:"numeric",timeZone:TZ}):"";
/* Simulated share in %, never "0%"/"100%" off a finite simulation; an official
   clinch or elimination mark (NHL API) overrides it. */
function nhlOdds(v,cl){
  if(cl==="e") return "out";
  if(cl&&"xyzp".indexOf(cl)>=0) return "clinched";
  if(v==null||!isFinite(v)) return "&mdash;";
  if(v>=99.5) return "&gt;99%"; if(v<0.5) return "&lt;1%";
  return Math.round(v)+"%";}
const nhlSL=id=>{const s=String(id||""); return s.length===8?s.slice(0,4)+"&ndash;"+s.slice(6):"";};
const nhlPrevLbl=()=>{const n=state.nhl||{};
  return nhlSL((n.stats_seasons&&n.stats_seasons.prev)||(n.prev_season&&n.prev_season.season))||"last season";};
const nhlStarted=()=>Object.values((state.nhl&&state.nhl.teams)||{}).some(t=>(t.gp||0)>0);
/* The record a team is read by: this season's once it has played, else last
   season's final line, labelled as such - never a wall of 0-0-0. */
function nhlRecOf(t){ if(!t) return null;
  if((t.gp||0)>0) return {r:t,cur:true,lbl:nhlSeasLbl()};
  if(t.prev&&t.prev.gp) return {r:t.prev,cur:false,lbl:nhlPrevLbl()};
  return null;}
const nhlWLO=r=>r?`${r.w}-${r.l}-${r.otl}`:"-";
/* A player's season line: `key` ("cur"/"prev") or his main one. */
function nhlLine(p,key){const s=(p&&p.stats)||{}, k=key||s.main||(s.cur?"cur":(s.prev?"prev":null));
  return k&&s[k]?{ln:s[k],key:k,lbl:k==="cur"?nhlSeasLbl():nhlPrevLbl()}:null;}
/* Rating badge: gb() for a rating; an unrated player gets a muted NR whose
   tooltip is the payload's reason in words (never a red "bad" colour). */
function nhlGb(p){ if(p&&p.rating!=null) return gb(p.rating);
  return `<span class="gb" title="Not rated: ${siteEsc((p&&p.nr)||"no rating")}"><span class="v" style="color:var(--faint)">NR</span></span>`;}
/* Per-payload indexes: xG ranks and rated-player counts. */
let nhlMemoKey=null, nhlMemoVal=null;
function nhlIdx(){const n=state.nhl; if(nhlMemoKey===n&&nhlMemoVal) return nhlMemoVal;
  const T=n.teams||{}, cs=Object.keys(T), xgR={};
  cs.slice().sort((a,b)=>(T[b].xg||0)-(T[a].xg||0)).forEach((c,i)=>{xgR[c]=i+1;});
  const P=Object.values(n.players||{});
  const sk=P.filter(p=>p.grp!=="G"&&p.rating!=null), nF=sk.filter(p=>p.grp==="F").length;
  nhlMemoVal={xgR,nSk:sk.length,nF,nD:sk.length-nF,nG:P.filter(p=>p.grp==="G"&&p.rating!=null).length,nT:cs.length};
  nhlMemoKey=n; return nhlMemoVal;}

/* ---------- policy: tier, call, grade ---------- */
function nhlCall(g){
  const hp=g.hp, pick=hp>=0.5?g.home:g.away, pp=Math.max(hp,1-hp), t=siteTier("nhl",g);
  // one rounding: the away side is 100 minus the rounded home side, so the two
  // printed percentages always add to 100 (hp 0.415 read 59% + 42% before)
  const pH=pctI(hp), pA=100-pH;
  return {hp,pick,pp,pH,pA,pPk:hp>=0.5?pH:pA,tier:t,early:t==="EARLY",call:t==="EARLY"?"EARLY":callOf(pp)};}
/* A value badge shows only where the edge layer cleared it for the row's tier:
   any non-EARLY row, or an EARLY row the edge layer disclosed on purpose
   (market/sched_edges with GATE_EARLY off stamps value.tier = "EARLY"). A
   badge priced before the EARLY gate carries no tier - and was priced on an
   older forecast - so it never shows on an EARLY row. */
const nhlBadge=g=>{const v=g&&g.value; return !!(v&&v.available&&(siteTier("nhl",g)!=="EARLY"||v.tier==="EARLY"));};
/* Postponed / cancelled (nhl_serve stamps ppd:1 and the NHL API state). */
function nhlOff(g){const s=String((g&&g.state)||"").toUpperCase();
  if(s==="CNCL"||s==="CANCELLED") return "Cancelled";
  return g&&(g.ppd||s==="PPD"||s==="POSTPONED")?"Postponed":null;}
/* Date, then puck drop, then id: a consistent comparator (the old nested
   ternary returned id order whenever a started later than b). */
const nhlCmp=(a,b)=>String(a.d).localeCompare(String(b.d))||String(a.start_utc||"").localeCompare(String(b.start_utc||""))||(a.id-b.id);
/* core's pRow with a true minus sign and no "+0.0". */
function nhlPRow(label,v){const w=Math.min(Math.abs(v)*3.0,50), neg=v<0;
  return `<div class="r"><span class="lab">${label}</span>
    <div class="tr"><div class="f ${neg?"neg":""}" style="${neg?"right:50%":"left:50%"};width:${w}%"></div></div>
    <span class="v ${neg?"neg":"pos"}">${nhlSg(v,1)}</span></div>`;}
const NHL_EARLY_WHY="EARLY tier: team ratings only - no goalie, lineup or roster input. Published as a lean, not a pick.";
const NHL_SCHED_PILL=`<span class="pill early" title="${NHL_EARLY_WHY}">SCHEDULED</span>`;
/* A played game: the pre-game forecast graded in the words the track record
   uses. EARLY and inside-55/45 forecasts are leans, never a PICK HIT/MISS. */
function nhlGrade(g,c){
  if(!nhlDone(g)) return "";
  const ok=(g.hs>g.as?g.home:g.away)===c.pick;
  if(g.replay) return `<span class="chip replay" title="No pre-game number was archived for this game: the probability shown is a walk-forward replay, listed on the track record and never rated">replay</span>`;
  if(c.early||c.call==="LEAN")
    return `<span class="chip lean" title="${c.early?"EARLY-tier lean (team ratings only): graded on the track record as a forecast, never counted as a pick":"Inside 55/45: a lean, not a pick"}">lean ${ok?"&#10003;":"&#10007;"}</span>`;
  return `<span class="chip ${ok?"hit":"miss"}">PICK ${ok?"HIT":"MISS"}</span>`;}
const NHL_GOALIE_NOTE="Priced before starting goalies are confirmed (game-day morning) - the one thing our own measurement says the market knows and this model does not.";
/* The EDGE bar (post-process value layer - odds never touch the model). It
   always carries the goalie disclosure; pick_policy.md makes it mandatory. */
function nhlValbar(g,page){
  if(!nhlBadge(g)) return "";
  const v=g.value;
  return `<div class="valbar" title="${siteEsc(v.note||NHL_GOALIE_NOTE)}"${page?' style="margin-bottom:6px"':""}><span class="vt">EDGE</span> ${v.team}
    <b>+${Math.round(v.ev_cur*100)}% EV</b><span class="vodds">@ ${v.cur_dec}${page&&v.open_dec?` (open ${v.open_dec})`:""}</span>
    <span class="vlive nog">no goalie conf.</span></div>`
    +(page?`<div class="sub" style="font-size:12px;margin:0 0 12px">${NHL_GOALIE_NOTE} A disagreement indicator
      against the market, not a pick. <a class="tl" href="${nhlH("record")}">Units &amp; record</a></div>`:"");}

/* ---------- rail (kept for the other leagues' pages) ---------- */
function nhlRailBtn(on){return `<button class="lg ${on?"on":""}" data-lg="nhl"><span>NHL</span><span class="n">${nhlYear()}</span></button>`;}
function nhlRail(){return siteRail("nhl");}
function nhlWire(){
  siteWireRail();
  $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.nhlRange=x.dataset.r;nhlPage();});}

/* ---------- BOARD ---------- */
function nhlCard(g){
  const c=nhlCall(g), done=nhlDone(g), hw=c.hp>=0.5, vOn=nhlBadge(g), off=done?null:nhlOff(g);
  const who=code=>{const r=nhlRecOf(nhlTeam(code));
    const rec=r?` <span class="era" title="${r.cur?r.lbl+" record":r.lbl+" final record - this season has not started"}">${nhlWLO(r.r)}${r.cur?"":" last season"}</span>`:"";
    return `<span class="who"><span class="sp">${nhlNick(code)}${rec}</span></span>`;};
  const odds=p=>c.early?"":`<span class="odds" title="fair American odds (no vig) from the model">${amOdds(p)}</span>`;
  const b2b=done||off?"":[[g.away,g.ab2b],[g.home,g.hb2b]].filter(x=>x[1])
    .map(([t])=>`<span class="lbadge proj" title="${t} played the day before: the model's back-to-back term applies">${t} B2B</span>`).join("");
  const lv=done?`<span class="lv final">Final ${g.as}-${g.hs}${nhlOT(g)}</span>`
    :(off?`<span class="lbadge proj" title="${off} by the NHL: the new date is to be announced">${off}</span>`
    :`<span class="lv" data-ng="nhl_${g.away}_${g.home}">${g.start_utc?fmtTime(g.start_utc):"time TBD"}</span>`);
  const call=done?nhlGrade(g,c):(c.early?NHL_SCHED_PILL:sitePill(c.tier,c.pp,c.pick));
  const ct=g.ct||{};
  const edge=[["elo","Elo"],["xg","xG"],["rest","rest"],["b2b","B2B"]]
    .filter(([k])=>ct[k]!=null&&(k==="elo"||k==="xg"||Math.abs(ct[k])>=0.3))
    .map(([k,l])=>`<span class="${ct[k]>=0?"p":"n"}">${l} ${nhlSg(ct[k],1)}</span>`).join("");
  const note=done?"":(off?`<div class="earlyn">${off} &mdash; new date to be announced. Not a pick.</div>`
    :(c.early?`<div class="earlyn">Not a pick &mdash; team ratings only, no goalie or lineup input.
    Current lean <b>${c.pick} ${c.pPk}%</b>${c.pp<=POL_LEAN_MAX?", inside 55/45":""}.</div>`:""));
  return `<a class="gc nhl${vOn?" hasval":""}${c.early&&!done?" early":""}" href="${nhlGameHref(g)}">
    ${nhlValbar(g,false)}
    <div class="top">${lv}${done?"":`<span class="lbadge tbd" title="${NHL_EARLY_WHY}">Team ratings</span>`}${b2b}${g.playoff?'<span class="lbadge proj">Playoff</span>':""}${call}</div>
    <div class="side ${hw?"":"win"}"><span class="ab">${g.away}</span>${who(g.away)}${odds(1-c.hp)}<span class="pc">${c.pA}%</span></div>
    <div class="side ${hw?"win":""}"><span class="ab">${g.home}</span>${who(g.home)}${odds(c.hp)}<span class="pc">${c.pH}%</span></div>
    <div class="pbar"><div class="h" style="width:${c.pH}%"></div><div class="mid"></div></div>
    ${note}
    <div class="edge" title="Each model term's push on ${g.home}'s win probability, in percentage points around 50%"><span class="k">${g.home} edge</span>${edge}</div>
  </a>`;
}
function nhlPage(){
  const n=state.nhl, today=nflToday(), mc=n.model_card||{};
  const sch=(n.schedule||[]).slice().sort(nhlCmp);
  const up=sch.filter(g=>!nhlDone(g)), played=sch.filter(nhlDone);
  let R=state.nhlRange;
  if(R==="year"||(R==="results"&&!played.length)||["today","tomorrow","week","month","results"].indexOf(R)<0)
    R=up.length?"week":"results";
  state.nhlRange=R;
  const pDays=[...new Set(played.map(g=>g.d))].sort().reverse();   // played dates, newest first
  const resDays=pDays.slice(0,state.nhlResN);
  const W={today:[today,today],tomorrow:[addDays(today,1),addDays(today,1)],week:[today,addDays(today,6)],
    month:[today,addDays(today,29)],results:resDays.length?[resDays[resDays.length-1],resDays[0]]:[today,today]}[R];
  const games=(R==="results"?played:sch).filter(g=>g.d>=W[0]&&g.d<=W[1]);
  const F=[["today","Today"],["tomorrow","Tomorrow"],["week","Week"],["month","Month"]];
  if(played.length) F.push(["results","Results"]);
  const filts=F.map(([k,t])=>`<button class="filt ${k===R?"on":""}" data-r="${k}">${t}</button>`).join("");
  // live EDGE badges, pinned above the date filter (MLB's rule: a badge on
  // tomorrow's slate must never be hidden by a narrow range)
  const vEdges=up.filter(nhlBadge).sort((a,b)=>(b.value.ev_cur||0)-(a.value.ev_cur||0));
  const vstrip=siteValStrip(vEdges.map(g=>({team:g.value.team,ev:g.value.ev_cur,dec:g.value.cur_dec,
    href:nhlGameHref(g),when:`${g.away}@${g.home} &middot; ${fmtDay(g.d,today)}`})),
    "NHL badges are priced before starting goalies are confirmed (game-day morning) &mdash; the information our own measurement says the market has and this model lacks.");
  const vs=n.value_status||{};
  const gateLine=(vs.gate_early&&vs.n_early)?` ${vs.n_early} game${vs.n_early===1?" is":"s are"} quoted by the market this week and carry
    <b>no EDGE badge</b>: a badge is a stronger claim than a pick, and an EARLY forecast is not a pick.`:"";
  const opens=(n.phase==="preseason"&&n.first_game)?` The ${nhlSeasLbl()} regular season opens <b>${fmtDay(n.first_game,today)}</b>.`:"";
  const legend=`<div class="polnote">Every NHL forecast is <span class="chip t-EARLY">EARLY</span>: the model runs on team
    ratings only &mdash; Elo, an expected-goals team rating, rest and back-to-backs &mdash; with no goalie, lineup or roster
    input. So each game is <b>scheduled, not a pick</b>: no pick pill and no fair odds; the model's lean is shown and
    labelled pre-information, and a played game grades the lean, never a pick.${gateLine}${opens}
    <a href="${nhlH("record")}">Track record by tier &rsaquo;</a></div>`;
  let body;
  if(!games.length){
    const nx=up.find(g=>g.d>=today);
    const nxN=nx?up.filter(g=>g.d===nx.d).length:0;
    body=`<div class="empty">No NHL games in this window.${nx?`<br><span class="sub">Next: <b>${fmtDay(nx.d,today)}</b> &middot;
      ${nxN} game${nxN===1?"":"s"}.</span> <button class="filt on" data-r="${nx.d<=addDays(today,6)?"week":"month"}">Show them</button>`
      :(played.length?` <button class="filt on" data-r="results">Latest results</button>`:"")}</div>`;
  }else{
    const days=[...new Set(games.map(g=>g.d))].sort(); if(R==="results") days.reverse();   // results: newest first
    body=days.map(d=>{const gs=games.filter(g=>g.d===d), nd=gs.filter(nhlDone).length;
      const np=gs.filter(g=>!nhlDone(g)&&siteTier("nhl",g)!=="EARLY").length;
      return `<div class="day"><span class="d">${fmtDay(d,today)}</span>
        <span class="c">${gs.length} game${gs.length===1?"":"s"} &middot; ${nd===gs.length?"all final":`${np} pick${np===1?"":"s"}`}</span>
        ${np||nd===gs.length?"":'<span class="proj">scheduled &middot; team ratings only</span>'}</div>
        <div class="grid">${gs.map(nhlCard).join("")}</div>`;}).join("");
    if(R==="results"&&pDays.length>resDays.length)
      body+=`<div style="text-align:center;margin-top:14px"><button class="filt on" id="nhlmore">Show earlier results</button></div>`;
  }
  $("#view").innerHTML=`<div class="controls"><div class="rail">${siteRail("nhl")}</div><div class="filters">${filts}</div></div>
    ${vstrip}${legend}${body}
    <div class="cols2 nhl-cols" style="margin-top:20px">${nhlBoardTable()}${nhlModelCard(mc)}</div>`;
  nhlWire();
  const more=$("#nhlmore"); if(more) more.onclick=()=>{state.nhlResN+=14; nhlPage();};
}
/* Board side panel: the real table once the season is under way, the
   season simulation's top eight before the first final. */
function nhlBoardTable(){
  const n=state.nhl, T=n.teams||{}, P=n.proj||{};
  if(nhlStarted()){
    const rows=Object.entries(T).sort((a,b)=>(b[1].pts-a[1].pts)||((a[1].gp||0)-(b[1].gp||0))||((b[1].rw||0)-(a[1].rw||0)))
      .slice(0,8).map(([c,t],i)=>`<tr ${nhlRow(nhlH("team",c))}><td><span class="num">${i+1}</span></td>
        <td class="a"><span class="ab" style="color:var(--accent)">${c}</span> <span class="sub">${nhlNick(c)}</span></td>
        <td><span class="num">${t.gp}</span></td><td><span class="num">${t.pts}</span></td>
        <td><span class="num">${nhlPct3(t.pts_pct)}</span></td><td><span class="num">${nhlWLO(t)}</span></td></tr>`).join("");
    return `<div class="panel"><h3>Top of the table <span class="sub">by points</span></h3><div class="twrap"><table>
      <thead><tr><th></th><th class="a">Team</th><th>GP</th><th>Pts</th><th>P%</th><th>Record</th></tr></thead>
      <tbody>${rows}</tbody></table></div><a class="tl" href="${nhlH("standings")}">Full standings &rarr;</a></div>`;
  }
  const rows=Object.keys(P).filter(c=>T[c]).sort((a,b)=>P[b].pts-P[a].pts).slice(0,8).map((c,i)=>{const p=P[c], t=T[c];
    return `<tr ${nhlRow(nhlH("team",c))}><td><span class="num">${i+1}</span></td>
      <td class="a"><span class="ab" style="color:var(--accent)">${c}</span> <span class="sub">${nhlNick(c)}</span></td>
      <td><span class="num">${p.pts.toFixed(0)}</span></td><td><span class="num">${nhlOdds(p.po)}</span></td>
      <td><span class="num">${Math.round(t.elo)}</span></td></tr>`;}).join("");
  return `<div class="panel"><h3>Projected top 8 <span class="sub">season simulation &middot; EARLY</span></h3>
    ${rows?`<div class="twrap"><table><thead><tr><th></th><th class="a">Team</th><th>Proj pts</th><th>Playoffs</th><th>Elo</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`:'<div class="sub">No projection in this build.</div>'}
    <a class="tl" href="${nhlH("standings")}">Standings &amp; projections &rarr;</a></div>`;
}
function nhlModelCard(mc){
  const box=(k,v,cls,title)=>`<div class="b"${title?` title="${title}"`:""}><div class="k">${k}</div><div class="v ${cls||""}">${v}</div></div>`;
  // Policy rule 3: NHL has one tier (EARLY), so this is that tier's own rate,
  // labelled as such, and only once enough games are graded to read it.
  const minN=typeof REC_MIN_N!=="undefined"?REC_MIN_N:30;
  const acc=mc.cur_season_acc??null, nG=mc.cur_season_n||0, show=acc!=null&&nG>=minN;
  // Two published references, each labelled with its own number so the tiles
  // reconcile: the gain is against a logistic fit on the Elo term alone
  // (test_ll + gain); the Elo core is the tuned Elo's own probabilities.
  const dE=mc.test_delta_vs_elo, eloOnly=mc.test_ll!=null&&dE!=null?mc.test_ll+dE:null;
  return `<div class="panel"><h3>Model card <span class="sub">market-blind &middot; locked holdout</span></h3><div class="statgrid">
    ${box("Test log loss",mc.test_ll!=null?mc.test_ll.toFixed(4):"&mdash;","","Locked TEST holdout, published once; lower is better")}
    ${box("vs Elo alone",dE!=null?`&minus;${dE.toFixed(4)}${eloOnly!=null?` <span class="sub" style="font-size:11px">(${eloOnly.toFixed(4)})</span>`:""}`:"&mdash;","pos",
      `Log-loss gain over a logistic fit on the Elo term alone (${eloOnly!=null?eloOnly.toFixed(4):"published"}) on the same TEST games - published once`)}
    ${box("Tuned Elo core",mc.baseline_elo_test!=null?mc.baseline_elo_test.toFixed(4):"&mdash;","",
      "The tuned Elo's own win probabilities, no blend, on the TEST holdout - published once. A different reference from the Elo-alone fit beside it, so the two tiles do not subtract to the gain")}
    ${box("Coin flip","0.6931")}
    ${box("Home win rate",mc.home_win_rate!=null?(mc.home_win_rate*100).toFixed(1)+"%":"&mdash;")}
    ${box("Games trained",mc.n_games_train?mc.n_games_train.toLocaleString(LOC):"&mdash;")}
    ${box(`${nhlSeasLbl()} leans right`,show?(acc*100).toFixed(1)+"%":"&mdash;","",
      show?`${nG} graded EARLY-tier forecasts (leans included) - not picks`:`${nG} graded so far - a rate is shown from ${minN}`)}
  </div>
  <div class="sub" style="margin-top:10px;font-size:12px">Inputs: ${(mc.features||[]).map(siteEsc).join(" &middot; ")}.
    No goalie, lineup or roster input, which is why every forecast is EARLY.
    <a class="tl" href="${nhlH("record")}">Graded forecasts by tier</a></div></div>`;
}

/* ---------- GAME PAGE ---------- */
function nhlGamePage(id){
  setNav("");
  const n=state.nhl, g=(n.schedule||[]).find(x=>String(x.id)===String(id));
  if(!g){$("#view").innerHTML=`<div class="empty">No NHL game <b>${siteEsc(id)}</b> in this build &mdash; only the
    ${nhlSeasLbl()} season is served. <a class="tl" href="${nhlH("")}">Back to the NHL board</a></div>`;return;}
  const c=nhlCall(g), done=nhlDone(g), ct=g.ct||{}, today=nflToday(), off=done?null:nhlOff(g);
  const NLBL={home:"Home ice",elo:"Team Elo",xg:"xG rating",rest:"Rest",b2b:"Back-to-back"};
  const ord=["home","elo","xg","rest","b2b"], keys=ord.filter(k=>NHL_CT.indexOf(k)>=0).concat(NHL_CT.filter(k=>ord.indexOf(k)<0));
  const why=keys.filter(k=>ct[k]!=null).map(k=>nhlPRow(NLBL[k]||k,ct[k])).join("");
  const sum=Object.values(ct).reduce((s,v)=>s+(+v||0),0);
  // The equation is printed in integer tenths so it holds exactly: 50% plus
  // the true margin equals the home probability. The terms decompose that
  // margin exactly before rounding; each bar is rounded to 0.1 pt, so their
  // sum can miss by a tenth or two - said so when it does.
  const t10=Math.round(c.hp*1000), m10=t10-500, s10=Math.round(sum*10);
  const eqn=`50% ${m10>=0?"+":"&minus;"} ${(Math.abs(m10)/10).toFixed(1)} = ${(t10/10).toFixed(1)}%, ${g.home}'s home win probability`
    +(s10!==m10&&why?` (each bar is rounded to 0.1 pt, so the bars add to ${nhlSg(s10/10,1)})`:"");
  const rest=v=>v==null?"?":(v>=5?"5+":v);
  const b2bTxt=[[g.away,g.ab2b],[g.home,g.hb2b]].filter(x=>x[1]).map(x=>x[0]);
  // result / live: a played game gets the MLB final panel from the payload's
  // official result; an unplayed one gets the live hook (ESPN, display only)
  let panel;
  if(done){const win=g.hs>g.as?g.home:g.away, pw=win===g.home?c.hp:1-c.hp;
    panel=`<div class="livepanel" style="display:flex;flex-wrap:wrap"><span class="sc">${g.away} ${g.as} &ndash; ${g.hs} ${g.home}</span>
      <span class="st">Final${nhlOT(g)}${g.playoff?" &middot; playoff":""} &middot; ${g.replay?"a walk-forward replay gave":"the model gave"} ${win} ${(pw*100).toFixed(1)}%
      &middot; ${g.replay?"replay, not rated":`log loss ${(-Math.log(Math.max(pw,1e-9))).toFixed(3)}`}</span><span style="margin-left:auto">${nhlGrade(g,c)}</span></div>`;}
  else panel=`<div class="livepanel" id="livepanel" data-ng="nhl_${g.away}_${g.home}" data-d="${g.d}" data-away="${g.away}" data-home="${g.home}"></div>`;
  const status=done
    ?(g.replay?"No pre-game number was archived for this game: the probability shown is a walk-forward replay, listed on the track record and never rated."
      :`The graded number is the last forecast published before puck drop${g.frozen_at?` (recorded ${nhlEtStamp(g.frozen_at)||g.frozen_at})`:""}; it is never changed after the game.`)
    :(off?`${off} by the NHL &mdash; the new date is to be announced. The forecast is recomputed from the current team ratings on every serve until the game is played.`
      :(g.near?"Re-served every few hours until puck drop; the last number published before the game is the one the track record grades."
      :"Beyond the 10-day slate: recomputed from the current team ratings on every serve, and recorded once the game enters the 10-day window."));
  const pAway=c.pA, pHome=c.pH;
  $("#view").innerHTML=`
    <a class="back" data-nhlback href="${nhlH("")}">&lsaquo; Board</a>
    <div class="gh"><span class="mt">${nhlTL(g.away,nhlName(g.away))} <span style="color:var(--faint)">@</span> ${nhlTL(g.home,nhlName(g.home))}</span>
      <span class="meta">${off?fmtDay(g.d,today)+` &middot; <b>${off.toLowerCase()}</b>`:nhlWhen(g)}</span>
      <span class="chip t-${c.tier}" title="${c.early?NHL_EARLY_WHY:""}">${c.tier}</span>${g.playoff?'<span class="chip po">PLAYOFF</span>':""}</div>
    ${panel}
    ${nhlValbar(g,true)}
    <div class="cols nhl-cols">
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="panel"><h3>${done?"Pre-game forecast":(c.early?"Current lean":"Our projection")} <span class="sub">${c.early?(done?"EARLY tier &mdash; graded as a lean, not a pick":"scheduled, not a pick"):""}</span></h3>
          <div class="proj"><div class="big">${pHome}<span class="u">%</span></div>
            <div class="pk">${g.home} win probability<b>${c.early||c.call==="LEAN"?"Lean":"Pick"}: ${c.pick} <span class="cf">${c.pPk}%</span></b>
            <span class="sub">${c.early?(done?"a lean, not a pick":"scheduled, not a pick")+" &mdash; team ratings only, no goalie or lineup input"
              :(c.call==="LEAN"?"inside 55/45 &mdash; the model has a side, not a case":"")}${c.early&&c.pp<=POL_LEAN_MAX?" &middot; inside 55/45, close to a coin flip":""}</span></div></div>
          <div class="bigbar"><div class="h" style="width:${pHome}%"></div><div class="mid"></div></div>
          <div class="barlab"><span>${g.away} ${pAway}%</span><span>${pHome}% ${g.home}</span></div>
          <div class="barlab" style="margin-top:16px;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase"><span>Why &mdash; model contributions</span><span>exact &middot; ${g.home} win-prob points</span></div>
          <div class="barlab" style="margin-top:4px;font-size:11px"><span>&#9664; favours ${g.away}</span><span>favours ${g.home} &#9654;</span></div>
          <div class="why" style="margin-top:6px">${why||'<div class="sub">no contribution data</div>'}</div>
          <div class="sub" style="margin-top:10px;font-size:12px;color:var(--faint)">Each bar is one model term's push on ${g.home}'s
            win probability, in points around 50%, taken straight from the market-blind model's coefficients: ${eqn}.
            The lean above is the larger side.
            Team Elo includes ${g.home}'s home ice (+30 Elo). Rest: ${g.away} ${rest(g.arest)} days, ${g.home} ${rest(g.hrest)}
            ${b2bTxt.length?`&middot; back-to-back: ${b2bTxt.join(" and ")}`:"&middot; neither team on a back-to-back"}.</div>
        </div>
        <div class="panel"><h3>Goaltending <span class="sub">display only &mdash; not a model input</span></h3>
          <div class="mup">${nhlGoalieCol(g.away,"a")}<div class="vs">vs</div>${nhlGoalieCol(g.home,"h")}</div>
          <div class="sub" style="margin-top:10px;font-size:12px;color:var(--faint)">Neither starter is known to the model or to this page:
            NHL starters are confirmed on game-day morning. Listed by workload; the rating is goals saved above expected
            (GSAx), 50 = an average goalie.</div></div>
        <div class="panel"><h3>Key skaters <span class="sub">GlassBox 5v5 RAPM &middot; display only</span></h3>
          <div class="cols2 nhl-cols">${nhlKeySkaters(g.away)}${nhlKeySkaters(g.home)}</div></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="panel"><h3>${g.away} vs ${g.home}</h3>${nhlH2H(g)}</div>
        <div class="panel"><h3>What the model knows <span class="sub">and what it does not</span></h3><div class="signals">
          <div class="sig good"><span class="ic">+</span><span><b>Team strength</b>: Elo from every result through
            ${nhlDateOnly(n.ratings_through)||"the last final"}, home ice worth +30, pulled 30% back toward average between seasons.</span></div>
          <div class="sig good"><span class="ic">+</span><span><b>Shot quality</b>: an expected-goals team rating (MoneyPuck xG),
            updated on chances created and allowed, never on results.</span></div>
          <div class="sig good"><span class="ic">+</span><span><b>Schedule</b>: rest days and back-to-backs from the league calendar.</span></div>
          <div class="sig warn"><span class="ic">&minus;</span><span><b>Starting goalies</b>: not an input. Our own measurement says
            goalie news is the main thing the market knows and this model does not.</span></div>
          <div class="sig warn"><span class="ic">&minus;</span><span><b>Lineups, injuries, scratches, trades</b>: not inputs &mdash;
            the players on this page are display only.</span></div></div>
          <div class="sub" style="margin-top:10px;font-size:12px">${status}</div></div>
        <div class="panel"><h3>Signals</h3><div class="signals">${nhlSignals(g,c)}</div></div>
      </div>
    </div>`;
}
function nhlH2H(g){
  const A=nhlTeam(g.away)||{}, H=nhlTeam(g.home)||{}, P=state.nhl.proj||{}, pa=P[g.away]||{}, ph=P[g.home]||{};
  const R=(l,a,h,fmt,b)=>cmpRow(l,a,h,x=>x==null||x===""?"&mdash;":(fmt?fmt(x):x),b);
  const per=(x,gp)=>gp&&x!=null?Math.round(100*x/gp)/100:null;
  const cur=(A.gp||0)>0||(H.gp||0)>0, ra=cur?A:(A.prev||null), rh=cur?H:(H.prev||null);
  let rows="";
  if(ra&&rh){
    rows+=`<tr><td colspan="3" class="lbl">${cur?nhlSeasLbl()+" so far":nhlPrevLbl()+" final &mdash; this season has not started"}</td></tr>`
      +R("record",nhlWLO(ra),nhlWLO(rh))
      +R("points",ra.pts,rh.pts,null,"hi")
      +R("points %",ra.pts_pct,rh.pts_pct,nhlPct3,"hi")
      +R("goal diff",ra.gd,rh.gd,nhlPM,"hi")
      +R("goals for / gm",per(ra.gf,ra.gp),per(rh.gf,rh.gp),v=>v.toFixed(2),"hi")
      +R("against / gm",per(ra.ga,ra.gp),per(rh.ga,rh.gp),v=>v.toFixed(2),"lo")
      +R("last 10",ra.l10,rh.l10)
      +R("streak",ra.streak,rh.streak)
      +R("road / home",ra.away,rh.home);
  }
  rows+=`<tr><td colspan="3" class="lbl">model &amp; ratings</td></tr>`
    +R("our Elo",A.elo,H.elo,v=>Math.round(v),"hi")
    +R("Elo rank",A.rank,H.rank,v=>"#"+v,"lo")
    +R("xG rating",A.xg,H.xg,v=>nhlSg(v,2),"hi")
    +R("lineup GlassBox",A.glassbox,H.glassbox,v=>v.toFixed(0),"hi")
    +R("rest days",g.arest,g.hrest,v=>v>=5?"5+":v,"hi")
    +R("back-to-back",g.ab2b?"yes":"no",g.hb2b?"yes":"no")
    +R("proj. points",pa.pts,ph.pts,v=>v.toFixed(0),"hi")
    +R("playoff odds",pa.po,ph.po,v=>nhlOdds(v),"hi");
  return `<table class="cmp">${rows}</table>
    <div class="sub" style="margin-top:8px;font-size:11.5px;color:var(--faint)">xG rating = expected-goal margin per game against
      an average team. Lineup GlassBox = the skaters' RAPM ratings, display only. Projections: season simulation, EARLY tier.</div>`;
}
function nhlSignals(g,c){
  const s=[], A=nhlTeam(g.away)||{}, H=nhlTeam(g.home)||{}, I=nhlIdx(), ct=g.ct||{}, was=nhlDone(g)?"was":"is";
  [[g.away,g.ab2b],[g.home,g.hb2b]].forEach(([t,f])=>{if(f) s.push(["warn",`${t} ${was} on the second night of a back-to-back
    &mdash; the model's back-to-back term moves ${g.home}'s win probability ${nhlSg(ct.b2b||0,1)} pts.`]);});
  const rd=(g.hrest||0)-(g.arest||0);
  if(!g.hb2b&&!g.ab2b&&Math.abs(rd)>=2) s.push(["good",`${rd>0?g.home:g.away} has ${Math.abs(rd)} more days' rest
    (rest term ${nhlSg(ct.rest||0,1)} pts for ${g.home}).`]);
  [[g.away,A],[g.home,H]].forEach(([code,t])=>{
    const xr=I.xgR[code]; if(t.rank==null||xr==null) return;
    if(xr-t.rank>=8) s.push(["warn",`We rate ${code} #${t.rank} on results (Elo) but only #${xr} on shot quality (xG) &mdash;
      its results have run ahead of its chances.`]);
    else if(t.rank-xr>=8) s.push(["good",`${code}'s shot quality (xG #${xr}) is well ahead of its results (Elo #${t.rank})
      &mdash; the chances say it is better than its record.`]);
    if(t.gb_rank!=null&&Math.abs(t.gb_rank-t.rank)>=12) s.push(["info",`${code}'s lineup rates #${t.gb_rank} of ${I.nT} by player
      RAPM but #${t.rank} by Elo (the lineup rating is display only, not a model input).`]);
    if((t.gp||0)>=10&&t.xgf!=null&&t.xga!=null&&(t.xg_gp||0)>=10){
      const luck=(t.gd-(t.xgf-t.xga))/t.gp;
      if(Math.abs(luck)>=0.4) s.push([luck>0?"warn":"good",`${code} is ${luck>0?"out":"under"}scoring its expected goals by
        ${Math.abs(luck).toFixed(2)} goals a game (finishing + goaltending) &mdash; ${luck>0?"regression risk":"due a bounce"}.`]);
    }
  });
  if(c.pp<=POL_LEAN_MAX) s.push(["info",`Inside 55/45: the model has a side, not a case.`]);
  return s.length?s.map(([k,t])=>`<div class="sig ${k==="info"?"":k}"><span class="ic">!</span><span>${t}</span></div>`).join("")
    :'<div class="sub">No standout signals.</div>';
}
function nhlGoalieCol(code,side){
  const t=nhlTeam(code)||{}, P=state.nhl.players||{};
  const gs=(t.goalies||[]).map(id=>P[id]).filter(Boolean).slice(0,2);
  if(!gs.length) return `<div class="col ${side}"><div class="sub">no goalie on the roster file</div></div>`;
  return `<div class="col ${side}">${gs.map(p=>{const L=nhlLine(p), ln=L&&L.ln;
    return `<div style="margin-bottom:6px"><div class="nm">${nhlPL(p)}</div>
      <div class="sub">${p.rating!=null?`GlassBox <b>${p.rating.toFixed(0)}</b> &middot; GSAx/60 ${nhlSg(p.gsax60,3)}`:`NR &middot; ${siteEsc(p.nr||"not rated")}`}</div>
      ${ln?`<div class="statline"><span>SV% <b>${nhlSv(ln.svp)}</b></span><span>GAA <b>${ln.gaa!=null?(+ln.gaa).toFixed(2):"-"}</b></span>
        <span>${ln.gp} GP</span><span>${L.lbl}${ln.tm&&ln.tm!==code?" &middot; "+ln.tm:""}</span></div>`
        :'<div class="statline"><span>no NHL line last season</span></div>'}</div>`;}).join("")}</div>`;
}
function nhlKeySkaters(code){
  const t=nhlTeam(code)||{}, P=state.nhl.players||{};
  const r=(t.roster||[]).map(id=>P[id]).filter(p=>p&&p.grp!=="G"&&p.rating!=null)
    .sort((a,b)=>b.rating-a.rating).slice(0,6);
  return `<div><div style="font-weight:700;margin-bottom:6px">${nhlTL(code,nhlName(code))}</div>
    ${r.length?`<div class="twrap"><table><thead><tr><th class="a">Skater</th><th>GlassBox</th><th>Net</th></tr></thead><tbody>
    ${r.map(p=>`<tr ${nhlRow(nhlH("player",p.id))}><td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${nhlPosL(p)}</span></td>
      <td>${gb(p.rating)}</td><td>${nhlSgN(p.net)}</td></tr>`).join("")}</tbody></table></div>`
    :'<div class="sub">no rated skaters on the roster file</div>'}</div>`;
}

/* ---------- TEAMS ---------- */
function nhlTeams(){
  const n=state.nhl, T=n.teams||{}, P=n.proj||{};
  const S=state.nhlTeamSort||"elo";
  const key={elo:c=>-(T[c].elo||0),gb:c=>-(T[c].glassbox==null?-1:T[c].glassbox),proj:c=>-((P[c]&&P[c].pts)||0)}[S]||(c=>-(T[c].elo||0));
  const codes=Object.keys(T).sort((a,b)=>key(a)-key(b));
  const card=c=>{const t=T[c], p=P[c], r=nhlRecOf(t);
    return `<div class="tcard" ${nhlRow(nhlH("team",c))}>
      <div class="h"><span class="code">${c}</span><span class="nm">${nhlName(c)}</span>${gb(t.glassbox)}</div>
      <div class="stat"><span>Elo <b>${Math.round(t.elo)}</b> #${t.rank}</span><span>xG <b>${nhlSg(t.xg,2)}</b></span>
        ${p?`<span>Proj <b>${p.pts.toFixed(0)}</b> pts</span>`:""}</div>
      <div class="stat"><span${r&&!r.cur?` title="${r.lbl} final record"`:""}>${r?(r.cur?"Rec":"Last season")+` <b>${nhlWLO(r.r)}</b>`:"Rec <b>&mdash;</b>"}</span>
        <span>${siteEsc(t.div||"")}</span>${p?`<span>Playoffs <b>${nhlOdds(p.po,t.clinch)}</b></span>`:""}</div></div>`;};
  $("#view").innerHTML=`<div class="controls"><div class="rail">${siteRail("nhl")}</div>
      ${chipRow([["elo","Model Elo"],["gb","Lineup GlassBox"],["proj","Projected points"]],S,"tsort")}</div>
    <div class="eyebrow">Database &middot; NHL</div><h1 class="pt">Teams</h1>
    <div class="sub" style="margin-bottom:14px"><b>Elo</b> is the model's team rating (1500 = average; results through
      ${nhlDateOnly(n.ratings_through)||"the last final"}, pulled 30% back toward average between seasons) and <b>xG</b> its
      expected-goal rating (xG margin per game vs an average team, updated on shot quality, never on results) &mdash; the two
      team inputs of the market-blind model. The badge is the <b>lineup GlassBox</b>: the 12 forwards and 6 defencemen with the
      most ice time, rated 0-100 on 5v5 RAPM (50 = average) &mdash; a display metric, not a model input. <b>Proj</b> = mean points
      in ${(n.proj_info&&n.proj_info.n_sims||20000).toLocaleString(LOC)} season simulations. Click a team.</div>
    <div class="tgrid">${codes.map(card).join("")}</div>`;
  siteWireRail();
  $("#view").querySelectorAll("[data-tsort]").forEach(x=>x.onclick=()=>{state.nhlTeamSort=x.dataset.tsort;nhlTeams();});
}
function nhlTeamPage(code){
  const n=state.nhl, t=(n.teams||{})[code];
  if(!t) return siteNotFound("nhl","team",code);
  const P=n.players||{}, pr=(n.proj||{})[code], I=nhlIdx(), today=nflToday();
  const cur=(t.gp||0)>0, r=cur?t:(t.prev||null), rl=cur?nhlSeasLbl():nhlPrevLbl();
  const tile=(k,v,cls,sub)=>`<div class="b"><div class="k">${k}</div><div class="v ${cls||""}">${v}${sub?` <span class="sub" style="font-size:12px">${sub}</span>`:""}</div></div>`;
  const perG=(x,gp)=>gp&&x!=null?(x/gp).toFixed(2):"-";
  const luck=r&&r.gp>=10&&r.xgf!=null&&r.xga!=null?(r.gd-(r.xgf-r.xga))/r.gp:null;
  const tiles=(r?[
      tile(cur?"Record":rl+" record",nhlWLO(r)),
      tile("Points",r.pts,"",nhlPct3(r.pts_pct)),
      tile("Goal diff",nhlPM(r.gd),r.gd>0?"pos":(r.gd<0?"neg":"")),
      tile("GF / gm",perG(r.gf,r.gp)),tile("GA / gm",perG(r.ga,r.gp)),
      tile(cur?"Last 10":"Final 10",r.l10||"-"),tile("Streak",r.streak||"-"),
      tile("Home",r.home||"-"),tile("Away",r.away||"-"),
      luck!=null?`<div class="b" title="(goal diff - expected-goal diff) per game: finishing and goaltending beyond the chances. Large + = regression risk"><div class="k">Goals vs xG</div><div class="v ${Math.abs(luck)>=0.3?(luck>0?"warnc":"pos"):""}">${nhlSg(luck,2)}</div></div>`:""]
    :[]).concat([
      tile("Our Elo",Math.round(t.elo),"",`#${t.rank}`),
      tile("xG rating",nhlSg(t.xg,2),+Math.abs(t.xg).toFixed(2)===0?"":(t.xg>0?"pos":"neg"),`#${I.xgR[code]}`),
      pr?tile("Proj points",pr.pts.toFixed(0),"",`&plusmn;${pr.sd.toFixed(0)}`):"",
      pr?tile("Playoff odds",nhlOdds(pr.po,cur?t.clinch:null)):""]).join("");
  const opens=!cur&&n.first_game?` &middot; season opens ${nhlShortDay(n.first_game)}`:"";
  const sub=[siteEsc(t.div||""),siteEsc(t.conf||""),
    cur?`${nhlSeasLbl()}: ${nhlWLO(t)}, ${t.pts} pts${t.div_rank?` &middot; #${t.div_rank} in division`:""}`
       :(t.prev?`${nhlPrevLbl()}: ${nhlWLO(t.prev)}, ${t.prev.pts} pts${t.prev.div_rank?` &middot; #${t.prev.div_rank} in division`:""}`:"")]
    .filter(Boolean).join(" &middot; ")+opens;
  const roster=(t.roster||[]).map(id=>P[id]).filter(Boolean);
  const use=t.use==="cur"?"cur":"prev", ul=use==="cur"?nhlSeasLbl():nhlPrevLbl();
  const usage=p=>{const L=nhlLine(p,use); if(!L||!L.ln.gp) return -1;
    return p.grp==="G"?(L.ln.toi_s||L.ln.gp*3600):(L.ln.toi_pg||0)*L.ln.gp;};
  const bySk=(a,b)=>(a.rating==null)-(b.rating==null)||(b.rating||0)-(a.rating||0)||usage(b)-usage(a);
  const F=roster.filter(p=>p.grp==="F").sort(bySk), D=roster.filter(p=>p.grp==="D").sort(bySk);
  const G=roster.filter(p=>p.grp==="G").sort((a,b)=>usage(b)-usage(a));
  const cell=(v,cls)=>`<td><span class="num ${cls||""}">${v==null?"-":v}</span></td>`;
  const tm=ln=>ln&&ln.tm&&ln.tm!==code?` <span class="sub" style="font-size:10.5px" title="${ul} line with ${ln.tm}">${ln.tm}</span>`:"";
  const skRow=p=>{const L=nhlLine(p,use), ln=L&&L.ln;
    return `<tr ${nhlRow(nhlH("player",p.id))}>
      <td class="a"><span class="num" style="color:var(--faint);display:inline-block;width:22px">${nhlPosL(p)}</span>
        <span style="font-weight:600">${siteEsc(p.name)}</span>${p.num!=null?` <span class="sub">#${p.num}</span>`:""}</td>
      ${cell(p.age)}<td><span class="num">${ln?ln.gp:"-"}</span>${tm(ln)}</td>
      ${cell(ln&&ln.g)}${cell(ln&&ln.a)}${cell(ln&&ln.p)}${cell(ln?nhlPM(ln.pm):null)}${cell(ln?nhlMMSS(ln.toi_pg):null)}
      <td>${nhlGb(p)}</td><td>${nhlSgN(p.net,2,p.rating==null)}</td><td>${nhlSgN(p.off,2,p.rating==null)}</td><td>${nhlSgN(p.def,2,p.rating==null)}</td></tr>`;};
  const gRow=p=>{const L=nhlLine(p,use), ln=L&&L.ln;
    return `<tr ${nhlRow(nhlH("player",p.id))}>
      <td class="a"><span class="num" style="color:var(--faint);display:inline-block;width:22px">G</span>
        <span style="font-weight:600">${siteEsc(p.name)}</span>${p.num!=null?` <span class="sub">#${p.num}</span>`:""}</td>
      ${cell(p.age)}<td><span class="num">${ln?ln.gp:"-"}</span>${tm(ln)}</td>${cell(ln&&ln.gs)}
      ${cell(ln?nhlWLO(ln):null)}${cell(ln?nhlSv(ln.svp):null)}${cell(ln&&ln.gaa!=null?(+ln.gaa).toFixed(2):null)}${cell(ln&&ln.so)}
      <td>${ln&&ln.gsax!=null?nhlSgN(ln.gsax,1):'<span class="num sub">-</span>'}</td>
      <td>${nhlGb(p)}</td><td>${p.gsax60!=null?nhlSgN(p.gsax60,3):'<span class="num sub">&mdash;</span>'}</td></tr>`;};
  const skHead=`<thead><tr><th class="a">Player</th><th>Age</th><th>GP</th><th>G</th><th>A</th><th>P</th><th>+/&minus;</th><th>TOI/GP</th>
    <th>GlassBox</th><th title="5v5 xG/60, teammate- and opponent-adjusted">Net</th><th>Off</th><th>Def</th></tr></thead>`;
  const tbl=(title,rows,head)=>rows.length?`<div class="subh">${title}</div><div class="twrap"><table>${head}<tbody>${rows}</tbody></table></div>`:"";
  // schedule & forecasts: every game of the season, the lean for this team
  const games=(n.schedule||[]).filter(g=>g.home===code||g.away===code).sort(nhlCmp);
  const nextI=games.findIndex(g=>!nhlDone(g));
  const sRows=games.map((g,i)=>{const home=g.home===code, opp=home?g.away:g.home, c=nhlCall(g), done=nhlDone(g), off=done?null:nhlOff(g);
    const b2b=!done&&!off&&(home?g.hb2b:g.ab2b)?' <span class="lbadge proj" title="second night of a back-to-back">B2B</span>':"";
    let res;
    if(done){const gf=home?g.hs:g.as, ga=home?g.as:g.hs, w=gf>ga;
      res=`<span class="num ${w?"pos":"neg"}">${w?"W":(g.last&&g.last!=="REG"?"OTL":"L")} ${gf}-${ga}${nhlOT(g)}</span>`;}
    else if(off) res=`<span class="lbadge proj" title="${off} by the NHL: new date to be announced">${off==="Cancelled"?"CNCL":"PPD"}</span>`;
    else res=`<span class="sub">${g.start_utc?fmtTime(g.start_utc):"TBD"}</span>`;
    const lean=done?nhlGrade(g,c):(c.pp<=POL_LEAN_MAX?`<span class="chip lean" title="inside 55/45">toss-up</span>`
      :`<span class="chip lean" title="EARLY-tier lean, not a pick">lean ${c.pick}</span>`);
    return `<tr ${nhlRow(nhlGameHref(g))}${i===nextI&&i>0?' style="border-top:2px solid var(--line-2)"':""}>
      <td class="a"><span class="num">${nhlShortDay(g.d)}</span>${g.playoff?' <span class="chip po">PO</span>':""}</td>
      <td class="a">${home?"vs":"@"} ${nhlTL(opp)}${b2b}</td><td>${res}</td>
      <td><span class="num ${c.pick===code?"pos":""}">${home?c.pH:c.pA}%</span></td><td>${lean}</td></tr>`;}).join("");
  const projPanel=pr?`<div class="panel"><h3>Season projection <span class="sub">${(n.proj_info&&n.proj_info.n_sims||20000).toLocaleString(LOC)} simulations &middot; EARLY</span></h3>
      <div class="proj"><div class="big">${pr.pts.toFixed(0)}<span class="u"> pts</span></div>
        <div class="pk">projected points<b>${pr.lo}&ndash;${pr.hi} <span class="sub" style="font-size:12px;font-family:var(--sans)">middle 80% of runs</span></b>
        <span class="sub">${pr.w.toFixed(1)}-${pr.l.toFixed(1)}-${pr.otl.toFixed(1)} over ${pr.rem} remaining games</span></div></div>
      <div class="statgrid" style="margin-top:10px">
        <div class="b"><div class="k">Playoffs</div><div class="v">${nhlOdds(pr.po,cur?t.clinch:null)}</div></div>
        <div class="b"><div class="k">Win division</div><div class="v">${nhlOdds(pr.div)}</div></div>
        <div class="b"><div class="k">Presidents' Tr.</div><div class="v">${nhlOdds(pr.pres)}</div></div>
        <div class="b"><div class="k">Points SD</div><div class="v">${pr.sd.toFixed(1)}</div></div></div>
      <div class="sub" style="margin-top:10px;font-size:12px">${siteEsc((n.proj_info&&n.proj_info.method)||"")} Team ratings only (no goalie or
        lineup input), so these are EARLY-tier projections, not picks.</div></div>`:"";
  const lineup=t.glassbox!=null?`<div style="margin-left:auto;text-align:right"><div class="sub">Lineup GlassBox</div>${gb(t.glassbox)}
      <div class="sub" style="margin-top:4px">F ${t.gb_f!=null?t.gb_f.toFixed(0):"-"} &middot; D ${t.gb_d!=null?t.gb_d.toFixed(0):"-"}
        &middot; G ${t.gb_g!=null?t.gb_g.toFixed(0):"-"} &middot; #${t.gb_rank||"-"} of ${I.nT}</div></div>`:"";
  const fill=t.fill||{}, filled=(fill.rapm||0)+(fill.prior||0);
  const rp=n.rapm||{};
  $("#view").innerHTML=`<a class="back" data-nhlback href="${nhlH("teams")}">&lsaquo; Teams</a>
    <div class="thead"><span class="code">${code}</span>
      <div><div style="font-size:15px;font-weight:600">${nhlName(code)}</div><div class="sub">${sub}</div></div>
      ${lineup}</div>
    <div class="tstats">${tiles}</div>
    ${!cur&&t.prev?`<div class="sub" style="margin:-6px 0 14px;font-size:12px">${nhlSeasLbl()} has not started for ${code}: the tiles show the ${nhlPrevLbl()} final line; Elo, xG and the projection are current.</div>`:""}
    <div class="cols2 nhl-cols">${projPanel}
      <div class="panel"><h3>Schedule &amp; forecasts <span class="sub">${games.length} games &middot; EARLY leans, not picks</span></h3>
        <div class="twrap" style="max-height:520px;overflow:auto"><table><thead><tr><th class="a">Date</th><th class="a">Opp</th><th>Time / result</th>
          <th title="${code}'s win probability">Win%</th><th></th></tr></thead><tbody>${sRows||'<tr><td class="sub" colspan="5">No games in this build.</td></tr>'}</tbody></table></div>
        <div class="sub" style="margin-top:8px;font-size:12px">Team ratings only &mdash; no goalie or lineup input &mdash; so every game is a
          labelled lean. <a class="tl" href="${nhlH("record")}">Track record</a></div></div></div>
    ${tbl("Forwards",F.map(skRow).join(""),skHead)}
    ${tbl("Defence",D.map(skRow).join(""),skHead)}
    ${tbl("Goaltending",G.map(gRow).join(""),`<thead><tr><th class="a">Player</th><th>Age</th><th>GP</th><th>GS</th><th>W-L-OT</th><th>SV%</th>
      <th>GAA</th><th>SO</th><th title="goals saved above expected, that season">GSAx</th><th>GlassBox</th><th title="rating window, per 60 minutes">GSAx/60</th></tr></thead>`)}
    <div class="sub" style="margin-top:12px;font-size:12px">Season columns: ${ul}${use==="prev"?" (the team has not played this season yet)":""}; a code
      beside GP is the team the line was played for. GlassBox is a display metric, never a model input: skaters on 5v5 RAPM
      (${siteEsc(rp.window||"")}), goalies on GSAx; Net/Off/Def in xG per 60 minutes. <b>NR</b> = not rated &mdash; hover for why.
      ${t.glassbox!=null?`Lineup GlassBox: ${t.n_f+t.n_d} of 18 lineup slots carry a displayed rating${filled?`; ${fill.rapm?fill.rapm+" filled from the partial RAPM estimate":""}${fill.rapm&&fill.prior?", ":""}${fill.prior?fill.prior+" at the league-average prior of 50":""}`:""}.`:""}
      ${rp.caveat?siteEsc(rp.caveat):""}</div>`;
}

/* ---------- PLAYERS (index + ladder) ---------- */
function nhlPlayers(){
  const n=state.nhl, P=Object.values(n.players||{}), I=nhlIdx(), rp=n.rapm||{}, gm=n.goalie_model||{};
  const pos=["all","F","C","L","R","D","G"].indexOf(state.nhlPos)>=0?state.nhlPos:"all", isG=pos==="G";
  const inPos=p=>pos==="all"?p.grp!=="G":((pos==="F"||pos==="D"||pos==="G")?p.grp===pos:p.pos===pos);
  const pool=P.filter(inPos), rated=pool.filter(p=>p.rating!=null), unrated=pool.filter(p=>p.rating==null);
  const SORTS=isG?[["r","Rating"],["gsax","GSAx/60"],["gp","Most games"]]
    :[["r","Rating"],["off","Offense"],["def","Defense"],["toi","Most 5v5 TOI"]];
  const sk=SORTS.some(s=>s[0]===state.nhlSort)?state.nhlSort:"r";
  const val={r:p=>p.rating,off:p=>p.off,def:p=>p.def,toi:p=>p.toi,gsax:p=>p.gsax60,gp:p=>p.g_win&&p.g_win.gp}[sk];
  const vol=p=>isG?((p.g_win&&p.g_win.gp)||0):(p.toi||0);
  const MINS=isG?[["0","All rated"],["50","50+ GP"],["150","150+ GP"],["nr",`Unrated (${unrated.length})`]]
    :[["0","All rated"],["2000","2,000+ min"],["3500","3,500+ min"],["nr",`Unrated (${unrated.length})`]];
  const mn=MINS.some(m=>m[0]===String(state.nhlMin))?String(state.nhlMin):"0";
  const num=v=>v==null||!isFinite(v)?-1e9:v;
  const ranked=rated.slice().sort((a,b)=>num(val(b))-num(val(a))||b.rating-a.rating);
  const rk=new Map(ranked.map((p,i)=>[p.id,i+1]));
  const useOf=p=>{const L=nhlLine(p); return L?(L.ln.gp||0):0;};
  const shown=mn==="nr"?unrated.slice().sort((a,b)=>useOf(b)-useOf(a)||(b.toi||0)-(a.toi||0)):ranked.filter(p=>vol(p)>=+mn);
  const majority=P.some(p=>p.stats&&p.stats.cur)?"cur":"prev";
  const season=p=>{const L=nhlLine(p); if(!L) return '<span class="sub">no NHL line</span>';
    const ln=L.ln, tag=L.key!==majority?` <span class="sub">(${L.lbl})</span>`:"";
    return p.grp==="G"?`${nhlSv(ln.svp)} SV% &middot; ${ln.gaa!=null?(+ln.gaa).toFixed(2):"-"} GAA &middot; ${ln.gp} GP${tag}`
      :`${ln.p} P in ${ln.gp} GP &middot; ${nhlMMSS(ln.toi_pg)}${tag}`;};
  const skRow=p=>`<tr ${nhlRow(nhlH("player",p.id))}><td><span class="num">${rk.get(p.id)}</span></td>
    <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${nhlPosL(p)}</span></td>
    <td class="a">${nhlTL(p.team)}</td><td>${gb(p.rating)}</td>
    <td>${nhlSgN(p.net)}</td><td>${nhlSgN(p.off)}</td><td>${nhlSgN(p.def)}</td>
    <td><span class="num">${nhlInt(p.toi)}</span></td><td class="a sub">${season(p)}</td></tr>`;
  const gRow=p=>`<tr ${nhlRow(nhlH("player",p.id))}><td><span class="num">${rk.get(p.id)}</span></td>
    <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">G</span></td>
    <td class="a">${nhlTL(p.team)}</td><td>${gb(p.rating)}</td><td>${nhlSgN(p.gsax60,3)}</td>
    <td>${p.g_win?nhlSgN(p.g_win.gsax,1):"-"}</td><td><span class="num">${p.g_win?p.g_win.gp:"-"}</span></td>
    <td class="a sub">${season(p)}</td></tr>`;
  const nrRow=p=>`<tr ${nhlRow(nhlH("player",p.id))}>
    <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${nhlPosL(p)}</span></td>
    <td class="a">${nhlTL(p.team)}</td><td class="a"><span class="nhl-nr">${siteEsc(p.nr||"not rated")}</span></td>
    <td class="a sub">${season(p)}</td></tr>`;
  const head=mn==="nr"?`<tr><th class="a">Player</th><th class="a">Team</th><th class="a">Why no rating</th><th class="a">Latest season</th></tr>`
    :isG?`<tr><th></th><th class="a">Goalie</th><th class="a">Team</th><th>GlassBox</th><th title="goals saved above expected per 60 minutes, rating window">GSAx/60</th>
      <th title="total GSAx over the rating window">GSAx</th><th title="games in the rating window">GP</th><th class="a">Latest season</th></tr>`
    :`<tr><th></th><th class="a">Skater</th><th class="a">Team</th><th>GlassBox</th><th title="5v5 xG/60: offense + defense">Net</th>
      <th>Off</th><th>Def</th><th title="5v5 minutes in the rating window">5v5 min</th><th class="a">Latest season</th></tr>`;
  const rows=shown.map(mn==="nr"?nrRow:(isG?gRow:skRow)).join("")
    ||`<tr><td class="sub" colspan="9">No players match.</td></tr>`;
  const gbKey=isG?"gb_g":(pos==="D"?"gb_d":(pos==="all"?"glassbox":"gb_f"));
  const TT=Object.entries(n.teams||{}).map(([c,t])=>[c,t[gbKey]]).filter(x=>x[1]!=null).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const grpName={all:"lineups",F:"forward groups",C:"forward groups",L:"forward groups",R:"forward groups",D:"defence groups",G:"No. 1 goalies"}[pos];
  const posName={all:"skaters",F:"forwards",C:"centres",L:"left wings",R:"right wings",D:"defencemen",G:"goalies"}[pos];
  $("#view").innerHTML=`<div class="controls"><div class="rail">${siteRail("nhl")}</div></div>
    <div class="eyebrow">Database &middot; NHL</div><h1 class="pt">Players</h1>
    <div class="sub" style="margin-bottom:10px">Every player on a current NHL roster (${P.length}). <b>Skaters</b> are rated on
      teammate- and opponent-adjusted 5v5 expected-goal impact (RAPM, xG per 60 minutes, ${siteEsc(rp.window||"")}; at least
      ${nhlInt(rp.min_toi||1000)} 5v5 minutes to be rated): <b>Off</b> = chances created, <b>Def</b> = chances suppressed (positive
      is better), <b>Net</b> = both. <b>Goalies</b> are rated on goals saved above expected (GSAx on MoneyPuck xG,
      ${siteEsc(gm.window||"")}, shrunk toward average). One 0-100 <b>GlassBox</b> scale: 50 = average, 15 points = 1 SD.
      <b>Display metrics &mdash; the game model has no player input.</b> Click anyone.</div>
    ${rp.caveat?`<div class="polnote warn" style="font-size:12px">${siteEsc(rp.caveat)}</div>`:""}
    <input class="psearch" id="nhlq" placeholder="Search ${P.length} NHL players&hellip;" autocomplete="off">
    <div id="nhlres"></div>
    <div class="grid" style="margin:8px 0 14px">
      <div class="panel"><h3>Rating distribution <span class="sub">${rated.length} rated ${posName}</span></h3>${histBars(rated.map(p=>p.rating))}</div>
      <div class="panel"><h3>Best team ${grpName} <span class="sub">lineup GlassBox &middot; display only</span></h3>
        <div class="twrap"><table><tbody>${TT.map(([c,v],i)=>`<tr ${nhlRow(nhlH("team",c))}>
          <td><span class="num">${i+1}</span></td><td class="a"><span class="ab" style="color:var(--accent)">${c}</span> <span class="sub">${nhlNick(c)}</span></td>
          <td>${gb(v)}</td></tr>`).join("")}</tbody></table></div></div>
    </div>
    <div class="controls nhl-chips" style="margin-bottom:8px">
      ${chipRow([["all","All skaters"],["F","Forwards"],["C","C"],["L","LW"],["R","RW"],["D","Defence"],["G","Goalies"]],pos,"npos")}
      ${mn==="nr"?"":chipRow(SORTS,sk,"nsort")}
      ${chipRow(MINS,mn,"nmin")}
    </div>
    <div class="sub" style="margin-bottom:6px"><b>${shown.length}</b> shown &middot; ${mn==="nr"
      ?`${posName} on a roster without a displayed rating, with the reason`
      :`${rated.length} rated ${posName}; # = rank by ${SORTS.find(s=>s[0]===sk)[1].toLowerCase()} among them`}</div>
    <div class="twrap"><table><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
  siteWireRail();
  $("#view").querySelectorAll("[data-npos]").forEach(x=>x.onclick=()=>{
    const was=state.nhlPos==="G"; state.nhlPos=x.dataset.npos;
    if(was!==(state.nhlPos==="G")){state.nhlMin=0; state.nhlSort="r";}
    nhlPlayers();});
  $("#view").querySelectorAll("[data-nsort]").forEach(x=>x.onclick=()=>{state.nhlSort=x.dataset.nsort;nhlPlayers();});
  $("#view").querySelectorAll("[data-nmin]").forEach(x=>x.onclick=()=>{state.nhlMin=x.dataset.nmin;nhlPlayers();});
  const q=$("#nhlq"), res=$("#nhlres");
  if(q) q.oninput=()=>{const s=norm(q.value);
    if(s.length<2){res.innerHTML="";return;}
    const hits=P.filter(p=>norm(p.name).includes(s)).sort((a,b)=>(b.rating!=null)-(a.rating!=null)||(b.rating||0)-(a.rating||0)).slice(0,20);
    res.innerHTML=hits.length?`<div class="twrap" style="margin-bottom:6px"><table><tbody>
      ${hits.map(p=>`<tr ${nhlRow(nhlH("player",p.id))}>
        <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${nhlPosL(p)} &middot; ${p.team}</span></td>
        <td>${nhlGb(p)}</td><td class="a sub">${p.rating==null?siteEsc(p.nr||"not rated"):season(p)}</td></tr>`).join("")}
      </tbody></table></div>`:`<div class="sub" style="margin:6px 0 10px">No players match.</div>`;};
}
function nhlPlayerPage(id){
  const n=state.nhl, p=(n.players||{})[id];
  if(!p) return siteNotFound("nhl","player",id);
  const I=nhlIdx(), isG=p.grp==="G", rp=n.rapm||{}, gm=n.goalie_model||{};
  const box=(k,v,cls,title)=>`<div class="b"${title?` title="${title}"`:""}><div class="k">${k}</div><div class="v ${cls||""}">${v==null?"-":v}</div></div>`;
  const sgC=(v,d)=>v==null?"":(+Math.abs(v).toFixed(d)===0?"":(v>0?"pos":"neg"));
  const ht=p.ht?`${Math.floor(p.ht/12)}&prime;${p.ht%12}&Prime;`:"";
  const bio=[teamLink(p.team,nhlName(p.team),"nhl"),NHL_POSF[p.pos]||p.pos,p.num!=null?"#"+p.num:"",
    p.age!=null?"age "+p.age:"",p.shoots?(isG?"catches ":"shoots ")+p.shoots:"",[ht,p.wt?p.wt+" lb":""].filter(Boolean).join(", "),
    siteEsc(p.nat||"")].filter(Boolean).join(" &middot; ");
  let rating, expl;
  if(isG){
    const w=p.g_win;
    rating=(p.rating!=null?[box(`Rank of ${I.nG}`,"#"+p.rk,"","among rated goalies"),
        box("GSAx/60",nhlSg(p.gsax60,3),sgC(p.gsax60,3),"goals saved above an average goalie per 60 minutes, shrunk toward average")]
      :[box("Rating","NR","sub")])
      .concat(w?[box("Games",w.gp),box("xGA",w.xga),box("GA",w.ga),box("GSAx",nhlSg(w.gsax,1),sgC(w.gsax,1),"expected goals against minus goals against, rating window")]:[])
      .concat(p.rel!=null?[box("Reliability",(+p.rel).toFixed(2),"","share of the estimate that is data rather than the league-average prior")]:[]).join("");
    expl=`GlassBox goalie rating (0-100): ${siteEsc(gm.scale||"")}. Window ${siteEsc(gm.window||"")}; at least ${gm.min_gp||10} games to be rated.`;
  }else{
    const F=p.grp==="F";
    rating=(p.rating!=null?[box(`Rank of ${I.nSk}`,"#"+p.rk,"","among rated skaters, forwards and defencemen pooled"),
        box(`${F?"F":"D"} rank of ${F?I.nF:I.nD}`,"#"+p.rk_pos,"",F?"among rated forwards":"among rated defencemen")]
      :[box("Rating","NR","sub")])
      .concat(p.net!=null?[box("Net xG/60",nhlSg(p.net,3),p.rating==null?"sub":sgC(p.net,3)),
        box("Offense",nhlSg(p.off,3),p.rating==null?"sub":sgC(p.off,3),"5v5 expected goals created per 60, vs average"),
        box("Defense",nhlSg(p.def,3),p.rating==null?"sub":sgC(p.def,3),"5v5 expected goals suppressed per 60, vs average (positive = fewer allowed)"),
        box("5v5 minutes",nhlInt(p.toi))]:[])
      .concat(p.rel!=null?[box("Reliability",(+p.rel).toFixed(2),"","share of the estimate that is data rather than the prior")]:[]).join("");
    expl=`GlassBox skater rating (0-100): ${siteEsc(rp.scale||"")}. Window ${siteEsc(rp.window||"")}${rp.built?`, fit ${nhlDateOnly(rp.built)}`:""}; at least
      ${nhlInt(rp.min_toi||1000)} 5v5 minutes to be rated.${rp.caveat?" "+siteEsc(rp.caveat):""}`;
  }
  const nr=p.rating==null?`<div class="polnote" style="margin:0 0 14px">Not rated: ${siteEsc(p.nr||"no rating")}.${!isG&&p.net!=null
    ?" The partial estimate below is greyed out &mdash; too little ice time to rank.":""}</div>`:"";
  const sk=[["GP","gp"],["G","g"],["A","a"],["P","p"],["+/&minus;","pm",nhlPM],["PPP","ppp"],["SOG","sog"],
    ["Sh%",null,ln=>ln.sog?(100*ln.g/ln.sog).toFixed(1):"-"],["PIM","pim"],["TOI/GP","toi_pg",nhlMMSS]];
  const gl=[["GP","gp"],["GS","gs"],["W-L-OT",null,nhlWLO,"nhl-sm"],["SV%","svp",nhlSv],["GAA","gaa",v=>(+v).toFixed(2)],["SO","so"],
    ["SA","sa"],["GA","ga"],["GSAx","gsax",v=>nhlSg(v,1)],["xGA","xga"]];
  const car=isG?[["GP","gp"],["W-L-OT",null,nhlWLO,"nhl-sm"],["SV%","svp",nhlSv],["GAA","gaa",v=>(+v).toFixed(2)],["SO","so"]]
    :[["GP","gp"],["G","g"],["A","a"],["P","p"],["+/&minus;","pm",nhlPM],["PPP","ppp"],["SOG","sog"],["P/GP",null,ln=>ln.gp?(ln.p/ln.gp).toFixed(2):"-"]];
  const grid=(ln,spec)=>`<div class="statgrid">${spec.map(([k,f,fmt,cls])=>{const v=f?ln[f]:ln;
    return box(k,v==null?null:(fmt?fmt(v):v),cls);}).join("")}</div>`;
  const s=p.stats||{}, panels=[];
  [["cur",nhlSeasLbl()],["prev",nhlPrevLbl()]].forEach(([k,l])=>{const ln=s[k]; if(!ln) return;
    panels.push(`<div class="panel"><h3>${l} season${ln.tm&&ln.tm!==p.team?` <span class="sub">with ${nhlTL(ln.tm)}</span>`:""}</h3>${grid(ln,isG?gl:sk)}</div>`);});
  if(s.career) panels.push(`<div class="panel"><h3>Career <span class="sub">NHL regular season</span></h3>${grid(s.career,car)}</div>`);
  $("#view").innerHTML=`<a class="back" data-nhlback href="${nhlH("team",p.team)}">&lsaquo; ${nhlName(p.team)}</a>
    <div class="phead"><span class="nm">${siteEsc(p.name)}</span><span class="sub">${bio}</span>
      <span style="margin-left:auto">${nhlGb(p)}</span></div>
    <div class="sub" style="margin-bottom:14px">${expl} The game model has no player, lineup or goalie term. Market-blind.</div>
    ${nr}
    <div class="panel" style="margin-bottom:14px"><h3>${isG?"Goalie rating":"Skater rating"} <span class="sub">${isG?siteEsc(gm.window||""):siteEsc(rp.window||"")}</span></h3>
      <div class="statgrid">${rating}</div></div>
    <div class="pgrid">${panels.join("")||'<div class="panel"><h3>Production</h3><div class="sub">No NHL regular-season line on file yet.</div></div>'}</div>`;
}

/* ---------- STANDINGS & PROJECTIONS ---------- */
function nhlStandings(){
  const n=state.nhl, T=n.teams||{}, P=n.proj||{}, codes=Object.keys(T), today=nflToday();
  const started=nhlStarted(), hasPrev=codes.some(c=>T[c].prev&&T[c].prev.gp), hasProj=codes.some(c=>P[c]);
  const opts=[]; if(started) opts.push(["cur",nhlSeasLbl()]); if(hasProj) opts.push(["proj","Projected"]);
  if(hasPrev) opts.push(["prev",nhlPrevLbl()+" final"]);
  if(!opts.length) opts.push(["cur",nhlSeasLbl()]);
  const S=opts.some(o=>o[0]===state.nhlStd)?state.nhlStd:opts[0][0]; state.nhlStd=S;
  const V=["div","wc","league"].indexOf(state.nhlStdView)>=0?state.nhlStdView:"div";
  const row=c=>S==="prev"?(T[c].prev||{}):T[c];
  const key=c=>{if(S==="proj") return [-((P[c]&&P[c].pts)||0)];
    const r=row(c); return [-(r.pts||0),(r.gp||0),-(r.rw||0),-(r.row||0),-(r.w||0),-(r.gd||0)];};
  const cmpK=(a,b)=>{const x=key(a), y=key(b); for(let i=0;i<x.length;i++) if(x[i]!==y[i]) return x[i]-y[i]; return a<b?-1:1;};
  const byRank=(cs,f)=>S!=="proj"&&cs.every(c=>row(c)[f]!=null)?cs.slice().sort((a,b)=>row(a)[f]-row(b)[f]):cs.slice().sort(cmpK);
  const divs=(n.divisions&&n.divisions.length)?n.divisions:[...new Set(codes.map(c=>T[c].div).filter(Boolean))];
  const confs=n.conferences&&Object.keys(n.conferences).length?n.conferences:{"":divs};
  const inDiv=d=>codes.filter(c=>T[c].div===d);
  const pct=v=>nhlOdds(v);
  let clinchSeen=false;
  const teamCell=c=>{const cl=S!=="proj"?row(c).clinch:null; if(cl) clinchSeen=true;
    return `<td class="a">${cl?`<span class="nhl-cl" title="${NHL_CLINCH[cl]||""}">${cl}</span>`:""}<span class="ab" style="color:var(--accent)">${c}</span>
      <span class="sub">${nhlNick(c)}</span></td>`;};
  const actHead=`<tr><th></th><th class="a">Team</th><th>GP</th><th>W</th><th>L</th><th title="overtime / shootout losses">OTL</th><th>Pts</th><th>P%</th>
    <th title="regulation wins">RW</th><th title="regulation + overtime wins">ROW</th><th>GF</th><th>GA</th><th>Diff</th><th>L10</th><th>Strk</th><th>Home</th><th>Away</th>
    <th title="goal diff minus expected-goal diff (finishing + goaltending); large + = regression risk">G&minus;xG</th>
    ${S==="cur"?`<th>Elo</th><th>xG</th><th title="season simulation">Proj</th><th>Playoffs</th>`:""}</tr>`;
  const actRow=(c,i,cut)=>{const r=row(c), t=T[c], p=P[c];
    const lk=r.gp>=10&&r.xgf!=null&&r.xga!=null?r.gd-(r.xgf-r.xga):null;
    return `<tr ${nhlRow(nhlH("team",c))}${cut?' class="nhl-cut"':""}><td><span class="num">${i+1}</span></td>${teamCell(c)}
      <td><span class="num">${r.gp??0}</span></td><td><span class="num">${r.w??0}</span></td><td><span class="num">${r.l??0}</span></td>
      <td><span class="num">${r.otl??0}</span></td><td><span class="num"><b>${r.pts??0}</b></span></td><td><span class="num">${r.gp?nhlPct3(r.pts_pct):"&mdash;"}</span></td>
      <td><span class="num">${r.rw??"-"}</span></td><td><span class="num">${r.row??"-"}</span></td>
      <td><span class="num">${r.gf??0}</span></td><td><span class="num">${r.ga??0}</span></td>
      <td><span class="num ${r.gd>0?"pos":(r.gd<0?"neg":"")}">${nhlPM(r.gd??0)}</span></td>
      <td><span class="num">${r.l10||"-"}</span></td><td><span class="num">${r.streak||"-"}</span></td>
      <td><span class="num">${r.home||"-"}</span></td><td><span class="num">${r.away||"-"}</span></td>
      <td>${lk==null?'<span class="num sub">&mdash;</span>':`<span class="num ${Math.abs(lk)/r.gp>=0.3?(lk>0?"warnc":"pos"):""}">${nhlSg(lk,1)}</span>`}</td>
      ${S==="cur"?`<td><span class="num">${Math.round(t.elo)}</span></td><td>${nhlSgN(t.xg,2)}</td>
        <td><span class="num">${p?p.pts.toFixed(0):"-"}</span></td><td><span class="num">${p?pct(p.po):"-"}</span></td>`:""}</tr>`;};
  const projHead=`<tr><th></th><th class="a">Team</th><th title="mean points over the simulations">Proj pts</th><th title="middle 80% of runs">Range</th>
    <th>W</th><th>L</th><th>OTL</th><th>Playoffs</th><th>Division</th><th title="best record in the league">Presidents'</th>
    <th>Elo</th><th>xG</th><th>Lineup</th>${started?"<th>Now</th>":""}</tr>`;
  const projRow=(c,i,cut)=>{const p=P[c]||{}, t=T[c];
    return `<tr ${nhlRow(nhlH("team",c))}${cut?' class="nhl-cut"':""}><td><span class="num">${i+1}</span></td>${teamCell(c)}
      <td><span class="num"><b>${p.pts!=null?p.pts.toFixed(0):"-"}</b></span></td><td><span class="num sub">${p.lo!=null?p.lo+"&ndash;"+p.hi:"-"}</span></td>
      <td><span class="num">${p.w!=null?p.w.toFixed(0):"-"}</span></td><td><span class="num">${p.l!=null?p.l.toFixed(0):"-"}</span></td>
      <td><span class="num">${p.otl!=null?p.otl.toFixed(0):"-"}</span></td>
      <td><span class="num">${pct(p.po)}</span></td><td><span class="num">${pct(p.div)}</span></td><td><span class="num">${pct(p.pres)}</span></td>
      <td><span class="num">${Math.round(t.elo)}</span></td><td>${nhlSgN(t.xg,2)}</td><td>${gb(t.glassbox)}</td>
      ${started?`<td><span class="num">${nhlWLO(t)}</span></td>`:""}</tr>`;};
  const head=S==="proj"?projHead:actHead, rowF=S==="proj"?projRow:actRow;
  const table=(title,cs,cutAt)=>`${title?`<div class="subh">${title}</div>`:""}<div class="twrap"><table><thead>${head}</thead>
    <tbody>${cs.map((c,i)=>rowF(c,i,cutAt!=null&&i===cutAt)).join("")}</tbody></table></div>`;
  let html="";
  if(V==="league") html=table("",byRank(codes,"league_rank"));
  else if(V==="div") html=Object.entries(confs).map(([cf,ds])=>`${cf?`<div class="eyebrow" style="margin-top:22px">${siteEsc(cf)} Conference</div>`:""}`
    +ds.map(d=>table(siteEsc(d),byRank(inDiv(d),"div_rank"))).join("")).join("");
  else html=Object.entries(confs).map(([cf,ds])=>{
    // NHL format: top three in each division, then two wild cards per conference
    const tops=ds.map(d=>byRank(inDiv(d),"div_rank").slice(0,3));
    const inTop=new Set([].concat(...tops));
    const rest=codes.filter(c=>ds.indexOf(T[c].div)>=0&&!inTop.has(c));
    const wc=S!=="proj"&&rest.every(c=>row(c).wc_rank!=null)?rest.slice().sort((a,b)=>row(a).wc_rank-row(b).wc_rank):rest.slice().sort(cmpK);
    return `<div class="eyebrow" style="margin-top:22px">${siteEsc(cf)} Conference</div>`
      +ds.map((d,i)=>table(siteEsc(d),tops[i])).join("")+table("Wild card",wc,1);}).join("");
  const banner=!started?`<div class="polnote">The ${nhlSeasLbl()} regular season ${n.first_game?`opens <b>${fmtDay(n.first_game,today)}</b>`:"has not started"}
      (${n.n_games?Math.round(2*n.n_games/Math.max(codes.length,1))+" games per team":"full schedule"}). Until the first final this page shows
      the season projection and the ${nhlPrevLbl()} final standings.</div>`:"";
  const pi=n.proj_info||{};
  const note=S==="proj"
    ?`Projected: every remaining game played ${(pi.n_sims||20000).toLocaleString(LOC)} times with the model's own win probability
      (${siteEsc(pi.method||"")}). Team ratings only &mdash; no goalie or lineup input &mdash; so these are EARLY-tier projections, not
      picks. Built ${nhlEtStamp(pi.generated)||"with this data"}.`
    :`Official order: points, then regulation wins, regulation + overtime wins, wins and goal differential${S==="cur"?" (NHL API ranks)":""}.
      <b>G&minus;xG</b> = goal diff minus expected-goal diff (finishing and goaltending beyond the chances; shown from 10 games, large + =
      regression risk).${S==="cur"?" <b>Elo</b> and <b>xG</b> are the model's current team ratings; <b>Proj</b> and <b>Playoffs</b> come from the season simulation.":""}
      ${n.sources&&n.sources.standings?` Standings from the NHL API, fetched ${nhlEtStamp(n.sources.standings)}.`:""}`;
  $("#view").innerHTML=`<div class="controls"><div class="rail">${siteRail("nhl")}</div></div>
    <div class="eyebrow">Database &middot; NHL</div><h1 class="pt">Standings</h1>
    <div class="sub" style="margin-bottom:10px"><b>Elo</b> = the model's market-blind team rating (1500 = average). <b>xG</b> = its
      expected-goal rating, the xG margin per game against an average team, updated on shot quality and never on results.
      <b>Lineup</b> = GlassBox 0-100 lineup rating, display only. Click any team.</div>
    ${banner}
    <div class="controls nhl-chips" style="margin-bottom:6px">${chipRow(opts,S,"nstd")}${chipRow([["div","Division"],["wc","Wild card"],["league","League"]],V,"nview")}</div>
    ${html}
    ${clinchSeen?`<div class="sub" style="margin-top:10px;font-size:12px">${Object.entries(NHL_CLINCH).map(([k,v])=>`<span class="nhl-cl">${k}</span>${v}`).join(" &middot; ")}</div>`:""}
    <div class="sub" style="margin-top:10px;font-size:12px">${note}</div>`;
  siteWireRail();
  $("#view").querySelectorAll("[data-nstd]").forEach(x=>x.onclick=()=>{state.nhlStd=x.dataset.nstd;nhlStandings();});
  $("#view").querySelectorAll("[data-nview]").forEach(x=>x.onclick=()=>{state.nhlStdView=x.dataset.nview;nhlStandings();});
}

/* ---------- back links that return where the reader came from ----------
   A short trail of the hashes this tab has visited. The core router renders
   on hashchange first; this listener runs right after it on the same event.
   Arriving at the page just before the current one on the trail is a step
   back (the browser's Back button or a back link) and pops the trail; any
   other move pushes. An NHL game, team or player page's back link then
   points at the page before it on the trail - so the browser's Back never
   turns a back link into a forward one - and keeps the page's own parent
   link when the trail is empty or leads outside the NHL pages. */
const nhlTrail=[];
const nhlTrailKey=h=>{const p=siteParse(h); return [p.lg||"",p.v,p.arg||""].join("|");};
function nhlTrailStep(oldHash,hash){
  const T=nhlTrail, k=nhlTrailKey(hash);
  if(!T.length&&oldHash!=null) T.push(oldHash);            // the page the tab opened on
  if(T.length&&nhlTrailKey(T[T.length-1])===k){T[T.length-1]=hash; return;}
  if(T.length>=2&&nhlTrailKey(T[T.length-2])===k){T.pop(); T[T.length-1]=hash; return;}
  T.push(hash); if(T.length>60) T.splice(0,T.length-60);
}
function nhlBackLabel(h){
  const P=siteParse(h);
  if(P.v==="game"){const gid=String(P.arg||"").replace(/^nhl-/,"");
    if(!/^nhl-/.test(String(P.arg||""))) return null;
    const g=state.nhl&&(state.nhl.schedule||[]).find(x=>String(x.id)===gid);
    return g?`${g.away} @ ${g.home}`:null;}
  if(P.lg!=="nhl") return null;
  if(P.v==="team"&&P.arg&&nhlTeam(P.arg)) return nhlName(P.arg);
  if(P.v==="player"&&P.arg&&state.nhl&&state.nhl.players&&state.nhl.players[P.arg]) return siteEsc(state.nhl.players[P.arg].name);
  return {"":"Board",standings:"Standings",teams:"Teams",players:"Players",record:"Track record"}[P.v]||null;
}
window.addEventListener("hashchange",e=>{try{
  const o=String((e&&e.oldURL)||""), i=o.indexOf("#");
  nhlTrailStep(i>=0?o.slice(i):null,location.hash);
  const a=document.querySelector("#view a.back[data-nhlback]"); if(!a||nhlTrail.length<2) return;
  const prev=nhlTrail[nhlTrail.length-2], lbl=nhlBackLabel(prev); if(!lbl) return;
  a.setAttribute("href",prev); a.innerHTML="&lsaquo; "+lbl;
}catch(_){}});
