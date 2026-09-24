/* NFL: board, season, standings, teams, players, positions, game page.
   Built on the shared chrome in core.js: siteTier/nflProb (which number a game
   shows and in which information tier), sitePill (PICK / LEAN / SCHEDULED),
   siteRail, siteValStrip, siteFresh, siteClock and league-qualified hrefs.
   documents/pick_policy.md applies everywhere a forecast is drawn:
     - EARLY (served more than a week before kickoff) is a SCHEDULED game: a
       labelled lean, no pick pill, no fair price, never graded as a pick;
     - inside 55/45 a PROJECTED forecast is a LEAN, not a PICK;
     - a played game shows the number published before kickoff (ph, frozen in
       the pre-game ledger) and results are tallied per tier, never pooled. */

/* ---------- NFL shared helpers ---------- */
const NFL_GN={QB:"Quarterbacks",RB:"Running backs",WR:"Receivers",TE:"Tight ends",
  OL:"Offensive line",DL:"Defensive line",LB:"Linebackers",DB:"Secondary"};
const NFL_PL={QB:"quarterbacks",RB:"running backs",WR:"receivers",TE:"tight ends",
  OL:"offensive linemen",DL:"defensive linemen",LB:"linebackers",DB:"defensive backs"};
const NFL_ONE={QB:"quarterback",RB:"running back",WR:"receiver",TE:"tight end",
  OL:"offensive lineman",DL:"defensive lineman",LB:"linebacker",DB:"defensive back"};
const NFL_FAMS=["QB","RB","WR","TE","OL","DL","LB","DB","K","P","LS"];
const NFL_NOGRADE="\u2013";          // rating.tier for "not active in the reference season"
const NFL_LUCK_MIN_G=6;              // pythagorean luck is noise below this many games
const NFL_STOP=' onclick="event.stopPropagation()"';   // a link inside a clickable row
const nflH=(v,arg)=>siteHref("nfl",v,arg);
const nflGameHref=g=>"#/game/"+g.w+"_"+g.away+"_"+g.home;
const nflRow=h=>`onclick="location.hash='${h}'"`;
const nflTL=(code,label)=>`<a class="tl" href="${nflH("team",code)}"${NFL_STOP}>${label||code}</a>`;
function nflPL(id,name){
  const p=id?state.nfl.players[id]:null, nm=siteEsc(name||(p&&p.name)||"");
  return p?`<a class="tl" href="${nflH("player",id)}"${NFL_STOP}>${nm}</a>`:nm;
}
const nflSS=()=>state.nfl.stats_season||state.nfl.season;            // season of players[].stats
const nflSP=()=>state.nfl.stats_prev_season||(nflSS()-1);             // season of players[].stats_prev
const nflRS=()=>state.nfl.standings_season||state.nfl.season;         // season of teams[].w/l/t
const nflRec=o=>o?`${o.w}-${o.l}${o.t?"-"+o.t:""}`:"-";
const nflPct=p=>p==null?"-":(p>=1?"1.000":(+p).toFixed(3).slice(1));   // (1).toFixed(3).slice(1) is ".000"
const nflPo=v=>v==null?"-":(v<0.5?"&lt;1%":(v>=99.5?"&gt;99%":(+v).toFixed(0)+"%"));
const nflSgn=v=>v==null?"-":(v>0?"+":"")+v;
const nflS1=v=>v==null?"-":((+v>=0?"+":"")+(+v).toFixed(1));
const nflS2=v=>v==null?"-":((+v>=0?"+":"")+(+v).toFixed(2));
const nflPerG=(x,gp)=>gp?(x/gp).toFixed(1):"-";
const nflNum=v=>v==null?"-":(typeof v==="number"?v.toLocaleString(LOC):v);
const nflKick=g=>g.start_utc||(g.d+"T"+(g.t||"00:00")+":00");
const nflByKick=(a,b)=>nflKick(a)<nflKick(b)?-1:(nflKick(a)>nflKick(b)?1:0);
const nflShortDay=iso=>new Date(iso+"T12:00:00Z").toLocaleDateString(LOC,{weekday:"short",month:"short",day:"numeric",timeZone:TZ});
const nflAt=iso=>{const t=Date.parse(iso||""); return Number.isFinite(t)
  ?siteEt(t,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})+" ET":null;};
const nflTierChip=t=>`<span class="chip t-${t}">${t}</span>`;
const nflRated=id=>{const p=id?state.nfl.players[id]:null; return p&&p.rating?p.rating.r:null;};
const nflSiteTag=g=>g.neutral?` <span class="sub" title="${g.intl?"international game":"neutral site"}: the listed home team has no crowd edge; the model still applies its intercept">(${g.intl?"intl":"neutral"})</span>`:"";

/* The week the board opens on: the week of the next game that is not final
   (a game played in the last two days whose result is not in yet still
   counts), else the last week of the season. */
function nflCurWeek(){
  const sch=(state.nfl&&state.nfl.schedule)||[]; if(!sch.length) return null;
  const from=addDays(nflToday(),-2);
  const open=sch.filter(g=>g.hs==null&&g.d>=from).sort(nflByKick);
  return open.length?open[0].w:Math.max(...sch.map(g=>g.w));
}
function nflNextGame(){
  const today=nflToday();
  return ((state.nfl&&state.nfl.schedule)||[]).filter(g=>g.hs==null&&g.d>=today).sort(nflByKick)[0]||null;
}

/* One game's forecast, as every NFL surface draws it: the number shown
   (nflProb: the live model inside a week, the season simulation beyond), its
   tier (the serve's stamp), and the call. An EARLY forecast carries no call. */
function nflCall(g){
  const pr=nflProb(g), hp=pr.hp, t=pr.tier, pp=Math.max(hp,1-hp);
  return {hp, t, pp, near:pr.near, pick:hp>=0.5?g.home:g.away,
    done:g.hs!=null&&g.as!=null, call:t==="EARLY"?"EARLY":callOf(pp), exact:hp===g.ph};
}
/* A played game's grade, worded by tier and call: only a PICK is a HIT/MISS;
   an EARLY forecast or a LEAN was right or wrong, never graded as a pick. */
function nflGrade(g,c){
  if(!c.done) return "";
  if(g.replay) return `<span class="chip replay" title="no pre-game number was archived: a walk-forward replay, kept out of every rate">REPLAY</span>`;
  if(g.hs===g.as) return `<span class="chip">TIE</span>`;
  const ok=(g.hs>g.as?g.home:g.away)===c.pick;
  if(c.call==="PICK") return `<span class="chip ${ok?"hit":"miss"}">${ok?"HIT &#10003;":"MISS &#10007;"}</span>`;
  const why=c.call==="EARLY"
    ?"EARLY: published more than a week before kickoff - a pre-information lean, never graded as a pick"
    :"inside 55/45 - a lean, not graded as a pick";
  return `<span class="chip lean" title="${why}">lean ${ok?"&#10003;":"&#10007;"}</span>`;
}
/* The call as a table cell: SCHEDULED leans muted, PICK/LEAN as the pill. */
function nflCallCell(g,c){
  if(c.call==="EARLY") return `<span class="sub" title="EARLY: scheduled, not a pick">lean</span> <span class="ab">${c.pick}</span> <span class="num sub">${pctI(c.pp)}%</span>`;
  if(c.done) return `<span class="chip ${c.call==="PICK"?"pick":"lean"}">${c.call}</span> <span class="ab">${c.pick}</span> <span class="num sub">${pctI(c.pp)}%</span>`;
  return sitePill(c.t,c.pp,c.pick);
}
/* Played-game results per tier - never pooled into one number. */
function nflTally(games){
  const o={pick:[0,0],lean:[0,0],early:[0,0]};
  games.forEach(g=>{if(g.hs==null||g.as==null||g.replay||g.hs===g.as) return;
    const c=nflCall(g), k=c.call==="EARLY"?"early":(c.call==="LEAN"?"lean":"pick");
    o[k][(g.hs>g.as?g.home:g.away)===c.pick?0:1]++;});
  const parts=[];
  if(o.pick[0]+o.pick[1]) parts.push(`${nflTierChip("PROJECTED")} picks <b>${o.pick[0]}-${o.pick[1]}</b>`);
  if(o.lean[0]+o.lean[1]) parts.push(`PROJECTED leans <b>${o.lean[0]}-${o.lean[1]}</b>`);
  if(o.early[0]+o.early[1]) parts.push(`${nflTierChip("EARLY")} leans <b>${o.early[0]}-${o.early[1]}</b> <span class="sub">(pre-information, not picks)</span>`);
  return parts.join(" &middot; ");
}
/* The QBs a game shows: a played game shows who STARTED (qb_start, a post-game
   fact); an unplayed game shows the QB1s the forecast used (qb, frozen with
   the number), falling back to today's depth chart only for unplayed rows. */
function nflGameQbs(g){
  const T=state.nfl.teams, done=g.hs!=null&&g.as!=null, f=g.qb||{}, s=g.qb_start||{};
  const one=(side,team)=>{
    const fq=f[side]&&f[side].name?f[side]:null, sq=s[side]&&s[side].name?s[side]:null;
    if(done) return {q:sq||null, fq, started:!!sq};
    return {q:fq||(T[team]&&T[team].qb1)||null, fq, started:false};
  };
  return {a:one("a",g.away), h:one("h",g.home)};
}
/* Freshness of the NFL data behind a page, from the payload's own stamps. */
function nflFreshLine(){
  const n=state.nfl, f=siteFresh("nfl"), sv=(n.model_card&&n.model_card.serve)||{};
  if(!f) return "";
  const parts=[`model served <b>${f.at?siteEt(f.at,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"})+" ET":"(no serve time in this build)"}</b>`,
    `results through <b>${f.wk?"Week "+f.wk:"none yet"}</b>`];
  const ru=nflAt(n.results_updated);
  if(ru&&n.results_updated!==n.served_at) parts.push(`results updated ${ru}`);
  const pt=sv.proj_through, rt=n.results_through;
  if(rt&&(!pt||rt.w>pt.w))
    parts.push(`<span class="warnc">season projections are simulated through ${pt?"Week "+pt.w:"no finals"} &mdash; later finals are graded but not yet re-simulated</span>`);
  if(f.pending) parts.push(`<span class="warnc">${f.pending} played game${f.pending===1?"":"s"} awaiting results</span>`);
  return parts.join(" &middot; ");
}
function nflLegend(){
  const f=siteFresh("nfl");
  return `<div class="polnote${f&&f.stale?" warn":""}">Picks are published only inside a week of kickoff, when the live model
    runs on projected depth-chart lineups and the week's injury report (${nflTierChip("PROJECTED")}). Further out a game is
    <b>scheduled, not a pick</b> (${nflTierChip("EARLY")}): its number is a pre-information lean (mostly the season simulation
    from team strength), shown with no fair price. Outside 55/45 a forecast is a <b>PICK</b>; inside it is a <b>LEAN</b>. There is no
    CONFIRMED tier yet &mdash; the model has no game-day inactives feed. A played game keeps the number published before kickoff.
    <a href="${nflH("record")}">Track record by tier &rsaquo;</a>
    <div class="sub" style="margin-top:4px">Data: ${nflFreshLine()}</div></div>`;
}
const NFL_INJ_CLS={Out:"nfl-out",Doubtful:"nfl-dbt",Questionable:"nfl-q"};
function nflStatusChip(p){
  const n=state.nfl, SL=n.status_labels||{};
  if(p.inj&&p.inj.status){const i=p.inj;
    return `<span class="chip ${NFL_INJ_CLS[i.status]||"nfl-q"}" title="Week ${i.week||"?"} injury report${i.injury?": "+siteEsc(i.injury):""}">${siteEsc(i.status)}</span>`;}
  if(p.status&&p.status!=="ACT")
    return `<span class="chip nfl-res" title="${siteEsc(SL[p.status]||p.status)}">${siteEsc(p.status==="DEV"?"PS":p.status)}</span>`;
  return "";
}
/* A player's season line in one short string (index, ladders, search). */
function nflStatShort(p,s){
  s=s||p.stats||{}; if(!s.g) return "&mdash;";
  const v=k=>s[k]||0;
  switch(p.fam){
    case "QB": return `${nflNum(v("pass_yds"))} yds &middot; ${v("pass_td")} TD &middot; ${v("ints")} INT`;
    case "RB": return `${nflNum(v("rush_yds"))} rush &middot; ${v("rec")} rec &middot; ${v("rush_td")+v("rec_td")} TD`;
    case "WR": case "TE": return `${v("rec")} rec &middot; ${nflNum(v("rec_yds"))} yds &middot; ${v("rec_td")} TD`;
    case "OL": case "LS": return `${s.g} G`;
    case "K": return `${v("fgm")}/${v("fga")} FG &middot; ${v("xpm")}/${v("xpa")} XP`;
    case "P": return `${v("punts")} punts &middot; ${v("punts")?(v("punt_yds")/v("punts")).toFixed(1):"-"} avg`;
    default: return `${v("tak")+v("ast")} tkl &middot; ${v("sk")} sk &middot; ${v("dint")} INT &middot; ${v("pd")} PD`;
  }
}
function nflGradeLegend(){
  const t=((state.nfl.model_card&&state.nfl.model_card.ratings)||{}).tiers;
  return "Grade = "+(t?siteEsc(String(t).replace(/^grade\s*=\s*/i,"")):"percentile within the position");
}
function nflRatingsLine(){
  const r=(state.nfl.model_card&&state.nfl.model_card.ratings)||{};
  return `${r.plays?r.plays.toLocaleString(LOC)+" plays":"every snap"}${r.seasons?", "+r.seasons:""}`;
}
function nflFrozenNote(){
  const n=state.nfl, r=(n.model_card&&n.model_card.ratings)||{};
  return r.through_season&&r.through_season<n.season
    ?`Player ratings are frozen at the end of ${r.through_season} until ${n.season} snap participation data is published;
      rosters, depth charts, injuries, records and stats are current.`:"";
}

/* ---------- NFL BOARD ---------- */
const NFL_EDGE_LBL={elo:"team",qb:"QB",units:"units",roster:"roster",ts:"players",hfa:"HFA",sched:"rest",luck:"luck",abs:"abs",avail:"inj"};
/* The card's edge strip: the exact breakdown (cx) of the live-model number,
   home-signed. Shown only when the card shows that number: a card showing the
   season simulation (EARLY, beyond a week) would be explained by bars for a
   different number. */
function nflEdgeRow(g,c){
  const cx=g.cx; if(!cx||!c.exact) return "";
  const ks=Object.keys(NFL_EDGE_LBL).filter(k=>cx[k]!=null&&Math.abs(cx[k])>=0.5)
    .sort((a,b)=>Math.abs(cx[b])-Math.abs(cx[a])).slice(0,5);
  if(!ks.length) return "";
  return `<div class="edge"><span class="k">${g.home} edge</span>${ks.map(k=>
    `<span class="${cx[k]>=0?"p":"n"}">${NFL_EDGE_LBL[k]} ${sgn(cx[k])}</span>`).join("")}</div>`;
}
function nflValBar(g){
  const v=g.value; if(!v||!v.available||g.d<nflToday()) return "";     // no price on a game already played
  return `<div class="valbar" title="${siteEsc(v.note||"")}"><span class="vt">EDGE</span> ${v.team} <b>+${Math.round(v.ev_cur*100)}% EV</b>
    <span class="vodds">@ ${v.cur_dec}</span><span class="vlive">&#9679; live</span></div>`;
}
function nflCard(g){
  const c=nflCall(g), hp=c.hp, homeWin=hp>=0.5, early=c.t==="EARLY"&&!c.done, today=nflToday();
  const overdue=!c.done&&g.d<today;
  const hasVal=g.value&&g.value.available;
  const place=g.neutral?(g.intl?" &middot; Intl":" &middot; Neutral"):"";
  // an EARLY card shows the live model's number when the game is inside a
  // calendar week (c.exact) and the season simulation beyond: label which
  const badge=c.done
    ?`<span class="lbadge tbd">Wk ${g.w}${place}</span>${nflTierChip(c.t)}`
    :(c.t==="EARLY"
      ?`<span class="lbadge tbd" title="served more than a week before kickoff: ${c.exact
          ?"the live model's early read on the current depth chart"
          :"the season simulation from team strength"}, not a pick">Wk ${g.w} &middot; ${c.exact?"Early read":"Season sim"}${place}</span>`
      :`<span class="lbadge proj" title="live model on projected depth-chart lineups and this week's injury report">Wk ${g.w} &middot; Proj lineup${place}</span>`);
  const clock=siteClock(g.t);
  const lv=c.done?`<span class="lv final">Final</span>`
    :`<span class="lv" data-ng="${g.away}_${g.home}" data-d="${g.d}">${overdue?"awaiting result":clock}</span>`;
  // played but not scored yet (overdue): the published call stays, the price goes
  const right=c.done?`<span class="nfl-r">${nflGrade(g,c)}</span>`:sitePill(c.t,c.pp,c.pick);
  const qb=nflGameQbs(g);
  const qbTag=o=>{if(!o.q||!o.q.name) return "";
    const r=nflRated(o.q.id);
    return `${siteEsc(o.q.name)}${r!=null?` <span class="era">${r.toFixed(0)}</span>`:""}${!c.done&&c.t==="PROJECTED"?' <span class="pj">proj</span>':""}`;};
  const odds=p=>(early||c.done||overdue)?"":`<span class="odds" title="fair American odds (no vig) from the model">${amOdds(p)}</span>`;
  const score=(mine,theirs)=>c.done?`<span class="nsc${mine<theirs?" lo":""}">${mine}</span>`:"";
  const lean=(early?`<div class="earlyn">Not a pick &mdash; served more than a week before kickoff, ${c.exact
        ?"from team strength and the current depth chart"
        :"from the season simulation (team strength only)"}. ${overdue
        ?`Pre-game lean <b>${c.pick} ${pctI(c.pp)}%</b>; it will not be graded as a pick.`
        :`Current lean <b>${c.pick} ${pctI(c.pp)}%</b>; it becomes a pick at the first serve inside a week of kickoff.`}</div>`:"");
  return `<a class="gc nflc${hasVal?" hasval":""}${early?" early":""}" href="${nflGameHref(g)}">
    ${nflValBar(g)}
    <div class="top">${lv}${badge}${right}</div>
    <div class="side ${homeWin?"":"win"}"><span class="ab">${g.away}</span>
      <span class="who"><span class="sp">${qbTag(qb.a)}</span></span>${odds(1-hp)}${score(g.as,g.hs)}<span class="pc">${pctI(1-hp)}%</span></div>
    <div class="side ${homeWin?"win":""}"><span class="ab">${g.home}</span>
      <span class="who"><span class="sp">${qbTag(qb.h)}</span></span>${odds(hp)}${score(g.hs,g.as)}<span class="pc">${pctI(hp)}%</span></div>
    <div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
    ${lean}${nflEdgeRow(g,c)}
  </a>`;
}
/* Unplayed games counted by call: a LEAN (inside 55/45) is not a PICK, and an
   EARLY game is scheduled, neither. */
function nflCallCount(games){
  const cs=games.filter(g=>g.hs==null).map(nflCall);
  return {nP:cs.filter(c=>c.call==="PICK").length, nL:cs.filter(c=>c.call==="LEAN").length,
    nE:cs.filter(c=>c.call==="EARLY").length};
}
const nflCallWords=k=>`${k.nP} pick${k.nP===1?"":"s"}${k.nL?` &middot; ${k.nL} lean${k.nL===1?"":"s"}`:""}`;
function nflDay(d,gs,today){
  const k=nflCallCount(gs);
  const fin=gs.filter(g=>g.hs!=null&&g.as!=null), nf=fin.length, open=gs.length-nf;
  const earlyFin=nf&&fin.every(g=>nflProb(g).tier==="EARLY");
  const tag=open&&!(k.nP+k.nL)?'<span class="proj">scheduled &middot; not picks</span>'
    :(!open&&earlyFin?'<span class="proj" title="published more than a week before kickoff - pre-information leans, never graded as picks">leans &middot; not picks</span>':"");
  return `<div class="day"><span class="d">${fmtDay(d,today)}</span>
    <span class="c">${gs.length} game${gs.length===1?"":"s"}${open&&(k.nP+k.nL)?` &middot; ${nflCallWords(k)}`:""}${nf?` &middot; ${nf===gs.length?"final":nf+" final"}`:""}</span>${tag}</div>
    <div class="grid">${gs.map(nflCard).join("")}</div>`;
}
function nflPage(){
  const n=state.nfl, mc=n.model_card||{}, today=nflToday();
  const sch=(n.schedule||[]).slice().sort(nflByKick), proj=n.proj||{}, T=n.teams||{};
  const R=["today","week","month","year"].indexOf(state.nflRange)>=0?state.nflRange:"week";
  const weeks=[...new Set(sch.map(g=>g.w))].sort((a,b)=>a-b), cw=nflCurWeek();
  const W=(state.nflWk!=null&&weeks.indexOf(state.nflWk)>=0)?state.nflWk:cw;
  let games, title;
  if(R==="today"){games=sch.filter(g=>g.d===today); title="Today";}
  else if(R==="week"){games=sch.filter(g=>g.w===W); title=W!=null?`Week ${W}`:"This week";}
  else if(R==="month"){games=sch.filter(g=>g.d>=today&&g.d<=addDays(today,29)); title="Next 30 days";}
  else{games=sch; title=`${n.season} season`;}
  const filts=[["today","Today"],["week","Week"],["month","Month"],["year","Season"]]
    .map(([k,t])=>`<button class="filt ${k===R?"on":""}" data-r="${k}">${t}</button>`).join("");
  // pinned VALUE strip (post-process EDGE layer, never a model input)
  const vg=sch.filter(g=>g.hs==null&&g.d>=today&&g.value&&g.value.available).sort((a,b)=>(b.value.ev_cur||0)-(a.value.ev_cur||0));
  const vstrip=siteValStrip(vg.map(g=>({team:g.value.team,ev:g.value.ev_cur,dec:g.value.cur_dec,href:nflGameHref(g),
    when:`${g.away}@${g.home} &middot; ${fmtDay(g.d,today)}`})),
    "NFL badges are priced off the weekly model build; injury and depth-chart news after that build is not in it.");
  const vs=n.value_status, vhold=vs&&vs.state==="suppressed"
    ?`<div class="polnote warn">EDGE badges are held for this build: ${siteEsc(vs.reason||"the model build is stale")}.</div>`:"";
  // heading for the window
  const nF=games.filter(g=>g.hs!=null&&g.as!=null).length, kc=nflCallCount(games);
  const span=games.length?(games[0].d===games[games.length-1].d?nflShortDay(games[0].d)
    :`${nflShortDay(games[0].d)} &ndash; ${nflShortDay(games[games.length-1].d)}`):"";
  const stepper=R==="week"&&W!=null?`<div class="filters nfl-stp">
      ${W>weeks[0]?`<button class="filt" data-wk="${W-1}">&lsaquo; Wk ${W-1}</button>`:""}
      <button class="filt ${W===cw?"on":""}" data-wk="${cw}">This week</button>
      ${W<weeks[weeks.length-1]?`<button class="filt" data-wk="${W+1}">Wk ${W+1} &rsaquo;</button>`:""}</div>`:"";
  const head=`<div class="nfl-wkh"><div><div class="eyebrow">NFL &middot; ${n.season} season</div>
      <h1 class="pt">${title} <span class="sub">${span?span+" &middot; ":""}${games.length} game${games.length===1?"":"s"} &middot; ${nflCallWords(kc)}${kc.nE?` &middot; ${kc.nE} scheduled`:""}${nF?` &middot; ${nF} final`:""}</span></h1></div>
    ${stepper}</div>`;
  let cards;
  if(!sch.length) cards=`<div class="empty">The ${n.season} NFL schedule is not in this build (${nflFreshLine()}).</div>`;
  else if(!games.length){
    const nx=nflNextGame();
    cards=nx?`<div class="empty">No NFL games ${R==="today"?"today":"in this window"}. Next: <b>${fmtDay(nx.d,today)} &middot; ${siteClock(nx.t)}</b>
        &mdash; ${nx.away} @ ${nx.home}, Week ${nx.w}.
        <div style="margin-top:12px"><button class="filt on" data-wk="${nx.w}">Show Week ${nx.w}</button></div></div>`
      :`<div class="empty">The ${n.season} regular season is complete &mdash; every result is on the <a class="tl" href="#/season">Season</a> page.</div>`;
  }else{
    const days=[...new Set(games.map(g=>g.d))].sort();
    cards=days.map(d=>nflDay(d,games.filter(g=>g.d===d),today)).join("");
  }
  // ---- season outlook, ratings, boards (always drawn, whatever the window)
  const rs=nflRS(), rsCur=rs===n.season;
  const pTop=Object.entries(proj).sort((x,y)=>y[1].w-x[1].w).slice(0,8).map(([c,p],i)=>{const t=T[c]||{};
    return `<tr ${nflRow(nflH("team",c))}><td><span class="num">${i+1}</span></td>
      <td class="a"><span class="ab" style="color:var(--accent)">${c}</span> <span class="sub">${t.nick||""}</span></td>
      <td><span class="num">${rsCur?nflRec(t):"-"}</span></td>
      <td><span class="num">${p.w.toFixed(1)}-${(17-p.w).toFixed(1)}</span></td>
      <td><span class="num">${nflPo(p.div)}</span></td><td><span class="num">${nflPo(p.po)}</span></td></tr>`;}).join("");
  const cal=(mc.calibration||[]).map(c=>`<tr><td class="a">${c.bucket}</td>
    <td><span class="num">${c.hit.toFixed(1)}%</span></td><td><span class="num">${c.n}</span></td></tr>`).join("");
  const pow=(n.power||[]).map(p=>{const t=T[p.code]||{};
    return `<tr ${nflRow(nflH("team",p.code))}><td><span class="num">${p.rank}</span></td>
      <td class="a"><span class="ab" style="color:var(--accent)">${p.code}</span> <span class="sub">${t.nick||p.nick||""}</span></td>
      <td><span class="num">${rsCur?nflRec(t):"-"}</span></td>
      <td><span class="num">${p.elo.toFixed(0)}</span></td>
      ${["off_pass","off_run","def_pass","def_run"].map(k=>`<td><span class="num ${p[k]>=0?"pos":"neg"}">${nflS1(p[k])}</span></td>`).join("")}
      <td>${gb(t.glassbox)}</td><td><span class="num">${t.rpow_rank?"#"+t.rpow_rank:"-"}</span></td></tr>`;}).join("");
  const powTitle=n.power_preseason?`${n.season} preseason power ratings`
    :`Power ratings <span class="sub" style="font-family:var(--sans);font-weight:400">through ${n.power_asof?nflShortDay(n.power_asof):"the latest final"}</span>`;
  const bd=pos=>((n.boards||{})[pos]||[]).map((p,i)=>`<tr ${p.id&&n.players[p.id]?nflRow(nflH("player",p.id)):""}>
    <td><span class="num">${i+1}</span></td><td class="a">${nflPL(p.id,p.player)}</td>
    <td><span class="num">${nflS2(p.z)}</span></td>
    <td><span class="num">${nflS2(p.cons)}</span></td><td><span class="num">${nflNum(p.n)}</span></td></tr>`).join("");
  const bs=n.boards_season||n.player_board_season||"";
  const btbl=(t,pos)=>`<div class="panel"><h3>${t} <span class="sub">${bs} per-play value</span></h3><div class="twrap"><table>
    <thead><tr><th></th><th class="a">Player</th><th>Value z</th><th>Floor</th><th>Plays</th></tr></thead>
    <tbody>${bd(pos)}</tbody></table></div></div>`;
  const mvp=(n.mvp||[]).map((m,i)=>`<tr ${m.id&&n.players[m.id]?nflRow(nflH("player",m.id)):""}><td><span class="num">${i+1}</span></td>
    <td class="a">${nflPL(m.id,m.player)} <span class="sub">${m.pos||""}</span></td>
    <td><span class="num pos">+${m.pts.toFixed(2)}</span></td><td><span class="num">${m.n_abs}</span></td></tr>`).join("");
  const sv=mc.serve||{};
  const model=`<div class="sub" style="margin:0 0 4px">Market-blind ${mc.n_features||14}-feature blend around an 11-vs-11
    per-snap TrueSkill (${nflRatingsLine()}) &middot; <b>${mc.accuracy!=null?mc.accuracy.toFixed(1)+"%":"-"}</b> accurate on a locked holdout
    (${mc.holdout||"scored once"}) &middot; log loss <b>${mc.test_log_loss!=null?mc.test_log_loss.toFixed(3):"-"}</b>
    (closing line ${mc.close_log_loss!=null?mc.close_log_loss.toFixed(3):"-"}) &middot; ${mc.training||""}.${mc.ratings_model
    ?` The player-ratings engine alone predicts at ${mc.ratings_model.acc}% &mdash; ahead of team Elo &mdash; and feeds the blend as a feature.`:""}
    The season simulation re-runs at every serve (${sv.sims?sv.sims.toLocaleString(LOC)+" runs":"Monte Carlo"}); team strength evolves inside every run.</div>`;
  $("#view").innerHTML=`<div class="controls"><div class="rail">${siteRail("nfl")}</div><div class="filters">${filts}</div></div>
    ${vstrip}${vhold}${nflLegend()}${head}${cards}
    <div class="subh" style="margin-top:28px">Season outlook</div>
    ${model}
    <div class="grid" style="margin-top:10px">
      ${Object.keys(proj).length?`<div class="panel"><h3>Projected standings <span class="sub">top 8 &middot; <a class="tl" href="${nflH("standings")}">all 32 &rarr;</a></span></h3>
        <div class="twrap"><table><thead><tr><th></th><th class="a">Team</th><th>${rs}</th><th>Proj W-L</th><th>Div</th><th>Playoffs</th></tr></thead><tbody>${pTop}</tbody></table></div>
        <div class="sub" style="margin-top:8px">Mean of ${sv.sims?sv.sims.toLocaleString(LOC):"the"} season simulations of the market-blind model;
          played games are fixed at their results${sv.proj_through?` (through Week ${sv.proj_through.w})`:""}.</div></div>`
        :`<div class="panel"><h3>Projected standings</h3><div class="sub">The season projection is not in this build.</div></div>`}
      <div class="panel"><h3>Calibration <span class="sub">locked holdout</span></h3><div class="twrap"><table>
        <thead><tr><th class="a">Model says</th><th>Actually wins</th><th>Games</th></tr></thead><tbody>${cal}</tbody></table></div>
        <div class="sub" style="margin-top:8px">Pick-side probability bucket vs how often the pick actually won, on the locked holdout.${mc.accuracy!=null&&mc.acc_home!=null
          ?` Accuracy ladder: home-always ${mc.acc_home}% &middot; Elo ${mc.acc_elo}% &middot; <b>GlassBox ${mc.accuracy}%</b>
          &middot; closing line ${mc.acc_close}%.`:""}</div></div>
    </div>
    <div class="subh">${powTitle}</div>
    <div class="sub" style="margin-bottom:8px">Elo is our market-blind team rating, walked through every final. Unit values are EPA per
      play vs league &times;100 (+ = good) from the pass/run split ratings. GlassBox = the roster's snap-weighted per-play
      TrueSkill (50 = league average); Lineup # ranks the projected starting 22.</div>
    <div class="twrap"><table><thead><tr><th>#</th><th class="a">Team</th><th>${rs}</th><th>Elo</th>
      <th>Pass off</th><th>Run off</th><th>Pass def</th><th>Run def</th><th>GlassBox</th><th>Lineup #</th></tr></thead><tbody>${pow}</tbody></table></div>
    <div class="grid" style="margin-top:16px">${btbl("Top QBs","QB")}${btbl("Top RBs","RB")}${btbl("Top WRs","WR")}${btbl("Top TEs","TE")}</div>
    <div class="sub" style="margin-top:6px">Per-play value boards: opponent-adjusted value per play as a z-score within the position,
      ranked by the conservative floor so a small sample cannot top the list (${bs} season, fixed).</div>
    ${mvp?`<div class="grid" style="margin-top:16px"><div class="panel"><h3>MVP impact <span class="sub">with-vs-without, points per game</span></h3><div class="twrap"><table>
      <thead><tr><th></th><th class="a">Player</th><th>Cost when out</th><th>Absences</th></tr></thead><tbody>${mvp}</tbody></table></div>
      <div class="sub" style="margin-top:8px">Measured team drop-off when the player missed games, opponent-adjusted &mdash;
        ${siteEsc(n.mvp_source||"a historical study")}. Elite QBs land inside the market's 4.5&ndash;7.0 point range.</div></div></div>`:""}`;
  siteWireRail();
  $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.nflRange=x.dataset.r;state.nflWk=null;board();});
  $("#view").querySelectorAll("[data-wk]").forEach(x=>x.onclick=()=>{if(x.disabled) return;
    state.nflWk=parseInt(x.dataset.wk,10);state.nflRange="week";board();});
}

/* ---------- NFL SEASON (every week) ---------- */
function nflSeason(wk){
  const n=state.nfl, sch=(n.schedule||[]).slice().sort(nflByKick), today=nflToday();
  if(!sch.length){$("#view").innerHTML=`<div class="empty">The ${n.season} NFL schedule is not in this build.
    <a class="tl" href="${nflH("")}">NFL board</a></div>`;return;}
  const weeks=[...new Set(sch.map(g=>g.w))].sort((a,b)=>a-b), cw=nflCurWeek();
  let W=parseInt(wk,10); if(weeks.indexOf(W)<0) W=cw;
  const chips=weeks.map(w=>{const gs=sch.filter(g=>g.w===w), fin=gs.every(g=>g.hs!=null);
    return `<button class="filt ${w===W?"on":""}" data-w="${w}" title="Week ${w}${fin?" - final":(w===cw?" - this week":"")}">${w===cw?"&bull; ":""}W${w}</button>`;}).join("");
  const games=sch.filter(g=>g.w===W);
  const row=g=>{const c=nflCall(g), hp=c.hp;
    let res;
    if(c.done) res=`<span class="num">${g.away} ${g.as}&ndash;${g.hs} ${g.home}</span> ${nflGrade(g,c)}`;
    else if(g.d<today) res=`<span class="chip ovd" title="played; the result is not in this build yet">awaiting result</span>`;
    else res=`<span class="sub">&mdash;</span>`;
    return `<tr ${nflRow(nflGameHref(g))}>
      <td class="a"><span class="sub">${nflShortDay(g.d)} &middot; ${siteClock(g.t)}</span></td>
      <td class="a">${nflTL(g.away)} <span class="sub">@</span> ${nflTL(g.home)}${nflSiteTag(g)}</td>
      <td class="a">${nflTierChip(c.t)}</td>
      <td class="a">${nflCallCell(g,c)}</td>
      <td class="pbc"><div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div></td>
      <td><span class="num">${pctI(hp)}%</span></td>
      <td class="a">${res}</td></tr>`;};
  const nF=games.filter(g=>g.hs!=null).length, kc=nflCallCount(games);
  const tally=nflTally(games);
  $("#view").innerHTML=`<div class="eyebrow">NFL &middot; ${n.season} season</div>
    <h1 class="pt">Season <span class="sub">every game, forecast before kickoff &middot; market-blind</span></h1>
    <div class="sub" style="margin-bottom:10px">Inside a week of kickoff the <b>live model</b> runs on projected depth-chart lineups
      (${nflTierChip("PROJECTED")} &mdash; a PICK outside 55/45, a LEAN inside). Further out the number is a
      pre-information lean, mostly the <b>season simulation</b> from team strength (${nflTierChip("EARLY")} &mdash; scheduled, not a pick). The bar and the
      percentage are the <b>home</b> side. A played game keeps its pre-game number.
      <a class="tl" href="${nflH("record")}">Track record by tier</a>.</div>
    <div class="sub" style="margin-bottom:12px">Data: ${nflFreshLine()}</div>
    <div class="filters" style="flex-wrap:wrap;margin:0 0 12px">${chips}</div>
    <div class="subh">Week ${W} <span class="sub" style="font-family:var(--sans);font-size:13px;font-weight:400">&middot; ${games.length} games
      &middot; ${nflCallWords(kc)}${kc.nE?` &middot; ${kc.nE} scheduled`:""} &middot; ${nF} final</span></div>
    ${tally?`<div class="sub" style="margin-bottom:8px">This week, by tier: ${tally}</div>`:""}
    <div class="twrap nfl-season"><table><thead><tr><th class="a">Kickoff (ET)</th><th class="a">Game</th><th class="a">Tier</th>
      <th class="a">Call</th><th class="pbc">Home win prob</th><th>Home</th><th class="a">Result</th></tr></thead>
      <tbody>${games.map(row).join("")}</tbody></table></div>`;
  $("#view").querySelectorAll("[data-w]").forEach(x=>x.onclick=()=>{location.hash="#/season/"+x.dataset.w;});
}

/* ---------- NFL STANDINGS ---------- */
const NFL_POST={WC:"Wild Card",DIV:"Divisional",CON:"Conference",SB:"Super Bowl","SB won":"Won Super Bowl"};
function nflLuckCell(t){
  if(t.luck==null) return `<span class="sub">-</span>`;
  if((t.gp||0)<NFL_LUCK_MIN_G) return `<span class="sub" title="shown from ${NFL_LUCK_MIN_G} games (${t.gp||0} played): a smaller sample is noise">&ndash;</span>`;
  return `<span class="num ${Math.abs(t.luck)>=0.10?(t.luck>0?"warnc":"pos"):""}">${sgn(t.luck*100)}</span>`;
}
function nflStandings(){
  const n=state.nfl, T=n.teams||{}, proj=n.proj||{}, hasProj=Object.keys(proj).length>0;
  const rs=nflRS(), cur=rs===n.season, thr=n.standings_through;
  const divs={}; Object.values(T).forEach(t=>{(divs[t.div]=divs[t.div]||[]).push(t);});
  const order=(n.divisions||Object.keys(divs).sort()).filter(d=>divs[d]);
  const sv=(n.model_card&&n.model_card.serve)||{};
  const row=t=>{const p=proj[t.code]||t.proj||{};
    return `<tr ${nflRow(nflH("team",t.code))}>
      <td class="a"><span class="ab" style="color:var(--accent)">${t.code}</span> <span class="sub">${t.nick||t.name}</span></td>
      <td><span class="num">${nflRec(t)}</span></td><td><span class="num">${nflPct(t.pct)}</span></td>
      <td><span class="num ${t.pt_diff>0?"pos":(t.pt_diff<0?"neg":"")}">${nflSgn(t.pt_diff)}</span></td>
      <td><span class="num">${t.home||"-"}</span></td><td><span class="num">${t.away||"-"}</span></td>
      <td><span class="num">${t.div_rec||"-"}</span></td><td><span class="num">${t.streak||"-"}</span></td>
      ${hasProj?`<td><span class="num" title="${p.sd!=null?"&plusmn; "+p.sd+" wins (1 sd)":""}">${p.w!=null?p.w.toFixed(1)+"-"+(17-p.w).toFixed(1):"-"}</span></td>
      <td><span class="num">${nflPo(p.div)}</span></td><td><span class="num">${nflPo(p.po)}</span></td>`:""}
      <td>${gb(t.glassbox)}</td>
      <td><span class="num">${t.elo?t.elo.toFixed(0):"-"}</span> <span class="sub">#${t.rank||"-"}</span></td>
      <td>${nflLuckCell(t)}</td></tr>`;};
  const head=`<thead><tr><th class="a">Team</th><th>W-L</th><th>Pct</th><th>Diff</th><th>Home</th><th>Away</th><th>Div</th><th>Strk</th>
    ${hasProj?"<th>Proj W-L</th><th>Win div</th><th>Playoffs</th>":""}<th>GlassBox</th><th>Elo</th><th>Luck</th></tr></thead>`;
  const tables=order.map(d=>`<div class="subh">${d}</div><div class="twrap"><table>${head}
    <tbody>${divs[d].slice().sort((a,b)=>(a.div_rank||9)-(b.div_rank||9)).map(row).join("")}</tbody></table></div>`).join("");
  // the previous season's final table, collapsed (teams[].prev)
  const anyPrev=Object.values(T).some(t=>t.prev), ps=anyPrev?(Object.values(T).find(t=>t.prev).prev.season):null;
  const prevRow=t=>{const q=t.prev||{};
    return `<tr ${nflRow(nflH("team",t.code))}><td class="a"><span class="ab" style="color:var(--accent)">${t.code}</span> <span class="sub">${t.nick||t.name}</span></td>
      <td><span class="num">${nflRec(q)}</span></td><td><span class="num">${nflPct(q.pct)}</span></td>
      <td><span class="num">${q.pf!=null?q.pf:"-"}</span></td><td><span class="num">${q.pa!=null?q.pa:"-"}</span></td>
      <td><span class="num ${q.pt_diff>0?"pos":(q.pt_diff<0?"neg":"")}">${nflSgn(q.pt_diff)}</span></td>
      <td class="a"><span class="sub">${q.post?(NFL_POST[q.post]||siteEsc(q.post)):"&mdash;"}</span></td></tr>`;};
  const prev=anyPrev&&cur?`<details class="sched"><summary>${ps} final standings <span class="sub">&middot; last season's records, division order and playoff run</span></summary>
    <div class="body">${order.map(d=>`<div class="subh" style="margin-top:12px">${d}</div><div class="twrap"><table>
      <thead><tr><th class="a">Team</th><th>W-L</th><th>Pct</th><th>PF</th><th>PA</th><th>Diff</th><th class="a">Playoffs (last round)</th></tr></thead>
      <tbody>${divs[d].slice().sort((a,b)=>((a.prev||{}).div_rank||9)-((b.prev||{}).div_rank||9)).map(prevRow).join("")}</tbody></table></div>`).join("")}</div></details>`:"";
  $("#view").innerHTML=`<div class="eyebrow">Database &middot; NFL</div>
    <h1 class="pt">Standings <span class="sub">${rs}${cur?(thr&&thr.w?` &middot; through Week ${thr.w}`:" &middot; no games played yet"):" final"}${hasProj?` &middot; ${n.season} projections`:""}</span></h1>
    <div class="sub" style="margin-bottom:6px"><b>W-L</b> is the ${rs} regular season${cur&&thr&&thr.w?` through Week ${thr.w}`:""}; division order applies
      the NFL tiebreak steps. ${hasProj?`<b>Proj W-L</b>, <b>Win div</b> and <b>Playoffs</b> are the mean over ${sv.sims?sv.sims.toLocaleString(LOC):"the"}
      season simulations of the market-blind model, played games fixed at their results.`:""} <b>GlassBox</b> is the roster's
      snap-weighted per-play TrueSkill (50 = league average). <b>Elo</b> is our market-blind team rating (# = its rank).
      <b>Luck</b> = win% minus the Pythagorean expectation from points scored and allowed, in percentage points (amber = due to regress);
      it is shown from ${NFL_LUCK_MIN_G} games. Click any team.</div>
    <div class="sub" style="margin-bottom:12px">Data: ${nflFreshLine()}</div>
    ${hasProj?"":`<div class="polnote warn">The season projection is not in this build &mdash; showing records only.</div>`}
    ${tables}${prev}`;
}

/* ---------- NFL TEAMS ---------- */
function nflTeams(){
  const n=state.nfl, T=Object.values(n.teams||{}).sort((a,b)=>(b.glassbox||0)-(a.glassbox||0)), rs=nflRS();
  const qbShort=q=>{if(!q||!q.name) return "-"; const p=q.name.split(" "); return p.length>1?(p[0].indexOf(".")>=0?p[0]:p[0][0]+".")+" "+p.slice(1).join(" "):q.name;};
  const card=(t,i)=>{const p=t.proj;
    return `<div class="tcard" ${nflRow(nflH("team",t.code))}>
    <div class="h"><span class="code">${t.code}</span><span class="nm">${t.name}</span>${gb(t.glassbox)}</div>
    <div class="stat"><span>${rs} <b>${nflRec(t)}</b></span><span>Diff <b>${nflSgn(t.pt_diff)}</b></span>
      <span>Proj <b>${p?p.w.toFixed(1)+"W":"-"}</b></span><span>PO <b>${p?nflPo(p.po):"-"}</b></span></div>
    <div class="stat"><span>Off <b>${t.gb_off!=null?t.gb_off.toFixed(0):"-"}</b></span><span>Def <b>${t.gb_def!=null?t.gb_def.toFixed(0):"-"}</b></span>
      <span>QB <b>${siteEsc(qbShort(t.qb1))}</b></span></div>
    <div class="stat"><span>${t.div} #${t.div_rank||"-"}</span><span>Elo <b>${t.elo?t.elo.toFixed(0):"-"}</b> #${t.rank||"-"}</span><span>GlassBox #${i+1}</span></div>
  </div>`;};
  $("#view").innerHTML=`<div class="eyebrow">Database &middot; NFL</div><h1 class="pt">Teams</h1>
    <div class="sub" style="margin-bottom:6px">The <b>GlassBox rating</b> (0-100) is the roster's snap-weighted average <b>per-play TrueSkill</b>
      rating &mdash; every snap is an 11-vs-11 match, all 22 players Bayesian-updated on the outcome (${nflRatingsLine()}).
      50 = league average at the position; Off = offensive personnel, Def = defensive. Records are the ${rs} regular season;
      QB1 comes from the current depth chart. Sorted by GlassBox. ${nflFrozenNote()}</div>
    <div class="sub" style="margin-bottom:14px">Data: ${nflFreshLine()}</div>
    <div class="tgrid nflt">${T.map(card).join("")}</div>`;
}

/* ---------- NFL TEAM PAGE ---------- */
const NFL_GROUPS=[
  ["Offense",["QB","RB","WR","TE"]],["Offensive line",["OL"]],["Defense",["DL","LB","DB"]],["Special teams",["K","P","LS"]]];
/* Roster stat columns per group; `f` lists the families a missing key means 0
   for (the builder drops zero-valued keys), anyone else shows blank. */
const NFL_COLS={
  "Offense":[["Pass yds","pass_yds",["QB"]],["Pass TD","pass_td",["QB"]],["INT","ints",["QB"]],
    ["Rush yds","rush_yds",["QB","RB"]],["Rec","rec",["RB","WR","TE"]],["Rec yds","rec_yds",["RB","WR","TE"]],
    ["TD",s=>(s.rush_td||0)+(s.rec_td||0),["RB","WR","TE"]]],
  "Offensive line":[],
  "Defense":[["Tkl","tak",["DL","LB","DB"]],["Ast","ast",["DL","LB","DB"]],["TFL","tfl",["DL","LB","DB"]],
    ["Sacks","sk",["DL","LB","DB"]],["INT","dint",["DL","LB","DB"]],["PD","pd",["DL","LB","DB"]],["FF","ff",["DL","LB","DB"]]],
  "Special teams":[["FG",s=>s.fga!=null?`${s.fgm||0}/${s.fga}`:null,[]],["Long",s=>s.fg_long!=null?s.fg_long:(s.punt_long!=null?s.punt_long:null),[]],
    ["XP",s=>s.xpa!=null?`${s.xpm||0}/${s.xpa}`:null,[]],["Punts","punts",["P"]],["Avg",s=>s.punts?(s.punt_yds/s.punts).toFixed(1):null,[]],["In 20","punt_in20",["P"]]]
};
function nflRosterTable(title,players,st){
  if(!players.length) return "";
  const cols=NFL_COLS[title]||[], stSh=title==="Special teams";
  const cell=(p,[,k,fams])=>{const s=p.stats||{};
    if(!s.g) return `<td><span class="sub">-</span></td>`;
    let v=typeof k==="function"?k(s):s[k];
    if(v==null&&typeof k!=="function"&&fams.indexOf(p.fam)>=0) v=0;
    return `<td><span class="num">${v==null?"":nflNum(v)}</span></td>`;};
  const rows=players.map(p=>{const sh=stSh?p.st_share:p.snap_share;
    return `<tr ${nflRow(nflH("player",p.id))}>
      <td class="a"><span class="num" style="color:var(--faint)">${p.pos}</span></td>
      <td class="a"><span class="player-link" style="font-weight:600">${siteEsc(p.name)}</span>${p.num!=null?` <span class="sub">#${p.num}</span>`:""} ${nflStatusChip(p)}</td>
      <td${p.rating&&!stSh?` title="TrueSkill &mu; ${p.rating.mu.toFixed(1)} &plusmn; ${p.rating.sigma.toFixed(1)}, ${nflNum(Math.round(p.rating.n_eff))} plays rated, grade ${p.rating.tier===NFL_NOGRADE?"none":siteEsc(p.rating.tier)}"`:""}>${stSh?'<span class="sub" title="specialists are outside the 11v11 rating model">n/a</span>':(p.rating?gb(p.rating.r):gb(null))}</td>
      <td><span class="num">${sh!=null&&(p.stats||{}).g?Math.round(sh*100)+"%":"-"}</span></td>
      <td><span class="num">${(p.stats||{}).g||"-"}</span></td>
      ${cols.map(c=>cell(p,c)).join("")}</tr>`;}).join("");
  return `<div class="subh">${title} <span class="sub" style="font-family:var(--sans);font-size:13px;font-weight:400">${players.length}</span></div>
    <div class="twrap"><table><thead><tr><th class="a">Pos</th><th class="a">Player</th><th>GlassBox</th><th>${stSh?"ST snap %":"Snap %"}</th><th>G</th>
      ${cols.map(c=>`<th>${c[0]}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
    ${title==="Offensive line"?`<div class="sub" style="margin-top:4px">Linemen have no box-score line; the rating is their measure.</div>`:""}`;
}
function nflSortRoster(P){
  const fo=f=>{const i=NFL_FAMS.indexOf(f); return i<0?99:i;};
  return P.slice().sort((a,b)=>fo(a.fam)-fo(b.fam)
    ||((b.rating?b.rating.r:-1)-(a.rating?a.rating.r:-1))
    ||((b.snap_share||0)-(a.snap_share||0)));
}
function nflTeamPage(code){
  const n=state.nfl, t=(n.teams||{})[code];
  if(!t) return siteNotFound("nfl","team",code);
  const today=nflToday(), rs=nflRS(), st=nflSS(), cur=rs===n.season;
  const allT=Object.values(n.teams), gbRank=t.glassbox!=null?1+allT.filter(x=>(x.glassbox||0)>t.glassbox).length:null;
  const P=(t.roster||[]).map(id=>n.players[id]).filter(Boolean);
  const inGroup=fams=>nflSortRoster(P.filter(p=>fams.indexOf(p.fam)>=0));
  const other=P.filter(p=>!NFL_GROUPS.some(([,f])=>f.indexOf(p.fam)>=0));
  const roster=NFL_GROUPS.map(([ttl,f])=>nflRosterTable(ttl,inGroup(f),st)).join("")
    +(other.length?nflRosterTable("Other",other,st):"");
  const listTbl=(ids,ttl,sub)=>{const L=nflSortRoster((ids||[]).map(id=>n.players[id]).filter(Boolean)); if(!L.length) return "";
    const SL=n.status_labels||{};
    return `<details class="sched"><summary>${ttl} &middot; ${L.length} <span class="sub">${sub}</span></summary><div class="body"><div class="twrap"><table>
      <thead><tr><th class="a">Pos</th><th class="a">Player</th><th class="a">Status</th><th>GlassBox</th><th class="a">${nflSP()} season</th></tr></thead>
      <tbody>${L.map(p=>`<tr ${nflRow(nflH("player",p.id))}><td class="a"><span class="num" style="color:var(--faint)">${p.pos}</span></td>
        <td class="a"><span class="player-link" style="font-weight:600">${siteEsc(p.name)}</span></td>
        <td class="a"><span class="sub">${siteEsc(SL[p.status]||p.status)}</span></td>
        <td>${p.rating?gb(p.rating.r):gb(null)}</td><td class="a"><span class="sub">${nflStatShort(p,p.stats_prev)}</span></td></tr>`).join("")}
      </tbody></table></div></div></details>`;};
  // schedule with forecasts and results
  const sch=(n.schedule||[]).filter(g=>g.home===code||g.away===code).slice().sort(nflByKick);
  const played=new Set(sch.map(g=>g.w)), allW=[...new Set((n.schedule||[]).map(g=>g.w))].sort((a,b)=>a-b);
  const srow=g=>{const c=nflCall(g), homeG=g.home===code, opp=homeG?g.away:g.home, pw=homeG?c.hp:1-c.hp;
    let res;
    if(c.done){const my=homeG?g.hs:g.as, th=homeG?g.as:g.hs, wl=my>th?"W":(my<th?"L":"T");
      res=`<span class="num ${wl==="W"?"pos":(wl==="L"?"neg":"")}">${wl} ${my}-${th}</span> ${nflGrade(g,c)}`;}
    else if(g.d<today) res=`<span class="chip ovd">awaiting result</span>`;
    else res=`<span class="sub">&mdash;</span>`;
    return `<tr ${nflRow(nflGameHref(g))}><td><span class="num">W${g.w}</span></td>
      <td class="a"><span class="sub">${nflShortDay(g.d)} &middot; ${siteClock(g.t)}</span></td>
      <td class="a">${homeG?"vs":"@"} ${nflTL(opp)}${nflSiteTag(g)}</td>
      <td class="a">${nflTierChip(c.t)}</td>
      <td><span class="num ${pw>=0.5?"pos":"neg"}">${pctI(pw)}%</span></td>
      <td class="a">${nflCallCell(g,c)}</td><td class="a">${res}</td></tr>`;};
  const rowsSch=allW.map(w=>{const g=sch.find(x=>x.w===w);
    return g?srow(g):`<tr class="recrow"><td><span class="num">W${w}</span></td><td class="a" colspan="6"><span class="sub">bye</span></td></tr>`;}).join("");
  const tally=nflTally(sch);
  const schTbl=sch.length?`<div class="subh">${n.season} schedule <span class="sub" style="font-family:var(--sans);font-size:13px;font-weight:400">&middot; forecasts and results</span></div>
    ${tally?`<div class="sub" style="margin-bottom:6px">Model on ${t.code} games, by tier: ${tally}</div>`:""}
    <div class="twrap"><table><thead><tr><th>Wk</th><th class="a">Kickoff (ET)</th><th class="a">Opponent</th><th class="a">Tier</th>
      <th>${t.code} win</th><th class="a">Call</th><th class="a">Result</th></tr></thead><tbody>${rowsSch}</tbody></table></div>
    <div class="sub" style="margin-top:4px">EARLY rows are scheduled games (pre-information leans, mostly the season simulation; not picks); PROJECTED rows are the live
      model inside a week of kickoff. Played games keep their pre-game number.</div>`:"";
  const p26=t.proj||{}, q=t.qb1, prev=t.prev;
  const kpi=(k,v,cls)=>`<div class="b"><div class="k">${k}</div><div class="v ${cls||""}">${v}</div></div>`;
  const luck=(t.gp||0)>=NFL_LUCK_MIN_G&&t.luck!=null
    ?kpi("Pythag luck",sgn(t.luck*100),Math.abs(t.luck)>=0.10?"warnc":"")
    :kpi("Pythag luck",`<span class="sub" style="font-size:12px" title="shown from ${NFL_LUCK_MIN_G} games">&ndash; (${t.gp||0} G)</span>`);
  const inj=(t.inj||[]);
  const injPanel=inj.length?`<div class="panel"><h3>Injury report <span class="sub">Week ${t.inj_week||"?"}</span></h3><div class="twrap"><table><tbody>
      ${inj.map(i=>`<tr ${i.id&&n.players[i.id]?nflRow(nflH("player",i.id)):""}><td class="a">${nflPL(i.id,i.name)} <span class="sub">${siteEsc(i.pos||"")}</span></td>
        <td class="a"><span class="chip ${NFL_INJ_CLS[i.status]||"nfl-q"}">${siteEsc(i.status)}</span></td><td class="a"><span class="sub">${siteEsc(i.injury||"")}</span></td></tr>`).join("")}
    </tbody></table></div><div class="sub" style="margin-top:6px">Out and Doubtful players are removed from the projected lineup;
      Questionable players enter the forecast through the availability simulation.</div></div>`
    :`<div class="panel"><h3>Injury report <span class="sub">Week ${t.inj_week||"?"}</span></h3><div class="sub">No player listed on the latest report.</div></div>`;
  $("#view").innerHTML=`<a class="back" href="${nflH("teams")}">&lsaquo; Teams</a>
    <div class="thead"><span class="code">${t.code}</span>
      <div><div style="font-size:15px;font-weight:600">${t.name}</div>
        <div class="sub">${t.div} &middot; #${t.div_rank||"-"} &middot; ${nflRec(t)} (${nflPct(t.pct)}) in ${rs}${cur&&n.standings_through&&n.standings_through.w?` through Week ${n.standings_through.w}`:""}${prev&&prev.season!==rs?` &middot; ${prev.season}: ${nflRec(prev)}${prev.post?" ("+(NFL_POST[prev.post]||prev.post)+")":""}`:""}</div>
        <div class="sub">QB1 ${q&&q.id&&n.players[q.id]?nflPL(q.id,q.name):(q?siteEsc(q.name):"-")}${t.lineup&&t.lineup.dt?` &middot; depth chart ${nflShortDay(t.lineup.dt)}`:""}</div></div>
      <div style="margin-left:auto;text-align:right"><div class="sub">GlassBox rating</div>${gb(t.glassbox)}
        <div class="sub" style="margin-top:4px">off ${t.gb_off!=null?t.gb_off.toFixed(0):"-"} &middot; def ${t.gb_def!=null?t.gb_def.toFixed(0):"-"} &middot; #${gbRank||"-"} of 32</div></div>
    </div>
    <div class="tstats">
      ${kpi("Point diff",nflSgn(t.pt_diff),t.pt_diff>0?"pos":(t.pt_diff<0?"neg":""))}
      ${kpi("Pts/gm",nflPerG(t.pf,t.gp))}${kpi("Allowed/gm",nflPerG(t.pa,t.gp))}
      ${kpi("Home",t.home||"-")}${kpi("Away",t.away||"-")}${kpi("Division",t.div_rec||"-")}${kpi("Streak",t.streak||"-")}
      ${kpi("Our Elo",`${t.elo?t.elo.toFixed(0):"-"} <span class="sub">#${t.rank||"-"}</span>`)}
      ${kpi("Lineup TS",t.rpow_rank?`#${t.rpow_rank} <span class="sub">of 32</span>`:"-")}
      ${kpi("Proj wins",p26.w!=null?`${p26.w.toFixed(1)}${p26.sd!=null?` <span class="sub">&plusmn;${p26.sd}</span>`:""}`:"-")}
      ${kpi("Win division",nflPo(p26.div))}${kpi("Playoffs",nflPo(p26.po))}
      ${luck}
    </div>
    <div class="sub" style="margin:-6px 0 14px">EPA units per play vs league &times;100: pass off ${nflS1(t.off_pass)} &middot; run off ${nflS1(t.off_run)}
      &middot; pass def ${nflS1(t.def_pass)} &middot; run def ${nflS1(t.def_run)} (+ = good). Data: ${nflFreshLine()}</div>
    <div class="grid">${t.lineup?nflLineupPanel(code,t,{}):""}${injPanel}</div>
    ${schTbl}
    <div class="subh" style="margin-top:22px">Roster <span class="sub" style="font-family:var(--sans);font-size:13px;font-weight:400">&middot; ${P.length} active &middot; stats are the ${st} regular season</span></div>
    ${roster}
    <div style="margin-top:6px">${listTbl(t.reserve,"Reserve lists","injured reserve, PUP, exempt")}${listTbl(t.practice,"Practice squad","not on the active roster")}</div>`;
}

/* ---------- NFL PLAYER PAGE ---------- */
function nflStatBoxes(fam,s,career){
  if(!s||!Object.keys(s).length) return "";
  const v=k=>s[k]!=null?s[k]:0, box=(k,x)=>`<div class="b"><div class="k">${k}</div><div class="v">${x==null?"-":nflNum(x)}</div></div>`;
  const rate=(a,b,d)=>b?(a/b).toFixed(d==null?1:d):null, pct=(a,b)=>b?(100*a/b).toFixed(1):null;
  let L;
  switch(fam){
    case "QB": L=[["Cmp-Att",`${v("cmp")}-${v("att")}`],["Cmp%",pct(v("cmp"),v("att"))],["Pass yds",v("pass_yds")],["Y/A",rate(v("pass_yds"),v("att"))],
      ["TD",v("pass_td")],["INT",v("ints")],["Rush yds",v("rush_yds")],["Rush TD",v("rush_td")]]; break;
    case "RB": L=[["Car",v("car")],["Rush yds",v("rush_yds")],["YPC",rate(v("rush_yds"),v("car"))],["Rush TD",v("rush_td")],
      ["Rec",v("rec")],["Rec yds",v("rec_yds")],["Rec TD",v("rec_td")]]; break;
    case "WR": case "TE": L=[["Tgt",v("tgt")],["Rec",v("rec")],["Catch%",pct(v("rec"),v("tgt"))],["Rec yds",v("rec_yds")],
      ["Y/R",rate(v("rec_yds"),v("rec"))],["TD",v("rec_td")]].concat(s.car?[["Rush yds",v("rush_yds")]]:[]); break;
    case "K": L=[["FG",`${v("fgm")}/${v("fga")}`],["FG%",pct(v("fgm"),v("fga"))],["Long",s.fg_long!=null?s.fg_long:null],["XP",`${v("xpm")}/${v("xpa")}`]]; break;
    case "P": L=[["Punts",v("punts")],["Yds",v("punt_yds")],["Avg",rate(v("punt_yds"),v("punts"))],["In 20",v("punt_in20")],["Long",s.punt_long!=null?s.punt_long:null]]; break;
    case "OL": case "LS": L=[]; break;
    default: L=[["Tkl",v("tak")],["Ast",v("ast")],["TFL",v("tfl")],["Sacks",v("sk")],["QB hits",v("qbh")],["INT",v("dint")],["PD",v("pd")],["FF",v("ff")]];
  }
  const lead=career
    ?[["Seasons",career.seasons!=null?career.seasons:null],["Years",career.first!=null?`${career.first}&ndash;${String(career.last).slice(2)}`:null]]
    :[["G",s.g!=null?s.g:null]];
  return `<div class="statgrid">${lead.concat(L).map(([k,x])=>box(k,x)).join("")}</div>`;
}
function nflPlayerPage(id){
  const n=state.nfl, p=(n.players||{})[id];
  if(!p) return siteNotFound("nfl","player",id);
  const t=(n.teams||{})[p.team], st=nflSS(), sp=nflSP(), R=p.rating, SL=n.status_labels||{};
  const rc=(n.model_card&&n.model_card.ratings)||{};
  const spec=["K","P","LS"].indexOf(p.fam)>=0;
  const peers=R?Object.values(n.players).filter(x=>x.rating&&x.rating.bucket===R.bucket).sort((a,b)=>b.rating.r-a.rating.r):[];
  const rk=R?peers.indexOf(p)+1:null;
  const famBucket={QB:"QB",RB:"RB",WR:"WR",TE:"TE",OL:"OL",DL:"DL",LB:"LB",DB:"DB"}[p.fam];
  const ht=p.ht?`${Math.floor(p.ht/12)}-${p.ht%12}`:null;
  const bio=[p.age?`age ${p.age}`:null, ht&&p.wt?`${ht}, ${p.wt} lb`:null, p.college?siteEsc(p.college):null,
    p.exp!=null?(p.exp===0?"rookie":`${p.exp} yr${p.exp===1?"":"s"} experience`):null,
    p.draft?`drafted ${p.draft.year} round ${p.draft.round}, #${p.draft.pick}${p.draft.team&&p.draft.team!==p.team?" by "+p.draft.team:""}`:null
  ].filter(Boolean).join(" &middot; ");
  const status=p.inj&&p.inj.status
    ?`<div class="polnote warn">Week ${p.inj.week||"?"} injury report: <b>${siteEsc(p.inj.status)}</b>${p.inj.injury?` &mdash; ${siteEsc(p.inj.injury)}`:""}.${
      p.inj.status==="Questionable"?" Questionable players enter the forecast through the availability simulation.":" Out and Doubtful players are removed from the projected lineup."}</div>`
    :(p.status&&p.status!=="ACT"?`<div class="polnote">Roster status: <b>${siteEsc(SL[p.status]||p.status)}</b>${p.status==="DEV"?" &mdash; not on the 53-man active roster.":""}</div>`:"");
  const expl=R
    ?`GlassBox rating (0-100): this player's per-play 11-vs-11 TrueSkill, conservative (&mu;&thinsp;&minus;&thinsp;3&sigma;) and
      scored among ${peers.length} rated ${NFL_PL[R.bucket]} &mdash; 50 = an average ${NFL_ONE[R.bucket]}.${famBucket&&famBucket!==R.bucket
      ?` The roster lists ${p.pos}; the rating group comes from the player's nflverse position, so he is rated among ${NFL_PL[R.bucket]}.`:""} Market-blind.`
    :(spec?"Kickers, punters and long snappers are outside the 11-vs-11 rating model, so there is no GlassBox rating."
      :`No GlassBox rating: not enough rated snaps through ${rc.through_season||"the last rated season"}.`);
  const panel=(title,body,empty)=>`<div class="panel"><h3>${title}</h3>${body||`<div class="sub">${empty}</div>`}</div>`;
  const cur=nflStatBoxes(p.fam,p.stats), prv=nflStatBoxes(p.fam,p.stats_prev), car=p.career?nflStatBoxes(p.fam,p.career,p.career):"";
  const olNote=(p.fam==="OL"||p.fam==="LS")?`<div class="sub" style="margin-top:6px">${p.fam==="OL"?"Linemen have no box-score line; the rating is their measure.":"Long snappers have no box-score line."}</div>`:"";
  const sh=v=>v==null?"-":Math.round(v*100)+"%";
  const ratingPanel=R?`<div class="panel"><h3>GlassBox rating <span class="sub">per-play 11v11 TrueSkill</span></h3>
      <div style="display:flex;align-items:center;gap:14px;margin:2px 0 10px">${gb(R.r)}
        <span class="sub">#${rk} of ${peers.length} ${NFL_PL[R.bucket]} &middot; grade <b>${R.tier===NFL_NOGRADE?"&ndash;":siteEsc(R.tier)}</b></span></div>
      <div class="statgrid nfl-rgrid">
        <div class="b"><div class="k">&mu; &plusmn; &sigma;</div><div class="v">${R.mu.toFixed(1)}&thinsp;&plusmn;&thinsp;${R.sigma.toFixed(1)}</div></div>
        <div class="b"><div class="k">Floor &mu;&minus;3&sigma;</div><div class="v">${(R.mu-3*R.sigma).toFixed(1)}</div></div>
        <div class="b"><div class="k">Plays rated</div><div class="v">${nflNum(Math.round(R.n_eff))}</div></div>
      </div>
      <div class="nfl-legend">${nflGradeLegend()}. ${nflFrozenNote()}</div>
      ${p.board?`<div class="sub" style="margin-top:8px">Per-play value (${n.player_board_season||sp}): z <b>${nflS2(p.board.z)}</b>
        (floor ${nflS2(p.board.cons)}) over ${nflNum(p.board.n)} plays &mdash; opponent-adjusted value per play vs the position.</div>`:""}
    </div>`:(spec?"":panel("GlassBox rating",`<div style="display:flex;align-items:center;gap:14px">${gb(null)}<span class="sub">not rated</span></div>`));
  const snaps=spec
    ?`<div class="sub" style="margin-top:8px">${st} special-teams snap share <b>${sh((p.stats||{}).g?p.st_share:null)}</b></div>`
    :`<div class="sub" style="margin-top:8px">${st} snap share <b>${sh((p.stats||{}).g?p.snap_share:null)}</b> of offense/defense${
      p.st_share!=null?` &middot; special teams <b>${sh(p.st_share)}</b>`:""} &middot; ${sp}: <b>${sh(p.snap_prev)}</b></div>`;
  $("#view").innerHTML=`<a class="back" href="${nflH("team",p.team)}">&lsaquo; ${t?t.name:p.team}</a>
    <div class="phead"><span class="nm">${siteEsc(p.name)}</span>
      <span class="sub">${teamLink(p.team,t?t.name:p.team,"nfl")} &middot; ${p.pos}${p.num!=null?" &middot; #"+p.num:""} ${nflStatusChip(p)}</span>
      <span style="margin-left:auto">${spec?"":gb(R?R.r:null)}</span></div>
    ${bio?`<div class="nfl-bio">${bio}</div>`:""}
    <div class="sub" style="margin-bottom:12px">${expl}</div>
    ${status}
    <div class="pgrid">
      ${ratingPanel}
      ${panel(`${st} season`,(cur?cur+olNote:`<div class="sub">No ${st} regular-season snaps yet.</div>`)+snaps)}
      ${panel(`${sp} season`,prv?prv+olNote:"",`No ${sp} regular-season snaps.`)}
      ${car?panel(`Career <span class="sub">regular season</span>`,car):""}
    </div>`;
}

/* ---------- NFL PLAYERS INDEX ---------- */
function nflPlayers(){
  const n=state.nfl, ALL=Object.values(n.players||{}), P=ALL.filter(p=>p.rating), st=nflSS();
  const GROUPS=["QB","RB","WR","TE","OL","DL","LB","DB"];
  const byg={}; GROUPS.forEach(g=>byg[g]=[]);
  P.forEach(p=>{if(byg[p.rating.bucket]) byg[p.rating.bucket].push(p);});
  GROUPS.forEach(g=>byg[g].sort((a,b)=>b.rating.r-a.rating.r));
  const row=(p,i)=>`<tr ${nflRow(nflH("player",p.id))}>
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${p.pos} &middot; ${p.team}</span> ${nflStatusChip(p)}</td>
    <td>${gb(p.rating.r)}</td>
    <td><span class="num">${p.rating.mu.toFixed(1)}</span></td>
    <td class="a"><span class="sub">${nflStatShort(p)}</span></td></tr>`;
  const panel=g=>`<div class="panel"><h3 style="cursor:pointer" onclick="location.hash='${nflH("pos",g)}'">${NFL_GN[g]} <span class="sub">full ladder &rarr;</span></h3><div class="twrap"><table>
    <thead><tr><th></th><th class="a">Player</th><th>GlassBox</th><th class="nocase">&mu;</th><th class="a">${st} season</th></tr></thead>
    <tbody>${byg[g].slice(0,10).map(row).join("")}</tbody></table></div></div>`;
  const proven=P.filter(p=>p.rating.n_eff>=1000).sort((a,b)=>a.rating.sigma-b.rating.sigma).slice(0,8);
  const wild=P.filter(p=>p.rating.n_eff>=300).sort((a,b)=>b.rating.sigma-a.rating.sigma).slice(0,8);
  const mini=(p,i)=>`<tr ${nflRow(nflH("player",p.id))}>
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${p.rating.bucket} &middot; ${p.team}</span></td>
    <td>${gb(p.rating.r)}</td>
    <td><span class="num">&sigma; ${p.rating.sigma.toFixed(2)}</span></td></tr>`;
  const r=(n.model_card&&n.model_card.ratings)||{};
  $("#view").innerHTML=`<div class="eyebrow">Database &middot; NFL</div><h1 class="pt">Players</h1>
    <div class="sub" style="margin-bottom:10px">Every snap is an <b>11-vs-11 TrueSkill match</b> &mdash; all 22 players on the field are
      Bayesian-updated on whether the offense beat the defense (<b>${nflRatingsLine()}</b>, garbage time down-weighted).
      Ratings are conservative (&mu;&thinsp;&minus;&thinsp;3&sigma;) and scored within position group, so a thin sample can't fake it
      (50 = an average player at the position). <b>${ALL.length.toLocaleString(LOC)}</b> players on the 32 rosters, reserve lists and
      practice squads; <b>${P.length.toLocaleString(LOC)}</b> rated. Stat lines are the ${st} regular season. ${nflFrozenNote()}</div>
    <input class="psearch" id="psearch" placeholder="Search all ${ALL.length.toLocaleString(LOC)} players&hellip;" autocomplete="off">
    <div id="psres"></div>
    <div class="grid" style="margin-top:8px">${GROUPS.map(panel).join("")}</div>
    <div class="sub" style="margin-top:8px">${nflGradeLegend()}.</div>
    <div class="grid" style="margin-top:16px">
      <div class="panel"><h3>Bankable <span class="sub">lowest &sigma;, 1,000+ plays &mdash; the engine is surest about these ratings</span></h3>
        <div class="twrap"><table><tbody>${proven.map(mini).join("")}</tbody></table></div></div>
      <div class="panel"><h3>Wild cards <span class="sub">highest &sigma;, 300+ plays &mdash; the most room to move once new snaps are rated</span></h3>
        <div class="twrap"><table><tbody>${wild.map(mini).join("")}</tbody></table></div></div>
    </div>`;
  const res=$("#psres");
  $("#psearch").oninput=e=>{
    const q=norm(e.target.value);
    if(q.length<2){res.innerHTML="";return;}
    const hits=ALL.filter(p=>norm(p.name).includes(q))
      .sort((a,b)=>((b.rating?b.rating.r:-1)-(a.rating?a.rating.r:-1))).slice(0,20);
    res.innerHTML=hits.length?`<div class="twrap" style="margin-bottom:6px"><table><tbody>
      ${hits.map(p=>`<tr ${nflRow(nflH("player",p.id))}>
        <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${p.pos} &middot; ${p.team}</span> ${nflStatusChip(p)}</td>
        <td>${["K","P","LS"].indexOf(p.fam)>=0?'<span class="sub">specialist</span>':gb(p.rating?p.rating.r:null)}</td>
        <td class="a"><span class="sub">${nflStatShort(p)}</span></td></tr>`).join("")}
      </tbody></table></div>`:`<div class="sub" style="margin:6px 0 10px">No players match.</div>`;
  };
}

/* ---------- NFL POSITION LADDER ---------- */
function nflPosPage(key){
  const n=state.nfl;
  if(!n||!NFL_GN[key]) return nflPlayers();   // render in place: a hash redirect here loops with route()
  if(state.nflPosFor!==key){state.nflPosFor=key;state.nflPosSub="";}
  const st=nflSS(), rc=(n.model_card&&n.model_card.ratings)||{};
  const all=Object.values(n.players||{}).filter(p=>p.rating&&p.rating.bucket===key);
  const mMu=all.reduce((s,p)=>s+p.rating.mu,0)/Math.max(all.length,1);
  const mSig=all.reduce((s,p)=>s+p.rating.sigma,0)/Math.max(all.length,1);
  // sample chips from this position's own distribution (medians differ ~4x across positions)
  const ne=all.map(p=>p.rating.n_eff).sort((a,b)=>a-b), q=f=>ne.length?Math.round(ne[Math.floor(f*(ne.length-1))]/100)*100:0;
  const mins=[[0,"All"]].concat([...new Set([q(0.5),q(0.75)])].filter(x=>x>0).map(x=>[x,`${x.toLocaleString(LOC)}+ plays`]));
  if(!mins.some(([k])=>k===state.posMin)) state.posMin=0;
  const posCount={}; all.forEach(p=>{posCount[p.pos]=(posCount[p.pos]||0)+1;});
  const subs=Object.entries(posCount).filter(([,c])=>c>=3).sort((a,b)=>b[1]-a[1]).map(([k])=>k);
  const rooms={};
  all.forEach(p=>{const w=Math.max(p.snap_share||0,0.05);
    (rooms[p.team]=rooms[p.team]||[0,0]); rooms[p.team][0]+=w*p.rating.r; rooms[p.team][1]+=w;});
  const best=Object.entries(rooms).map(([t,[s,w]])=>[t,s/w]).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const row=(p,i)=>{const r=p.rating;
    return `<tr ${nflRow(nflH("player",p.id))}>
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${siteEsc(p.name)}</span> <span class="sub">${p.pos}</span> ${nflStatusChip(p)}</td>
    <td class="a">${nflTL(p.team)}</td>
    <td>${gb(r.r)}</td>
    <td><span class="num">${r.mu.toFixed(1)}&thinsp;&plusmn;&thinsp;${r.sigma.toFixed(1)}</span></td>
    <td><span class="num">${(r.mu-3*r.sigma).toFixed(1)}</span></td>
    <td><span class="num">${nflNum(Math.round(r.n_eff))}</span></td>
    <td><span class="num">${r.tier===NFL_NOGRADE?"&ndash;":siteEsc(r.tier)}</span></td>
    <td class="a"><span class="sub">${nflStatShort(p)}</span></td></tr>`;};
  const renderBody=()=>{
    const qy=norm(($("#posq")||{}).value||"");
    let P=all.filter(p=>p.rating.n_eff>=state.posMin&&(!state.nflPosSub||p.pos===state.nflPosSub));
    if(qy.length>=2) P=P.filter(p=>norm(p.name).includes(qy));
    const S={r:(a,b)=>b.rating.r-a.rating.r, mu:(a,b)=>b.rating.mu-a.rating.mu,
             sig:(a,b)=>a.rating.sigma-b.rating.sigma, n:(a,b)=>b.rating.n_eff-a.rating.n_eff};
    P.sort(S[state.posSort]||S.r);
    $("#posbody").innerHTML=P.map(row).join("")||`<tr><td class="sub" colspan="9">No players match.</td></tr>`;
    $("#posn").textContent=P.length;
  };
  $("#view").innerHTML=`<a class="back" href="${nflH("players")}">&lsaquo; Players</a>
    <div class="eyebrow">TrueSkill ladder &middot; NFL</div><h1 class="pt">${NFL_GN[key]}</h1>
    <div class="sub" style="margin-bottom:10px"><b>${all.length}</b> rated ${NFL_PL[key]} &middot; league &mu; ${mMu.toFixed(1)}, avg &sigma; ${mSig.toFixed(2)}
      &middot; ratings are &mu;&thinsp;&minus;&thinsp;3&sigma; scored within this position (50 = an average ${NFL_ONE[key]}); the floor is what the
      engine will vouch for. The rating group is the player's nflverse position, so a roster DE can sit among linebackers.
      ${nflGradeLegend()}. Stat line = ${st} regular season. ${nflFrozenNote()}</div>
    <div class="grid" style="margin-bottom:14px">
      <div class="panel"><h3>Rating distribution</h3>${histBars(all.map(p=>p.rating.r))}</div>
      <div class="panel"><h3>Best ${key} rooms <span class="sub">snap-weighted team average</span></h3>
        <div class="twrap"><table><tbody>${best.map(([t,v],i)=>`<tr ${nflRow(nflH("team",t))}>
          <td><span class="num">${i+1}</span></td><td class="a"><span class="ab" style="color:var(--accent)">${t}</span> <span class="sub">${n.teams[t]?n.teams[t].name:""}</span></td>
          <td>${gb(v)}</td></tr>`).join("")}</tbody></table></div></div>
    </div>
    <div class="controls" style="margin-bottom:10px">
      ${chipRow([["r","Rating"],["mu","Raw &mu;"],["sig","Most certain"],["n","Most plays"]],state.posSort,"psort")}
      ${chipRow(mins,state.posMin,"pmin")}
      ${subs.length>1?chipRow([["","All pos"]].concat(subs.map(s=>[s,`${s} ${posCount[s]}`])),state.nflPosSub,"psub"):""}
      <input class="psearch" id="posq" placeholder="Filter by name&hellip;" autocomplete="off" style="max-width:220px;margin:0">
    </div>
    <div class="sub" style="margin-bottom:6px"><span id="posn">${all.length}</span> shown</div>
    <div class="twrap"><table>
      <thead><tr><th></th><th class="a">Player</th><th class="a">Team</th><th>Rating</th><th class="nocase">&mu;&thinsp;&plusmn;&thinsp;&sigma;</th><th class="nocase">&mu;&minus;3&sigma;</th><th>Plays</th><th>Grade</th><th class="a">${st} season</th></tr></thead>
      <tbody id="posbody"></tbody></table></div>`;
  renderBody();
  $("#view").querySelectorAll("[data-psort]").forEach(x=>x.onclick=()=>{state.posSort=x.dataset.psort;nflPosPage(key);});
  $("#view").querySelectorAll("[data-pmin]").forEach(x=>x.onclick=()=>{state.posMin=parseInt(x.dataset.pmin,10)||0;nflPosPage(key);});
  $("#view").querySelectorAll("[data-psub]").forEach(x=>x.onclick=()=>{state.nflPosSub=x.dataset.psub||"";nflPosPage(key);});
  $("#posq").oninput=renderBody;
}

/* ---------- NFL GAME PAGE ---------- */
const NFL_CX_LBL={base:"Home edge (model avg.)",elo:"Team strength (Elo)",qb:"Quarterback",units:"Pass/run units (EPA)",
  roster:"Roster quality",ts:"Player ratings (11v11)",hfa:"Home field (team)",sched:"Rest &amp; body clock",
  luck:"Fumble luck",abs:"Absences",avail:"Questionable players"};
function nflQbLine(p){
  const st=nflSS(), sp=nflSP();
  if(!p) return `<span class="sub">not in this build's player table</span>`;
  const s=p.stats||{}, v=k=>s[k]||0, q=p.stats_prev||{}, w=k=>q[k]||0;
  const cur=s.g?`<div class="statline"><span>${st}</span><span>Cmp% <b>${v("att")?(100*v("cmp")/v("att")).toFixed(1):"-"}</b></span>
      <span>Yds <b>${nflNum(v("pass_yds"))}</b></span><span>TD <b>${v("pass_td")}</b></span><span>INT <b>${v("ints")}</b></span>
      <span>Y/A <b>${v("att")?(v("pass_yds")/v("att")).toFixed(1):"-"}</b></span><span>Rush <b>${v("rush_yds")}</b></span></div>`
    :`<div class="statline"><span>no ${st} snaps yet</span></div>`;
  const prev=q.g?`<div class="statline"><span>${sp}</span><span><b>${nflNum(w("pass_yds"))}</b> yds</span><span><b>${w("pass_td")}</b> TD</span>
      <span><b>${w("ints")}</b> INT</span><span>${q.g} G</span></div>`:"";
  return cur+prev;
}
function nflLineupPanel(code,t,opt){
  const n=state.nfl, L=t&&t.lineup; if(!L) return "";
  const st=nflSS(), age=L.dt?Math.round((Date.parse(nflToday()+"T12:00:00Z")-Date.parse(L.dt+"T12:00:00Z"))/864e5):null;
  const stale=age!=null&&age>7;
  const lrow=e=>{const p=e.id?n.players[e.id]:null;
    return `<tr ${p?nflRow(nflH("player",e.id)):""}>
    <td class="a"><span class="sub" style="font-size:11px;letter-spacing:.04em">${e.slot}</span></td>
    <td class="a"><span class="${p?"player-link":""}">${siteEsc(e.name)}</span>${p?" "+nflStatusChip(p):""}${e.src==="usage"
      ?` <span class="sub" title="promoted: real ${st} usage and rating beat the listed starter">&uarr; usage</span>`:""}</td>
    <td>${gb(e.r)}</td>
    <td><span class="num">${Math.round((e.share||0)*100)}%</span></td></tr>`;};
  const tbl=(side,title)=>L[side]&&L[side].length?`<div class="subh">${title}</div><div class="twrap"><table>
    <thead><tr><th class="a"></th><th class="a">Player</th><th>Rating</th><th>${st} snap %</th></tr></thead>
    <tbody>${L[side].map(lrow).join("")}</tbody></table></div>`:"";
  return `<div class="panel"><h3>${code} ${opt&&opt.title?opt.title:"projected lineup"} <span class="sub ${stale?"warnc":""}">depth chart ${L.dt?nflShortDay(L.dt):"?"}${stale?` &mdash; ${age} days old`:""}${L.inj_week?` &middot; injury report wk ${L.inj_week}`:""}</span></h3>
    ${tbl("off","Offense")}${tbl("def","Defense")}
    <div class="sub" style="margin-top:8px">Depth-chart starters with Out/Doubtful players removed, promoted (&uarr;) when real usage and
      rating say the chart is stale. Re-read at every model serve.</div></div>`;
}
function nflGamePage(key){
  const n=state.nfl, sch=n.schedule||[], today=nflToday();
  const parts=String(key).split("_"), w=parseInt(parts[0],10), away=parts[1], home=parts[2];
  const g=sch.find(x=>x.w===w&&x.away===away&&x.home===home);
  if(!g){$("#view").innerHTML=`<div class="empty">${sch.length?`No NFL game <b>${siteEsc(key)}</b> in the ${n.season} schedule.`
      :"The NFL schedule is not in this build."} <a class="tl" href="${nflH("")}">NFL board</a> &middot; <a class="tl" href="#/season">Season</a></div>`;return;}
  const A=n.teams[away]||{code:away,name:away}, H=n.teams[home]||{code:home,name:home};
  const c=nflCall(g), hp=c.hp, early=c.t==="EARLY", overdue=!c.done&&g.d<today, sv=(n.model_card&&n.model_card.serve)||{};
  const rs=nflRS(), rsCur=rs===n.season;
  // ---- projection / result
  const sub=(c.done||overdue)
    ?(c.call==="EARLY"?"EARLY: published more than a week before kickoff &mdash; a pre-information lean, not graded as a pick"
      :c.call==="LEAN"?"inside 55/45 &mdash; a lean, not graded as a pick":"PROJECTED: live model inside a week of kickoff &mdash; graded as a pick")
    :(early?`scheduled, not a pick &mdash; served more than a week before kickoff, ${c.exact?"team strength and the current depth chart":"season simulation from team strength only"};
        it becomes a pick at the first serve inside a week of kickoff`
      :c.call==="LEAN"?"inside 55/45 &mdash; the model has a side, not a case"
      :`live model &middot; projected depth-chart lineups${(A.inj_week||H.inj_week)?` and the Week ${H.inj_week||A.inj_week} injury report`:""}`);
  const callWord=early||c.call==="EARLY"?"Lean":(c.call==="LEAN"?"Lean":"Pick");
  let result="";
  if(c.done){
    const win=g.hs>g.as?home:(g.hs<g.as?away:null), pw=win===home?c.hp:(win===away?1-c.hp:null);
    result=`<div class="nfl-result">${win?`Model gave ${win} <b>${pctI(pw)}%</b> before kickoff &middot; log loss <b>${(-Math.log(Math.max(pw,1e-9))).toFixed(3)}</b>`
      :"Tie &mdash; no winner, not graded"} ${nflGrade(g,c)}</div>`;
  }
  const receipt=g.replay
    ?`<span class="chip replay">REPLAY</span> no pre-game number was archived for this game; the number is a walk-forward replay, kept out of every rate.`
    :(g.frozen_at?`Pre-game number locked ${nflAt(g.frozen_at)}${g.start_utc?` &mdash; ${(()=>{const h=(Date.parse(g.start_utc)-Date.parse(g.frozen_at))/36e5;
        return h>=48?Math.round(h/24)+" days":Math.max(0,Math.round(h))+" hours";})()} before kickoff`:""}, in the ${c.t} tier.`
      :(g.receipt_mismatch?"The ledger receipt did not match the published number, so this row is shown as EARLY."
        :(c.done?"":`The number published at kickoff is the one graded; the latest serve was ${nflAt(n.served_at)||"?"}.`)));
  const fair=!early&&!c.done&&!overdue?`<div class="sub" style="margin-top:8px">Fair odds (no vig, from the model): ${away} <span class="num">${amOdds(1-hp)}</span>
      &middot; ${home} <span class="num">${amOdds(hp)}</span></div>`:"";
  // ---- why: the exact breakdown of the live-model number
  const cx=g.cx, ORDER=(sv.cx_order&&sv.cx_order.length?sv.cx_order:["base","elo","qb","units","roster","ts","hfa","sched","luck","abs","avail"]);
  let why;
  if(cx){
    const ks=ORDER.filter(k=>cx[k]!=null&&(["base","elo","qb","units"].indexOf(k)>=0||Math.abs(cx[k])>=0.05));
    const sum=ORDER.reduce((s,k)=>s+(cx[k]||0),0);
    const notes=[];
    if(g.neutral) notes.push(`${g.intl?"International":"Neutral-site"} game: the listed home team has no crowd, but the locked model applies its intercept (${sgn(cx.base||0)}) to it anyway.`);
    if(g.early) notes.push(`${away} plays a ${siteClock(g.t)} kickoff after travelling east &mdash; the body-clock term.`);
    if(g.hrest!=null&&g.arest!=null&&g.hrest!==g.arest) notes.push(`Rest: ${away} ${g.arest} days, ${home} ${g.hrest} days.`);
    const Q=g.q?[].concat((g.q.a||[]).map(x=>[away,x]),(g.q.h||[]).map(x=>[home,x])):[];
    if(Q.length) notes.push(`The availability simulation toggled ${Q.map(([tm,x])=>`${nflPL(x.id,x.name)} (${tm} ${siteEsc(x.pos||"")})`).join(", ")} &mdash; Questionable on the injury report.`);
    why=`<h3 style="margin-top:16px">Why &mdash; exact contributions <span class="sub">${c.exact?"":"of the live model's "+pctI(g.ph)+"%"}</span></h3>
      <div class="nfl-dir"><span>&#9664; ${away}</span><span class="c">home win-probability points</span><span>${home} &#9654;</span></div>
      <div class="why nflwhy">${ks.map(k=>pRow(NFL_CX_LBL[k]||k,cx[k])).join("")}</div>
      <div class="sub" style="margin-top:8px">50 + bars = <b>${(50+sum).toFixed(1)}%</b>, the live model's ${home} number. Each bar is the
        exact change in probability when its group is added, in this order, so the bars add up.${c.exact?"":` The lean above is the
        season simulation's ${pctI(hp)}%, which also plays out every earlier game &mdash; the bars explain the live model, not the simulation.`}</div>
      ${notes.length?`<div class="nfl-defs">${notes.join(" ")}</div>`:""}
      <div class="nfl-defs"><b>Home edge</b> = the blend's intercept, the average home advantage. <b>Elo</b> = team strength.
        <b>Quarterback</b> = the QB1s' rating gap. <b>Units</b> = EPA per play, pass and run, offense against defense (grouped: their split
        is collinear). <b>Roster quality</b> = the expected lineup's production on the current roster. <b>Player ratings</b> = the per-snap
        11v11 TrueSkill of the projected lineups. <b>Home field (team)</b> = this stadium's edge beyond the average. <b>Rest &amp; body
        clock</b> = rest-day gap plus a western team in an early eastern kickoff. <b>Fumble luck</b> = fumble-recovery luck, which regresses.
        <b>Questionable players</b> = the availability simulation over this week's Questionable tags. Market-blind.</div>`;
  }else{
    // A row the serve could not reconcile carries no cx: fall back to the
    // slope-scaled ct, which is a linearisation and says so.
    const ct=g.ct||{}, ks=["elo","qb","units","roster","ts","hfa","sched","luck","abs","avail"].filter(k=>ct[k]!=null&&Math.abs(ct[k])>=0.05);
    why=`<h3 style="margin-top:16px">Why &mdash; blend contributions</h3><div class="nfl-approx"><div class="sub">Approximate: each bar
        is one group's push scaled to probability points. They do <b>not</b> add up to the number above &mdash; the blend combines
        them through a logistic curve.</div></div>
      <div class="why nflwhy">${ks.map(k=>pRow(NFL_CX_LBL[k],ct[k])).join("")||'<div class="sub">no contribution data</div>'}</div>`;
  }
  // ---- quarterbacks
  const qb=nflGameQbs(g);
  const qcol=(o,side,team)=>{const p=o.q&&o.q.id?n.players[o.q.id]:null, R=p&&p.rating;
    return `<div class="col ${side}"><div class="nm">${o.q?nflPL(o.q.id,o.q.name):"-"}</div>
      <div class="sub">${teamLink(team,team,"nfl")}${R?` &middot; GlassBox <b>${R.r.toFixed(0)}</b> &middot; &mu; ${R.mu.toFixed(1)} &plusmn; ${R.sigma.toFixed(1)}`:(p?" &middot; not rated":"")}</div>
      ${o.q?nflQbLine(p):""}</div>`;};
  const swap=c.done?["a","h"].filter(s=>qb[s].fq&&qb[s].q&&qb[s].fq.id!==qb[s].q.id)
    .map(s=>`the forecast assumed ${siteEsc(qb[s].fq.name)} for ${s==="a"?away:home}; ${siteEsc(qb[s].q.name)} started`):[];
  const qbPanel=`<div class="panel nflqb"><h3>${c.done?"Starting quarterbacks":"Quarterbacks"} <span class="sub">${c.done?"who started":"the QB1s this forecast uses"}</span></h3>
    <div class="mup">${qcol(qb.a,"a",away)}<div class="vs">vs</div>${qcol(qb.h,"h",home)}</div>
    <div class="sub" style="margin-top:10px">GlassBox = per-play 11v11 TrueSkill, conservative and position-normalized (50 = average QB).
      ${swap.length?"Note: "+swap.join("; ")+".":""}</div></div>`;
  // ---- head to head
  const f0=x=>x==null?"-":Math.round(+x), gp=o=>o.gp||0;
  const hasRec=rsCur&&(gp(A)||gp(H));
  const teamCmp=`<table class="cmp">
    ${cmpRow(`${rs} record`,nflRec(A),nflRec(H))}
    ${hasRec?cmpRow("win%",A.pct,H.pct,nflPct,"hi"):""}
    ${hasRec?cmpRow("point diff",A.pt_diff,H.pt_diff,nflSgn,"hi"):""}
    ${hasRec?cmpRow("pts / gm",gp(A)?A.pf/gp(A):null,gp(H)?H.pf/gp(H):null,x=>x==null?"-":x.toFixed(1),"hi"):""}
    ${hasRec?cmpRow("allowed / gm",gp(A)?A.pa/gp(A):null,gp(H)?H.pa/gp(H):null,x=>x==null?"-":x.toFixed(1),"lo"):""}
    ${hasRec?cmpRow("away / home",A.away||"-",H.home||"-"):""}
    ${hasRec?cmpRow("streak",A.streak||"-",H.streak||"-"):""}
    ${A.prev&&H.prev&&A.prev.season!==rs?cmpRow(`${A.prev.season} record`,nflRec(A.prev),nflRec(H.prev)):""}
    ${cmpRow("our Elo",A.elo,H.elo,f0,"hi")}
    ${cmpRow("Elo rank",A.rank?"#"+A.rank:"-",H.rank?"#"+H.rank:"-")}
    ${cmpRow("roster GlassBox",A.glassbox,H.glassbox,x=>x==null?"-":(+x).toFixed(0),"hi")}
    ${cmpRow("off / def GB",`${A.gb_off!=null?A.gb_off.toFixed(0):"-"} / ${A.gb_def!=null?A.gb_def.toFixed(0):"-"}`,`${H.gb_off!=null?H.gb_off.toFixed(0):"-"} / ${H.gb_def!=null?H.gb_def.toFixed(0):"-"}`)}
    ${A.rpow_rank&&H.rpow_rank?cmpRow("lineup TS rank","#"+A.rpow_rank,"#"+H.rpow_rank):""}
    ${cmpRow("pass off EPA",A.off_pass,H.off_pass,nflS1,"hi")}
    ${cmpRow("run off EPA",A.off_run,H.off_run,nflS1,"hi")}
    ${cmpRow("pass def EPA",A.def_pass,H.def_pass,nflS1,"hi")}
    ${cmpRow("run def EPA",A.def_run,H.def_run,nflS1,"hi")}
    ${cmpRow("rest days",g.arest,g.hrest,x=>x==null?"-":x,"hi")}
    ${A.proj&&H.proj?cmpRow(`proj ${n.season} wins`,A.proj.w,H.proj.w,x=>x==null?"-":x.toFixed(1),"hi"):""}
    ${A.proj&&H.proj?cmpRow("playoff odds",A.proj.po,H.proj.po,nflPo,"hi"):""}
  </table>`;
  // ---- signals
  const sig=[];
  const ra=qb.a.q?nflRated(qb.a.q.id):null, rh=qb.h.q?nflRated(qb.h.q.id):null;
  if(ra!=null&&rh!=null&&Math.abs(rh-ra)>=8){const hb=rh>ra;
    sig.push([hb?"good":"warn",`QB edge: ${hb?home:away} (${siteEsc((hb?qb.h:qb.a).q.name)}, GlassBox ${(hb?rh:ra).toFixed(0)} vs ${(hb?ra:rh).toFixed(0)})`]);}
  if(g.hrest!=null&&g.arest!=null){const d=g.hrest-g.arest;
    if(Math.abs(d)>=3){const more=d>0?home:away, a=Math.max(g.hrest,g.arest), b=Math.min(g.hrest,g.arest);
      sig.push([d>0?"good":"warn",`Rest edge: ${more} has ${a} days' rest vs ${b}${a>=13?" (off a bye)":""}`]);}}
  if(g.early) sig.push(["warn",`Body clock: ${away} travels east for a ${siteClock(g.t)} kickoff`]);
  if(g.neutral) sig.push(["warn",`${g.intl?"International":"Neutral-site"} game: ${home} is the listed home team without a home crowd`]);
  if(!c.done){[[away,A],[home,H]].forEach(([code,T])=>{if(T.inj_week!==g.w) return;
    (T.inj||[]).filter(i=>i.status==="Out"||i.status==="Doubtful").forEach(i=>{const r=nflRated(i.id);
      if((i.pos==="QB")||(r!=null&&r>=60)) sig.push(["warn",`${code}: ${siteEsc(i.name)} (${siteEsc(i.pos||"")}${r!=null?", GlassBox "+r.toFixed(0):""}) is ${i.status}`]);});});}
  if(rsCur&&gp(A)>=4&&gp(H)>=4){
    const all=Object.values(n.teams), recRank=t=>1+all.filter(x=>(x.pct||0)>(t.pct||0)).length;
    [[home,H],[away,A]].forEach(([code,T])=>{if(T.rank&&T.rank+8<=recRank(T))
      sig.push(["good",`We rate ${code} higher than its record (Elo #${T.rank} vs #${recRank(T)} by win%)`]);});}
  [[home,H],[away,A]].forEach(([code,T])=>{if(rsCur&&gp(T)>=NFL_LUCK_MIN_G&&T.luck!=null&&Math.abs(T.luck)>=0.10)
    sig.push([T.luck>0?"warn":"good",`${code} is ${T.luck>0?"outperforming":"underperforming"} its point differential (${sgn(T.luck*100)} pts of win%)${T.luck>0?" &mdash; regression risk":" &mdash; due to bounce back"}`]);});
  if(cx&&Math.abs(cx.luck||0)>=1) sig.push(["warn",`Fumble luck moves this number ${sgn(cx.luck)} pts toward ${cx.luck>0?home:away} &mdash; it regresses`]);
  // ---- injuries (this week's report, unplayed games in that week)
  const injList=(code,T)=>{const L=(T.inj||[]); if(!L.length) return `<div class="sub">${code}: nobody listed.</div>`;
    return `<div class="sub" style="margin:6px 0 2px"><b>${code}</b></div>${L.map(i=>`<div class="nfl-injr"><span class="chip ${NFL_INJ_CLS[i.status]||"nfl-q"}">${siteEsc(i.status)}</span>
      ${nflPL(i.id,i.name)} <span class="sub">${siteEsc(i.pos||"")}${i.injury?" &middot; "+siteEsc(i.injury):""}</span></div>`).join("")}`;};
  const injPanel=!c.done&&(A.inj_week===g.w||H.inj_week===g.w)
    ?`<div class="panel"><h3>Injury report <span class="sub">Week ${g.w}</span></h3>${injList(away,A)}${injList(home,H)}
      <div class="sub" style="margin-top:8px">Out/Doubtful players are removed from the projected lineups; Questionable players enter through the
        availability simulation.</div></div>`:"";
  const val=nflValBar(g);
  const live=c.done
    ?`<div class="livepanel" style="display:flex"><span class="sc">${away} ${g.as} &ndash; ${g.hs} ${home}</span><span class="st">Final</span></div>`
    :`<div class="livepanel" id="livepanel" data-ng="${away}_${home}" data-d="${g.d}" data-away="${away}" data-home="${home}"></div>`;
  const lineups=!c.done&&(A.lineup||H.lineup)
    ?`<div class="grid" style="margin-top:14px">${nflLineupPanel(away,A,{title:early?"current lineup":"projected lineup"})}${nflLineupPanel(home,H,{title:early?"current lineup":"projected lineup"})}</div>`
    :(c.done?`<div class="sub" style="margin-top:14px">Lineup panels show today's depth chart, so they are hidden on a played game; the
        quarterbacks above are the game's actual starters.</div>`:"");
  $("#view").innerHTML=`<a class="back" href="${nflH("")}">&lsaquo; NFL board</a>
    <a class="back" href="#/season/${g.w}" style="margin-left:14px">&lsaquo; Week ${g.w}</a>
    <div class="gh"><span class="mt">${teamLink(away,away,"nfl")} <span style="color:var(--faint)">@</span> ${teamLink(home,home,"nfl")}</span>
      <span class="meta">Week ${g.w} &middot; ${fmtDay(g.d,today)} &middot; ${siteClock(g.t)}${g.neutral?(g.intl?" &middot; international, neutral site":" &middot; neutral site"):""}</span></div>
    <div class="sub" style="margin-bottom:10px">${A.name} at ${H.name} &middot; ${nflTierChip(c.t)}${overdue?` &middot; <span class="chip ovd">played &mdash; result not in this build yet</span>`:""}
      &middot; Data: ${nflFreshLine()}</div>
    ${live}
    <div class="cols">
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="panel"><h3>${c.done?"Pre-game forecast":(overdue?(early?"Pre-game lean":"Pre-game forecast"):(early?"Scheduled &mdash; current lean":"Our projection"))}</h3>
          ${val}
          <div class="proj"><div class="big">${pctI(hp)}<span class="u">%</span></div>
            <div class="pk">${home} win probability<b>${callWord}: ${c.pick} <span class="cf">${pctI(c.pp)}%</span></b>
              <span class="sub">${sub}</span></div></div>
          <div class="bigbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
          <div class="barlab"><span>${away} ${pctI(1-hp)}%</span><span>${pctI(hp)}% ${home}</span></div>
          ${fair}${result}
          ${receipt?`<div class="sub" style="margin-top:8px">${receipt}</div>`:""}
          ${why}
        </div>
        ${qbPanel}
      </div>
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="panel"><h3>${away} vs ${home}${c.done?' <span class="sub">team numbers as of today, not at kickoff</span>':""}</h3>${teamCmp}</div>
        <div class="panel"><h3>Signals</h3><div class="signals">
          ${sig.length?sig.map(([k,x])=>`<div class="sig ${k}"><span class="ic">!</span><span>${x}</span></div>`).join(""):'<div class="sub">No standout signals.</div>'}
        </div></div>
        ${injPanel}
      </div>
    </div>
    ${lineups}`;
}
