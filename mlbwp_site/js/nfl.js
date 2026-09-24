/* NFL: board, season, standings, teams, players, positions, game page. */
/* ---------- NFL (preseason) ---------- */
function nflPage(){
  const n=state.nfl, mc=n.model_card, b=state.board;
  const rail=b.leagues.map(l=>{
    if(l.active)
      return `<button class="lg" data-lg="${l.code}"><span>${l.name}</span><span class="n">${l.n_games}</span></button>`;
    if(l.code==="nfl")
      return `<button class="lg on" data-lg="nfl"><span>NFL</span><span class="n">2026</span></button>`;
    return `<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`;}).join("")+(state.nhl?nhlRailBtn(state.league==="nhl"):"");
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
        <span class="who"><span class="sp">${qbOf(g.away)}</span></span><span class="odds" title="fair American odds (no vig) from the model">${amOdds(1-hp)}</span><span class="pc">${pctI(1-hp)}%</span></div>
      <div class="side ${homeWin?"win":""}"><span class="ab">${g.home}</span>
        <span class="who"><span class="sp">${qbOf(g.home)}</span></span><span class="odds" title="fair American odds (no vig) from the model">${amOdds(hp)}</span><span class="pc">${pctI(hp)}%</span></div>
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
      &middot; ${mc.n_features} features &middot; ${mc.n_tests} documented tests &middot; ${mc.training}.${mc.ratings_model?` The <b>player-ratings engine alone</b> predicts at ${mc.ratings_model.acc}% accuracy &mdash; beating team Elo &mdash; and feeds the model as a feature.`:""}</div>
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
  const panel=g=>`<div class="panel"><h3 style="cursor:pointer" onclick="location.hash='#/pos/${g}'">${GN[g]} <span class="sub" style="font-weight:400">full ladder &rarr;</span></h3><div class="twrap"><table>
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

function nflPosPage(key){
  const n=state.nfl;
  const GN={QB:"Quarterbacks",RB:"Running backs",WR:"Receivers",TE:"Tight ends",
            OL:"Offensive line",DL:"Defensive line",LB:"Linebackers",DB:"Secondary"};
  if(!n||!GN[key]){location.hash="#/players";return;}
  const all=Object.values(n.players||{}).filter(p=>p.rating&&p.rating.bucket===key);
  const mMu=all.reduce((s,p)=>s+p.rating.mu,0)/Math.max(all.length,1);
  const mSig=all.reduce((s,p)=>s+p.rating.sigma,0)/Math.max(all.length,1);
  const rooms={};
  all.forEach(p=>{const w=Math.max(p.snap_share||0,0.05);
    (rooms[p.team]=rooms[p.team]||[0,0]); rooms[p.team][0]+=w*p.rating.r; rooms[p.team][1]+=w;});
  const best=Object.entries(rooms).map(([t,[s,w]])=>[t,s/w]).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const row=(p,i)=>{const r=p.rating;
    return `<tr onclick="location.hash='#/player/${p.id}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span></td>
    <td>${teamLink(p.team)}</td>
    <td>${gb(r.r)}</td>
    <td><span class="num">${r.mu.toFixed(1)}&thinsp;&plusmn;&thinsp;${r.sigma.toFixed(1)}</span></td>
    <td><span class="num">${(r.mu-3*r.sigma).toFixed(1)}</span></td>
    <td><span class="num">${r.n_eff.toLocaleString()}</span></td>
    <td><span class="num">${r.tier}</span></td></tr>`;};
  const renderBody=()=>{
    const q=norm(($("#posq")||{}).value||"");
    let P=all.filter(p=>p.rating.n_eff>=state.posMin);
    if(q.length>=2) P=P.filter(p=>norm(p.name).includes(q));
    const S={r:(a,b)=>b.rating.r-a.rating.r, mu:(a,b)=>b.rating.mu-a.rating.mu,
             sig:(a,b)=>a.rating.sigma-b.rating.sigma, n:(a,b)=>b.rating.n_eff-a.rating.n_eff};
    P.sort(S[state.posSort]||S.r);
    $("#posbody").innerHTML=P.map(row).join("")||`<tr><td class="sub" colspan="8">No players match.</td></tr>`;
    $("#posn").textContent=P.length;
  };
  $("#view").innerHTML=`<a class="back" href="#/players">&lsaquo; Players</a>
    <div class="eyebrow">TrueSkill ladder &middot; NFL</div><h1 class="pt">${GN[key]}</h1>
    <div class="sub" style="margin-bottom:10px"><b>${all.length}</b> rated ${GN[key].toLowerCase()} &middot; league &mu; ${mMu.toFixed(1)}, avg &sigma; ${mSig.toFixed(2)}
      &middot; ratings are &mu;&thinsp;&minus;&thinsp;3&sigma; z-scored within this position (50 = average ${key}). Conservative = the floor the engine will vouch for.</div>
    <div class="grid" style="margin-bottom:14px">
      <div class="panel"><h3>Rating distribution</h3>${histBars(all.map(p=>p.rating.r))}</div>
      <div class="panel"><h3>Best ${key} rooms <span class="sub" style="font-weight:400">snap-weighted team average</span></h3>
        <div class="twrap"><table><tbody>${best.map(([t,v],i)=>`<tr onclick="location.hash='#/team/${t}'">
          <td><span class="num">${i+1}</span></td><td class="a"><span class="ab" style="color:var(--accent)">${t}</span> <span class="sub">${n.teams[t]?n.teams[t].name:""}</span></td>
          <td>${gb(v)}</td></tr>`).join("")}</tbody></table></div></div>
    </div>
    <div class="controls" style="margin-bottom:10px">
      ${chipRow([["r","Rating"],["mu","Raw μ"],["sig","Most certain"],["n","Most plays"]],state.posSort,"psort")}
      ${chipRow([[0,"All"],[500,"500+ plays"],[1500,"1,500+ plays"]],state.posMin,"pmin")}
      <input class="psearch" id="posq" placeholder="Filter by name&hellip;" autocomplete="off" style="max-width:220px;margin:0">
    </div>
    <div class="sub" style="margin-bottom:6px"><span id="posn">${all.length}</span> shown</div>
    <div class="twrap"><table>
      <thead><tr><th></th><th>Player</th><th>Team</th><th>Rating</th><th class="nocase">&mu;&thinsp;&plusmn;&thinsp;&sigma;</th><th class="nocase">&mu;&minus;3&sigma;</th><th>Plays</th><th>Tier</th></tr></thead>
      <tbody id="posbody"></tbody></table></div>`;
  renderBody();
  $("#view").querySelectorAll("[data-psort]").forEach(x=>x.onclick=()=>{state.posSort=x.dataset.psort;nflPosPage(key);});
  $("#view").querySelectorAll("[data-pmin]").forEach(x=>x.onclick=()=>{state.posMin=parseInt(x.dataset.pmin);nflPosPage(key);});
  $("#posq").oninput=renderBody;
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
             ts:"Player ratings (11v11 TS)", luck:"Fumble luck", abs:"Injuries / absences", avail:"Questionable players (availability sim)"};
  const ORDER=["elo","qb","units","roster","hfa","sched","ts","luck","abs","avail"];
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
    ${(A&&A.rpow_rank)?cmpRow("lineup TS power","#"+A.rpow_rank,"#"+H.rpow_rank,x=>x):""}
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
          <span class="who"><span class="sp">${qcell(A)}</span></span><span class="odds" title="fair American odds (no vig) from the model">${amOdds(1-hp)}</span><span class="pc">${pctI(1-hp)}%</span></div>
        <div class="side ${homeWin?"win":""}"><span class="ab">${teamLink(home)}</span>
          <span class="who"><span class="sp">${qcell(H)}</span></span><span class="odds" title="fair American odds (no vig) from the model">${amOdds(hp)}</span><span class="pc">${pctI(hp)}%</span></div>
        <div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
        <div class="sub" style="margin-top:10px">Model ${pctI(g.ph)}% home &middot; simulation ${pctI(g.pmc!=null?g.pmc:g.ph)}% home.
          Games inside a week serve the model; further out, the 20,000-run sim (team strength evolves inside every sim).</div></div>
      <div class="panel why"><h3>Why &mdash; blend contributions <span class="sub" style="font-weight:400">home prob points</span></h3>
        ${whyRows||`<div class="sub">no contribution data</div>`}
        <div class="sub" style="margin-top:10px">Each bar is one feature group's push, taken from the market-blind blend's own
          coefficients and scaled to home-probability points. They rank and size the model's reasons faithfully, but they do
          <b>not</b> add up to the probability above &mdash; the blend combines them through a logistic curve, not by addition.
          Pass/run/EPA are grouped (their split is collinear; the sum is what the model believes).</div></div>
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

