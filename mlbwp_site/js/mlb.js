/* MLB pages: players, positions, game, standings, teams, player.
   Every link these pages draw is league-qualified (siteHref("mlb",...)): a bare
   #/team/SEA opened in a new tab or shared resolves against the READER's last
   league and opened the Seattle Seahawks or Kraken instead of the Mariners. */
/* ---------- PLAYERS TAB ---------- */
function playersPage(){
  if(state.league==="nfl"&&state.nfl) return nflPlayers();
  if(state.league==="nhl"&&state.nhl) return nhlPlayers();
  const db=state.db, P=Object.values(db.players||{});
  const bats=P.filter(p=>p.role==="batter"&&p.ts100!=null).sort((a,b)=>b.ts100-a.ts100);
  const pits=P.filter(p=>p.role==="pitcher"&&p.ts100!=null).sort((a,b)=>b.ts100-a.ts100);
  const brow=(p,i)=>`<tr onclick="location.hash='${siteHref("mlb","player",p.id)}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
    <td>${gb(p.ts100)}</td>
    <td><span class="num">${p.ts_mu!=null?p.ts_mu.toFixed(1):"-"}</span></td>
    <td class="sub">${p.bat?`${p.bat.ops!=null?p.bat.ops:"-"} OPS &middot; ${p.bat.hr||0} HR &middot; ${p.bat.sb||0} SB`:"&mdash;"}</td></tr>`;
  const prow=(p,i)=>`<tr onclick="location.hash='${siteHref("mlb","player",p.id)}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
    <td>${gb(p.ts100)}</td>
    <td><span class="num">${p.ts_mu!=null?p.ts_mu.toFixed(1):"-"}</span></td>
    <td class="sub">${p.pit?`${p.pit.era!=null?p.pit.era:"-"} ERA &middot; ${p.pit.whip!=null?p.pit.whip:"-"} WHIP &middot; ${p.pit.so||0} K`:"&mdash;"}</td></tr>`;
  const tbl=(title,rows,statHdr,key)=>`<div class="panel"><h3 style="cursor:pointer" onclick="location.hash='${siteHref("mlb","pos",key)}'">${title} <span class="sub" style="font-weight:400">all &rarr;</span></h3><div class="twrap"><table>
    <thead><tr><th></th><th>Player</th><th>GlassBox</th><th class="nocase">&mu;</th><th>${statHdr}</th></tr></thead>
    <tbody>${rows}</tbody></table></div></div>`;
  $("#view").innerHTML=`<div class="eyebrow">Database &middot; MLB</div><h1 class="pt">Players</h1>
    <div class="sub" style="margin-bottom:10px">Every plate appearance is a <b>TrueSkill duel</b>: batter vs pitcher, both Bayesian-updated
      on the outcome. The 0-100 <b>GlassBox rating</b> comes straight from that ladder (50 = league average); &mu; is the raw skill estimate.
      ${bats.length+pits.length} rated players. Click anyone.</div>
    <input class="psearch" id="psearch" placeholder="Search ${P.length} players&hellip;" autocomplete="off">
    <div id="psres"></div>
    <div class="grid" style="margin-top:8px">
      ${tbl("Top hitters",bats.slice(0,15).map(brow).join(""),"2026 season","hitters")}
      ${tbl("Top pitchers",pits.slice(0,15).map(prow).join(""),"2026 season","pitchers")}
    </div>`;
  const res=$("#psres");
  $("#psearch").oninput=e=>{
    const q=norm(e.target.value);
    if(q.length<2){res.innerHTML="";return;}
    const hits=P.filter(p=>norm(p.name).includes(q)).slice(0,20);
    res.innerHTML=hits.length?`<div class="twrap" style="margin-bottom:6px"><table><tbody>
      ${hits.map(p=>`<tr onclick="location.hash='${siteHref("mlb","player",p.id)}'">
        <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
        <td>${gb(p.ts100)}</td>
        <td class="sub">${p.role==="pitcher"?(p.pit?`${p.pit.era!=null?p.pit.era:"-"} ERA`:""):(p.bat?`${p.bat.ops!=null?p.bat.ops:"-"} OPS`:"")}</td></tr>`).join("")}
      </tbody></table></div>`:`<div class="sub" style="margin:6px 0 10px">No players match.</div>`;
  };
}

/* ---------- POSITION DEEP-DIVE ---------- */
function histBars(vals){
  const B=24, c=new Array(B).fill(0);
  vals.forEach(v=>c[Math.max(0,Math.min(B-1,Math.floor(v/100*B)))]++);
  const m=Math.max(...c,1);
  return `<div style="display:flex;align-items:flex-end;gap:2px;height:56px;margin:6px 0 2px">
    ${c.map((x,i)=>`<div style="flex:1;background:${i>=B/2?"var(--accent)":"var(--line-2)"};opacity:${x?1:.22};height:${Math.max(4,Math.round(100*x/m))}%;border-radius:2px" title="${Math.round(100*i/B)}-${Math.round(100*(i+1)/B)}: ${x}"></div>`).join("")}
  </div><div class="sub" style="display:flex;justify-content:space-between"><span>0</span><span>50 = league avg</span><span>100</span></div>`;
}
function chipRow(items,cur,attr){
  return `<div class="filters" style="margin:0 0 0 0">${items.map(([k,t])=>
    `<button class="filt ${String(k)===String(cur)?"on":""}" data-${attr}="${k}">${t}</button>`).join("")}</div>`;
}
function posPage(key){
  if(state.posKey!==key){state.posKey=key;state.posSort="r";state.posMin=0;}
  if(state.league==="nfl"&&state.nfl) return nflPosPage(key);
  // NHL has no position ladders: its players page carries the position
  // filters. Render it in place. Redirecting by writing location.hash here
  // fought route()'s own canonical-hash rewrite and looped forever
  // (188 hashchanges in 1.5s on #/nhl/pos/X).
  if(state.league==="nhl") return playersPage();
  return mlbPosPage(key);
}

function mlbPosPage(key){
  const db=state.db;
  if(!db||(key!=="hitters"&&key!=="pitchers")) return playersPage();   // render, never re-route mid-route
  const isBat=key==="hitters";
  const all=Object.values(db.players||{}).filter(p=>p.role===(isBat?"batter":"pitcher")&&p.ts100!=null);
  const mMu=all.reduce((s,p)=>s+(p.ts_mu||25),0)/Math.max(all.length,1);
  const T=Object.values(db.teams||{}).map(t=>[t.abbr,isBat?t.ts_off:t.ts_pit,t.code])
    .filter(x=>x[1]!=null).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const line=p=>isBat?(p.bat?`${p.bat.ops!=null?p.bat.ops:"-"} OPS &middot; ${p.bat.hr||0} HR &middot; ${p.bat.rbi||0} RBI`:"&mdash;")
                     :(p.pit?`${p.pit.era!=null?p.pit.era:"-"} ERA &middot; ${p.pit.whip!=null?p.pit.whip:"-"} WHIP &middot; ${p.pit.so||0} K`:"&mdash;");
  const vol=p=>isBat?((p.bat&&p.bat.pa)||0):((p.pit&&p.pit.ip)||0);
  const row=(p,i)=>`<tr onclick="location.hash='${siteHref("mlb","player",p.id)}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos}</span></td>
    <td>${teamLink(p.team,p.team,"mlb")}</td>
    <td>${gb(p.ts100)}</td>
    <td><span class="num">${p.ts_mu!=null?p.ts_mu.toFixed(1):"-"}</span></td>
    <td class="sub">${line(p)}</td></tr>`;
  const renderBody=()=>{
    const q=norm(($("#posq")||{}).value||"");
    let P=all.filter(p=>vol(p)>=state.posMin);
    if(q.length>=2) P=P.filter(p=>norm(p.name).includes(q));
    const S={r:(a,b)=>b.ts100-a.ts100, mu:(a,b)=>(b.ts_mu||0)-(a.ts_mu||0),
             vol:(a,b)=>vol(b)-vol(a)};
    P.sort(S[state.posSort]||S.r);
    $("#posbody").innerHTML=P.map(row).join("")||`<tr><td class="sub" colspan="6">No players match.</td></tr>`;
    $("#posn").textContent=P.length;
  };
  $("#view").innerHTML=`<a class="back" href="${siteHref("mlb","players")}">&lsaquo; Players</a>
    <div class="eyebrow">TrueSkill ladder &middot; MLB</div><h1 class="pt">${isBat?"Hitters":"Pitchers"}</h1>
    <div class="sub" style="margin-bottom:10px"><b>${all.length}</b> rated ${key} &middot; league &mu; ${mMu.toFixed(1)}
      &middot; every plate appearance is a batter-vs-pitcher TrueSkill duel; 50 = league average. Click anyone.</div>
    <div class="grid" style="margin-bottom:14px">
      <div class="panel"><h3>Rating distribution</h3>${histBars(all.map(p=>p.ts100))}</div>
      <div class="panel"><h3>Best team ${isBat?"lineups":"staffs"} <span class="sub" style="font-weight:400">roster TrueSkill average</span></h3>
        <div class="twrap"><table><tbody>${T.map(([ab,v,code],i)=>`<tr onclick="location.hash='${siteHref("mlb","team",code)}'">
          <td><span class="num">${i+1}</span></td><td class="a"><span class="ab" style="color:var(--accent)">${ab}</span></td>
          <td>${gb(v)}</td></tr>`).join("")}</tbody></table></div></div>
    </div>
    <div class="controls" style="margin-bottom:10px">
      ${chipRow([["r","Rating"],["mu","Raw μ"],["vol",isBat?"Most PA":"Most IP"]],state.posSort,"psort")}
      ${chipRow(isBat?[[0,"All"],[100,"100+ PA"],[300,"300+ PA"]]:[[0,"All"],[30,"30+ IP"],[100,"100+ IP"]],state.posMin,"pmin")}
      <input class="psearch" id="posq" placeholder="Filter by name&hellip;" autocomplete="off" style="max-width:220px;margin:0">
    </div>
    <div class="sub" style="margin-bottom:6px"><span id="posn">${all.length}</span> shown</div>
    <div class="twrap"><table>
      <thead><tr><th></th><th>Player</th><th>Team</th><th>GlassBox</th><th class="nocase">&mu;</th><th>2026 season</th></tr></thead>
      <tbody id="posbody"></tbody></table></div>`;
  renderBody();
  $("#view").querySelectorAll("[data-psort]").forEach(x=>x.onclick=()=>{state.posSort=x.dataset.psort;mlbPosPage(key);});
  $("#view").querySelectorAll("[data-pmin]").forEach(x=>x.onclick=()=>{state.posMin=parseInt(x.dataset.pmin);mlbPosPage(key);});
  $("#posq").oninput=renderBody;
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
  const g=findGame(pk); if(!g){$("#view").innerHTML=`<div class="empty">Game not found. <a href="${siteHref("mlb","")}">Back to board</a></div>`;return;}
  const db=state.db, hp=g.home_win_prob, homeWin=hp>=0.5, e=g.edge||{};
  const A=db.teams[g.away], H=db.teams[g.home];
  const pa=findPlayerByName(g.away_sp), ph=findPlayerByName(g.home_sp);
  const conf=Math.max(hp,1-hp), gTier=infoTier(g);
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
    <a class="back" href="${siteHref("mlb","")}">&lsaquo; Board</a>
    <div class="gh"><span class="mt">${teamLink(g.away,g.away_abbr,"mlb")} <span style="color:var(--faint)">@</span> ${teamLink(g.home,g.home_abbr,"mlb")}</span>
      <span class="meta">${fmtDay(g.date,state.board.generated)} &middot; ${fmtTime(g.start_utc)}</span></div>
    <div class="livepanel" id="livepanel" data-pk="${g.game_pk}" data-home="${g.home_abbr}" data-away="${g.away_abbr}" data-pick="${g.pick}"></div>
    <div class="cols">
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="panel"><h3>Our projection</h3>
          <div class="proj"><div class="big">${pctI(hp)}<span class="u">%</span></div>
            <div class="pk">${g.home_abbr} win probability<b>${gTier==="EARLY"?"Lean":callOf(conf)==="LEAN"?"Lean":"Pick"}: ${g.pick} <span class="cf">${pctI(conf)}%</span></b>
            <span class="sub">${gTier==="EARLY"
              ?"scheduled, not a pick — no starter announced, team ratings only"
              :callOf(conf)==="LEAN"?"inside 55/45 — the model has a side, not a case"
              :(gTier==="CONFIRMED"?"official lineup posted":"starters named, lineup projected")}</span></div></div>
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
              <div class="sub">${teamLink(g.away,g.away_abbr,"mlb")}${pa&&pa.ts100!=null?" &middot; GlassBox "+pa.ts100:""}</div>
              ${statline(pa)}</div>
            <div class="vs">vs</div>
            <div class="col h"><div class="nm">${ph?playerLinkByName(g.home_sp):g.home_sp}</div>
              <div class="sub">${teamLink(g.home,g.home_abbr,"mlb")}${ph&&ph.ts100!=null?" &middot; GlassBox "+ph.ts100:""}</div>
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
  if(state.league==="nhl"&&state.nhl) return nhlStandings();
  const db=state.db, T=Object.values(db.teams);
  const divs={}; T.forEach(t=>{(divs[t.div]=divs[t.div]||[]).push(t);});
  const order=["AL East","AL Central","AL West","NL East","NL Central","NL West"];
  let html=`<div class="eyebrow">Database</div><h1 class="pt">Standings</h1>
    <div class="sub" style="margin-bottom:14px"><b>GlassBox</b> is the roster's 0-100 TrueSkill rating (50 = league average). <b>Luck</b> = win% minus pythagorean (large + = regression risk). <b>Elo</b> is our market-blind rating. Click any team.</div>`;
  html+=order.filter(d=>divs[d]).map(d=>{
    const rows=divs[d].sort((a,b)=>a.div_rank-b.div_rank).map(t=>`
      <tr onclick="location.hash='${siteHref("mlb","team",t.code)}'">
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
  if(state.league==="nhl"&&state.nhl) return nhlTeams();
  const db=state.db;
  const T=Object.values(db.teams).sort((a,b)=>(b.ts100||0)-(a.ts100||0));
  const card=t=>`<div class="tcard" onclick="location.hash='${siteHref("mlb","team",t.code)}'">
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
  if(state.league==="nhl"&&state.nhl) return nhlTeamPage(code);
  const db=state.db, t=db.teams[code];
  if(!t){$("#view").innerHTML=`<div class="empty">Team not found. <a href="${siteHref("mlb","teams")}">All teams</a></div>`;return;}
  const roster=(t.roster||[]).map(id=>db.players[String(id)]).filter(Boolean);
  const bat=roster.filter(p=>p.role==="batter");
  const pit=roster.filter(p=>p.role==="pitcher");
  const cell=(v,cls)=>`<td><span class="num ${cls||""}">${v==null?"-":v}</span></td>`;
  const prow=p=>`<tr onclick="location.hash='${siteHref("mlb","player",p.id)}'">
    <td class="a"><span class="num" style="color:var(--faint)">${p.pos}</span>
      <span style="margin-left:8px;font-weight:600">${p.name}</span></td>
    ${p.role==="batter"
      ? cell(p.bat&&p.bat.avg,"")+cell(p.bat&&p.bat.hr)+cell(p.bat&&p.bat.rbi)+cell(p.bat&&p.bat.ops)
      : cell(p.pit&&p.pit.era)+cell(p.pit&&p.pit.so)+cell(p.pit&&p.pit.whip)+cell(p.pit&&(p.pit.gs>0?p.pit.gs+" GS":(p.pit.sv||0)+" SV"))}
    <td>${gb(p.ts100)}</td></tr>`;
  const tbl=(title,rows,cols)=>`<div class="subh">${title}</div><div class="twrap"><table>
    <thead><tr><th>Player</th>${cols.map(c=>`<th>${c}</th>`).join("")}<th>GlassBox</th></tr></thead>
    <tbody>${rows.map(prow).join("")}</tbody></table></div>`;
  $("#view").innerHTML=`<a class="back" href="${siteHref("mlb","teams")}">&lsaquo; Teams</a>
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
  if(state.league==="nhl"&&state.nhl&&state.nhl.players[id]) return nhlPlayerPage(id);
  const db=state.db, p=db.players[String(id)];
  if(!p){$("#view").innerHTML=`<div class="empty">Player not found. <a href="${siteHref("mlb","teams")}">Teams</a></div>`;return;}
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
  $("#view").innerHTML=`<a class="back" href="${siteHref("mlb","team",p.team)}">&lsaquo; ${t?t.name:p.team}</a>
    <div class="phead"><span class="nm">${p.name}</span>
      <span class="sub">${teamLink(p.team,p.team,"mlb")} &middot; ${p.pos}${p.num?" &middot; #"+p.num:""}</span>
      <span style="margin-left:auto">${gb(p.ts100)}</span></div>
    <div class="sub" style="margin-bottom:16px">GlassBox rating (0-100) &mdash; this player's per-plate-appearance TrueSkill rating vs league-average ${p.role}s. Market-blind.</div>
    <div class="pgrid">
      <div class="panel"><h3>2026 season</h3><div class="statgrid">${season}</div></div>
      <div class="panel"><h3>Career</h3><div class="statgrid">${career}</div></div>
    </div>`;
}

/* ---------- TRACK RECORD (#/record) ---------- */
/* Every graded PRE-GAME pick, per league, scored honestly. MLB comes from the
   served ledger block (board.record.mlb): the board is forward-only, so a played
   game's probability lives nowhere else. NFL/NHL are read straight off their
   payload schedules, where each played game still carries its frozen pre-game
   probability (ph / hp). No market numbers, no refitting - the model's own picks,
   graded against the result. */
