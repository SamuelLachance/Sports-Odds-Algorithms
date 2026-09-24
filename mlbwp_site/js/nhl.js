/* NHL: rail, board, cards, teams, players, standings, game page. */
function nhlRailBtn(on){return `<button class="lg ${on?"on":""}" data-lg="nhl"><span>NHL</span><span class="n">${nhlYear()}</span></button>`;}
function nhlRail(){const b=state.board;
  return b.leagues.map(l=>{
    if(l.active) return `<button class="lg" data-lg="${l.code}"><span>${l.name}</span><span class="n">${l.n_games}</span></button>`;
    if(l.code==="nfl"&&state.nfl) return `<button class="lg" data-lg="nfl"><span>NFL</span><span class="n">2026</span></button>`;
    return `<button class="lg" disabled><span>${l.name}</span><span class="soon">soon</span></button>`;}).join("")+nhlRailBtn(true);}
function nhlWire(){
  $("#view").querySelectorAll("[data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);board();});
  $("#view").querySelectorAll("[data-r]").forEach(x=>x.onclick=()=>{state.nhlRange=x.dataset.r;nhlPage();});}
function nhlCard(g){
  const hp=g.hp, homeWin=hp>=0.5, done=g.hs!=null;
  const pick=homeWin?g.home:g.away, pp=Math.max(hp,1-hp), val=g.value;
  /* The NHL badge is priced WITHOUT goalie confirmation. Our own measurement
     (data/market_projection.json) puts the closing line's orthogonal information
     - goalie/scratch news - at +0.00183, MORE than our entire 0.00171 gap to the
     market. Disclosing that is not optional; see documents/pick_policy.md. */
  const valbar=(val&&val.available)?`<div class="valbar" title="Priced before starting goalies are confirmed — the one thing our own measurement says the market knows and we do not."><span class="vt">EDGE</span> ${val.team} <b>+${Math.round(val.ev_cur*100)}% EV</b><span class="vodds">@ ${val.cur_dec}</span><span class="vlive">no goalie conf.</span></div>`:"";
  let badge=`<span class="lbadge off">Model</span>`+(g.playoff?`<span class="lbadge proj">Playoff</span>`:"");
  if(done){const winner=g.hs>g.as?g.home:g.away;
    badge=(winner===pick?`<span class="lbadge off">HIT</span>`:`<span class="lbadge tbd">MISS</span>`)+((g.last&&g.last!=="REG")?`<span class="lbadge proj">${g.last}</span>`:"");}
  const ng=done?"":` data-ng="nhl_${g.away}_${g.home}"`;    // live-score hook (nhl_ prefix: codes overlap NFL)
  return `<a class="gc${val&&val.available?' hasval':''}" href="#/game/nhl-${g.id}">
    ${valbar}
    <div class="top"><span class="lv"${ng}>${done?`FINAL ${g.as}-${g.hs}`:g.d.slice(5)}</span>${badge}
      <span class="pill ${tier(pp)==="strong"?"strong":""}">${pick} ${pctI(pp)}%</span></div>
    <div class="side ${homeWin?"":"win"}"><span class="ab">${g.away}</span><span class="who"><span class="sp">${nhlCity(g.away)}</span></span><span class="odds" title="fair American odds (no vig) from the model">${amOdds(1-hp)}</span><span class="pc">${pctI(1-hp)}%</span></div>
    <div class="side ${homeWin?"win":""}"><span class="ab">${g.home}</span><span class="who"><span class="sp">${nhlCity(g.home)}</span></span><span class="odds" title="fair American odds (no vig) from the model">${amOdds(hp)}</span><span class="pc">${pctI(hp)}%</span></div>
    <div class="pbar"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div></a>`;}
function nhlPage(){
  const n=state.nhl, mc=n.model_card;
  const sch=(n.schedule||[]).filter(g=>!g.playoff);
  const today=new Date().toLocaleDateString("en-CA",{timeZone:TZ});
  const NR={today:[today,today],week:[today,addDays(today,6)],month:[today,addDays(today,29)],year:["2000-01-01","2099-01-01"]}[state.nhlRange];
  const games=sch.filter(g=>g.d>=NR[0]&&g.d<=NR[1]);
  const filts=[["today","Today"],["week","Week"],["month","Month"],["year","Year"]]
    .map(([k,t])=>`<button class="filt ${k===state.nhlRange?"on":""}" data-r="${k}">${t}</button>`).join("");
  let cards;
  if(!games.length){cards=`<div class="empty">No NHL games in this window &mdash; the season runs October&nbsp;&ndash;&nbsp;June. Pick <b>Year</b> to see every ${nhlSeasLbl()} game with its model pick and result.</div>`;}
  else{const days=[...new Set(games.map(g=>g.d))].sort();
    if(state.nhlRange==="year") days.reverse();   // offseason: newest games first
    cards=days.map(d=>{const gs=games.filter(g=>g.d===d);
      return `<div class="day"><span class="d">${fmtDay(d,today)}</span><span class="c">${gs.length} games</span></div>
        <div class="grid">${gs.map(nhlCard).join("")}</div>`;}).join("");}
  const std=Object.entries(n.teams).sort((a,b)=>b[1].pts-a[1].pts).slice(0,8).map(([c,t],i)=>
    `<tr onclick="location.hash='#/team/${c}'"><td><span class="num">${i+1}</span></td>
      <td class="a"><span class="ab" style="color:var(--accent)">${c}</span> <span class="sub">${nhlCity(c)}</span></td>
      <td><span class="num">${t.pts}</span></td><td><span class="num">${t.w}-${t.l}-${t.otl}</span></td>
      <td><span class="num">${t.elo.toFixed(0)}</span></td></tr>`).join("");
  const cardMc=`<div class="panel"><h3>Model card</h3><div class="statgrid">
    <div class="b"><span class="k">Test log loss</span><span class="v">${mc.test_ll.toFixed(4)}</span></div>
    <div class="b"><span class="k">vs bare Elo</span><span class="v pos">&minus;${(mc.test_delta_vs_elo).toFixed(4)}</span></div>
    <div class="b"><span class="k">Base Elo</span><span class="v">${mc.baseline_elo_test.toFixed(4)}</span></div>
    <div class="b"><span class="k">Home win rate</span><span class="v">${(mc.home_win_rate*100).toFixed(1)}%</span></div>
    <div class="b"><span class="k">${nhlSeasLbl()} acc</span><span class="v">${mc.cur_season_acc?(mc.cur_season_acc*100).toFixed(1)+"%":"&mdash;"}</span></div>
    <div class="b"><span class="k">Features</span><span class="v" style="font-size:12px">${mc.features.join(" &middot; ")}</span></div>
  </div></div>`;
  $("#view").innerHTML=`<div class="controls"><div class="rail">${nhlRail()}</div><div class="filters">${filts}</div></div>
    <p class="sub" style="margin:2px 2px 14px">Market-blind: tuned Elo + rest + back-to-back + expected-goals team rating. Every game shows the model's pre-game pick and the result (HIT/MISS).</p>
    ${cards}
    <div class="cols2" style="margin-top:20px">
      <div class="panel"><h3>Top of the standings</h3><div class="twrap"><table>
        <thead><tr><th></th><th>Team</th><th>Pts</th><th>Record</th><th>Elo</th></tr></thead>
        <tbody>${std}</tbody></table></div><a class="tl" href="#/standings">Full standings &rarr;</a></div>
      ${cardMc}
    </div>`;
  nhlWire();
}
function nhlTeams(){
  const n=state.nhl;
  const cards=Object.entries(n.teams).sort((a,b)=>a[1].rank-b[1].rank).map(([c,t])=>
    `<div class="tcard" onclick="location.hash='#/team/${c}'">
      <div class="h"><span class="code" style="color:var(--accent)">${c}</span><span class="nm">${nhlCity(c)}</span><span class="num">#${t.rank}</span></div>
      <div class="stat"><span>Elo <b>${t.elo.toFixed(0)}</b></span><span>xG <b>${sgn(t.xg)}</b></span><span>Pts <b>${t.pts}</b></span></div>
      <div class="stat"><span>Rec <b>${t.w}-${t.l}-${t.otl}</b></span><span>GF <b>${t.gf}</b></span><span>GA <b>${t.ga}</b></span></div></div>`).join("");
  $("#view").innerHTML=`<div class="controls"><div class="rail">${nhlRail()}</div></div>
    <h1 class="pt">NHL teams <span class="sub">GLASSBOX ratings &middot; ${nhlSeasLbl()}</span></h1>
    <div class="tgrid">${cards}</div>`;
  nhlWire();
}
function nhlTeamPage(code){
  const n=state.nhl, t=n.teams[code];
  if(!t){$("#view").innerHTML=`<div class="empty">Unknown team.</div>`;return;}
  const games=(n.schedule||[]).filter(g=>g.home===code||g.away===code);
  const rows=games.map(g=>{const home=g.home===code,opp=home?g.away:g.home,hp=home?g.hp:1-g.hp;
    const done=g.hs!=null; let res="";
    if(done){const gf=home?g.hs:g.as,ga=home?g.as:g.hs;const w=gf>ga;res=`<span class="num ${w?"pos":"neg"}">${w?"W":"L"} ${gf}-${ga}${g.last&&g.last!=="REG"?"/"+g.last:""}</span>`;}
    return `<tr onclick="location.hash='#/game/nhl-${g.id}'"><td>${g.d.slice(5)}</td>
      <td class="a">${home?"vs":"@"} <span class="ab">${opp}</span></td>
      <td><span class="num">${pctI(hp)}%</span></td><td>${res}</td></tr>`;}).join("");
  const top=(t.top||[]).map(pid=>{const p=n.players[pid];if(!p)return "";
    return `<tr onclick="location.hash='#/player/${pid}'">
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos}</span></td>
    <td><span class="gb ${gbTier(p.rating)}">${p.rating.toFixed(0)}</span></td>
    <td><span class="num ${p.off>=0?"pos":"neg"}">${sgn(p.off)}</span></td>
    <td><span class="num ${p.def>=0?"pos":"neg"}">${sgn(p.def)}</span></td></tr>`;}).join("");
  $("#view").innerHTML=`<a class="back" href="#/teams">&larr; Teams</a>
    <h1 class="pt">${nhlCity(code)} <span class="sub">${code} &middot; #${t.rank} &middot; ${t.w}-${t.l}-${t.otl}, ${t.pts} pts</span></h1>
    <div class="statgrid">
      <div class="b"><span class="k">Elo</span><span class="v">${t.elo.toFixed(0)}</span></div>
      <div class="b"><span class="k">xG rating</span><span class="v ${t.xg>=0?"pos":"neg"}">${sgn(t.xg)}</span></div>
      <div class="b"><span class="k">Goals for</span><span class="v">${t.gf}</span></div>
      <div class="b"><span class="k">Goals against</span><span class="v">${t.ga}</span></div></div>
    <div class="cols2" style="margin-top:16px">
      <div class="panel"><h3>Top skaters <span class="sub">xG/60 RAPM</span></h3><div class="twrap"><table>
        <thead><tr><th>Player</th><th>Rtg</th><th>Off</th><th>Def</th></tr></thead><tbody>${top||`<tr><td colspan=4 class="sub">no rated skaters</td></tr>`}</tbody></table></div></div>
      <div class="panel"><h3>Schedule &amp; picks</h3><div class="twrap"><table>
        <thead><tr><th>Date</th><th>Opp</th><th>Win%</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table></div></div>
    </div>`;
}
function nhlPlayers(){
  const n=state.nhl, ps=Object.entries(n.players);
  const sortK=state.nhlSort;
  const sorted=ps.slice().sort((a,b)=>b[1][sortK]-a[1][sortK]);
  const sortBtns=[["net","Two-way"],["off","Offense"],["def","Defense"]]
    .map(([k,t])=>`<button class="filt ${k===sortK?"on":""}" data-s="${k}">${t}</button>`).join("");
  const rows=sorted.slice(0,300).map(([id,p],i)=>`<tr onclick="location.hash='#/player/${id}'">
    <td><span class="num">${i+1}</span></td>
    <td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
    <td><span class="gb ${gbTier(p.rating)}">${p.rating.toFixed(0)}</span></td>
    <td><span class="num ${p.net>=0?"pos":"neg"}">${sgn(p.net)}</span></td>
    <td><span class="num ${p.off>=0?"pos":"neg"}">${sgn(p.off)}</span></td>
    <td><span class="num ${p.def>=0?"pos":"neg"}">${sgn(p.def)}</span></td>
    <td><span class="num">${p.toi.toFixed(0)}</span></td></tr>`).join("");
  $("#view").innerHTML=`<div class="controls"><div class="rail">${nhlRail()}</div><div class="filters">${sortBtns}</div></div>
    <h1 class="pt">NHL skater ratings <span class="sub">teammate-adjusted xG/60 (RAPM) &middot; 5v5 &middot; last 4 seasons</span></h1>
    <p class="sub" style="margin:2px 2px 12px">Offense = danger created, Defense = danger suppressed, both per 60 min adjusted for teammates &amp; opponents. Display metric &mdash; not a model input.
    <input class="psearch" id="nhlsrch" placeholder="search player&hellip;"></p>
    <div class="twrap"><table>
      <thead><tr><th></th><th>Player</th><th>Rtg</th><th>Net</th><th>Off</th><th>Def</th><th>TOI</th></tr></thead>
      <tbody id="nhlrows">${rows}</tbody></table></div>`;
  $("#view").querySelectorAll("[data-lg]").forEach(x=>x.onclick=()=>{setLeague(x.dataset.lg);board();});
  $("#view").querySelectorAll("[data-s]").forEach(x=>x.onclick=()=>{state.nhlSort=x.dataset.s;nhlPlayers();});
  const srch=$("#nhlsrch");
  if(srch)srch.oninput=()=>{const q=norm(srch.value);
    const f=q?sorted.filter(([id,p])=>norm(p.name).includes(q)):sorted;
    $("#nhlrows").innerHTML=f.slice(0,300).map(([id,p],i)=>`<tr onclick="location.hash='#/player/${id}'">
      <td><span class="num">${i+1}</span></td><td class="a"><span class="player-link">${p.name}</span> <span class="sub">${p.pos} &middot; ${p.team}</span></td>
      <td><span class="gb ${gbTier(p.rating)}">${p.rating.toFixed(0)}</span></td>
      <td><span class="num ${p.net>=0?"pos":"neg"}">${sgn(p.net)}</span></td>
      <td><span class="num ${p.off>=0?"pos":"neg"}">${sgn(p.off)}</span></td>
      <td><span class="num ${p.def>=0?"pos":"neg"}">${sgn(p.def)}</span></td>
      <td><span class="num">${p.toi.toFixed(0)}</span></td></tr>`).join("");};
}
function nhlPlayerPage(id){
  const n=state.nhl, p=n.players[id];
  if(!p){$("#view").innerHTML=`<div class="empty">Unknown player.</div>`;return;}
  $("#view").innerHTML=`<a class="back" href="#/players">&larr; Skater ratings</a>
    <h1 class="pt">${p.name} <span class="sub">${p.pos} &middot; <a class="tl" href="#/team/${p.team}">${nhlCity(p.team)}</a></span></h1>
    <div class="mup"><span class="gb ${gbTier(p.rating)}" style="font-size:24px;padding:8px 14px">${p.rating.toFixed(0)}</span>
      <span class="statline">teammate-adjusted xG/60 rating (5v5, last 4 seasons)</span></div>
    <div class="statgrid" style="margin-top:14px">
      <div class="b"><span class="k">Net xG/60</span><span class="v ${p.net>=0?"pos":"neg"}">${sgn(p.net)}</span></div>
      <div class="b"><span class="k">Offense (create)</span><span class="v ${p.off>=0?"pos":"neg"}">${sgn(p.off)}</span></div>
      <div class="b"><span class="k">Defense (suppress)</span><span class="v ${p.def>=0?"pos":"neg"}">${sgn(p.def)}</span></div>
      <div class="b"><span class="k">5v5 minutes</span><span class="v">${p.toi.toFixed(0)}</span></div></div>`;
}
function nhlStandings(){
  const n=state.nhl;
  const rows=Object.entries(n.teams).sort((a,b)=>b[1].pts-a[1].pts).map(([c,t],i)=>
    `<tr onclick="location.hash='#/team/${c}'"><td><span class="num">${i+1}</span></td>
      <td class="a"><span class="ab" style="color:var(--accent)">${c}</span> <span class="sub">${nhlCity(c)}</span></td>
      <td><span class="num">${t.w}-${t.l}-${t.otl}</span></td><td><span class="num">${t.pts}</span></td>
      <td><span class="num">${t.gf}-${t.ga}</span></td>
      <td><span class="num">${t.elo.toFixed(0)}</span></td>
      <td><span class="num ${t.xg>=0?"pos":"neg"}">${sgn(t.xg)}</span></td></tr>`).join("");
  $("#view").innerHTML=`<div class="controls"><div class="rail">${nhlRail()}</div></div>
    <h1 class="pt">NHL standings <span class="sub">${nhlSeasLbl()} &middot; GLASSBOX ratings</span></h1>
    <div class="twrap"><table>
      <thead><tr><th></th><th>Team</th><th>Record</th><th>Pts</th><th>GF-GA</th><th>Elo</th><th>xG</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  nhlWire();
}
function nhlGamePage(id){
  const n=state.nhl, g=(n.schedule||[]).find(x=>String(x.id)===String(id));
  if(!g){$("#view").innerHTML=`<div class="empty">Game not found.</div>`;return;}
  const th=n.teams[g.home]||{}, ta=n.teams[g.away]||{}, hp=g.hp, done=g.hs!=null;
  const pick=hp>=0.5?g.home:g.away, pp=Math.max(hp,1-hp);
  const cmpRow=(lbl,av,hv,fmt)=>`<tr><td class="tr">${fmt?fmt(av):av}</td><td class="f">${lbl}</td><td class="r">${fmt?fmt(hv):hv}</td></tr>`;
  const ct=g.ct||{};
  const NLBL={elo:"Team strength (Elo + home ice)", rest:"Rest advantage",
              b2b:"Back-to-back", xg:"Expected-goals rating"};
  // elo always shows: it is the model's core and a 0.0 there is information.
  const nWhy=NHL_CT.filter(k=>ct[k]!=null&&(Math.abs(ct[k])>=0.05||k==="elo"))
    .sort((x,y)=>Math.abs(ct[y])-Math.abs(ct[x])).map(k=>pRow(NLBL[k],ct[k])).join("");
  const val=g.value;
  const valbar=(val&&val.available)?`<div class="valbar" style="margin-bottom:14px"><span class="vt">EDGE</span> ${val.team} <b>+${Math.round(val.ev_cur*100)}% EV</b> <span class="vodds">@ ${val.cur_dec} (open ${val.open_dec})</span></div>`:"";
  const res=done?`<div class="mup"><span class="statline">Final: <b>${g.away} ${g.as} &ndash; ${g.hs} ${g.home}</b>${g.last&&g.last!=="REG"?" ("+g.last+")":""} &middot; model ${((hp>=.5)===(g.hs>g.as)?'<span class="pos">HIT</span>':'<span class="neg">MISS</span>')}</span></div>`:"";
  $("#view").innerHTML=`<a class="back" href="#/">&larr; Board</a>
    <h1 class="pt">${nhlCity(g.away)} @ ${nhlCity(g.home)} <span class="sub">${g.d}${g.playoff?" &middot; Playoff":""}</span></h1>
    ${valbar}
    <div class="mup"><span class="ab">${g.away}</span>
      <span class="pill ${tier(pp)==="strong"?"strong":""}" style="font-size:18px">${pick} ${pctI(pp)}%</span>
      <span class="ab">${g.home}</span></div>
    <div class="pbar" style="margin:10px 0 18px"><div class="h" style="width:${pctI(hp)}%"></div><div class="mid"></div></div>
    ${res}
    <div class="panel why"><h3>Why &mdash; model contributions <span class="sub" style="font-weight:400">home prob points</span></h3>
      ${nWhy||`<div class="sub">no contribution data</div>`}
      <div class="sub" style="margin-top:10px">Each bar is one term's push on the home win probability away from 50/50,
        taken straight from the market-blind model's own coefficients. These are exact:
        50% plus the bars equals the number above.</div></div>
    <div class="panel"><h3>Head to head</h3><div class="twrap cmp"><table>
      <thead><tr><th class="tr">${g.away}</th><th class="f"></th><th class="r">${g.home}</th></tr></thead>
      <tbody>
        ${cmpRow("Elo rating",ta.elo,th.elo,v=>v?v.toFixed(0):"&mdash;")}
        ${cmpRow("xG rating",ta.xg,th.xg,v=>v!=null?sgn(v):"&mdash;")}
        ${cmpRow("Record",ta.w!=null?`${ta.w}-${ta.l}-${ta.otl}`:"&mdash;",th.w!=null?`${th.w}-${th.l}-${th.otl}`:"&mdash;")}
        ${cmpRow("Points",ta.pts,th.pts)}
        ${cmpRow("Fair odds",amOdds(1-hp),amOdds(hp))}
      </tbody></table></div></div>`;
}

