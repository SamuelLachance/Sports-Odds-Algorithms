const BASE_PATH = document.querySelector('meta[name="base-path"]')?.content || "";
const USE_STATIC_API =
  Boolean(BASE_PATH) || window.location.hostname.endsWith("github.io");

const state = {
  slate: null,
  tracking: null,
  teamsIndex: null,
  teamProfiles: {},
  worldCup: null,
  selectedLeague: "all",
  trackingPeriod: "all_time",
  worldCupTab: "overview",
};

const appRoot = document.getElementById("appRoot");
const leagueMenu = document.getElementById("leagueMenu");
const gameMenu = document.getElementById("gameMenu");
const sidebarGames = document.getElementById("sidebarGames");
const sidebarGamesTitle = document.getElementById("sidebarGamesTitle");
const footerUpdated = document.getElementById("footerUpdated");
const themeToggle = document.getElementById("themeToggle");

function api(path) {
  if (USE_STATIC_API) {
    return `${BASE_PATH}/api/${path}`;
  }
  return `/api/${path}`;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      typeof payload?.detail === "string" ? payload.detail : "Request failed",
    );
  }
  return payload;
}

function parseRoute() {
  const hash = location.hash.replace(/^#/, "") || "/";
  const parts = hash.split("/").filter(Boolean);
  return { path: parts[0] || "", parts };
}

function navigate(hash) {
  location.hash = hash;
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("soa-theme", theme);
}

function initTheme() {
  setTheme(localStorage.getItem("soa-theme") === "light" ? "light" : "dark");
}

function formatTime(iso) {
  if (!iso) return "TBD";
  return new Date(iso).toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatOdds(v) {
  if (v == null || v === 0) return "—";
  return v > 0 ? `+${v}` : `${v}`;
}

function formatSpread(v) {
  if (v == null) return "—";
  const rounded = Number(v);
  const text = Number.isInteger(rounded) ? `${rounded}` : rounded.toFixed(1);
  return rounded > 0 ? `+${text}` : text;
}

function pickMarketLabel(pick) {
  if (pick.bet_type === "spread") {
    return `Spread ${formatSpread(pick.spread_line)} (${formatOdds(pick.spread_odds ?? pick.market_odds)})`;
  }
  if (pick.side === "draw") {
    return `Draw ${formatOdds(pick.market_odds)}`;
  }
  return formatOdds(pick.market_odds);
}

function pickModelLabel(pick) {
  if (pick.bet_type === "spread" && pick.model_margin != null) {
    const sideMargin = pick.side === "home" ? pick.model_margin : -pick.model_margin;
    return `Margin ${formatSpread(sideMargin)}`;
  }
  return formatOdds(pick.model_projection);
}

function confClass(c) {
  return `confidence-${c || "low"}`;
}

function gamesForLeague(league) {
  const games = state.slate?.games || [];
  return league === "all" ? games : games.filter((g) => g.league === league);
}

function gameById(id) {
  return (state.slate?.games || []).find((g) => g.event_id === id);
}

function renderLeagueMenu() {
  const leagues = state.teamsIndex?.leagues || [];
  const items = [
    `<li><a href="#/games" class="league-link ${state.selectedLeague === "all" ? "active" : ""}" data-league="all">All sports</a></li>`,
    ...leagues.map(
      (lg) =>
        `<li><a href="#/games/${lg.id}" class="league-link ${state.selectedLeague === lg.id ? "active" : ""}" data-league="${lg.id}">${lg.name} <span class="count">${lg.team_count}</span></a></li>`,
    ),
  ];
  leagueMenu.innerHTML = items.join("");
  leagueMenu.querySelectorAll(".league-link").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      const lg = el.dataset.league;
      navigate(lg === "all" ? "#/games" : `#/games/${lg}`);
    });
  });
}

function renderGameSubmenu(league) {
  const games = gamesForLeague(league);
  if (!games.length) {
    sidebarGames.hidden = true;
    return;
  }
  sidebarGames.hidden = false;
  sidebarGamesTitle.textContent =
    league === "all" ? "Today's games" : `${league.toUpperCase()} games`;
  gameMenu.innerHTML = games
    .map((g) => {
      const away = g.matchup.away.name;
      const home = g.matchup.home.name;
      return `<li><a href="#/game/${g.event_id}" class="game-link">${away} @ ${home}</a></li>`;
    })
    .join("");
}

function factorBars(factors) {
  if (!factors?.length) return "<p class='muted'>No factor data.</p>";
  return factors
    .map((f) => {
      const w = Math.min(Math.abs(f.value), 100);
      const dir = f.favors === "away" ? "away" : f.favors === "home" ? "home" : "neutral";
      return `<div class="factor-item compact"><div class="factor-head"><strong>${f.label}</strong><span>${f.value > 0 ? "+" : ""}${f.value.toFixed(1)} · ${dir}</span></div><div class="factor-bar"><span style="width:${w}%"></span></div></div>`;
    })
    .join("");
}

function pickCard(pick, extra = "") {
  return `<article class="pick-card ${confClass(pick.confidence)}">
    <div class="pick-top"><span class="league-pill">${pick.league_name || pick.league}</span><span class="strategy-pill">${pick.strategy_label || pick.strategy}</span></div>
    <h3>${pick.team_name}</h3>
    <p class="pick-matchup">${pick.matchup || extra}</p>
    <p class="pick-time">${formatTime(pick.start_time)}</p>
    <div class="pick-odds">
      <div><span>${pick.bet_type === "spread" ? "Spread" : "Market"}</span><strong>${pickMarketLabel(pick)}</strong></div>
      <div><span>Model</span><strong>${pickModelLabel(pick)}</strong></div>
      <div><span>Edge</span><strong>+${pick.edge}</strong></div>
    </div>
    <p class="pick-reason">${pick.reason}</p>
  </article>`;
}

function algoBreakdown(m) {
  if (!m) return "";
  const legacy = m.legacy;
  const power = m.power;
  const basketball = m.basketball_pred;
  const baseball = m.baseball_pred;
  const soccer = m.soccer_pred;
  const legacyThreeway = m.legacy_threeway;
  const powerThreeway = m.power_threeway;
  if (!legacy && !power && !basketball && !baseball && !soccer && !legacyThreeway) return "";
  const parts = [];
  const layerTag =
    m.blend_layers === 3 ? "3-layer" : m.blend_layers === 2 ? "2-layer" : "";
  if (layerTag) {
    parts.push(layerTag);
  }
  if (legacy) {
    parts.push(`Legacy V2: ${legacy.win_probability}% (${legacy.favorite_side})`);
  }
  if (power) {
    parts.push(`Power: ${power.home_win_probability}% home (${power.home_power} vs ${power.away_power})`);
  }
  if (basketball) {
    const margin =
      basketball.predicted_margin != null
        ? ` margin ${basketball.predicted_margin}`
        : "";
    parts.push(`Matrix: ${basketball.home_win_probability}% home${margin}`);
  }
  if (baseball) {
    const elo =
      baseball.elo_exp != null ? ` Elo ${baseball.elo_exp}%` : "";
    parts.push(`MLB-Model: ${baseball.home_win_probability}% home${elo}`);
  }
  if (legacyThreeway) {
    parts.push(
      `Legacy 1X2: ${legacyThreeway.home_win_probability}/${legacyThreeway.draw_probability}/${legacyThreeway.away_win_probability}%`
    );
  }
  if (powerThreeway) {
    parts.push(
      `Power 1X2: ${powerThreeway.home_win_probability}/${powerThreeway.draw_probability}/${powerThreeway.away_win_probability}%`
    );
  }
  if (soccer) {
    const xg =
      soccer.expected_home_goals != null
        ? ` xG ${soccer.expected_home_goals}-${soccer.expected_away_goals}`
        : "";
    parts.push(
      `Football-predictor: ${soccer.home_win_probability}/${soccer.draw_probability}/${soccer.away_win_probability}%${xg}`
    );
  }
  if (m.blend_note) {
    parts.push(m.blend_note);
  }
  return parts.length
    ? `<div class="algo-blend panel-sub"><span class="blend-label">Model blend</span><small>${parts.join(" · ")}</small></div>`
    : "";
}

function algoCenter(game) {
  const m = game.model;
  const mk = game.market;
  const away = game.matchup.away;
  const home = game.matchup.home;
  const fav = m.favorite_side === "home" ? home.name : away.name;
  const top = game.top_pick;
  const isSoccer = Boolean(m.threeway);
  const algoLabel = m.algorithm === "Unified" ? "Unified model" : "Algo V2 win probability";
  const probBlock = isSoccer
    ? `<div class="algo-probability">
        <span>3-way unified probabilities</span>
        <div class="odds-row game-odds threeway-probs">
          <div class="odds-chip"><span>${away.name}</span><strong>${m.away_win_probability ?? "—"}%</strong><small>Model ${formatOdds(m.away_projection)}</small></div>
          <div class="odds-chip"><span>Draw</span><strong>${m.draw_probability ?? "—"}%</strong><small>Model ${formatOdds(m.draw_projection)}</small></div>
          <div class="odds-chip"><span>${home.name}</span><strong>${m.home_win_probability ?? "—"}%</strong><small>Model ${formatOdds(m.home_projection)}</small></div>
        </div>
      </div>`
    : `<div class="algo-probability">
        <span>${algoLabel}</span>
        <strong class="prob-value">${m.win_probability}%</strong>
        <small>Model favorite: ${fav}</small>
      </div>`;
  const oddsRow = isSoccer
    ? `<div class="odds-row game-odds">
        <div class="odds-chip"><span>${away.name}</span><strong>${formatOdds(mk.away_moneyline)}</strong><small>Model ${formatOdds(m.away_projection)}</small></div>
        <div class="odds-chip"><span>Draw</span><strong>${formatOdds(mk.draw_moneyline)}</strong><small>Model ${formatOdds(m.draw_projection)}</small></div>
        <div class="odds-chip"><span>${home.name}</span><strong>${formatOdds(mk.home_moneyline)}</strong><small>Model ${formatOdds(m.home_projection)}</small></div>
        <div class="odds-chip"><span>O-U</span><strong>${mk.over_under ?? "—"}</strong><small>${mk.provider || "ESPN"}</small></div>
      </div>`
    : `<div class="odds-row game-odds">
        <div class="odds-chip"><span>${away.name}</span><strong>${formatOdds(mk.away_moneyline)}</strong><small>Model ${formatOdds(m.away_projection)}</small></div>
        <div class="odds-chip"><span>${home.name}</span><strong>${formatOdds(mk.home_moneyline)}</strong><small>Model ${formatOdds(m.home_projection)}</small></div>
        <div class="odds-chip"><span>Spread / O-U</span><strong>${mk.spread ?? "—"} / ${mk.over_under ?? "—"}</strong><small>${mk.provider || "ESPN"}</small></div>
      </div>`;
  return `<section class="algo-hero panel">
    <div class="algo-hero-head">
      <span class="league-pill">${game.league_name}</span>
      <h1>${away.name} <span class="at">@</span> ${home.name}</h1>
      <p class="game-meta">${formatTime(game.start_time)} · ${game.status_detail || game.status}</p>
    </div>
    <div class="algo-core">
      ${probBlock}
      ${algoBreakdown(m)}
      ${oddsRow}
    </div>
    ${top ? `<div class="game-pick ${confClass(top.confidence)}"><strong>${top.strategy_label}</strong><span>${top.team_name} · ${pickMarketLabel(top)} vs model ${pickModelLabel(top)} (+${top.edge})</span><p>${top.reason}</p></div>` : `<div class="game-pick neutral"><strong>No value flag</strong><span>Model leans ${fav}; lines do not beat model price today.</span></div>`}
    <details class="factor-details" open><summary>Algo factor breakdown</summary><div class="factor-list">${factorBars(m.factors)}</div></details>
    ${(game.recommendations || []).length ? `<div class="rec-list"><h3>All model recommendations</h3>${game.recommendations.map((p) => pickCard({ ...p, league_name: game.league_name, matchup: `${away.name} @ ${home.name}`, start_time: game.start_time })).join("")}</div>` : ""}
  </section>`;
}

function viewDashboard() {
  const slate = state.slate || {};
  const summary = slate.summary || {};
  const picks = slate.recommended_bets || [];
  const games = slate.games || [];
  const leagues = summary.leagues || [...new Set(games.map((g) => g.league))];
  const tracking = state.tracking?.all_time || state.tracking?.summary || {};
  const dateLabel = slate.date_label || "Today";
  const minEdge = summary.min_edge ?? slate.min_recommended_edge ?? 40;
  const leagueCounts = games.reduce((acc, g) => {
    acc[g.league_name || g.league] = (acc[g.league_name || g.league] || 0) + 1;
    return acc;
  }, {});
  const slateBreakdown = Object.entries(leagueCounts)
    .map(([name, count]) => `${name} (${count})`)
    .join(" · ");

  appRoot.innerHTML = `
    <section class="tracking-hero panel home-hero">
      <div class="tracking-hero-top">
        <div>
          <h1>Sharp Odds dashboard</h1>
          <p>Today's slate · ${dateLabel} · Unified model across ${leagues.length || 0} leagues.</p>
          <p class="muted">${slateBreakdown || "No games on today's slate yet."}</p>
        </div>
        <div class="tracking-hero-stats home-stats">
          <div><span>Games</span><strong>${summary.games_analyzed ?? games.length}</strong></div>
          <div><span>Algo picks</span><strong>${summary.recommended_bets ?? picks.length}</strong></div>
          <div><span>Min edge</span><strong>+${minEdge}</strong></div>
          <div><span>All-time ROI</span><strong>${tracking.roi_percent ?? 0}%</strong></div>
        </div>
      </div>
    </section>

    <div class="rollup-grid home-quick-links">
      <a class="rollup-card panel home-link-card" href="#/games">
        <h4>Games</h4>
        <strong class="rollup-record">${games.length}</strong>
        <span>Full algo breakdowns for every matchup</span>
      </a>
      <a class="rollup-card panel home-link-card" href="#/picks">
        <h4>Algo picks</h4>
        <strong class="rollup-record">${picks.length}</strong>
        <span>Value bets at +${minEdge} edge or higher</span>
      </a>
      <a class="rollup-card panel home-link-card" href="#/teams">
        <h4>Teams</h4>
        <strong class="rollup-record">${leagues.length}</strong>
        <span>Season stats and recent form</span>
      </a>
      <a class="rollup-card panel home-link-card" href="#/worldcup">
        <h4>World Cup</h4>
        <strong class="rollup-record">${state.worldCup?.summary?.total_matches ?? 104}</strong>
        <span>2026 hub · groups, bracket, unified preds</span>
      </a>
      <a class="rollup-card panel home-link-card" href="#/tracking">
        <h4>Tracking</h4>
        <strong class="rollup-record">${tracking.record || "0-0"}</strong>
        <span>${tracking.units > 0 ? "+" : ""}${tracking.units ?? 0}u · ${tracking.pending ?? 0} pending</span>
      </a>
    </div>

    <section class="section">
      <div class="section-head"><h2>Top algo picks</h2><a class="text-link" href="#/picks">View all →</a></div>
      <div class="picks-grid">${picks.length ? picks.slice(0, 6).map((p) => pickCard(p)).join("") : `<div class="panel empty-panel">No bets meet the +${minEdge} minimum edge threshold today.</div>`}</div>
    </section>`;
}

function renderTrackingSummary() {
  const periods = [
    ["daily", "Today"],
    ["weekly", "This week"],
    ["monthly", "This month"],
    ["yearly", "This year"],
    ["all_time", "All time"],
  ];
  return `<div class="rollup-grid">${periods
    .map(([key, label]) => {
      const row =
        key === "all_time"
          ? state.tracking?.all_time
          : (state.tracking?.[key] || [])[0];
      if (!row) return `<div class="rollup-card panel"><h4>${label}</h4><p class="muted">No data yet</p></div>`;
      return `<div class="rollup-card panel"><h4>${label}</h4><strong class="rollup-record">${row.record || "0-0"}</strong><span>${row.units > 0 ? "+" : ""}${row.units ?? 0}u · ROI ${row.roi_percent ?? 0}%</span><small>${row.bets ?? 0} bets · ${row.pending ?? 0} pending</small></div>`;
    })
    .join("")}</div>`;
}

function viewPicks() {
  const picks = state.slate?.recommended_bets || [];
  const slate = state.slate || {};
  const minEdge = slate.summary?.min_edge ?? slate.min_recommended_edge ?? 40;
  appRoot.innerHTML = `<section class="page-head"><h1>Algo picks</h1><p>Only bets with +${minEdge} edge or higher vs the unified fair prices (3-layer value agreement required where applicable).</p></section>
    <div class="picks-grid">${picks.length ? picks.map((p) => pickCard(p)).join("") : `<div class="panel empty-panel">No bets meet the +${minEdge} minimum edge threshold today.</div>`}</div>`;
}

function viewGames(league) {
  state.selectedLeague = league || "all";
  renderLeagueMenu();
  renderGameSubmenu(state.selectedLeague);
  const games = gamesForLeague(state.selectedLeague);
  appRoot.innerHTML = `<section class="page-head"><h1>Games</h1><p>Select a matchup for full algo analysis. Use the sidebar for league and game navigation.</p></section>
    <div class="slate-list">${games.length ? games.map((g) => gameListCard(g)).join("") : '<div class="panel empty-panel">No games for this filter.</div>'}</div>`;
}

function gameListCard(game) {
  const away = game.matchup.away;
  const home = game.matchup.home;
  const m = game.model;
  const fav = m.favorite_side === "home" ? home.name : away.name;
  return `<article class="game-card panel clickable" data-game="${game.event_id}">
    <div class="game-head"><div><span class="league-pill">${game.league_name}</span><h3>${away.name} @ ${home.name}</h3><p class="game-meta">${formatTime(game.start_time)}</p></div>
    <div class="win-chip"><span>Unified</span><strong>${fav}</strong><small>${m.win_probability}%</small></div></div>
    <a class="btn btn-secondary btn-sm" href="#/game/${game.event_id}">Open algo breakdown →</a>
  </article>`;
}

function viewGame(eventId) {
  const game = gameById(eventId);
  if (!game) {
    appRoot.innerHTML = '<div class="panel empty-panel">Game not found on today\'s slate.</div>';
    return;
  }
  state.selectedLeague = game.league;
  renderLeagueMenu();
  renderGameSubmenu(game.league);
  appRoot.innerHTML = algoCenter(game);
}

function viewTeams(league) {
  const leagues = state.teamsIndex?.leagues || [];
  const filtered = league ? leagues.filter((l) => l.id === league) : leagues;
  appRoot.innerHTML = `<section class="page-head"><h1>Teams</h1><p>Every team across ${leagues.length} leagues — select for season stats and recent form.</p></section>
    ${filtered
      .map(
        (lg) => `<div class="team-league-block"><h2>${lg.name}</h2><div class="team-grid">${lg.teams
          .map(
            (t) =>
              `<a class="team-tile panel" href="#/team/${lg.id}/${t.abbr}"><strong>${t.label}</strong><span>${t.abbr.toUpperCase()}</span></a>`,
          )
          .join("")}</div></div>`,
      )
      .join("")}`;
}

async function loadTeamProfile(league, abbr) {
  const key = `${league}/${abbr}`;
  if (state.teamProfiles[key]) return state.teamProfiles[key];
  const url = USE_STATIC_API
    ? api(`team-profiles/${league}/${abbr}.json`)
    : api(`teams/${league}/${abbr}`);
  try {
    const profile = await fetchJson(url);
    state.teamProfiles[key] = profile;
    return profile;
  } catch {
    return null;
  }
}

async function viewTeam(league, abbr) {
  appRoot.innerHTML = '<div class="panel empty-panel">Loading team profile…</div>';
  const profile = await loadTeamProfile(league, abbr);
  if (!profile) {
    appRoot.innerHTML = '<div class="panel empty-panel">Team profile unavailable.</div>';
    return;
  }
  const stats = profile.season_stats;
  appRoot.innerHTML = `<section class="page-head"><span class="league-pill">${profile.league_name}</span><h1>${profile.label}</h1><p>Season ${profile.season_year} · Data through ${profile.cutoff_date}</p></section>
    <div class="stat-grid dashboard-stats">
      <div class="stat-box"><span class="stat-label">Record</span><strong>${stats ? `${stats.wins}-${stats.losses}` : "—"}</strong></div>
      <div class="stat-box"><span class="stat-label">Win %</span><strong>${stats?.win_pct ?? "—"}%</strong></div>
      <div class="stat-box"><span class="stat-label">Games</span><strong>${stats?.games_played ?? 0}</strong></div>
      <div class="stat-box"><span class="stat-label">Seasons used</span><strong>${(profile.seasons_used || []).join(" + ") || profile.season_year}</strong></div>
    </div>
    <section class="section"><h2>Recent games</h2>
    <div class="recent-games">${(profile.recent_games || []).length ? profile.recent_games.map((g) => `<div class="recent-row panel"><span class="result-badge ${g.result === "W" ? "win" : "loss"}">${g.result}</span><span>${g.date}</span><span>vs ${g.opponent}</span><strong>${g.score[0]}–${g.score[1]}</strong></div>`).join("") : '<p class="muted">No recent game log.</p>'}</div></section>
    <a class="btn btn-secondary" href="#/teams/${league}">← ${profile.league_name} teams</a>`;
}

function statusBadge(status, units) {
  if (status === "pending") return '<span class="status pending">Pending</span>';
  if (status === "push") return '<span class="status push">Push</span>';
  if (status === "win") return `<span class="status win">Win +${units?.toFixed?.(2) ?? units}u</span>`;
  return `<span class="status loss">Loss ${units?.toFixed?.(2) ?? units}u</span>`;
}

function periodLabel(key) {
  const labels = {
    daily: "Day",
    weekly: "Week",
    monthly: "Month",
    yearly: "Year",
    all_time: "All time",
  };
  return labels[key] || key;
}

function renderPeriodTable(periodKey) {
  const rows =
    periodKey === "all_time"
      ? state.tracking?.all_time
        ? [{ ...state.tracking.all_time, label: "All time", key: "all" }]
        : []
      : state.tracking?.[periodKey] || [];
  if (!rows.length) {
    return `<p class="muted">No ${periodLabel(periodKey).toLowerCase()} data yet — bets are logged each day at 3am.</p>`;
  }
  return `<table class="data-table"><thead><tr><th>${periodLabel(periodKey)}</th><th>Record</th><th>Units</th><th>ROI</th><th>Bets</th><th>Pending</th></tr></thead><tbody>${rows
    .map(
      (r) =>
        `<tr><td>${r.label || r.key}</td><td>${r.record || "0-0"}</td><td>${r.units > 0 ? "+" : ""}${r.units ?? 0}u</td><td>${r.roi_percent ?? 0}%</td><td>${r.bets ?? 0}</td><td>${r.pending ?? 0}</td></tr>`,
    )
    .join("")}</tbody></table>`;
}

function renderUnitsChart(periodKey) {
  const rows = state.tracking?.[periodKey] || [];
  if (!rows.length) return "";
  const max = Math.max(...rows.map((r) => Math.abs(r.units || 0)), 1);
  const bars = [...rows].reverse().slice(-12);
  return `<div class="units-chart">${bars
    .map((r) => {
      const h = Math.max(8, (Math.abs(r.units || 0) / max) * 100);
      const cls = (r.units || 0) >= 0 ? "up" : "down";
      return `<div class="units-bar-wrap"><div class="units-bar ${cls}" style="height:${h}%"></div><span>${(r.label || r.key).split(" ")[0]}</span></div>`;
    })
    .join("")}</div>`;
}

function viewTracking() {
  const period = state.trackingPeriod;
  const all = state.tracking?.all_time || state.tracking?.summary || {};
  const bets = state.tracking?.bets || [];
  const since = state.tracking?.tracking_since || "—";
  const minEdge =
    state.slate?.summary?.min_edge ??
    state.slate?.min_recommended_edge ??
    state.tracking?.min_recommended_edge ??
    40;

  appRoot.innerHTML = `
    <section class="tracking-hero panel">
      <div class="tracking-hero-top">
        <div>
          <h1>Performance tracking</h1>
          <p>Every algo bet with +${minEdge} edge is logged, graded at closing odds, and rolled up day → week → month → year → all time.</p>
          <p class="muted">Tracking since ${since} · ${state.tracking?.timezone || "America/Toronto"}</p>
        </div>
        <div class="tracking-hero-stats">
          <div><span>Record</span><strong>${all.record || "0-0"}</strong></div>
          <div><span>Units</span><strong>${all.units > 0 ? "+" : ""}${all.units ?? 0}u</strong></div>
          <div><span>ROI</span><strong>${all.roi_percent ?? 0}%</strong></div>
          <div><span>Pending</span><strong>${all.pending ?? 0}</strong></div>
        </div>
      </div>
    </section>

    <div class="period-tabs">${["daily", "weekly", "monthly", "yearly", "all_time"]
      .map(
        (p) =>
          `<button type="button" class="period-tab ${period === p ? "active" : ""}" data-period="${p}">${periodLabel(p)}</button>`,
      )
      .join("")}</div>

    <div class="rollup-grid">${renderTrackingSummary()}</div>

    <section class="section panel">
      <h2>${periodLabel(period)} breakdown</h2>
      ${period !== "all_time" ? renderUnitsChart(period) : ""}
      ${renderPeriodTable(period)}
    </section>

    <section class="section">
      <div class="section-head"><h2>Bet log (${bets.length})</h2></div>
      <div class="bet-log">${bets.length ? bets.map((b) => `<article class="bet-row panel"><div class="bet-row-top"><div><strong>${b.team_name}</strong><span class="league-pill">${b.league_name}</span>${statusBadge(b.status, b.units)}</div><span class="edge-tag">+${b.edge} edge</span></div>
      <p class="muted">${b.matchup} · ${b.date}</p>
      <div class="pick-odds compact"><div><span>${b.bet_type === "spread" ? "Spread" : "Market"}</span><strong>${b.bet_type === "spread" ? formatSpread(b.spread_line) + " (" + formatOdds(b.spread_odds ?? b.market_odds) + ")" : formatOdds(b.market_odds)}</strong></div><div><span>Model</span><strong>${b.bet_type === "spread" && b.model_margin != null ? "Margin " + formatSpread(b.side === "home" ? b.model_margin : -b.model_margin) : formatOdds(b.model_projection)}</strong></div><div><span>Strategy</span><strong>${b.strategy_label}</strong></div></div>
      ${b.final_score ? `<p class="final-score">Final: ${b.final_score}</p>` : ""}</article>`).join("") : `<div class="panel empty-panel">No tracked bets yet. Picks with +${minEdge} edge are logged on each daily rebuild.</div>`}</div>
    </section>`;

  appRoot.querySelectorAll(".period-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.trackingPeriod = btn.dataset.period;
      viewTracking();
    });
  });
}

function wcPred(eventId) {
  return state.worldCup?.predictions?.[eventId];
}

function wcMatchById(eventId) {
  return (state.worldCup?.matches || []).find((m) => m.event_id === eventId);
}

function wcThreewayPred(model) {
  if (!model?.threeway) return "";
  return `${model.home_win_probability}% / ${model.draw_probability}% / ${model.away_win_probability}%`;
}

function wcMatchCard(match) {
  const pred = wcPred(match.event_id);
  const model = pred?.model;
  const score = match.completed ? match.scoreline : match.status_detail || "Scheduled";
  const pick = pred?.top_pick;
  let predBlock = "";
  if (model?.threeway) {
    predBlock = `<div class="wc-pred-chip"><span>Unified 1X2</span><strong>${wcThreewayPred(model)}</strong><small>H / D / A</small></div>`;
  } else if (model) {
    const fav = model.favorite_side === "home" ? match.home.name : match.away.name;
    predBlock = `<div class="wc-pred-chip"><span>Unified</span><strong>${fav}</strong><small>${model.win_probability}%</small></div>`;
  } else if (match.is_placeholder) {
    predBlock = `<div class="wc-pred-chip muted-chip"><span>TBD</span><strong>—</strong><small>Teams not set</small></div>`;
  } else {
    predBlock = `<div class="wc-pred-chip muted-chip"><span>Model</span><strong>—</strong><small>Unavailable</small></div>`;
  }
  const pickBadge = pick
    ? `<span class="wc-pick-badge">Pick: ${pick.team_name} (+${pick.edge})</span>`
    : "";
  return `<article class="wc-match-card panel ${match.completed ? "completed" : ""}">
    <div class="wc-match-head">
      <div>
        <span class="league-pill">${match.round_label}${match.group ? ` · Group ${match.group}` : ""}</span>
        <h4>${match.away.name} <span class="at">@</span> ${match.home.name}</h4>
        <p class="game-meta">${formatTime(match.start_time)}${match.venue ? ` · ${match.venue}` : ""}</p>
      </div>
      <div class="wc-score-block">
        <span class="wc-score">${score}</span>
        ${pickBadge}
      </div>
    </div>
    <div class="wc-match-foot">
      ${predBlock}
      <a class="btn btn-secondary btn-sm" href="#/worldcup/match/${match.event_id}">Full breakdown →</a>
    </div>
  </article>`;
}

function renderGroupTable(group) {
  const rows = group.standings || [];
  return `<div class="wc-group-panel panel">
    <h3>Group ${group.id}</h3>
    <table class="data-table wc-standings-table">
      <thead><tr><th>#</th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th>Form</th></tr></thead>
      <tbody>${rows
        .map(
          (r) =>
            `<tr class="zone-${r.zone}"><td>${r.position}</td><td><strong>${r.team}</strong></td><td>${r.played}</td><td>${r.wins}</td><td>${r.draws}</td><td>${r.losses}</td><td>${r.goals_for}</td><td>${r.goals_against}</td><td>${r.goal_diff > 0 ? "+" : ""}${r.goal_diff}</td><td><strong>${r.points}</strong></td><td class="wc-form">${(r.form || []).join("")}</td></tr>`,
        )
        .join("")}</tbody>
    </table>
    <div class="wc-group-matches">${(group.matches || []).map((m) => wcMatchCard(m)).join("")}</div>
  </div>`;
}

function renderKnockoutBracket(hub) {
  const rounds = (hub.rounds || []).filter((r) => r.slug !== "group-stage");
  return `<div class="wc-bracket">${rounds
    .map(
      (round) => `<section class="wc-bracket-round panel">
        <h3>${round.label}</h3>
        <div class="wc-bracket-matches">${(round.matches || []).map((m) => wcMatchCard(m)).join("")}</div>
      </section>`,
    )
    .join("")}</div>`;
}

function wcSimProbChip(probs, xg) {
  let block = "";
  if (probs) {
    block += `<div class="wc-pred-chip"><span>Unified 1X2</span><strong>${probs.home}% / ${probs.draw}% / ${probs.away}%</strong><small>H / D / A</small></div>`;
  }
  if (xg) {
    block += `<div class="wc-pred-chip"><span>Poisson xG</span><strong>${xg.home} – ${xg.away}</strong><small>Blended 3-layer</small></div>`;
  }
  return block;
}

function wcSimMatchCard(match) {
  const probs = match.model_probs;
  const xg = match.expected_goals;
  const winnerBadge =
    match.winner === "home"
      ? `<span class="wc-sim-winner-badge">${match.home.name}</span>`
      : match.winner === "away"
        ? `<span class="wc-sim-winner-badge">${match.away.name}</span>`
        : `<span class="wc-sim-winner-badge draw">Draw</span>`;
  return `<article class="wc-match-card wc-sim-match panel">
    <div class="wc-match-head">
      <div>
        <span class="league-pill">${match.round_label}${match.group ? ` · Group ${match.group}` : ""}</span>
        <h4>${match.away.name} <span class="at">@</span> ${match.home.name}</h4>
        ${match.resolved_from_placeholder ? '<p class="game-meta muted">Resolved from bracket placeholder</p>' : ""}
      </div>
      <div class="wc-score-block">
        <span class="wc-score wc-sim-score">${match.scoreline}</span>
        ${winnerBadge}
      </div>
    </div>
    <div class="wc-match-foot">${wcSimProbChip(probs, xg)}</div>
  </article>`;
}

function renderBracketTreeNode(node) {
  if (!node || node.placeholder) {
    return `<div class="wc-tree-node wc-tree-placeholder panel"><span class="muted">TBD</span></div>`;
  }
  const away = node.away?.name || "TBD";
  const home = node.home?.name || "TBD";
  const awayScore = node.away?.score ?? "–";
  const homeScore = node.home?.score ?? "–";
  const awayWin = node.away?.winner ? " winner" : "";
  const homeWin = node.home?.winner ? " winner" : "";
  const children = (node.children || [])
    .map((child) => renderBracketTreeNode(child))
    .join("");
  return `<div class="wc-tree-node panel">
    <div class="wc-tree-match">
      <div class="wc-tree-team${awayWin}"><span>${away}</span><strong>${awayScore}</strong></div>
      <div class="wc-tree-team${homeWin}"><span>${home}</span><strong>${homeScore}</strong></div>
      ${node.scoreline ? `<div class="wc-tree-scoreline">${node.scoreline}</div>` : ""}
    </div>
    ${children ? `<div class="wc-tree-children">${children}</div>` : ""}
  </div>`;
}

function renderKnockoutBracketTree(bracketTree) {
  if (!bracketTree) return "";
  const columns = (bracketTree.columns || []).filter((c) => c.slug !== "3rd-place-match");
  const columnsHtml = columns
    .map(
      (col) => `<div class="wc-tree-column">
        <h4>${col.label}</h4>
        <div class="wc-tree-column-matches">${(col.matches || [])
          .map(
            (m) => `<div class="wc-tree-leaf panel">
              <div class="wc-tree-team${m.away?.winner ? " winner" : ""}"><span>${m.away?.name || "TBD"}</span><strong>${m.away?.score ?? "–"}</strong></div>
              <div class="wc-tree-team${m.home?.winner ? " winner" : ""}"><span>${m.home?.name || "TBD"}</span><strong>${m.home?.score ?? "–"}</strong></div>
              <div class="wc-tree-scoreline">${m.scoreline || ""}</div>
            </div>`,
          )
          .join("")}</div>
      </div>`,
    )
    .join("");

  const finalHtml = bracketTree.final
    ? `<div class="wc-tree-column wc-tree-final-col">
        <h4>Final</h4>
        ${renderBracketTreeNode(bracketTree.final)}
      </div>`
    : "";

  const thirdHtml = bracketTree.third_place_match
    ? `<div class="wc-tree-column wc-tree-third-col">
        <h4>Third place</h4>
        <div class="wc-tree-leaf panel">
          <div class="wc-tree-team${bracketTree.third_place_match.away?.winner ? " winner" : ""}"><span>${bracketTree.third_place_match.away?.name}</span><strong>${bracketTree.third_place_match.away?.score}</strong></div>
          <div class="wc-tree-team${bracketTree.third_place_match.home?.winner ? " winner" : ""}"><span>${bracketTree.third_place_match.home?.name}</span><strong>${bracketTree.third_place_match.home?.score}</strong></div>
          <div class="wc-tree-scoreline">${bracketTree.third_place_match.scoreline || ""}</div>
        </div>
      </div>`
    : "";

  return `<section class="section panel"><h2>Knockout bracket tree</h2>
    <p class="muted">Full knockout path from Round of 32 through the Final — scores sampled via Poisson xG from the unified 3-layer model.</p>
    <div class="wc-bracket-tree">${columnsHtml}${finalHtml}${thirdHtml}</div></section>`;
}

function renderMonteCarloBlock(mc) {
  if (!mc) return "";
  const champRows = (mc.champion_probability || [])
    .map(
      (r) =>
        `<tr><td><strong>${r.team}</strong></td><td>${r.probability}%</td><td>${r.count}/${mc.iterations}</td></tr>`,
    )
    .join("");
  return `<section class="section panel"><h2>Monte Carlo tournament (${mc.iterations} runs)</h2>
    <p class="muted">Each run simulates all 104 matches using unified 3-layer 1X2 probabilities and Dixon–Coles Poisson scores. Representative path below uses the most common champion (${mc.mode_champion || "—"}).</p>
    <table class="data-table"><thead><tr><th>Team</th><th>Win %</th><th>Count</th></tr></thead>
    <tbody>${champRows || '<tr><td colspan="3">No data</td></tr>'}</tbody></table></section>`;
}

function renderSimulationTab(hub) {
  const sim = hub.simulation;
  if (!sim) {
    return '<div class="panel empty-panel">Simulation data not available yet. Rebuild the World Cup hub.</div>';
  }
  const summary = sim.summary || {};
  const rounds = sim.rounds || [];
  const standings = sim.simulated_standings || {};

  const mc = sim.monte_carlo || {};
  const podium = `
    <section class="wc-sim-hero panel">
      <p class="wc-sim-label">Representative champion · unified 3-layer + Poisson xG</p>
      <h2 class="wc-sim-champion">${sim.champion || "TBD"}</h2>
      <p class="wc-sim-final">${sim.final_scoreline ? `Final: ${sim.final_scoreline}` : ""}${mc.mode_champion && mc.mode_champion !== sim.champion ? ` · MC mode: ${mc.mode_champion}` : mc.mode_champion ? ` · ${(mc.champion_probability?.[0]?.probability ?? "—")}% MC win rate` : ""}</p>
      <div class="wc-sim-podium">
        <div><span>Runner-up</span><strong>${sim.runner_up || "—"}</strong></div>
        <div><span>Third place</span><strong>${sim.third_place || "—"}</strong></div>
        <div><span>Matches simulated</span><strong>${summary.total_simulated ?? 104}</strong></div>
      </div>
    </section>`;

  const stats = `
    <div class="rollup-grid">
      <div class="rollup-card panel"><h4>Group stage</h4><strong class="rollup-record">${summary.group_stage ?? 72}</strong><span>Unified 1X2 + Poisson scores</span></div>
      <div class="rollup-card panel"><h4>Knockout</h4><strong class="rollup-record">${summary.knockout ?? 32}</strong><span>Extra time / pens on ties</span></div>
      <div class="rollup-card panel"><h4>Monte Carlo</h4><strong class="rollup-record">${mc.iterations ?? summary.mc_iterations ?? 500}</strong><span>Full tournament iterations</span></div>
    </div>`;

  const standingsHtml = `<section class="section panel"><h2>Simulated group standings</h2>
    <div class="wc-groups-grid">${Object.keys(standings)
      .sort()
      .map((gid) => {
        const rows = standings[gid] || [];
        return `<div class="wc-group-panel"><h3>Group ${gid}</h3>
          <table class="data-table wc-standings-table"><thead><tr><th>#</th><th>Team</th><th>Pts</th><th>GD</th></tr></thead>
          <tbody>${rows
            .map(
              (r) =>
                `<tr class="zone-${r.zone}"><td>${r.position}</td><td><strong>${r.team}</strong></td><td>${r.points}</td><td>${r.goal_diff > 0 ? "+" : ""}${r.goal_diff}</td></tr>`,
            )
            .join("")}</tbody></table></div>`;
      })
      .join("")}</div></section>`;

  const roundsHtml = `<div class="wc-bracket wc-sim-bracket">${rounds
    .map(
      (round) => `<section class="wc-bracket-round panel">
        <h3>${round.label} <span class="muted">(${(round.matches || []).length})</span></h3>
        <div class="wc-bracket-matches">${(round.matches || []).map((m) => wcSimMatchCard(m)).join("")}</div>
      </section>`,
    )
    .join("")}</div>`;

  return `${podium}${stats}${renderMonteCarloBlock(mc)}${renderKnockoutBracketTree(sim.bracket_tree)}${standingsHtml}${roundsHtml}`;
}

function viewWorldCup(subPath) {
  const hub = state.worldCup;
  if (!hub) {
    appRoot.innerHTML = '<div class="panel empty-panel">Loading World Cup 2026 data…</div>';
    return;
  }
  const summary = hub.summary || {};
  const fmt = hub.format || {};
  const tab = subPath || state.worldCupTab || "overview";
  state.worldCupTab = tab;

  const tabs = [
    ["overview", "Overview"],
    ["simulation", "Simulation"],
    ["groups", "Groups"],
    ["knockout", "Knockout"],
    ["matches", "All matches"],
  ];

  let body = "";
  if (tab === "simulation") {
    body = renderSimulationTab(hub);
  } else if (tab === "groups") {
    body = `<div class="wc-groups-grid">${Object.values(hub.groups || {})
      .map((g) => renderGroupTable(g))
      .join("")}</div>
      <section class="section panel"><h2>Best third-placed teams</h2>
      <p class="muted">Top 8 of 12 third-place finishers advance to the Round of 32.</p>
      <table class="data-table"><thead><tr><th>Rank</th><th>Group</th><th>Team</th><th>Pts</th><th>GD</th><th>GF</th><th>Status</th></tr></thead>
      <tbody>${(hub.third_place_ranking || [])
        .map(
          (r) =>
            `<tr class="${r.third_place_qualified ? "zone-qualified" : ""}"><td>${r.third_place_rank}</td><td>${r.group}</td><td><strong>${r.team}</strong></td><td>${r.points}</td><td>${r.goal_diff > 0 ? "+" : ""}${r.goal_diff}</td><td>${r.goals_for}</td><td>${r.third_place_qualified ? "Advances" : "Out"}</td></tr>`,
        )
        .join("")}</tbody></table></section>`;
  } else if (tab === "knockout") {
    body = `<p class="muted panel">Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final. Placeholder slots fill in as groups finish.</p>${renderKnockoutBracket(hub)}`;
  } else if (tab === "matches") {
    body = `<div class="wc-all-matches">${(hub.matches || []).map((m) => wcMatchCard(m)).join("")}</div>`;
  } else {
    const recs = hub.recommended_bets || [];
    body = `
      <section class="wc-format panel">
        <h2>2026 format</h2>
        <ul class="wc-format-list">
          <li><strong>48 teams</strong> in <strong>12 groups of 4</strong> — each team plays 3 group matches</li>
          <li><strong>32 advance:</strong> top 2 per group + 8 best third-placed teams</li>
          <li><strong>104 matches</strong> total · ${fmt.dates?.start || "Jun 11"} → ${fmt.dates?.final || "Jul 19"}</li>
          <li>Hosts: ${(fmt.hosts || []).join(", ")}</li>
        </ul>
      </section>
      <div class="rollup-grid">
        <div class="rollup-card panel"><h4>Matches</h4><strong class="rollup-record">${summary.total_matches ?? 0}</strong><span>${summary.completed ?? 0} completed · ${summary.upcoming ?? 0} upcoming</span></div>
        <div class="rollup-card panel"><h4>Unified predictions</h4><strong class="rollup-record">${summary.predictions_count ?? 0}</strong><span>3-layer soccer model on every fixed fixture</span></div>
        <div class="rollup-card panel"><h4>Algo picks</h4><strong class="rollup-record">${summary.recommended_bets ?? 0}</strong><span>Edge + agreement filters</span></div>
        <div class="rollup-card panel"><h4>Group stage</h4><strong class="rollup-record">${summary.group_stage_matches ?? 72}</strong><span>72 matches · 12 groups</span></div>
      </div>
      <section class="section"><div class="section-head"><h2>Recommended World Cup bets</h2></div>
      <div class="picks-grid">${recs.length ? recs.map((p) => pickCard({ ...p, league_name: "FIFA World Cup" })).join("") : '<div class="panel empty-panel">No World Cup bets where all 3 layers find value right now.</div>'}</div></section>
      <section class="section"><div class="section-head"><h2>Latest results & upcoming</h2></div>
      <div class="wc-all-matches">${(hub.matches || []).slice(0, 12).map((m) => wcMatchCard(m)).join("")}</div></section>`;
  }

  appRoot.innerHTML = `
    <section class="wc-hero panel">
      <div class="wc-hero-inner">
        <span class="league-pill">FIFA World Cup 2026</span>
        <h1>World Cup hub</h1>
        <p>Unified 3-layer predictions for every match · live scores · group tables · full knockout bracket.</p>
        <p class="muted">Canada · Mexico · United States · ${summary.total_matches ?? 104} matches</p>
      </div>
    </section>
    <div class="wc-subtabs">${tabs
      .map(
        ([id, label]) =>
          `<button type="button" class="wc-subtab ${tab === id ? "active" : ""}" data-wc-tab="${id}">${label}</button>`,
      )
      .join("")}</div>
    ${body}`;

  appRoot.querySelectorAll(".wc-subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.worldCupTab = btn.dataset.wcTab;
      navigate("#/worldcup");
      viewWorldCup(state.worldCupTab);
    });
  });
}

function viewWorldCupMatch(eventId) {
  const match = wcMatchById(eventId);
  const pred = wcPred(eventId);
  if (!match) {
    appRoot.innerHTML = '<div class="panel empty-panel">Match not found.</div>';
    return;
  }
  if (pred?.model && pred?.matchup && pred?.market) {
    const game = {
      event_id: match.event_id,
      league: "worldcup",
      league_name: "FIFA World Cup",
      name: match.name,
      start_time: match.start_time,
      status: match.status,
      status_detail: match.status_detail,
      matchup: {
        away: { ...pred.matchup.away, name: match.away.name },
        home: { ...pred.matchup.home, name: match.home.name },
      },
      market: pred.market,
      model: pred.model,
      top_pick: pred.top_pick,
      recommendations: pred.recommendations || [],
    };
    appRoot.innerHTML = `${algoCenter(game)}
      <p class="muted" style="margin-top:1rem">${match.round_label}${match.group ? ` · Group ${match.group}` : ""}${match.venue ? ` · ${match.venue}` : ""}${match.scoreline ? ` · Final: ${match.scoreline}` : ""}</p>
      <a class="btn btn-secondary" href="#/worldcup">← World Cup hub</a>`;
    return;
  }
  appRoot.innerHTML = `<section class="page-head"><h1>${match.away.name} @ ${match.home.name}</h1>
    <p>${match.round_label} · ${formatTime(match.start_time)}</p></section>
    ${wcMatchCard(match)}
    <a class="btn btn-secondary" href="#/worldcup">← World Cup hub</a>`;
}

function highlightNav(route) {
  document.querySelectorAll("#mainNav a").forEach((a) => {
    const r = a.dataset.route;
    const active =
      r === `/${route.path}` ||
      (route.path === "" && r === "/") ||
      (route.path === "worldcup" && r === "/worldcup");
    a.classList.toggle("active", active);
  });
}

async function render() {
  const route = parseRoute();
  highlightNav(route);
  try {
    if (route.path === "picks") viewPicks();
    else if (route.path === "games") viewGames(route.parts[1]);
    else if (route.path === "game") viewGame(route.parts[1]);
    else if (route.path === "teams") viewTeams(route.parts[1]);
    else if (route.path === "team") await viewTeam(route.parts[1], route.parts[2]);
    else if (route.path === "tracking") viewTracking();
    else if (route.path === "worldcup") {
      if (route.parts[0] === "match") viewWorldCupMatch(route.parts[1]);
      else viewWorldCup(route.parts[0]);
    } else viewDashboard();
  } catch (err) {
    appRoot.innerHTML = `<div class="panel empty-panel error-panel">${err.message}</div>`;
  }
}

async function loadPlatform() {
  const slate = await fetchJson(
    USE_STATIC_API ? api("daily-slate.json") : api("daily/slate"),
  );
  state.slate = slate;

  try {
    state.tracking = await fetchJson(
      USE_STATIC_API ? api("tracking.json") : api("tracking"),
    );
  } catch {
    state.tracking = {
      bets: [],
      summary: { record: "0-0", units: 0, roi_percent: 0, pending: 0 },
      all_time: { record: "0-0", units: 0, roi_percent: 0, pending: 0 },
      daily: [],
      weekly: [],
      monthly: [],
      yearly: [],
    };
  }

  try {
    state.teamsIndex = await fetchJson(
      USE_STATIC_API ? api("teams-index.json") : api("teams"),
    );
  } catch {
    state.teamsIndex = { leagues: [] };
  }

  try {
    state.worldCup = await fetchJson(
      USE_STATIC_API ? api("world-cup.json") : api("worldcup"),
    );
  } catch {
    state.worldCup = null;
  }

  const stamp = slate.generated_at ? new Date(slate.generated_at) : new Date();
  footerUpdated.textContent = `Updated ${stamp.toLocaleString()}`;
  renderLeagueMenu();

  await render();
}

window.addEventListener("hashchange", () => render());
themeToggle.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
});

initTheme();
loadPlatform().catch((err) => {
  appRoot.innerHTML = `<div class="panel empty-panel error-panel">${err.message}</div>`;
});
