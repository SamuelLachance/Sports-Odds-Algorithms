const BASE_PATH = document.querySelector('meta[name="base-path"]')?.content || "";
const USE_STATIC_API =
  Boolean(BASE_PATH) || window.location.hostname.endsWith("github.io");

const state = {
  slate: null,
  tracking: null,
  teamsIndex: null,
  teamProfiles: {},
  selectedLeague: "all",
  trackingPeriod: "all_time",
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
      <div><span>Market</span><strong>${formatOdds(pick.market_odds)}</strong></div>
      <div><span>Model</span><strong>${formatOdds(pick.model_projection)}</strong></div>
      <div><span>Edge</span><strong>+${pick.edge}</strong></div>
    </div>
    <p class="pick-reason">${pick.reason}</p>
  </article>`;
}

function algoCenter(game) {
  const m = game.model;
  const mk = game.market;
  const away = game.matchup.away;
  const home = game.matchup.home;
  const fav = m.favorite_side === "home" ? home.name : away.name;
  const top = game.top_pick;
  return `<section class="algo-hero panel">
    <div class="algo-hero-head">
      <span class="league-pill">${game.league_name}</span>
      <h1>${away.name} <span class="at">@</span> ${home.name}</h1>
      <p class="game-meta">${formatTime(game.start_time)} · ${game.status_detail || game.status}</p>
    </div>
    <div class="algo-core">
      <div class="algo-probability">
        <span>Algo V2 win probability</span>
        <strong class="prob-value">${m.win_probability}%</strong>
        <small>Model favorite: ${fav}</small>
      </div>
      <div class="odds-row game-odds">
        <div class="odds-chip"><span>${away.name}</span><strong>${formatOdds(mk.away_moneyline)}</strong><small>Model ${formatOdds(m.away_projection)}</small></div>
        <div class="odds-chip"><span>${home.name}</span><strong>${formatOdds(mk.home_moneyline)}</strong><small>Model ${formatOdds(m.home_projection)}</small></div>
        <div class="odds-chip"><span>Spread / O-U</span><strong>${mk.spread ?? "—"} / ${mk.over_under ?? "—"}</strong><small>${mk.provider || "ESPN"}</small></div>
      </div>
    </div>
    ${top ? `<div class="game-pick ${confClass(top.confidence)}"><strong>${top.strategy_label}</strong><span>${top.team_name} · ${formatOdds(top.market_odds)} vs model ${formatOdds(top.model_projection)} (+${top.edge})</span><p>${top.reason}</p></div>` : `<div class="game-pick neutral"><strong>No value flag</strong><span>Model leans ${fav}; lines do not beat model price today.</span></div>`}
    <details class="factor-details" open><summary>Algo factor breakdown</summary><div class="factor-list">${factorBars(m.factors)}</div></details>
    ${(game.recommendations || []).length ? `<div class="rec-list"><h3>All model recommendations</h3>${game.recommendations.map((p) => pickCard({ ...p, league_name: game.league_name, matchup: `${away.name} @ ${home.name}`, start_time: game.start_time })).join("")}</div>` : ""}
  </section>`;
}

function viewDashboard() {
  const picks = state.slate?.recommended_bets || [];
  appRoot.innerHTML = `
    <section class="page-head page-head-lean">
      <h1>Today's algo picks</h1>
      <p>Algo V2 value bets with +50 edge or higher. Browse <a class="text-link" href="#/games">games</a>, <a class="text-link" href="#/teams">teams</a>, or <a class="text-link" href="#/tracking">tracking</a>.</p>
    </section>
    <div class="picks-grid">${picks.length ? picks.map((p) => pickCard(p)).join("") : '<div class="panel empty-panel">No bets meet the +50 minimum edge threshold today.</div>'}</div>`;
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
  appRoot.innerHTML = `<section class="page-head"><h1>Algo picks</h1><p>Only bets with +50 edge or higher vs Algo V2 fair prices.</p></section>
    <div class="picks-grid">${picks.length ? picks.map((p) => pickCard(p)).join("") : '<div class="panel empty-panel">No bets meet the +50 minimum edge threshold today.</div>'}</div>`;
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
    <div class="win-chip"><span>Algo</span><strong>${fav}</strong><small>${m.win_probability}%</small></div></div>
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

function viewTracking() {
  const period = state.trackingPeriod;
  const rows =
    period === "all_time"
      ? [state.tracking?.all_time]
      : state.tracking?.[period] || [];
  const bets = state.tracking?.bets || [];
  appRoot.innerHTML = `<section class="page-head"><h1>Bet tracking</h1><p>Every algo value bet logged day-to-day with win/loss grading at closing moneyline odds (1u flat).</p></section>
    <div class="period-tabs">${["daily", "weekly", "monthly", "yearly", "all_time"]
      .map(
        (p) =>
          `<button type="button" class="period-tab ${period === p ? "active" : ""}" data-period="${p}">${p.replace("_", " ")}</button>`,
      )
      .join("")}</div>
    <div class="rollup-grid">${renderTrackingSummary()}</div>
    <section class="section"><h2>Period rollups</h2>
    <table class="data-table"><thead><tr><th>Period</th><th>Record</th><th>Units</th><th>ROI</th><th>Bets</th></tr></thead>
    <tbody>${(rows || [])
      .filter(Boolean)
      .map(
        (r) =>
          `<tr><td>${r.label || r.key}</td><td>${r.record}</td><td>${r.units > 0 ? "+" : ""}${r.units}u</td><td>${r.roi_percent}%</td><td>${r.bets}</td></tr>`,
      )
      .join("") || "<tr><td colspan='5'>No settled periods yet.</td></tr>"}</tbody></table></section>
    <section class="section"><h2>Bet log</h2>
    <div class="bet-log">${bets.length ? bets.map((b) => `<article class="bet-row panel"><div class="bet-row-top"><div><strong>${b.team_name}</strong><span class="league-pill">${b.league_name}</span>${statusBadge(b.status, b.units)}</div><span class="edge-tag">+${b.edge} edge</span></div>
    <p class="muted">${b.matchup} · ${b.date}</p>
    <div class="pick-odds compact"><div><span>Market</span><strong>${formatOdds(b.market_odds)}</strong></div><div><span>Model</span><strong>${formatOdds(b.model_projection)}</strong></div><div><span>Strategy</span><strong>${b.strategy_label}</strong></div></div>
    ${b.final_score ? `<p class="final-score">Final: ${b.final_score}</p>` : ""}</article>`).join("") : '<div class="panel empty-panel">Tracking starts when value bets are published on the daily slate.</div>'}</div></section>`;
  appRoot.querySelectorAll(".period-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.trackingPeriod = btn.dataset.period;
      viewTracking();
    });
  });
}

function highlightNav(route) {
  document.querySelectorAll("#mainNav a").forEach((a) => {
    const r = a.dataset.route;
    a.classList.toggle("active", r === `/${route.path}` || (route.path === "" && r === "/"));
  });
}

async function render() {
  const route = parseRoute();
  highlightNav(route);
  try {
    if (route.path === "picks") viewPicks();
    else if (route.path === "games") viewGames(route.parts[0]);
    else if (route.path === "game") viewGame(route.parts[0]);
    else if (route.path === "teams") viewTeams(route.parts[0]);
    else if (route.path === "team") await viewTeam(route.parts[0], route.parts[1]);
    else if (route.path === "tracking") viewTracking();
    else viewDashboard();
  } catch (err) {
    appRoot.innerHTML = `<div class="panel empty-panel error-panel">${err.message}</div>`;
  }
}

async function loadPlatform() {
  const [slate, tracking, teamsIndex] = await Promise.all([
    fetchJson(USE_STATIC_API ? api("daily-slate.json") : api("daily/slate")),
    fetchJson(USE_STATIC_API ? api("tracking.json") : api("tracking")),
    fetchJson(USE_STATIC_API ? api("teams-index.json") : api("teams")),
  ]);
  state.slate = slate;
  state.tracking = tracking;
  state.teamsIndex = teamsIndex;
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
