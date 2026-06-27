const META_BASE_PATH =
  document.querySelector('meta[name="base-path"]')?.content ?? "";
const IS_GITHUB_IO = window.location.hostname.endsWith("github.io");
const IS_LOCAL_DEV =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";

// GitHub project pages live under /Repo-Name; custom domains serve from /.
const BASE_PATH = IS_GITHUB_IO
  ? META_BASE_PATH || "/Sports-Odds-Algorithms"
  : "";

// Prebuilt JSON in /api unless we're hitting the local FastAPI dev server.
const USE_STATIC_API = !IS_LOCAL_DEV;

const state = {
  slate: null,
  tracking: null,
  teamsIndex: null,
  teamProfiles: {},
  dbManifest: null,
  dbCache: {},
  selectedLeague: "all",
  sidebarLeague: null,
  trackingPeriod: "all_time",
};

const appRoot = document.getElementById("appRoot");
const leagueMenu = document.getElementById("leagueMenu");
const leagueBrowse = document.getElementById("leagueBrowse");
const teamMenu = document.getElementById("teamMenu");
const sidebarTeams = document.getElementById("sidebarTeams");
const sidebarTeamsTitle = document.getElementById("sidebarTeamsTitle");
const sidebarNav = document.getElementById("sidebarNav");
const gameMenu = document.getElementById("gameMenu");
const sidebarGames = document.getElementById("sidebarGames");
const sidebarGamesTitle = document.getElementById("sidebarGamesTitle");
const footerUpdated = document.getElementById("footerUpdated");
const themeToggle = document.getElementById("themeToggle");
const navToggle = document.getElementById("navToggle");
const mainNav = document.getElementById("mainNav");
const mobileBottomNav = document.getElementById("mobileBottomNav");

function breadcrumbs(items) {
  if (!items?.length) return "";
  return `<nav class="breadcrumbs" aria-label="Breadcrumb">${items
    .map((item, i) => {
      const isLast = i === items.length - 1;
      if (isLast) {
        return `<span class="breadcrumb-current" aria-current="page">${item.label}</span>`;
      }
      return `<a href="${item.href}">${item.label}</a><span class="breadcrumb-sep" aria-hidden="true">/</span>`;
    })
    .join("")}</nav>`;
}

function api(path) {
  const prefix = BASE_PATH.replace(/\/$/, "");
  if (USE_STATIC_API) {
    return `${prefix}/api/${path}`;
  }
  return `/api/${path}`;
}

async function fetchJson(url) {
  const separator = url.includes("?") ? "&" : "?";
  const response = await fetch(`${url}${separator}_=${Date.now()}`, {
    cache: "no-store",
  });
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
  const saved = localStorage.getItem("soa-theme");
  setTheme(saved === "dark" ? "dark" : "light");
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
    const teamMargin = pick.side === "home" ? pick.model_margin : -pick.model_margin;
    const role = pick.side === "home" ? "Home" : "Away";
    const fair =
      pick.model_projection != null ? ` · fair ${formatOdds(pick.model_projection)}` : "";
    return `${role} margin ${formatSpread(teamMargin)}${fair}`;
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

function leagueHref(league) {
  return `#/teams/${league}`;
}

function teamHref(league, abbr) {
  return `#/team/${league}/${abbr}`;
}

function playerHref(league, playerId) {
  return `#/player/${league}/${playerId}`;
}

function leaguesForBrowse() {
  const manifest = state.dbManifest?.leagues || [];
  if (manifest.length) {
    return manifest.map((lg) => ({
      id: lg.id,
      name: lg.name,
      category: lg.category || "Other",
      team_count: lg.team_count || 0,
      games_today: lg.games_today || 0,
    }));
  }
  return (state.teamsIndex?.leagues || []).map((lg) => ({
    id: lg.id,
    name: lg.name,
    category: lg.category || "Other",
    team_count: (lg.teams || []).length,
    games_today: gamesForLeague(lg.id).length,
  }));
}

function teamsForLeague(league) {
  const fromIndex = state.teamsIndex?.leagues?.find((lg) => lg.id === league);
  if (fromIndex?.teams?.length) return fromIndex.teams;
  const cached = state.dbCache[`league:${league}`];
  const standings = cached?.standings?.teams || [];
  return standings.map((row) => ({
    abbr: row.abbr,
    label: row.name || row.abbr?.toUpperCase(),
  }));
}

function renderSidebarNav(route) {
  if (!sidebarNav) return;
  const path = route?.path || "";
  const items = [
    { href: "#/", label: "Home", active: !path },
    { href: "#/games", label: "Games", active: path === "games" || path === "game" },
    { href: "#/teams", label: "Leagues", active: path === "teams" || path === "team" || path === "player" },
    { href: "#/picks", label: "Algo picks", active: path === "picks" },
    { href: "#/tracking", label: "Tracking", active: path === "tracking" },
  ];
  sidebarNav.innerHTML = items
    .map(
      (item) =>
        `<li><a href="${item.href}" class="sidebar-nav-link ${item.active ? "active" : ""}">${item.label}</a></li>`,
    )
    .join("");
}

function renderLeagueBrowse(activeLeague) {
  if (!leagueBrowse) return;
  const leagues = leaguesForBrowse();
  if (!leagues.length) {
    leagueBrowse.innerHTML = `<p class="sidebar-empty">League index loading…</p>`;
    return;
  }
  const byCategory = new Map();
  for (const lg of leagues) {
    const cat = lg.category || "Other";
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat).push(lg);
  }
  const categories = [...byCategory.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  leagueBrowse.innerHTML = categories
    .map(([category, rows]) => {
      const leagueRows = rows
        .sort((a, b) => a.name.localeCompare(b.name))
        .map((lg) => {
          const active = activeLeague === lg.id ? " active" : "";
          const gamesToday = lg.games_today ? ` · ${lg.games_today} today` : "";
          return `<a href="${leagueHref(lg.id)}" class="browse-league${active}" data-league="${lg.id}">
            <span class="browse-league-name">${lg.name}</span>
            <span class="browse-league-meta">${lg.team_count || 0} teams${gamesToday}</span>
          </a>`;
        })
        .join("");
      return `<div class="browse-category"><div class="browse-category-label">${category}</div>${leagueRows}</div>`;
    })
    .join("");
}

function renderTeamSubmenu(league, activeAbbr) {
  if (!sidebarTeams || !teamMenu) return;
  const teams = teamsForLeague(league);
  if (!teams.length) {
    sidebarTeams.hidden = true;
    return;
  }
  const leagueName =
    leaguesForBrowse().find((lg) => lg.id === league)?.name || league.toUpperCase();
  sidebarTeams.hidden = false;
  sidebarTeamsTitle.textContent = leagueName;
  teamMenu.innerHTML = teams
    .map((team) => {
      const abbr = (team.abbr || "").toLowerCase();
      const active = activeAbbr === abbr ? " active" : "";
      return `<li><a href="${teamHref(league, abbr)}" class="team-link${active}">${team.label || abbr.toUpperCase()}</a></li>`;
    })
    .join("");
}

function renderSidebar(route) {
  renderSidebarNav(route);
  renderLeagueBrowse(state.sidebarLeague);
  if (state.sidebarLeague) {
    const activeAbbr =
      route?.path === "team" ? (route.parts[2] || "").toLowerCase() : null;
    renderTeamSubmenu(state.sidebarLeague, activeAbbr);
  } else {
    if (sidebarTeams) sidebarTeams.hidden = true;
  }
}

function leaguesForSidebar() {
  const fromIndex = state.teamsIndex?.leagues || [];
  if (fromIndex.length) return fromIndex;

  const games = state.slate?.games || [];
  const byLeague = new Map();
  for (const game of games) {
    const id = game.league;
    if (!id || byLeague.has(id)) continue;
    byLeague.set(id, {
      id,
      name: game.league_name || id.toUpperCase(),
      team_count: games.filter((g) => g.league === id).length,
    });
  }
  return [...byLeague.values()].sort((a, b) => a.name.localeCompare(b.name));
}

function renderLeagueMenu() {
  const leagues = leaguesForSidebar();
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
  const hockey = m.hockey_pred;
  const football = m.football_pred;
  const soccer = m.soccer_pred;
  const dbRating = m.db_rating;
  if (!legacy && !power && !basketball && !baseball && !hockey && !football && !soccer && !dbRating) return "";
  const parts = [];
  const layerTag =
    m.blend_layers >= 4
      ? "4-layer"
      : m.blend_layers === 3
        ? "3-layer"
        : m.blend_layers === 2
          ? "2-layer"
          : "";
  if (layerTag) {
    parts.push(layerTag);
  }
  if (legacy) {
    const legacyHome =
      legacy.home_win_probability ??
      (legacy.favorite_side === "home"
        ? legacy.win_probability
        : 100 - legacy.win_probability);
    parts.push(`Legacy V2: ${legacyHome}% home`);
  }
  if (power) {
    parts.push(`Power: ${power.home_win_probability}% home (${power.home_power} vs ${power.away_power})`);
  }
  if (basketball) {
    const margin =
      basketball.predicted_margin != null
        ? ` margin ${formatSpread(-basketball.predicted_margin)}`
        : "";
    parts.push(`Matrix: ${basketball.home_win_probability}% home${margin}`);
  }
  if (baseball) {
    const elo =
      baseball.elo_exp != null ? ` Elo ${baseball.elo_exp}%` : "";
    parts.push(`MLB-Model: ${baseball.home_win_probability}% home${elo}`);
  }
  if (hockey) {
    const xg =
      hockey.expected_home_goals != null
        ? ` xG ${hockey.expected_home_goals}-${hockey.expected_away_goals}`
        : "";
    parts.push(`Hockey-predictions: ${hockey.home_win_probability}% home${xg}`);
  }
  if (football) {
    const spread =
      football.projected_spread != null
        ? ` spread ${football.projected_spread > 0 ? "+" : ""}${football.projected_spread}`
        : "";
    parts.push(`nfelo: ${football.home_win_probability}% home${spread}`);
  }
  if (soccer) {
    const xg =
      soccer.expected_home_goals != null
        ? ` xG ${soccer.expected_home_goals}-${soccer.expected_away_goals}`
        : "";
    parts.push(
      `Soccer: ${soccer.home_win_probability}% / ${soccer.draw_probability}% / ${soccer.away_win_probability}%${xg}`,
    );
  }
  if (dbRating) {
    const sparse =
      dbRating.sparse_schedule_factor > 0.35 ? " · sparse boost" : "";
    parts.push(
      `DB Ratings (${dbRating.source_home}): ${dbRating.home_rating} vs ${dbRating.away_rating} → ${dbRating.home_win_probability}% home${sparse}`,
    );
  }
  const context = m.soccer_context;
  if (context?.factors?.length) {
    const ctxParts = context.factors.map(
      (f) => `${f.label}${f.detail ? ` (${f.detail})` : ""}`,
    );
    parts.push(`Context: ${ctxParts.join("; ")}`);
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
  const threeway = m.threeway;
  const algoLabel = threeway
    ? "1X2 model probabilities"
    : m.algorithm === "Unified"
      ? "Unified model"
      : "Algo V2 win probability";
  const probBlock = threeway
    ? `<div class="algo-probability threeway">
        <span>${algoLabel}</span>
        <div class="threeway-grid">
          <div><small>${away.name}</small><strong>${m.away_win_probability}%</strong></div>
          <div><small>Draw</small><strong>${m.draw_probability}%</strong></div>
          <div><small>${home.name}</small><strong>${m.home_win_probability}%</strong></div>
        </div>
        ${m.soccer_pred?.expected_home_goals != null ? `<small>Projected score ${m.soccer_pred.expected_away_goals}-${m.soccer_pred.expected_home_goals}</small>` : ""}
        ${m.soccer_context?.factors?.length ? `<details class="factor-details"><summary>Context factors (ESPN)</summary><div class="factor-list">${m.soccer_context.factors.map((f) => `<div class="factor-row"><span>${f.label}</span><small>${f.detail || ""}</small></div>`).join("")}</div></details>` : ""}
      </div>`
    : `<div class="algo-probability">
        <span>${algoLabel}</span>
        <strong class="prob-value">${m.win_probability}%</strong>
        <small>Model favorite: ${fav}</small>
      </div>`;
  const drawChip =
    threeway && mk.draw_moneyline != null
      ? `<div class="odds-chip"><span>Draw</span><strong>${formatOdds(mk.draw_moneyline)}</strong><small>Model ${formatOdds(m.draw_projection)}</small></div>`
      : "";
  const oddsRow = `<div class="odds-row game-odds">
        <div class="odds-chip"><span>${away.name}</span><strong>${formatOdds(mk.away_moneyline)}</strong><small>Model ${formatOdds(m.away_projection)}</small></div>
        ${drawChip}
        <div class="odds-chip"><span>${home.name}</span><strong>${formatOdds(mk.home_moneyline)}</strong><small>Model ${formatOdds(m.home_projection)}</small></div>
        ${threeway ? "" : `<div class="odds-chip"><span>Spread / O-U</span><strong>${mk.spread ?? "—"} / ${mk.over_under ?? "—"}</strong><small>${mk.provider || "ESPN"}</small></div>`}
      </div>`;
  return `<section class="algo-hero panel">
    ${breadcrumbs([
      { label: "Home", href: "#/" },
      { label: "Games", href: "#/games" },
      { label: `${away.name} @ ${home.name}` },
    ])}
    <div class="algo-hero-head">
      <span class="league-pill">${game.league_name}</span>
      <h1>${away.name} <span class="at">@</span> ${home.name}</h1>
      <p class="game-meta">${formatTime(game.start_time)} · ${game.status_detail || game.status}</p>
      <p class="db-game-links"><a href="${teamHref(game.league, game.matchup?.away?.abbr)}">${away.name}</a> · <a href="${teamHref(game.league, game.matchup?.home?.abbr)}">${home.name}</a> · <a href="${leagueHref(game.league)}">${game.league_name} league</a></p>
    </div>
    <div class="algo-core">
      ${probBlock}
      ${algoBreakdown(m)}
      ${oddsRow}
    </div>
    ${game.eligible_for_official_picks === false ? `<div class="game-pick neutral"><strong>Predictions only</strong><span>Soccer 1X2 model and fair prices are shown; official algo picks exclude soccer.</span></div>` : top ? `<div class="game-pick ${confClass(top.confidence)}"><strong>${top.strategy_label}</strong><span>${top.team_name} · ${pickMarketLabel(top)} vs model ${pickModelLabel(top)} (+${top.edge})</span><p>${top.reason}</p></div>` : `<div class="game-pick neutral"><strong>No value flag</strong><span>Model leans ${fav}; lines do not beat model price today.</span></div>`}
    <details class="factor-details" open><summary>Algo factor breakdown</summary><div class="factor-list">${factorBars(m.factors)}</div></details>
    ${game.eligible_for_official_picks !== false && (game.recommendations || []).length ? `<div class="rec-list"><h3>All model recommendations</h3>${game.recommendations.map((p) => pickCard({ ...p, league_name: game.league_name, matchup: `${away.name} @ ${home.name}`, start_time: game.start_time })).join("")}</div>` : ""}
  </section>`;
}

function viewDashboard() {
  state.sidebarLeague = null;
  renderSidebar(parseRoute());
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
        <h4>Leagues</h4>
        <strong class="rollup-record">${state.dbManifest?.league_count || leagues.length}</strong>
        <span>Standings, rosters, ratings, and team sheets</span>
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
  state.sidebarLeague = null;
  renderSidebar(parseRoute());
  const picks = state.slate?.recommended_bets || [];
  const slate = state.slate || {};
  const minEdge = slate.summary?.min_edge ?? slate.min_recommended_edge ?? 40;
  appRoot.innerHTML = `<section class="page-head"><h1>Algo picks</h1><p>Only bets with +${minEdge} edge or higher vs the unified fair prices (3-layer value agreement required where applicable).</p></section>
    <div class="picks-grid">${picks.length ? picks.map((p) => pickCard(p)).join("") : `<div class="panel empty-panel">No bets meet the +${minEdge} minimum edge threshold today.</div>`}</div>`;
}

function viewGames(league) {
  state.selectedLeague = league || "all";
  state.sidebarLeague = league && league !== "all" ? league : null;
  renderLeagueMenu();
  renderGameSubmenu(state.selectedLeague);
  renderSidebar(parseRoute());
  const games = gamesForLeague(state.selectedLeague);
  appRoot.innerHTML = `<section class="page-head"><h1>Games</h1><p>Today's matchups with full algo breakdowns. Filter by league in the sidebar or open a team sheet from any game.</p></section>
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
  state.sidebarLeague = game.league;
  renderLeagueMenu();
  renderGameSubmenu(game.league);
  renderSidebar(parseRoute());
  appRoot.innerHTML = algoCenter(game);
}

async function viewLeaguesHub() {
  state.sidebarLeague = null;
  renderSidebar(parseRoute());
  const manifest = state.dbManifest || (await loadDbManifest().catch(() => null));
  const leagues = manifest?.leagues || leaguesForBrowse();
  appRoot.innerHTML = `${breadcrumbs([{ label: "Home", href: "#/" }, { label: "Leagues" }])}<section class="page-head"><h1>Leagues</h1>
    <p>Browse every league — standings, power ratings, news, rosters, and today's betting board. Pick a league from the sidebar or below.</p>
    ${manifest ? `<p class="muted">${manifest.players_built || 0} player profiles · ${manifest.teams_built || 0} team sheets · Updated ${new Date(manifest.generated_at).toLocaleString()}</p>` : ""}</section>
    <div class="db-league-grid">${leagues
      .map(
        (lg) => `<a class="db-league-card panel" href="${leagueHref(lg.id)}">
          <span class="league-pill">${lg.category || "League"}</span>
          <h3>${lg.name}</h3>
          <p class="muted">${lg.team_count || 0} teams · ${lg.games_today || 0} games today</p>
        </a>`,
      )
      .join("")}</div>`;
}

async function viewLeaguePage(league) {
  state.sidebarLeague = league;
  renderSidebar(parseRoute());
  const data = await loadLeagueDb(league);
  renderSidebar(parseRoute());
  const teams = (data.standings?.teams || []).slice(0, 60);
  const betting = data.betting || {};
  const games = betting.games_today || [];
  const leagueName = data.league?.name || league.toUpperCase();
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Leagues", href: "#/teams" },
    { label: leagueName },
  ])}<section class="page-head">
    <span class="league-pill">${data.league?.category}</span>
    <h1>${leagueName}</h1>
    <p>${data.profile?.description || "Standings, ratings, news, and team sheets for betting research."}</p>
    <p class="muted">Season ${data.season_year} · ${betting.game_count || 0} games today · Updated ${new Date(data.generated_at).toLocaleString()}</p>
  </section>
  ${games.length ? `<section class="section"><div class="section-head"><h2>Today's board (${games.length})</h2></div>
    <div class="db-bet-grid">${games.map((g) => renderBettingGameCard(g, league)).join("")}</div></section>` : ""}
  <div class="db-grid-two">
    <section class="section"><div class="section-head"><h2>Standings</h2></div>${renderStandingsTable(data.standings, league)}</section>
    <section class="section"><div class="section-head"><h2>Power ratings</h2></div><div class="panel">${renderRatingsSummary(data.ratings)}</div></section>
  </div>
  <section class="section"><div class="section-head"><h2>Latest news</h2></div>${renderNewsList(data.news)}</section>
  <section class="section"><div class="section-head"><h2>Teams</h2><a class="text-link" href="#/teams">All leagues →</a></div>
    <div class="team-grid">${teams
      .map(
        (row) => `<a class="team-card panel" href="${teamHref(league, row.abbr)}">
          <strong>${row.name}</strong>
          <span class="muted">#${row.rank ?? "—"} · ${row.wins ?? "—"}-${row.losses ?? "—"}</span>
        </a>`,
      )
      .join("")}</div>
  </section>`;
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
  state.sidebarLeague = league;
  renderSidebar(parseRoute());
  appRoot.innerHTML = '<div class="panel empty-panel">Loading team…</div>';

  const normalizedAbbr = (abbr || "").toLowerCase();
  const [teamDb, profile] = await Promise.all([
    loadTeamDb(league, normalizedAbbr).catch(() => null),
    loadTeamProfile(league, normalizedAbbr),
  ]);

  if (!teamDb && !profile) {
    appRoot.innerHTML = `<div class="panel empty-panel">Team sheet not available for ${normalizedAbbr.toUpperCase()} yet. It is built on each daily rebuild when the team is on the slate.</div>
      <a class="btn btn-secondary" href="${leagueHref(league)}">← ${league.toUpperCase()}</a>`;
    return;
  }

  const leagueName =
    profile?.league_name ||
    leaguesForBrowse().find((lg) => lg.id === league)?.name ||
    league.toUpperCase();
  const teamName =
    teamDb?.team?.name || profile?.label || normalizedAbbr.toUpperCase();
  const trends = teamDb?.trends || {};
  const projection = teamDb?.projection || {};
  const ratings = teamDb?.ratings || {};
  const betting = teamDb?.betting || {};
  const upcoming = betting.upcoming_games || [];
  const profileStats = profile?.season_stats;
  const recentGames =
    (teamDb?.recent_games || []).length
      ? teamDb.recent_games
      : (profile?.recent_games || []).map((g) => ({
          result: g.result,
          date: g.date,
          opponent: g.opponent,
          score: Array.isArray(g.score) ? `${g.score[0]}–${g.score[1]}` : g.score,
          location: g.location || "",
        }));

  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Leagues", href: "#/teams" },
    { label: leagueName, href: leagueHref(league) },
    { label: teamName },
  ])}<section class="page-head">
    ${teamDb?.team?.logo ? `<img class="db-team-logo" src="${teamDb.team.logo}" alt="">` : ""}
    <span class="league-pill">${leagueName}</span>
    <h1>${teamName}</h1>
    <p>${teamDb?.team?.record_summary || ""} ${teamDb?.team?.standing_summary ? "· " + teamDb.team.standing_summary : ""}</p>
    ${teamDb?.team?.coach ? `<p class="muted">Head coach: ${teamDb.team.coach}</p>` : ""}
    ${profile ? `<p class="muted">Season ${profile.season_year} · Data through ${profile.cutoff_date}</p>` : ""}
    <p><a href="${leagueHref(league)}">← ${leagueName} league</a></p>
  </section>
  ${upcoming.length ? `<section class="section"><div class="section-head"><h2>Upcoming — betting context</h2></div>
    <div class="db-bet-grid">${upcoming.map((g) => renderBettingGameCard(g, league)).join("")}</div></section>` : ""}
  ${(teamDb?.injuries || []).length ? `<section class="section panel"><h2>Injuries / availability</h2><ul class="db-recent">${teamDb.injuries.map((p) => `<li><strong>${p.name}</strong> (${p.position}) — ${p.status}</li>`).join("")}</ul></section>` : ""}
  <div class="stat-grid">
    <div class="stat-card panel"><span>Record</span><strong>${profileStats ? `${profileStats.wins}-${profileStats.losses}` : teamDb?.standing ? `${teamDb.standing.wins ?? "—"}-${teamDb.standing.losses ?? "—"}` : "—"}</strong></div>
    <div class="stat-card panel"><span>Win %</span><strong>${profileStats?.win_pct != null ? `${profileStats.win_pct}%` : trends.win_percent != null ? `${(trends.win_percent * 100).toFixed(1)}%` : "—"}</strong></div>
    <div class="stat-card panel"><span>Streak</span><strong>${trends.streak || "—"}</strong></div>
    <div class="stat-card panel"><span>Power rating</span><strong>${ratings.power?.power ?? projection.power_rating ?? "—"}</strong></div>
    <div class="stat-card panel"><span>Last 5</span><strong>${trends.last_5 || "—"}</strong></div>
    <div class="stat-card panel"><span>Projected pace</span><strong>${projection.projected_wins_pace ?? "—"} wins</strong></div>
  </div>
  <div class="db-grid-two">
    <section class="section panel"><h2>Recent games</h2>
      ${recentGames.length ? `<ul class="db-recent">${recentGames
        .map((g) => `<li><strong>${g.result}</strong> ${g.date} vs ${g.opponent} (${g.score})${g.location ? " · " + g.location : ""}</li>`)
        .join("")}</ul>` : `<p class="muted">No recent game log.</p>`}
    </section>
    <section class="section panel"><h2>Season stats</h2>
      ${(teamDb?.stats?.categories || [])
        .slice(0, 2)
        .map(
          (cat) => `<h3>${cat.name}</h3><ul class="db-stat-list">${(cat.stats || [])
            .slice(0, 8)
            .map((s) => `<li><span>${s.name}</span><strong>${s.display ?? s.value}</strong></li>`)
            .join("")}</ul>`,
        )
        .join("") || (profileStats ? `<ul class="db-stat-list"><li><span>Games played</span><strong>${profileStats.games_played ?? "—"}</strong></li><li><span>Win %</span><strong>${profileStats.win_pct ?? "—"}%</strong></li></ul>` : `<p class="muted">Season stats unavailable.</p>`)}
    </section>
  </div>
  ${(teamDb?.roster || teamDb?.players_index || []).length ? `<section class="section panel"><h2>Roster (${(teamDb.roster || []).length} players)</h2>
    <div class="db-roster-grid">${(teamDb.players_index || teamDb.roster || [])
      .map(
        (p) => `<a class="db-player" href="${playerHref(league, p.id)}">
          ${p.headshot ? `<img src="${p.headshot}" alt="">` : ""}
          <strong>${p.name}</strong>
          <span class="muted">${p.position || ""}${p.jersey ? " #" + p.jersey : ""}${p.status ? " · " + p.status : ""}</span>
        </a>`,
      )
      .join("")}</div>
  </section>` : ""}
  <a class="btn btn-secondary" href="${leagueHref(league)}">← ${leagueName}</a>`;
}

async function viewPlayer(league, playerId) {
  state.sidebarLeague = league;
  renderSidebar(parseRoute());
  let player;
  try {
    player = await loadPlayerDb(league, playerId);
  } catch {
    appRoot.innerHTML = `<div class="panel empty-panel">Player profile not found. Player sheets are built for slate teams on each daily rebuild.</div>
      <a class="btn btn-secondary" href="${leagueHref(league)}">← ${league.toUpperCase()}</a>`;
    return;
  }
  const info = player.player || {};
  const teamAbbr = (player.team_abbr || "").toLowerCase();
  const leagueName =
    leaguesForBrowse().find((lg) => lg.id === league)?.name || league.toUpperCase();
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Leagues", href: "#/teams" },
    { label: leagueName, href: leagueHref(league) },
    ...(teamAbbr ? [{ label: teamAbbr.toUpperCase(), href: teamHref(league, teamAbbr) }] : []),
    { label: info.name || "Player" },
  ])}<section class="page-head db-player-head">
    ${info.headshot ? `<img class="db-player-photo" src="${info.headshot}" alt="">` : ""}
    <div>
      <span class="league-pill">${leagueName}</span>
      <h1>${info.name || "Player"}</h1>
      <p>${info.position || ""}${info.jersey ? " · #" + info.jersey : ""} · ${info.status || "Active"}</p>
      <p class="muted">${info.height || ""} ${info.weight || ""} · Age ${info.age ?? "—"} · ${info.experience ?? "—"} yrs exp</p>
      ${teamAbbr ? `<a href="${teamHref(league, teamAbbr)}">← ${teamAbbr.toUpperCase()}</a>` : ""}
    </div>
  </section>
  <div class="db-grid-two">
    <section class="section panel"><h2>Season stats</h2>
      ${(player.season_stats || []).concat(player.overview_stats || []).slice(0, 2).map((cat) =>
        `<h3>${cat.name}</h3><ul class="db-stat-list">${(cat.stats || []).slice(0, 12).map((s) =>
          `<li><span>${s.name}</span><strong>${s.display ?? s.value}</strong></li>`).join("")}</ul>`).join("") || `<p class="muted">Stats unavailable.</p>`}
    </section>
    <section class="section panel"><h2>Recent games</h2>
      ${(player.game_log || []).length ? `<ul class="db-recent">${player.game_log.map((g) =>
        `<li>${g.date || ""} ${g.opponent || ""} · ${g.score || ""}</li>`).join("")}</ul>` : `<p class="muted">No game log.</p>`}
    </section>
  </div>
  ${(player.news || []).length ? `<section class="section"><h2>Player news</h2>${renderNewsList(player.news.map((n) => ({ ...n, link: n.link })))}</section>` : ""}
  ${teamAbbr ? `<a class="btn btn-secondary" href="${teamHref(league, teamAbbr)}">← Team</a>` : `<a class="btn btn-secondary" href="${leagueHref(league)}">← ${leagueName}</a>`}`;
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
  state.sidebarLeague = null;
  renderSidebar(parseRoute());
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
      <div class="pick-odds compact"><div><span>${b.bet_type === "spread" ? "Spread" : "Market"}</span><strong>${b.bet_type === "spread" ? formatSpread(b.spread_line) + " (" + formatOdds(b.spread_odds ?? b.market_odds) + ")" : formatOdds(b.market_odds)}</strong></div><div><span>Model</span><strong>${b.bet_type === "spread" && b.model_margin != null ? (b.side === "home" ? "Home" : "Away") + " margin " + formatSpread(b.side === "home" ? b.model_margin : -b.model_margin) : formatOdds(b.model_projection)}</strong></div><div><span>Strategy</span><strong>${b.strategy_label}</strong></div></div>
      ${b.final_score ? `<p class="final-score">Final: ${b.final_score}</p>` : ""}</article>`).join("") : `<div class="panel empty-panel">No tracked bets yet. Picks with +${minEdge} edge are logged on each daily rebuild.</div>`}</div>
    </section>`;

  appRoot.querySelectorAll(".period-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.trackingPeriod = btn.dataset.period;
      viewTracking();
    });
  });
}

function dbApi(path) {
  return api(`db/${path}`);
}

async function loadDbManifest() {
  if (state.dbManifest) return state.dbManifest;
  state.dbManifest = await fetchJson(
    USE_STATIC_API ? dbApi("manifest.json") : api("db/manifest"),
  );
  return state.dbManifest;
}

async function loadLeagueDb(league) {
  const key = `league:${league}`;
  if (state.dbCache[key]) return state.dbCache[key];
  const payload = await fetchJson(
    USE_STATIC_API
      ? dbApi(`${league}/league.json`)
      : api(`db/${league}/league`),
  );
  state.dbCache[key] = payload;
  return payload;
}

async function loadTeamDb(league, abbr) {
  const key = `team:${league}:${abbr}`;
  if (state.dbCache[key]) return state.dbCache[key];
  const payload = await fetchJson(
    USE_STATIC_API
      ? dbApi(`${league}/teams/${abbr}.json`)
      : api(`db/${league}/teams/${abbr}`),
  );
  state.dbCache[key] = payload;
  return payload;
}

async function loadPlayerDb(league, playerId) {
  const key = `player:${league}:${playerId}`;
  if (state.dbCache[key]) return state.dbCache[key];
  const payload = await fetchJson(
    USE_STATIC_API
      ? dbApi(`${league}/players/${playerId}.json`)
      : api(`db/${league}/players/${playerId}`),
  );
  state.dbCache[key] = payload;
  return payload;
}

function renderBettingGameCard(sheet, league) {
  const model = sheet.model || {};
  const market = sheet.market || {};
  const matchup = sheet.matchup || {};
  const away = matchup.away?.name || "Away";
  const home = matchup.home?.name || "Home";
  const agreement = model.agreement || {};
  const picks = sheet.recommendations || [];
  const top = sheet.top_pick;
  const official = sheet.eligible_for_official_picks !== false;
  return `<article class="db-bet-card panel">
    <div class="db-bet-head">
      <span class="league-pill">${sheet.league_name || league}</span>
      <a href="#/game/${sheet.event_id}"><strong>${away} @ ${home}</strong></a>
      <span class="muted">${formatTime(sheet.start_time)}</span>
    </div>
    <div class="odds-row compact">
      <div class="odds-chip"><span>${away} ML</span><strong>${formatOdds(market.away_moneyline)}</strong><small>Model ${formatOdds(model.away_projection)}</small></div>
      <div class="odds-chip"><span>${home} ML</span><strong>${formatOdds(market.home_moneyline)}</strong><small>Model ${formatOdds(model.home_projection)}</small></div>
      ${market.spread != null ? `<div class="odds-chip"><span>Spread</span><strong>${formatSpread(market.spread)}</strong></div>` : ""}
    </div>
    <div class="db-bet-model">
      <span>Unified model</span>
      <strong>${model.win_probability ?? "—"}%</strong>
      <small>Fav: ${model.favorite_side || "—"} · Blend ${model.blend_layers || "—"} layers</small>
      ${agreement.required ? `<small>Agreement: ${agreement.agreed ? "✓ all layers" : "✗ split"} (${(agreement.value_sides || []).join(", ") || "none"})</small>` : ""}
    </div>
    ${official && top ? `<div class="game-pick ${confClass(top.confidence)}"><strong>${top.strategy_label}</strong> ${top.team_name} +${top.edge} edge</div>` : !official ? `<div class="game-pick neutral"><strong>Predictions only</strong> (soccer excluded from official picks)</div>` : `<div class="game-pick neutral"><strong>No official pick</strong> — model/market gap below threshold</div>`}
    <div class="db-bet-links">
      <a href="${teamHref(league, matchup.home?.abbr)}">${home}</a>
      <a href="${teamHref(league, matchup.away?.abbr)}">${away}</a>
      <a href="#/game/${sheet.event_id}">Full analysis →</a>
    </div>
  </article>`;
}

function renderRatingsSummary(ratings) {
  const power = ratings?.power || {};
  const teams = Object.entries(power).slice(0, 8);
  if (!teams.length) return `<p class="muted">Model ratings unavailable for this league snapshot.</p>`;
  return `<ul class="db-stat-list">${teams
    .map(([key, row]) => `<li><span>${row.name || key}</span><strong>${row.power}</strong></li>`)
    .join("")}</ul>`;
}

function renderStandingsTable(standings, league) {
  const rows = standings?.teams || [];
  if (!rows.length) return `<div class="panel empty-panel">Standings unavailable for this league.</div>`;
  return `<div class="db-table-wrap panel"><table class="db-table"><thead><tr>
    <th>#</th><th>Team</th><th>W-L</th><th>Win%</th><th>GB</th><th>Streak</th><th>Diff</th>
  </tr></thead><tbody>${rows
    .map((row) => {
      const wins = row.wins ?? "—";
      const losses = row.losses ?? "—";
      const wp = row.win_percent != null ? `${(row.win_percent * 100).toFixed(1)}%` : "—";
      const abbr = row.abbr || "";
      return `<tr><td>${row.rank ?? "—"}</td><td><a href="${teamHref(league, abbr)}"><strong>${row.name}</strong></a></td><td>${wins}-${losses}</td><td>${wp}</td><td>${row.games_behind ?? "—"}</td><td>${row.streak ?? "—"}</td><td>${row.point_differential ?? "—"}</td></tr>`;
    })
    .join("")}</tbody></table></div>`;
}

function renderNewsList(news) {
  if (!news?.length) return `<div class="panel empty-panel">No recent headlines.</div>`;
  return `<div class="db-news-list">${news
    .map(
      (item) => `<article class="db-news panel"><h3>${item.headline || "Update"}</h3>
      <p class="muted">${item.published ? new Date(item.published).toLocaleString() : ""}</p>
      <p>${item.description || ""}</p>
      ${item.link ? `<a href="${item.link}" target="_blank" rel="noopener">Read on ESPN</a>` : ""}</article>`,
    )
    .join("")}</div>`;
}

function highlightNav(route) {
  const path = route.path || "";
  const activeRoute =
    path === "" ? "/" : path === "team" || path === "player" ? "/teams" : `/${path}`;

  document.querySelectorAll("#mainNav a, #mobileBottomNav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === activeRoute);
  });
}

function closeMobileNav() {
  mainNav?.classList.remove("open");
  navToggle?.setAttribute("aria-expanded", "false");
  navToggle?.setAttribute("aria-label", "Open menu");
}

function toggleMobileNav() {
  const open = mainNav?.classList.toggle("open");
  navToggle?.setAttribute("aria-expanded", open ? "true" : "false");
  navToggle?.setAttribute("aria-label", open ? "Close menu" : "Open menu");
}

async function render() {
  const route = parseRoute();

  if (route.path === "database") {
    const league = route.parts[1];
    const seg = route.parts[2];
    const third = route.parts[3];
    if (league && seg === "player" && third) navigate(`#/player/${league}/${third}`);
    else if (league && seg && seg !== "player") navigate(`#/team/${league}/${seg}`);
    else if (league) navigate(`#/teams/${league}`);
    else navigate("#/teams");
    return;
  }

  highlightNav(route);
  closeMobileNav();
  try {
    if (route.path === "picks") viewPicks();
    else if (route.path === "games") viewGames(route.parts[1]);
    else if (route.path === "game") viewGame(route.parts[1]);
    else if (route.path === "teams") {
      if (route.parts[1]) await viewLeaguePage(route.parts[1]);
      else await viewLeaguesHub();
    }
    else if (route.path === "team") await viewTeam(route.parts[1], route.parts[2]);
    else if (route.path === "player") await viewPlayer(route.parts[1], route.parts[2]);
    else if (route.path === "tracking") viewTracking();
    else viewDashboard();
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
    state.dbManifest = await fetchJson(
      USE_STATIC_API ? dbApi("manifest.json") : api("db/manifest"),
    );
  } catch {
    state.dbManifest = null;
  }

  const stamp = slate.generated_at ? new Date(slate.generated_at) : new Date();
  footerUpdated.textContent = `Updated ${stamp.toLocaleString()}`;
  renderLeagueMenu();
  renderSidebar(parseRoute());

  await render();
}

window.addEventListener("hashchange", () => render());
themeToggle.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
});
navToggle?.addEventListener("click", toggleMobileNav);
mainNav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", closeMobileNav);
});

initTheme();
loadPlatform().catch((err) => {
  appRoot.innerHTML = `<div class="panel empty-panel error-panel">${err.message}</div>`;
});
