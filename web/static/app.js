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
  trackingPeriod: "all_time",
};

const appRoot = document.getElementById("appRoot");
const leagueMenu = document.getElementById("leagueMenu");
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
    const sideMargin = pick.side === "home" ? pick.model_margin : -pick.model_margin;
    const fair =
      pick.model_projection != null ? ` · fair ${formatOdds(pick.model_projection)}` : "";
    return `Margin ${formatSpread(sideMargin)}${fair}`;
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
        ? ` margin ${basketball.predicted_margin}`
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
      <p class="db-game-links"><a href="#/database/${game.league}/${game.matchup?.away?.abbr}">${away.name} sheet</a> · <a href="#/database/${game.league}/${game.matchup?.home?.abbr}">${home.name} sheet</a> · <a href="#/database/${game.league}">League board</a></p>
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
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Teams", href: "#/teams" },
    { label: profile.label },
  ])}<section class="page-head"><span class="league-pill">${profile.league_name}</span><h1>${profile.label}</h1><p>Season ${profile.season_year} · Data through ${profile.cutoff_date}</p></section>
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
      <a href="#/database/${league}/${matchup.home?.abbr}">${home} sheet</a>
      <a href="#/database/${league}/${matchup.away?.abbr}">${away} sheet</a>
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
      return `<tr><td>${row.rank ?? "—"}</td><td><a href="#/database/${league}/${abbr}"><strong>${row.name}</strong></a></td><td>${wins}-${losses}</td><td>${wp}</td><td>${row.games_behind ?? "—"}</td><td>${row.streak ?? "—"}</td><td>${row.point_differential ?? "—"}</td></tr>`;
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

async function viewDatabase() {
  const manifest = await loadDbManifest();
  const leagues = manifest.leagues || [];
  appRoot.innerHTML = `${breadcrumbs([{ label: "Home", href: "#/" }, { label: "Database" }])}<section class="page-head"><h1>Sports Database</h1>
    <p>Complete betting research hub — standings, player files, team sheets, model ratings, and today's market edges across ${manifest.league_count || leagues.length} leagues.</p>
    <p class="muted">${manifest.players_built || 0} player files · ${manifest.teams_built || 0} team sheets · Schema v${manifest.schema_version || 2}</p></section>
    <div class="db-league-grid">${leagues
      .map(
        (lg) => `<a class="db-league-card panel" href="#/database/${lg.id}">
          <span class="league-pill">${lg.category}</span>
          <h3>${lg.name}</h3>
          <p class="muted">${lg.team_count || 0} teams · ${lg.games_today || 0} games today · ${lg.players_built || 0} players built</p>
        </a>`,
      )
      .join("")}</div>`;
}

async function viewDatabaseLeague(league) {
  const data = await loadLeagueDb(league);
  const teams = (data.standings?.teams || []).slice(0, 40);
  const betting = data.betting || {};
  const games = betting.games_today || [];
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Database", href: "#/database" },
    { label: data.league?.name || league.toUpperCase() },
  ])}<section class="page-head">
    <span class="league-pill">${data.league?.category}</span>
    <h1>${data.league?.name} — League Sheet</h1>
    <p>${data.profile?.description || "Full league database for betting decisions."}</p>
    <p class="muted">Season ${data.season_year} · ${betting.game_count || 0} games today · Updated ${new Date(data.generated_at).toLocaleString()}</p>
  </section>
  ${games.length ? `<section class="section"><div class="section-head"><h2>Today's betting board (${games.length})</h2></div>
    <div class="db-bet-grid">${games.map((g) => renderBettingGameCard(g, league)).join("")}</div></section>` : ""}
  <div class="db-grid-two">
    <section class="section"><div class="section-head"><h2>Standings</h2></div>${renderStandingsTable(data.standings, league)}</section>
    <section class="section"><div class="section-head"><h2>Power ratings</h2></div><div class="panel">${renderRatingsSummary(data.ratings)}</div></section>
  </div>
  <section class="section"><div class="section-head"><h2>Latest news</h2></div>${renderNewsList(data.news)}</section>
  <section class="section"><div class="section-head"><h2>Teams</h2></div>
    <div class="team-grid">${teams
      .map(
        (row) => `<a class="team-card panel" href="#/database/${league}/${row.abbr}">
          <strong>${row.name}</strong>
          <span class="muted">#${row.rank ?? "—"} · ${row.wins ?? "—"}-${row.losses ?? "—"}</span>
        </a>`,
      )
      .join("")}</div>
  </section>
  <a class="btn btn-secondary" href="#/database">← All leagues</a>`;
}

async function viewDatabaseTeam(league, abbr) {
  let team;
  try {
    team = await loadTeamDb(league, abbr);
  } catch {
    appRoot.innerHTML = `<div class="panel empty-panel">Team sheet not built yet for ${abbr.toUpperCase()}. It appears after the next daily rebuild when the team is on the slate.</div>
      <a class="btn btn-secondary" href="#/database/${league}">← ${league.toUpperCase()}</a>`;
    return;
  }
  const trends = team.trends || {};
  const projection = team.projection || {};
  const ratings = team.ratings || {};
  const betting = team.betting || {};
  const upcoming = betting.upcoming_games || [];
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Database", href: "#/database" },
    { label: league.toUpperCase(), href: `#/database/${league}` },
    { label: team.team?.name || abbr.toUpperCase() },
  ])}<section class="page-head">
    ${team.team?.logo ? `<img class="db-team-logo" src="${team.team.logo}" alt="">` : ""}
    <span class="league-pill">${league.toUpperCase()}</span>
    <h1>${team.team?.name || abbr.toUpperCase()} — Team Sheet</h1>
    <p>${team.team?.record_summary || ""} ${team.team?.standing_summary ? "· " + team.team.standing_summary : ""}</p>
    ${team.team?.coach ? `<p class="muted">Head coach: ${team.team.coach}</p>` : ""}
    <p><a href="#/teams/${league}/${abbr}">Season profile</a> · <a href="#/database/${league}">League sheet</a></p>
  </section>
  ${upcoming.length ? `<section class="section"><div class="section-head"><h2>Upcoming — betting context</h2></div>
    <div class="db-bet-grid">${upcoming.map((g) => renderBettingGameCard(g, league)).join("")}</div></section>` : ""}
  ${(team.injuries || []).length ? `<section class="section panel"><h2>Injuries / availability</h2><ul class="db-recent">${team.injuries.map((p) => `<li><strong>${p.name}</strong> (${p.position}) — ${p.status}</li>`).join("")}</ul></section>` : ""}
  <div class="stat-grid">
    <div class="stat-card panel"><span>Streak</span><strong>${trends.streak || "—"}</strong></div>
    <div class="stat-card panel"><span>Last 5</span><strong>${trends.last_5 || "—"}</strong></div>
    <div class="stat-card panel"><span>Power rating</span><strong>${ratings.power?.power ?? projection.power_rating ?? "—"}</strong></div>
    <div class="stat-card panel"><span>Projected pace</span><strong>${projection.projected_wins_pace ?? "—"} wins</strong></div>
  </div>
  <div class="db-grid-two">
    <section class="section panel"><h2>Recent games</h2>
      ${(team.recent_games || []).length ? `<ul class="db-recent">${team.recent_games
        .map((g) => `<li><strong>${g.result}</strong> ${g.date} vs ${g.opponent} (${g.score}) · ${g.location}</li>`)
        .join("")}</ul>` : `<p class="muted">No recent game log in snapshot.</p>`}
    </section>
    <section class="section panel"><h2>Season stats</h2>
      ${(team.stats?.categories || [])
        .slice(0, 2)
        .map(
          (cat) => `<h3>${cat.name}</h3><ul class="db-stat-list">${(cat.stats || [])
            .slice(0, 8)
            .map((s) => `<li><span>${s.name}</span><strong>${s.display ?? s.value}</strong></li>`)
            .join("")}</ul>`,
        )
        .join("") || `<p class="muted">Season stats unavailable.</p>`}
    </section>
  </div>
  <section class="section panel"><h2>Roster (${(team.roster || []).length} players · ${(team.players_built || []).length} full files)</h2>
    <div class="db-roster-grid">${(team.players_index || team.roster || [])
      .map(
        (p) => `<a class="db-player" href="#/database/${league}/player/${p.id}">
          ${p.headshot ? `<img src="${p.headshot}" alt="">` : ""}
          <strong>${p.name}</strong>
          <span class="muted">${p.position || ""}${p.jersey ? " #" + p.jersey : ""}${p.status ? " · " + p.status : ""}</span>
        </a>`,
      )
      .join("")}</div>
  </section>
  <a class="btn btn-secondary" href="#/database/${league}">← ${league.toUpperCase()} league sheet</a>`;
}

async function viewDatabasePlayer(league, playerId) {
  let player;
  try {
    player = await loadPlayerDb(league, playerId);
  } catch {
    appRoot.innerHTML = `<div class="panel empty-panel">Player file not found. Player sheets are built for slate teams on each daily rebuild.</div>
      <a class="btn btn-secondary" href="#/database/${league}">← ${league.toUpperCase()}</a>`;
    return;
  }
  const info = player.player || {};
  const teamAbbr = player.team_abbr || "";
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Database", href: "#/database" },
    { label: league.toUpperCase(), href: `#/database/${league}` },
    ...(teamAbbr ? [{ label: teamAbbr.toUpperCase(), href: `#/database/${league}/${teamAbbr}` }] : []),
    { label: info.name || "Player" },
  ])}<section class="page-head db-player-head">
    ${info.headshot ? `<img class="db-player-photo" src="${info.headshot}" alt="">` : ""}
    <div>
      <span class="league-pill">${league.toUpperCase()}</span>
      <h1>${info.name || "Player"}</h1>
      <p>${info.position || ""}${info.jersey ? " · #" + info.jersey : ""} · ${info.status || "Active"}</p>
      <p class="muted">${info.height || ""} ${info.weight || ""} · Age ${info.age ?? "—"} · ${info.experience ?? "—"} yrs exp</p>
      ${teamAbbr ? `<a href="#/database/${league}/${teamAbbr}">← Team sheet</a>` : ""}
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
  <a class="btn btn-secondary" href="#/database/${league}/${teamAbbr}">← Team sheet</a>`;
}

function highlightNav(route) {
  const isActive = (r) =>
    r === `/${route.path}` || (route.path === "" && r === "/");

  document.querySelectorAll("#mainNav a, #mobileBottomNav a").forEach((a) => {
    a.classList.toggle("active", isActive(a.dataset.route));
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
  highlightNav(route);
  closeMobileNav();
  try {
    if (route.path === "picks") viewPicks();
    else if (route.path === "games") viewGames(route.parts[1]);
    else if (route.path === "game") viewGame(route.parts[1]);
    else if (route.path === "teams") viewTeams(route.parts[1]);
    else if (route.path === "team") await viewTeam(route.parts[1], route.parts[2]);
    else if (route.path === "tracking") viewTracking();
    else if (route.path === "database") {
      const league = route.parts[1];
      const seg = route.parts[2];
      const third = route.parts[3];
      if (league && seg === "player" && third) await viewDatabasePlayer(league, third);
      else if (league && seg && seg !== "player") await viewDatabaseTeam(league, seg);
      else if (league) await viewDatabaseLeague(league);
      else await viewDatabase();
    }
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
