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

async function fetchJson(url, { timeoutMs = 30000 } = {}) {
  const separator = url.includes("?") ? "&" : "?";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${url}${separator}_=${Date.now()}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        typeof payload?.detail === "string" ? payload.detail : "Request failed",
      );
    }
    return payload;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timed out — the site may still be rebuilding.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
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
  const themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) themeColor.content = theme === "light" ? "#f8fafc" : "#151a28";
}

function initTheme() {
  const saved = localStorage.getItem("soa-theme");
  setTheme(saved === "light" ? "light" : "dark");
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

function formatGameDate(value) {
  if (!value) return "—";
  if (value.includes("T")) {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    }
  }
  const mdy = String(value).match(/^(\d{1,2})-(\d{1,2})-(\d{4})$/);
  if (mdy) {
    const parsed = new Date(Number(mdy[3]), Number(mdy[1]) - 1, Number(mdy[2]));
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    }
  }
  const parsed = new Date(value);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }
  return value;
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

/** Round algo/model ratings for display (avoids JS float artifacts like 11.399999…). */
function formatRating(value, decimals = 1) {
  if (value == null || value === "" || Number.isNaN(Number(value))) return "—";
  const factor = 10 ** decimals;
  const rounded = Math.round(Number(value) * factor) / factor;
  if (decimals === 0 || Number.isInteger(rounded)) return String(Math.round(rounded));
  return rounded.toFixed(decimals);
}

function formatRatingDiff(value, decimals = 1) {
  const text = formatRating(value, decimals);
  if (text === "—") return text;
  const num = Number(text);
  return num > 0 ? `+${text}` : text;
}

function ratingTierClass(rating) {
  if (rating == null || Number.isNaN(Number(rating))) return "unknown";
  const n = Number(rating);
  if (n >= 85) return "elite";
  if (n >= 75) return "good";
  if (n >= 55) return "average";
  return "low";
}

function titleCaseCategory(name) {
  if (!name) return "Stats";
  const text = String(name).trim();
  if (!text) return "Stats";
  return text
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatStatValue(stat) {
  const raw = stat?.display ?? stat?.displayValue ?? stat?.value;
  if (raw == null || raw === "") return "—";
  return String(raw);
}

function renderSeasonStatsSection(categories, profileStats, options = {}) {
  const blocks = (categories || [])
    .map((cat) => ({
      name: titleCaseCategory(cat.name),
      stats: (cat.stats || []).filter(
        (stat) =>
          stat &&
          (stat.name ||
            stat.display != null ||
            stat.displayValue != null ||
            stat.value != null),
      ),
    }))
    .filter((cat) => cat.stats.length > 0);

  const rowCount = blocks.reduce((total, cat) => total + cat.stats.length, 0);
  if (!rowCount) {
    if (profileStats?.games_played != null || profileStats?.win_pct != null) {
      return `<ul class="db-stat-list">
        <li><span>Games played</span><strong>${profileStats.games_played ?? "—"}</strong></li>
        <li><span>Win %</span><strong>${profileStats.win_pct ?? "—"}%</strong></li>
      </ul>`;
    }
    return `<p class="muted">${options.emptyMessage || "Season stats unavailable."}</p>`;
  }

  return `<div class="fm-season-stats">${blocks
    .map(
      (cat, index) => `<details class="fm-stat-details"${index === 0 ? " open" : ""}>
        <summary>${cat.name}<span class="fm-stat-count">${cat.stats.length}</span></summary>
        <ul class="db-stat-list">${cat.stats
          .map(
            (stat) =>
              `<li><span>${stat.name || "—"}</span><strong>${formatStatValue(stat)}</strong></li>`,
          )
          .join("")}</ul>
      </details>`,
    )
    .join("")}</div>`;
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

function pickTeamAbbr(pick, matchup) {
  if (pick?.team_abbr) return String(pick.team_abbr).toLowerCase();
  const side = pick?.side;
  if (side && matchup?.[side]?.abbr) return String(matchup[side].abbr).toLowerCase();
  return null;
}

function teamNameLink(league, abbr, name, className = "team-link-inline") {
  const lg = league || "";
  const token = (abbr || "").toLowerCase();
  if (!lg || !token || !name) return name || "";
  return `<a href="${teamHref(lg, token)}" class="${className}">${name}</a>`;
}

function pickTeamNameLink(pick, league, matchup) {
  const abbr = pickTeamAbbr(pick, matchup);
  return abbr ? teamNameLink(league, abbr, pick.team_name) : pick.team_name || "";
}

function matchupLinks(league, away, home, className = "team-link-inline") {
  const awayName = away?.name || away || "Away";
  const homeName = home?.name || home || "Home";
  const awayAbbr = away?.abbr || null;
  const homeAbbr = home?.abbr || null;
  return `${teamNameLink(league, awayAbbr, awayName, className)} <span class="at">@</span> ${teamNameLink(league, homeAbbr, homeName, className)}`;
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

function pickGameHref(pick) {
  if (pick?.event_id) return `#/game/${pick.event_id}`;
  const games = state.slate?.games || [];
  const match = games.find(
    (game) =>
      game.league === pick?.league &&
      (pick?.matchup === `${game.matchup?.away?.name} @ ${game.matchup?.home?.name}` ||
        pick?.team_name === game.matchup?.home?.name ||
        pick?.team_name === game.matchup?.away?.name),
  );
  return match ? `#/game/${match.event_id}` : null;
}

function pickCard(pick, extra = "") {
  const gameHref = pickGameHref(pick);
  const tag = gameHref ? "a" : "article";
  const linkAttrs = gameHref
    ? ` href="${gameHref}" class="pick-card pick-card-link ${confClass(pick.confidence)}"`
    : ` class="pick-card ${confClass(pick.confidence)}"`;
  return `<${tag}${linkAttrs}>
    <div class="pick-top"><span class="league-pill">${pick.league_name || pick.league}</span><span class="strategy-pill">${pick.strategy_label || pick.strategy}</span></div>
    <h3>${pick.team_name || ""}</h3>
    <p class="pick-matchup">${pick.matchup || extra}</p>
    <p class="pick-time">${formatTime(pick.start_time)}</p>
    <div class="pick-odds">
      <div><span>${pick.bet_type === "spread" ? "Spread" : "Market"}</span><strong>${pickMarketLabel(pick)}</strong></div>
      <div><span>Model</span><strong>${pickModelLabel(pick)}</strong></div>
      <div><span>Edge</span><strong>+${pick.edge}</strong></div>
    </div>
    <p class="pick-reason">${pick.reason}</p>
    ${gameHref ? `<span class="pick-open-hint">Open game prediction →</span>` : ""}
  </${tag}>`;
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
    parts.push(
      `Power: ${power.home_win_probability}% home (${formatRating(power.home_power)} vs ${formatRating(power.away_power)})`,
    );
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
      `DB Ratings (${dbRating.source_home}): ${formatRating(dbRating.home_rating)} vs ${formatRating(dbRating.away_rating)} → ${dbRating.home_win_probability}% home${sparse}`,
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
          <div><small>${teamNameLink(game.league, away.abbr, away.name)}</small><strong>${m.away_win_probability}%</strong></div>
          <div><small>Draw</small><strong>${m.draw_probability}%</strong></div>
          <div><small>${teamNameLink(game.league, home.abbr, home.name)}</small><strong>${m.home_win_probability}%</strong></div>
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
        <div class="odds-chip"><span>${teamNameLink(game.league, away.abbr, away.name)}</span><strong>${formatOdds(mk.away_moneyline)}</strong><small>Model ${formatOdds(m.away_projection)}</small></div>
        ${drawChip}
        <div class="odds-chip"><span>${teamNameLink(game.league, home.abbr, home.name)}</span><strong>${formatOdds(mk.home_moneyline)}</strong><small>Model ${formatOdds(m.home_projection)}</small></div>
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
      <h1>${matchupLinks(game.league, away, home)}</h1>
      <p class="game-meta">${formatTime(game.start_time)} · ${game.status_detail || game.status}</p>
      <p class="db-game-links"><a href="${teamHref(game.league, game.matchup?.away?.abbr)}">${away.name}</a> · <a href="${teamHref(game.league, game.matchup?.home?.abbr)}">${home.name}</a> · <a href="${leagueHref(game.league)}">${game.league_name} league</a></p>
    </div>
    <div class="algo-core">
      ${probBlock}
      ${algoBreakdown(m)}
      ${oddsRow}
    </div>
    ${game.eligible_for_official_picks === false ? `<div class="game-pick neutral"><strong>Predictions only</strong><span>Soccer 1X2 model and fair prices are shown; official algo picks exclude soccer.</span></div>` : top ? `<div class="game-pick ${confClass(top.confidence)}"><strong>${top.strategy_label}</strong><span>${pickTeamNameLink(top, game.league, game.matchup)} · ${pickMarketLabel(top)} vs model ${pickModelLabel(top)} (+${top.edge})</span><p>${top.reason}</p></div>` : `<div class="game-pick neutral"><strong>No value flag</strong><span>Model leans ${fav}; lines do not beat model price today.</span></div>`}
    <details class="factor-details" open><summary>Algo factor breakdown</summary><div class="factor-list">${factorBars(m.factors)}</div></details>
    ${game.eligible_for_official_picks !== false && (game.recommendations || []).length ? `<div class="rec-list"><h3>All model recommendations</h3>${game.recommendations.map((p) => pickCard({ ...p, league: game.league, league_name: game.league_name, matchup: `${away.name} @ ${home.name}`, matchup_obj: game.matchup, start_time: game.start_time })).join("")}</div>` : ""}
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
    <div class="game-head"><div><span class="league-pill">${game.league_name}</span><h3>${matchupLinks(game.league, away, home)}</h3><p class="game-meta">${formatTime(game.start_time)}</p></div>
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
    ${manifest ? `<p class="muted">${manifest.roster_profiles || manifest.players_built || 0} roster profiles · ${manifest.teams_built || 0} team sheets · Updated ${new Date(manifest.generated_at).toLocaleString()}</p>` : ""}</section>
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
  let data;
  try {
    data = await loadLeagueDb(league);
  } catch {
    appRoot.innerHTML = `<div class="panel empty-panel">League data for ${league.toUpperCase()} is not available yet. Try again after the next daily sync.</div>
      <a class="btn btn-secondary" href="#/teams">← All leagues</a>`;
    return;
  }
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
        (row) => {
          const power = powerRatingForAbbr(data.ratings, row.abbr);
          const record = `${row.wins ?? "—"}-${row.losses ?? "—"}`;
          const ratingLabel = power != null ? ` · PWR ${formatRating(power)}` : "";
          return `<a class="team-card panel" href="${teamHref(league, row.abbr)}">
          <strong>${row.name}</strong>
          <span class="muted">#${row.rank ?? "—"} · ${record}${ratingLabel}</span>
        </a>`;
        },
      )
      .join("")}</div>
  </section>`;
}

function playerStatusClass(status) {
  const value = (status || "").toLowerCase();
  if (!value || value === "active" || value === "starter" || value === "active roster") return "active";
  if (value.includes("inj") || value.includes("out") || value.includes("doubt")) return "injured";
  if (value.includes("suspend") || value.includes("inactive")) return "inactive";
  return "other";
}

function playerRatingTier(rating) {
  return ratingTierClass(rating);
}

function formatPlayerRating(rating) {
  const formatted = formatRating(rating, 0);
  return formatted === "—" ? null : formatted;
}

function ratingSourceLabel(source, layer) {
  const modelLayers = {
    basketball_matrix: "Matrix model",
    hockey_poisson: "Poisson xG",
    baseball_elo: "Elo",
    nfelo: "nfelo",
    soccer_elo: "Soccer Elo",
    power_ratings: "Power ratings",
  };
  if (source === "model" && layer) {
    return modelLayers[layer] || String(layer).replace(/_/g, " ");
  }
  return "Model";
}

function renderPlayerRatingBadge(rating, { large = false, source = null, layer = null } = {}) {
  const formatted = formatPlayerRating(rating);
  if (formatted == null) return "";
  const tier = playerRatingTier(rating);
  const cls = large ? "fm-player-rating fm-player-rating--lg" : "fm-player-rating";
  const tooltip = ratingSourceLabel(source, layer);
  return `<span class="${cls} fm-player-rating--${tier}" title="${tooltip}">${formatted}</span>`;
}

function htmlAttr(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

function playerSilhouetteClass(league) {
  const meta = leaguesForBrowse().find((lg) => lg.id === league);
  const category = (meta?.category || "").toLowerCase();
  if (category.includes("soccer")) return "fm-player-photo--silhouette-soccer";
  if (category.includes("hockey")) return "fm-player-photo--silhouette-hockey";
  if (category.includes("baseball")) return "fm-player-photo--silhouette-baseball";
  if (category.includes("football")) return "fm-player-photo--silhouette-football";
  return "fm-player-photo--silhouette";
}

function headshotCdnUrl(league, athleteId) {
  if (athleteId == null || athleteId === "") return null;
  const sportPaths = {
    nba: "nba",
    wnba: "wnba",
    cbb: "mens-college-basketball",
    nfl: "nfl",
    cfb: "college-football",
    nhl: "nhl",
    ncaah: "hockey",
    ncaawh: "hockey",
    mlb: "mlb",
    ncaabb: "baseball",
    mls: "soccer",
    epl: "soccer",
    laliga: "soccer",
    bundesliga: "soccer",
    seriea: "soccer",
    ligue1: "soccer",
    ucl: "soccer",
    worldcup: "soccer",
    fifa_friendlies: "soccer",
    concacaf_wcq: "soccer",
    concacaf_gold: "soccer",
    concacaf_nations: "soccer",
    uefa_euro: "soccer",
    uefa_nations: "soccer",
    copa_america: "soccer",
  };
  const sport = sportPaths[league];
  if (!sport) return null;
  return `https://a.espncdn.com/i/headshots/${sport}/players/full/${athleteId}.png`;
}

function renderPlayerPhoto(player, league, { large = false } = {}) {
  const name = player.name || player.displayName || "Player";
  const alt = htmlAttr(name);
  const sizeClass = large ? " fm-player-photo--lg" : "";
  const silhouetteCls = `fm-player-photo fm-player-photo--silhouette${sizeClass} ${playerSilhouetteClass(league)}`.trim();
  const headshot = player.headshot || headshotCdnUrl(league, player.id);
  if (headshot) {
    const src = htmlAttr(headshot);
    return `<span class="fm-player-photo-slot${sizeClass}">
      <img class="fm-player-photo${sizeClass}" src="${src}" alt="${alt}" loading="lazy"
        onerror="this.classList.add('fm-player-photo--hidden');this.nextElementSibling?.classList.remove('fm-player-photo--hidden')">
      <span class="${silhouetteCls} fm-player-photo--hidden" role="img" aria-label="${alt}"></span>
    </span>`;
  }
  return `<span class="${silhouetteCls}" role="img" aria-label="${alt}"></span>`;
}

function renderPlayerCard(player, league, deepIds) {
  const pid = String(player.id || "");
  const isDeep = !deepIds || deepIds.has(pid);
  const statusClass = playerStatusClass(player.status);
  const rating = player.algo_rating ?? player.player?.algo_rating;
  const ratingSource = player.rating_source ?? player.player?.rating_source;
  const ratingLayer = player.rating_layer ?? player.player?.rating_layer;
  return `<a class="fm-player-card${isDeep ? " fm-player-card--deep" : ""}" href="${playerHref(league, pid)}">
    <div class="fm-player-photo-wrap">
      ${renderPlayerPhoto(player, league)}
      ${player.jersey ? `<span class="fm-player-jersey">#${player.jersey}</span>` : ""}
      ${renderPlayerRatingBadge(rating, { source: ratingSource, layer: ratingLayer })}
    </div>
    <div class="fm-player-body">
      <strong class="fm-player-name">${player.name || "Player"}</strong>
      <span class="fm-player-meta">${player.position || "—"}${player.experience != null ? ` · ${player.experience} yr` : ""}</span>
      ${player.status ? `<span class="fm-player-status fm-player-status--${statusClass}">${player.status}</span>` : ""}
    </div>
  </a>`;
}

function renderRosterSection(teamDb, league) {
  const roster = teamDb?.roster || [];
  if (!roster.length) return "";
  const deepIds = new Set((teamDb?.players_built || []).map(String));
  const positions = [...new Set(roster.map((p) => p.position).filter(Boolean))].sort();
  const grouped = positions.length > 1 && positions.length <= 12;
  const cards = grouped
    ? positions
        .map((pos) => {
          const group = roster.filter((p) => p.position === pos);
          return `<div class="fm-roster-group"><h3 class="fm-roster-pos">${pos}</h3><div class="fm-roster-grid">${group.map((p) => renderPlayerCard(p, league, deepIds)).join("")}</div></div>`;
        })
        .join("")
    : `<div class="fm-roster-grid">${roster.map((p) => renderPlayerCard(p, league, deepIds)).join("")}</div>`;
  return `<section class="section panel fm-roster-section">
    <div class="section-head"><h2>Squad · ${roster.length} players</h2>
      <span class="muted">${deepIds.size ? `${deepIds.size} with full stats` : "Roster profiles"}</span></div>
    ${cards}
  </section>`;
}

function parseRecordFromText(text) {
  const match = String(text || "").match(/(\d+)\s*[-–]\s*(\d+)/);
  if (!match) return null;
  return { wins: Number(match[1]), losses: Number(match[2]) };
}

function parseRankFromStandingSummary(text) {
  const match = String(text || "").match(/(\d+)\s*(?:st|nd|rd|th)\b/i);
  return match ? Number(match[1]) : null;
}

function formatTeamRecord(standing, team) {
  if (standing?.wins != null) {
    const ties = standing.ties ? `-${standing.ties}` : "";
    return `${standing.wins}-${standing.losses}${ties}`;
  }
  const fromSummary = parseRecordFromText(team?.record_summary);
  if (fromSummary) return `${fromSummary.wins}-${fromSummary.losses}`;
  return "—";
}

async function enrichTeamStanding(league, abbr, teamDb) {
  if (!teamDb) return teamDb;
  const hasRank = teamDb.standing?.rank != null;
  const hasRecord = teamDb.standing?.wins != null;
  if (hasRank && hasRecord) return teamDb;

  try {
    const leagueData = await loadLeagueDb(league);
    const row = (leagueData.standings?.teams || []).find(
      (entry) => (entry.abbr || "").toLowerCase() === abbr,
    );
    if (row) {
      teamDb.standing = { ...row, ...(teamDb.standing || {}) };
      if (!teamDb.trends) teamDb.trends = {};
      if (!teamDb.trends.streak && row.streak) teamDb.trends.streak = row.streak;
      if (teamDb.trends.games_behind == null && row.games_behind != null) {
        teamDb.trends.games_behind = row.games_behind;
      }
      if (teamDb.trends.point_differential == null && row.point_differential != null) {
        teamDb.trends.point_differential = row.point_differential;
      }
      if (!teamDb.projection) teamDb.projection = {};
      if (teamDb.projection.rank == null) {
        teamDb.projection.rank = row.rank ?? row.playoff_seed;
      }
    }
  } catch {
    /* league standings optional */
  }
  return teamDb;
}

function resolveTeamHeroContext(teamDb, profile, recentGames) {
  const team = teamDb?.team || {};
  const standing = { ...(teamDb?.standing || {}) };
  const trends = { ...(teamDb?.trends || {}) };
  const projection = { ...(teamDb?.projection || {}) };

  if (standing.wins == null) {
    const fromSummary = parseRecordFromText(team.record_summary);
    if (fromSummary) Object.assign(standing, fromSummary);
  }
  if (standing.wins == null && profile?.season_stats?.wins != null) {
    standing.wins = profile.season_stats.wins;
    standing.losses = profile.season_stats.losses;
  }

  if (standing.rank == null) {
    standing.rank =
      projection.rank ??
      projection.playoff_seed ??
      standing.playoff_seed ??
      parseRankFromStandingSummary(team.standing_summary);
  }

  if (!trends.last_5 && recentGames?.length) {
    trends.last_5 = recentGames
      .slice(0, 5)
      .map((game) => game.result || "")
      .filter(Boolean)
      .join("");
  }

  return { team, standing, trends, projection };
}

function renderTeamHero(teamDb, leagueName, teamName, profile, recentGames) {
  const { team, standing, trends, projection } = resolveTeamHeroContext(
    teamDb,
    profile,
    recentGames,
  );
  const ratings = teamDb?.ratings || {};
  const avgRating = ratings.avg_player_rating;
  const ratingDiff =
    ratings.rating_diff ?? (avgRating != null ? Number(avgRating) - 50 : null);
  const powerValue = ratings.power?.power ?? projection.power_rating;
  return `<section class="fm-team-hero panel">
    <div class="fm-team-hero-main">
      ${team.logo ? `<img class="fm-team-logo" src="${team.logo}" alt="">` : ""}
      <div>
        <span class="league-pill">${leagueName}</span>
        <h1>${teamName}</h1>
        <p class="fm-team-record">${team.record_summary || ""}${team.standing_summary ? " · " + team.standing_summary : ""}</p>
        ${team.coach ? `<p class="muted">Head coach · ${team.coach}</p>` : ""}
        ${profile ? `<p class="muted">Season ${profile.season_year} · Through ${profile.cutoff_date}</p>` : ""}
      </div>
    </div>
    <div class="fm-team-hero-stats">
      <div class="fm-hero-stat"><span>Record</span><strong>${formatTeamRecord(standing, team)}</strong></div>
      <div class="fm-hero-stat"><span>Rank</span><strong>${standing.rank ?? "—"}</strong></div>
      <div class="fm-hero-stat"><span>Streak</span><strong>${trends.streak || "—"}</strong></div>
      <div class="fm-hero-stat"><span>Last 5</span><strong>${trends.last_5 || "—"}</strong></div>
      <div class="fm-hero-stat"><span>Power</span><strong>${formatRating(powerValue)}</strong></div>
      <div class="fm-hero-stat"><span>Avg player</span><strong class="fm-hero-rating fm-hero-rating--${playerRatingTier(avgRating)}">${formatRating(avgRating)}</strong></div>
      <div class="fm-hero-stat"><span>Diff</span><strong>${formatRatingDiff(ratingDiff)}</strong></div>
      <div class="fm-hero-stat"><span>Pace</span><strong>${formatRating(projection.projected_wins_pace)}</strong></div>
    </div>
  </section>`;
}

function teamLabelForAbbr(league, abbr) {
  const token = (abbr || "").toLowerCase();
  if (!token) return "";
  const team = teamsForLeague(league).find((row) => (row.abbr || "").toLowerCase() === token);
  return team?.label || team?.name || token.toUpperCase();
}

function gameLocationPrefix(location) {
  const loc = (location || "").toLowerCase();
  if (loc === "away" || loc === "@") return "@";
  if (loc === "home" || loc === "vs") return "vs";
  return loc || "vs";
}

function normalizeRecentGameScore(score) {
  if (Array.isArray(score)) return `${score[0]}–${score[1]}`;
  return score || "";
}

function parseAtVsOpponent(raw, location) {
  const text = (raw || "").trim();
  if (!text) return { opponent: "", opponent_abbr: null, location: location || "" };
  if (text.startsWith("@")) {
    const abbr = text.slice(1).trim();
    return { opponent: abbr, opponent_abbr: abbr.toLowerCase(), location: location || "away" };
  }
  if (text.toLowerCase().startsWith("vs")) {
    const abbr = text.replace(/^vs\.?\s*/i, "").trim();
    return { opponent: abbr, opponent_abbr: abbr.toLowerCase(), location: location || "home" };
  }
  if (/^[a-z]{2,5}$/i.test(text)) {
    return { opponent: text, opponent_abbr: text.toLowerCase(), location: location || "" };
  }
  return { opponent: text, opponent_abbr: null, location: location || "" };
}

function normalizeProfileRecentGame(g) {
  const parsed = parseAtVsOpponent(g.opponent, g.location);
  const oppAbbr =
    g.opponent_abbr ||
    parsed.opponent_abbr ||
    (g.opponent && /^[a-z]{2,5}$/i.test(g.opponent) ? g.opponent.toLowerCase() : null);
  return {
    result: g.result,
    date: g.date,
    opponent: g.opponent && g.opponent !== parsed.opponent ? g.opponent : parsed.opponent || g.opponent,
    opponent_abbr: oppAbbr,
    score: normalizeRecentGameScore(g.score),
    location: g.location || parsed.location || "",
  };
}

function mergeRecentGames(dbGames, profileGames) {
  const fromDb = dbGames || [];
  const fromProfile = (profileGames || []).map(normalizeProfileRecentGame);
  if (!fromDb.length) return fromProfile;
  if (!fromProfile.length) return fromDb;

  const profileByDate = new Map(fromProfile.map((g) => [g.date, g]));
  return fromDb.map((g) => {
    const missingOpp = !g.opponent || g.opponent === "?";
    if (!missingOpp && g.opponent_abbr) return g;
    const fallback = profileByDate.get(g.date);
    if (!fallback) return g;
    return {
      ...g,
      opponent: missingOpp ? fallback.opponent || g.opponent : g.opponent,
      opponent_abbr: g.opponent_abbr || fallback.opponent_abbr || null,
      location: g.location || fallback.location || "",
    };
  });
}

function recentGameOpponentLink(league, game) {
  const abbr = game.opponent_abbr || null;
  const fallbackName = abbr ? teamLabelForAbbr(league, abbr) : "";
  const name = game.opponent || fallbackName || "?";
  if (!abbr || name === "?") return name;
  const displayName =
    name.length > 5 || /\s/.test(name) ? name : teamLabelForAbbr(league, name) || name;
  return teamNameLink(league, abbr, displayName);
}

function renderRecentGameRow(league, game) {
  const parsed = parseAtVsOpponent(game.opponent, game.location);
  const abbr = game.opponent_abbr || parsed.opponent_abbr || null;
  const location = game.location || parsed.location || "";
  const result = game.result || "";
  const badgeClass = result === "W" ? "win" : result === "L" ? "loss" : "";
  const score = normalizeRecentGameScore(game.score);
  const prefix = gameLocationPrefix(location);
  const opponentMarkup = recentGameOpponentLink(league, {
    ...game,
    opponent_abbr: abbr,
    opponent: game.opponent || parsed.opponent,
  });
  return `<li class="recent-row">
    <span class="result-badge ${badgeClass}">${result || "—"}</span>
    <span class="recent-date">${formatGameDate(game.date)}</span>
    <span class="recent-matchup"><span class="recent-loc">${prefix}</span> ${opponentMarkup}</span>
    <span class="recent-score">${score}</span>
  </li>`;
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
  let teamDb = await loadTeamDb(league, normalizedAbbr).catch(() => null);
  const profile = await loadTeamProfile(league, normalizedAbbr);
  teamDb = await enrichTeamStanding(league, normalizedAbbr, teamDb);

  if (!teamDb && !profile) {
    appRoot.innerHTML = `<div class="panel empty-panel">Team sheet not available for ${normalizedAbbr.toUpperCase()} yet. Daily rebuilds populate all teams — try again after the next sync.</div>
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
  const recentGames = mergeRecentGames(teamDb?.recent_games, profile?.recent_games);

  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Leagues", href: "#/teams" },
    { label: leagueName, href: leagueHref(league) },
    { label: teamName },
  ])}${renderTeamHero(teamDb, leagueName, teamName, profile, recentGames)}
  <p class="fm-team-nav"><a href="${leagueHref(league)}">← ${leagueName} league</a></p>
  ${upcoming.length ? `<section class="section"><div class="section-head"><h2>Upcoming — betting context</h2></div>
    <div class="db-bet-grid">${upcoming.map((g) => renderBettingGameCard(g, league)).join("")}</div></section>` : ""}
  ${(teamDb?.injuries || []).length ? `<section class="section panel fm-injuries"><h2>Injuries & availability</h2><ul class="db-recent">${teamDb.injuries.map((p) => `<li><strong>${p.name}</strong> <span class="muted">(${p.position})</span> — <span class="fm-injury-status">${p.status}</span></li>`).join("")}</ul></section>` : ""}
  <div class="stat-grid fm-team-stats">
    <div class="stat-card panel"><span>Win %</span><strong>${profileStats?.win_pct != null ? `${profileStats.win_pct}%` : trends.win_percent != null ? `${(trends.win_percent * 100).toFixed(1)}%` : "—"}</strong></div>
    <div class="stat-card panel"><span>GB</span><strong>${trends.games_behind ?? teamDb?.standing?.games_behind ?? "—"}</strong></div>
    <div class="stat-card panel"><span>Pt diff</span><strong>${formatRating(trends.point_differential ?? teamDb?.standing?.point_differential, 0)}</strong></div>
    <div class="stat-card panel"><span>Roster</span><strong>${(teamDb?.roster_count ?? (teamDb?.roster || []).length) || "—"}</strong></div>
  </div>
  <div class="db-grid-two">
    <section class="section panel"><h2>Recent games</h2>
      ${recentGames.length ? `<ul class="db-recent recent-games">${recentGames
        .map((g) => renderRecentGameRow(league, g))
        .join("")}</ul>` : `<p class="muted">No recent game log.</p>`}
    </section>
    <section class="section panel"><h2>Season stats</h2>
      ${renderSeasonStatsSection(teamDb?.stats?.categories, profileStats)}
    </section>
  </div>
  ${renderRosterSection(teamDb, league)}
  <a class="btn btn-secondary" href="${leagueHref(league)}">← ${leagueName}</a>`;
}

async function viewPlayer(league, playerId) {
  state.sidebarLeague = league;
  renderSidebar(parseRoute());
  let player;
  try {
    player = await loadPlayerDb(league, playerId);
  } catch {
    appRoot.innerHTML = `<div class="panel empty-panel">Player profile not found for this id. Roster cards are rebuilt on each daily sync.</div>
      <a class="btn btn-secondary" href="${leagueHref(league)}">← ${league.toUpperCase()}</a>`;
    return;
  }
  const info = player.player || {};
  const playerRating = player.algo_rating ?? info.algo_rating;
  const ratingLayer = player.rating_layer ?? info.rating_layer;
  const ratingSource = player.rating_source ?? info.rating_source;
  const ratingTooltip = ratingSourceLabel(ratingSource, ratingLayer);
  const teamAbbr = (player.team_abbr || "").toLowerCase();
  const profileDepth = player.profile_depth || "full";
  const isRosterOnly = profileDepth === "roster";
  const isStatsOnly = profileDepth === "stats";
  const leagueName =
    leaguesForBrowse().find((lg) => lg.id === league)?.name || league.toUpperCase();
  const statBlocks = (player.season_stats || []).concat(player.overview_stats || []);
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Leagues", href: "#/teams" },
    { label: leagueName, href: leagueHref(league) },
    ...(teamAbbr ? [{ label: teamAbbr.toUpperCase(), href: teamHref(league, teamAbbr) }] : []),
    { label: info.name || "Player" },
  ])}<section class="fm-player-hero panel">
    <div class="fm-player-hero-photo">
      ${renderPlayerPhoto(info, league, { large: true })}
      ${info.jersey ? `<span class="fm-player-jersey fm-player-jersey--lg">#${info.jersey}</span>` : ""}
    </div>
    <div class="fm-player-hero-body">
      <span class="league-pill">${leagueName}</span>
      <h1>${info.name || "Player"}</h1>
      <p class="fm-player-role">${info.position || "—"} · ${info.status || "Active"}</p>
      ${renderPlayerRatingBadge(playerRating, { large: true, source: ratingSource, layer: ratingLayer }) ? `<div class="fm-player-rating-hero">${renderPlayerRatingBadge(playerRating, { large: true, source: ratingSource, layer: ratingLayer })}<span class="fm-player-rating-label">${ratingTooltip}</span></div>` : ""}
      <div class="fm-player-bio">
        <span>${info.height || "—"}</span>
        <span>${info.weight || "—"}</span>
        <span>Age ${info.age ?? "—"}</span>
        <span>${info.experience != null ? info.experience + " yrs exp" : "—"}</span>
      </div>
      ${teamAbbr ? `<a class="fm-player-team-link" href="${teamHref(league, teamAbbr)}">${teamAbbr.toUpperCase()} squad →</a>` : ""}
      ${isRosterOnly ? `<p class="muted fm-roster-only-note">Roster profile — season stats load on daily rebuild.</p>` : ""}
      ${isStatsOnly ? `<p class="muted fm-roster-only-note">Season stats profile — game log loads for featured slate players.</p>` : ""}
    </div>
  </section>
  <div class="db-grid-two">
    <section class="section panel"><h2>Season stats</h2>
      ${renderSeasonStatsSection(statBlocks, null, {
        emptyMessage: isRosterOnly
          ? "Season stats not loaded for this player yet."
          : "Stats unavailable.",
      })}
    </section>
    <section class="section panel"><h2>Recent games</h2>
      ${(player.game_log || []).length ? `<ul class="db-recent recent-games">${player.game_log.map((g) =>
        renderRecentGameRow(league, g)).join("")}</ul>` : `<p class="muted">${isRosterOnly || isStatsOnly ? "Game log loads with full player profile." : "No game log."}</p>`}
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
      <div class="bet-log">${bets.length ? bets.map((b) => `<article class="bet-row panel"><div class="bet-row-top"><div><strong>${b.team_abbr ? teamNameLink(b.league, b.team_abbr, b.team_name) : b.team_name}</strong><span class="league-pill">${b.league_name}</span>${statusBadge(b.status, b.units)}</div><span class="edge-tag">+${b.edge} edge</span></div>
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
  try {
    const payload = await fetchJson(
      USE_STATIC_API
        ? dbApi(`${league}/teams/${abbr}.json`)
        : api(`db/${league}/teams/${abbr}`),
    );
    state.dbCache[key] = payload;
    return payload;
  } catch {
    return null;
  }
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
      <strong>${matchupLinks(league, matchup.away, matchup.home)}</strong>
      <span class="muted">${formatTime(sheet.start_time)}</span>
    </div>
    <div class="odds-row compact">
      <div class="odds-chip"><span>${teamNameLink(league, matchup.away?.abbr, away)} ML</span><strong>${formatOdds(market.away_moneyline)}</strong><small>Model ${formatOdds(model.away_projection)}</small></div>
      <div class="odds-chip"><span>${teamNameLink(league, matchup.home?.abbr, home)} ML</span><strong>${formatOdds(market.home_moneyline)}</strong><small>Model ${formatOdds(model.home_projection)}</small></div>
      ${market.spread != null ? `<div class="odds-chip"><span>Spread</span><strong>${formatSpread(market.spread)}</strong></div>` : ""}
    </div>
    <div class="db-bet-model">
      <span>Unified model</span>
      <strong>${model.win_probability ?? "—"}%</strong>
      <small>Fav: ${model.favorite_side || "—"} · Blend ${model.blend_layers || "—"} layers</small>
      ${agreement.required ? `<small>Agreement: ${agreement.agreed ? "✓ all layers" : "✗ split"} (${(agreement.value_sides || []).join(", ") || "none"})</small>` : ""}
    </div>
    ${official && top ? `<div class="game-pick ${confClass(top.confidence)}"><strong>${top.strategy_label}</strong> ${pickTeamNameLink(top, league, matchup)} +${top.edge} edge</div>` : !official ? `<div class="game-pick neutral"><strong>Predictions only</strong> (soccer excluded from official picks)</div>` : `<div class="game-pick neutral"><strong>No official pick</strong> — model/market gap below threshold</div>`}
    <div class="db-bet-links">
      <a href="${teamHref(league, matchup.home?.abbr)}">${home}</a>
      <a href="${teamHref(league, matchup.away?.abbr)}">${away}</a>
      <a href="#/game/${sheet.event_id}">Full analysis →</a>
    </div>
  </article>`;
}

function powerRatingForAbbr(ratings, abbr) {
  const key = (abbr || "").toLowerCase();
  return ratings?.power?.[key]?.power ?? null;
}

function renderRatingsSummary(ratings) {
  const power = ratings?.power || {};
  const teams = Object.entries(power)
    .sort((a, b) => (b[1].power ?? 0) - (a[1].power ?? 0))
    .slice(0, 12);
  if (!teams.length) return `<p class="muted">Model ratings unavailable for this league snapshot.</p>`;
  return `<ul class="db-stat-list">${teams
    .map(([key, row]) => `<li><span>${row.name || key}</span><strong>${formatRating(row.power)}</strong></li>`)
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
      return `<tr><td>${row.rank ?? "—"}</td><td><a href="${teamHref(league, abbr)}"><strong>${row.name}</strong></a></td><td>${wins}-${losses}</td><td>${wp}</td><td>${row.games_behind ?? "—"}</td><td>${row.streak ?? "—"}</td><td>${formatRating(row.point_differential, 0)}</td></tr>`;
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
  let slate;
  try {
    slate = await fetchJson(
      USE_STATIC_API ? api("daily-slate.json") : api("daily/slate"),
    );
  } catch (err) {
    appRoot.innerHTML = `<div class="panel empty-panel error-panel">Could not load daily slate: ${err.message}. The site may still be rebuilding — refresh in a few minutes.</div>`;
    return;
  }
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

  try {
    await render();
  } catch (err) {
    appRoot.innerHTML = `<div class="panel empty-panel error-panel">${err.message}</div>`;
  }
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
