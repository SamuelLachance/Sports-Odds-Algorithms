const BASE_PATH = document.querySelector('meta[name="base-path"]')?.content || "";
const USE_STATIC_API =
  Boolean(BASE_PATH) || window.location.hostname.endsWith("github.io");

const refreshBtn = document.getElementById("refreshBtn");
const picksGrid = document.getElementById("picksGrid");
const picksEmpty = document.getElementById("picksEmpty");
const slateList = document.getElementById("slateList");
const leagueFilter = document.getElementById("leagueFilter");
const boardError = document.getElementById("boardError");
const themeToggle = document.getElementById("themeToggle");
const statGames = document.getElementById("statGames");
const statPicks = document.getElementById("statPicks");
const statLeagues = document.getElementById("statLeagues");
const updatedAt = document.getElementById("updatedAt");

let boardData = null;

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("soa-theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("soa-theme");
  setTheme(saved === "light" ? "light" : "dark");
}

function slateApiUrl() {
  if (USE_STATIC_API) {
    return `${BASE_PATH}/api/daily-slate.json`;
  }
  return "/api/daily/slate";
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : "Unable to load today's betting board.";
    throw new Error(message);
  }
  return payload;
}

function formatTime(iso) {
  if (!iso) return "TBD";
  const date = new Date(iso);
  return date.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatOdds(value) {
  if (value === null || value === undefined || value === 0) return "—";
  return value > 0 ? `+${value}` : `${value}`;
}

function confidenceClass(confidence) {
  return `confidence-${confidence || "low"}`;
}

function renderStats(data) {
  statGames.textContent = String(data.summary?.games_analyzed ?? 0);
  statPicks.textContent = String(data.summary?.recommended_bets ?? 0);
  statLeagues.textContent = (data.summary?.leagues || []).join(" · ").toUpperCase() || "—";
  const stamp = data.generated_at ? new Date(data.generated_at) : new Date();
  updatedAt.textContent = `Board updated ${stamp.toLocaleString()}`;
}

function renderPickCard(pick) {
  const article = document.createElement("article");
  article.className = `pick-card ${confidenceClass(pick.confidence)}`;
  article.innerHTML = `
    <div class="pick-top">
      <span class="league-pill">${pick.league_name || pick.league?.toUpperCase()}</span>
      <span class="strategy-pill">${pick.strategy_label || pick.strategy}</span>
    </div>
    <h3>${pick.team_name}</h3>
    <p class="pick-matchup">${pick.matchup}</p>
    <p class="pick-time">${formatTime(pick.start_time)}</p>
    <div class="pick-odds">
      <div><span>Market</span><strong>${formatOdds(pick.market_odds)}</strong></div>
      <div><span>Model</span><strong>${formatOdds(pick.model_projection)}</strong></div>
      <div><span>Edge</span><strong>+${pick.edge}</strong></div>
    </div>
    <p class="pick-reason">${pick.reason}</p>
  `;
  return article;
}

function renderPicks(data) {
  picksGrid.innerHTML = "";
  const picks = data.recommended_bets || [];

  if (!picks.length) {
    const empty = document.createElement("div");
    empty.className = "panel empty-panel";
    empty.textContent =
      "No clear value bets on today's board. The model didn't find lines beating the sportsbook on current moneylines.";
    picksGrid.appendChild(empty);
    return;
  }

  for (const pick of picks) {
    picksGrid.appendChild(renderPickCard(pick));
  }
}

function renderFactorBars(factors) {
  if (!factors?.length) return "";
  return factors
    .map((factor) => {
      const width = Math.min(Math.abs(factor.value), 100);
      const direction =
        factor.favors === "away"
          ? "away"
          : factor.favors === "home"
            ? "home"
            : "neutral";
      return `
        <div class="factor-item compact">
          <div class="factor-head">
            <strong>${factor.label}</strong>
            <span>${factor.value > 0 ? "+" : ""}${factor.value.toFixed(1)}% · ${direction}</span>
          </div>
          <div class="factor-bar"><span style="width:${width}%"></span></div>
        </div>
      `;
    })
    .join("");
}

function renderGameCard(game) {
  const article = document.createElement("article");
  article.className = "game-card panel";
  article.dataset.league = game.league;

  const away = game.matchup.away;
  const home = game.matchup.home;
  const model = game.model;
  const market = game.market;
  const topPick = game.top_pick;
  const favorite =
    model.favorite_side === "home" ? home.name : away.name;

  article.innerHTML = `
    <div class="game-head">
      <div>
        <span class="league-pill">${game.league_name}</span>
        <h3>${away.name} <span class="at">@</span> ${home.name}</h3>
        <p class="game-meta">${formatTime(game.start_time)} · ${game.status_detail || game.status}</p>
      </div>
      <div class="win-chip">
        <span>Model favorite</span>
        <strong>${favorite}</strong>
        <small>${model.win_probability}%</small>
      </div>
    </div>

    <div class="odds-row game-odds">
      <div class="odds-chip">
        <span>${away.name} ML</span>
        <strong>${formatOdds(market.away_moneyline)}</strong>
        <small>Model ${formatOdds(model.away_projection)}</small>
      </div>
      <div class="odds-chip">
        <span>${home.name} ML</span>
        <strong>${formatOdds(market.home_moneyline)}</strong>
        <small>Model ${formatOdds(model.home_projection)}</small>
      </div>
      <div class="odds-chip">
        <span>Spread / Total</span>
        <strong>${market.spread ?? "—"} / ${market.over_under ?? "—"}</strong>
        <small>${market.provider || "ESPN odds"}</small>
      </div>
    </div>

    ${
      topPick
        ? `<div class="game-pick ${confidenceClass(topPick.confidence)}">
            <strong>${topPick.strategy_label}</strong>
            <span>${topPick.team_name} · ${formatOdds(topPick.market_odds)} vs model ${formatOdds(topPick.model_projection)} (+${topPick.edge})</span>
            <p>${topPick.reason}</p>
          </div>`
        : `<div class="game-pick neutral">
            <strong>No value flag</strong>
            <span>Model leans ${favorite} but sportsbook lines are not beating the model price.</span>
          </div>`
    }

    <details class="factor-details">
      <summary>Factor breakdown</summary>
      <div class="factor-list">${renderFactorBars(model.factors)}</div>
    </details>
  `;
  return article;
}

function renderSlate(data) {
  slateList.innerHTML = "";
  const filter = leagueFilter.value;
  const games = (data.games || []).filter(
    (game) => filter === "all" || game.league === filter,
  );

  if (!games.length) {
    const empty = document.createElement("div");
    empty.className = "panel empty-panel";
    empty.textContent = "No upcoming games found for this filter.";
    slateList.appendChild(empty);
    return;
  }

  for (const game of games) {
    slateList.appendChild(renderGameCard(game));
  }
}

function showError(message) {
  boardError.hidden = false;
  boardError.textContent = message;
}

function clearError() {
  boardError.hidden = true;
  boardError.textContent = "";
}

async function loadBoard() {
  clearError();
  refreshBtn.disabled = true;
  refreshBtn.textContent = "Refreshing...";
  picksGrid.innerHTML = `<div class="panel empty-panel">Analyzing today's matchups...</div>`;
  slateList.innerHTML = `<div class="panel empty-panel">Loading games...</div>`;

  try {
    boardData = await fetchJson(slateApiUrl());
    renderStats(boardData);
    renderPicks(boardData);
    renderSlate(boardData);
  } catch (error) {
    showError(error.message);
    picksGrid.innerHTML = `<div class="panel empty-panel">Could not load recommendations.</div>`;
    slateList.innerHTML = `<div class="panel empty-panel">Could not load games.</div>`;
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = "Refresh today's board";
  }
}

refreshBtn.addEventListener("click", () => {
  loadBoard().catch((error) => showError(error.message));
});

leagueFilter.addEventListener("change", () => {
  if (boardData) renderSlate(boardData);
});

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  setTheme(next);
});

initTheme();
loadBoard().catch((error) => showError(error.message));
