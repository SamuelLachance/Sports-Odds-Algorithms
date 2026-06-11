const leagueSelect = document.getElementById("league");
const awaySelect = document.getElementById("awayTeam");
const homeSelect = document.getElementById("homeTeam");
const seasonSelect = document.getElementById("seasonYear");
const form = document.getElementById("predictForm");
const formError = document.getElementById("formError");
const submitBtn = document.getElementById("submitBtn");
const resultsEmpty = document.getElementById("resultsEmpty");
const resultsContent = document.getElementById("resultsContent");
const themeToggle = document.getElementById("themeToggle");

const DEMO_DEFAULTS = {
  nba: {
    away: "portland-trail-blazers",
    home: "golden-state-warriors",
    date: "4-16-2017",
    season: "2017",
  },
  nhl: {
    away: "pittsburgh-penguins",
    home: "washington-capitals",
    date: "4-12-2017",
    season: "2017",
  },
  mlb: {
    away: "chicago-cubs",
    home: "cleveland-indians",
    date: "10-25-2016",
    season: "2016",
  },
};

function slugToLabel(slug) {
  return slug.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("soa-theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("soa-theme");
  setTheme(saved === "light" ? "light" : "dark");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string"
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item.msg || JSON.stringify(item)).join(", ")
        : "Request failed";
    throw new Error(message);
  }
  return payload;
}

function fillSelect(select, items, valueKey, labelKey) {
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item[valueKey];
    option.textContent = item[labelKey];
    select.appendChild(option);
  }
}

async function loadLeagues() {
  const leagues = await fetchJson("/api/leagues");
  fillSelect(leagueSelect, leagues, "id", "name");
}

async function loadLeagueData() {
  const league = leagueSelect.value;
  const [teams, seasons] = await Promise.all([
    fetchJson(`/api/leagues/${league}/teams`),
    fetchJson(`/api/leagues/${league}/seasons`),
  ]);

  fillSelect(awaySelect, teams, "slug", "label");
  fillSelect(homeSelect, teams, "slug", "label");
  fillSelect(seasonSelect, seasons.map((year) => ({ year })), "year", "year");

  const defaults = DEMO_DEFAULTS[league] || DEMO_DEFAULTS.nba;
  awaySelect.value = defaults.away;
  homeSelect.value = defaults.home;
  document.getElementById("date").value = defaults.date;
  seasonSelect.value = defaults.season;
}

function showError(message) {
  formError.hidden = false;
  formError.textContent = message;
}

function clearError() {
  formError.hidden = true;
  formError.textContent = "";
}

function updateRing(probability) {
  const ring = document.getElementById("probRing");
  const circumference = 326.7;
  const offset = circumference - (Math.min(probability, 100) / 100) * circumference;
  ring.style.strokeDashoffset = String(offset);
}

function renderFactors(factors) {
  const container = document.getElementById("factorList");
  container.innerHTML = "";

  for (const factor of factors) {
    const item = document.createElement("div");
    item.className = "factor-item";

    const width = Math.min(Math.abs(factor.value), 100);
    const direction =
      factor.favors === "away"
        ? "favors away"
        : factor.favors === "home"
          ? "favors home"
          : "neutral";

    item.innerHTML = `
      <div class="factor-head">
        <strong>${factor.label}</strong>
        <span>${factor.value > 0 ? "+" : ""}${factor.value.toFixed(2)}% · ${direction}</span>
      </div>
      <div class="factor-bar"><span style="width:${width}%"></span></div>
    `;
    container.appendChild(item);
  }
}

function renderResults(data) {
  resultsEmpty.hidden = true;
  resultsContent.hidden = false;

  document.getElementById("awayName").textContent = slugToLabel(data.matchup.away.slug);
  document.getElementById("homeName").textContent = slugToLabel(data.matchup.home.slug);
  document.getElementById("favoriteName").textContent = slugToLabel(data.prediction.favorite_team);
  document.getElementById("predictionMeta").textContent =
    `${data.algorithm} · ${data.date} · ${data.league.toUpperCase()} ${data.season_year}`;
  document.getElementById("winProbability").textContent = `${data.prediction.win_probability}%`;
  document.getElementById("favoriteOdds").textContent = data.prediction.american_odds.favorite;
  document.getElementById("underdogOdds").textContent = data.prediction.american_odds.underdog;
  document.getElementById("totalScore").textContent = `${data.prediction.total_score}%`;

  updateRing(data.prediction.win_probability);
  renderFactors(data.factors);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  submitBtn.disabled = true;
  submitBtn.textContent = "Calculating...";

  try {
    const payload = {
      league: leagueSelect.value,
      away_team: awaySelect.value,
      home_team: homeSelect.value,
      date: document.getElementById("date").value.trim(),
      season_year: seasonSelect.value,
      algorithm: document.getElementById("algorithm").value,
    };

    const result = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    renderResults(result);
  } catch (error) {
    showError(error.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Calculate odds";
  }
});

leagueSelect.addEventListener("change", () => {
  loadLeagueData().catch((error) => showError(error.message));
});

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  setTheme(next);
});

initTheme();

loadLeagues()
  .then(loadLeagueData)
  .catch((error) => showError(error.message));
