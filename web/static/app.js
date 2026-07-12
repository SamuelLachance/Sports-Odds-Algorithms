const APP_BUILD_VERSION = "2026-07-12-wave2-ux";
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
const navToggle = document.getElementById("navToggle");

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function minHubacekConfidence(source) {
  const s = source?.summary || source || state.tracking || {};
  return s.min_win_confidence_pp ?? source?.min_win_confidence_pp ?? 20;
}

function hubacekPickRule(source) {
  const s = source?.summary || source || state.tracking || {};
  const gap = s.min_market_gap_pp ?? source?.min_market_gap_pp ?? 2;
  const ev = s.min_ev_pct ?? source?.min_ev_pct ?? 2;
  const conf = minHubacekConfidence(source);
  return `Hubáček spot: decorrelated model beats the book by ≥ ${gap} pp with ≥ ${ev}% EV and |p−50| ≥ ${conf} pp`;
}

/** Leagues with live models but official Hubáček tracking disabled. */
const PREDICTIONS_ONLY_LEAGUES = new Set(["nfl", "cfb", "cbb", "ncaabb"]);

function isPredictionsOnlyLeague(league) {
  return PREDICTIONS_ONLY_LEAGUES.has(String(league || "").toLowerCase());
}

function predictionsOnlyPill(league) {
  if (!isPredictionsOnlyLeague(league)) return "";
  return `<span class="edge-badge edge-badge--ref" title="Model live; official Hubáček picks disabled until backtests clear">Predictions only</span>`;
}

function officialEmptyPanel(hubacekRule) {
  return `<div class="panel empty-panel empty-panel--official">
    <strong>No official picks today</strong>
    <p class="muted">Nothing cleared the official Hubáček gate on tracked leagues (NBA, WNBA, NHL, MLB, calibrated soccer). NFL, CFB, and CBB stay predictions-only — see Model predictions below when model value spots exist.</p>
    <p class="muted empty-panel-rule">${escapeHtml(hubacekRule)}</p>
  </div>`;
}

function modelAnalysisEmptyPanel() {
  return `<div class="panel empty-panel empty-panel--ref">
    <strong>No model value spots today</strong>
    <p class="muted">Reference-only analysis (including NFL/CFB/CBB) found no value spots on today's slate. Full probabilities stay on each game page.</p>
  </div>`;
}

function gamesFilterEmptyPanel(league) {
  if (league && league !== "all" && isPredictionsOnlyLeague(league)) {
    const label = String(league).toUpperCase();
    return `<div class="panel empty-panel empty-panel--ref">
      <strong>No ${escapeHtml(label)} games on today's slate</strong>
      <p class="muted">${escapeHtml(label)} is predictions-only — GradientBoost v2 is live when games appear, but spots are not logged as official Hubáček picks.</p>
    </div>`;
  }
  return `<div class="panel empty-panel">No games for this filter.</div>`;
}

function officialBetTypeForLeague(league) {
  const id = (league || "").toLowerCase();
  if (["nba", "wnba", "cbb", "ncaabb", "nfl", "cfb"].includes(id)) return "spread";
  if (
    [
      "mls",
      "epl",
      "laliga",
      "bundesliga",
      "seriea",
      "ligue1",
      "ucl",
      "worldcup",
      "fifa_friendlies",
      "concacaf_wcq",
      "concacaf_gold",
      "concacaf_nations",
      "uefa_euro",
      "uefa_nations",
      "copa_america",
    ].includes(id)
  ) {
    return "soccer_1x2";
  }
  return "moneyline";
}

function officialBetType(game) {
  return game?.official_bet_type || officialBetTypeForLeague(game?.league);
}

function usesSpreadOfficialPicks(game) {
  return officialBetType(game) === "spread";
}

function officialMarketPhrase(game) {
  const t = officialBetType(game);
  if (t === "spread") return "spread";
  if (t === "soccer_1x2") return "1X2";
  return "moneyline";
}
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

const SPORT_ALGO_LABELS = {
  Algo_V1: "Algo V1 win probability",
  BasketballMatrix: "Matrix completion win probability",
  MLBRunCast: "MLB RunCast win probability",
  MLBGradientBoost:
    "MLB GradientBoost v2 (statsapi features + XGB/LR ensemble, isotonic-calibrated)",
  NHLGradientBoost:
    "NHL GradientBoost v2 (MoneyPuck xG + goalie GSAx + XGB/LR ensemble, isotonic-calibrated)",
  HockeyPuckCast: "PuckCast xG win probability",
  SoccerPathA: "Soccer Path A 1X2 (Elo + Pi + Dixon–Coles + XGB + context)",
  "SoccerGradientBoost v2":
    "Soccer GradientBoost v2 1X2 (25-season top-5 history, Elo + Dixon–Coles + SoT xG + XGB/LR ensemble, isotonic-calibrated)",
  "WNBAGradientBoost v2":
    "WNBA GradientBoost v2 (30-season history, Elo + four factors + pace + rest/travel, XGB/LR ensemble, isotonic-calibrated)",
  "NBAGradientBoost v2":
    "NBA GradientBoost v2 (30-season history, Elo + four factors + pace + rest/travel, XGB/LR ensemble, isotonic-calibrated)",
  Unified: "Unified model",
};

const SPORT_LAYER_LABELS = {
  Algo_V1: "Algo V1",
  BasketballMatrix: "BasketballMatrix",
  MLBRunCast: "MLB RunCast",
  MLBGradientBoost: "MLB GradientBoost v2",
  NHLGradientBoost: "NHL GradientBoost v2",
  HockeyPuckCast: "PuckCast",
  SharpBaseball: "SharpBaseball",
  SoccerPathA: "Soccer Path A",
  SharpSoccer: "Soccer Path A",
  "SoccerGradientBoost v2": "Soccer GradientBoost v2",
  "WNBAGradientBoost v2": "WNBA GradientBoost v2",
  "NBAGradientBoost v2": "NBA GradientBoost v2",
};

function primaryAlgoLabel(model) {
  if (!model) return "Model win probability";
  if (model.threeway) return "1X2 model probabilities";
  return SPORT_ALGO_LABELS[model.algorithm] || "Algo V2 win probability";
}

function primaryAlgoShort(model) {
  if (!model) return "Model";
  if (model.threeway) return "1X2";
  const short = {
    Algo_V1: "Algo V1",
    BasketballMatrix: "Matrix",
    MLBRunCast: "MLB RunCast",
    MLBGradientBoost: "MLB GB v2",
    NHLGradientBoost: "NHL GB v2",
    HockeyPuckCast: "PuckCast",
    SoccerPathA: "Soccer Path A",
    "SoccerGradientBoost v2": "Soccer GB v2",
    "WNBAGradientBoost v2": "WNBA GB v2",
    "NBAGradientBoost v2": "NBA GB v2",
    Unified: "Unified",
  };
  return short[model.algorithm] || model.algorithm || "Algo V2";
}

function sportLayerDisplayName(layer) {
  if (!layer?.algorithm) return null;
  return SPORT_LAYER_LABELS[layer.algorithm] || layer.algorithm;
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

function pickBestPriceHint(pick) {
  if (pick.best_available_odds == null) return "";
  const espn = pick.bet_type === "spread" ? pick.spread_odds ?? pick.market_odds : pick.market_odds;
  if (espn != null && Number(pick.best_available_odds) === Number(espn)) return "";
  const edge =
    pick.best_vs_espn_pp != null
      ? ` · +${pick.best_vs_espn_pp}pp vs ESPN`
      : pick.n_books
        ? ` · ${pick.n_books} books`
        : "";
  return `<div><span>Best book</span><strong>${formatOdds(pick.best_available_odds)}${edge}</strong></div>`;
}

/** Thin-sample international / tournament leagues (mirrors context_signals). */
const SPARSE_SAMPLE_LEAGUES = new Set([
  "worldcup",
  "fifa_friendlies",
  "copa_america",
  "concacaf_wcq",
  "concacaf_gold",
  "concacaf_nations",
  "uefa_euro",
  "uefa_nations",
]);

function isSparseSampleLeague(league) {
  const key = String(league || "").toLowerCase();
  if (!key) return false;
  return SPARSE_SAMPLE_LEAGUES.has(key) || key.startsWith("concacaf_");
}

function formatSignedPct(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  const text = Math.abs(n) < 10 && digits > 0 ? n.toFixed(digits) : String(Math.round(n * 10) / 10);
  return n > 0 ? `+${text}` : text;
}

function formatUnits(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  const text = Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/\.?0+$/, "");
  return `${n > 0 ? "+" : ""}${text}u`;
}

/** Quarter-Kelly stake (0.25–3u) from pick kelly_pct / stake_units when present. */
function stakeUnitsFromPick(pick) {
  if (pick?.stake_units != null && !Number.isNaN(Number(pick.stake_units))) {
    return Number(pick.stake_units);
  }
  const kellyPct = pick?.kelly_pct;
  if (kellyPct == null || Number(kellyPct) <= 0) return null;
  const kellyFrac = Number(kellyPct) / 100;
  let base = (kellyFrac / 4) * 100;
  const ev = Number(pick?.ev_pct);
  if (!Number.isNaN(ev)) {
    if (ev > 40) base *= 0.85;
    else if (ev > 25) base *= 0.92;
  }
  return Math.round(Math.min(3, Math.max(0.25, base)) * 100) / 100;
}

function openingSteamMeta(model) {
  if (!model || typeof model !== "object") return null;
  if (model.opening_steam?.steam_signal) return model.opening_steam;
  for (const key of [
    "soccer_pred",
    "basketball_pred",
    "baseball_pred",
    "hockey_pred",
    "football_pred",
  ]) {
    const steam = model[key]?.opening_steam;
    if (steam?.steam_signal) return steam;
  }
  return null;
}

function edgeQualityBadges(pick, game) {
  const badges = [];
  const league = pick?.league || game?.league;
  const model = game?.model || {};
  const steam = openingSteamMeta(model);
  const sparseFactor = model?.db_rating?.sparse_schedule_factor;
  const gamesProxy =
    pick?.games_played_proxy ?? model?.context_signals?.games_played_proxy;

  if (
    steam ||
    pick?.opening_edge ||
    (pick?.best_vs_espn_pp != null && Number(pick.best_vs_espn_pp) > 0)
  ) {
    badges.push({
      key: "opening",
      label: "Opening-edge",
      title: "Early-line or shopped price edge — bet early before the close",
    });
  }

  const isOfficial =
    pick?.tracked === true ||
    pick?.strategy === "hubacek" ||
    String(pick?.strategy_label || "")
      .toLowerCase()
      .includes("hubáček") ||
    String(pick?.strategy_label || "")
      .toLowerCase()
      .includes("hubacek");
  if (isOfficial && pick?.tracked !== false) {
    badges.push({
      key: "close",
      label: "Close-floor",
      title: "Clears Hubáček floors sized to beat the closing line",
    });
  }

  if (
    isSparseSampleLeague(league) ||
    (gamesProxy != null && Number(gamesProxy) < 8) ||
    (sparseFactor != null && Number(sparseFactor) > 0.35)
  ) {
    badges.push({
      key: "sparse",
      label: "Sparse-sample",
      title: "Thin sample — EV is capped; treat size and confidence cautiously",
    });
  }

  return badges;
}

function renderEdgeBadges(pick, game) {
  const badges = edgeQualityBadges(pick, game);
  if (!badges.length) return "";
  return `<div class="edge-badge-row">${badges
    .map(
      (b) =>
        `<span class="edge-badge edge-badge--${b.key}" title="${escapeHtml(b.title)}">${escapeHtml(b.label)}</span>`,
    )
    .join("")}</div>`;
}

function sparseLeaguePill(league) {
  if (!isSparseSampleLeague(league)) return "";
  return `<span class="edge-badge edge-badge--sparse" title="International / tournament slate — thin sample">Sparse sample</span>`;
}

function disclaimerBar(extra = "") {
  return `<aside class="disclaimer-bar" role="note">
    <strong>Research decision-support</strong>
    <span>Bet early, shop lines, size to units.${extra ? ` ${extra}` : ""}</span>
  </aside>`;
}

function contextCallout(model) {
  const adj = model?.context_adjustment_pp;
  const signals = model?.context_signals;
  if (adj == null && !signals) return "";
  const total = adj ?? signals?.total_pp;
  if (total == null && !signals) return "";
  const parts = [];
  if (signals?.flb_pp != null && Math.abs(Number(signals.flb_pp)) >= 0.05) {
    parts.push(`FLB ${formatSignedPct(signals.flb_pp)} pp`);
  }
  if (signals?.steam_pp != null && Math.abs(Number(signals.steam_pp)) >= 0.05) {
    parts.push(`Steam ${formatSignedPct(signals.steam_pp)} pp`);
  }
  if (signals?.news_pp != null && Math.abs(Number(signals.news_pp)) >= 0.05) {
    parts.push(`News ${formatSignedPct(signals.news_pp)} pp`);
  }
  const detail = parts.length ? parts.join(" · ") : "Context layer applied";
  const signed = total != null ? formatSignedPct(total) : "—";
  return `<div class="context-panel">
    <div class="context-panel-head">
      <span class="panel-kicker">Context layer</span>
      <strong>${signed} pp home</strong>
    </div>
    <p class="muted">${escapeHtml(detail)}</p>
  </div>`;
}

function lineShoppingPanel(market, pick) {
  const mk = market || {};
  const nBooks = mk.n_books ?? pick?.n_books;
  const bestPick = pick?.best_available_odds;
  const hasBest =
    bestPick != null ||
    mk.best_home_ml != null ||
    mk.best_away_ml != null ||
    mk.best_home_spread != null ||
    mk.best_away_spread != null;
  if (!nBooks && !hasBest) return "";

  const rows = [];
  if (bestPick != null) {
    const vs =
      pick?.best_vs_espn_pp != null
        ? ` · +${pick.best_vs_espn_pp}pp vs ESPN`
        : "";
    rows.push(
      `<div><span>Best for pick</span><strong>${formatOdds(bestPick)}${vs}</strong></div>`,
    );
  }
  if (mk.best_away_ml != null) {
    rows.push(
      `<div><span>Best away ML</span><strong>${formatOdds(mk.best_away_ml)}</strong></div>`,
    );
  }
  if (mk.best_home_ml != null) {
    rows.push(
      `<div><span>Best home ML</span><strong>${formatOdds(mk.best_home_ml)}</strong></div>`,
    );
  }
  if (mk.best_away_spread != null) {
    rows.push(
      `<div><span>Best away spread</span><strong>${formatOdds(mk.best_away_spread)}</strong></div>`,
    );
  }
  if (mk.best_home_spread != null) {
    rows.push(
      `<div><span>Best home spread</span><strong>${formatOdds(mk.best_home_spread)}</strong></div>`,
    );
  }
  if (!rows.length && nBooks) {
    rows.push(`<div><span>Books scanned</span><strong>${nBooks}</strong></div>`);
  }

  return `<div class="line-shop-panel">
    <div class="line-shop-head">
      <span class="panel-kicker">Line shopping</span>
      ${nBooks ? `<span class="muted">${nBooks} books</span>` : ""}
    </div>
    <div class="line-shop-grid">${rows.join("")}</div>
    <p class="muted line-shop-caption">Shop the best available price — ESPN quote is a reference, not always the best book.</p>
  </div>`;
}

function hubacekThreeTrackPanel(game, pick) {
  const m = game?.model || {};
  const top = pick || game?.top_pick;
  if (!m && !top) return "";

  const predictionsOnly = isPredictionsOnlyGame(game);
  const displayProb = m.threeway
    ? null
    : m.win_probability;
  const displayLabel = m.threeway
    ? `1X2 ${m.home_win_probability ?? "—"} / ${m.draw_probability ?? "—"} / ${m.away_win_probability ?? "—"}`
    : displayProb != null
      ? `${displayProb}%`
      : "—";

  const pickProb = top?.win_probability;
  const evProb =
    top?.base_win_probability ??
    m.pre_decorrelation_home_win_probability ??
    m.pre_context_home_win_probability ??
    null;

  const kicker = predictionsOnly ? "Model probability tracks" : "Hubáček 3-track";
  const gateLabel = predictionsOnly ? "Decorrelated" : "Pick gate";
  const gateHint = predictionsOnly
    ? "Shown for research — not an official pick gate"
    : "Decorrelated prob for gap / confidence";
  const caption = predictionsOnly
    ? `<p class="muted three-track-caption">Predictions-only league — these tracks inform the board, but do not create official Hubáček bets.</p>`
    : "";

  return `<div class="three-track-panel${predictionsOnly ? " three-track-panel--ref" : ""}">
    <div class="three-track-head">
      <span class="panel-kicker">${kicker}</span>
      <span class="muted">Display · ${gateLabel} · Honest EV</span>
    </div>
    <div class="three-track-grid">
      <div>
        <span>Display</span>
        <strong>${displayLabel}</strong>
        <small>Blended model shown on the board</small>
      </div>
      <div>
        <span>${gateLabel}</span>
        <strong>${pickProb != null ? `${pickProb}%` : "—"}</strong>
        <small>${gateHint}</small>
      </div>
      <div>
        <span>Honest EV</span>
        <strong>${evProb != null ? `${evProb}%` : pickProb != null ? `${pickProb}%` : "—"}</strong>
        <small>Calibrated pre-decorrelation for EV &amp; Kelly</small>
      </div>
    </div>
    ${caption}
    <a class="text-link three-track-link" href="#/methodology">How this works →</a>
  </div>`;
}

function fairOddsBetStrip(game, pick) {
  const top = pick || game?.top_pick;
  if (!top) return "";
  const stake = stakeUnitsFromPick(top);
  const fair = pickModelLabel(top);
  const best =
    top.best_available_odds != null
      ? formatOdds(top.best_available_odds)
      : null;
  const market = pickMarketLabel(top);
  return `<div class="fair-bet-strip">
    <div><span>Fair / model</span><strong>${fair}</strong></div>
    <div><span>Market</span><strong>${market}</strong></div>
    ${best ? `<div><span>Best book</span><strong>${best}</strong></div>` : ""}
    ${stake != null ? `<div><span>Suggested stake</span><strong>${stake}u</strong></div>` : ""}
    ${top.ev_pct != null ? `<div><span>Honest EV</span><strong>+${top.ev_pct}%</strong></div>` : ""}
  </div>`;
}

function clvCaption() {
  return `<p class="clv-caption muted">CLV (closing-line value): positive = you beat the close on implied probability. Primary metric is implied-prob CLV; payout CLV is shown when available as a secondary check. <a class="text-link" href="#/methodology">How this works →</a></p>`;
}

function formatClvBlock(bet) {
  if (bet?.clv_pct == null && bet?.clv_payout_pct == null) return "";
  const implied = bet.clv_pct;
  const payout = bet.clv_payout_pct;
  const impliedCls =
    implied == null ? "" : Number(implied) >= 0 ? "clv-positive" : "clv-negative";
  const payoutCls =
    payout == null ? "" : Number(payout) >= 0 ? "clv-positive" : "clv-negative";
  return `<div class="clv-metrics">
    ${implied != null ? `<span class="${impliedCls}">CLV ${formatSignedPct(implied)}%</span>` : ""}
    ${payout != null ? `<span class="clv-secondary ${payoutCls}">Payout CLV ${formatSignedPct(payout)}%</span>` : ""}
  </div>`;
}

function pickBetTypeLabel(pick) {
  if (pick?.bet_type === "spread") return "Spread";
  if (pick?.side === "draw" || pick?.bet_type === "soccer_1x2") return "1X2";
  return "Moneyline";
}

function pickSideLabel(pick) {
  if (!pick) return "";
  if (pick.side === "draw") return "Draw";
  if (pick.side === "home") return "Home";
  if (pick.side === "away") return "Away";
  return pick.side || "";
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
  if (!lg || !token || !name) return escapeHtml(name || "");
  return `<a href="${teamHref(lg, token)}" class="${className}">${escapeHtml(name)}</a>`;
}

function pickTeamNameLink(pick, league, matchup) {
  const abbr = pickTeamAbbr(pick, matchup);
  return abbr ? teamNameLink(league, abbr, pick.team_name) : escapeHtml(pick.team_name || "");
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
    { href: "#/tracking", label: "CLV tracking", active: path === "tracking" },
    { href: "#/methodology", label: "Methodology", active: path === "methodology" },
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

function pickCard(pick, extra = "", game = null) {
  const gameHref = pickGameHref(pick);
  const tag = gameHref ? "a" : "article";
  const linkAttrs = gameHref
    ? ` href="${gameHref}" class="pick-card pick-card-link ${confClass(pick.confidence)}"`
    : ` class="pick-card ${confClass(pick.confidence)}"`;
  const trackedPill =
    pick.tracked === false
      ? `<span class="strategy-pill strategy-pill--muted">Not tracked</span>`
      : pick.tracked === true
        ? `<span class="strategy-pill strategy-pill--official">Official</span>`
        : "";
  const stake = stakeUnitsFromPick(pick);
  const gap = pick.model_market_gap_pp;
  const confLabel = pick.confidence
    ? String(pick.confidence).charAt(0).toUpperCase() + String(pick.confidence).slice(1)
    : "—";
  return `<${tag}${linkAttrs}>
    <div class="pick-top">
      <span class="league-pill">${escapeHtml(pick.league_name || pick.league || "")}</span>
      <span class="strategy-pill">${escapeHtml(pick.strategy_label || pick.strategy || "")}</span>
      ${trackedPill}
      ${predictionsOnlyPill(pick.league)}
      ${sparseLeaguePill(pick.league)}
    </div>
    ${renderEdgeBadges(pick, game)}
    <div class="pick-side-row">
      <span class="pick-side-tag">${escapeHtml(pickSideLabel(pick))} · ${escapeHtml(pickBetTypeLabel(pick))}</span>
      ${stake != null ? `<span class="pick-stake">${stake}u</span>` : ""}
    </div>
    <h3>${escapeHtml(pick.team_name || "")}</h3>
    <p class="pick-matchup">${escapeHtml(pick.matchup || extra || "")}</p>
    <p class="pick-time">${formatTime(pick.start_time)}</p>
    <div class="pick-odds">
      <div><span>${pick.bet_type === "spread" ? "Spread" : "Market"}</span><strong>${pickMarketLabel(pick)}</strong></div>
      ${pickBestPriceHint(pick)}
      <div><span>Model</span><strong>${pickModelLabel(pick)}</strong></div>
      ${gap != null ? `<div><span>Market gap</span><strong>${formatSignedPct(gap)} pp</strong></div>` : pick.edge != null ? `<div><span>Edge</span><strong>+${pick.edge}</strong></div>` : ""}
      ${pick.ev_pct != null ? `<div><span>Honest EV</span><strong>+${pick.ev_pct}%</strong></div>` : ""}
      ${pick.expected_units != null ? `<div><span>CLV / EV units</span><strong>${formatUnits(pick.expected_units)}</strong></div>` : ""}
      <div><span>Confidence</span><strong>${escapeHtml(confLabel)}</strong></div>
      ${pick.kelly_pct != null ? `<div><span>Kelly</span><strong>${pick.kelly_pct}%</strong></div>` : ""}
    </div>
    <p class="pick-reason">${escapeHtml(pick.reason || "")}</p>
    ${gameHref ? `<span class="pick-open-hint">Open game prediction →</span>` : ""}
  </${tag}>`;
}

function algoBreakdown(m, game) {
  if (!m) return "";
  const legacy = m.legacy;
  const power = m.power;
  const basketball = m.basketball_pred;
  const baseball = m.baseball_pred;
  const hockey = m.hockey_pred;
  const football = m.football_pred;
  const soccer = m.soccer_pred;
  const dbRating = m.db_rating;
  const ensemble = m.ensemble_ml;
  const singleModel = (m.blend_layers || 0) === 1;
  if (
    !legacy &&
    !power &&
    !basketball &&
    !baseball &&
    !hockey &&
    !football &&
    !soccer &&
    !dbRating &&
    !ensemble
  ) {
    return "";
  }
  const parts = [];
  if (m.blend_mode === "ensemble_ml" || ensemble) {
    parts.push("EnsembleML");
  }
  const layerTag =
    m.blend_layers >= 4
      ? "4-layer"
      : m.blend_layers === 3
        ? "3-layer"
        : m.blend_layers === 2
          ? "2-layer"
          : "";
  if (layerTag && !singleModel) {
    parts.push(layerTag);
  }
  if (!singleModel && legacy) {
    const legacyHome =
      legacy.home_win_probability ??
      (legacy.favorite_side === "home"
        ? legacy.win_probability
        : 100 - legacy.win_probability);
    parts.push(`Legacy V2: ${legacyHome}% home`);
  }
  if (singleModel && legacy?.algorithm === "Algo_V1") {
    const legacyHome =
      legacy.home_win_probability ??
      (legacy.favorite_side === "home"
        ? legacy.win_probability
        : 100 - legacy.win_probability);
    parts.push(`Algo V1: ${legacyHome}% home · total ${formatRating(legacy.total_score, 2)}`);
  }
  if (!singleModel && power) {
    parts.push(
      `Power: ${power.home_win_probability}% home (${formatRating(power.home_power)} vs ${formatRating(power.away_power)})`,
    );
  }
  if (basketball) {
    const name = sportLayerDisplayName(basketball) || "BasketballMatrix";
    const scores =
      basketball.predicted_home_score != null && basketball.predicted_away_score != null
        ? ` · ${basketball.predicted_away_score}-${basketball.predicted_home_score}`
        : "";
    const orPace =
      basketball.predicted_home_offensive_rating != null &&
      basketball.predicted_away_offensive_rating != null
        ? ` · OR ${formatRating(basketball.predicted_home_offensive_rating)}/${formatRating(basketball.predicted_away_offensive_rating)}`
        : "";
    const pace =
      basketball.predicted_pace != null
        ? ` pace ${formatRating(basketball.predicted_pace)}`
        : "";
    const elo =
      basketball.home_elo != null && basketball.away_elo != null
        ? ` · Elo ${formatRating(basketball.away_elo, 0)}-${formatRating(basketball.home_elo, 0)}`
        : "";
    const netRtg =
      basketball.home_net_rtg != null && basketball.away_net_rtg != null
        ? ` · net rtg ${formatRatingDiff(basketball.away_net_rtg)}/${formatRatingDiff(basketball.home_net_rtg)}`
        : "";
    const teamPace =
      basketball.predicted_pace == null &&
      basketball.home_pace != null &&
      basketball.away_pace != null
        ? ` · pace ${formatRating((Number(basketball.home_pace) + Number(basketball.away_pace)) / 2)}`
        : "";
    const b2b =
      basketball.home_b2b || basketball.away_b2b
        ? ` · B2B ${[basketball.away_b2b ? "away" : null, basketball.home_b2b ? "home" : null].filter(Boolean).join("+")}`
        : "";
    parts.push(
      `${name}: ${basketball.home_win_probability}% home${scores}${orPace}${pace}${elo}${netRtg}${teamPace}${b2b}`,
    );
  }
  if (baseball) {
    const name = sportLayerDisplayName(baseball) || "SharpBaseball";
    const runs =
      baseball.predicted_home_runs != null && baseball.predicted_away_runs != null
        ? ` · ${baseball.predicted_away_runs}-${baseball.predicted_home_runs} runs`
        : baseball.elo_exp != null
          ? ` · Elo ${baseball.elo_exp}%`
          : "";
    const pitchers =
      baseball.away_probable_pitcher || baseball.home_probable_pitcher
        ? ` · SP ${baseball.away_probable_pitcher || "TBD"} vs ${baseball.home_probable_pitcher || "TBD"}`
        : "";
    const fip =
      baseball.away_sp_fip != null && baseball.home_sp_fip != null
        ? ` (FIP ${formatRating(baseball.away_sp_fip, 2)}/${formatRating(baseball.home_sp_fip, 2)})`
        : "";
    const elo =
      baseball.home_elo != null && baseball.away_elo != null
        ? ` · Elo ${formatRating(baseball.away_elo, 0)}-${formatRating(baseball.home_elo, 0)}`
        : "";
    const park =
      baseball.park_factor != null && Math.abs(baseball.park_factor - 1) >= 0.03
        ? ` · park ${formatRating(baseball.park_factor, 2)}x`
        : "";
    const decorr = baseball.market_decorrelated ? " · pick decorr" : "";
    parts.push(
      `${name}: ${baseball.home_win_probability}% home${runs}${pitchers}${fip}${elo}${park}${decorr}`,
    );
  }
  if (hockey && m.algorithm !== "Algo_V1") {
    const name = sportLayerDisplayName(hockey) || "Hockey";
    const xg =
      hockey.expected_home_goals != null
        ? ` · proj goals ${hockey.expected_away_goals}-${hockey.expected_home_goals}`
        : "";
    const goalies =
      hockey.away_goalie || hockey.home_goalie
        ? ` · G: ${hockey.away_goalie || "TBD"}${hockey.away_goalie_confirmed ? " ✓" : ""} vs ${hockey.home_goalie || "TBD"}${hockey.home_goalie_confirmed ? " ✓" : ""}`
        : "";
    const gsax =
      hockey.home_goalie_gsax100 != null && hockey.away_goalie_gsax100 != null
        ? ` · GSAx/100 ${formatRatingDiff(hockey.away_goalie_gsax100, 2)} vs ${formatRatingDiff(hockey.home_goalie_gsax100, 2)}`
        : "";
    const elo =
      hockey.home_elo != null && hockey.away_elo != null
        ? ` · Elo ${formatRating(hockey.away_elo, 0)} vs ${formatRating(hockey.home_elo, 0)}`
        : "";
    const xgRates =
      hockey.home_xgf_pg != null && hockey.away_xgf_pg != null
        ? ` · xGF/gm ${formatRating(hockey.away_xgf_pg, 2)} vs ${formatRating(hockey.home_xgf_pg, 2)}`
        : "";
    const rest = [];
    if (hockey.away_b2b) rest.push("away B2B");
    if (hockey.home_b2b) rest.push("home B2B");
    const restTag = rest.length ? ` · ${rest.join(", ")}` : "";
    const decorr = hockey.market_decorrelated ? " · pick decorr" : "";
    parts.push(
      `${name}: ${hockey.home_win_probability}% home${xg}${goalies}${gsax}${elo}${xgRates}${restTag}${decorr}`,
    );
  }
  if (!singleModel && football) {
    parts.push(`nfelo: ${football.home_win_probability}% home`);
  }
  if (soccer) {
    const name = sportLayerDisplayName(soccer) || "Soccer Path A";
    const xg =
      soccer.expected_home_goals != null
        ? ` · xG ${soccer.expected_away_goals}-${soccer.expected_home_goals}`
        : "";
    const elo =
      soccer.elo_home != null && soccer.elo_away != null
        ? ` · Elo ${formatRating(soccer.elo_away, 0)} vs ${formatRating(soccer.elo_home, 0)}`
        : "";
    const promoted = [];
    if (soccer.away_promoted) promoted.push("away promoted");
    if (soccer.home_promoted) promoted.push("home promoted");
    const promotedTag = promoted.length ? ` · ${promoted.join(", ")}` : "";
    const restParts = [];
    if (soccer.home_rest_days != null && soccer.away_rest_days != null) {
      const restGap = Number(soccer.home_rest_days) - Number(soccer.away_rest_days);
      if (Math.abs(restGap) >= 3) {
        restParts.push(
          `rest ${formatRating(soccer.away_rest_days, 0)}d vs ${formatRating(soccer.home_rest_days, 0)}d`,
        );
      }
    }
    const restTag = restParts.length ? ` · ${restParts.join(", ")}` : "";
    const enrich = [];
    if (soccer.market_calibrated) enrich.push("market-calibrated");
    if (soccer.market_decorrelated) enrich.push("pick decorr");
    if (soccer.opening_steam?.steam_signal) enrich.push("opening steam");
    if (m.context_adjusted) enrich.push("ESPN context");
    const suffix = enrich.length ? ` · ${enrich.join(", ")}` : "";
    parts.push(
      `${name}: ${soccer.home_win_probability}% / ${soccer.draw_probability}% / ${soccer.away_win_probability}%${xg}${elo}${promotedTag}${restTag}${suffix}`,
    );
  }
  if (dbRating) {
    const sparse =
      dbRating.sparse_schedule_factor > 0.35 ? " · sparse boost" : "";
    parts.push(
      `DB Ratings (${dbRating.source_home}): ${formatRating(dbRating.home_rating)} vs ${formatRating(dbRating.away_rating)} → ${dbRating.home_win_probability}% home${sparse}`,
    );
  }
  if (ensemble) {
    const margin =
      ensemble.predicted_home_margin != null
        ? ` margin ${ensemble.predicted_home_margin}`
        : "";
    parts.push(`EnsembleML: ${ensemble.home_win_probability}% home${margin}`);
  }
  const context = m.soccer_context;
  if (context?.factors?.length) {
    const ctxParts = context.factors.map(
      (f) => `${f.label}${f.detail ? ` (${f.detail})` : ""}`,
    );
    parts.push(`Context: ${ctxParts.join("; ")}`);
  }
  if (m.blend_note && !singleModel) {
    parts.push(m.blend_note);
  }
  if (m.home_spread_margin != null && game && usesSpreadOfficialPicks(game)) {
    parts.push(`Spread margin: home ${formatSpread(m.home_spread_margin)}`);
  }
  return parts.length
    ? `<div class="algo-blend panel-sub"><span class="blend-label">${singleModel ? "Sport model" : "Model blend"}</span><small>${parts.join(" · ")}</small></div>`
    : "";
}

function factorsSectionTitle(m) {
  if (m?.algorithm === "Algo_V1") return "Algo V1 factor breakdown";
  return (m?.blend_layers || 0) === 1 ? "Model inputs" : "Algo factor breakdown";
}

function isPredictionsOnlyGame(game) {
  return game?.eligible_for_official_picks === false;
}

function gameValuePickBlock(game, top, isSoccer) {
  const predictionsOnly = isPredictionsOnlyGame(game);
  const stake = top ? stakeUnitsFromPick(top) : null;
  const stakeBit = stake != null ? ` · ${stake}u` : "";
  if (predictionsOnly) {
    if (top) {
      return `<div class="game-pick neutral">
        <strong>Model value analysis</strong>
        ${renderEdgeBadges(top, game)}
        <span>${pickTeamNameLink(top, game.league, game.matchup)} · ${pickSideLabel(top)} ${pickBetTypeLabel(top)} · ${pickMarketLabel(top)} vs model ${pickModelLabel(top)} (+${top.ev_pct != null ? top.ev_pct + "% EV" : ""}${top.edge != null ? ", +" + top.edge + " edge" : ""})${stakeBit} — not an official Hubáček pick</span>
        <p>${escapeHtml(top.reason || "")}</p>
        ${fairOddsBetStrip(game, top)}
      </div>`;
    }
    return `<div class="game-pick neutral"><strong>Predictions only</strong><span>${isSoccer ? "Full 1X2 model probabilities and fair prices are shown below. Value spots appear under Model predictions on the picks page but are not tracked as official Hubáček bets." : "Model probabilities and fair prices are shown. NFL, CFB, and CBB stay predictions-only until walk-forward backtests clear — not an official Hubáček pick."}</span></div>`;
  }
  if (top) {
    return `<div class="game-pick ${confClass(top.confidence)}">
      <strong>${escapeHtml(top.strategy_label || "Official pick")}</strong>
      ${renderEdgeBadges(top, game)}
      <span>${pickTeamNameLink(top, game.league, game.matchup)} · ${pickSideLabel(top)} ${pickBetTypeLabel(top)} · ${pickMarketLabel(top)} vs model ${pickModelLabel(top)} (+${top.ev_pct != null ? top.ev_pct + "% EV" : ""}${top.edge != null ? ", +" + top.edge + " edge" : ""})${stakeBit}</span>
      <p>${escapeHtml(top.reason || "")}</p>
      ${fairOddsBetStrip(game, top)}
    </div>`;
  }
  if (isSoccer) {
    return `<div class="game-pick neutral"><strong>No official 1X2 pick</strong><span>${hubacekPickRule(state.slate)} not met on home/draw/away lines.</span></div>`;
  }
  return `<div class="game-pick neutral"><strong>No official pick</strong><span>${hubacekPickRule(state.slate)} not met on today's ${officialMarketPhrase(game)} line.</span></div>`;
}

function gameRecommendationsList(game, away, home) {
  const recs = game.recommendations || [];
  if (!recs.length) return "";
  const title = isPredictionsOnlyGame(game)
    ? "Model analysis (not tracked)"
    : "All model recommendations";
  return `<div class="rec-list"><h3>${title}</h3>${recs
    .map((p) =>
      pickCard({
        ...p,
        league: game.league,
        league_name: game.league_name,
        matchup: `${away.name} @ ${home.name}`,
        matchup_obj: game.matchup,
        start_time: game.start_time,
      }),
    )
    .join("")}</div>`;
}

function algoCenter(game) {
  const m = game.model;
  const mk = game.market;
  const away = game.matchup.away;
  const home = game.matchup.home;
  const fav = m.favorite_side === "home" ? home.name : away.name;
  const top = game.top_pick;
  const threeway = m.threeway;
  const isSoccer = Boolean(threeway);
  const algoLabel = threeway ? "1X2 model probabilities" : primaryAlgoLabel(m);
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
        <small>Model favorite: ${fav}${m.home_spread_margin != null && usesSpreadOfficialPicks(game) ? ` · spread margin ${formatSpread(m.home_spread_margin)}` : ""}</small>
      </div>`;
  const drawChip =
    threeway && mk.draw_moneyline != null
      ? `<div class="odds-chip"><span>Draw</span><strong>${formatOdds(mk.draw_moneyline)}</strong><small>Model ${formatOdds(m.draw_projection)}</small></div>`
      : "";
  const oddsRow = `<div class="odds-row game-odds">
        <div class="odds-chip"><span>${teamNameLink(game.league, away.abbr, away.name)}</span><strong>${formatOdds(mk.away_moneyline)}</strong><small>Model ${formatOdds(m.away_projection)}</small></div>
        ${drawChip}
        <div class="odds-chip"><span>${teamNameLink(game.league, home.abbr, home.name)}</span><strong>${formatOdds(mk.home_moneyline)}</strong><small>Model ${formatOdds(m.home_projection)}</small></div>
        ${threeway ? "" : usesSpreadOfficialPicks(game) ? `<div class="odds-chip"><span>Spread / O-U</span><strong>${mk.spread ?? "—"} / ${mk.over_under ?? "—"}</strong><small>${mk.provider || "ESPN"}</small></div>` : mk.over_under != null ? `<div class="odds-chip"><span>O-U</span><strong>${mk.over_under}</strong><small>${mk.provider || "ESPN"}</small></div>` : ""}
      </div>`;
  return `<section class="algo-hero panel">
    ${breadcrumbs([
      { label: "Home", href: "#/" },
      { label: "Games", href: "#/games" },
      { label: `${away.name} @ ${home.name}` },
    ])}
    <div class="algo-hero-head">
      <span class="league-pill">${escapeHtml(game.league_name || "")}</span>
      ${sparseLeaguePill(game.league)}
      ${predictionsOnlyPill(game.league)}
      <h1>${matchupLinks(game.league, away, home)}</h1>
      <p class="game-meta">${formatTime(game.start_time)} · ${escapeHtml(game.status_detail || game.status || "")}</p>
      <p class="db-game-links"><a href="${teamHref(game.league, game.matchup?.away?.abbr)}">${escapeHtml(away.name)}</a> · <a href="${teamHref(game.league, game.matchup?.home?.abbr)}">${escapeHtml(home.name)}</a> · <a href="${leagueHref(game.league)}">${escapeHtml(game.league_name || "")} league</a></p>
    </div>
    ${disclaimerBar("Fair odds below are model prices — shop the best book before staking.")}
    <div class="algo-core">
      ${probBlock}
      ${algoBreakdown(m, game)}
      ${oddsRow}
    </div>
    ${hubacekThreeTrackPanel(game, top)}
    ${contextCallout(m)}
    ${lineShoppingPanel(mk, top)}
    ${gameValuePickBlock(game, top, isSoccer)}
    <details class="factor-details" open><summary>${factorsSectionTitle(m)}</summary><div class="factor-list">${factorBars(m.factors)}</div></details>
    ${gameRecommendationsList(game, away, home)}
  </section>`;
}

function viewDashboard() {
  state.sidebarLeague = null;
  renderSidebar(parseRoute());
  const slate = state.slate || {};
  const summary = slate.summary || {};
  const picks = slate.recommended_bets || [];
  const modelAnalysis = slate.model_analysis_bets || [];
  const games = slate.games || [];
  const leagues = summary.leagues || [...new Set(games.map((g) => g.league))];
  const tracking = state.tracking?.all_time || state.tracking?.summary || {};
  const dateLabel = slate.date_label || "Today";
  const minConf = minHubacekConfidence(slate);
  const hubacekRule = hubacekPickRule(slate);
  const leagueCounts = games.reduce((acc, g) => {
    acc[g.league_name || g.league] = (acc[g.league_name || g.league] || 0) + 1;
    return acc;
  }, {});
  const slateBreakdown = Object.entries(leagueCounts)
    .map(([name, count]) => `${name} (${count})`)
    .join(" · ");
  const contextGames = games.filter(
    (g) => g.model?.context_adjustment_pp != null && Number(g.model.context_adjustment_pp) !== 0,
  );

  appRoot.innerHTML = `
    <section class="tracking-hero panel home-hero">
      <div class="tracking-hero-top">
        <div>
          <h1>Sharp Odds dashboard</h1>
          <p>Today's slate · ${escapeHtml(dateLabel)} · Official Hubáček picks first — stake units, honest EV, and shopped prices when available.</p>
          <p class="muted">${escapeHtml(slateBreakdown || "No games on today's slate yet.")}</p>
        </div>
        <div class="tracking-hero-stats home-stats">
          <div><span>Games</span><strong>${summary.games_analyzed ?? games.length}</strong></div>
          <div><span>Algo picks</span><strong>${summary.recommended_bets ?? picks.length}</strong></div>
          <div><span>Min conf</span><strong>${minConf}+ pp</strong></div>
          <div title="ROI (per staked unit)"><span>All-time ROI</span><strong>${tracking.roi_percent ?? 0}%</strong></div>
        </div>
      </div>
      ${disclaimerBar()}
    </section>

    <div class="rollup-grid home-quick-links">
      <a class="rollup-card panel home-link-card" href="#/picks">
        <h4>Algo picks</h4>
        <strong class="rollup-record">${picks.length}</strong>
        <span>Official Hubáček spots with stake &amp; EV</span>
      </a>
      <a class="rollup-card panel home-link-card" href="#/games">
        <h4>Games</h4>
        <strong class="rollup-record">${games.length}</strong>
        <span>Full algo breakdowns for every matchup</span>
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

    ${contextGames.length ? `<section class="section"><div class="section-head"><h2>Context layer active</h2></div>
      <div class="context-callout-list">${contextGames.slice(0, 4).map((g) => {
        const away = g.matchup?.away?.name || "Away";
        const home = g.matchup?.home?.name || "Home";
        return `<a class="context-panel context-panel--link" href="#/game/${g.event_id}">
          <div class="context-panel-head"><span class="panel-kicker">${escapeHtml(g.league_name || g.league || "")}</span><strong>${formatSignedPct(g.model.context_adjustment_pp)} pp</strong></div>
          <p>${escapeHtml(away)} @ ${escapeHtml(home)}</p>
        </a>`;
      }).join("")}</div></section>` : ""}

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Top official picks</h2>
          <p class="muted section-intro">${escapeHtml(hubacekRule)}. Stake = quarter-Kelly (0.25–3u). Bet early for CLV.</p>
        </div>
        <a class="text-link" href="#/picks">View all →</a>
      </div>
      <div class="picks-grid">${picks.length ? picks.slice(0, 6).map((p) => {
        const g = gameById(p.event_id);
        return pickCard(p, "", g);
      }).join("") : officialEmptyPanel(hubacekRule)}</div>
    </section>
    ${modelAnalysis.length ? `<section class="section"><div class="section-head"><h2>Model predictions (not tracked)</h2><a class="text-link" href="#/picks#model-predictions">View all →</a></div><p class="muted section-intro">Same analysis as game pages — reference only, not logged in official bet history. NFL/CFB/CBB never appear as official Hubáček picks.</p><div class="picks-grid">${modelAnalysis.slice(0, 6).map((p) => pickCard(p)).join("")}</div></section>` : ""}`;
}

function renderTrackingSummary() {
  const periods = [
    ["daily", "Today"],
    ["weekly", "This week"],
    ["monthly", "This month"],
    ["yearly", "This year"],
    ["all_time", "All time"],
  ];
  return `<div class="rollup-grid rollup-grid--tracking">${periods
    .map(([key, label]) => {
      const row =
        key === "all_time"
          ? state.tracking?.all_time
          : (state.tracking?.[key] || [])[0];
      if (!row) {
        return `<div class="rollup-card panel rollup-card--empty"><h4>${label}</h4><p class="muted">No graded bets yet</p><small class="muted">Rollups fill as official picks settle</small></div>`;
      }
      const unitsCls =
        Number(row.units) > 0 ? "clv-positive" : Number(row.units) < 0 ? "clv-negative" : "";
      return `<div class="rollup-card panel"><h4>${label}</h4><strong class="rollup-record">${row.record || "0-0"}</strong><span class="${unitsCls}" title="ROI (per staked unit)">${row.units > 0 ? "+" : ""}${row.units ?? 0}u · ROI ${row.roi_percent ?? 0}%</span><small>${row.bets ?? 0} bets · ${row.pending ?? 0} pending</small></div>`;
    })
    .join("")}</div>`;
}

function viewPicks() {
  state.sidebarLeague = null;
  renderSidebar(parseRoute());
  const picks = state.slate?.recommended_bets || [];
  const modelAnalysis = state.slate?.model_analysis_bets || [];
  const slate = state.slate || {};
  const minConf = minHubacekConfidence(slate);
  const hubacekRule = hubacekPickRule(slate);
  appRoot.innerHTML = `
    <section class="page-head">
      <h1>Algo picks</h1>
      <p><strong>Official Hubáček tracking</strong> covers NBA, WNBA, NHL, MLB, and calibrated soccer leagues only. <strong>NFL, CFB, and CBB</strong> are predictions-only — GradientBoost v2 models are live, but they never enter the official Hubáček book until walk-forward backtests clear. On tracked leagues, official spots fire when the decorrelated model beats the de-vigged book by a backtested per-league gap with honest EV and confidence floors (|p−50| ≥ ${minConf} pp on moneylines). Stakes are quarter-Kelly (0.25–3u).</p>
      <p class="muted">Official gate: ${escapeHtml(hubacekRule)}</p>
    </section>
    ${disclaimerBar()}
    <section class="section picks-section-official">
      <div class="section-head">
        <div>
          <h2>Official picks</h2>
          <p class="muted section-intro">Tracked in performance history. Each card shows side, market, stake, honest EV, market gap, and best available odds when line shopping is present.</p>
        </div>
        <span class="section-count">${picks.length}</span>
      </div>
      <div class="picks-grid">${picks.length ? picks.map((p) => pickCard(p, "", gameById(p.event_id))).join("") : officialEmptyPanel(hubacekRule)}</div>
    </section>
    <section class="section picks-section-reference" id="model-predictions">
      <div class="section-head">
        <div>
          <h2>Model predictions (not tracked)</h2>
          <p class="muted section-intro">Reference-only value spots — including NFL/CFB/CBB (models live, official Hubáček picks disabled) plus MLS, UCL, and other non-tracked analysis. Full probabilities stay on each game page.</p>
        </div>
        <span class="section-count">${modelAnalysis.length}</span>
      </div>
      <div class="picks-grid">${modelAnalysis.length ? modelAnalysis.map((p) => pickCard(p)).join("") : modelAnalysisEmptyPanel()}</div>
    </section>`;
}

function viewGames(league) {
  state.selectedLeague = league || "all";
  state.sidebarLeague = league && league !== "all" ? league : null;
  renderLeagueMenu();
  renderGameSubmenu(state.selectedLeague);
  renderSidebar(parseRoute());
  const games = gamesForLeague(state.selectedLeague);
  appRoot.innerHTML = `<section class="page-head"><h1>Games</h1><p>Today's matchups with full algo breakdowns. Filter by league in the sidebar or open a team sheet from any game.</p></section>
    <div class="slate-list">${games.length ? games.map((g) => gameListCard(g)).join("") : gamesFilterEmptyPanel(state.selectedLeague)}</div>`;
}

function gameListCard(game) {
  const away = game.matchup.away;
  const home = game.matchup.home;
  const m = game.model;
  const fav = m.favorite_side === "home" ? home.name : away.name;
  const top = game.top_pick;
  const stake = top ? stakeUnitsFromPick(top) : null;
  const predictionsOnly = isPredictionsOnlyGame(game);
  let betStrip = "";
  if (top) {
    betStrip = `<div class="game-bet-strip ${predictionsOnly ? "game-bet-strip--ref" : confClass(top.confidence)}">
      <div class="game-bet-strip-main">
        <strong>${predictionsOnly ? "Model value" : escapeHtml(top.strategy_label || "Pick")}</strong>
        <span>${escapeHtml(top.team_name || "")} · ${pickSideLabel(top)} ${pickBetTypeLabel(top)}</span>
      </div>
      <div class="game-bet-strip-metrics">
        ${top.ev_pct != null ? `<span>+${top.ev_pct}% EV</span>` : ""}
        ${stake != null ? `<span>${stake}u</span>` : ""}
        ${top.best_available_odds != null ? `<span>Best ${formatOdds(top.best_available_odds)}</span>` : ""}
      </div>
      ${renderEdgeBadges(top, game)}
    </div>`;
  } else if (m?.context_adjustment_pp != null && Number(m.context_adjustment_pp) !== 0) {
    betStrip = contextCallout(m);
  }
  return `<article class="game-card panel clickable" data-game="${game.event_id}">
    <div class="game-head"><div><span class="league-pill">${escapeHtml(game.league_name || "")}</span>${sparseLeaguePill(game.league)}${predictionsOnlyPill(game.league)}<h3>${matchupLinks(game.league, away, home)}</h3><p class="game-meta">${formatTime(game.start_time)}</p></div>
    <div class="win-chip"><span>${primaryAlgoShort(m)}</span><strong>${escapeHtml(fav)}</strong><small>${m.win_probability}%</small></div></div>
    ${betStrip}
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
    appRoot.innerHTML = `<div class="panel empty-panel">League data for ${escapeHtml(league.toUpperCase())} is not available yet. Try again after the next daily sync.</div>
      <a class="btn btn-secondary" href="#/teams">← All leagues</a>`;
    return;
  }
  renderSidebar(parseRoute());
  const teams = (data.standings?.teams || []).slice(0, 60);
  const betting = data.betting || {};
  const games = betting.games_today || [];
  const leagueName = data.league?.name || league.toUpperCase();
  const predictionsOnly = isPredictionsOnlyLeague(league);
  const boardEmpty = !games.length
    ? `<section class="section"><div class="section-head"><h2>Today's board</h2></div>
        <div class="panel empty-panel${predictionsOnly ? " empty-panel--ref" : ""}">
          <strong>No games today</strong>
          <p class="muted">${predictionsOnly
            ? `${escapeHtml(leagueName)} is predictions-only — when games appear, GradientBoost v2 projections show on the board but are not logged as official Hubáček picks.`
            : "Nothing scheduled for this league on today's slate."}</p>
        </div></section>`
    : "";
  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Leagues", href: "#/teams" },
    { label: leagueName },
  ])}<section class="page-head">
    <span class="league-pill">${data.league?.category}</span>
    ${predictionsOnlyPill(league)}
    <h1>${leagueName}</h1>
    <p>${data.profile?.description || "Standings, ratings, news, and team sheets for betting research."}</p>
    <p class="muted">Season ${data.season_year} · ${betting.game_count || 0} games today · Updated ${new Date(data.generated_at).toLocaleString()}</p>
  </section>
  ${games.length ? `<section class="section"><div class="section-head"><h2>Today's board (${games.length})</h2></div>
    <div class="db-bet-grid">${games.map((g) => renderBettingGameCard(g, league)).join("")}</div></section>` : boardEmpty}
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

function ratingSourceLabel(source, layer, year) {
  const external = {
    "2k": "2K",
    madden: "Madden",
    nhl: "NHL",
    mlb_ts: "MLB The Show",
    fc: "EA FC",
    fm: "FM",
    prior: "Estimated",
    derived: "Estimated",
    model: "Model",
    cached: "Cached",
  };
  const key = String(source || "").toLowerCase();
  if (key && key !== "model") {
    const label = external[key] || (key ? key.toUpperCase() : "Rating");
    if (year) return `${label} '${String(year).slice(-2)}`;
    return label;
  }
  const modelLayers = {
    basketball_matrix: "Matrix model",
    hockey_poisson: "Poisson xG",
    baseball_elo: "Elo",
    nfelo: "nfelo",
    soccer_elo: "Soccer Path A",
    power_ratings: "Power ratings",
  };
  if (key === "model" && layer) {
    return modelLayers[layer] || String(layer).replace(/_/g, " ");
  }
  return "Model";
}

function renderPlayerRatingBadge(rating, { large = false, source = null, layer = null, year = null } = {}) {
  const formatted = formatPlayerRating(rating);
  if (formatted == null) return "";
  const tier = playerRatingTier(rating);
  const cls = large ? "fm-player-rating fm-player-rating--lg" : "fm-player-rating";
  const tooltip = ratingSourceLabel(source, layer, year);
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
  const ratingYear = player.rating_year ?? player.player?.rating_year;
  return `<a class="fm-player-card${isDeep ? " fm-player-card--deep" : ""}" href="${playerHref(league, pid)}">
    <div class="fm-player-photo-wrap">
      ${renderPlayerPhoto(player, league)}
      ${player.jersey ? `<span class="fm-player-jersey">#${player.jersey}</span>` : ""}
      ${renderPlayerRatingBadge(rating, { source: ratingSource, layer: ratingLayer, year: ratingYear })}
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
    appRoot.innerHTML = `<div class="panel empty-panel">Team sheet not available for ${escapeHtml(normalizedAbbr.toUpperCase())} yet. Daily rebuilds populate all teams — try again after the next sync.</div>
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
  const injuries = teamDb?.injuries || [];
  let leagueNewsSource = teamDb?.news || state.dbCache[`league:${league}`]?.news || [];
  if (!leagueNewsSource.length) {
    try {
      const leagueData = await loadLeagueDb(league);
      leagueNewsSource = leagueData?.news || [];
    } catch {
      /* league news optional */
    }
  }
  const teamNews = leagueNewsSource
    .filter((item) => {
      const blob = `${item.headline || ""} ${item.description || ""}`.toLowerCase();
      const tokens = [teamName, normalizedAbbr, teamDb?.team?.abbr]
        .filter(Boolean)
        .map((t) => String(t).toLowerCase());
      return tokens.some((t) => t.length >= 2 && blob.includes(t));
    })
    .slice(0, 6);

  appRoot.innerHTML = `${breadcrumbs([
    { label: "Home", href: "#/" },
    { label: "Leagues", href: "#/teams" },
    { label: leagueName, href: leagueHref(league) },
    { label: teamName },
  ])}${renderTeamHero(teamDb, leagueName, teamName, profile, recentGames)}
  <p class="fm-team-nav"><a href="${leagueHref(league)}">← ${leagueName} league</a></p>
  ${upcoming.length ? `<section class="section"><div class="section-head"><h2>Upcoming — betting context</h2></div>
    <div class="db-bet-grid">${upcoming.map((g) => renderBettingGameCard(g, league)).join("")}</div></section>` : ""}
  ${injuries.length || teamNews.length ? `<section class="section panel availability-risk">
    <div class="section-head"><h2>Availability risk</h2><span class="edge-badge edge-badge--sparse">Research</span></div>
    <p class="muted section-intro">Injuries and recent news that can move lines — check before staking.</p>
    <div class="availability-grid">
      <div class="availability-block"><h3>Injuries</h3>${injuries.length ? `<ul class="db-recent">${injuries.map((p) => `<li><strong>${escapeHtml(p.name || "")}</strong> <span class="muted">(${escapeHtml(p.position || "")})</span> — <span class="fm-injury-status">${escapeHtml(p.status || "")}</span></li>`).join("")}</ul>` : `<p class="muted">No injury report in this snapshot.</p>`}</div>
      ${teamNews.length ? `<div class="availability-block"><h3>News</h3>${renderNewsList(teamNews)}</div>` : `<div class="availability-block"><h3>News</h3><p class="muted">No team-tagged headlines in the league snapshot. <a class="text-link" href="${leagueHref(league)}">Browse league news →</a></p></div>`}
    </div>
  </section>` : ""}
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
  const ratingYear = player.rating_year ?? info.rating_year;
  const ratingTooltip = ratingSourceLabel(ratingSource, ratingLayer, ratingYear);
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
  return `<table class="data-table"><thead><tr><th>${periodLabel(periodKey)}</th><th>Record</th><th>Units</th><th>ROI (per staked unit)</th><th>Bets</th><th>Pending</th></tr></thead><tbody>${rows
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
  const minConf = minHubacekConfidence(state.slate);
  const hubacekRule = hubacekPickRule(state.slate);
  const graded = bets.filter((b) => b.status && b.status !== "pending");
  const youngBook = bets.length > 0 && graded.length < 30;
  const decidedCount = (Number(all.wins) || 0) + (Number(all.losses) || 0);
  const provisionalSample = decidedCount < 30;

  appRoot.innerHTML = `
    <section class="tracking-hero panel">
      <div class="tracking-hero-top">
        <div>
          <h1>Performance tracking</h1>
          <p>Official Hubáček bets are logged with odds frozen at record time, graded on ESPN finals with closing-line value (CLV), staked at quarter-Kelly (0.25–3u), and rolled up day → week → month → year → all time.</p>
          <p class="muted">Tracking since ${escapeHtml(String(since))} · ${escapeHtml(state.tracking?.timezone || "America/Toronto")}</p>
          <p class="muted tracking-rule">${escapeHtml(hubacekRule)}</p>
        </div>
        <div class="tracking-hero-stats">
          <div><span>Record</span><strong>${all.record || "0-0"}</strong>${provisionalSample ? `<small class="edge-badge edge-badge--sparse" title="Fewer than 30 decided bets — treat results as provisional">Provisional — small sample</small>` : ""}</div>
          <div><span>Units</span><strong class="${Number(all.units) > 0 ? "clv-positive" : Number(all.units) < 0 ? "clv-negative" : ""}">${all.units > 0 ? "+" : ""}${all.units ?? 0}u</strong></div>
          <div><span>ROI (per staked unit)</span><strong>${all.roi_percent ?? 0}%</strong></div>
          <div><span>Pending</span><strong>${all.pending ?? 0}</strong></div>
        </div>
      </div>
      <aside class="disclaimer-bar roi-caveat-bar" role="note">
        <strong>Open vs close ROI</strong>
        <span>Opening-line backtest ROI is an <em>upper bound</em>. Live morning tracking usually locks later consensus prices and grades closer to the close — expect live ROI below the open-line study.</span>
      </aside>
      ${disclaimerBar("CLV and ROI need sample size — treat early results as provisional.")}
      ${clvCaption()}
    </section>

    <div class="panel tracking-scope-note" role="note">
      <strong>Official book scope</strong>
      <p class="muted">Rollups cover NBA, WNBA, NHL, MLB, and calibrated soccer only. <span class="edge-badge edge-badge--ref">Predictions only</span> leagues (NFL, CFB, CBB) are excluded from this book — they never appear as empty 0–0 official rollups.</p>
    </div>

    ${youngBook ? `<div class="panel empty-panel tracking-empty-expect">
      <strong>Young track record</strong>
      <p class="muted">Only ${graded.length} graded official bets so far. Expect noisy ROI until dozens of Hubáček spots settle. Focus on process: bet early, shop lines, and size to units — not short-term win rate.</p>
    </div>` : ""}

    <div class="period-tabs">${["daily", "weekly", "monthly", "yearly", "all_time"]
      .map(
        (p) =>
          `<button type="button" class="period-tab ${period === p ? "active" : ""}" data-period="${p}">${periodLabel(p)}</button>`,
      )
      .join("")}</div>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Period rollups</h2>
          <p class="muted section-intro">Official Hubáček spots only — predictions-only leagues omitted.</p>
        </div>
      </div>
      ${renderTrackingSummary()}
    </section>

    <section class="section panel">
      <h2>${periodLabel(period)} breakdown</h2>
      ${period !== "all_time" ? renderUnitsChart(period) : ""}
      ${renderPeriodTable(period)}
    </section>

    <section class="section">
      <div class="section-head"><h2>Bet log (${bets.length})</h2></div>
      <div class="bet-log">${bets.length ? bets.map((b) => `<article class="bet-row panel">
        <div class="bet-row-top">
          <div>
            <strong>${b.team_abbr ? teamNameLink(b.league, b.team_abbr, b.team_name) : escapeHtml(b.team_name || "")}</strong>
            <span class="league-pill">${escapeHtml(b.league_name || b.league || "")}</span>
            ${predictionsOnlyPill(b.league)}
            ${sparseLeaguePill(b.league)}
            ${statusBadge(b.status, b.units)}
          </div>
          <span class="edge-tag">+${b.edge ?? "—"} edge</span>
        </div>
        <p class="muted">${escapeHtml(b.matchup || "")} · ${escapeHtml(b.date || "")}${b.stake_units ? ` · ${b.stake_units}u stake` : ""}</p>
        ${formatClvBlock(b)}
        <div class="pick-odds compact">
          <div><span>${b.bet_type === "spread" ? "Spread" : "Market"}</span><strong>${b.bet_type === "spread" ? formatSpread(b.spread_line) + " (" + formatOdds(b.spread_odds ?? b.market_odds) + ")" : formatOdds(b.market_odds)}</strong></div>
          <div><span>Model</span><strong>${b.bet_type === "spread" && b.model_margin != null ? (b.side === "home" ? "Home" : "Away") + " margin " + formatSpread(b.side === "home" ? b.model_margin : -b.model_margin) : formatOdds(b.model_projection)}</strong></div>
          <div><span>Strategy</span><strong>${escapeHtml(b.strategy_label || "")}</strong></div>
        </div>
        ${b.final_score ? `<p class="final-score">Final: ${escapeHtml(b.final_score)}</p>` : ""}
      </article>`).join("") : `<div class="panel empty-panel">
        <strong>No tracked bets yet</strong>
        <p class="muted">Official picks need +EV and |p−50| ≥ ${minConf} pp (Hubáček). Once logged, this log shows stake, implied-prob CLV, and payout CLV when closing odds are available.</p>
      </div>`}</div>
    </section>`;

  appRoot.querySelectorAll(".period-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.trackingPeriod = btn.dataset.period;
      viewTracking();
    });
  });
}

function viewMethodology() {
  state.sidebarLeague = null;
  renderSidebar(parseRoute());
  appRoot.innerHTML = `
    ${breadcrumbs([{ label: "Home", href: "#/" }, { label: "Methodology" }])}
    <section class="page-head">
      <h1>Methodology &amp; research</h1>
      <p>How the board turns model output into tracked picks: the prediction stack, the decorrelation gate behind official bets, closing-line value, the context layer, and stake sizing — plus an honest list of what is still unproven.</p>
    </section>
    ${disclaimerBar()}

    <section class="section panel methodology-body">
      <h2>Prediction stack</h2>
      <p>Major leagues often run a <strong>single-model GradientBoost v2</strong> as the primary signal — <code>nba_v2</code>, <code>wnba_v2</code>, <code>nhl_v2</code>, <code>mlb_v2</code>, <code>soccer_v2</code>, plus <code>nfl_v2</code>, <code>cfb_v2</code>, and <code>cbb_v2</code> for predictions — rather than three equal layers on every board. Where a blend still applies, layers are weighted per league:</p>
      <ul>
        <li><strong>GradientBoost v2</strong> — sport-specific ensembles under <code>web/{nba,wnba,nhl,mlb,soccer,nfl,cfb,cbb}_v2</code>; the main live model for those leagues when trained weights are present.</li>
        <li><strong>Legacy Algo V2 / power ratings</strong> — efficiency engine and margin-based ratings used as baselines or fallbacks, especially where v2 is thin or absent.</li>
        <li><strong>Other sport layers</strong> — MLB RunCast, Dixon–Coles for soccer, BasketballMatrix / Torvik fallback for CBB, and nfelo-style ratings as supporting football signals.</li>
      </ul>
      <p>A per-league <strong>meta-stack</strong> decides how much to trust each available layer, and a <strong>temperature calibration</strong> pass rescales the probability so that, against history, "60%" actually means about 60%. Official Hubáček tracking is enabled only for NBA, WNBA, NHL, MLB, and calibrated soccer; NFL/CFB/CBB remain predictions-only until backtests clear.</p>
    </section>

    <section class="section panel methodology-body">
      <h2>Hubáček decorrelation</h2>
      <p>An accurate model is not enough to win. If the model mostly agrees with the bookmaker, its value spots cluster on prices where the vig exceeds the tiny disagreement, and the strategy grinds to a loss. Hubáček, Šourek &amp; Železný showed that training a model to <em>decorrelate</em> from the bookmaker — while staying accurate — is what turns predictions into a profitable betting strategy.</p>
      <p>The site applies this as a 3-track split, shown on every game page:</p>
      <ul>
        <li><strong>Display probability</strong> — the full blended model, shown on the board.</li>
        <li><strong>Pick gate probability</strong> — a decorrelated probability that must beat the de-vigged market by a backtested per-league gap (with a confidence floor) before a pick becomes official.</li>
        <li><strong>Honest EV</strong> — expected value computed from the calibrated <em>pre-decorrelation</em> probability, so the decorrelation shove never inflates the printed edge or the Kelly stake.</li>
      </ul>
      <p class="muted methodology-cite">Hubáček, O., Šourek, G., &amp; Železný, F. (2019). “Exploiting sports-betting market using machine learning.” <em>International Journal of Forecasting</em>, 35(2). <a class="text-link" href="https://doi.org/10.1016/j.ijforecast.2019.01.001" target="_blank" rel="noopener">doi:10.1016/j.ijforecast.2019.01.001</a></p>
      <p class="muted">NFL, CFB, and CBB show the same probability tracks for research, but the Hubáček official gate stays off until walk-forward backtests clear — those leagues never appear in the official tracking book.</p>
    </section>

    <section class="section panel methodology-body">
      <h2>Closing Line Value (CLV)</h2>
      <p>CLV compares the implied probability of the odds you took against the implied probability of the closing odds: <strong>positive CLV means you beat the close</strong>. Because the closing line aggregates all late information and sharp money, consistently beating it is the best available predictor of long-run profit — far more informative than a short-run win–loss record, which is mostly variance at small sample sizes.</p>
      <p>Every official pick is logged with odds frozen at record time and graded for implied-probability CLV once closing odds land. One caveat: the closing reference here is the <strong>ESPN consensus feed, not a sharp book</strong>, so CLV on this site is an approximation of true closing value.</p>
    </section>

    <section class="section panel methodology-body">
      <h2>Context layer</h2>
      <p>A small fourth layer nudges probabilities by at most ±3 percentage points in total, based on market-bias research rather than model output:</p>
      <ul>
        <li><strong>Favorite–longshot bias (FLB)</strong> — bettors systematically overpay for longshots and underpay for heavy favorites. Recent work on MLB moneyline markets finds the bias present in <em>opening</em> odds and gone by the close, so a capped nudge (≤1.5 pp) toward strong favorites is applied at open-like prices only.</li>
        <li><strong>Steam signal</strong> — open→current line movement as a proxy for informed money (≤1 pp).</li>
        <li><strong>News keywords</strong> — injury/suspension and hot-streak heuristics from headlines (≤2 pp).</li>
        <li><strong>Sparse-sample EV caps</strong> — printed EV is capped for thin international tournaments (World Cup, friendlies, continental qualifiers) where the sample is too small to trust extreme edges.</li>
      </ul>
    </section>

    <section class="section panel methodology-body">
      <h2>Stake sizing</h2>
      <p>Stakes are <strong>quarter-Kelly</strong>, clamped to <strong>0.25–3 units</strong> (1u ≈ 1% of bankroll), with two conservative adjustments on top:</p>
      <ul>
        <li><strong>Correlation haircut</strong> — same-slate bets share leagues, sides, and market regimes, so stakes are shrunk about 15% versus raw quarter-Kelly.</li>
        <li><strong>EV dampener</strong> — printed EV above 25% is scaled ×0.92 and above 40% ×0.85, because extreme printed EV is usually sample noise and should never size a bet up.</li>
      </ul>
    </section>

    <section class="section panel methodology-body">
      <h2>Honest limitations</h2>
      <ul>
        <li><strong>Beating the close is unproven for NBA.</strong> A pilot of the NBA model graded at real closing lines returned −1.5% ATS with negative CLV. Whatever edge exists lives earlier in the day, not at the close.</li>
        <li><strong>MLB edge concentrates at the open.</strong> Walk-forward testing showed roughly +35% ROI against opening lines versus −3.2% at the close — consistent with the finding that FLB mispricing disappears as the market matures.</li>
        <li><strong>Opening-line backtests overstate live ROI.</strong> Historical ROI vs opening prices is an upper bound; live morning tracking locks later consensus prices and usually grades worse than the open-line study.</li>
        <li><strong>The live tracked sample is very young.</strong> Until dozens of graded bets settle, record, units, and ROI are noise; process metrics (CLV, stake discipline) matter more.</li>
        <li><strong>This is research, not investment advice.</strong> Nothing here guarantees profit; treat the board as decision support for studying betting markets.</li>
      </ul>
    </section>`;
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
  const top = sheet.top_pick;
  const predictionsOnly = sheet.eligible_for_official_picks === false;
  const stake = top ? stakeUnitsFromPick(top) : null;
  return `<article class="db-bet-card panel">
    <div class="db-bet-head">
      <span class="league-pill">${escapeHtml(sheet.league_name || league || "")}</span>
      ${sparseLeaguePill(league)}
      ${predictionsOnlyPill(league)}
      <strong>${matchupLinks(league, matchup.away, matchup.home)}</strong>
      <span class="muted">${formatTime(sheet.start_time)}</span>
    </div>
    <div class="odds-row compact">
      <div class="odds-chip"><span>${teamNameLink(league, matchup.away?.abbr, away)} ML</span><strong>${formatOdds(market.away_moneyline)}</strong><small>Model ${formatOdds(model.away_projection)}</small></div>
      <div class="odds-chip"><span>${teamNameLink(league, matchup.home?.abbr, home)} ML</span><strong>${formatOdds(market.home_moneyline)}</strong><small>Model ${formatOdds(model.home_projection)}</small></div>
      ${usesSpreadOfficialPicks({ league, official_bet_type: sheet.official_bet_type }) && market.spread != null ? `<div class="odds-chip"><span>Spread</span><strong>${formatSpread(market.spread)}</strong></div>` : ""}
    </div>
    <div class="db-bet-model">
      <span>Unified model</span>
      <strong>${model.win_probability ?? "—"}%</strong>
      <small>Fav: ${model.favorite_side || "—"} · Blend ${model.blend_layers || "—"} layers</small>
      ${agreement.required ? `<small>Agreement: ${agreement.agreed ? "all layers" : "split"} (${(agreement.value_sides || []).join(", ") || "none"})</small>` : ""}
    </div>
    ${top ? `<div class="game-bet-strip ${predictionsOnly ? "game-bet-strip--ref" : confClass(top.confidence)}">
      <div class="game-bet-strip-main">
        <strong>${predictionsOnly ? "Model value" : escapeHtml(top.strategy_label || "Pick")}</strong>
        <span>${pickTeamNameLink(top, league, matchup)} · ${pickSideLabel(top)} ${pickBetTypeLabel(top)}${top.ev_pct != null ? ` · +${top.ev_pct}% EV` : ""}${stake != null ? ` · ${stake}u` : ""}${predictionsOnly ? " — not tracked" : ""}</span>
      </div>
      ${renderEdgeBadges(top, sheet)}
    </div>` : predictionsOnly ? `<div class="game-pick neutral"><strong>Predictions only</strong> — model shown, not an official Hubáček pick</div>` : `<div class="game-pick neutral"><strong>No official pick</strong> — ${hubacekPickRule(state.slate)} not met</div>`}
    ${lineShoppingPanel(market, top)}
    <div class="db-bet-links">
      <a href="${teamHref(league, matchup.home?.abbr)}">${escapeHtml(home)}</a>
      <a href="${teamHref(league, matchup.away?.abbr)}">${escapeHtml(away)}</a>
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
    else if (route.path === "methodology") viewMethodology();
    else viewDashboard();
  } catch (err) {
    appRoot.innerHTML = `<div class="panel empty-panel error-panel">${escapeHtml(err.message)}</div>`;
  }
}

async function loadPlatform() {
  let slate;
  try {
    slate = await fetchJson(
      USE_STATIC_API ? api("daily-slate.json") : api("daily/slate"),
    );
  } catch (err) {
    appRoot.innerHTML = `<div class="panel empty-panel error-panel">Could not load daily slate: ${escapeHtml(err.message)}. The site may still be rebuilding — refresh in a few minutes.</div>`;
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
    appRoot.innerHTML = `<div class="panel empty-panel error-panel">${escapeHtml(err.message)}</div>`;
  }
}

window.addEventListener("hashchange", () => render());
navToggle?.addEventListener("click", toggleMobileNav);
mainNav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", closeMobileNav);
});

loadPlatform().catch((err) => {
  appRoot.innerHTML = `<div class="panel empty-panel error-panel">${escapeHtml(err.message)}</div>`;
});
