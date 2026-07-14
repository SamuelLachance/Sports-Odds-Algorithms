"""Live multi-book odds enrichment for daily slate markets.

Soft-fails on network/parse errors so the slate build never blocks on books.
Defaults ON for NBA/NHL/MLB/WNBA interactively; OFF under FAST_DAILY_BUILD
unless LIVE_MULTI_BOOK=1 forces it on (Pages sets that explicitly).

A global wall-time budget (LIVE_MULTI_BOOK_BUDGET_S, default 120s) caps the
cumulative time spent fetching books per build; once exceeded, enrichment
becomes a no-op for the remaining games.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from web.bet_advisor import normalize_american_odds
from web.clv_service import american_to_implied_prob
from web.mlb_odds_espn import MAX_MLB_RUN_LINE, _consensus_mlb, _provider_line_mlb
from web.nba_odds_espn import (
    MAX_NBA_SPREAD,
    _consensus,
    _get_json,
    _median,
    _nested_american,
    _provider_line,
    _valid_american,
)
from web.nhl_odds_espn import MAX_NHL_PUCK_LINE, _consensus_nhl, _provider_line_nhl

MULTI_BOOK_LEAGUES = frozenset({"nba", "nhl", "mlb", "wnba", "nfl"})

_HANDICAP_MAX: dict[str, float] = {
    "nba": MAX_NBA_SPREAD,
    "wnba": MAX_NBA_SPREAD,
    "mlb": MAX_MLB_RUN_LINE,
    "nhl": MAX_NHL_PUCK_LINE,
}

# ESPN core odds paths (sport/league segment).
_ODDS_PATH: dict[str, str] = {
    "nba": "basketball/leagues/nba",
    "wnba": "basketball/leagues/wnba",
    "nhl": "hockey/leagues/nhl",
    "mlb": "baseball/leagues/mlb",
}

_DEFAULT_TIMEOUT_S = 5

# Global wall-time budget for book fetches across one build (env override).
_DEFAULT_BUDGET_S = 120.0
_budget_spent_s: float = 0.0


def _budget_limit_s() -> float:
    raw = (os.environ.get("LIVE_MULTI_BOOK_BUDGET_S") or "").strip()
    if not raw:
        return _DEFAULT_BUDGET_S
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return _DEFAULT_BUDGET_S


def enrichment_budget_exhausted() -> bool:
    """True once cumulative fetch wall-time has used up the build budget."""
    return _budget_spent_s >= _budget_limit_s()


def enrichment_budget_remaining_s() -> float:
    """Seconds left in the build's multi-book fetch budget (never negative)."""
    return max(_budget_limit_s() - _budget_spent_s, 0.0)


def reset_enrichment_budget() -> None:
    """Reset the cumulative fetch-time accumulator (tests / new builds)."""
    global _budget_spent_s
    _budget_spent_s = 0.0


def _charge_budget(elapsed_s: float) -> None:
    global _budget_spent_s
    _budget_spent_s += max(float(elapsed_s), 0.0)


def odds_path_for_league(league: str) -> str | None:
    """ESPN core odds path segment for a multi-book league, or None if unknown."""
    return _ODDS_PATH.get((league or "").lower())


def multi_book_enabled(league: str) -> bool:
    """LIVE_MULTI_BOOK defaults ON for interactive use; OFF during fast CI builds.

    Set LIVE_MULTI_BOOK=1 to force on (even in FAST_DAILY_BUILD).
    Set LIVE_MULTI_BOOK=0/false/no/off to force off.
    """
    flag = (os.environ.get("LIVE_MULTI_BOOK") or "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return league.lower() in MULTI_BOOK_LEAGUES
    # Default: skip multi-book during fast Pages builds (timeout risk).
    fast = (os.environ.get("FAST_DAILY_BUILD") or "").strip().lower()
    if fast in {"1", "true", "yes", "on"}:
        return False
    return league.lower() in MULTI_BOOK_LEAGUES


def line_shopping_status() -> str:
    """Honesty label for whether this build attempted multi-book enrichment.

    Returns:
      - ``on`` — multi-book fetches run for NBA/NHL/MLB/WNBA
      - ``skipped_fast_build`` — FAST_DAILY_BUILD skipped enrichment (default)
      - ``off`` — LIVE_MULTI_BOOK explicitly disabled
    """
    flag = (os.environ.get("LIVE_MULTI_BOOK") or "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return "off"
    if flag in {"1", "true", "yes", "on"}:
        return "on"
    fast = (os.environ.get("FAST_DAILY_BUILD") or "").strip().lower()
    if fast in {"1", "true", "yes", "on"}:
        return "skipped_fast_build"
    return "on"


def _odds_url(league: str, event_id: str, competition_id: str) -> str | None:
    path = odds_path_for_league(league)
    if not path:
        return None
    return (
        f"https://sports.core.api.espn.com/v2/sports/{path}/"
        f"events/{event_id}/competitions/{competition_id}/odds"
    )


def _as_int_odds(value: Any) -> int | None:
    # bool is a subclass of int; reject before False→0→EVEN +100.
    # Text EVEN/PK must match normalize_american_odds (not float()→None drop).
    if isinstance(value, bool):
        return None
    return normalize_american_odds(value)


def _median_open_moneyline(values: list[float]) -> float | None:
    """Median open ML; fail closed on mixed-sign pools (median 0 → fake EVEN).

    Individual books already pass ``_valid_american``; even-book medians can
    still land in the |x| < 100 dead zone or average opposite favorites to 0.
    """
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    nonzero = [v for v in clean if v != 0.0]
    if nonzero and len({v > 0 for v in nonzero}) > 1:
        return None
    return _valid_american(_median(clean))


def best_american_odds(values: list[Any]) -> int | None:
    """Best price for the bettor = lowest implied probability."""
    cleaned: list[int] = []
    for value in values:
        odds = _as_int_odds(value)
        if odds is not None:
            cleaned.append(odds)
    if not cleaned:
        return None
    return min(
        cleaned,
        key=lambda o: american_to_implied_prob(o) if american_to_implied_prob(o) is not None else 1.0,
    )


# --- Multi-book merge + custom de-vigged consensus -------------------------

# Sharp-book weights for the custom consensus (Pinnacle-class > offshore
# early-openers > retail). ESPN-sourced retail books weigh like retail.
SHARP_BOOK_WEIGHTS: dict[str, float] = {
    "pinnacle": 3.0,
    "betonlineag": 2.5,
    "lowvig": 2.0,
    "bovada": 1.2,
    "mybookieag": 1.2,
    "draftkings": 1.0,
    "fanduel": 1.0,
    "betmgm": 1.0,
    "williamhill_us": 1.0,
    "betrivers": 1.0,
}

_BOOK_DISPLAY: dict[str, str] = {
    "betonlineag": "BetOnline",
    "pinnacle": "Pinnacle",
    "lowvig": "LowVig",
    "bovada": "Bovada",
    "mybookieag": "MyBookie",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "williamhill_us": "Caesars",
    "betrivers": "BetRivers",
}


def _canon_book(name: str) -> str:
    """Canonical lowercase book key; espn:draftkings == draftkings."""
    key = str(name or "").strip().lower()
    if key.startswith("espn:"):
        key = key[5:]
    key = key.replace(" ", "").replace("_us", "_us")
    aliases = {"caesars": "williamhill_us", "espnbet": "espnbet", "betonline": "betonlineag"}
    return aliases.get(key, key)


def book_display_name(name: str) -> str:
    canon = _canon_book(name)
    return _BOOK_DISPLAY.get(canon, str(name or "").replace("espn:", "").strip() or canon)


def merge_book_prices(
    espn_book_lines: list[dict[str, Any]] | None,
    store_prices: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge ESPN per-provider rows with fresh early-lines store prices.

    Dedupe by canonical book (ESPN row wins ties — it is fetched live during
    this build). Rows: {book, display, home_ml, away_ml, home_spread, source}.
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in store_prices or []:
        canon = _canon_book(row.get("book"))
        if not canon:
            continue
        merged[canon] = {
            "book": canon,
            "display": book_display_name(canon),
            "home_ml": _as_int_odds(row.get("home_ml")),
            "away_ml": _as_int_odds(row.get("away_ml")),
            "home_spread": row.get("home_spread"),
            "source": row.get("source") or "odds_api",
        }
    for row in espn_book_lines or []:
        canon = _canon_book(row.get("name"))
        if not canon:
            continue
        merged[canon] = {
            "book": canon,
            "display": book_display_name(row.get("name")),
            "home_ml": _as_int_odds(row.get("home_ml")),
            "away_ml": _as_int_odds(row.get("away_ml")),
            "home_spread": row.get("home_spread"),
            "source": "espn",
        }
    return [r for r in merged.values() if r.get("home_ml") is not None or r.get("away_ml") is not None]


def best_ml_with_book(
    book_prices: list[dict[str, Any]],
    *,
    side: str,
) -> tuple[int, str] | None:
    """(best price, display name) for a side across merged books."""
    key = "home_ml" if side == "home" else "away_ml"
    best: tuple[int, str] | None = None
    for row in book_prices:
        odds = _as_int_odds(row.get(key))
        if odds is None:
            continue
        imp = american_to_implied_prob(odds)
        if imp is None:
            continue
        if best is None or imp < (american_to_implied_prob(best[0]) or 1.0):
            best = (odds, str(row.get("display") or row.get("book") or ""))
    return best


def _prob_to_american(prob: float) -> int:
    prob = min(max(prob, 1e-4), 1 - 1e-4)
    if prob >= 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


def custom_consensus_from_books(
    book_prices: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """De-vigged, sharp-weighted median consensus across merged books.

    Each two-sided book is de-vigged (home implied / total implied), then the
    weighted median of home fair probs is taken with SHARP_BOOK_WEIGHTS —
    robust to one book's stale/outlier number, and vig differences between
    books cancel. Returns fair prob + fair (no-vig) American lines.
    """
    probs: list[tuple[float, float, str]] = []  # (fair_home_prob, weight, book)
    for row in book_prices:
        h = _as_int_odds(row.get("home_ml"))
        a = _as_int_odds(row.get("away_ml"))
        if h is None or a is None:
            continue
        ph = american_to_implied_prob(h)
        pa = american_to_implied_prob(a)
        if ph is None or pa is None or ph + pa <= 0:
            continue
        canon = _canon_book(row.get("book"))
        weight = SHARP_BOOK_WEIGHTS.get(canon, 1.0)
        probs.append((ph / (ph + pa), weight, canon))
    if not probs:
        return None
    probs.sort(key=lambda t: t[0])
    total_w = sum(w for _, w, _ in probs)
    acc = 0.0
    fair = probs[-1][0]
    for prob, weight, _ in probs:
        acc += weight
        if acc >= total_w / 2.0:
            fair = prob
            break
    return {
        "fair_home_prob": round(fair, 4),
        "fair_home_ml": _prob_to_american(fair),
        "fair_away_ml": _prob_to_american(1.0 - fair),
        "books_used": len(probs),
        "method": "devig_sharp_weighted_median",
    }


def _spreads_equal(left: Any, right: Any) -> bool:
    """True when both handicaps parse to the same finite line (exact match)."""
    if left is None or right is None:
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    return abs(a - b) < 1e-9


def best_spread_odds_for_line(
    lines: list[dict[str, Any]],
    *,
    odds_key: str,
    spread_key: str,
    target_spread: float | None,
) -> int | None:
    """Best juice only among books quoting ``target_spread`` (not a different line)."""
    if target_spread is None or isinstance(target_spread, bool):
        return None
    matched = [
        line.get(odds_key)
        for line in lines
        if _spreads_equal(line.get(spread_key), target_spread)
    ]
    return best_american_odds(matched)


def shopping_edge_pp(espn_odds: int | None, best_odds: int | None) -> float | None:
    """Implied-prob edge (pp) of best book vs ESPN single-provider quote."""
    espn = _as_int_odds(espn_odds)
    best = _as_int_odds(best_odds)
    if espn is None or best is None:
        return None
    espn_p = american_to_implied_prob(espn)
    best_p = american_to_implied_prob(best)
    if espn_p is None or best_p is None:
        return None
    return round((espn_p - best_p) * 100.0, 2)


def _provider_open_moneylines(item: dict[str, Any]) -> tuple[float | None, float | None]:
    """One book's opening moneylines (items[].homeTeamOdds.open.moneyLine.american)."""
    home = item.get("homeTeamOdds") or {}
    away = item.get("awayTeamOdds") or {}
    open_home = _valid_american(_nested_american(home, "open", "moneyLine"))
    open_away = _valid_american(_nested_american(away, "open", "moneyLine"))
    return open_home, open_away


_LINE_FIELDS = (
    "home_close_ml",
    "away_close_ml",
    "home_close_spread",
    "away_close_spread",
    "home_spread_odds",
    "away_spread_odds",
)


def _provider_line_for_league(
    league: str | None,
    item: dict[str, Any],
) -> dict[str, float | None]:
    """Dispatch to sport-specific parsers so fake ML-sized handicaps are dropped."""
    key = (league or "nba").lower()
    if key == "mlb":
        return _provider_line_mlb(item)
    if key == "nhl":
        return _provider_line_nhl(item)
    return _provider_line(item, max_handicap_abs=_HANDICAP_MAX.get(key, MAX_NBA_SPREAD))


def summarize_book_items(
    items: list[dict[str, Any]],
    *,
    league: str | None = None,
) -> dict[str, Any]:
    """Build consensus + best-price + opening-line fields from ESPN core odds items.

    ``n_books`` counts only providers with at least one parsed *close* market
    number; open-only shells still feed steam medians but do not inflate
    ``book_providers``. Median consensus opens (``open_home_moneyline`` /
    ``open_away_moneyline``) feed the steam signal.
    """
    filtered = [
        item
        for item in items
        if "live" not in ((item.get("provider") or {}).get("name", "").lower())
    ]
    if not filtered:
        return {}

    league_key = (league or "nba").lower()
    max_handicap = _HANDICAP_MAX.get(league_key, MAX_NBA_SPREAD)

    lines: list[dict[str, float | None]] = []
    providers: list[str] = []
    open_home_values: list[float] = []
    open_away_values: list[float] = []
    book_lines: list[dict[str, Any]] = []
    for item in filtered:
        line = _provider_line_for_league(league_key, item)
        open_home, open_away = _provider_open_moneylines(item)
        has_close = any(line.get(field) is not None for field in _LINE_FIELDS)
        # Open-only providers still feed steam medians, but must not inflate
        # n_books / book_providers (those gate close-consensus line shopping).
        if open_home is not None and open_away is not None:
            open_home_values.append(open_home)
            open_away_values.append(open_away)
        prov_name = ((item.get("provider") or {}).get("name") or "").strip()
        if prov_name:
            # Per-provider rows for the early-lines store + per-book UI.
            book_lines.append(
                {
                    "name": prov_name,
                    "home_ml": line.get("home_close_ml"),
                    "away_ml": line.get("away_close_ml"),
                    "home_spread": line.get("home_close_spread"),
                    "total": line.get("close_total"),
                    "open_home_ml": open_home,
                    "open_away_ml": open_away,
                    "open_home_spread": line.get("home_open_spread"),
                    "open_total": line.get("open_total"),
                }
            )
        if not has_close:
            continue
        lines.append(line)
        if prov_name and prov_name not in providers:
            providers.append(prov_name)

    # Open-only slate: still return steam medians with n_books=0 (do not drop
    # opens via an empty early return). Completely empty input stays {}.
    if not lines and not open_home_values and not open_away_values:
        return {}

    if lines:
        # Prefer league-aware consensus (MLB/NHL validate run/puck lines and
        # reject mixed-sign / even-book dead-zone medians; NBA path uses
        # MAX_NBA_SPREAD / raised college caps via max_handicap_abs).
        if league_key == "mlb":
            consensus = _consensus_mlb(filtered)
        elif league_key == "nhl":
            consensus = _consensus_nhl(filtered)
        else:
            consensus = _consensus(filtered, max_handicap_abs=max_handicap)

        best_home_ml = best_american_odds([line.get("home_close_ml") for line in lines])
        best_away_ml = best_american_odds([line.get("away_close_ml") for line in lines])
        consensus_home_spread = consensus.get("home_close_spread")
        consensus_away_spread = consensus.get("away_close_spread")
        # Juice from a different handicap is not shoppable for the consensus line.
        best_home_spread = best_spread_odds_for_line(
            lines,
            odds_key="home_spread_odds",
            spread_key="home_close_spread",
            target_spread=consensus_home_spread,
        )
        best_away_spread = best_spread_odds_for_line(
            lines,
            odds_key="away_spread_odds",
            spread_key="away_close_spread",
            target_spread=consensus_away_spread,
        )

        consensus_home_ml = _as_int_odds(consensus.get("home_close_ml"))
        consensus_away_ml = _as_int_odds(consensus.get("away_close_ml"))

        summary: dict[str, Any] = {
            "n_books": len(lines),
            "book_providers": providers,
            "best_home_ml": best_home_ml,
            "best_away_ml": best_away_ml,
            "best_home_spread": best_home_spread,
            "best_away_spread": best_away_spread,
            "consensus_home_ml": consensus_home_ml,
            "consensus_away_ml": consensus_away_ml,
            "consensus_home_spread": consensus_home_spread,
            "consensus_away_spread": consensus_away_spread,
        }
    else:
        summary = {
            "n_books": 0,
            "book_providers": [],
            "best_home_ml": None,
            "best_away_ml": None,
            "best_home_spread": None,
            "best_away_spread": None,
            "consensus_home_ml": None,
            "consensus_away_ml": None,
            "consensus_home_spread": None,
            "consensus_away_spread": None,
        }
    # Open ML medians: reject mixed-sign pools (median 0 → fake EVEN) and
    # require both sides valid so dead-zone home + valid away never unpairs.
    open_home_raw = _median_open_moneyline(open_home_values) if open_home_values else None
    open_away_raw = _median_open_moneyline(open_away_values) if open_away_values else None
    open_home_ml = _as_int_odds(open_home_raw) if open_home_raw is not None else None
    open_away_ml = _as_int_odds(open_away_raw) if open_away_raw is not None else None
    if open_home_ml is not None and open_away_ml is not None:
        summary["open_home_moneyline"] = open_home_ml
        summary["open_away_moneyline"] = open_away_ml
    if book_lines:
        summary["book_lines"] = book_lines
    return summary


def line_shopping_edge_from_market(
    enrichment: dict[str, Any],
    *,
    espn_home_ml: int | None,
    espn_away_ml: int | None,
) -> float | None:
    """Max ML shopping edge (pp) across home/away vs ESPN quote."""
    edges = [
        shopping_edge_pp(espn_home_ml, enrichment.get("best_home_ml")),
        shopping_edge_pp(espn_away_ml, enrichment.get("best_away_ml")),
    ]
    present = [e for e in edges if e is not None]
    if not present:
        return None
    return max(present)


def fetch_multi_book_odds(
    league: str,
    event_id: str,
    *,
    competition_id: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Fetch multi-book consensus for one event. Empty dict on failure.

    Clear no-op paths (disabled league, missing event id, unknown odds path,
    or exhausted LIVE_MULTI_BOOK_BUDGET_S) return {} without touching the
    network. Network/parse failures also return {}; only attempted fetches
    charge wall-time against the budget.
    """
    if not multi_book_enabled(league) or not event_id:
        return {}
    if enrichment_budget_exhausted():
        return {}
    url = _odds_url(league, event_id, competition_id or event_id)
    if not url:
        # League allowed by flag but missing from _ODDS_PATH — do not KeyError.
        return {}
    timeout_s = timeout if timeout is not None else _DEFAULT_TIMEOUT_S
    started = time.monotonic()
    payload: dict[str, Any] = {}
    try:
        payload = _get_json(url, timeout=timeout_s, retries=1) or {}
    except Exception:  # noqa: BLE001 — soft-fail
        payload = {}
    finally:
        _charge_budget(time.monotonic() - started)
    items = payload.get("items") or []
    if not items:
        return {}
    return summarize_book_items(items, league=league)


def apply_enrichment_to_market(
    market: dict[str, Any],
    enrichment: dict[str, Any],
) -> dict[str, Any]:
    """Pure merge of a summarize_book_items payload onto a market dict."""
    if not enrichment:
        return market
    edge = line_shopping_edge_from_market(
        enrichment,
        espn_home_ml=market.get("home_moneyline"),
        espn_away_ml=market.get("away_moneyline"),
    )
    updated = {
        **market,
        **enrichment,
    }
    if edge is not None:
        updated["line_shopping_edge_pp"] = edge
    return updated


def enrich_market_dict(
    market: dict[str, Any],
    league: str,
    event_id: str,
    *,
    competition_id: str | None = None,
) -> dict[str, Any]:
    """Attach multi-book fields onto a game market dict. Soft-fail preserves input.

    Returns the input unchanged once the global fetch budget is exhausted so a
    slow book API can never stall the rest of the slate build.
    """
    if not multi_book_enabled(league):
        return market
    if enrichment_budget_exhausted():
        return market
    try:
        enrichment = fetch_multi_book_odds(
            league,
            event_id,
            competition_id=competition_id,
        )
    except Exception:  # noqa: BLE001
        return market
    return apply_enrichment_to_market(market, enrichment)


def best_available_for_pick(
    market: dict[str, Any],
    *,
    side: str,
    bet_type: str = "moneyline",
) -> int | None:
    """Best shopped American price for a recommended pick side."""
    if bet_type == "spread":
        if side == "home":
            return market.get("best_home_spread")
        if side == "away":
            return market.get("best_away_spread")
        return None
    if side == "home":
        return market.get("best_home_ml")
    if side == "away":
        return market.get("best_away_ml")
    return None


def line_shopping_fields_for_pick(
    market: dict[str, Any],
    *,
    side: str,
    bet_type: str = "moneyline",
    model_prob_pct: float | None = None,
) -> dict[str, Any]:
    """Subset of market enrichment to attach onto recommended_bets / pick cards.

    Official EV/Kelly stay on ESPN ``market_odds``. When a better book price
    exists, also report ``ev_pct_at_best`` so the UI does not imply Honest EV
    is executable at the shopped price.

    ``line_shopping_edge_pp`` on a pick is the shopping edge for *this* side,
    not the game-level max across home/away.
    """
    if not market.get("n_books"):
        return {}
    best = best_available_for_pick(market, side=side, bet_type=bet_type)
    fields: dict[str, Any] = {
        "n_books": market.get("n_books"),
        "consensus_home_ml": market.get("consensus_home_ml"),
        "consensus_away_ml": market.get("consensus_away_ml"),
    }
    if best is not None:
        fields["best_available_odds"] = best
        espn = (
            market.get("home_spread_odds" if side == "home" else "away_spread_odds")
            if bet_type == "spread"
            else market.get("home_moneyline" if side == "home" else "away_moneyline")
        )
        # Use _as_int_odds — bare int("-110.0") raises and can abort pick attach.
        espn_odds = _as_int_odds(espn)
        best_odds = _as_int_odds(best)
        # Spread juice from a different handicap is not a real shopping edge.
        side_edge = None
        if bet_type == "spread":
            if _spreads_equal(
                market.get("spread"), market.get("consensus_home_spread")
            ):
                side_edge = shopping_edge_pp(espn_odds, best_odds)
        else:
            side_edge = shopping_edge_pp(espn_odds, best_odds)
        if side_edge is not None:
            fields["best_vs_espn_pp"] = side_edge
            # Side-only edge — never inherit game-level max across home/away.
            fields["line_shopping_edge_pp"] = side_edge
        # bool / NaN / ±inf must not invent shopped EV (True→1%, inf→99.9% clamp).
        if (
            model_prob_pct is not None
            and not isinstance(model_prob_pct, bool)
            and best_odds is not None
        ):
            try:
                prob = float(model_prob_pct)
            except (TypeError, ValueError):
                prob = float("nan")
            if math.isfinite(prob):
                from web.bet_advisor import expected_value_pct

                fields["ev_pct_at_best"] = round(
                    expected_value_pct(prob, best_odds), 2
                )
    return {k: v for k, v in fields.items() if v is not None}
