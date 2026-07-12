"""Live multi-book odds enrichment for daily slate markets.

Soft-fails on network/parse errors so the slate build never blocks on books.
Enable with LIVE_MULTI_BOOK=1 (default ON for NBA/NHL/MLB/WNBA).

A global wall-time budget (LIVE_MULTI_BOOK_BUDGET_S, default 120s) caps the
cumulative time spent fetching books per build; once exceeded, enrichment
becomes a no-op for the remaining games.
"""

from __future__ import annotations

import os
import time
from typing import Any

from web.clv_service import american_to_implied_prob
from web.mlb_odds_espn import MAX_MLB_RUN_LINE, _provider_line_mlb
from web.nba_odds_espn import (
    MAX_NBA_SPREAD,
    _consensus,
    _get_json,
    _median,
    _nested_american,
    _provider_line,
    _valid_american,
)
from web.nhl_odds_espn import MAX_NHL_PUCK_LINE, _provider_line_nhl

MULTI_BOOK_LEAGUES = frozenset({"nba", "nhl", "mlb", "wnba"})

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
    validated = _valid_american(value if isinstance(value, (int, float)) else None)
    if validated is None and value is not None:
        try:
            validated = _valid_american(float(value))
        except (TypeError, ValueError):
            return None
    if validated is None:
        return None
    return int(round(validated))


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


def shopping_edge_pp(espn_odds: int | None, best_odds: int | None) -> float | None:
    """Implied-prob edge (pp) of best book vs ESPN single-provider quote."""
    if espn_odds is None or best_odds is None:
        return None
    espn_p = american_to_implied_prob(int(espn_odds))
    best_p = american_to_implied_prob(int(best_odds))
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

    ``n_books`` counts only provider items that yielded at least one parsed
    market number; ``book_providers`` lists their names. Median consensus opens
    (``open_home_moneyline`` / ``open_away_moneyline``) feed the steam signal.
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
    for item in filtered:
        line = _provider_line_for_league(league_key, item)
        open_home, open_away = _provider_open_moneylines(item)
        has_close = any(line.get(field) is not None for field in _LINE_FIELDS)
        if not has_close and open_home is None and open_away is None:
            continue
        lines.append(line)
        name = ((item.get("provider") or {}).get("name") or "").strip()
        if name and name not in providers:
            providers.append(name)
        if open_home is not None:
            open_home_values.append(open_home)
        if open_away is not None:
            open_away_values.append(open_away)

    if not lines:
        return {}

    # Prefer league-aware line parse for consensus medians (MLB/NHL validate
    # run/puck lines; NBA path validates with MAX_NBA_SPREAD).
    if league_key in {"mlb", "nhl"}:
        consensus = {
            field: _median([line.get(field) for line in lines])  # type: ignore[arg-type]
            for field in _LINE_FIELDS
        }
    else:
        consensus = _consensus(filtered, max_handicap_abs=max_handicap)

    best_home_ml = best_american_odds([line.get("home_close_ml") for line in lines])
    best_away_ml = best_american_odds([line.get("away_close_ml") for line in lines])
    best_home_spread = best_american_odds([line.get("home_spread_odds") for line in lines])
    best_away_spread = best_american_odds([line.get("away_spread_odds") for line in lines])

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
        "consensus_home_spread": consensus.get("home_close_spread"),
        "consensus_away_spread": consensus.get("away_close_spread"),
    }
    open_home_ml = _as_int_odds(_median(open_home_values)) if open_home_values else None
    open_away_ml = _as_int_odds(_median(open_away_values)) if open_away_values else None
    if open_home_ml is not None:
        summary["open_home_moneyline"] = open_home_ml
    if open_away_ml is not None:
        summary["open_away_moneyline"] = open_away_ml
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
        side_edge = shopping_edge_pp(espn_odds, best_odds)
        if side_edge is not None:
            fields["best_vs_espn_pp"] = side_edge
            # Side-only edge — never inherit game-level max across home/away.
            fields["line_shopping_edge_pp"] = side_edge
        if model_prob_pct is not None and best_odds is not None:
            from web.bet_advisor import expected_value_pct

            fields["ev_pct_at_best"] = round(
                expected_value_pct(float(model_prob_pct), best_odds), 2
            )
    return {k: v for k, v in fields.items() if v is not None}
