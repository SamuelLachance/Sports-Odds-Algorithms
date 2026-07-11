"""Live multi-book odds enrichment for daily slate markets.

Soft-fails on network/parse errors so the slate build never blocks on books.
Enable with LIVE_MULTI_BOOK=1 (default ON for NBA/NHL/MLB/WNBA).
"""

from __future__ import annotations

import os
from typing import Any

from web.clv_service import american_to_implied_prob
from web.nba_odds_espn import (
    _consensus,
    _get_json,
    _provider_line,
    _valid_american,
)

MULTI_BOOK_LEAGUES = frozenset({"nba", "nhl", "mlb", "wnba"})

# ESPN core odds paths (sport/league segment).
_ODDS_PATH: dict[str, str] = {
    "nba": "basketball/leagues/nba",
    "wnba": "basketball/leagues/wnba",
    "nhl": "hockey/leagues/nhl",
    "mlb": "baseball/leagues/mlb",
}

_DEFAULT_TIMEOUT_S = 5


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


def _odds_url(league: str, event_id: str, competition_id: str) -> str:
    path = _ODDS_PATH[league.lower()]
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


def summarize_book_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build consensus + best-price fields from ESPN core odds items."""
    filtered = [
        item
        for item in items
        if "live" not in ((item.get("provider") or {}).get("name", "").lower())
    ]
    if not filtered:
        return {}

    lines = [_provider_line(item) for item in filtered]
    consensus = _consensus(filtered)

    best_home_ml = best_american_odds([line.get("home_close_ml") for line in lines])
    best_away_ml = best_american_odds([line.get("away_close_ml") for line in lines])
    best_home_spread = best_american_odds([line.get("home_spread_odds") for line in lines])
    best_away_spread = best_american_odds([line.get("away_spread_odds") for line in lines])

    consensus_home_ml = _as_int_odds(consensus.get("home_close_ml"))
    consensus_away_ml = _as_int_odds(consensus.get("away_close_ml"))

    return {
        "n_books": int(consensus.get("n_books") or len(filtered)),
        "best_home_ml": best_home_ml,
        "best_away_ml": best_away_ml,
        "best_home_spread": best_home_spread,
        "best_away_spread": best_away_spread,
        "consensus_home_ml": consensus_home_ml,
        "consensus_away_ml": consensus_away_ml,
        "consensus_home_spread": consensus.get("home_close_spread"),
        "consensus_away_spread": consensus.get("away_close_spread"),
    }


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
    """Fetch multi-book consensus for one event. Empty dict on failure."""
    if not multi_book_enabled(league) or not event_id:
        return {}
    comp = competition_id or event_id
    timeout_s = timeout if timeout is not None else _DEFAULT_TIMEOUT_S
    try:
        payload = _get_json(
            _odds_url(league, event_id, comp),
            timeout=timeout_s,
            retries=1,
        )
    except Exception:  # noqa: BLE001 — soft-fail
        return {}
    items = payload.get("items") or []
    if not items and competition_id is None and event_id:
        # Rare: competition id differs; nothing else to try without scoreboard.
        return {}
    return summarize_book_items(items)


def enrich_market_dict(
    market: dict[str, Any],
    league: str,
    event_id: str,
    *,
    competition_id: str | None = None,
) -> dict[str, Any]:
    """Attach multi-book fields onto a game market dict. Soft-fail preserves input."""
    if not multi_book_enabled(league):
        return market
    try:
        enrichment = fetch_multi_book_odds(
            league,
            event_id,
            competition_id=competition_id,
        )
    except Exception:  # noqa: BLE001
        return market
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
) -> dict[str, Any]:
    """Subset of market enrichment to attach onto recommended_bets / pick cards."""
    if not market.get("n_books"):
        return {}
    best = best_available_for_pick(market, side=side, bet_type=bet_type)
    fields: dict[str, Any] = {
        "n_books": market.get("n_books"),
        "line_shopping_edge_pp": market.get("line_shopping_edge_pp"),
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
        side_edge = shopping_edge_pp(
            int(espn) if espn is not None else None,
            best,
        )
        if side_edge is not None:
            fields["best_vs_espn_pp"] = side_edge
    return {k: v for k, v in fields.items() if v is not None}
