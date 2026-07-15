"""Market-derived feature registry for decorrelated (β) models.

``use_market=False`` on a HybridConfig only strips the two direct market
columns; a truly odds-blind Hubáček arm must also drop every feature derived
from betting lines (steam, line moves, totals, book counts...). Name-based
matching keeps this robust across leagues whose engines emit different
subsets.
"""

from __future__ import annotations

_EXACT = frozenset(
    {
        "has_market",
        "mkt_home_prob",
        "has_spread",
        "mkt_home_spread",
        "has_open_line",
        "spread_move",
        "ml_steam_pp",
        "has_steam",
        "n_books",
        "mkt_total",
        "has_total",
        "total_move",
        "steam_to_fav",
        "reverse_line_move",
        "chalk_steam",
        "total_line",
        "model_total_vs_line",
        "big_fav_ml_proxy",
        "mkt_open_home",
        "mkt_open_draw",
        "mkt_open_away",
    }
)

_PREFIXES = ("mkt_",)
_SUBSTRINGS = ("steam", "line_move")


def is_market_derived(column: str) -> bool:
    name = str(column)
    if name in _EXACT:
        return True
    if any(name.startswith(p) for p in _PREFIXES):
        return True
    return any(s in name for s in _SUBSTRINGS)


def strip_market_features(columns: list[str]) -> list[str]:
    return [c for c in columns if not is_market_derived(c)]
