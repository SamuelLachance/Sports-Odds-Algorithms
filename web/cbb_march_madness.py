"""March Madness tournament predictions — separate from regular-season CBB."""

from __future__ import annotations

from typing import Any


def run_march_madness_pred(
    *,
    home_abbr: str,
    away_abbr: str,
    round_name: str | None = None,
    neutral_site: bool = False,
) -> dict[str, Any]:
    """
    Tournament-only prediction stub. NOT wired into daily_service or blend_service.

    Regular-season slate uses run_cbb_pred_model via cbb_pred_model.
    """
    _ = round_name
    return {
        "algorithm": "CBBMarchMadness",
        "source": "march-madness-stub",
        "home_abbr": home_abbr.lower(),
        "away_abbr": away_abbr.lower(),
        "neutral_site": neutral_site,
        "home_win_probability": 50.0,
        "note": "Tournament pipeline stub — not used for regular season.",
    }
