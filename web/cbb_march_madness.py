"""STUB / QUARANTINED — March Madness tournament predictions.

Not wired into daily_service, blend_service, or the public board.
Safe to import; returns placeholder 50/50 probabilities only.

Regular-season CBB uses ``web.cbb_pred_model`` / BasketballMatrix.
"""

from __future__ import annotations

from typing import Any

# Explicit quarantine flag for callers / greps.
IS_STUB = True
STUB_NOTE = (
    "Tournament pipeline stub — quarantined; not used for regular season or official picks."
)


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
        "stub": True,
        "home_abbr": home_abbr.lower(),
        "away_abbr": away_abbr.lower(),
        "neutral_site": neutral_site,
        "home_win_probability": 50.0,
        "note": STUB_NOTE,
    }
