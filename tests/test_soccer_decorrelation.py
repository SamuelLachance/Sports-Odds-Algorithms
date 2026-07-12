"""Soccer Path A decorrelation unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_decorrelation import (  # noqa: E402
    decorrelate_threeway_from_market,
    devig_threeway_from_odds,
)


def test_devig_threeway_sums_to_100() -> None:
    probs = devig_threeway_from_odds(-150, 280, 400)
    assert probs is not None
    assert abs(sum(probs) - 100.0) < 0.05


def test_devig_threeway_invalid_american_fails_closed() -> None:
    """Match two-way de-vig: bad magnitudes return None instead of raising."""
    from web.bet_advisor import devig_two_way_probs

    assert devig_two_way_probs(-50, 130) == (None, None)
    assert devig_threeway_from_odds(-50, 280, 400) is None
    assert devig_threeway_from_odds(-150, 75, 400) is None
    # EVEN 0 normalizes to +100 and still de-vigs.
    even = devig_threeway_from_odds(-150, 0, 400)
    assert even is not None
    assert abs(sum(even) - 100.0) < 0.05


def test_decorrelate_moves_away_from_market() -> None:
    model = (55.0, 25.0, 20.0)
    market = (45.0, 30.0, 25.0)
    adjusted = decorrelate_threeway_from_market(model, market)
    assert adjusted[0] > model[0]
    assert abs(sum(adjusted) - 100.0) < 0.05


if __name__ == "__main__":
    test_devig_threeway_sums_to_100()
    test_devig_threeway_invalid_american_fails_closed()
    test_decorrelate_moves_away_from_market()
    print("test_soccer_decorrelation.py: all tests passed")
