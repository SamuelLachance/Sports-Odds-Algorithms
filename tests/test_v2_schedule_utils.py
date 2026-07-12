"""Unit tests for shared v2 schedule-window helpers."""

from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.v2_schedule_utils import (  # noqa: E402
    count_games_in_last_n_days,
    iso_day,
)


def test_iso_day_normalizes_date_and_string() -> None:
    assert iso_day(date(2026, 7, 12)) == "2026-07-12"
    assert iso_day("2026-07-12T19:00:00Z") == "2026-07-12"
    assert iso_day("") is None
    assert iso_day("bad") is None
    assert iso_day(None) is None
    assert iso_day("2026-99-99") is None
    assert iso_day("2026-02-30") is None


def test_count_games_in_last_n_days_matches_day_diff_semantics() -> None:
    recent = [
        "2026-07-01",
        "2026-07-05",
        "2026-07-08",
        "2026-07-11",
        "not-a-date",
    ]
    # Window (Jul 5 .. Jul 11] exclusive of game day Jul 12, inclusive of Jul 5.
    assert count_games_in_last_n_days(recent, "2026-07-12", days=7) == 3
    assert count_games_in_last_n_days(recent, date(2026, 7, 12), days=7) == 3
    # Narrower 3-day window: Jul 9..11 → only Jul 11.
    assert count_games_in_last_n_days(recent, "2026-07-12", days=3) == 1
    # Same-day entries must not count (0 < days).
    assert count_games_in_last_n_days(["2026-07-12"], "2026-07-12", days=7) == 0
    assert count_games_in_last_n_days([], "2026-07-12", days=7) == 0
    assert count_games_in_last_n_days(recent, "2026-07-12", days=0) == 0
    assert count_games_in_last_n_days(recent, "bad-date", days=7) == 0


def test_nba_team_state_uses_shared_window_helper() -> None:
    from web.nba_v2.feature_engine import TeamState

    team = TeamState("bos")
    team.recent_dates = ["2026-07-05", "2026-07-08", "2026-07-11"]
    assert team.games_in_last7(date(2026, 7, 12)) == 3
    assert team.games_in_last3(date(2026, 7, 12)) == 1
