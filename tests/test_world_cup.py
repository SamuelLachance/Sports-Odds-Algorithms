"""World Cup 2026 hub tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.wc_groups import (  # noqa: E402
    WORLD_CUP_2026_GROUPS,
    team_group,
    is_placeholder_team,
)
from web.world_cup_service import (  # noqa: E402
    _compute_group_standings,
    fetch_world_cup_events,
)


def test_twelve_groups_of_four() -> None:
    assert len(WORLD_CUP_2026_GROUPS) == 12
    assert sum(len(t) for t in WORLD_CUP_2026_GROUPS.values()) == 48


def test_team_group_mapping() -> None:
    assert team_group("Mexico") == "A"
    assert team_group("Canada") == "B"
    assert team_group("Bosnia-Herzegovina") == "B"
    assert team_group("United States") == "D"
    assert team_group("Czechia") == "A"


def test_placeholder_detection() -> None:
    assert is_placeholder_team("Group B 2nd Place")
    assert is_placeholder_team("Round of 32 1 Winner")
    assert not is_placeholder_team("Brazil")


def test_standings_from_results() -> None:
    matches = [
        {
            "round_slug": "group-stage",
            "group": "A",
            "completed": True,
            "away": {"name": "South Africa", "score": 0},
            "home": {"name": "Mexico", "score": 2},
        },
        {
            "round_slug": "group-stage",
            "group": "A",
            "completed": True,
            "away": {"name": "Czechia", "score": 1},
            "home": {"name": "South Korea", "score": 2},
        },
    ]
    tables = _compute_group_standings(matches)
    mexico = next(r for r in tables["A"] if r["team"] == "Mexico")
    assert mexico["points"] == 3
    assert mexico["position"] == 1


def test_fetch_events_count() -> None:
    events = fetch_world_cup_events()
    assert len(events) >= 100


if __name__ == "__main__":
    test_twelve_groups_of_four()
    test_team_group_mapping()
    test_placeholder_detection()
    test_standings_from_results()
    test_fetch_events_count()
    print("test_world_cup.py: all tests passed")
