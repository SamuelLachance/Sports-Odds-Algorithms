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
from web.wc_simulation import (  # noqa: E402
    derive_scores,
    extract_model_probs,
    extract_unified_lambdas,
    extract_unified_probs,
    pick_knockout_outcome,
    pick_outcome,
    resolve_placeholder,
    sample_poisson_score,
    simulate_tournament,
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


def test_pick_outcome_highest_probability() -> None:
    assert pick_outcome({"home": 20.0, "draw": 55.0, "away": 25.0}) == "draw"
    assert pick_outcome({"home": 60.0, "draw": 25.0, "away": 15.0}) == "home"


def test_knockout_no_draw_rule() -> None:
    assert pick_knockout_outcome({"home": 30.0, "draw": 40.0, "away": 30.0}) in {"home", "away"}
    assert pick_knockout_outcome({"home": 30.0, "draw": 40.0, "away": 30.0}) != "draw"
    assert pick_knockout_outcome({"home": 10.0, "draw": 20.0, "away": 70.0}) == "away"


def test_resolve_group_winner_and_second() -> None:
    standings = {
        "A": [
            {"team": "Mexico", "position": 1},
            {"team": "South Korea", "position": 2},
            {"team": "Czechia", "position": 3},
        ]
    }
    ctx = {
        "group_standings": standings,
        "third_place_ranking": [],
        "round_winners": {},
        "used_third_place_teams": set(),
    }
    assert resolve_placeholder("Group A Winner", **ctx) == "Mexico"
    assert resolve_placeholder("Group A 2nd Place", **ctx) == "South Korea"


def test_resolve_third_place_groups() -> None:
    third_place_ranking = [
        {"team": "Brazil", "group": "C", "third_place_rank": 1, "third_place_qualified": True},
        {"team": "Japan", "group": "F", "third_place_rank": 2, "third_place_qualified": True},
        {"team": "Ghana", "group": "L", "third_place_rank": 9, "third_place_qualified": False},
    ]
    used: set[str] = set()
    team = resolve_placeholder(
        "Third Place Group C/D/F/G/H",
        group_standings={},
        third_place_ranking=third_place_ranking,
        round_winners={},
        used_third_place_teams=used,
    )
    assert team == "Brazil"
    team2 = resolve_placeholder(
        "Third Place Group C/D/F/G/H",
        group_standings={},
        third_place_ranking=third_place_ranking,
        round_winners={},
        used_third_place_teams=used,
    )
    assert team2 == "Japan"


def test_resolve_round_winner_placeholder() -> None:
    round_winners = {"round-of-32": {3: "Argentina"}}
    assert (
        resolve_placeholder(
            "Round of 32 3 Winner",
            group_standings={},
            third_place_ranking=[],
            round_winners=round_winners,
            used_third_place_teams=set(),
        )
        == "Argentina"
    )


def test_derive_scores_from_expected_goals() -> None:
    pred = {
        "model": {
            "soccer_pred": {
                "expected_home_goals": 2.4,
                "expected_away_goals": 0.8,
            }
        }
    }
    home, away = derive_scores("home", pred, allow_draw=True)
    assert home > away


def test_mini_tournament_simulation() -> None:
    """Group A round-robin + final between 1st and 2nd using mock predictions."""
    teams = WORLD_CUP_2026_GROUPS["A"]
    pairings = [
        (teams[0], teams[1]),
        (teams[2], teams[3]),
        (teams[0], teams[2]),
        (teams[1], teams[3]),
        (teams[0], teams[3]),
        (teams[1], teams[2]),
    ]
    matches = []
    predictions = {}
    for idx, (away, home) in enumerate(pairings, start=1):
        eid = f"g{idx}"
        matches.append(
            {
                "event_id": eid,
                "round_slug": "group-stage",
                "round_label": "Group stage",
                "group": "A",
                "start_time": f"2026-06-{10 + idx}T18:00Z",
                "away": {"name": away, "abbr": "TBD"},
                "home": {"name": home, "abbr": "TBD"},
            }
        )
        home_bias = 55 + (idx % 3) * 5
        predictions[eid] = {
            "model": {
                "threeway": True,
                "home_win_probability": home_bias,
                "draw_probability": 20,
                "away_win_probability": 100 - home_bias - 20,
            }
        }

    matches.append(
        {
            "event_id": "f1",
            "round_slug": "final",
            "round_label": "Final",
            "start_time": "2026-07-19T18:00Z",
            "away": {"name": "Group A Winner", "abbr": "TBD"},
            "home": {"name": "Group A 2nd Place", "abbr": "TBD"},
        }
    )
    predictions["f1"] = {
        "model": {
            "threeway": True,
            "home_win_probability": 52,
            "draw_probability": 28,
            "away_win_probability": 20,
        }
    }

    result = simulate_tournament(matches, predictions, mc_iterations=30, mc_seed=7)
    assert "monte_carlo" in result
    assert result["method"] == "unified_3_layer_monte_carlo_poisson"
    assert result["summary"]["total_simulated"] == 7
    assert result["summary"]["group_stage"] == 6
    assert result["summary"]["knockout"] == 1
    assert result["champion"] in teams
    assert result["runner_up"] in teams
    assert result["champion"] != result["runner_up"]
    assert result["bracket_tree"]
    assert result["monte_carlo"]["iterations"] == 30
    final = next(m for m in result["matches"] if m["event_id"] == "f1")
    assert final["outcome"] in {"home", "away"}
    assert final["resolved_from_placeholder"] is True


def test_extract_unified_lambdas_blends_layers() -> None:
    pred = {
        "model": {
            "threeway": True,
            "home_win_probability": 45,
            "draw_probability": 28,
            "away_win_probability": 27,
            "legacy_threeway": {"home_win_probability": 50, "draw_probability": 25, "away_win_probability": 25},
            "power_threeway": {"home_win_probability": 48, "draw_probability": 27, "away_win_probability": 25},
            "soccer_pred": {
                "home_win_probability": 46,
                "draw_probability": 26,
                "away_win_probability": 28,
                "expected_home_goals": 1.55,
                "expected_away_goals": 1.05,
            },
        }
    }
    lam_h, lam_a = extract_unified_lambdas(pred)
    assert lam_h > lam_a
    assert 0.5 <= lam_h <= 3.5


def test_poisson_sample_produces_valid_score() -> None:
    rng = __import__("random").Random(99)
    home, away, outcome = sample_poisson_score(1.4, 1.0, rng, allow_draw=True, knockout=False)
    assert home >= 0 and away >= 0
    assert outcome in {"home", "draw", "away"}


def test_extract_model_probs_from_hub_shape() -> None:
    pred = {
        "model": {
            "threeway": True,
            "home_win_probability": 45.2,
            "draw_probability": 28.1,
            "away_win_probability": 26.7,
        }
    }
    probs = extract_model_probs(pred)
    assert probs["home"] == 45.2
    unified = extract_unified_probs(pred)
    assert unified["home"] == 45.2
    assert probs["draw"] == 28.1
    assert probs["away"] == 26.7


if __name__ == "__main__":
    test_twelve_groups_of_four()
    test_team_group_mapping()
    test_placeholder_detection()
    test_standings_from_results()
    test_pick_outcome_highest_probability()
    test_knockout_no_draw_rule()
    test_resolve_group_winner_and_second()
    test_resolve_third_place_groups()
    test_resolve_round_winner_placeholder()
    test_derive_scores_from_expected_goals()
    test_mini_tournament_simulation()
    test_extract_unified_lambdas_blends_layers()
    test_poisson_sample_produces_valid_score()
    test_extract_model_probs_from_hub_shape()
    test_fetch_events_count()
    print("test_world_cup.py: all tests passed")
