"""Tests for hockey PuckCast prediction layer."""

from web.hockey_features import build_matchup_features, build_training_matrix
from web.hockey_pred_model import (
    TeamMetrics,
    calculate_expected_goals,
    calculate_win_probability,
    predict_matchup_from_model,
)


def test_calculate_expected_goals_home_favorite() -> None:
    home = TeamMetrics("tor", goals_for_pg=3.4, goals_against_pg=2.8, games_played=10)
    away = TeamMetrics("mtl", goals_for_pg=2.9, goals_against_pg=3.3, games_played=10)
    home_xg, away_xg = calculate_expected_goals(home, away)
    assert home_xg > away_xg


def test_calculate_win_probability_sums_near_one() -> None:
    probs = calculate_win_probability(3.2, 2.8)
    assert abs(probs.home_win + probs.away_win - 1.0) < 0.02


def test_predict_matchup_from_model() -> None:
    model = {
        "league": "nhl",
        "cutoff_date": "12-31-2025",
        "dated_games": [],
        "team_metrics": {
            "bos": TeamMetrics("bos", 3.5, 2.6, 15),
            "mtl": TeamMetrics("mtl", 2.8, 3.2, 15),
        },
        "team_game_counts": {"bos": 15, "mtl": 15},
        "xg_rates": {"bos": (3.5, 2.6), "mtl": (2.8, 3.2)},
        "classifier": None,
    }
    result = predict_matchup_from_model(model, "bos", "mtl", use_goalie_edge=False)
    assert result is not None
    assert 0 < result["home_win_probability"] < 100
    assert result["expected_home_goals"] > result["expected_away_goals"]


def test_build_training_matrix_leak_free() -> None:
    dated = []
    for i in range(25):
        day = f"2025-01-{i + 1:02d}"
        home_win = i % 2 == 0
        dated.append(
            (
                day,
                (
                    "aaa",
                    "bbb",
                    "A",
                    "B",
                    3 if home_win else 1,
                    1 if home_win else 3,
                ),
            )
        )
    x_rows, y_rows, names = build_training_matrix(dated, "2025-02-01", "nhl")
    assert len(names) > 10
    assert len(x_rows) == len(y_rows)
    assert len(x_rows) >= 10


def test_build_training_matrix_skips_ties_not_away_wins() -> None:
    """Tied games must be omitted — labeling them 0 trains them as away wins."""
    dated = []
    for i in range(25):
        day = f"2025-01-{i + 1:02d}"
        home_win = i % 2 == 0
        dated.append(
            (
                day,
                (
                    "aaa",
                    "bbb",
                    "A",
                    "B",
                    3 if home_win else 1,
                    1 if home_win else 3,
                ),
            )
        )
    # Insert a tie late enough that both teams have history.
    dated.append(("2025-01-26", ("aaa", "bbb", "A", "B", 2, 2)))
    dated.append(("2025-01-27", ("aaa", "bbb", "A", "B", 4, 1)))

    x_rows, y_rows, _names = build_training_matrix(dated, "2025-02-01", "ncaah")
    assert len(x_rows) == len(y_rows)
    assert all(label in (0, 1) for label in y_rows)
    # Rebuild without the tie: labels must match (tie skipped entirely).
    dated_no_tie = [row for row in dated if row[1][4] != row[1][5]]
    _x2, y2, _ = build_training_matrix(dated_no_tie, "2025-02-01", "ncaah")
    assert y_rows == y2


def test_build_matchup_features_keys() -> None:
    dated = [
        ("2025-01-01", ("aaa", "bbb", "A", "B", 3, 2)),
        ("2025-01-03", ("bbb", "aaa", "B", "A", 2, 4)),
    ]
    feats = build_matchup_features(dated, "2025-01-10", "aaa", "bbb", "nhl")
    assert "home_gf_pg_season" in feats
    assert "away_rest_days" in feats
