"""Quick smoke tests for core algorithms and the web prediction service."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def test_nba_example() -> None:
    from web.predict_service import predict_match

    result = predict_match(
        league="nba",
        away_slug="portland-trail-blazers",
        home_slug="golden-state-warriors",
        date="4-16-2017",
        season_year="2017",
        algo_version="Algo_V2",
    )
    probability = result["prediction"]["win_probability"]
    assert probability == 71.32, f"Expected 71.32, got {probability}"
    print("NBA example prediction OK:", probability)


def test_api_import() -> None:
    from web.app import app

    assert app.title == "Sports Odds Algorithms"
    print("FastAPI app import OK")


if __name__ == "__main__":
    test_api_import()
    test_nba_example()
    print("All smoke tests passed.")
