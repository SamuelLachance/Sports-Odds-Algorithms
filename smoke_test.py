"""Quick smoke tests for core algorithms and the web prediction service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
    # Bundled Algo_V2 golden reference (GSW vs POR, 2017-04-16).
    assert abs(float(probability) - 71.32) < 0.05, f"Expected ~71.32, got {probability}"
    print("NBA example prediction OK:", probability)


def test_api_import() -> None:
    from web.app import app

    assert app.title == "Sports Odds Algorithms"
    print("FastAPI app import OK")


def test_football_cbb_v2_artifacts_optional() -> None:
    """Report nfl/cfb/cbb v2 artifact presence; never fail when models are absent."""
    checks = (
        ("nfl_v2", "web.nfl_v2.live"),
        ("cfb_v2", "web.cfb_v2.live"),
        ("cbb_v2", "web.cbb_v2.live"),
    )
    for label, module_path in checks:
        try:
            module = __import__(module_path, fromlist=["artifacts_available"])
            available = bool(module.artifacts_available())
        except Exception as exc:  # noqa: BLE001 - smoke must stay non-fatal
            print(f"{label} artifacts_available: error ({exc})")
            continue
        print(f"{label} artifacts_available: {available}")


def test_toronto_event_date_helper() -> None:
    """Regression: EDT kickoffs must use America/Toronto, not UTC[:10] / UTC−5."""
    from web.season_games import _event_date_iso

    # 04:30Z in June = 00:30 Toronto (EDT) same calendar day.
    assert _event_date_iso("2024-06-15T04:30:00Z") == "2024-06-15"
    # Late-ET tip that rolls UTC past midnight must stay prior Toronto evening.
    assert _event_date_iso("2026-01-16T03:00:00Z") == "2026-01-15"
    assert _event_date_iso("2026-01-15") == "2026-01-15"


def test_docs_do_not_overstate_nba_ml_close_roi() -> None:
    """Honesty guard: NBA ML pilot close ROI is negative; docs must not flip the sign."""
    readme = Path(ROOT) / "README.md"
    agents = Path(ROOT) / "AGENTS.md"
    report = Path(ROOT) / "data" / "supplemental" / "nba-ml" / "backtest_report.md"
    for path in (readme, agents, report):
        assert path.is_file(), path
    report_text = report.read_text(encoding="utf-8")
    assert "ROI: -1.46%" in report_text or "ROI: -1.5" in report_text
    assert "ROI: -3.13%" in report_text or "ROI: -3%" in report_text
    readme_text = readme.read_text(encoding="utf-8")
    assert "−1.5%" in readme_text or "-1.5%" in readme_text
    assert "does not beat the NBA close" in readme_text
    # Forbid upside-down claims that the pilot is a closing-line winner.
    lowered = readme_text.lower()
    assert "beats the nba close" not in lowered
    assert "positive closing-line roi" not in lowered
    # Cron must use comma hours, not step syntax that misstates deploy times.
    agents_text = agents.read_text(encoding="utf-8")
    assert "0 4,10,16,22" in agents_text
    assert "0 4/10/16/22" not in agents_text
    assert "ROI close négatif" in agents_text or "roi close négatif" in agents_text.lower()
    # Public README must match AGENTS / pages.yml (slash form was a prior docs lie).
    assert "0 4,10,16,22" in readme_text
    assert "0 4/10/16/22" not in readme_text
    assert "every push" not in readme_text.lower()
    assert "paths-ignore" in readme_text

def test_core_v2_artifacts_optional() -> None:
    """Report nba/wnba/nhl/mlb/soccer v2 presence; never fail when absent."""
    checks = (
        ("nba_v2", "web.nba_v2.live"),
        ("wnba_v2", "web.wnba_v2.live"),
        ("nhl_v2", "web.nhl_v2.live"),
        ("mlb_v2", "web.mlb_v2.live"),
        ("soccer_v2", "web.soccer_v2.live"),
    )
    for label, module_path in checks:
        try:
            module = __import__(module_path, fromlist=["artifacts_available"])
            available = bool(module.artifacts_available())
        except Exception as exc:  # noqa: BLE001 - smoke must stay non-fatal
            print(f"{label} artifacts_available: error ({exc})")
            continue
        print(f"{label} artifacts_available: {available}")


if __name__ == "__main__":
    test_api_import()
    test_nba_example()
    test_toronto_event_date_helper()
    test_docs_do_not_overstate_nba_ml_close_roi()
    test_football_cbb_v2_artifacts_optional()
    test_core_v2_artifacts_optional()
    print("All smoke tests passed.")
