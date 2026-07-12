"""America/Toronto cutoff helpers used by tuning / odds-fetch scripts."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.tracking_service import toronto_cutoff_mdy  # noqa: E402


def test_toronto_cutoff_mdy_differs_from_utc_near_midnight() -> None:
    # 04:00 UTC on Jan 15 = still Jan 14 evening in America/Toronto (EST).
    utc = datetime(2026, 1, 15, 4, 0, tzinfo=ZoneInfo("UTC"))
    assert toronto_cutoff_mdy(utc) == "1-14-2026"


def test_tune_and_fetch_scripts_use_toronto_today() -> None:
    scripts = (
        "tune_pick_strategy.py",
        "tune_sports_meta.py",
        "tune_soccer_meta.py",
        "train_ensemble_ml.py",
        "evaluate_stacking_logloss.py",
        "fetch_nba_odds.py",
        "fetch_nhl_odds.py",
        "fetch_mlb_odds.py",
        "fetch_wnba_odds.py",
    )
    for name in scripts:
        text = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "toronto_today" in text or "toronto_cutoff_mdy" in text, name
        assert "date.today()" not in text, name
