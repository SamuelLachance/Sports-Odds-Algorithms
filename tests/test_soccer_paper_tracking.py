"""Soccer paper-tracking edge cases (corrupt log, bad outcomes, blend hook)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_load_paper_log_tolerates_corrupt_and_non_list_bets(tmp_path, monkeypatch) -> None:
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)
    payload = paper._load_paper_log()
    assert payload == {"version": 1, "bets": []}

    path.write_text(json.dumps({"version": 1, "bets": "oops"}), encoding="utf-8")
    payload = paper._load_paper_log()
    assert payload["bets"] == []


def test_grade_skips_unknown_outcome_and_bad_market_ml(tmp_path, monkeypatch) -> None:
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "bets": [
                    {
                        "key": "epl:1:home",
                        "league": "epl",
                        "event_id": "1",
                        "pick_outcome": "home",
                        "market_ml": "not-an-int",
                    },
                    {
                        "key": "epl:2:bogus",
                        "league": "epl",
                        "event_id": "2",
                        "pick_outcome": "bogus",
                    },
                    {
                        "key": "epl:3:away",
                        "league": "epl",
                        "event_id": "3",
                        "pick_outcome": "away",
                        "market_ml": 150,
                    },
                    "not-a-dict",
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "web.tracking_service._fetch_event_result",
        lambda *_a, **_k: (2, 1),
    )
    summary = paper.grade_paper_picks()
    # Bad/missing market_ml stays pending; only priced picks settle.
    assert summary["newly_graded"] == 1
    assert summary["settled"] == 1
    assert summary["wins"] == 1
    assert summary["units"] == 1.5
    reloaded = paper._load_paper_log()
    unpriced = next(b for b in reloaded["bets"] if isinstance(b, dict) and b["event_id"] == "1")
    assert "status" not in unpriced
    assert "units" not in unpriced
    graded = next(b for b in reloaded["bets"] if isinstance(b, dict) and b["event_id"] == "3")
    assert graded["status"] == "win"
    assert graded["units"] == 1.5


def test_maybe_record_from_blend_uses_away_win_fallback(tmp_path, monkeypatch) -> None:
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)

    paper.maybe_record_from_blend(
        {
            "soccer_pick_signals": {
                "high_confidence_disagreement": True,
                "model_best_outcome": "away",
                "max_edge_pp": "4.5",
            },
            "away_win_probability": 41.25,
            "soccer_pred": {},
        },
        league="epl",
        event_id="99",
        home_abbr="ars",
        away_abbr="liv",
        home_name="Arsenal",
        away_name="Liverpool",
        game_date="2026-07-12",
        home_ml=-120,
        draw_ml=260,
        away_ml=300,
    )
    payload = paper._load_paper_log()
    assert len(payload["bets"]) == 1
    assert payload["bets"][0]["pick_outcome"] == "away"
    assert payload["bets"][0]["model_prob"] == 41.25
    assert payload["bets"][0]["edge_pp"] == 4.5

    # empty event_id / crash inputs must not raise
    paper.maybe_record_from_blend(
        None,  # type: ignore[arg-type]
        league="epl",
        event_id="",
        home_abbr="a",
        away_abbr="b",
        home_name="A",
        away_name="B",
        game_date="2026-07-12",
        home_ml=None,
        draw_ml=None,
        away_ml=None,
    )


def test_maybe_record_from_blend_skips_missing_model_prob(tmp_path, monkeypatch) -> None:
    """Missing pick/fallback probs must not invent model_prob=0.0 paper rows."""
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)

    paper.maybe_record_from_blend(
        {
            "soccer_pick_signals": {
                "high_confidence_disagreement": True,
                "model_best_outcome": "home",
                "max_edge_pp": 5,
            },
            "soccer_pred": {},
        },
        league="epl",
        event_id="1",
        home_abbr="ars",
        away_abbr="liv",
        home_name="Arsenal",
        away_name="Liverpool",
        game_date="2026-07-12",
        home_ml=-120,
        draw_ml=250,
        away_ml=300,
    )
    assert paper._load_paper_log()["bets"] == []
