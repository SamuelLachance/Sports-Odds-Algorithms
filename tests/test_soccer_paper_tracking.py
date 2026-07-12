"""Soccer paper-tracking edge cases (corrupt log, bad outcomes, blend hook)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_safe_int_rejects_bool_false_poison() -> None:
    """False must not become market_ml=0 (later normalized to EVEN +100)."""
    import web.soccer_paper_tracking as paper

    assert paper._safe_int(False) is None
    assert paper._safe_int(True) is None
    assert paper._safe_int(0) == 0
    assert paper._safe_int("-110") == -110


def test_record_pick_drops_bool_market_ml(tmp_path, monkeypatch) -> None:
    """False must not become EVEN/+100 — refuse the unpriced paper row entirely."""
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)
    paper.record_soccer_paper_pick(
        league="epl",
        event_id="1",
        home_abbr="A",
        away_abbr="B",
        home_name="A FC",
        away_name="B FC",
        game_date="2024-01-01",
        pick_outcome="home",
        model_prob=0.55,
        market_ml=False,  # type: ignore[arg-type]
        edge_pp=5.0,
    )
    assert not path.is_file()
    assert paper._load_paper_log()["bets"] == []


def test_load_paper_log_corrupt_returns_none_not_empty(tmp_path, monkeypatch) -> None:
    """Corrupt / non-list bets must fail closed so grading cannot wipe the file."""
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)
    assert paper._load_paper_log() is None

    path.write_text(json.dumps({"version": 1, "bets": "oops"}), encoding="utf-8")
    assert paper._load_paper_log() is None


def test_grade_corrupt_ledger_does_not_wipe_file(tmp_path, monkeypatch) -> None:
    """Pages grading on a corrupt JSON must not overwrite the ledger with []."""
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    corrupt = "{not-json-but-was-a-ledger"
    path.write_text(corrupt, encoding="utf-8")
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)

    summary = paper.grade_paper_picks()
    assert summary.get("load_error") is True
    assert summary["newly_graded"] == 0
    assert path.read_text(encoding="utf-8") == corrupt


def test_grade_outcome_title_case_draw() -> None:
    import web.soccer_paper_tracking as paper

    assert paper._grade_outcome("Draw", 1, 1) == "win"
    assert paper._grade_outcome("HOME", 1, 2) == "win"


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


def test_grade_leaves_invalid_american_odds_pending(tmp_path, monkeypatch) -> None:
    """|odds| < 100 must stay pending — not settle as 0u win / −1u loss."""
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "bets": [
                    {
                        "key": "epl:50:home",
                        "league": "epl",
                        "event_id": "50",
                        "pick_outcome": "home",
                        "market_ml": 50,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "web.tracking_service._fetch_event_result",
        lambda *_a, **_k: (1, 2),
    )
    summary = paper.grade_paper_picks()
    assert summary["newly_graded"] == 0
    assert summary["settled"] == 0
    reloaded = paper._load_paper_log()
    bet = reloaded["bets"][0]
    assert "status" not in bet
    assert "units" not in bet


def test_maybe_record_skips_unavailable_path_a_stub(tmp_path, monkeypatch) -> None:
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)
    paper.maybe_record_from_blend(
        {
            "blend_mode": "soccer_path_a_unavailable",
            "soccer_pred": None,
            "home_win_probability": 33.33,
            "soccer_pick_signals": {
                "high_confidence_disagreement": True,
                "model_best_outcome": "home",
                "max_edge_pp": 9.0,
            },
        },
        league="epl",
        event_id="77",
        home_abbr="ars",
        away_abbr="liv",
        home_name="Arsenal",
        away_name="Liverpool",
        game_date="2026-07-12",
        home_ml=400,
        draw_ml=300,
        away_ml=-150,
    )
    assert paper._load_paper_log()["bets"] == []

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


def test_maybe_record_skips_unpriced_market_ml(tmp_path, monkeypatch) -> None:
    """Unexecutable prices must not enter the ledger (grading would leave them forever)."""
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)

    paper.maybe_record_from_blend(
        {
            "soccer_pred": {"pick_draw_probability": 0.4},
            "soccer_pick_signals": {
                "high_confidence_disagreement": True,
                "best_edge_outcome": "draw",
                "max_edge_pp": 8.0,
            },
        },
        league="epl",
        event_id="unpriced",
        home_abbr="ars",
        away_abbr="che",
        home_name="Arsenal",
        away_name="Chelsea",
        game_date="2026-07-12",
        home_ml=-110,
        draw_ml=None,
        away_ml=220,
    )
    assert paper._load_paper_log()["bets"] == []

    paper.record_soccer_paper_pick(
        league="epl",
        event_id="direct",
        home_abbr="ars",
        away_abbr="che",
        home_name="Arsenal",
        away_name="Chelsea",
        game_date="2026-07-12",
        pick_outcome="home",
        model_prob=55.0,
        market_ml=None,
        edge_pp=5.0,
    )
    assert paper._load_paper_log()["bets"] == []

def test_pick_signals_best_edge_outcome_can_differ_from_model_best() -> None:
    """Max edge can sit on draw while the model favorite is home."""
    from web.soccer_pick_signals import (
        HIGH_CONFIDENCE_DISAGREEMENT_PP,
        build_soccer_pick_signals,
    )

    signals = build_soccer_pick_signals(
        home_prob=45.0,
        draw_prob=30.0,
        away_prob=25.0,
        # Fair draw ~+233; fat +350 gives a large underdog edge on draw.
        home_ml=-110,
        draw_ml=350,
        away_ml=220,
    )
    assert signals["model_best_outcome"] == "home"
    assert signals["best_edge_outcome"] == "draw"
    assert signals["max_edge_pp"] >= HIGH_CONFIDENCE_DISAGREEMENT_PP
    assert signals["high_confidence_disagreement"] is True


def test_maybe_record_papers_best_edge_outcome_not_model_favorite(
    tmp_path, monkeypatch
) -> None:
    import web.soccer_paper_tracking as paper

    path = tmp_path / "soccer_paper_tracking.json"
    monkeypatch.setattr(paper, "PAPER_TRACKING_PATH", path)

    paper.maybe_record_from_blend(
        {
            "soccer_pick_signals": {
                "high_confidence_disagreement": True,
                "model_best_outcome": "home",
                "best_edge_outcome": "draw",
                "max_edge_pp": 12.0,
            },
            "soccer_pred": {
                "pick_home_win_probability": 45.0,
                "pick_draw_probability": 30.0,
                "pick_away_win_probability": 25.0,
            },
        },
        league="epl",
        event_id="edge-draw",
        home_abbr="ars",
        away_abbr="liv",
        home_name="Arsenal",
        away_name="Liverpool",
        game_date="2026-07-12",
        home_ml=-110,
        draw_ml=350,
        away_ml=220,
    )
    bets = paper._load_paper_log()["bets"]
    assert len(bets) == 1
    assert bets[0]["pick_outcome"] == "draw"
    assert bets[0]["model_prob"] == 30.0
    assert bets[0]["market_ml"] == 350
    assert bets[0]["edge_pp"] == 12.0


def test_grade_paper_picks_persists_summary_when_nothing_new(
    tmp_path, monkeypatch
) -> None:
    """Pages re-runs must refresh on-disk summary even with newly_graded=0."""
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
                        "market_ml": -120,
                        "status": "win",
                        "units": 0.833,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "web.tracking_service._fetch_event_result",
        lambda *_a, **_k: None,
    )
    summary = paper.grade_paper_picks()
    assert summary["newly_graded"] == 0
    assert summary["settled"] == 1
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["summary"]["settled"] == 1
    assert reloaded["summary"]["units"] == 0.833
