"""News-headline plumbing + single-fetch enrichment wiring in daily_service.

All network boundaries are mocked; no real HTTP in these tests.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.daily_service import (  # noqa: E402
    _league_news_headlines,
    predict_live_game,
)
from web.espn_client import MarketOdds, ScheduledGame  # noqa: E402


def test_league_news_headlines_extracts_and_caps_at_40() -> None:
    payload = {"articles": [{"headline": f"Headline {i}"} for i in range(60)]}
    with patch.dict("os.environ", {"NEWS_SIGNALS": "1"}, clear=False):
        with patch(
            "web.sports_db.espn_fetch.fetch_league_news", return_value=payload
        ) as mock_fetch:
            headlines = _league_news_headlines("nba")
    assert len(headlines) == 40
    assert headlines[0] == "Headline 0"
    mock_fetch.assert_called_once_with("nba", limit=40)


def test_league_news_headlines_disabled_via_env() -> None:
    with patch.dict("os.environ", {"NEWS_SIGNALS": "0"}, clear=False):
        with patch("web.sports_db.espn_fetch.fetch_league_news") as mock_fetch:
            assert _league_news_headlines("nba") == []
    mock_fetch.assert_not_called()


def test_league_news_headlines_soft_fail_and_empty() -> None:
    with patch.dict("os.environ", {"NEWS_SIGNALS": "1"}, clear=False):
        with patch(
            "web.sports_db.espn_fetch.fetch_league_news",
            side_effect=OSError("espn down"),
        ):
            assert _league_news_headlines("nba") == []
        with patch("web.sports_db.espn_fetch.fetch_league_news", return_value=None):
            assert _league_news_headlines("nba") == []


def test_league_news_headlines_skips_blank_and_nonstring_entries() -> None:
    payload = {
        "articles": [
            {"headline": ""},
            {"title": "From title"},
            "junk-entry",
            {"headline": None},
            {"headline": "  Real headline  "},
        ]
    }
    with patch.dict("os.environ", {"NEWS_SIGNALS": "1"}, clear=False):
        with patch("web.sports_db.espn_fetch.fetch_league_news", return_value=payload):
            assert _league_news_headlines("nba") == ["From title", "Real headline"]


def test_blend_predictions_accepts_headlines_kwarg() -> None:
    from web.blend_service import blend_predictions

    assert "headlines" in inspect.signature(blend_predictions).parameters


def _scheduled_game() -> ScheduledGame:
    return ScheduledGame(
        league="nba",
        event_id="401999",
        name="Miami Heat at Boston Celtics",
        start_time="2026-07-10T23:00Z",
        status="pre",
        status_detail="7:00 PM ET",
        away_abbr="MIA",
        home_abbr="BOS",
        away_name="Miami Heat",
        home_name="Boston Celtics",
        away_espn_id="14",
        home_espn_id="2",
        market=MarketOdds(away_moneyline=100, home_moneyline=-110),
    )


def test_predict_live_game_single_fetch_opens_and_headlines() -> None:
    """Enrichment fetched once; opens + headlines reach the context hook;
    the same enrichment payload lands on the final market dict."""
    game = _scheduled_game()
    headlines = ["Celtics star sidelined with injury"]
    enrichment = {
        "n_books": 2,
        "book_providers": ["BookA", "BookB"],
        "best_home_ml": -105,
        "best_away_ml": 105,
        "open_home_moneyline": 120,
        "open_away_moneyline": -140,
        "consensus_home_ml": -108,
        "consensus_away_ml": 102,
    }
    blended_stub = {
        "algorithm": "Unified",
        "blend_mode": "blended",
        "total_score": -55.0,
        "win_probability": 55.0,
        "favorite_side": "home",
        "blended_home_win_probability": 55.0,
    }
    captured: dict = {}

    def _fake_context(blended, *, market=None, headlines=None, **kwargs):
        captured["context_market"] = dict(market or {})
        captured["context_headlines"] = list(headlines or [])
        return {**blended, "context_adjustment_pp": 0.0}

    algo_instance = MagicMock()
    algo_instance.calculate_V2.return_value = {"total": -10.0}
    odds_instance = MagicMock()
    odds_instance.analyze2.return_value = {}

    with (
        patch(
            "web.daily_service.resolve_team",
            side_effect=lambda league, abbr, name: (
                ["bos", "boston-celtics"] if abbr == "BOS" else ["mia", "miami-heat"]
            ),
        ),
        patch("web.daily_service.load_live_team_data", return_value=[{"seed": 1}]),
        patch("algo.Algo", return_value=algo_instance),
        patch("odds_calculator.Odds_Calculator", return_value=odds_instance),
        patch(
            "web.live_odds_enrichment.fetch_multi_book_odds",
            return_value=enrichment,
        ) as mock_fetch,
        patch(
            "web.daily_service.blend_predictions", return_value=dict(blended_stub)
        ) as mock_blend,
        patch("web.context_signals.apply_context_to_blend", side_effect=_fake_context),
        patch(
            "web.daily_service.apply_ensemble_ml",
            side_effect=lambda blended, league, **kw: blended,
        ),
        patch(
            "web.daily_service.ensure_hubacek_in_blend",
            side_effect=lambda blended, **kw: blended,
        ),
        patch("web.daily_service.compute_model_agreement", return_value={}),
        patch("web.daily_service.get_pick_thresholds", return_value={}),
        patch("web.daily_service.official_pick_binary_probs", return_value=(45.0, 55.0)),
        patch("web.daily_service.evaluate_official_picks_for_game", return_value=[]),
    ):
        result = predict_live_game(game, headlines=headlines)

    # Books fetched exactly once — never re-fetched for the market dict.
    assert mock_fetch.call_count == 1

    # Headlines flow into blend_predictions and the context hook.
    assert mock_blend.call_args.kwargs["headlines"] == headlines
    assert captured["context_headlines"] == headlines

    # Multi-book consensus opens activate the steam signal in the hook.
    assert captured["context_market"]["open_home_moneyline"] == 120
    assert captured["context_market"]["open_away_moneyline"] == -140
    assert captured["context_market"]["home_moneyline"] == -110

    # Same enrichment payload reused on the final market dict.
    market = result["market"]
    assert market["n_books"] == 2
    assert market["open_home_moneyline"] == 120
    assert market["best_home_ml"] == -105
    assert market["line_shopping_edge_pp"] is not None


def test_predict_live_game_enrichment_soft_fail() -> None:
    """A books outage must not change core outputs — market stays bare."""
    game = _scheduled_game()
    blended_stub = {
        "algorithm": "Unified",
        "blend_mode": "blended",
        "total_score": -55.0,
        "win_probability": 55.0,
        "favorite_side": "home",
        "blended_home_win_probability": 55.0,
        "context_adjustment_pp": 0.0,
    }
    algo_instance = MagicMock()
    algo_instance.calculate_V2.return_value = {"total": -10.0}
    odds_instance = MagicMock()
    odds_instance.analyze2.return_value = {}

    with (
        patch(
            "web.daily_service.resolve_team",
            side_effect=lambda league, abbr, name: (
                ["bos", "boston-celtics"] if abbr == "BOS" else ["mia", "miami-heat"]
            ),
        ),
        patch("web.daily_service.load_live_team_data", return_value=[{"seed": 1}]),
        patch("algo.Algo", return_value=algo_instance),
        patch("odds_calculator.Odds_Calculator", return_value=odds_instance),
        patch(
            "web.live_odds_enrichment.fetch_multi_book_odds",
            side_effect=OSError("books down"),
        ),
        patch("web.daily_service.blend_predictions", return_value=dict(blended_stub)),
        patch(
            "web.daily_service.apply_ensemble_ml",
            side_effect=lambda blended, league, **kw: blended,
        ),
        patch(
            "web.daily_service.ensure_hubacek_in_blend",
            side_effect=lambda blended, **kw: blended,
        ),
        patch("web.daily_service.compute_model_agreement", return_value={}),
        patch("web.daily_service.get_pick_thresholds", return_value={}),
        patch("web.daily_service.official_pick_binary_probs", return_value=(45.0, 55.0)),
        patch("web.daily_service.evaluate_official_picks_for_game", return_value=[]),
    ):
        result = predict_live_game(game)

    market = result["market"]
    assert market["home_moneyline"] == -110
    assert "n_books" not in market
    assert result["model"]["win_probability"] == 55.0


def test_get_daily_slate_isolates_league_and_prewarm_failures() -> None:
    """One league's scoreboard/prewarm/readiness failure must not abort the slate."""
    from web.daily_service import get_daily_slate
    from web.league_profiles import SUPPORTED_LEAGUES

    nba_game = _scheduled_game()
    mlb_game = ScheduledGame(
        league="mlb",
        event_id="401800",
        name="Yankees at Red Sox",
        start_time="2026-07-10T23:00Z",
        status="pre",
        status_detail="Scheduled",
        home_abbr="bos",
        away_abbr="nyy",
        home_name="Boston Red Sox",
        away_name="New York Yankees",
        home_espn_id="2",
        away_espn_id="10",
        market=MarketOdds(
            home_moneyline=-120,
            away_moneyline=100,
            spread=-1.5,
            home_spread_odds=-110,
            away_spread_odds=-110,
        ),
    )

    def fake_scoreboard(league: str, days_ahead: int = 0):
        if league == "nhl":
            raise RuntimeError("nhl scoreboard down")
        if league == "nba":
            return [nba_game]
        if league == "mlb":
            return [mlb_game]
        return []

    predicted = []

    def fake_predict(game, headlines=None):
        predicted.append(game.league)
        return {
            "event_id": game.event_id,
            "league": game.league,
            "league_name": game.league.upper(),
            "matchup": {
                "away": {"name": game.away_name},
                "home": {"name": game.home_name},
            },
            "start_time": game.start_time,
            "eligible_for_official_picks": False,
            "recommendations": [],
            "top_pick": None,
            "model": {},
            "market": {},
            "name": game.name,
            "status": game.status,
            "status_detail": game.status_detail,
            "cutoff_date": "7-10-2026",
            "season_year": "2026",
        }

    def _prewarm_maybe_fail(league, *_a, **_k):
        if league == "nba":
            raise RuntimeError("prewarm boom")

    with (
        patch("web.daily_service.SUPPORTED_LEAGUES", ("nhl", "nba", "mlb")),
        patch("web.daily_service.fetch_scoreboard", side_effect=fake_scoreboard),
        patch("web.daily_service._actionable_games", side_effect=lambda games: games),
        patch(
            "web.daily_service.is_league_ready_for_daily_slate",
            return_value=True,
        ),
        patch(
            "web.daily_service._prewarm_league_models",
            side_effect=_prewarm_maybe_fail,
        ),
        patch("web.daily_service._league_news_headlines", return_value=[]),
        patch("web.daily_service.predict_live_game", side_effect=fake_predict),
        patch("web.daily_service.passes_hubacek_tracked_pick", return_value=False),
        patch(
            "web.daily_service.official_hubacek_thresholds",
            return_value={"min_edge_pp": 0, "min_ev_pct": 0},
        ),
    ):
        slate = get_daily_slate(days_ahead=0)

    assert len(SUPPORTED_LEAGUES) >= 3  # sanity: real registry still intact
    assert {g["league"] for g in slate["games"]} == {"nba", "mlb"}
    assert predicted == ["nba", "mlb"]
    error_leagues = {e["league"] for e in slate["errors"]}
    assert "nhl" in error_leagues
    assert "nba" in error_leagues  # prewarm recorded, games still analyzed
    assert any("prewarm" in (e.get("error") or "").lower() for e in slate["errors"])
    # Public slate errors must not echo raw exception strings.
    assert not any("boom" in (e.get("error") or "").lower() for e in slate["errors"])
    assert not any("scoreboard down" in (e.get("error") or "").lower() for e in slate["errors"])
