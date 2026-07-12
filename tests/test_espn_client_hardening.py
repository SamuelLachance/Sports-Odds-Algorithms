"""ESPN client timeout / throttle helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web import espn_client  # noqa: E402


def test_default_timeout_is_hardened() -> None:
    assert espn_client._DEFAULT_TIMEOUT_S <= 15
    assert espn_client._MIN_REQUEST_INTERVAL_S > 0


def test_retry_after_prefers_header() -> None:
    exc = MagicMock()
    exc.headers = {"Retry-After": "2.5"}
    assert espn_client._retry_after_seconds(exc, 0) == 2.5


def test_retry_after_falls_back_to_backoff() -> None:
    exc = MagicMock()
    exc.headers = {}
    assert espn_client._retry_after_seconds(exc, 0) == 0.75
    assert espn_client._retry_after_seconds(exc, 1) == 1.5


def test_throttle_is_threadsafe_callable() -> None:
    """Throttle lock exists so parallel scoreboard workers can space starts."""
    from unittest.mock import patch

    assert hasattr(espn_client, "_throttle_lock")
    # Avoid real sleeps so the suite stays quiet and non-flaky under load.
    with patch("web.espn_client.time.sleep"):
        espn_client._throttle()
        espn_client._throttle()


def test_fetch_scoreboard_raises_when_all_requests_fail() -> None:
    """Total ESPN outage must not look like an empty schedule."""
    import urllib.error
    from unittest.mock import patch

    def boom(_url: str):
        raise urllib.error.URLError("timed out")

    with patch("web.espn_client._fetch_json", side_effect=boom):
        try:
            espn_client.fetch_scoreboard("nba", on_date=__import__("datetime").date(2026, 7, 12))
            raised = False
        except espn_client.ScoreboardFetchError:
            raised = True
    assert raised


def test_fetch_scoreboard_soft_fails_timeout_error() -> None:
    """TimeoutError after retries must soft-fail like URLError (not abort the date loop)."""
    from datetime import date
    from unittest.mock import patch

    def boom(_url: str):
        raise TimeoutError("espn timed out")

    with patch("web.espn_client._fetch_json", side_effect=boom):
        try:
            espn_client.fetch_scoreboard("nba", on_date=date(2026, 7, 12))
            raised = False
        except espn_client.ScoreboardFetchError:
            raised = True
    assert raised


def test_fetch_scoreboard_soft_fails_json_decode_error() -> None:
    import json
    from datetime import date
    from unittest.mock import patch

    def boom(_url: str):
        raise json.JSONDecodeError("bad", "doc", 0)

    with patch("web.espn_client._fetch_json", side_effect=boom):
        try:
            espn_client.fetch_scoreboard("nhl", on_date=date(2026, 7, 12))
            raised = False
        except espn_client.ScoreboardFetchError:
            raised = True
    assert raised


def test_fetch_scoreboard_empty_payload_is_not_a_fetch_error() -> None:
    """A successful empty scoreboard is a real empty slate, not ScoreboardFetchError."""
    from datetime import date
    from unittest.mock import patch

    with patch("web.espn_client._fetch_json", return_value={"events": []}):
        games = espn_client.fetch_scoreboard("nba", on_date=date(2026, 7, 12))
    assert games == []


def test_parse_american_odds_maps_even_zero_to_plus_100() -> None:
    """ESPN numeric 0 (even money) must become +100, not invalid American 0."""
    assert espn_client._parse_american_odds(0) == 100
    assert espn_client._parse_american_odds("EVEN") == 100
    assert espn_client._parse_american_odds("even") == 100
    assert espn_client._parse_american_odds("PK") == 100
    assert espn_client._parse_american_odds("pk") == 100
    assert espn_client._parse_american_odds(-110) == -110
    assert espn_client._parse_american_odds(None) is None


def test_parse_american_odds_rejects_invalid_magnitude() -> None:
    """|odds| < 100 (except ESPN 0→EVEN) must not enter the daily slate."""
    assert espn_client._parse_american_odds(50) is None
    assert espn_client._parse_american_odds(-50) is None
    assert espn_client._parse_american_odds("+75") is None
    assert espn_client._parse_american_odds(100) == 100
    assert espn_client._parse_american_odds(-100) == -100


def test_parse_spread_line_accepts_lowercase_pk() -> None:
    assert espn_client._parse_spread_line("pk") == 0.0
    assert espn_client._parse_spread_line("PK") == 0.0
    assert espn_client._parse_spread_line("even") == 0.0


def test_live_mlb_nhl_reject_moneyline_sized_spreads() -> None:
    """Live scoreboard must drop ML dumps into spread like historical collectors."""
    event = {
        "id": "1",
        "name": "Away @ Home",
        "date": "2026-07-12T17:00Z",
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "away",
                        "team": {
                            "abbreviation": "NYY",
                            "displayName": "Yankees",
                            "id": "1",
                        },
                    },
                    {
                        "homeAway": "home",
                        "team": {
                            "abbreviation": "BOS",
                            "displayName": "Red Sox",
                            "id": "2",
                        },
                    },
                ],
                "odds": [
                    {
                        "spread": -152,
                        "moneyline": {
                            "away": {"close": {"odds": 140}},
                            "home": {"close": {"odds": -160}},
                        },
                    }
                ],
                "status": {"type": {"state": "pre", "shortDetail": "Scheduled"}},
            }
        ],
    }
    mlb = espn_client._parse_event(event, "mlb")
    nhl = espn_client._parse_event(event, "nhl")
    assert mlb is not None and mlb.market.spread is None
    assert nhl is not None and nhl.market.spread is None

    event["competitions"][0]["odds"][0]["spread"] = -1.5
    mlb_ok = espn_client._parse_event(event, "mlb")
    assert mlb_ok is not None and mlb_ok.market.spread == -1.5
