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
