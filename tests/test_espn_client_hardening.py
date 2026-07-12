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
    assert hasattr(espn_client, "_throttle_lock")
    espn_client._throttle()
    espn_client._throttle()
