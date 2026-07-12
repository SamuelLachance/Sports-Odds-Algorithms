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


def test_parse_american_odds_accepts_float_json_values() -> None:
    """ESPN/core JSON often emits American prices as floats; do not drop them."""
    assert espn_client._parse_american_odds(-110.0) == -110
    assert espn_client._parse_american_odds(150.0) == 150
    assert espn_client._parse_american_odds("-110.0") == -110
    assert espn_client._parse_american_odds("+150.0") == 150
    assert espn_client._parse_american_odds(0.0) == 100
    assert espn_client._parse_american_odds(50.0) is None


def test_parse_american_odds_rejects_invalid_magnitude() -> None:
    """|odds| < 100 (except ESPN 0→EVEN) must not enter the daily slate."""
    assert espn_client._parse_american_odds(50) is None
    assert espn_client._parse_american_odds(-50) is None
    assert espn_client._parse_american_odds("+75") is None
    assert espn_client._parse_american_odds(100) == 100
    assert espn_client._parse_american_odds(-100) == -100
    # Non-finite must fail closed — never OverflowError into the slate.
    assert espn_client._parse_american_odds(float("inf")) is None
    assert espn_client._parse_american_odds(float("-inf")) is None
    assert espn_client._parse_american_odds(float("nan")) is None
    assert espn_client._parse_american_odds("inf") is None


def test_parse_spread_line_accepts_lowercase_pk() -> None:
    assert espn_client._parse_spread_line("pk") == 0.0
    assert espn_client._parse_spread_line("PK") == 0.0
    assert espn_client._parse_spread_line("even") == 0.0


def test_parse_spread_line_rejects_nan_and_inf() -> None:
    """Non-finite spreads must not slip onto the live slate via abs(nan) checks."""
    assert espn_client._parse_spread_line(float("nan")) is None
    assert espn_client._parse_spread_line("nan") is None
    assert espn_client._parse_spread_line(float("inf")) is None
    assert espn_client._parse_spread_line("-inf") is None
    clamped = espn_client._clamp_live_spread(
        "nba", espn_client._parse_spread_line(float("nan")), -110, -110
    )
    assert clamped == (None, -110, -110)


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


def test_live_cfb_cbb_keep_blowout_spreads() -> None:
    """Live college caps must match collectors so real blowouts are not dropped."""
    event = {
        "id": "1",
        "name": "Away @ Home",
        "date": "2026-09-12T17:00Z",
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "away",
                        "team": {
                            "abbreviation": "UGA",
                            "displayName": "Georgia",
                            "id": "1",
                        },
                    },
                    {
                        "homeAway": "home",
                        "team": {
                            "abbreviation": "ALA",
                            "displayName": "Alabama",
                            "id": "2",
                        },
                    },
                ],
                "odds": [
                    {
                        "spread": -55.5,
                        "moneyline": {
                            "away": {"close": {"odds": 1800}},
                            "home": {"close": {"odds": -5000}},
                        },
                    }
                ],
                "status": {"type": {"state": "pre", "shortDetail": "Scheduled"}},
            }
        ],
    }
    cfb = espn_client._parse_event(event, "cfb")
    assert cfb is not None and cfb.market.spread == -55.5

    event["competitions"][0]["odds"][0]["spread"] = -45.5
    cbb = espn_client._parse_event(event, "cbb")
    assert cbb is not None and cbb.market.spread == -45.5

    # ML-sized dumps still rejected even with the raised college caps.
    event["competitions"][0]["odds"][0]["spread"] = -152
    cfb_ml = espn_client._parse_event(event, "cfb")
    assert cfb_ml is not None and cfb_ml.market.spread is None

    # Common juice (−110) must not slip through CFB's raised 120 cap.
    event["competitions"][0]["odds"][0]["spread"] = -110
    cfb_juice = espn_client._parse_event(event, "cfb")
    assert cfb_juice is not None and cfb_juice.market.spread is None
    event["competitions"][0]["odds"][0]["spread"] = -105
    cfb_soft = espn_client._parse_event(event, "cfb")
    assert cfb_soft is not None and cfb_soft.market.spread is None


def test_fetch_scoreboard_default_date_uses_toronto_not_utc() -> None:
    """Default scoreboard window must match America/Toronto slate labeling."""
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    # 04:00 UTC on Jan 15 = still Jan 14 evening in America/Toronto (EST).
    frozen_utc = datetime(2026, 1, 15, 4, 0, tzinfo=ZoneInfo("UTC"))
    captured: list[str] = []

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return frozen_utc.astimezone(tz) if tz is not None else frozen_utc

    def fake_fetch(url: str, *args, **kwargs):
        if "dates=" in url:
            captured.append(url.split("dates=")[1].split("&")[0])
        return {"events": []}

    with (
        patch("web.espn_client.datetime", _FrozenDateTime),
        patch.object(espn_client, "_fetch_json", side_effect=fake_fetch),
    ):
        espn_client.fetch_scoreboard("nba", days_ahead=0)

    assert captured == ["20260114"]


def test_event_before_cutoff_includes_late_et_prior_toronto_day() -> None:
    """Late-ET games must use Toronto calendar day, not UTC midnight cutoff."""
    from web.espn_client import iso_to_project_date
    from web.live_data import _event_before_cutoff, _parse_cutoff

    # Feb 8 10pm ET = Feb 9 03:00Z → Toronto date 2-8-2025
    assert iso_to_project_date("2025-02-09T03:00:00Z") == "2-8-2025"

    event = {
        "date": "2025-02-09T03:00:00Z",
        "competitions": [{"status": {"type": {"completed": True}}}],
    }
    cutoff = _parse_cutoff("2-9-2025")
    assert _event_before_cutoff(event, cutoff) is True

    # Same Toronto day as cutoff must be excluded (strictly before).
    same_day = {
        "date": "2025-02-09T18:00:00Z",
        "competitions": [{"status": {"type": {"completed": True}}}],
    }
    assert _event_before_cutoff(same_day, cutoff) is False


def test_parse_cutoff_accepts_iso_and_slate_labels() -> None:
    """NFL/CFB v2 live passes YYYY-MM-DD; legacy slate paths use M-D-YYYY."""
    from web.live_data import _parse_cutoff

    iso = _parse_cutoff("2025-09-15")
    assert (iso.year, iso.month, iso.day) == (2025, 9, 15)

    slate = _parse_cutoff("9-15-2025")
    assert (slate.year, slate.month, slate.day) == (2025, 9, 15)

    padded = _parse_cutoff("09-15-2025")
    assert (padded.year, padded.month, padded.day) == (2025, 9, 15)

    # Previously ISO was misread as month=15 → ValueError, so ESPN fetch never ran.
    assert _parse_cutoff("2024-12-01").month == 12


def test_fetch_team_schedule_does_not_cache_empty_payload() -> None:
    """Empty ESPN schedules must not poison the process-lifetime schedule cache."""
    from unittest.mock import patch

    espn_client.clear_schedule_cache()
    url_suffix = "teams/1/schedule?season=2026"
    with patch.object(espn_client, "_fetch_json", return_value={"events": []}):
        first = espn_client.fetch_team_schedule("nba", "1", 2026)
    assert first == []
    assert not any(url_suffix in key for key in espn_client._SCHEDULE_CACHE)

    payload = {"events": [{"id": "401"}]}
    with patch.object(espn_client, "_fetch_json", return_value=payload) as fetch:
        second = espn_client.fetch_team_schedule("nba", "1", 2026)
    assert second == payload["events"]
    assert fetch.call_count == 1
    # Non-empty result is cached for subsequent callers.
    with patch.object(espn_client, "_fetch_json", side_effect=AssertionError("cache miss")):
        third = espn_client.fetch_team_schedule("nba", "1", 2026)
    assert third == payload["events"]
    espn_client.clear_schedule_cache()
