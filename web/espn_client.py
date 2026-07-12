"""ESPN public API client for live schedules and odds."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from web.league_profiles import (
    LEAGUE_PROFILES,
    get_league_profile,
)

ESPN_ABBR_ALIASES: dict[str, dict[str, str]] = {
    "nba": {"NY": "ny", "NO": "no", "SA": "sa", "GS": "gs", "UTAH": "utah"},
    "nhl": {
        "NJ": "nj", "TB": "tb", "LA": "la", "SJ": "sj", "WSH": "wsh",
        "VGK": "vgk", "CAR": "car", "NSH": "nsh", "UTA": "uta",
    },
    "mlb": {
        "CHW": "chw", "SD": "sd", "SF": "sf", "TB": "tb", "KC": "kc",
        "CWS": "chw", "AZ": "ari", "WSH": "wsh",
    },
    "nfl": {"JAX": "jax", "LA": "la", "LV": "lv", "WSH": "wsh"},
    "cfb": {"OSU": "osu", "USC": "usc", "MIA": "mia"},
    "cbb": {"UConn": "uconn"},
}

LEAGUE_CONFIG = {
    league_id: {
        "sport_path": profile["sport_path"],
        "display": profile["name"],
    }
    for league_id, profile in LEAGUE_PROFILES.items()
}

# Reject ML-sized values dumped into live spread fields (same policy as collectors).
_MAX_LIVE_SPREAD_ABS: dict[str, float] = {
    "mlb": 7.0,
    "nhl": 5.0,
    "nba": 40.0,
    "wnba": 40.0,
    "nfl": 40.0,
    # Align with college closing collectors (scripts/fetch_{cfb,cbb}_odds.py).
    "cfb": 120.0,
    "cbb": 60.0,
}


@dataclass
class MarketOdds:
    away_moneyline: int | None
    home_moneyline: int | None
    draw_moneyline: int | None = None
    spread: float | None = None
    over_under: float | None = None
    provider: str | None = None
    away_spread_odds: int | None = None
    home_spread_odds: int | None = None


@dataclass
class ScheduledGame:
    league: str
    event_id: str
    name: str
    start_time: str
    status: str
    status_detail: str
    away_abbr: str
    home_abbr: str
    away_name: str
    home_name: str
    away_espn_id: str
    home_espn_id: str
    market: MarketOdds


# Soft client-side throttle + shorter timeouts to avoid ESPN rate-limit storms.
# Slot reservation happens under the lock; sleep is outside so parallel
# scoreboard workers can overlap in-flight HTTP after spacing start times.
_MIN_REQUEST_INTERVAL_S = 0.12
_DEFAULT_TIMEOUT_S = 15
_last_request_at = 0.0
_throttle_lock = threading.Lock()


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_REQUEST_INTERVAL_S - (now - _last_request_at)
        if wait > 0:
            _last_request_at = now + wait
        else:
            wait = 0.0
            _last_request_at = now
    if wait > 0:
        time.sleep(wait)


def _retry_after_seconds(exc: urllib.error.HTTPError, attempt: int) -> float:
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            return min(max(float(header), 0.5), 30.0)
        except ValueError:
            pass
    # 429 / 5xx: exponential backoff with a small floor.
    return min(0.75 * (2**attempt), 8.0)


def _fetch_json(url: str, timeout: int | None = None, retries: int = 3) -> dict[str, Any]:
    timeout_s = _DEFAULT_TIMEOUT_S if timeout is None else timeout
    last_error: BaseException | None = None
    for attempt in range(max(1, retries)):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
        )
        try:
            _throttle()
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            # Retry rate-limits and transient server errors only.
            if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < retries:
                time.sleep(_retry_after_seconds(exc, attempt))
                continue
            raise
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
        except TimeoutError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise urllib.error.URLError("ESPN request failed")


def _parse_american_odds(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    upper = text.upper()
    if upper in {"EVEN", "PK"}:
        return 100
    if upper in {"OFF", "N/A", "NA"}:
        return None
    try:
        if text.startswith("+"):
            odds = int(text)
        else:
            odds = int(text)
    except ValueError:
        return None
    # ESPN often encodes even money as numeric 0; American odds require |x| >= 100.
    if odds == 0:
        return 100
    if abs(odds) < 100:
        return None
    return odds


def _parse_spread_line(value: str | int | float | None) -> float | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    upper = text.upper()
    if upper in {"PK", "EVEN"}:
        return 0.0
    if upper in {"OFF", "N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_spread(odds_block: dict[str, Any] | None) -> tuple[float | None, int | None, int | None]:
    """Return home consensus spread and per-side spread juice from ESPN odds."""
    if not odds_block:
        return None, None, None

    home_spread = _parse_spread_line(odds_block.get("spread"))
    point_spread = odds_block.get("pointSpread") or {}
    away_close = (point_spread.get("away") or {}).get("close") or {}
    home_close = (point_spread.get("home") or {}).get("close") or {}

    if home_spread is None:
        home_spread = _parse_spread_line(home_close.get("line"))

    away_spread_odds = _parse_american_odds(away_close.get("odds"))
    home_spread_odds = _parse_american_odds(home_close.get("odds"))
    return home_spread, away_spread_odds, home_spread_odds


def _clamp_live_spread(
    league: str,
    home_spread: float | None,
    away_spread_odds: int | None,
    home_spread_odds: int | None,
) -> tuple[float | None, int | None, int | None]:
    """Drop ML/juice dumps and out-of-band handicaps from live spread fields.

    College caps (CFB ≤120) must still reject common juice (−105…−120): those
    magnitudes are American odds, never real point spreads.
    """
    if home_spread is not None and abs(home_spread) >= 100.0:
        return None, None, None
    max_abs = _MAX_LIVE_SPREAD_ABS.get(league.lower())
    if (
        max_abs is not None
        and home_spread is not None
        and abs(home_spread) > max_abs
    ):
        return None, None, None
    return home_spread, away_spread_odds, home_spread_odds


def _extract_moneyline(odds_block: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not odds_block:
        return None, None

    moneyline = odds_block.get("moneyline") or {}
    away_close = (moneyline.get("away") or {}).get("close") or {}
    home_close = (moneyline.get("home") or {}).get("close") or {}
    away_ml = _parse_american_odds(away_close.get("odds"))
    home_ml = _parse_american_odds(home_close.get("odds"))

    if away_ml is None:
        away_ml = _parse_american_odds((odds_block.get("awayTeamOdds") or {}).get("moneyLine"))
    if home_ml is None:
        home_ml = _parse_american_odds((odds_block.get("homeTeamOdds") or {}).get("moneyLine"))

    return away_ml, home_ml


def _extract_draw_moneyline(odds_block: dict[str, Any] | None) -> int | None:
    if not odds_block:
        return None

    moneyline = odds_block.get("moneyline") or {}
    draw_close = (moneyline.get("draw") or {}).get("close") or {}
    draw_ml = _parse_american_odds(draw_close.get("odds"))
    if draw_ml is not None:
        return draw_ml

    return _parse_american_odds((odds_block.get("drawOdds") or {}).get("moneyLine"))


def _format_status(competition: dict[str, Any]) -> tuple[str, str]:
    status = competition.get("status") or {}
    status_type = status.get("type") or {}
    return (
        status_type.get("state") or status_type.get("name") or "unknown",
        status_type.get("shortDetail") or status_type.get("detail") or "",
    )


def _parse_event(event: dict[str, Any], league: str) -> ScheduledGame | None:
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    if not away or not home:
        return None

    odds_block = (competition.get("odds") or [None])[0]
    away_ml, home_ml = _extract_moneyline(odds_block)
    draw_ml = _extract_draw_moneyline(odds_block)
    home_spread, away_spread_odds, home_spread_odds = _clamp_live_spread(
        league, *_extract_spread(odds_block)
    )
    state, detail = _format_status(competition)
    away_team = away.get("team") or {}
    home_team = home.get("team") or {}

    return ScheduledGame(
        league=league,
        event_id=str(event.get("id") or ""),
        name=event.get("name") or "",
        start_time=event.get("date") or "",
        status=state,
        status_detail=detail,
        away_abbr=(away_team.get("abbreviation") or "").upper(),
        home_abbr=(home_team.get("abbreviation") or "").upper(),
        away_name=away_team.get("displayName") or "",
        home_name=home_team.get("displayName") or "",
        away_espn_id=str(away_team.get("id") or ""),
        home_espn_id=str(home_team.get("id") or ""),
        market=MarketOdds(
            away_moneyline=away_ml,
            home_moneyline=home_ml,
            draw_moneyline=draw_ml,
            spread=home_spread,
            over_under=(odds_block or {}).get("overUnder"),
            provider=((odds_block or {}).get("provider") or {}).get("name"),
            away_spread_odds=away_spread_odds,
            home_spread_odds=home_spread_odds,
        ),
    )


class ScoreboardFetchError(RuntimeError):
    """Raised when every ESPN scoreboard request fails (network), not an empty slate."""


def fetch_scoreboard(
    league: str,
    on_date: date | None = None,
    days_ahead: int = 0,
) -> list[ScheduledGame]:
    league = league.lower()
    profile = get_league_profile(league)
    games: list[ScheduledGame] = []
    seen: set[str] = set()
    attempts = 0
    failures = 0

    if on_date is None:
        # Align with daily slate labeling (America/Toronto), not bare UTC date.today().
        from zoneinfo import ZoneInfo

        base = datetime.now(ZoneInfo("America/Toronto")).date()
    else:
        base = on_date
    dates_to_check = [
        base.fromordinal(base.toordinal() + offset) for offset in range(0, days_ahead + 1)
    ]

    for check_date in dates_to_check:
        date_param = check_date.strftime("%Y%m%d")
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{profile['sport_path']}/scoreboard?dates={date_param}"
        )
        attempts += 1
        try:
            payload = _fetch_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            # Soft-fail per date so a timeout on day 0 does not abort days_ahead.
            failures += 1
            continue

        for event in payload.get("events") or []:
            event_id = str(event.get("id") or "")
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            parsed = _parse_event(event, league)
            if parsed:
                games.append(parsed)

    if not games and on_date is None:
        url = f"https://site.api.espn.com/apis/site/v2/sports/{profile['sport_path']}/scoreboard"
        attempts += 1
        try:
            payload = _fetch_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            failures += 1
            payload = None

        if payload is not None:
            for event in payload.get("events") or []:
                event_id = str(event.get("id") or "")
                if not event_id or event_id in seen:
                    continue
                seen.add(event_id)
                parsed = _parse_event(event, league)
                if parsed:
                    games.append(parsed)

    # Distinguish total network failure from a genuinely empty schedule.
    if attempts > 0 and failures == attempts and not games:
        raise ScoreboardFetchError(f"ESPN scoreboard unavailable for {league}")

    return games


_SCHEDULE_CACHE: dict[str, list[dict[str, Any]]] = {}


def fetch_team_schedule(league: str, espn_team_id: str, season: int) -> list[dict[str, Any]]:
    profile = get_league_profile(league)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/teams/{espn_team_id}/schedule?season={season}"
    )
    if url in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[url]
    try:
        payload = _fetch_json(url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []
    events = payload.get("events") or []
    _SCHEDULE_CACHE[url] = events
    return events


def clear_schedule_cache() -> None:
    _SCHEDULE_CACHE.clear()


def iso_to_project_date(iso_value: str) -> str:
    """Convert an ISO kickoff to project cutoff M-D-YYYY in America/Toronto.

    Slate readiness and date labels use Toronto; using UTC here caused
    late-ET / early-UTC games to land on the wrong calendar day.
    """
    from zoneinfo import ZoneInfo

    parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone(ZoneInfo("America/Toronto"))
    return f"{local.month}-{local.day}-{local.year}"


def current_season_year(league: str, cutoff: date) -> int:
    """ESPN season year for the in-progress season on the cutoff date.

    Conventions match ESPN's ``?season=`` parameter:
    - NBA/NHL/CBB/NCAAH: ending calendar year (2025-26 → 2026)
    - NFL/CFB: starting / fall year (2025 season includes Feb 2026 playoffs)
    - WNBA / MLB / soccer: calendar year
    """
    year = cutoff.year
    month = cutoff.month

    # Spring / calendar-year baseball (MLB, NCAA baseball, winter leagues, WBC).
    if league in {
        "mlb",
        "ncaabb",
        "dwl",
        "pwl",
        "vwl",
        "lmp",
        "wbc",
        "mls",
        "epl",
        "laliga",
        "bundesliga",
        "seriea",
        "ligue1",
        "ucl",
        "worldcup",
        "fifa_friendlies",
        "concacaf_wcq",
        "concacaf_gold",
        "concacaf_nations",
        "uefa_euro",
        "uefa_nations",
        "copa_america",
        "wnba",
    }:
        return year

    # NFL: Sep–Feb uses the fall start year (playoffs stay on prior season).
    if league == "nfl":
        return year if month >= 3 else year - 1

    # CFB: Aug–Jan bowls use the fall start year (not CBB's ending-year scheme).
    if league == "cfb":
        return year if month >= 7 else year - 1

    # CBB: Aug+ rolls into the spring ending year (2025-26 → 2026).
    if league == "cbb":
        if month >= 8:
            return year + 1
        return year

    # NBA / NHL / college hockey: Oct+ rolls into the spring ending year.
    if league in {"nba", "nhl", "ncaah", "ncaawh"}:
        if month >= 10:
            return year + 1
        return year

    return year


def prior_season_year(league: str, cutoff: date) -> int:
    return current_season_year(league, cutoff) - 1


def guess_season_years(league: str, cutoff: date) -> list[int]:
    """Return [current, prior] ESPN season years (for tests and legacy callers)."""
    return [current_season_year(league, cutoff), prior_season_year(league, cutoff)]
