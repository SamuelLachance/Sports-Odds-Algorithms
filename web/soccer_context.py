"""Fourth-layer context adjustments for soccer (ESPN public data only).

Signals applied when available (graceful degradation otherwise):
- Recent form (scoreboard ``form`` string, e.g. LWWWD)
- Season style proxies from completed-game stats: possession, shots, direct play,
  pressing intensity (tackles + interceptions)
- Injuries / suspensions (ESPN core injuries API — often empty off-season)
- Venue (neutral-site flag, stadium name)

Not claimed (unavailable reliably from free ESPN endpoints):
- Pre-match formations / lineups (rosters populate post-match only)
- Coach-change flags (roster coach metadata is stale / missing)
- Weather, pitch surface, PPDA
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

from web.espn_client import _fetch_json
from web.league_profiles import POWER_SCOREBOARD_LOOKBACK, get_league_profile, is_soccer_league
from web.live_data import _event_before_cutoff, _parse_cutoff, _score_value
from web.season_games import _fetch_scoreboard_payload, _normalize_abbr

STYLE_LOOKBACK_DAYS = 60
STYLE_SCOREBOARD_STEP = 2
MIN_STYLE_GAMES = 3
MAX_INJURY_IMPACT_PP = 2.5
MAX_FACTOR_SHIFT_PP = 1.8
MAX_TOTAL_SHIFT_PP = 5.0


@dataclass
class TeamStyleProfile:
    games: int = 0
    possession_pct: float | None = None
    shots_per_game: float | None = None
    direct_index: float | None = None
    press_index: float | None = None

    def label(self) -> str | None:
        if self.games < MIN_STYLE_GAMES:
            return None
        tags: list[str] = []
        if self.possession_pct is not None:
            if self.possession_pct >= 55:
                tags.append("possession")
            elif self.possession_pct <= 45:
                tags.append("direct")
        if self.press_index is not None and self.press_index >= 22:
            tags.append("pressing")
        if self.direct_index is not None and self.direct_index >= 0.14:
            tags.append("long-ball")
        return "/".join(tags) if tags else "balanced"


@dataclass
class MatchContextSignals:
    home_form: str | None = None
    away_form: str | None = None
    neutral_site: bool = False
    venue_name: str | None = None
    home_injuries: int = 0
    away_injuries: int = 0
    home_style: TeamStyleProfile | None = None
    away_style: TeamStyleProfile | None = None
    sources: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)


def _stat_value(stats: list[dict[str, Any]], name: str) -> float | None:
    for item in stats:
        if item.get("name") != name:
            continue
        raw = item.get("displayValue")
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def _league_core_slug(league: str) -> str:
    profile = get_league_profile(league)
    return profile["sport_path"].split("/", 1)[1]


def parse_form_score(form: str | None, sample: int = 5) -> float | None:
    """Return average points per recent result (W=1, D=0.5, L=0)."""
    if not form:
        return None
    letters = [ch.upper() for ch in form if ch.upper() in {"W", "D", "L"}]
    if not letters:
        return None
    recent = letters[-sample:]
    points = sum(1.0 if ch == "W" else 0.5 if ch == "D" else 0.0 for ch in recent)
    return points / len(recent)


def _accumulate_style(
    profiles: dict[str, dict[str, float]],
    team_key: str,
    stats: list[dict[str, Any]],
) -> None:
    possession = _stat_value(stats, "possessionPct")
    shots = _stat_value(stats, "totalShots")
    passes = _stat_value(stats, "totalPasses")
    longballs = _stat_value(stats, "totalLongBalls")
    tackles = _stat_value(stats, "effectiveTackles") or _stat_value(stats, "totalTackles")
    interceptions = _stat_value(stats, "interceptions")

    bucket = profiles.setdefault(
        team_key,
        {
            "games": 0.0,
            "possession_sum": 0.0,
            "possession_n": 0.0,
            "shots_sum": 0.0,
            "shots_n": 0.0,
            "direct_sum": 0.0,
            "direct_n": 0.0,
            "press_sum": 0.0,
            "press_n": 0.0,
        },
    )
    bucket["games"] += 1.0
    if possession is not None:
        bucket["possession_sum"] += possession
        bucket["possession_n"] += 1.0
    if shots is not None:
        bucket["shots_sum"] += shots
        bucket["shots_n"] += 1.0
    if passes and passes > 0 and longballs is not None:
        bucket["direct_sum"] += longballs / passes
        bucket["direct_n"] += 1.0
    if tackles is not None or interceptions is not None:
        bucket["press_sum"] += (tackles or 0.0) + (interceptions or 0.0)
        bucket["press_n"] += 1.0


def _finalize_style(raw: dict[str, float]) -> TeamStyleProfile:
    games = int(raw.get("games", 0))
    profile = TeamStyleProfile(games=games)
    if raw.get("possession_n", 0) > 0:
        profile.possession_pct = round(raw["possession_sum"] / raw["possession_n"], 1)
    if raw.get("shots_n", 0) > 0:
        profile.shots_per_game = round(raw["shots_sum"] / raw["shots_n"], 1)
    if raw.get("direct_n", 0) > 0:
        profile.direct_index = round(raw["direct_sum"] / raw["direct_n"], 3)
    if raw.get("press_n", 0) > 0:
        profile.press_index = round(raw["press_sum"] / raw["press_n"], 1)
    return profile


def _parse_completed_event_styles(
    league: str,
    event: dict[str, Any],
    cutoff: datetime,
) -> list[tuple[str, list[dict[str, Any]]]] | None:
    if not _event_before_cutoff(event, cutoff):
        return None
    competition = (event.get("competitions") or [{}])[0]
    status = (competition.get("status") or {}).get("type") or {}
    if not status.get("completed"):
        return None

    rows: list[tuple[str, list[dict[str, Any]]]] = []
    for competitor in competition.get("competitors") or []:
        team = competitor.get("team") or {}
        abbr = _normalize_abbr(league, team.get("abbreviation", ""))
        if not abbr:
            continue
        stats = competitor.get("statistics") or []
        if not stats:
            return None
        if _score_value(competitor) is None:
            return None
        rows.append((abbr, stats))
    return rows or None


@lru_cache(maxsize=16)
def get_team_style_profiles(league: str, cutoff_date: str) -> dict[str, TeamStyleProfile]:
    """Aggregate per-team style metrics from recent completed ESPN scoreboards."""
    league = league.lower()
    if not is_soccer_league(league):
        return {}

    cutoff = _parse_cutoff(cutoff_date)
    lookback = min(POWER_SCOREBOARD_LOOKBACK.get(league, POWER_SCOREBOARD_LOOKBACK["default"]), STYLE_LOOKBACK_DAYS)
    cutoff_day = date(cutoff.year, cutoff.month, cutoff.day)
    raw_profiles: dict[str, dict[str, float]] = {}

    for offset in range(1, lookback + 1, STYLE_SCOREBOARD_STEP):
        day = cutoff_day - timedelta(days=offset)
        try:
            payload = _fetch_scoreboard_payload(league, day)
        except OSError:
            continue
        for event in payload.get("events") or []:
            parsed = _parse_completed_event_styles(league, event, cutoff)
            if not parsed:
                continue
            for team_key, stats in parsed:
                _accumulate_style(raw_profiles, team_key, stats)

    return {key: _finalize_style(raw) for key, raw in raw_profiles.items()}


def _fetch_injury_count(league: str, espn_team_id: str) -> int:
    if not espn_team_id:
        return 0
    slug = _league_core_slug(league)
    url = (
        f"https://sports.core.api.espn.com/v2/sports/soccer/leagues/{slug}"
        f"/teams/{espn_team_id}/injuries"
    )
    try:
        payload = _fetch_json(url, timeout=12)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return 0
    return int(payload.get("count") or 0)


def _fetch_summary_header(league: str, event_id: str) -> dict[str, Any] | None:
    if not event_id:
        return None
    profile = get_league_profile(league)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/summary?event={event_id}"
    )
    try:
        payload = _fetch_json(url, timeout=15)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    competitions = (payload.get("header") or {}).get("competitions") or []
    return competitions[0] if competitions else None


def fetch_match_context_signals(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> MatchContextSignals:
    """Collect match-day context from ESPN (best-effort)."""
    league = league.lower()
    signals = MatchContextSignals()
    if not is_soccer_league(league):
        return signals

    styles = get_team_style_profiles(league, cutoff_date)
    home_key = _normalize_abbr(league, home_abbr)
    away_key = _normalize_abbr(league, away_abbr)
    # ESPN soccer abbr aliases are sparse; also try raw lower abbr.
    signals.home_style = styles.get(home_key) or styles.get(home_abbr.lower())
    signals.away_style = styles.get(away_key) or styles.get(away_abbr.lower())
    if signals.home_style and signals.home_style.games >= MIN_STYLE_GAMES:
        signals.sources.append("season_style")
    else:
        signals.unavailable.append("home_style")
    if signals.away_style and signals.away_style.games >= MIN_STYLE_GAMES:
        if "season_style" not in signals.sources:
            signals.sources.append("season_style")
    else:
        signals.unavailable.append("away_style")

    header = _fetch_summary_header(league, event_id or "")
    if header:
        signals.neutral_site = bool(header.get("neutralSite"))
        venue = header.get("venue") or {}
        signals.venue_name = venue.get("fullName")
        if signals.venue_name or signals.neutral_site:
            signals.sources.append("venue")
        for competitor in header.get("competitors") or []:
            side = competitor.get("homeAway")
            form = competitor.get("form")
            if side == "home":
                signals.home_form = form
            elif side == "away":
                signals.away_form = form
        if signals.home_form or signals.away_form:
            signals.sources.append("form")
        else:
            signals.unavailable.append("form")
    else:
        signals.unavailable.extend(["form", "venue"])

    if home_espn_id:
        signals.home_injuries = _fetch_injury_count(league, home_espn_id)
    if away_espn_id:
        signals.away_injuries = _fetch_injury_count(league, away_espn_id)
    if home_espn_id or away_espn_id:
        if signals.home_injuries or signals.away_injuries:
            signals.sources.append("injuries")
        else:
            signals.unavailable.append("injuries")

    signals.unavailable.extend(
        [
            "formation",
            "coach_change",
            "weather",
            "pitch_surface",
        ]
    )
    return signals


def _style_clash_draw_boost(home: TeamStyleProfile, away: TeamStyleProfile) -> float:
    """Similar styles and moderate shot volumes → slightly higher draw."""
    if home.games < MIN_STYLE_GAMES or away.games < MIN_STYLE_GAMES:
        return 0.0
    if home.possession_pct is None or away.possession_pct is None:
        return 0.0
    poss_gap = abs(home.possession_pct - away.possession_pct)
    if poss_gap > 12:
        return 0.0
    shots_gap = 0.0
    if home.shots_per_game is not None and away.shots_per_game is not None:
        shots_gap = abs(home.shots_per_game - away.shots_per_game)
    if shots_gap > 4:
        return 0.0
    return min(1.2, 0.4 + (12 - poss_gap) * 0.05)


def _style_home_edge(home: TeamStyleProfile, away: TeamStyleProfile) -> float:
    """Possession / shot edge nudges home win probability."""
    if home.games < MIN_STYLE_GAMES or away.games < MIN_STYLE_GAMES:
        return 0.0
    edge = 0.0
    if home.possession_pct is not None and away.possession_pct is not None:
        edge += (home.possession_pct - away.possession_pct) * 0.04
    if home.shots_per_game is not None and away.shots_per_game is not None:
        edge += (home.shots_per_game - away.shots_per_game) * 0.15
    if home.press_index is not None and away.press_index is not None:
        edge += (home.press_index - away.press_index) * 0.03
    return max(-MAX_FACTOR_SHIFT_PP, min(MAX_FACTOR_SHIFT_PP, edge))


def compute_context_adjustments(
    signals: MatchContextSignals,
) -> tuple[float, float, float, list[dict[str, Any]]]:
    """Return (home_shift, draw_shift, away_shift) in percentage points."""
    home_shift = 0.0
    draw_shift = 0.0
    away_shift = 0.0
    factors: list[dict[str, Any]] = []

    home_form = parse_form_score(signals.home_form)
    away_form = parse_form_score(signals.away_form)
    if home_form is not None and away_form is not None:
        delta = (home_form - away_form) * MAX_FACTOR_SHIFT_PP
        delta = max(-MAX_FACTOR_SHIFT_PP, min(MAX_FACTOR_SHIFT_PP, delta))
        home_shift += delta
        away_shift -= delta * 0.85
        draw_shift -= delta * 0.15
        factors.append(
            {
                "key": "form",
                "label": "Recent form",
                "detail": f"{signals.home_form or '?'} vs {signals.away_form or '?'}",
                "home_shift": round(delta, 2),
                "source": "espn_scoreboard",
            }
        )

    if signals.home_style and signals.away_style:
        style_edge = _style_home_edge(signals.home_style, signals.away_style)
        draw_boost = _style_clash_draw_boost(signals.home_style, signals.away_style)
        if abs(style_edge) >= 0.05:
            home_shift += style_edge
            away_shift -= style_edge * 0.7
            draw_shift -= style_edge * 0.3
            factors.append(
                {
                    "key": "style",
                    "label": "Style of play",
                    "detail": (
                        f"{signals.home_style.label() or 'n/a'}"
                        f" vs {signals.away_style.label() or 'n/a'}"
                    ),
                    "home_shift": round(style_edge, 2),
                    "source": "espn_match_stats",
                }
            )
        if draw_boost >= 0.2:
            draw_shift += draw_boost
            home_shift -= draw_boost * 0.45
            away_shift -= draw_boost * 0.45
            factors.append(
                {
                    "key": "style_draw",
                    "label": "Style draw bias",
                    "detail": "Similar possession/shots profiles",
                    "draw_shift": round(draw_boost, 2),
                    "source": "espn_match_stats",
                }
            )

    injury_delta = signals.away_injuries - signals.home_injuries
    if injury_delta:
        per = min(MAX_INJURY_IMPACT_PP, abs(injury_delta) * 0.6)
        signed = per if injury_delta > 0 else -per
        home_shift += signed
        away_shift -= signed * 0.85
        draw_shift -= signed * 0.15
        factors.append(
            {
                "key": "injuries",
                "label": "Injuries listed",
                "detail": f"home {signals.home_injuries} · away {signals.away_injuries}",
                "home_shift": round(signed, 2),
                "source": "espn_core_injuries",
            }
        )

    if signals.neutral_site:
        neutral_home = -1.2
        neutral_draw = 0.5
        home_shift += neutral_home
        draw_shift += neutral_draw
        away_shift += 0.7
        factors.append(
            {
                "key": "venue",
                "label": "Neutral venue",
                "detail": signals.venue_name or "Neutral site",
                "home_shift": neutral_home,
                "draw_shift": neutral_draw,
                "source": "espn_summary",
            }
        )

    total = abs(home_shift) + abs(draw_shift) + abs(away_shift)
    if total > MAX_TOTAL_SHIFT_PP and total > 0:
        scale = MAX_TOTAL_SHIFT_PP / total
        home_shift *= scale
        draw_shift *= scale
        away_shift *= scale
        for factor in factors:
            for key in ("home_shift", "draw_shift", "away_shift"):
                if key in factor:
                    factor[key] = round(float(factor[key]) * scale, 2)

    return home_shift, draw_shift, away_shift, factors


def apply_threeway_context_shifts(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    home_shift: float,
    draw_shift: float,
    away_shift: float,
) -> tuple[float, float, float]:
    """Apply capped percentage-point shifts and renormalize to 100%."""
    home = max(0.5, home_prob + home_shift)
    draw = max(0.5, draw_prob + draw_shift)
    away = max(0.5, away_prob + away_shift)
    total = home + draw + away
    if total <= 0:
        return home_prob, draw_prob, away_prob
    scale = 100.0 / total
    return round(home * scale, 2), round(draw * scale, 2), round(away * scale, 2)


def build_soccer_context_payload(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    *,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> dict[str, Any] | None:
    """Build context layer payload; None when no actionable factors."""
    signals = fetch_match_context_signals(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        event_id=event_id,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
    )
    home_shift, draw_shift, away_shift, factors = compute_context_adjustments(signals)
    if not factors:
        return None

    adj_home, adj_draw, adj_away = apply_threeway_context_shifts(
        home_prob, draw_prob, away_prob, home_shift, draw_shift, away_shift
    )
    return {
        "algorithm": "SoccerContext",
        "source": "espn-public",
        "home_win_probability": adj_home,
        "draw_probability": adj_draw,
        "away_win_probability": adj_away,
        "shifts": {
            "home": round(home_shift, 2),
            "draw": round(draw_shift, 2),
            "away": round(away_shift, 2),
        },
        "factors": factors,
        "signals": {
            "home_form": signals.home_form,
            "away_form": signals.away_form,
            "neutral_site": signals.neutral_site,
            "venue": signals.venue_name,
            "home_injuries": signals.home_injuries,
            "away_injuries": signals.away_injuries,
            "home_style": signals.home_style.label() if signals.home_style else None,
            "away_style": signals.away_style.label() if signals.away_style else None,
            "sources": signals.sources,
            "unavailable": signals.unavailable,
        },
    }
