"""Track algo recommended bets, grade results, and build performance rollups."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from web.bet_advisor import spread_line_for_side
from web.hubacek_picks import (
    official_hubacek_thresholds,
    passes_hubacek_tracked_pick,
)
from web.league_profiles import (
    eligible_for_official_picks,
    is_soccer_league,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_FILE = PROJECT_ROOT / "data" / "tracking.json"
TIMEZONE_LABEL = "America/Toronto"
TORONTO = ZoneInfo(TIMEZONE_LABEL)


def toronto_today() -> date:
    return datetime.now(TORONTO).date()

BetResult = Literal["pending", "win", "loss", "push"]

# Stake sizing: quarter-Kelly, expressed in units where 1u = 1% of bankroll.
MIN_STAKE_UNITS = 0.25
MAX_STAKE_UNITS = 3.0
DEFAULT_STAKE_UNITS = 1.0


def stake_units_from_kelly(
    kelly_pct: float | None,
    *,
    ev_pct: float | None = None,
    correlation_penalty: float = 0.0,
) -> float:
    """Quarter-Kelly stake in units (1u = 1% bankroll), clamped to 0.25–3u.

    Optionally routes through ``portfolio_stake_units`` for a correlation haircut
    and EV soft dampener. Default ``correlation_penalty=0`` preserves classic
    quarter-Kelly sizing; pass ~0.15 when sizing many same-slate correlated bets.
    """
    if kelly_pct is None:
        return DEFAULT_STAKE_UNITS
    try:
        kelly_frac = float(kelly_pct) / 100.0
    except (TypeError, ValueError):
        return DEFAULT_STAKE_UNITS
    if kelly_frac <= 0:
        return DEFAULT_STAKE_UNITS

    from web.portfolio_sizing import portfolio_stake_units

    return portfolio_stake_units(
        float(ev_pct) if ev_pct is not None else 0.0,
        kelly_frac,
        bankroll_units=100.0,
        max_units=MAX_STAKE_UNITS,
        min_units=MIN_STAKE_UNITS,
        correlation_penalty=correlation_penalty,
    )


def calculate_units(stake: float, american_odds: int, result: BetResult) -> float:
    try:
        stake_f = float(stake)
    except (TypeError, ValueError):
        stake_f = DEFAULT_STAKE_UNITS
    if stake_f < 0:
        stake_f = DEFAULT_STAKE_UNITS

    if result == "push":
        return 0.0
    if result == "loss":
        return -stake_f
    try:
        odds = int(american_odds)
    except (TypeError, ValueError):
        return 0.0
    # ESPN EVEN (0) → even money; other |odds| < 100 are invalid → 0u on win.
    if odds == 0:
        return stake_f
    if abs(odds) < 100:
        return 0.0
    if odds > 0:
        return stake_f * (odds / 100.0)
    return stake_f * (100.0 / abs(odds))


def _empty_store() -> dict[str, Any]:
    return {"version": 1, "bets": []}


def _normalize_store(store: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure a tracking store always has a list ``bets`` (empty seasons / corrupt file).

    Also collapses legacy duplicate rows that share the same event/side/type key
    so rollups never double-count P&L from older date-keyed recordings.
    """
    if not isinstance(store, dict):
        return _empty_store()
    bets = store.get("bets")
    if not isinstance(bets, list):
        bets = []
    # Drop non-dict rows so rollups never crash on corrupt entries.
    bets = [b for b in bets if isinstance(b, dict)]
    bets = _dedupe_bets_by_key(bets)
    version = store.get("version", 1)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = 1
    return {"version": version, "bets": bets}


def _bet_preference_rank(bet: dict[str, Any]) -> tuple[int, str]:
    """Prefer settled rows over pending, then newest ``recorded_at``."""
    status = str(bet.get("status") or "pending").lower()
    decided = 1 if status in {"win", "loss", "push"} else 0
    stamp = str(bet.get("recorded_at") or bet.get("closing_snapshot_at") or bet.get("date") or "")
    return (decided, stamp)


def _dedupe_bets_by_key(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per ``event_id:side[:bet_type]``; preserve insertion order of winners."""
    chosen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    orphans: list[dict[str, Any]] = []
    for bet in bets:
        event_id = str(bet.get("event_id") or "")
        side = str(bet.get("side") or "")
        if not event_id or not side:
            orphans.append(bet)
            continue
        key = _bet_key(event_id, side, bet.get("bet_type") or "moneyline")
        prior = chosen.get(key)
        if prior is None:
            chosen[key] = bet
            order.append(key)
            continue
        if _bet_preference_rank(bet) > _bet_preference_rank(prior):
            chosen[key] = bet
    return [chosen[k] for k in order] + orphans


def load_store() -> dict[str, Any]:
    if not TRACKING_FILE.exists():
        return _empty_store()
    try:
        return _normalize_store(json.loads(TRACKING_FILE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return _empty_store()


def save_store(store: dict[str, Any]) -> None:
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_store(store)
    TRACKING_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def _bet_key(event_id: str, side: str, bet_type: str = "moneyline") -> str:
    """Stable identity for a tracked pick (no slate calendar date).

    Horizon games can appear on consecutive daily builds with different
    ``date_label`` values; including the slate day would duplicate pending
    rows and double-count P&L after grading.
    """
    if bet_type == "moneyline":
        return f"{event_id}:{side}"
    return f"{event_id}:{side}:{bet_type}"


def _grading_spread_line(bet: dict[str, Any]) -> float | None:
    consensus = bet.get("consensus_spread")
    if consensus is not None:
        return spread_line_for_side(float(consensus), bet["side"])
    if bet.get("spread_line") is not None:
        return float(bet["spread_line"])
    return None


def _resolve_grading_odds(bet: dict[str, Any]) -> int | None:
    bet_type = bet.get("bet_type") or "moneyline"
    if bet_type == "spread":
        # Same priority as _recorded_and_closing_odds (CLV): posted spread juice
        # first. Do not use `or` — ESPN EVEN is 0 (falsy but valid).
        for key in ("spread_odds", "consensus_odds", "market_odds"):
            value = bet.get(key)
            if value is not None:
                return int(value)
        # Missing posted juice — leave ungraded (do not invent -110).
        return None
    market_odds = bet.get("market_odds")
    if market_odds is None:
        return None
    return int(market_odds)


def _parse_date_label(value: str) -> date:
    return date.fromisoformat(value)


def _parse_start_time(value: Any) -> datetime | None:
    """Parse an ISO 8601 kickoff time ("2026-07-10T19:00Z") to aware UTC, else None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _consensus_closing_ml(game: dict[str, Any] | None, side: str) -> int | None:
    """Multi-book consensus closing moneyline from a slate game's market dict."""
    if not isinstance(game, dict):
        return None
    market = game.get("market")
    if not isinstance(market, dict):
        return None
    key = {"home": "consensus_home_ml", "away": "consensus_away_ml"}.get(side)
    value = market.get(key) if key else None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _official_tracked_bets(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in bets if eligible_for_official_picks(b.get("league") or "")]


def record_from_slate(store: dict[str, Any], slate: dict[str, Any]) -> dict[str, Any]:
    """Log Hubáček official picks from the daily slate.

    Odds and model numbers are frozen at record time; later slate runs only
    refresh a closing-line snapshot on still-pending bets (for CLV grading).
    New bets are only created pre-kickoff — a pick first seen after its game
    started was never actionable and is skipped (pending bets still update).
    """
    store = _normalize_store(store)
    date_label = slate.get("date_label") or toronto_today().isoformat()
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    # Index by event/side/type only so a pick seen on consecutive slate days
    # updates closing odds instead of creating a second pending row.
    index: dict[str, dict[str, Any]] = {}
    for b in store["bets"]:
        if not b.get("event_id") or not b.get("side"):
            continue
        key = _bet_key(
            str(b.get("event_id") or ""),
            str(b.get("side") or ""),
            b.get("bet_type") or "moneyline",
        )
        prior = index.get(key)
        if prior is None:
            index[key] = b
            continue
        # Prefer settled over pending so a graded row is never reopened.
        if _bet_preference_rank(b) > _bet_preference_rank(prior):
            index[key] = b
    games_by_event = {
        str(g.get("event_id")): g
        for g in slate.get("games") or []
        if isinstance(g, dict) and g.get("event_id")
    }

    for pick in slate.get("recommended_bets") or []:
        if not eligible_for_official_picks(pick.get("league") or ""):
            continue
        if not passes_hubacek_tracked_pick(pick):
            continue
        event_id = pick.get("event_id")
        side = pick.get("side")
        if not event_id or not side:
            continue

        bet_type = pick.get("bet_type") or "moneyline"
        key = _bet_key(event_id, side, bet_type)
        existing = index.get(key)
        if existing:
            if existing.get("status") == "pending":
                closing_ml = pick.get("market_odds")
                closing_source = "espn"
                consensus_ml = _consensus_closing_ml(games_by_event.get(str(event_id)), side)
                if bet_type != "spread" and consensus_ml is not None:
                    closing_ml = consensus_ml
                    closing_source = "consensus"
                existing.update(
                    {
                        "closing_market_odds": closing_ml,
                        "closing_spread_odds": pick.get("spread_odds"),
                        "closing_consensus_spread": pick.get("consensus_spread"),
                        "closing_source": closing_source,
                        "closing_snapshot_at": now,
                    }
                )
            continue

        start_time = _parse_start_time(pick.get("start_time"))
        if start_time is not None and start_time <= now_dt:
            # First seen post-kickoff: never actionable, no closing snapshot possible.
            continue

        bet = {
            "id": key,
            "date": date_label,
            "event_id": event_id,
            "league": pick.get("league"),
            "league_name": pick.get("league_name"),
            "side": side,
            "team_name": pick.get("team_name"),
            "team_slug": pick.get("team_slug"),
            "team_abbr": pick.get("team_abbr"),
            "matchup": pick.get("matchup"),
            "start_time": pick.get("start_time"),
            "strategy": pick.get("strategy"),
            "strategy_label": pick.get("strategy_label"),
            "confidence": pick.get("confidence"),
            "edge": pick.get("edge"),
            "ev_pct": pick.get("ev_pct"),
            "model_projection": pick.get("model_projection"),
            "market_odds": pick.get("market_odds"),
            "win_probability": pick.get("win_probability"),
            "reason": pick.get("reason"),
            "bet_type": bet_type,
            "spread_line": pick.get("spread_line"),
            "spread_odds": pick.get("spread_odds"),
            "consensus_spread": pick.get("consensus_spread"),
            "consensus_odds": pick.get("consensus_odds"),
            "consensus_label": pick.get("consensus_label"),
            "model_margin": pick.get("model_margin"),
            "model_market_gap_pp": pick.get("model_market_gap_pp"),
            "kelly_pct": pick.get("kelly_pct"),
            "status": "pending",
            "units": 0.0,
            "stake_units": stake_units_from_kelly(
                pick.get("kelly_pct"),
                ev_pct=pick.get("ev_pct"),
            ),
            "recorded_at": now,
            "recorded_pre_start": True,
        }
        store["bets"].append(bet)
        index[key] = bet

    return store


def _scores_from_event(event: dict[str, Any]) -> tuple[int | None, int | None, bool]:
    competition = (event.get("competitions") or [{}])[0]
    status = (competition.get("status") or {}).get("type") or {}
    completed = bool(status.get("completed")) or status.get("state") in {"post"}
    competitors = competition.get("competitors") or []
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    if not away or not home:
        return None, None, False

    def score(comp: dict[str, Any]) -> int | None:
        block = comp.get("score")
        if isinstance(block, dict):
            raw = block.get("value")
        else:
            raw = block
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    return score(away), score(home), completed


def _scoreboard_dates_for_bet(bet: dict[str, Any]) -> list[date]:
    """Dates to scan on ESPN scoreboards when direct event lookup fails."""
    dates: list[date] = []
    start_time = bet.get("start_time")
    if start_time:
        try:
            parsed = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            center = parsed.astimezone(TORONTO).date()
            for offset in (-1, 0, 1):
                candidate = center.fromordinal(center.toordinal() + offset)
                if candidate not in dates:
                    dates.append(candidate)
        except (ValueError, TypeError):
            pass

    try:
        recorded = _parse_date_label(bet["date"])
        if recorded not in dates:
            dates.append(recorded)
    except (KeyError, ValueError):
        pass
    return dates


def _fetch_event_result(league: str, event_id: str) -> tuple[int, int] | None:
    """Fetch final scores for a single ESPN event (works across calendar dates)."""
    from web.espn_client import _fetch_json, get_league_profile

    profile = get_league_profile(league)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/summary?event={event_id}"
    )
    try:
        payload = _fetch_json(url)
    except Exception:
        return None

    competitions = (payload.get("header") or {}).get("competitions") or []
    if not competitions:
        return None

    away_score, home_score, completed = _scores_from_event(
        {"competitions": competitions}
    )
    if not completed or away_score is None or home_score is None:
        return None
    return away_score, home_score


def _lookup_scoreboard_result(
    bet: dict[str, Any],
    cache: dict[tuple[str, date], dict[str, tuple[int, int]]],
) -> tuple[int, int] | None:
    league = bet.get("league") or ""
    event_id = bet.get("event_id") or ""
    if not league or not event_id:
        return None

    for on_date in _scoreboard_dates_for_bet(bet):
        cache_key = (league, on_date)
        if cache_key not in cache:
            cache[cache_key] = _fetch_results_by_event(league, on_date)
        scores = cache[cache_key].get(event_id)
        if scores:
            return scores
    return None


def _fetch_results_by_event(league: str, on_date: date) -> dict[str, tuple[int, int]]:
    from web.espn_client import _fetch_json, get_league_profile

    profile = get_league_profile(league)
    date_param = on_date.strftime("%Y%m%d")
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/scoreboard?dates={date_param}"
    )
    try:
        payload = _fetch_json(url)
    except Exception:
        return {}

    results: dict[str, tuple[int, int]] = {}
    for event in payload.get("events") or []:
        away_score, home_score, completed = _scores_from_event(event)
        if not completed or away_score is None or home_score is None:
            continue
        event_id = str(event.get("id") or "")
        if event_id:
            results[event_id] = (away_score, home_score)
    return results


def _grade_spread_bet(
    side: str,
    spread: float,
    away_score: int,
    home_score: int,
) -> BetResult:
    if side == "home":
        adjusted = home_score + spread
        if adjusted == away_score:
            return "push"
        return "win" if adjusted > away_score else "loss"

    adjusted = away_score + spread
    if adjusted == home_score:
        return "push"
    return "win" if adjusted > home_score else "loss"


def _recorded_and_closing_odds(bet: dict[str, Any]) -> tuple[int | None, int | None]:
    bet_type = bet.get("bet_type") or "moneyline"
    if bet_type == "spread":
        recorded = _resolve_grading_odds(bet)
        closing = bet.get("closing_spread_odds")
    else:
        recorded = bet.get("market_odds")
        closing = bet.get("closing_market_odds")
    try:
        rec = int(recorded) if recorded is not None else None
        close = int(closing) if closing is not None else None
    except (TypeError, ValueError):
        return None, None
    return rec, close


def _clv_payout_pct(recorded: int, closing: int) -> float | None:
    """Legacy decimal-payout CLV: (recorded_dec / closing_dec - 1) * 100."""
    from web.bet_advisor import american_to_decimal

    try:
        recorded_decimal = american_to_decimal(int(recorded))
        closing_decimal = american_to_decimal(int(closing))
    except (TypeError, ValueError):
        return None
    if closing_decimal <= 1.0:
        return None
    return round((recorded_decimal / closing_decimal - 1.0) * 100.0, 2)


def _clv_pct(bet: dict[str, Any]) -> float | None:
    """Industry-standard implied-probability CLV vs closing price."""
    from web.clv_service import clv_vs_market_pct

    recorded, closing = _recorded_and_closing_odds(bet)
    if recorded is None or closing is None:
        return None
    return clv_vs_market_pct(recorded, closing)


def _clv_fields(bet: dict[str, Any]) -> dict[str, float]:
    """Return clv_pct (implied-prob) and clv_payout_pct (legacy decimal ratio)."""
    recorded, closing = _recorded_and_closing_odds(bet)
    if recorded is None or closing is None:
        return {}
    from web.clv_service import clv_vs_market_pct

    fields: dict[str, float] = {}
    implied = clv_vs_market_pct(recorded, closing)
    if implied is not None:
        fields["clv_pct"] = implied
    payout = _clv_payout_pct(recorded, closing)
    if payout is not None:
        fields["clv_payout_pct"] = payout
    return fields


def grade_bet(
    bet: dict[str, Any],
    away_score: int,
    home_score: int,
) -> dict[str, Any]:
    bet_type = bet.get("bet_type") or "moneyline"
    side = bet["side"]

    if bet_type == "spread":
        spread = _grading_spread_line(bet)
        if spread is None:
            return bet
        status = _grade_spread_bet(side, spread, away_score, home_score)
    elif side == "draw":
        status = "win" if away_score == home_score else "loss"
    elif away_score == home_score:
        if is_soccer_league(bet.get("league") or ""):
            status = "loss"
        else:
            status = "push"
    elif side == "away":
        status = "win" if away_score > home_score else "loss"
    else:
        status = "win" if home_score > away_score else "loss"

    odds = _resolve_grading_odds(bet)
    if odds is None:
        return bet
    units = calculate_units(_bet_stake_units(bet), odds, status)
    graded = {
        **bet,
        "status": status,
        "units": units,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "final_score": f"{away_score}–{home_score}",
    }
    graded.update(_clv_fields(bet))
    return graded


def grade_pending(store: dict[str, Any]) -> dict[str, Any]:
    store = _normalize_store(store)
    pending = [
        b
        for b in store["bets"]
        if b.get("status") == "pending" and eligible_for_official_picks(b.get("league") or "")
    ]
    if not pending:
        return store

    scoreboard_cache: dict[tuple[str, date], dict[str, tuple[int, int]]] = {}
    for bet in pending:
        event_id = bet.get("event_id")
        league = bet.get("league")
        if not event_id or not league:
            continue

        scores = _fetch_event_result(league, event_id)
        if not scores:
            scores = _lookup_scoreboard_result(bet, scoreboard_cache)
        if not scores:
            continue

        graded = grade_bet(bet, scores[0], scores[1])
        bet_id = bet.get("id")
        if not bet_id:
            continue
        idx = next(
            (i for i, b in enumerate(store["bets"]) if b.get("id") == bet_id),
            None,
        )
        if idx is None:
            continue
        store["bets"][idx] = graded

    return store


def _rollup_label_daily(d: date) -> str:
    return d.isoformat()


def _rollup_label_weekly(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _rollup_label_monthly(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def _rollup_label_yearly(d: date) -> str:
    return str(d.year)


def _bet_stake_units(bet: dict[str, Any]) -> float:
    """Stake in units for a bet, defaulting to 1u on missing/bad/non-positive data."""
    try:
        stake = float(bet.get("stake_units") if bet.get("stake_units") is not None else DEFAULT_STAKE_UNITS)
    except (TypeError, ValueError):
        return DEFAULT_STAKE_UNITS
    if stake <= 0 or stake != stake:  # NaN check
        return DEFAULT_STAKE_UNITS
    return stake


def _bet_units(bet: dict[str, Any]) -> float:
    """Settled P&L units; coerce bad values to 0 so rollups never raise."""
    try:
        value = float(bet.get("units") or 0)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return value


def _summarize_bets(bets: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up a bet list. Empty seasons / push-only windows yield ROI 0, never /0."""
    bets = [b for b in bets if isinstance(b, dict)]
    wins = sum(1 for b in bets if b.get("status") == "win")
    losses = sum(1 for b in bets if b.get("status") == "loss")
    pushes = sum(1 for b in bets if b.get("status") == "push")
    pending = sum(1 for b in bets if b.get("status") == "pending")
    settled_units = sum(_bet_units(b) for b in bets if b.get("status") != "pending")
    # ROI per staked unit: pushes return the stake, so only win/loss stakes count.
    staked_units = sum(_bet_stake_units(b) for b in bets if b.get("status") in ("win", "loss"))
    if staked_units > 0:
        roi = settled_units / staked_units * 100.0
    else:
        roi = 0.0
    return {
        "bets": len(bets),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": pending,
        "units": round(settled_units, 3),
        "staked_units": round(staked_units, 2),
        "roi_percent": round(roi, 2),
        "record": f"{wins}-{losses}" + (f"-{pushes}" if pushes else ""),
    }


def build_period_rollups(store: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    store = _normalize_store(store)
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {
        "daily": {},
        "weekly": {},
        "monthly": {},
        "yearly": {},
    }

    for bet in _official_tracked_bets(store["bets"]):
        raw_date = bet.get("date")
        if not raw_date:
            continue
        try:
            d = _parse_date_label(str(raw_date))
        except ValueError:
            continue
        for period, key in [
            ("daily", _rollup_label_daily(d)),
            ("weekly", _rollup_label_weekly(d)),
            ("monthly", _rollup_label_monthly(d)),
            ("yearly", _rollup_label_yearly(d)),
        ]:
            buckets[period].setdefault(key, []).append(bet)

    result: dict[str, list[dict[str, Any]]] = {}
    for period, groups in buckets.items():
        rows = []
        for key in sorted(groups.keys(), reverse=True):
            summary = _summarize_bets(groups[key])
            rows.append({"key": key, "label": key, **summary})
        result[period] = rows
    return result


def build_tracking_response(store: dict[str, Any]) -> dict[str, Any]:
    store = _normalize_store(store)
    official_bets = _official_tracked_bets(store["bets"])
    sorted_bets = sorted(
        official_bets,
        key=lambda b: (b.get("date", ""), -(b.get("edge") or 0)),
        reverse=True,
    )
    all_time = _summarize_bets(official_bets)
    periods = build_period_rollups(store)
    tracking_since = sorted_bets[-1].get("date") if sorted_bets else None

    return {
        "bets": sorted_bets,
        "summary": all_time,
        "daily": periods["daily"],
        "weekly": periods["weekly"],
        "monthly": periods["monthly"],
        "yearly": periods["yearly"],
        "all_time": all_time,
        "tracking_since": tracking_since,
        "timezone": TIMEZONE_LABEL,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Tracks Hubáček official picks (decorrelated model beats the book by "
            "≥2 pp with ≥2% honest EV and a per-bet-type confidence bar) from each "
            "daily slate. Basketball/football spreads graded ATS at the recorded "
            "book spread; hockey/baseball at the recorded moneyline; soccer 1X2 at "
            "the recorded price. Odds are frozen at record time and CLV is logged "
            "at grading. Stakes are quarter-Kelly (0.25–3u, 1u = 1% bankroll)."
        ),
        **official_hubacek_thresholds(),
    }


def prune_below_min_ev(store: dict[str, Any]) -> dict[str, Any]:
    """Drop still-pending bets that no longer qualify. Graded bets are immutable."""
    store = _normalize_store(store)
    store["bets"] = [
        b
        for b in store["bets"]
        if b.get("status") not in (None, "pending") or passes_hubacek_tracked_pick(b)
    ]
    return store


def prune_below_min_edge(store: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias."""
    return prune_below_min_ev(store)


def update_tracking(slate: dict[str, Any]) -> dict[str, Any]:
    store = _normalize_store(load_store())
    store = prune_below_min_edge(store)
    store = record_from_slate(store, slate)
    store = grade_pending(store)
    save_store(store)
    return build_tracking_response(store)
