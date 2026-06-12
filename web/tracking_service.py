"""Track algo recommended bets, grade results, and build performance rollups."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from web.espn_client import fetch_scoreboard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_FILE = PROJECT_ROOT / "data" / "tracking.json"
TIMEZONE_LABEL = "America/Toronto"

BetResult = Literal["pending", "win", "loss", "push"]


def calculate_units(stake: float, american_odds: int, result: BetResult) -> float:
    if result == "push":
        return 0.0
    if result == "loss":
        return -stake
    if american_odds > 0:
        return stake * (american_odds / 100)
    return stake * (100 / abs(american_odds))


def _empty_store() -> dict[str, Any]:
    return {"version": 1, "bets": []}


def load_store() -> dict[str, Any]:
    if not TRACKING_FILE.exists():
        return _empty_store()
    try:
        return json.loads(TRACKING_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_store()


def save_store(store: dict[str, Any]) -> None:
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACKING_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _bet_key(date_label: str, event_id: str, side: str) -> str:
    return f"{date_label}:{event_id}:{side}"


def _parse_date_label(value: str) -> date:
    return date.fromisoformat(value)


def record_from_slate(store: dict[str, Any], slate: dict[str, Any]) -> dict[str, Any]:
    """Log value bets from the daily slate (edge > 0)."""
    date_label = slate.get("date_label") or date.today().isoformat()
    now = datetime.now(timezone.utc).isoformat()
    index = {_bet_key(b["date"], b["event_id"], b["side"]): b for b in store["bets"]}

    seen: set[str] = set()
    picks: list[dict[str, Any]] = []

    for pick in slate.get("recommended_bets") or []:
        if (pick.get("edge") or 0) > 0:
            picks.append(pick)

    for game in slate.get("games") or []:
        for pick in game.get("recommendations") or []:
            if (pick.get("edge") or 0) <= 0:
                continue
            enriched = {
                **pick,
                "league": game["league"],
                "league_name": game["league_name"],
                "event_id": game["event_id"],
                "matchup": (
                    f"{game['matchup']['away']['name']} @ {game['matchup']['home']['name']}"
                ),
                "start_time": game.get("start_time"),
            }
            key = f"{enriched['event_id']}:{enriched['side']}"
            if key not in seen:
                picks.append(enriched)
                seen.add(key)

    for pick in picks:
        if (pick.get("edge") or 0) <= 0:
            continue
        event_id = pick.get("event_id")
        side = pick.get("side")
        if not event_id or not side:
            continue

        key = _bet_key(date_label, event_id, side)
        existing = index.get(key)
        if existing:
            existing.update(
                {
                    "edge": pick.get("edge"),
                    "strategy": pick.get("strategy"),
                    "strategy_label": pick.get("strategy_label"),
                    "confidence": pick.get("confidence"),
                    "model_projection": pick.get("model_projection"),
                    "market_odds": pick.get("market_odds"),
                    "win_probability": pick.get("win_probability"),
                    "reason": pick.get("reason"),
                    "team_name": pick.get("team_name"),
                    "team_slug": pick.get("team_slug"),
                }
            )
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
            "matchup": pick.get("matchup"),
            "start_time": pick.get("start_time"),
            "strategy": pick.get("strategy"),
            "strategy_label": pick.get("strategy_label"),
            "confidence": pick.get("confidence"),
            "edge": pick.get("edge"),
            "model_projection": pick.get("model_projection"),
            "market_odds": pick.get("market_odds"),
            "win_probability": pick.get("win_probability"),
            "reason": pick.get("reason"),
            "status": "pending",
            "units": 0.0,
            "stake_units": 1.0,
            "recorded_at": now,
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


def grade_bet(
    bet: dict[str, Any],
    away_score: int,
    home_score: int,
) -> dict[str, Any]:
    side = bet["side"]
    if away_score == home_score:
        status: BetResult = "push"
    elif side == "away":
        status = "win" if away_score > home_score else "loss"
    else:
        status = "win" if home_score > away_score else "loss"

    odds = int(bet.get("market_odds") or -110)
    units = calculate_units(float(bet.get("stake_units") or 1), odds, status)
    return {
        **bet,
        "status": status,
        "units": units,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "final_score": f"{away_score}–{home_score}",
    }


def grade_pending(store: dict[str, Any]) -> dict[str, Any]:
    pending = [b for b in store["bets"] if b.get("status") == "pending"]
    if not pending:
        return store

    by_date_league: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for bet in pending:
        bucket = (bet["date"], bet["league"])
        by_date_league.setdefault(bucket, []).append(bet)

    for (date_label, league), bets in by_date_league.items():
        try:
            on_date = _parse_date_label(date_label)
        except ValueError:
            continue
        results = _fetch_results_by_event(league, on_date)
        for bet in bets:
            scores = results.get(bet["event_id"])
            if not scores:
                continue
            graded = grade_bet(bet, scores[0], scores[1])
            idx = next(i for i, b in enumerate(store["bets"]) if b["id"] == bet["id"])
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


def _summarize_bets(bets: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for b in bets if b.get("status") == "win")
    losses = sum(1 for b in bets if b.get("status") == "loss")
    pushes = sum(1 for b in bets if b.get("status") == "push")
    pending = sum(1 for b in bets if b.get("status") == "pending")
    settled_units = sum(b.get("units") or 0 for b in bets if b.get("status") != "pending")
    settled = wins + losses + pushes
    roi = (settled_units / settled * 100) if settled else 0.0
    return {
        "bets": len(bets),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": pending,
        "units": round(settled_units, 3),
        "roi_percent": round(roi, 2),
        "record": f"{wins}-{losses}" + (f"-{pushes}" if pushes else ""),
    }


def build_period_rollups(store: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {
        "daily": {},
        "weekly": {},
        "monthly": {},
        "yearly": {},
    }

    for bet in store["bets"]:
        try:
            d = _parse_date_label(bet["date"])
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
    sorted_bets = sorted(
        store["bets"],
        key=lambda b: (b.get("date", ""), -(b.get("edge") or 0)),
        reverse=True,
    )
    all_time = _summarize_bets(store["bets"])
    periods = build_period_rollups(store)
    tracking_since = sorted_bets[-1]["date"] if sorted_bets else None

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
            "Tracks algo value bets (positive edge vs model) from each daily slate. "
            "Graded at closing moneyline odds; 1u flat stake."
        ),
    }


def update_tracking(slate: dict[str, Any]) -> dict[str, Any]:
    store = load_store()
    store = record_from_slate(store, slate)
    store = grade_pending(store)
    save_store(store)
    return build_tracking_response(store)
