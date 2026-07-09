"""Paper-track soccer value signals (not official Hubáček picks)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_TRACKING_PATH = PROJECT_ROOT / "data" / "soccer_paper_tracking.json"


def _load_paper_log() -> dict[str, Any]:
    if not PAPER_TRACKING_PATH.is_file():
        return {"version": 1, "bets": []}
    try:
        payload = json.loads(PAPER_TRACKING_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "bets": []}
    if not isinstance(payload, dict):
        return {"version": 1, "bets": []}
    payload.setdefault("version", 1)
    payload.setdefault("bets", [])
    return payload


def record_soccer_paper_pick(
    *,
    league: str,
    event_id: str,
    home_abbr: str,
    away_abbr: str,
    home_name: str,
    away_name: str,
    game_date: str,
    pick_outcome: str,
    model_prob: float,
    market_ml: int | None,
    edge_pp: float,
    signals: dict[str, Any] | None = None,
) -> None:
    """Append a paper pick when soccer Path A flags a high-confidence disagreement."""
    payload = _load_paper_log()
    bets: list[dict[str, Any]] = payload["bets"]
    dedupe_key = f"{league}:{event_id}:{pick_outcome}"
    if any(bet.get("key") == dedupe_key for bet in bets):
        return
    bets.append(
        {
            "key": dedupe_key,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "league": league,
            "event_id": event_id,
            "game_date": game_date,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_name": home_name,
            "away_name": away_name,
            "pick_outcome": pick_outcome,
            "model_prob": round(model_prob, 2),
            "market_ml": market_ml,
            "edge_pp": round(edge_pp, 2),
            "signals": signals or {},
            "paper": True,
        }
    )
    PAPER_TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TRACKING_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save_paper_log(payload: dict[str, Any]) -> None:
    PAPER_TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TRACKING_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _grade_outcome(pick_outcome: str, away_score: int, home_score: int) -> str:
    if pick_outcome == "draw":
        return "win" if away_score == home_score else "loss"
    if pick_outcome == "home":
        return "win" if home_score > away_score else "loss"
    return "win" if away_score > home_score else "loss"


def grade_paper_picks() -> dict[str, Any]:
    """Grade pending paper picks against ESPN finals; returns a summary."""
    from web.tracking_service import _fetch_event_result, calculate_units

    payload = _load_paper_log()
    graded = 0
    for bet in payload["bets"]:
        if bet.get("status") in {"win", "loss"}:
            continue
        league = bet.get("league")
        event_id = bet.get("event_id")
        if not league or not event_id:
            continue
        scores = _fetch_event_result(league, str(event_id))
        if not scores:
            continue
        away_score, home_score = scores
        status = _grade_outcome(bet.get("pick_outcome") or "", away_score, home_score)
        bet["status"] = status
        bet["final_score"] = f"{away_score}–{home_score}"
        bet["graded_at"] = datetime.now(timezone.utc).isoformat()
        market_ml = bet.get("market_ml")
        if market_ml is not None:
            bet["units"] = round(calculate_units(1.0, int(market_ml), status), 3)
        graded += 1

    if graded:
        _save_paper_log(payload)

    settled = [b for b in payload["bets"] if b.get("status") in {"win", "loss"}]
    wins = sum(1 for b in settled if b["status"] == "win")
    units = sum(b.get("units") or 0.0 for b in settled)
    summary = {
        "picks": len(payload["bets"]),
        "settled": len(settled),
        "wins": wins,
        "losses": len(settled) - wins,
        "units": round(units, 3),
        "newly_graded": graded,
    }
    payload["summary"] = summary
    if graded:
        _save_paper_log(payload)
    return summary


def maybe_record_from_blend(
    blended: dict[str, Any],
    *,
    league: str,
    event_id: str,
    home_abbr: str,
    away_abbr: str,
    home_name: str,
    away_name: str,
    game_date: str,
    home_ml: int | None,
    draw_ml: int | None,
    away_ml: int | None,
) -> None:
    signals = blended.get("soccer_pick_signals") or {}
    if not signals.get("high_confidence_disagreement"):
        return
    outcome = signals.get("model_best_outcome")
    if outcome not in {"home", "draw", "away"}:
        return
    soccer_pred = blended.get("soccer_pred") or {}
    prob_key = {
        "home": "pick_home_win_probability",
        "draw": "pick_draw_probability",
        "away": "pick_away_win_probability",
    }[outcome]
    model_prob = float(soccer_pred.get(prob_key) or blended.get(f"{outcome.replace('home', 'home_win')}_probability") or 0)
    market_ml = {"home": home_ml, "draw": draw_ml, "away": away_ml}[outcome]
    record_soccer_paper_pick(
        league=league,
        event_id=event_id,
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        home_name=home_name,
        away_name=away_name,
        game_date=game_date,
        pick_outcome=outcome,
        model_prob=model_prob,
        market_ml=market_ml,
        edge_pp=float(signals.get("max_edge_pp") or 0),
        signals=signals,
    )
