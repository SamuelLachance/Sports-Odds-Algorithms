"""Paper-track soccer value signals (not official Hubáček picks)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_TRACKING_PATH = PROJECT_ROOT / "data" / "soccer_paper_tracking.json"

_VALID_OUTCOMES = frozenset({"home", "draw", "away"})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value is False:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    bets = payload.get("bets")
    if not isinstance(bets, list):
        payload["bets"] = []
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
    if not league or not event_id:
        return
    if pick_outcome not in _VALID_OUTCOMES:
        return
    payload = _load_paper_log()
    bets: list[dict[str, Any]] = payload["bets"]
    dedupe_key = f"{league}:{event_id}:{pick_outcome}"
    if any(isinstance(bet, dict) and bet.get("key") == dedupe_key for bet in bets):
        return
    bets.append(
        {
            "key": dedupe_key,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "league": league,
            "event_id": str(event_id),
            "game_date": game_date,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_name": home_name,
            "away_name": away_name,
            "pick_outcome": pick_outcome,
            "model_prob": round(_safe_float(model_prob), 2),
            "market_ml": _safe_int(market_ml),
            "edge_pp": round(_safe_float(edge_pp), 2),
            "signals": signals if isinstance(signals, dict) else {},
            "paper": True,
        }
    )
    _save_paper_log(payload)


def _save_paper_log(payload: dict[str, Any]) -> None:
    PAPER_TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TRACKING_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _grade_outcome(pick_outcome: str, away_score: int, home_score: int) -> str | None:
    if pick_outcome == "draw":
        return "win" if away_score == home_score else "loss"
    if pick_outcome == "home":
        return "win" if home_score > away_score else "loss"
    if pick_outcome == "away":
        return "win" if away_score > home_score else "loss"
    return None


def grade_paper_picks() -> dict[str, Any]:
    """Grade pending paper picks against ESPN finals; returns a summary."""
    from web.tracking_service import _fetch_event_result, calculate_units

    payload = _load_paper_log()
    graded = 0
    for bet in payload["bets"]:
        if not isinstance(bet, dict):
            continue
        if bet.get("status") in {"win", "loss"}:
            continue
        league = bet.get("league")
        event_id = bet.get("event_id")
        pick_outcome = bet.get("pick_outcome") or ""
        if not league or not event_id or pick_outcome not in _VALID_OUTCOMES:
            continue
        try:
            scores = _fetch_event_result(league, str(event_id))
        except Exception:  # noqa: BLE001 - grading must not abort the Pages build
            continue
        if not scores:
            continue
        try:
            away_score, home_score = int(scores[0]), int(scores[1])
        except (TypeError, ValueError, IndexError):
            continue
        status = _grade_outcome(pick_outcome, away_score, home_score)
        if status is None:
            continue
        # Fail closed: without a posted price, leave pending so P&L is not
        # polluted with 0u losses (or omitted win payouts).
        market_ml = _safe_int(bet.get("market_ml"))
        if market_ml is None:
            continue
        try:
            units = round(calculate_units(1.0, market_ml, status), 3)
        except (TypeError, ValueError):
            continue
        bet["status"] = status
        bet["final_score"] = f"{away_score}–{home_score}"
        bet["graded_at"] = datetime.now(timezone.utc).isoformat()
        bet["units"] = units
        graded += 1

    settled = [b for b in payload["bets"] if isinstance(b, dict) and b.get("status") in {"win", "loss"}]
    wins = sum(1 for b in settled if b["status"] == "win")
    units = sum(_safe_float(b.get("units")) for b in settled)
    summary = {
        "picks": len([b for b in payload["bets"] if isinstance(b, dict)]),
        "settled": len(settled),
        "wins": wins,
        "losses": len(settled) - wins,
        "units": round(units, 3),
        "newly_graded": graded,
    }
    payload["summary"] = summary
    # Persist summary even when nothing newly graded so Pages/deploy
    # summaries stay consistent with the on-disk ledger.
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
    try:
        if not isinstance(blended, dict) or not league or not event_id:
            return
        signals = blended.get("soccer_pick_signals") or {}
        if not isinstance(signals, dict):
            return
        if not signals.get("high_confidence_disagreement"):
            return
        # Paper the outcome that carries the edge, not merely the model favorite.
        outcome = signals.get("best_edge_outcome") or signals.get("model_best_outcome")
        if outcome not in _VALID_OUTCOMES:
            return
        soccer_pred = blended.get("soccer_pred") or {}
        if not isinstance(soccer_pred, dict):
            soccer_pred = {}
        prob_key = {
            "home": "pick_home_win_probability",
            "draw": "pick_draw_probability",
            "away": "pick_away_win_probability",
        }[outcome]
        fallback_key = {
            "home": "home_win_probability",
            "draw": "draw_probability",
            "away": "away_win_probability",
        }[outcome]
        raw_prob = soccer_pred.get(prob_key)
        if raw_prob is None:
            raw_prob = blended.get(fallback_key)
        # Fail closed: missing probs must not invent model_prob=0.0 junk rows.
        if raw_prob is None:
            return
        try:
            model_prob = float(raw_prob)
        except (TypeError, ValueError):
            return
        if model_prob <= 0:
            return
        market_ml = {"home": home_ml, "draw": draw_ml, "away": away_ml}[outcome]
        record_soccer_paper_pick(
            league=league,
            event_id=str(event_id),
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_name=home_name,
            away_name=away_name,
            game_date=game_date,
            pick_outcome=outcome,
            model_prob=model_prob,
            market_ml=market_ml,
            edge_pp=_safe_float(signals.get("max_edge_pp")),
            signals=signals,
        )
    except Exception:  # noqa: BLE001 - paper tracking must never break the slate
        return
