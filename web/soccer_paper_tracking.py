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
