"""Daily slate analysis and bet recommendations across NBA, NHL, and MLB."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from web.bet_advisor import (  # noqa: E402
    evaluate_picks,
    evaluate_soccer_picks,
    evaluate_spread_picks,
    model_moneylines,
    pick_to_dict,
    projections_from_win_probs,
    soccer_model_moneylines,
    soccer_threeway_probs,
)
from web.baseball_pred_model import get_baseball_pred_context, is_baseball_league  # noqa: E402
from web.basketball_pred_model import get_basketball_pred_context, is_basketball_league  # noqa: E402
from web.soccer_pred_model import get_soccer_pred_context  # noqa: E402
from web.blend_service import blend_predictions, compute_model_agreement  # noqa: E402
from web.season_games import prewarm_league_power  # noqa: E402
from web.espn_client import (  # noqa: E402
    ScheduledGame,
    current_season_year,
    fetch_scoreboard,
    iso_to_project_date,
)
from web.league_profiles import (  # noqa: E402
    LEAGUE_PROFILES,
    MIN_RECOMMENDED_EDGE,
    SUPPORTED_LEAGUES,
    get_algo_league,
    is_soccer_league,
    uses_spread_bets,
)
from web.live_data import load_live_team_data, resolve_team  # noqa: E402
from web.predict_service import FACTOR_LABELS  # noqa: E402


def _ensure_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _today_cutoff(game: ScheduledGame) -> str:
    if game.start_time:
        return iso_to_project_date(game.start_time)
    today = date.today()
    return f"{today.month}-{today.day}-{today.year}"


def _is_actionable_soon(game: ScheduledGame, horizon_days: int = 3) -> bool:
    if not game.start_time:
        return True
    start = datetime.fromisoformat(game.start_time.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return now - timedelta(days=1) <= start <= now + timedelta(days=horizon_days)


def _season_year_from_cutoff(league: str, cutoff_date: str) -> str:
    month, day, year = cutoff_date.split("-")
    season = current_season_year(league, date(int(year), int(month), int(day)))
    return str(season)


def predict_live_game(game: ScheduledGame) -> dict[str, Any]:
    _ensure_project_root()
    from algo import Algo
    from odds_calculator import Odds_Calculator

    away = resolve_team(game.league, game.away_abbr, game.away_name)
    home = resolve_team(game.league, game.home_abbr, game.home_name)
    if not away or not home:
        raise ValueError(
            f"Unknown teams for {game.league}: {game.away_abbr} @ {game.home_abbr}"
        )

    cutoff = _today_cutoff(game)
    season_year = _season_year_from_cutoff(game.league, cutoff)

    data_away = load_live_team_data(game.league, away, game.away_espn_id, cutoff)
    data_home = load_live_team_data(game.league, home, game.home_espn_id, cutoff)

    if not data_away or not data_home:
        raise ValueError(
            f"Insufficient season data for {away[1]} or {home[1]} before {cutoff}."
        )

    algo_league = get_algo_league(game.league)
    odds_calculator = Odds_Calculator(algo_league)
    algo = Algo(algo_league)

    with redirect_stdout(io.StringIO()):
        returned_away = odds_calculator.analyze2(away, home, data_away, "away")
        returned_home = odds_calculator.analyze2(home, away, data_home, "home")
        algo_data = algo.calculate_V2(cutoff, returned_away, returned_home)

    legacy_total = float(algo_data["total"])
    legacy_win_probability = abs(legacy_total)

    blended = blend_predictions(
        legacy_total_score=legacy_total,
        legacy_win_probability=legacy_win_probability,
        league=game.league,
        cutoff_date=cutoff,
        home_abbr=home[0],
        away_abbr=away[0],
        home_name=game.home_name,
        away_name=game.away_name,
    )

    total = float(blended["total_score"])
    win_probability = float(blended["win_probability"])
    favorite_side = blended["favorite_side"]

    if blended.get("blended_home_win_probability") is not None:
        home_prob = float(blended["blended_home_win_probability"])
        away_prob = 100.0 - home_prob
        away_proj, home_proj = projections_from_win_probs(home_prob, away_prob)
    else:
        away_proj, home_proj = model_moneylines(total)

    factors = []
    for key, label in FACTOR_LABELS.items():
        if key not in algo_data:
            continue
        value = float(algo_data[key])
        factors.append(
            {
                "key": key,
                "label": label,
                "value": value,
                "favors": "away" if value > 0 else "home" if value < 0 else "neutral",
            }
        )

    model_agreement = compute_model_agreement(blended, game.league)

    model_payload: dict[str, Any] = {
        **blended,
        "model_agreement": model_agreement,
        "away_projection": away_proj,
        "home_projection": home_proj,
        "factors": factors,
    }

    if uses_spread_bets(game.league):
        picks = evaluate_spread_picks(
            league=game.league,
            away_name=game.away_name,
            home_name=game.home_name,
            away_slug=away[1],
            home_slug=home[1],
            total_score=total,
            win_probability=win_probability,
            consensus_spread=game.market.spread,
            away_spread_odds=game.market.away_spread_odds,
            home_spread_odds=game.market.home_spread_odds,
        )
    elif is_soccer_league(game.league):
        if blended.get("threeway"):
            home_prob = float(blended["home_win_probability"])
            draw_prob = float(blended["draw_probability"])
            away_prob = float(blended["away_win_probability"])
            total = float(blended["total_score"])
            win_probability = float(blended["win_probability"])
            favorite_side = blended["favorite_side"]
        else:
            home_prob, draw_prob, away_prob = soccer_threeway_probs(total, game.league)
        away_proj, draw_proj, home_proj = soccer_model_moneylines(
            home_prob, draw_prob, away_prob
        )
        model_payload.update(
            {
                "threeway": True,
                "home_win_probability": round(home_prob, 2),
                "draw_probability": round(draw_prob, 2),
                "away_win_probability": round(away_prob, 2),
                "draw_projection": draw_proj,
                "total_score": round(total, 2),
                "win_probability": round(win_probability, 2),
                "favorite_side": favorite_side,
            }
        )
        model_payload["away_projection"] = away_proj
        model_payload["home_projection"] = home_proj
        picks = evaluate_soccer_picks(
            away_name=game.away_name,
            home_name=game.home_name,
            away_slug=away[1],
            home_slug=home[1],
            total_score=total,
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            away_proj=away_proj,
            draw_proj=draw_proj,
            home_proj=home_proj,
            away_market=game.market.away_moneyline,
            draw_market=game.market.draw_moneyline,
            home_market=game.market.home_moneyline,
        )
    else:
        picks = evaluate_picks(
            away_name=game.away_name,
            home_name=game.home_name,
            away_slug=away[1],
            home_slug=home[1],
            total_score=total,
            win_probability=win_probability,
            away_market=game.market.away_moneyline,
            home_market=game.market.home_moneyline,
        )

    if model_agreement.get("required") == 3 and not model_agreement.get("agreed"):
        picks = []

    return {
        "event_id": game.event_id,
        "league": game.league,
        "league_name": LEAGUE_PROFILES[game.league]["name"],
        "name": game.name,
        "start_time": game.start_time,
        "status": game.status,
        "status_detail": game.status_detail,
        "cutoff_date": cutoff,
        "season_year": season_year,
        "matchup": {
            "away": {"abbr": away[0], "slug": away[1], "name": game.away_name},
            "home": {"abbr": home[0], "slug": home[1], "name": game.home_name},
        },
        "market": {
            "provider": game.market.provider,
            "away_moneyline": game.market.away_moneyline,
            "home_moneyline": game.market.home_moneyline,
            "draw_moneyline": game.market.draw_moneyline,
            "spread": game.market.spread,
            "away_spread_odds": game.market.away_spread_odds,
            "home_spread_odds": game.market.home_spread_odds,
            "over_under": game.market.over_under,
        },
        "model": model_payload,
        "recommendations": [pick_to_dict(pick) for pick in picks],
        "top_pick": pick_to_dict(picks[0]) if picks else None,
    }


def get_daily_slate(days_ahead: int = 0) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for league in SUPPORTED_LEAGUES:
        try:
            scheduled = fetch_scoreboard(league, days_ahead=days_ahead)
        except Exception as exc:  # noqa: BLE001
            errors.append({"league": league, "error": str(exc)})
            continue

        power_cutoffs = {
            _today_cutoff(game)
            for game in scheduled
            if game.status not in {"in", "post"} and _is_actionable_soon(game)
        }
        for cutoff in power_cutoffs:
            try:
                prewarm_league_power(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"Power prewarm failed ({cutoff}): {exc}",
                    }
                )
            if is_basketball_league(league):
                try:
                    get_basketball_pred_context(league, cutoff)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "league": league,
                            "error": f"Basketball matrix prewarm failed ({cutoff}): {exc}",
                        }
                    )
            if is_baseball_league(league):
                try:
                    get_baseball_pred_context(league, cutoff)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "league": league,
                            "error": f"Baseball model prewarm failed ({cutoff}): {exc}",
                        }
                    )
            if is_soccer_league(league):
                try:
                    get_soccer_pred_context(league, cutoff)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "league": league,
                            "error": f"Soccer model prewarm failed ({cutoff}): {exc}",
                        }
                    )

        for game in scheduled:
            if game.status in {"in", "post"}:
                continue
            if not _is_actionable_soon(game):
                continue
            try:
                all_games.append(predict_live_game(game))
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "game": game.name,
                        "error": str(exc),
                    }
                )

    recommendations = []
    for game in all_games:
        for pick in game.get("recommendations") or []:
            recommendations.append(
                {
                    **pick,
                    "league": game["league"],
                    "league_name": game["league_name"],
                    "event_id": game["event_id"],
                    "matchup": f"{game['matchup']['away']['name']} @ {game['matchup']['home']['name']}",
                    "start_time": game["start_time"],
                }
            )

    best_by_event: dict[str, dict[str, Any]] = {}
    for rec in recommendations:
        event_id = rec.get("event_id") or ""
        if not event_id:
            continue
        current = best_by_event.get(event_id)
        if current is None or rec.get("edge", 0) > current.get("edge", 0):
            best_by_event[event_id] = rec
    recommendations = sorted(
        best_by_event.values(),
        key=lambda item: item.get("edge", 0),
        reverse=True,
    )

    return {
        "generated_at": generated_at,
        "date_label": date.today().isoformat(),
        "summary": {
            "games_analyzed": len(all_games),
            "recommended_bets": len(
                [r for r in recommendations if r.get("edge", 0) >= MIN_RECOMMENDED_EDGE]
            ),
            "min_edge": MIN_RECOMMENDED_EDGE,
            "leagues": list({game["league"] for game in all_games}),
        },
        "recommended_bets": [
            r for r in recommendations if r.get("edge", 0) >= MIN_RECOMMENDED_EDGE
        ][:20],
        "min_recommended_edge": MIN_RECOMMENDED_EDGE,
        "games": all_games,
        "errors": errors,
    }
