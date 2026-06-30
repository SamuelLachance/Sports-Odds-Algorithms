"""Daily slate analysis and bet recommendations across all supported leagues."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORONTO = ZoneInfo("America/Toronto")


def _toronto_today() -> date:
    return datetime.now(TORONTO).date()

from web.bet_advisor import (  # noqa: E402
    evaluate_picks,
    evaluate_soccer_picks,
    evaluate_spread_picks,
    model_moneylines,
    pick_to_dict,
    projections_from_win_probs,
    resolve_binary_win_probs,
    soccer_model_moneylines,
)
from web.baseball_pred_model import get_baseball_pred_context, is_baseball_league  # noqa: E402
from web.basketball_pred_model import get_basketball_pred_context, is_basketball_league  # noqa: E402
from web.football_pred_model import get_football_pred_context, is_football_league  # noqa: E402
from web.hockey_pred_model import get_hockey_pred_context, is_hockey_league  # noqa: E402
from web.league_profiles import is_soccer_league  # noqa: E402
from web.soccer_pred_model import get_soccer_pred_context  # noqa: E402
from web.blend_service import blend_predictions, blended_home_spread_margin, compute_model_agreement  # noqa: E402
from web.season_games import prewarm_league_power  # noqa: E402
from web.espn_client import (  # noqa: E402
    ScheduledGame,
    current_season_year,
    fetch_scoreboard,
    iso_to_project_date,
)
from web.league_profiles import (  # noqa: E402
    LEAGUE_PROFILES,
    MIN_EXPECTED_VALUE_PCT,
    MIN_RECOMMENDED_EDGE,
    SUPPORTED_LEAGUES,
    eligible_for_official_picks,
    get_algo_league,
    uses_spread_bets,
)
from web.league_readiness import is_league_ready_for_daily_slate  # noqa: E402
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
    today = _toronto_today()
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


def _enrich_pick_with_team_abbr(
    pick_dict: dict[str, Any], matchup: dict[str, Any]
) -> dict[str, Any]:
    side = pick_dict.get("side")
    if side in ("away", "home"):
        abbr = (matchup.get(side) or {}).get("abbr")
        if abbr:
            return {**pick_dict, "team_abbr": abbr}
    return pick_dict


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
        event_id=game.event_id,
        home_espn_id=game.home_espn_id,
        away_espn_id=game.away_espn_id,
        home_slug=home[1],
        away_slug=away[1],
    )

    total = float(blended["total_score"])
    win_probability = float(blended["win_probability"])
    favorite_side = blended["favorite_side"]

    if blended.get("threeway"):
        home_prob = float(blended["home_win_probability"])
        draw_prob = float(blended["draw_probability"])
        away_prob = float(blended["away_win_probability"])
        away_proj, draw_proj, home_proj = soccer_model_moneylines(
            home_prob, draw_prob, away_prob
        )
    elif blended.get("blended_home_win_probability") is not None:
        home_prob = float(blended["blended_home_win_probability"])
        away_prob = 100.0 - home_prob
        away_proj, home_proj = projections_from_win_probs(home_prob, away_prob)
        draw_proj = None
    else:
        away_proj, home_proj = model_moneylines(total)
        draw_proj = None

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

    market_payload = {
        "away_moneyline": game.market.away_moneyline,
        "home_moneyline": game.market.home_moneyline,
        "draw_moneyline": game.market.draw_moneyline,
        "spread": game.market.spread,
        "away_spread_odds": game.market.away_spread_odds,
        "home_spread_odds": game.market.home_spread_odds,
    }
    model_agreement = compute_model_agreement(
        blended, game.league, market=market_payload
    )
    value_agreed = (
        model_agreement.get("required") == 3 and model_agreement.get("agreed")
    )
    pick_min_edge = MIN_RECOMMENDED_EDGE
    value_sides = set(
        model_agreement.get("value_sides")
        or model_agreement.get("value_outcomes")
        or []
    )

    model_payload: dict[str, Any] = {
        **blended,
        "model_agreement": model_agreement,
        "away_projection": away_proj,
        "home_projection": home_proj,
        "factors": factors,
    }
    if draw_proj is not None:
        model_payload["draw_projection"] = draw_proj

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
            model_margin_home=blended_home_spread_margin(blended, game.league),
            min_edge=pick_min_edge,
        )
        if not picks:
            ml_away_prob, ml_home_prob = resolve_binary_win_probs(blended, total)
            picks = evaluate_picks(
                away_name=game.away_name,
                home_name=game.home_name,
                away_slug=away[1],
                home_slug=home[1],
                total_score=total,
                win_probability=win_probability,
                away_market=game.market.away_moneyline,
                home_market=game.market.home_moneyline,
                away_prob=ml_away_prob,
                home_prob=ml_home_prob,
                min_edge=pick_min_edge,
            )
    elif is_soccer_league(game.league):
        soccer_pred = blended.get("soccer_pred") or {}
        picks = evaluate_soccer_picks(
            away_name=game.away_name,
            home_name=game.home_name,
            away_slug=away[1],
            home_slug=home[1],
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            away_proj=away_proj,
            draw_proj=draw_proj,
            home_proj=home_proj,
            away_market=game.market.away_moneyline,
            draw_market=game.market.draw_moneyline,
            home_market=game.market.home_moneyline,
            expected_home_goals=blended.get("expected_home_goals")
            or soccer_pred.get("expected_home_goals"),
            expected_away_goals=blended.get("expected_away_goals")
            or soccer_pred.get("expected_away_goals"),
            min_edge=pick_min_edge,
        )
    else:
        ml_away_prob, ml_home_prob = resolve_binary_win_probs(blended, total)
        picks = evaluate_picks(
            away_name=game.away_name,
            home_name=game.home_name,
            away_slug=away[1],
            home_slug=home[1],
            total_score=total,
            win_probability=win_probability,
            away_market=game.market.away_moneyline,
            home_market=game.market.home_moneyline,
            away_prob=ml_away_prob,
            home_prob=ml_home_prob,
            min_edge=pick_min_edge,
        )

    if model_agreement.get("required") == 3 and not model_agreement.get("agreed"):
        picks = []
    elif value_agreed and value_sides:
        picks = [pick for pick in picks if pick.side in value_sides]
        picks = picks[:1] if picks else []

    official_picks = picks if eligible_for_official_picks(game.league) else []

    matchup = {
        "away": {"abbr": away[0], "slug": away[1], "name": game.away_name},
        "home": {"abbr": home[0], "slug": home[1], "name": game.home_name},
    }

    def _enrich_pick(pick_dict: dict[str, Any]) -> dict[str, Any]:
        return _enrich_pick_with_team_abbr(pick_dict, matchup)

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
        "matchup": matchup,
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
        "eligible_for_official_picks": eligible_for_official_picks(game.league),
        "recommendations": [_enrich_pick(pick_to_dict(pick)) for pick in official_picks],
        "top_pick": _enrich_pick(pick_to_dict(official_picks[0])) if official_picks else None,
    }


def _slate_cutoff_date() -> str:
    today = _toronto_today()
    return f"{today.month}-{today.day}-{today.year}"


def _actionable_games(scheduled: list[ScheduledGame]) -> list[ScheduledGame]:
    return [
        game
        for game in scheduled
        if game.status not in {"in", "post"} and _is_actionable_soon(game)
    ]


def _prewarm_league_models(
    league: str,
    cutoffs: set[str],
    errors: list[dict[str, str]],
) -> None:
    for cutoff in cutoffs:
        try:
            prewarm_league_power(league, cutoff)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {"league": league, "error": f"Power prewarm failed ({cutoff}): {exc}"}
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
        if is_hockey_league(league):
            try:
                get_hockey_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"Hockey model prewarm failed ({cutoff}): {exc}",
                    }
                )
        if is_football_league(league):
            try:
                get_football_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"Football model prewarm failed ({cutoff}): {exc}",
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


def get_daily_slate(days_ahead: int = 0) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    slate_cutoff = _slate_cutoff_date()

    for league in SUPPORTED_LEAGUES:
        try:
            scheduled = fetch_scoreboard(league, days_ahead=days_ahead)
        except Exception as exc:  # noqa: BLE001
            errors.append({"league": league, "error": str(exc)})
            continue

        actionable = _actionable_games(scheduled)
        if not actionable:
            continue

        if not is_league_ready_for_daily_slate(league, slate_cutoff):
            continue

        power_cutoffs = {_today_cutoff(game) for game in actionable}
        _prewarm_league_models(league, power_cutoffs, errors)

        for game in actionable:
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
        if not game.get("eligible_for_official_picks", True):
            continue
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
        if current is None or (
            rec.get("ev_pct", 0),
            rec.get("edge", 0),
        ) > (
            current.get("ev_pct", 0),
            current.get("edge", 0),
        ):
            best_by_event[event_id] = rec
    def _meets_recommendation_threshold(game: dict[str, Any], rec: dict[str, Any]) -> bool:
        edge = rec.get("edge", 0)
        ev_pct = rec.get("ev_pct", 0)
        if ev_pct < MIN_EXPECTED_VALUE_PCT and edge < MIN_RECOMMENDED_EDGE:
            return False
        agreement = (game.get("model") or {}).get("model_agreement") or {}
        if agreement.get("required") == 3 and agreement.get("agreed"):
            value_sides = set(
                agreement.get("value_sides") or agreement.get("value_outcomes") or []
            )
            return rec.get("side") in value_sides
        return True

    games_by_event = {game["event_id"]: game for game in all_games}
    recommendations = sorted(
        best_by_event.values(),
        key=lambda item: (item.get("ev_pct", 0), item.get("edge", 0)),
        reverse=True,
    )

    qualifying = [
        r
        for r in recommendations
        if _meets_recommendation_threshold(games_by_event.get(r.get("event_id", ""), {}), r)
    ]

    return {
        "generated_at": generated_at,
        "date_label": _toronto_today().isoformat(),
        "summary": {
            "games_analyzed": len(all_games),
            "recommended_bets": len(qualifying),
            "min_edge": MIN_RECOMMENDED_EDGE,
            "min_ev_pct": MIN_EXPECTED_VALUE_PCT,
            "leagues": list({game["league"] for game in all_games}),
        },
        "recommended_bets": qualifying[:20],
        "min_recommended_edge": MIN_RECOMMENDED_EDGE,
        "min_expected_value_pct": MIN_EXPECTED_VALUE_PCT,
        "games": all_games,
        "errors": errors,
    }
