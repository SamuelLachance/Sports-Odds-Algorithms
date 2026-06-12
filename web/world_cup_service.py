"""FIFA World Cup 2026 hub — full tournament slate, standings, unified predictions."""

from __future__ import annotations

import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from typing import Any

from web.daily_service import predict_live_game
from web.espn_client import ScheduledGame, _fetch_json, _parse_event, iso_to_project_date
from web.league_profiles import get_league_profile
from web.season_games import prewarm_league_power
from web.soccer_pred_model import get_soccer_pred_context
from web.wc_groups import (
    WORLD_CUP_2026_GROUPS,
    WORLD_CUP_FORMAT,
    is_placeholder_team,
    normalize_team_name,
    team_group,
)

TOURNAMENT_START = date(2026, 6, 11)
TOURNAMENT_END = date(2026, 7, 19)

ROUND_ORDER = [
    ("group-stage", "Group stage"),
    ("round-of-32", "Round of 32"),
    ("round-of-16", "Round of 16"),
    ("quarterfinals", "Quarter-finals"),
    ("semifinals", "Semi-finals"),
    ("3rd-place-match", "Third-place play-off"),
    ("final", "Final"),
]

ROUND_SLUGS = {slug for slug, _ in ROUND_ORDER}


def fetch_world_cup_events() -> list[dict[str, Any]]:
    """Load every World Cup 2026 event from ESPN (group + knockout)."""
    profile = get_league_profile("worldcup")
    base = f"https://site.api.espn.com/apis/site/v2/sports/{profile['sport_path']}"
    seen: set[str] = set()
    events: list[dict[str, Any]] = []

    # Range fetch (up to 100 events) plus per-day sweep for stragglers.
    urls = [
        f"{base}/scoreboard?dates={TOURNAMENT_START.strftime('%Y%m%d')}-{TOURNAMENT_END.strftime('%Y%m%d')}",
    ]
    day = TOURNAMENT_START
    while day <= TOURNAMENT_END:
        urls.append(f"{base}/scoreboard?dates={day.strftime('%Y%m%d')}")
        day = day.fromordinal(day.toordinal() + 1)

    for url in urls:
        try:
            payload = _fetch_json(url)
        except urllib.error.URLError:
            continue
        for raw in payload.get("events") or []:
            eid = str(raw.get("id") or "")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            events.append(raw)

    events.sort(key=lambda e: e.get("date") or "")
    return events


def _score(comp: dict[str, Any]) -> int:
    score_obj = comp.get("score")
    if isinstance(score_obj, dict):
        raw = score_obj.get("value")
    else:
        raw = score_obj
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _parse_wc_match(raw: dict[str, Any]) -> dict[str, Any]:
    comp = (raw.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    if not away or not home:
        return {}

    away_team = away.get("team") or {}
    home_team = home.get("team") or {}
    season = raw.get("season") or {}
    round_slug = season.get("slug") or "group-stage"
    status = (comp.get("status") or {}).get("type") or {}
    venue = (comp.get("venue") or {}).get("fullName") or (raw.get("venue") or {}).get("fullName")

    away_name = away_team.get("displayName") or ""
    home_name = home_team.get("displayName") or ""
    away_score = _score(away)
    home_score = _score(home)
    completed = bool(status.get("completed"))
    state = status.get("state") or status.get("name") or "unknown"

    group_id = None
    if round_slug == "group-stage":
        ga = team_group(away_name)
        gb = team_group(home_name)
        group_id = ga if ga and ga == gb else ga or gb

    scheduled = _parse_event(raw, "worldcup")

    return {
        "event_id": str(raw.get("id") or ""),
        "name": raw.get("name") or "",
        "short_name": raw.get("shortName") or "",
        "start_time": raw.get("date") or "",
        "round_slug": round_slug,
        "round_label": dict(ROUND_ORDER).get(round_slug, round_slug.replace("-", " ").title()),
        "group": group_id,
        "venue": venue,
        "status": state,
        "status_detail": status.get("shortDetail") or status.get("detail") or "",
        "completed": completed,
        "away": {
            "name": away_name,
            "abbr": (away_team.get("abbreviation") or "").upper(),
            "espn_id": str(away_team.get("id") or ""),
            "score": away_score if completed or state in {"in", "post"} else None,
            "winner": away.get("winner") is True,
        },
        "home": {
            "name": home_name,
            "abbr": (home_team.get("abbreviation") or "").upper(),
            "espn_id": str(home_team.get("id") or ""),
            "score": home_score if completed or state in {"in", "post"} else None,
            "winner": home.get("winner") is True,
        },
        "scoreline": f"{away_score}–{home_score}" if completed or state in {"in", "post"} else None,
        "is_placeholder": is_placeholder_team(away_name) or is_placeholder_team(home_name),
        "scheduled_game": scheduled,
    }


def _empty_row(team: str, group: str) -> dict[str, Any]:
    return {
        "team": team,
        "group": group,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
        "points": 0,
        "form": [],
    }


def _apply_result(row: dict[str, Any], gf: int, ga: int, form_char: str) -> None:
    row["played"] += 1
    row["goals_for"] += gf
    row["goals_against"] += ga
    row["goal_diff"] = row["goals_for"] - row["goals_against"]
    row["form"] = (row["form"] + [form_char])[-5:]
    if gf > ga:
        row["wins"] += 1
        row["points"] += 3
    elif gf < ga:
        row["losses"] += 1
    else:
        row["draws"] += 1
        row["points"] += 1


def _compute_group_standings(matches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {
        gid: {team: _empty_row(team, gid) for team in teams}
        for gid, teams in WORLD_CUP_2026_GROUPS.items()
    }

    for match in matches:
        if match.get("round_slug") != "group-stage" or not match.get("completed"):
            continue
        group_id = match.get("group")
        if not group_id or group_id not in tables:
            continue
        away_name = normalize_team_name(match["away"]["name"])
        home_name = normalize_team_name(match["home"]["name"])
        away_score = match["away"]["score"] or 0
        home_score = match["home"]["score"] or 0

        away_row = None
        home_row = None
        for key, row in tables[group_id].items():
            if key.lower() == away_name.lower() or away_name.lower() in key.lower():
                away_row = row
            if key.lower() == home_name.lower() or home_name.lower() in key.lower():
                home_row = row
        if not away_row or not home_row:
            continue

        if away_score > home_score:
            _apply_result(away_row, away_score, home_score, "W")
            _apply_result(home_row, home_score, away_score, "L")
        elif away_score < home_score:
            _apply_result(away_row, away_score, home_score, "L")
            _apply_result(home_row, home_score, away_score, "W")
        else:
            _apply_result(away_row, away_score, home_score, "D")
            _apply_result(home_row, home_score, away_score, "D")

    result: dict[str, list[dict[str, Any]]] = {}
    for gid, rows in tables.items():
        sorted_rows = sorted(
            rows.values(),
            key=lambda r: (r["points"], r["goal_diff"], r["goals_for"], r["team"]),
            reverse=True,
        )
        for idx, row in enumerate(sorted_rows, start=1):
            row["position"] = idx
            row["zone"] = (
                "qualified"
                if idx <= 2
                else "third_place_race"
                if idx == 3
                else "eliminated"
            )
        result[gid] = sorted_rows
    return result


def _compute_third_place_ranking(
    group_standings: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    thirds: list[dict[str, Any]] = []
    for gid, rows in group_standings.items():
        if len(rows) >= 3:
            third = dict(rows[2])
            third["group"] = gid
            thirds.append(third)
    thirds.sort(
        key=lambda r: (r["points"], r["goal_diff"], r["goals_for"], r["team"]),
        reverse=True,
    )
    for idx, row in enumerate(thirds, start=1):
        row["third_place_rank"] = idx
        row["third_place_qualified"] = idx <= 8
    return thirds


def _predict_match(scheduled: ScheduledGame | None) -> dict[str, Any] | None:
    if not scheduled:
        return None
    if is_placeholder_team(scheduled.away_name) or is_placeholder_team(scheduled.home_name):
        return None
    try:
        return predict_live_game(scheduled)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def get_world_cup_hub() -> dict[str, Any]:
    cutoff = iso_to_project_date(datetime.now(timezone.utc).isoformat())
    try:
        prewarm_league_power("worldcup", cutoff)
        get_soccer_pred_context("worldcup", cutoff)
    except Exception:
        pass

    raw_events = fetch_world_cup_events()
    matches = [_parse_wc_match(e) for e in raw_events]
    matches = [m for m in matches if m.get("event_id")]

    group_matches = [m for m in matches if m.get("round_slug") == "group-stage"]
    knockout_matches = [m for m in matches if m.get("round_slug") != "group-stage"]

    group_standings = _compute_group_standings(matches)
    third_place = _compute_third_place_ranking(group_standings)

    predictions: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    recommended: list[dict[str, Any]] = []

    predict_jobs: list[tuple[str, ScheduledGame, str, str]] = []
    for match in matches:
        if match.get("is_placeholder"):
            continue
        scheduled: ScheduledGame | None = match.get("scheduled_game")
        if not scheduled:
            continue
        predict_jobs.append(
            (match["event_id"], scheduled, match.get("name") or "", match.get("start_time") or "")
        )

    workers = min(8, max(1, len(predict_jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_predict_match, scheduled): (event_id, name, start_time)
            for event_id, scheduled, name, start_time in predict_jobs
        }
        for future in as_completed(futures):
            event_id, name, start_time = futures[future]
            pred = future.result()
            if pred is None:
                continue
            if pred.get("error"):
                errors.append({"event_id": event_id, "game": name, "error": pred["error"]})
                continue
            predictions[event_id] = {
                "matchup": pred.get("matchup"),
                "market": pred.get("market"),
                "model": pred.get("model"),
                "top_pick": pred.get("top_pick"),
                "recommendations": pred.get("recommendations"),
            }
            if pred.get("top_pick"):
                pick = {**pred["top_pick"], "event_id": event_id, "start_time": start_time}
                recommended.append(pick)

    from web.wc_simulation import simulate_tournament

    simulation = simulate_tournament(matches, predictions, predict_fn=_predict_match)
    for match in matches:
        match.pop("scheduled_game", None)

    by_round: dict[str, list[dict[str, Any]]] = {slug: [] for slug, _ in ROUND_ORDER}
    for match in matches:
        slug = match.get("round_slug") or "group-stage"
        if slug in by_round:
            by_round[slug].append(match)

    completed = sum(1 for m in matches if m.get("completed"))
    upcoming = sum(1 for m in matches if not m.get("completed") and not m.get("is_placeholder"))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tournament": "FIFA World Cup 2026",
        "format": WORLD_CUP_FORMAT,
        "summary": {
            "total_matches": len(matches),
            "group_stage_matches": len(group_matches),
            "knockout_matches": len(knockout_matches),
            "completed": completed,
            "upcoming": upcoming,
            "predictions_count": len(predictions),
            "recommended_bets": len(recommended),
        },
        "groups": {
            gid: {
                "id": gid,
                "name": f"Group {gid}",
                "teams": WORLD_CUP_2026_GROUPS[gid],
                "standings": group_standings.get(gid, []),
                "matches": [m for m in group_matches if m.get("group") == gid],
            }
            for gid in WORLD_CUP_2026_GROUPS
        },
        "third_place_ranking": third_place,
        "rounds": [
            {"slug": slug, "label": label, "matches": by_round.get(slug, [])}
            for slug, label in ROUND_ORDER
        ],
        "knockout_bracket": [
            m for m in knockout_matches if m.get("round_slug") != "3rd-place-match"
        ],
        "matches": matches,
        "predictions": predictions,
        "recommended_bets": recommended,
        "simulation": simulation,
        "errors": errors,
    }
