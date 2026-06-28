"""Normalize ESPN payloads into stable database JSON shapes."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 2


def _stat_map(stats: list[dict[str, Any]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stat in stats or []:
        name = stat.get("name") or stat.get("abbreviation")
        if not name:
            continue
        out[str(name)] = {
            "value": stat.get("value"),
            "display": stat.get("displayValue"),
        }
    return out


def parse_standings(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"groups": [], "teams": []}

    groups: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], group_name: str | None = None) -> None:
        name = node.get("name") or group_name
        entries = (node.get("standings") or {}).get("entries") or []
        group_rows: list[dict[str, Any]] = []
        for entry in entries:
            team = entry.get("team") or {}
            stats = _stat_map(entry.get("stats"))
            row = {
                "team_id": team.get("id"),
                "abbr": (team.get("abbreviation") or "").lower(),
                "name": team.get("displayName") or team.get("name"),
                "rank": stats.get("rank", {}).get("value"),
                "wins": stats.get("wins", {}).get("value"),
                "losses": stats.get("losses", {}).get("value"),
                "ties": stats.get("ties", {}).get("value"),
                "win_percent": stats.get("winPercent", {}).get("value"),
                "games_behind": stats.get("gamesBehind", {}).get("value"),
                "streak": stats.get("streak", {}).get("display"),
                "points_for": stats.get("avgPointsFor", {}).get("value")
                or stats.get("pointsFor", {}).get("value"),
                "points_against": stats.get("avgPointsAgainst", {}).get("value")
                or stats.get("pointsAgainst", {}).get("value"),
                "point_differential": stats.get("differential", {}).get("value")
                or stats.get("pointDifferential", {}).get("value"),
                "playoff_seed": stats.get("playoffSeed", {}).get("value"),
                "clincher": stats.get("clincher", {}).get("display"),
                "stats": stats,
            }
            group_rows.append(row)
            flat.append({**row, "group": name})

        if group_rows:
            groups.append({"name": name, "teams": group_rows})

        for child in node.get("children") or []:
            walk(child, child.get("name"))

    if payload.get("children"):
        for child in payload["children"]:
            walk(child)
    else:
        walk(payload)

    flat.sort(
        key=lambda row: (
            row.get("rank") if row.get("rank") is not None else 999,
            -(row.get("win_percent") or 0),
        ),
    )
    return {"groups": groups, "teams": flat}


def parse_news(payload: dict[str, Any] | None, limit: int = 12) -> list[dict[str, Any]]:
    articles = (payload or {}).get("articles") or []
    rows: list[dict[str, Any]] = []
    for article in articles[:limit]:
        rows.append(
            {
                "id": article.get("id"),
                "headline": article.get("headline"),
                "description": article.get("description"),
                "published": article.get("published"),
                "type": article.get("type"),
                "link": (article.get("links") or {}).get("web", {}).get("href"),
                "images": [
                    img.get("url")
                    for img in (article.get("images") or [])
                    if img.get("url")
                ][:1],
            }
        )
    return rows


def parse_rankings(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows: list[dict[str, Any]] = []
    for block in payload.get("rankings") or []:
        poll_name = block.get("name") or block.get("shortName")
        for rank_row in (block.get("ranks") or []):
            team = rank_row.get("team") or {}
            rows.append(
                {
                    "poll": poll_name,
                    "rank": rank_row.get("current"),
                    "previous": rank_row.get("previous"),
                    "team": team.get("displayName") or team.get("name"),
                    "abbr": (team.get("abbreviation") or "").lower(),
                    "record": rank_row.get("recordSummary"),
                    "points": rank_row.get("points"),
                    "trend": rank_row.get("trend"),
                }
            )
    return rows


def _dict_or_str_label(value: Any, *, abbrev_key: str = "abbreviation", name_key: str = "name") -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return value.get(abbrev_key) or value.get(name_key) or value.get("displayName")
    return None


def _iter_roster_athletes(athletes: list[Any]) -> list[dict[str, Any]]:
    """Flatten ESPN roster groups (NFL offense/defense, NHL position buckets)."""
    rows: list[dict[str, Any]] = []
    for entry in athletes:
        if not isinstance(entry, dict):
            continue
        items = entry.get("items")
        if items:
            group_position = entry.get("position")
            for athlete in items:
                if not isinstance(athlete, dict):
                    continue
                row = dict(athlete)
                if group_position and not row.get("position"):
                    row["position"] = group_position
                rows.append(row)
        else:
            rows.append(entry)
    return rows


def _parse_roster_athlete(athlete: dict[str, Any]) -> dict[str, Any]:
    experience = athlete.get("experience")
    if isinstance(experience, dict):
        experience_years = experience.get("years")
    elif isinstance(experience, (int, float)):
        experience_years = experience
    else:
        experience_years = None

    status = athlete.get("status")
    if isinstance(status, dict):
        status_name = status.get("name")
    else:
        status_name = status if isinstance(status, str) else None

    headshot = athlete.get("headshot")
    if isinstance(headshot, dict):
        headshot_url = headshot.get("href")
    else:
        headshot_url = headshot if isinstance(headshot, str) else None

    return {
        "id": athlete.get("id"),
        "name": athlete.get("displayName") or athlete.get("fullName"),
        "position": _dict_or_str_label(athlete.get("position")),
        "jersey": athlete.get("jersey"),
        "age": athlete.get("age"),
        "height": athlete.get("displayHeight"),
        "weight": athlete.get("displayWeight"),
        "experience": experience_years,
        "status": status_name,
        "headshot": headshot_url,
    }


def parse_roster(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    athletes = _iter_roster_athletes((payload or {}).get("athletes") or [])
    return [_parse_roster_athlete(athlete) for athlete in athletes]


def parse_team_statistics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"categories": []}
    categories: list[dict[str, Any]] = []

    results = payload.get("results") or {}
    stats_root = results.get("stats") if isinstance(results.get("stats"), dict) else {}
    raw_categories = stats_root.get("categories") or []

    for split in raw_categories:
        stats = []
        for stat in split.get("stats") or []:
            stats.append(
                {
                    "name": stat.get("displayName") or stat.get("name"),
                    "display": stat.get("displayValue"),
                    "value": stat.get("value"),
                    "rank": stat.get("rank"),
                }
            )
        categories.append({"name": split.get("displayName") or split.get("name"), "stats": stats})

    return {"categories": categories}


def parse_team_summary(
    roster_payload: dict[str, Any] | None,
    stats_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    team = (roster_payload or {}).get("team") or (stats_payload or {}).get("team") or {}
    coach = ((roster_payload or {}).get("coach") or [{}])[0] if roster_payload else {}
    return {
        "espn_id": team.get("id"),
        "abbr": (team.get("abbreviation") or "").lower(),
        "name": team.get("displayName") or team.get("name"),
        "location": team.get("location"),
        "color": team.get("color"),
        "logo": (team.get("logo") or (team.get("logos") or [{}])[0].get("href") if team.get("logos") else team.get("logo")),
        "record_summary": team.get("recordSummary") or team.get("standingSummary"),
        "season_summary": team.get("seasonSummary"),
        "standing_summary": team.get("standingSummary"),
        "coach": " ".join(
            part for part in (coach.get("firstName"), coach.get("lastName")) if part
        ).strip()
        or None,
    }


def build_trends(
    standing_row: dict[str, Any] | None,
    recent_games: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    recent = recent_games or []
    wins = sum(1 for g in recent if g.get("result") == "W")
    losses = sum(1 for g in recent if g.get("result") == "L")
    return {
        "streak": (standing_row or {}).get("streak"),
        "last_5": "".join(g.get("result", "?") for g in recent[:5]),
        "last_5_record": f"{wins}-{losses}" if recent else None,
        "win_percent": (standing_row or {}).get("win_percent"),
        "games_behind": (standing_row or {}).get("games_behind"),
        "point_differential": (standing_row or {}).get("point_differential"),
    }


def build_projection(standing_row: dict[str, Any] | None, power_rating: float | None = None) -> dict[str, Any]:
    row = standing_row or {}
    win_pct = row.get("win_percent")
    projected_wins = None
    if win_pct is not None:
        try:
            projected_wins = round(float(win_pct) * 82, 1)  # default 82-game season proxy
        except (TypeError, ValueError):
            projected_wins = None
    return {
        "playoff_seed": row.get("playoff_seed"),
        "clincher": row.get("clincher"),
        "projected_wins_pace": projected_wins,
        "power_rating": power_rating,
        "rank": row.get("rank"),
    }


def _parse_stat_categories(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    categories: list[dict[str, Any]] = []
    raw_categories = payload.get("categories")
    if not raw_categories:
        results = payload.get("results") or {}
        stats_root = results.get("stats") if isinstance(results.get("stats"), dict) else {}
        raw_categories = stats_root.get("categories") or []
    for cat in raw_categories:
        stats = []
        for stat in cat.get("stats") or []:
            stats.append(
                {
                    "name": stat.get("displayName") or stat.get("name"),
                    "short_name": stat.get("shortDisplayName") or stat.get("abbreviation"),
                    "display": stat.get("displayValue"),
                    "value": stat.get("value"),
                    "rank": stat.get("rank"),
                }
            )
        categories.append(
            {
                "name": cat.get("displayName") or cat.get("name"),
                "stats": stats,
            }
        )
    return categories


def parse_player_overview_stats(overview: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not overview:
        return []
    statistics = overview.get("statistics") or {}
    splits = statistics.get("splits") or []
    categories: list[dict[str, Any]] = []
    for split in splits:
        labels = split.get("labels") or split.get("displayNames") or []
        stats = split.get("stats") or []
        rows = []
        if labels and stats and isinstance(stats[0], list):
            for idx, label in enumerate(labels):
                value = stats[0][idx] if stats[0] and idx < len(stats[0]) else None
                rows.append({"name": label, "display": value, "value": value})
        else:
            for item in stats:
                if isinstance(item, dict):
                    rows.append(
                        {
                            "name": item.get("displayName") or item.get("name"),
                            "display": item.get("displayValue"),
                            "value": item.get("value"),
                        }
                    )
        categories.append(
            {
                "name": split.get("displayName") or statistics.get("displayName") or "Season",
                "stats": rows,
            }
        )
    return categories


def _parse_at_vs(at_vs: str | None) -> tuple[str, str | None, str]:
    """Return (location, opponent_abbr, opponent_label) from ESPN atVs text."""
    raw = (at_vs or "").strip()
    if not raw:
        return "", None, ""
    if raw.startswith("@"):
        abbr = raw[1:].strip()
        return "away", abbr.lower() if abbr else None, abbr
    if raw.lower().startswith("vs"):
        abbr = raw[2:].strip().lstrip(".")
        return "home", abbr.lower() if abbr else None, abbr
    if len(raw) <= 5 and raw.isalpha():
        return "", raw.lower(), raw.upper()
    return "", None, raw


def parse_player_game_log(overview: dict[str, Any] | None, limit: int = 10) -> list[dict[str, Any]]:
    game_log = (overview or {}).get("gameLog") or {}
    events = game_log.get("events") or {}
    rows: list[dict[str, Any]] = []
    for event_id, event in events.items():
        location, opp_abbr, opp_label = _parse_at_vs(event.get("atVs"))
        score_text = (event.get("score") or "").strip()
        result = score_text[:1].upper() if score_text[:1].upper() in {"W", "L", "T"} else None
        rows.append(
            {
                "event_id": event_id,
                "date": event.get("gameDate"),
                "opponent": opp_label,
                "opponent_abbr": opp_abbr,
                "location": location,
                "result": result,
                "score": score_text[2:].strip() if result and len(score_text) > 2 else score_text,
                "home_score": event.get("homeTeamScore"),
                "away_score": event.get("awayTeamScore"),
                "stats": event.get("stats"),
            }
        )
    rows.sort(key=lambda row: row.get("date") or "", reverse=True)
    return rows[:limit]


def parse_player_news(overview: dict[str, Any] | None, limit: int = 5) -> list[dict[str, Any]]:
    articles = (overview or {}).get("news") or []
    rows: list[dict[str, Any]] = []
    for article in articles[:limit]:
        rows.append(
            {
                "headline": article.get("headline"),
                "description": article.get("description"),
                "published": article.get("published"),
                "link": (article.get("links") or {}).get("web", {}).get("href"),
            }
        )
    return rows


def roster_index_row(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": player.get("id"),
        "name": player.get("name"),
        "position": player.get("position"),
        "jersey": player.get("jersey"),
        "status": player.get("status"),
        "headshot": player.get("headshot"),
        "age": player.get("age"),
        "experience": player.get("experience"),
    }


def build_player_roster_snapshot(
    *,
    league: str,
    player_id: str,
    roster_row: dict[str, Any] | None,
    team_abbr: str,
) -> dict[str, Any]:
    roster_row = roster_row or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "league": league,
        "team_abbr": team_abbr,
        "profile_depth": "roster",
        "player": {
            "id": player_id,
            "name": roster_row.get("name"),
            "position": roster_row.get("position"),
            "jersey": roster_row.get("jersey"),
            "age": roster_row.get("age"),
            "height": roster_row.get("height"),
            "weight": roster_row.get("weight"),
            "experience": roster_row.get("experience"),
            "status": roster_row.get("status"),
            "headshot": roster_row.get("headshot"),
        },
        "season_stats": [],
        "overview_stats": [],
        "game_log": [],
        "news": [],
    }


def build_player_snapshot(
    *,
    league: str,
    player_id: str,
    roster_row: dict[str, Any] | None,
    overview: dict[str, Any] | None,
    stats_payload: dict[str, Any] | None,
    team_abbr: str,
) -> dict[str, Any]:
    roster_row = roster_row or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "league": league,
        "team_abbr": team_abbr,
        "profile_depth": "full",
        "player": {
            "id": player_id,
            "name": roster_row.get("name"),
            "position": roster_row.get("position"),
            "jersey": roster_row.get("jersey"),
            "age": roster_row.get("age"),
            "height": roster_row.get("height"),
            "weight": roster_row.get("weight"),
            "experience": roster_row.get("experience"),
            "status": roster_row.get("status"),
            "headshot": roster_row.get("headshot"),
        },
        "season_stats": _parse_stat_categories(stats_payload),
        "overview_stats": parse_player_overview_stats(overview),
        "game_log": parse_player_game_log(overview),
        "news": parse_player_news(overview),
        "next_game": (overview or {}).get("nextGame"),
        "fantasy": (overview or {}).get("fantasy"),
        "awards": (overview or {}).get("awards"),
    }
