"""Normalize ESPN payloads into stable database JSON shapes."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1


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


def parse_roster(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    athletes = (payload or {}).get("athletes") or []
    players: list[dict[str, Any]] = []
    for athlete in athletes:
        pos = athlete.get("position") or {}
        players.append(
            {
                "id": athlete.get("id"),
                "name": athlete.get("displayName") or athlete.get("fullName"),
                "position": pos.get("abbreviation") or pos.get("name"),
                "jersey": athlete.get("jersey"),
                "age": athlete.get("age"),
                "height": athlete.get("displayHeight"),
                "weight": athlete.get("displayWeight"),
                "experience": (athlete.get("experience") or {}).get("years"),
                "status": (athlete.get("status") or {}).get("name"),
                "headshot": (athlete.get("headshot") or {}).get("href"),
            }
        )
    return players


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
