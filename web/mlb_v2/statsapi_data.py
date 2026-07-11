"""MLB Stats API fetchers shared by offline history scripts and live inference."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

BASE_URL = "https://statsapi.mlb.com/api/v1"
USER_AGENT = "Sports-Odds-Algorithms/2.0 (mlb-v2)"
PITCHER_CHUNK = 40


def get_json(url: str, *, retries: int = 4, timeout: int = 90) -> dict[str, Any]:
    # stdlib urllib (not requests) so CI needs no extra dependency.
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise OSError(f"Failed after {retries} tries: {url}") from last_error


def parse_ip_outs(raw: Any) -> int:
    """'6.2' innings -> 20 outs."""
    text = str(raw or "0")
    if "." in text:
        whole, _, frac = text.partition(".")
        return int(whole or 0) * 3 + int(frac or 0)
    return int(text or 0) * 3


def fetch_season_games(season: int, *, include_names: bool = False) -> list[dict[str, Any]]:
    url = (
        f"{BASE_URL}/schedule?sportId=1&season={season}&gameType=R"
        "&hydrate=probablePitcher"
    )
    payload = get_json(url)
    games: list[dict[str, Any]] = []
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            status = (game.get("status") or {}).get("codedGameState") or ""
            teams = game.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            home_team = home.get("team") or {}
            away_team = away.get("team") or {}
            if not home_team.get("id") or not away_team.get("id"):
                continue
            row: dict[str, Any] = {
                "gamePk": game.get("gamePk"),
                "date": game.get("officialDate") or str(day.get("date") or ""),
                "status": status,
                "day_night": game.get("dayNight") or "",
                "double_header": game.get("doubleHeader") or "N",
                "game_number": int(game.get("gameNumber") or 1),
                "venue_id": (game.get("venue") or {}).get("id"),
                "home_id": int(home_team["id"]),
                "away_id": int(away_team["id"]),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_pp_id": ((home.get("probablePitcher") or {}).get("id")),
                "away_pp_id": ((away.get("probablePitcher") or {}).get("id")),
            }
            if include_names:
                row["home_pp_name"] = (home.get("probablePitcher") or {}).get("fullName")
                row["away_pp_name"] = (away.get("probablePitcher") or {}).get("fullName")
                row["game_datetime"] = game.get("gameDate")
            games.append(row)
    games.sort(key=lambda g: (g["date"], g.get("gamePk") or 0))
    return games


def fetch_pitcher_logs(season: int, pitcher_ids: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def _chunk(ids: list[int]) -> dict[str, Any]:
        joined = ",".join(str(i) for i in ids)
        url = (
            f"{BASE_URL}/people?personIds={joined}"
            f"&hydrate=stats(group=[pitching],type=[gameLog],season={season})"
        )
        payload = get_json(url)
        out: dict[str, Any] = {}
        for person in payload.get("people") or []:
            pid = person.get("id")
            if not pid:
                continue
            hand = ((person.get("pitchHand") or {}).get("code") or "R").upper()
            logs: list[dict[str, Any]] = []
            for block in person.get("stats") or []:
                for split in block.get("splits") or []:
                    stat = split.get("stat") or {}
                    if str(split.get("season") or "") != str(season):
                        continue
                    logs.append(
                        {
                            "date": split.get("date") or "",
                            "is_home": bool(split.get("isHome")),
                            "team_id": ((split.get("team") or {}).get("id")),
                            "opp_id": ((split.get("opponent") or {}).get("id")),
                            "gs": int(stat.get("gamesStarted") or 0),
                            "outs": parse_ip_outs(stat.get("inningsPitched")),
                            "er": int(stat.get("earnedRuns") or 0),
                            "r": int(stat.get("runs") or 0),
                            "so": int(stat.get("strikeOuts") or 0),
                            "bb": int(stat.get("baseOnBalls") or 0),
                            "h": int(stat.get("hits") or 0),
                            "hr": int(stat.get("homeRuns") or 0),
                            "hbp": int(stat.get("hitByPitch") or 0),
                            "bf": int(stat.get("battersFaced") or 0),
                        }
                    )
            logs.sort(key=lambda item: item["date"])
            out[str(pid)] = {"hand": hand, "logs": logs}
        return out

    chunks = [
        pitcher_ids[i : i + PITCHER_CHUNK] for i in range(0, len(pitcher_ids), PITCHER_CHUNK)
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for chunk_result in pool.map(_chunk, chunks):
            result.update(chunk_result)
    return result


def fetch_team_ids(season: int) -> list[int]:
    payload = get_json(f"{BASE_URL}/teams?sportId=1&season={season}")
    return sorted(int(team["id"]) for team in payload.get("teams") or [] if team.get("id"))


def fetch_team_logs(season: int, team_ids: list[int], group: str) -> dict[str, Any]:
    def _one(team_id: int) -> tuple[int, list[dict[str, Any]]]:
        url = f"{BASE_URL}/teams/{team_id}/stats?stats=gameLog&group={group}&season={season}"
        try:
            payload = get_json(url)
        except OSError:
            return team_id, []
        rows: list[dict[str, Any]] = []
        for block in payload.get("stats") or []:
            for split in block.get("splits") or []:
                stat = split.get("stat") or {}
                row: dict[str, Any] = {"date": split.get("date") or ""}
                if group == "hitting":
                    row.update(
                        {
                            "r": int(stat.get("runs") or 0),
                            "h": int(stat.get("hits") or 0),
                            "hr": int(stat.get("homeRuns") or 0),
                            "d2": int(stat.get("doubles") or 0),
                            "d3": int(stat.get("triples") or 0),
                            "bb": int(stat.get("baseOnBalls") or 0),
                            "so": int(stat.get("strikeOuts") or 0),
                            "ab": int(stat.get("atBats") or 0),
                            "pa": int(stat.get("plateAppearances") or 0),
                            "obp": float(stat.get("obp") or 0.0),
                            "slg": float(stat.get("slg") or 0.0),
                        }
                    )
                else:
                    row.update(
                        {
                            "outs": parse_ip_outs(stat.get("inningsPitched")),
                            "er": int(stat.get("earnedRuns") or 0),
                            "r": int(stat.get("runs") or 0),
                            "so": int(stat.get("strikeOuts") or 0),
                            "bb": int(stat.get("baseOnBalls") or 0),
                            "h": int(stat.get("hits") or 0),
                            "hr": int(stat.get("homeRuns") or 0),
                            "bf": int(stat.get("battersFaced") or 0),
                        }
                    )
                rows.append(row)
        rows.sort(key=lambda item: item["date"])
        return team_id, rows

    result: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for team_id, rows in pool.map(_one, team_ids):
            result[str(team_id)] = rows
    return result
