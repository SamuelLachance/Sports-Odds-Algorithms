"""External database player/team ratings (not derived from our stats)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.bet_advisor import soccer_threeway_probs
from web.league_profiles import (
    LEAGUE_PROFILES,
    SOCCER_DRAW_BASE,
    is_soccer_league,
)
from web.season_games import load_league_completed_games

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "db_ratings_cache"
CACHE_TTL = timedelta(hours=12)

DB_BASE_WEIGHT = 0.12
DB_SPARSE_BOOST = 0.28
SPARSE_GAME_REFERENCE = 15

USER_AGENT = "Sports-Odds-Algorithms/2.0"


def _category(league: str) -> str:
    return LEAGUE_PROFILES[league.lower()]["category"]


def _fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def _cache_path(source: str, key: str) -> Path:
    safe = re.sub(r"[^a-z0-9._-]+", "_", key.lower())
    return CACHE_DIR / source / f"{safe}.json"


def _read_cache(source: str, key: str) -> dict[str, Any] | None:
    path = _cache_path(source, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    fetched = payload.get("fetched_at")
    if not fetched:
        return None
    ts = datetime.fromisoformat(fetched)
    if datetime.now(timezone.utc) - ts > CACHE_TTL:
        return None
    return payload


def _write_cache(source: str, key: str, payload: dict[str, Any]) -> None:
    path = _cache_path(source, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _top_n_average(values: list[int], n: int = 8) -> float | None:
    if not values:
        return None
    top = sorted(values, reverse=True)[:n]
    return sum(top) / len(top)


def _extract_ovr_values(html: str) -> list[int]:
    ovrs = [int(x) for x in re.findall(r"\b(\d{2,3})\s*OVR\b", html, re.I)]
    if ovrs:
        return ovrs
    return [int(x) for x in re.findall(r">(\d{2})<", html) if 50 <= int(x) <= 99]


def _rating_diff_to_home_prob(home_rating: float, away_rating: float, scale: float) -> float:
    if scale <= 0:
        return 50.0
    diff = home_rating - away_rating
    prob = 1.0 / (1.0 + 10.0 ** (-diff / scale))
    return min(max(prob * 100.0, 1.0), 99.0)


def _soccer_rating_prob(home_rating: float, away_rating: float) -> float:
    return _rating_diff_to_home_prob(home_rating, away_rating, scale=400.0)


def _ovr_rating_prob(home_rating: float, away_rating: float) -> float:
    return _rating_diff_to_home_prob(home_rating, away_rating, scale=16.0)


@lru_cache(maxsize=1)
def _soccer_rating_homepage_index() -> dict[str, float]:
    """Parse soccer-rating.com homepage for lineup strength proxies."""
    try:
        html = _fetch_html("https://www.soccer-rating.com/")
    except (urllib.error.URLError, TimeoutError):
        return {}

    index: dict[str, float] = {}
    for match in re.finditer(
        r"Tip:\s*([^<(]+?)\s*\(([A-Z0-9]{2,4})\).*?Rating\s*(-?\d+)/(-?\d+)",
        html,
        re.S,
    ):
        team_label = match.group(1).strip()
        rating_pair = abs(int(match.group(3))) + abs(int(match.group(4)))
        index[team_label.lower()] = float(1500 + rating_pair * 8)

    for block in re.finditer(
        r"([A-Za-z][^<\n|]{2,40})\s+(\d{2})\s+(\d{2})",
        html,
    ):
        name = block.group(1).strip()
        if len(name) < 3 or name.lower() in {"rating", "lineup", "value"}:
            continue
        avg = (int(block.group(2)) + int(block.group(3))) / 2.0
        index[name.lower()] = 1200.0 + avg * 10.0

    return index


def _scrape_soccer_team_rating(team_name: str) -> dict[str, Any] | None:
    key = team_name.lower()
    cached = _read_cache("soccer-rating", key)
    if cached:
        return cached

    homepage = _soccer_rating_homepage_index()
    for label, rating in homepage.items():
        if key in label or label in key:
            payload = {
                "source": "soccer-rating.com",
                "team": team_name,
                "rating": rating,
                "method": "homepage_lineup_index",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_cache("soccer-rating", key, payload)
            return payload

    slug = urllib.parse.quote(team_name)
    try:
        html = _fetch_html(f"https://www.soccer-rating.com/{slug}/1")
    except (urllib.error.URLError, TimeoutError):
        return None

    if team_name.lower() not in html.lower():
        return None

    total = re.search(r"Rating Total:\s*</td>\s*<td[^>]*>\s*([\d.]+)", html)
    home = re.search(r"Rating Home:\s*</td>\s*<td[^>]*>\s*([\d.]+)", html)
    away = re.search(r"Rating Away:\s*</td>\s*<td[^>]*>\s*([\d.]+)", html)
    if not total:
        return None

    payload = {
        "source": "soccer-rating.com",
        "team": team_name,
        "rating": float(total.group(1)),
        "rating_home": float(home.group(1)) if home else None,
        "rating_away": float(away.group(1)) if away else None,
        "method": "team_page_total",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache("soccer-rating", key, payload)
    return payload


def _scrape_2k_team(slug: str, *, wnba: bool = False) -> dict[str, Any] | None:
    source = "2kratings-wnba" if wnba else "2kratings-nba"
    cached = _read_cache(source, slug)
    if cached:
        return cached

    prefix = "/wnba-teams" if wnba else "/teams"
    try:
        html = _fetch_html(f"https://www.2kratings.com{prefix}/{slug}")
    except (urllib.error.URLError, TimeoutError):
        return None

    team_match = re.search(r"Team Overall Rating of (\d+)", html)
    if team_match:
        rating = float(team_match.group(1))
        method = "team_overall"
    else:
        ovrs = _extract_ovr_values(html)
        avg = _top_n_average(ovrs, 8)
        if avg is None:
            return None
        rating = avg
        method = "top8_player_ovr"

    payload = {
        "source": "2kratings.com",
        "team_slug": slug,
        "rating": rating,
        "method": method,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache(source, slug, payload)
    return payload


def _scrape_madden_team(slug: str) -> dict[str, Any] | None:
    cached = _read_cache("maddenratings", slug)
    if cached:
        return cached

    try:
        html = _fetch_html(f"https://www.maddenratings.com/teams/{slug}")
    except (urllib.error.URLError, TimeoutError):
        return None

    team_match = re.search(r"Team Overall Rating of (\d+)", html)
    if team_match:
        rating = float(team_match.group(1))
        method = "team_overall"
    else:
        ovrs = _extract_ovr_values(html)
        avg = _top_n_average(ovrs, 8)
        if avg is None:
            return None
        rating = avg
        method = "top8_player_ovr"

    payload = {
        "source": "maddenratings.com",
        "team_slug": slug,
        "rating": rating,
        "method": method,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache("maddenratings", slug, payload)
    return payload


def _scrape_theshow_team(slug: str) -> dict[str, Any] | None:
    cached = _read_cache("theshowratings", slug)
    if cached:
        return cached

    try:
        html = _fetch_html(f"https://www.theshowratings.com/teams/{slug}")
    except (urllib.error.URLError, TimeoutError):
        return None

    ovrs = _extract_ovr_values(html)
    rating = _top_n_average(ovrs, 8)
    if rating is None:
        return None

    payload = {
        "source": "theshowratings.com",
        "team_slug": slug,
        "rating": rating,
        "method": "top8_player_ovr",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache("theshowratings", slug, payload)
    return payload


def _scrape_nhl_team(slug: str) -> dict[str, Any] | None:
    cached = _read_cache("nhlratings", slug)
    if cached:
        return cached

    try:
        html = _fetch_html(f"https://www.nhlratings.net/teams/{slug}")
    except (urllib.error.URLError, TimeoutError):
        return None

    team_match = re.search(r"Team Overall Rating of (\d+)", html)
    if team_match:
        rating = float(team_match.group(1))
        method = "team_overall"
    else:
        ovrs = _extract_ovr_values(html)
        rating = _top_n_average(ovrs, 8)
        if rating is None:
            return None
        method = "top8_player_ovr"

    payload = {
        "source": "nhlratings.net",
        "team_slug": slug,
        "rating": rating,
        "method": method,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache("nhlratings", slug, payload)
    return payload


def get_team_db_rating(
    league: str,
    *,
    team_slug: str,
    team_name: str | None = None,
) -> dict[str, Any] | None:
    league = league.lower()
    category = _category(league)

    if category == "soccer":
        return _scrape_soccer_team_rating(team_name or team_slug.replace("-", " ").title())

    if category == "basketball":
        if league == "wnba":
            return _scrape_2k_team(team_slug, wnba=True)
        return _scrape_2k_team(team_slug)

    if category == "football":
        return _scrape_madden_team(team_slug)

    if category == "baseball":
        return _scrape_theshow_team(team_slug)

    if category == "hockey":
        return _scrape_nhl_team(team_slug)

    return None


def _team_game_count(league: str, cutoff_date: str, abbr: str) -> int:
    games = load_league_completed_games(league, cutoff_date)
    abbr = abbr.lower()
    return sum(1 for home, away, *_rest in games if abbr in (home, away))


def sparse_schedule_factor(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> float:
    home_games = _team_game_count(league, cutoff_date, home_abbr)
    away_games = _team_game_count(league, cutoff_date, away_abbr)
    min_games = min(home_games, away_games)
    if min_games >= SPARSE_GAME_REFERENCE:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (min_games / SPARSE_GAME_REFERENCE)))


def db_rating_blend_weight(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    has_db_layer: bool,
) -> float:
    if not has_db_layer:
        return 0.0
    sparse = sparse_schedule_factor(league, cutoff_date, home_abbr, away_abbr)
    return DB_BASE_WEIGHT + sparse * DB_SPARSE_BOOST


def run_db_rating_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_slug: str,
    away_slug: str,
    home_name: str | None = None,
    away_name: str | None = None,
) -> dict[str, Any] | None:
    home_entry = get_team_db_rating(
        league, team_slug=home_slug, team_name=home_name
    )
    away_entry = get_team_db_rating(
        league, team_slug=away_slug, team_name=away_name
    )
    if not home_entry or not away_entry:
        return None

    home_rating = float(home_entry["rating"])
    away_rating = float(away_entry["rating"])
    category = _category(league)

    if is_soccer_league(league):
        home_for_prob = float(home_entry.get("rating_home") or home_rating)
        away_for_prob = float(away_entry.get("rating_away") or away_rating)
        home_prob = _soccer_rating_prob(home_for_prob, away_for_prob)
        draw_base = SOCCER_DRAW_BASE.get(league, SOCCER_DRAW_BASE["default"])
        binary_total = home_prob if home_prob >= 50 else -(100.0 - home_prob)
        home_p, draw_p, away_p = soccer_threeway_probs(binary_total, league)
        threeway = {
            "home_win_probability": round(home_p, 2),
            "draw_probability": round(draw_p, 2),
            "away_win_probability": round(away_p, 2),
        }
    else:
        home_prob = _ovr_rating_prob(home_rating, away_rating)
        threeway = None

    sparse = sparse_schedule_factor(league, cutoff_date, home_abbr, away_abbr)
    return {
        "algorithm": "DB Ratings",
        "source_home": home_entry["source"],
        "source_away": away_entry["source"],
        "home_rating": round(home_rating, 2),
        "away_rating": round(away_rating, 2),
        "home_win_probability": round(home_prob, 2),
        "away_win_probability": round(100.0 - home_prob, 2),
        "threeway": threeway,
        "method_home": home_entry.get("method"),
        "method_away": away_entry.get("method"),
        "sparse_schedule_factor": round(sparse, 3),
        "blend_weight_hint": round(
            db_rating_blend_weight(
                league,
                cutoff_date,
                home_abbr,
                away_abbr,
                has_db_layer=True,
            ),
            3,
        ),
    }


def _redistribute_weights(
    weights: dict[str, float],
    db_weight: float,
) -> dict[str, float]:
    if db_weight <= 0:
        total = sum(weights.values()) or 1.0
        return {key: value / total for key, value in weights.items()}

    remaining = max(0.0, 1.0 - db_weight)
    base_total = sum(weights.values()) or 1.0
    merged = {key: (value / base_total) * remaining for key, value in weights.items()}
    merged["db_rating"] = db_weight
    return merged


def _get_layer_threeway(blended: dict[str, Any], key: str) -> tuple[float, float, float] | None:
    if key == "db_rating":
        layer = blended.get("db_rating") or {}
    elif key == "legacy":
        layer = blended.get("legacy_threeway") or blended.get("legacy") or {}
    elif key == "power":
        layer = blended.get("power_threeway") or blended.get("power") or {}
    else:
        layer = blended.get(key) or {}

    if "draw_probability" not in layer:
        return None
    return (
        float(layer["home_win_probability"]),
        float(layer["draw_probability"]),
        float(layer["away_win_probability"]),
    )


def _get_layer_home_prob(
    blended: dict[str, Any],
    key: str,
    db_payload: dict[str, Any] | None = None,
) -> float | None:
    if key == "db_rating":
        if not db_payload:
            return None
        return float(db_payload["home_win_probability"])
    layer = blended.get(key) or {}
    if layer.get("home_win_probability") is not None:
        return float(layer["home_win_probability"])
    return None


def apply_db_rating_blend(
    blended: dict[str, Any],
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_slug: str,
    away_slug: str,
    home_name: str | None = None,
    away_name: str | None = None,
) -> dict[str, Any]:
    db_payload = run_db_rating_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_slug=home_slug,
        away_slug=away_slug,
        home_name=home_name,
        away_name=away_name,
    )
    if not db_payload:
        return blended

    db_weight = db_rating_blend_weight(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        has_db_layer=True,
    )
    if db_weight <= 0:
        return blended

    result = dict(blended)
    weights = dict(result.get("blend_weights") or {})
    if not weights:
        layers = max(int(result.get("blend_layers") or 2), 2)
        per = 1.0 / layers
        weights = {"legacy": per, "power": per}
        for key in (
            "basketball_pred",
            "baseball_pred",
            "hockey_pred",
            "football_pred",
            "soccer_pred",
        ):
            if result.get(key):
                weights[key] = per
                break

    result["blend_weights"] = _redistribute_weights(weights, db_weight)
    result["db_rating"] = db_payload
    result["blend_layers"] = int(result.get("blend_layers") or len(weights)) + 1

    if result.get("threeway") and db_payload.get("threeway"):
        home_p = draw_p = away_p = 0.0
        for key, weight in result["blend_weights"].items():
            if key == "db_rating":
                triple = (
                    float(db_payload["threeway"]["home_win_probability"]),
                    float(db_payload["threeway"]["draw_probability"]),
                    float(db_payload["threeway"]["away_win_probability"]),
                )
            else:
                triple = _get_layer_threeway(result, key)
            if not triple:
                continue
            home_p += weight * triple[0]
            draw_p += weight * triple[1]
            away_p += weight * triple[2]

        total = home_p + draw_p + away_p
        if total > 0:
            home_p, draw_p, away_p = (
                home_p / total * 100.0,
                draw_p / total * 100.0,
                away_p / total * 100.0,
            )
        from web.soccer_blend import threeway_probs_to_total_score

        total_score, win_prob, favorite = threeway_probs_to_total_score(
            home_p, draw_p, away_p
        )
        result.update(
            {
                "home_win_probability": round(home_p, 2),
                "draw_probability": round(draw_p, 2),
                "away_win_probability": round(away_p, 2),
                "total_score": round(total_score, 2),
                "win_probability": round(win_prob, 2),
                "favorite_side": favorite,
                "blended_threeway": {
                    "home_win_probability": round(home_p, 2),
                    "draw_probability": round(draw_p, 2),
                    "away_win_probability": round(away_p, 2),
                },
            }
        )
    else:
        from web.blend_service import home_win_prob_to_total_score

        home_prob = 0.0
        for key, weight in result["blend_weights"].items():
            layer_home = _get_layer_home_prob(result, key, db_payload)
            if layer_home is None:
                continue
            home_prob += weight * layer_home
        home_prob = min(max(home_prob, 0.0), 100.0)
        total, win_prob = home_win_prob_to_total_score(home_prob)
        result["blended_home_win_probability"] = round(home_prob, 2)
        result["total_score"] = round(total, 2)
        result["win_probability"] = round(win_prob, 2)
        result["favorite_side"] = "home" if total <= 0 else "away"

    sparse = db_payload.get("sparse_schedule_factor", 0)
    boost = " (boosted for sparse schedule)" if sparse and sparse > 0.35 else ""
    source = db_payload.get("source_home", "database")
    note = result.get("blend_note") or ""
    db_note = f"DB player ratings from {source}{boost}."
    result["blend_note"] = f"{note} {db_note}".strip() if note else db_note
    return result
