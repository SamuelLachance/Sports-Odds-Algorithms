"""Build static site + JSON API for GitHub Pages."""

from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
STATIC_SRC = PROJECT_ROOT / "web" / "static"
REPO_NAME = "Sports-Odds-Algorithms"
BASE_PATH = f"/{REPO_NAME}"

sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import DEFAULT_DATES, DEMO_SEASONS, SUPPORTED_LEAGUES  # noqa: E402
from web.predict_service import (  # noqa: E402
    ALGO_VERSIONS,
    get_leagues,
    get_seasons,
    get_teams,
    predict_match,
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def copy_static_assets() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(STATIC_SRC / name, DOCS_DIR / name)

    index_path = DOCS_DIR / "index.html"
    html = index_path.read_text(encoding="utf-8")
    html = html.replace(
        '<meta name="viewport"',
        f'<meta name="base-path" content="{BASE_PATH}">\n  <meta name="viewport"',
    )
    html = html.replace('href="/static/styles.css"', 'href="styles.css"')
    html = html.replace('src="/static/app.js"', 'src="app.js"')
    html = html.replace(
        "https://github.com/JamesQuintero/Sports-Odds-Algorithms",
        "https://github.com/SamuelLachance/Sports-Odds-Algorithms",
    )
    index_path.write_text(html, encoding="utf-8")

    (DOCS_DIR / ".nojekyll").touch()


def build_api_metadata() -> None:
    write_json(DOCS_DIR / "api" / "leagues.json", get_leagues())

    for league in SUPPORTED_LEAGUES:
        write_json(DOCS_DIR / "api" / "leagues" / league / "teams.json", get_teams(league))
        write_json(DOCS_DIR / "api" / "leagues" / league / "seasons.json", get_seasons(league))


def _season_date(league: str, season: str) -> str:
    if league in DEMO_SEASONS and season == DEMO_SEASONS[league]:
        return DEFAULT_DATES.get(league, f"4-16-{season}")
    if league == "mlb":
        return f"10-25-{season}"
    if league in {"mls", "epl", "laliga", "bundesliga", "seriea", "ligue1"}:
        return f"4-15-{season}"
    if league == "cbb":
        return f"3-15-{season}"
    if league == "cfb":
        return f"12-15-{season}"
    return DEFAULT_DATES.get(league, f"4-16-{season}")


def _predict_job(args: tuple[str, str, str, str, str, str]) -> tuple[str, dict | None]:
    sys.path.insert(0, str(PROJECT_ROOT))
    league, away_slug, home_slug, date, season, algo = args
    try:
        from web.predict_service import predict_match as run_predict

        result = run_predict(league, away_slug, home_slug, date, season, algo)
    except (ValueError, KeyError, IndexError, OSError):
        return "", None

    rel = (
        f"api/predict/{league}/{away_slug}/{home_slug}/{date}/{season}/{algo}.json"
    )
    return rel, result


def build_prediction_cache() -> tuple[int, int]:
    """Build precomputed predictions. Default: demo seasons only (fast CI deploy)."""
    full_build = os.environ.get("FULL_BUILD", "").lower() in {"1", "true", "yes"}
    jobs: list[tuple[str, str, str, str, str, str]] = []

    for league in SUPPORTED_LEAGUES:
        teams = get_teams(league)
        slugs = [team["slug"] for team in teams]
        if full_build:
            seasons = get_seasons(league)
        elif league in DEMO_SEASONS:
            seasons = [DEMO_SEASONS[league]]
        else:
            continue

        for season in seasons:
            date = _season_date(league, season)
            for away_slug, home_slug in permutations(slugs, 2):
                for algo in ALGO_VERSIONS:
                    jobs.append((league, away_slug, home_slug, date, season, algo))

    print(f"Building {len(jobs)} prediction files (full_build={full_build})")

    built = 0
    skipped = 0
    workers = min(8, max(1, (os.cpu_count() or 4) - 1))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_predict_job, job) for job in jobs]
        for future in as_completed(futures):
            rel, result = future.result()
            if not rel or result is None:
                skipped += 1
                continue
            write_json(DOCS_DIR / rel, result)
            built += 1

    return built, skipped


def build_world_cup_hub() -> dict:
    from web.world_cup_service import get_world_cup_hub

    print("Building FIFA World Cup 2026 hub (full tournament predictions)...")
    hub = get_world_cup_hub()
    write_json(DOCS_DIR / "api" / "world-cup.json", hub)
    return hub


def build_daily_slate() -> dict:
    from web.daily_service import get_daily_slate
    from web.team_service import build_team_profiles_for_slate, build_teams_index
    from web.tracking_service import update_tracking

    print("Building daily betting slate...")
    slate = get_daily_slate()
    write_json(DOCS_DIR / "api" / "daily-slate.json", slate)

    print("Building teams index and slate team profiles...")
    write_json(DOCS_DIR / "api" / "teams-index.json", build_teams_index())
    for rel_key, profile in build_team_profiles_for_slate(slate).items():
        league, abbr = rel_key.split("/", 1)
        write_json(DOCS_DIR / "api" / "team-profiles" / league / f"{abbr}.json", profile)

    print("Updating bet tracking rollups...")
    write_json(DOCS_DIR / "api" / "tracking.json", update_tracking(slate))
    return slate


def main() -> None:
    copy_static_assets()
    build_api_metadata()
    build_daily_slate()
    build_world_cup_hub()
    built, skipped = build_prediction_cache()
    print(f"GitHub Pages build complete: {DOCS_DIR}")
    print(f"Prediction cache: {built} files written, {skipped} skipped")


if __name__ == "__main__":
    main()
