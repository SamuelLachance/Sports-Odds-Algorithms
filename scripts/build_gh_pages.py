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

DEFAULT_DATES = {
    "nba": "4-16-2017",
    "nhl": "4-12-2017",
    "mlb": "10-25-2016",
}

DEMO_SEASONS = {
    "nba": "2017",
    "nhl": "2017",
    "mlb": "2016",
}

sys.path.insert(0, str(PROJECT_ROOT))

from web.predict_service import (  # noqa: E402
    ALGO_VERSIONS,
    SUPPORTED_LEAGUES,
    get_leagues,
    get_seasons,
    get_teams,
    predict_match,
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def copy_static_assets() -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True)

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
    if league == "mlb":
        return DEFAULT_DATES["mlb"] if season == DEMO_SEASONS["mlb"] else f"10-25-{season}"
    return DEFAULT_DATES[league] if season == DEMO_SEASONS[league] else f"4-16-{season}"


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
        seasons = get_seasons(league) if full_build else [DEMO_SEASONS[league]]

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


def build_daily_slate() -> None:
    from web.daily_service import get_daily_slate

    print("Building daily betting slate...")
    write_json(DOCS_DIR / "api" / "daily-slate.json", get_daily_slate())


def main() -> None:
    copy_static_assets()
    build_api_metadata()
    build_daily_slate()
    built, skipped = build_prediction_cache()
    print(f"GitHub Pages build complete: {DOCS_DIR}")
    print(f"Prediction cache: {built} files written, {skipped} skipped")


if __name__ == "__main__":
    main()
