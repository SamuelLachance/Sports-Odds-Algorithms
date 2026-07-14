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


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _fast_daily_build() -> bool:
    """Skip the precomputed prediction cache (CI daily deploy)."""
    return _env_flag("FAST_DAILY_BUILD")


# Brand canonical origin (custom domain). Hash SPA routes are not distinct URLs
# for crawlers, so the sitemap only lists the real document.
SITE_ORIGIN = "https://sharpsheettips.com"
PAGES_MIRROR = f"https://samuellachance.github.io{BASE_PATH}"


def write_seo_files() -> None:
    """robots.txt + sitemap.xml for the GitHub Pages static site."""
    robots = (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {SITE_ORIGIN}/sitemap.xml\n"
    )
    (DOCS_DIR / "robots.txt").write_text(robots, encoding="utf-8")

    # Fragments (#/picks etc.) are ignored by search engines — only index the document.
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "  <url>\n"
        f"    <loc>{SITE_ORIGIN}/</loc>\n"
        "    <changefreq>daily</changefreq>\n"
        "  </url>\n"
        "  <url>\n"
        f"    <loc>{PAGES_MIRROR}/</loc>\n"
        "    <changefreq>daily</changefreq>\n"
        "  </url>\n"
        "</urlset>\n"
    )
    (DOCS_DIR / "sitemap.xml").write_text(sitemap, encoding="utf-8")


def copy_static_assets() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("index.html", "styles.css", "app.js", "favicon.svg", "404.html"):
        shutil.copy2(STATIC_SRC / name, DOCS_DIR / name)

    index_path = DOCS_DIR / "index.html"
    app_path = DOCS_DIR / "app.js"
    asset_version = str(int(app_path.stat().st_mtime_ns // 1_000_000))
    html = index_path.read_text(encoding="utf-8")
    html = html.replace(
        '<meta name="viewport"',
        f'<meta name="base-path" content="{BASE_PATH}">\n  <meta name="viewport"',
    )
    html = html.replace('href="/static/styles.css"', f'href="styles.css?v={asset_version}"')
    html = html.replace('src="/static/app.js"', f'src="app.js?v={asset_version}"')
    html = html.replace('href="/static/favicon.svg"', "href=\"favicon.svg\"")
    html = html.replace(
        "https://github.com/JamesQuintero/Sports-Odds-Algorithms",
        "https://github.com/SamuelLachance/Sports-Odds-Algorithms",
    )
    index_path.write_text(html, encoding="utf-8")

    write_seo_files()
    (DOCS_DIR / ".nojekyll").touch()


def build_api_metadata() -> None:
    write_json(DOCS_DIR / "api" / "leagues.json", get_leagues())

    if _fast_daily_build():
        print("Skipping per-league teams/seasons cache (FAST_DAILY_BUILD)")
        return

    for league in SUPPORTED_LEAGUES:
        write_json(DOCS_DIR / "api" / "leagues" / league / "teams.json", get_teams(league))
        write_json(DOCS_DIR / "api" / "leagues" / league / "seasons.json", get_seasons(league))


def _season_date(league: str, season: str) -> str:
    if league in DEMO_SEASONS and season == DEMO_SEASONS[league]:
        return DEFAULT_DATES.get(league, f"4-16-{season}")
    if league == "mlb":
        return f"10-25-{season}"
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
    except Exception:  # noqa: BLE001 — one bad matchup must not abort the Pages pool
        return "", None

    if not isinstance(result, dict):
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


def build_daily_slate(*, refresh_sports_db: bool = True) -> dict:
    from web.daily_service import get_daily_slate
    from web.team_service import (
        build_league_sidebar_index,
        build_team_profiles_for_slate,
        build_teams_index,
    )
    from web.tracking_service import update_tracking

    print("Building daily betting slate...")
    slate = get_daily_slate()
    write_json(DOCS_DIR / "api" / "daily-slate.json", slate)

    # Look-ahead slates (48h edge window): model-only predictions publish before
    # lines open; the board's line-watch restacks them client-side when books
    # post. Light artifacts only — no tracking, team profiles, or db patching —
    # and no multi-book/news for future days (lines mostly don't exist yet),
    # keeping the Pages build inside its timeout. LOOKAHEAD_SLATES=0 disables.
    future_slates: list[dict] = []
    if (os.environ.get("LOOKAHEAD_SLATES") or "1").strip().lower() not in ("0", "false"):
        saved_env = {k: os.environ.get(k) for k in ("LIVE_MULTI_BOOK", "NEWS_SIGNALS")}
        os.environ["LIVE_MULTI_BOOK"] = "0"
        os.environ["NEWS_SIGNALS"] = "0"
        try:
            for days_ahead in (1, 2):
                try:
                    print(f"Building look-ahead slate (+{days_ahead}d)...")
                    future = get_daily_slate(days_ahead)
                    write_json(DOCS_DIR / "api" / f"slate-plus{days_ahead}.json", future)
                    future_slates.append(future)
                except Exception as exc:  # noqa: BLE001 — look-ahead must not block deploy
                    print(f"Look-ahead slate +{days_ahead}d failed: {exc}")
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    if _fast_daily_build():
        print("Building lightweight teams index for sidebar (FAST_DAILY_BUILD)")
        write_json(
            DOCS_DIR / "api" / "teams-index.json",
            build_league_sidebar_index(slate),
        )
        print("Skipping slate team profiles (FAST_DAILY_BUILD)")
    else:
        print("Building teams index and slate team profiles...")
        write_json(DOCS_DIR / "api" / "teams-index.json", build_teams_index())
        for rel_key, profile in build_team_profiles_for_slate(slate).items():
            league, abbr = rel_key.split("/", 1)
            write_json(DOCS_DIR / "api" / "team-profiles" / league / f"{abbr}.json", profile)

    print("Updating bet tracking rollups...")
    # Look-ahead qualifying picks (open lines only) become official bets placed
    # in advance; the same dedupe key refreshes them on later runs.
    write_json(DOCS_DIR / "api" / "tracking.json", update_tracking(slate, extra_slates=future_slates))

    print("Grading soccer paper picks...")
    try:
        from web.soccer_paper_tracking import grade_paper_picks

        paper_summary = grade_paper_picks()
        print(
            f"Soccer paper record: {paper_summary['wins']}-{paper_summary['losses']} "
            f"({paper_summary['units']:+.2f}u, {paper_summary['newly_graded']} newly graded)"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Soccer paper grading skipped: {exc}")

    from web.sports_db import build_sports_database
    from web.sports_db.build import TEAM_BUILD_WORKERS

    full_build = _env_flag("FULL_BUILD")
    if refresh_sports_db:
        print(f"Building sports database (fast={not full_build})...")
        build_sports_database(
            DOCS_DIR / "api" / "db",
            slate=slate,
            fast=not full_build,
            max_workers=4,
            team_workers=TEAM_BUILD_WORKERS if _fast_daily_build() else 6,
        )
    else:
        print("Skipping sports DB rebuild (restored API cache)", flush=True)
        from web.sports_db.build import patch_betting_from_slate

        patch_stats = patch_betting_from_slate(DOCS_DIR / "api" / "db", slate)
        print(
            "Patched betting.games_today from fresh slate: "
            f"{patch_stats['leagues_patched']} leagues, "
            f"{patch_stats['teams_patched']} team sheets, "
            f"manifest={'updated' if patch_stats['manifest_updated'] else 'skipped'}",
            flush=True,
        )
    return slate


PAGES_API_CACHE = PROJECT_ROOT / ".build-cache" / "pages-api"


def seed_pages_api_cache() -> bool:
    """Restore prior docs/api tree (teams, db, predict cache) for fast CI deploys."""
    if not PAGES_API_CACHE.is_dir():
        return False
    target = DOCS_DIR / "api"
    if target.is_dir():
        shutil.rmtree(target)
    shutil.copytree(PAGES_API_CACHE, target)
    print(f"Seeded Pages API from cache: {PAGES_API_CACHE}", flush=True)
    return True


def save_pages_api_cache() -> None:
    source = DOCS_DIR / "api"
    if not source.is_dir():
        return
    if PAGES_API_CACHE.is_dir():
        shutil.rmtree(PAGES_API_CACHE)
    shutil.copytree(source, PAGES_API_CACHE)
    print(f"Saved Pages API cache: {PAGES_API_CACHE}", flush=True)


def seed_sports_db_cache() -> bool:
    """Restore prior sports DB output so fast builds can reuse recent-game rows."""
    cache_root = PROJECT_ROOT / ".build-cache" / "sports-db" / "db"
    target = DOCS_DIR / "api" / "db"
    if not cache_root.is_dir():
        return False
    if target.is_dir():
        shutil.rmtree(target)
    shutil.copytree(cache_root, target)
    print(f"Seeded sports DB from cache: {cache_root}")
    return True


def save_sports_db_cache() -> None:
    """Persist sports DB output for the next CI run."""
    source = DOCS_DIR / "api" / "db"
    if not source.is_dir():
        return
    cache_root = PROJECT_ROOT / ".build-cache" / "sports-db" / "db"
    if cache_root.is_dir():
        shutil.rmtree(cache_root)
    shutil.copytree(source, cache_root)
    print(f"Saved sports DB cache: {cache_root}")


def main() -> None:
    fast_build = _fast_daily_build()
    print(f"GitHub Pages build mode: {'fast daily' if fast_build else 'full'}", flush=True)

    copy_static_assets()

    api_cached = seed_pages_api_cache() if fast_build else False
    if api_cached:
        print("Using cached Pages API; refreshing slate + tracking only", flush=True)
    else:
        build_api_metadata()
        if fast_build:
            seed_sports_db_cache()

    refresh_sports_db = not (fast_build and api_cached)
    build_daily_slate(refresh_sports_db=refresh_sports_db)

    if fast_build:
        print("Skipping prediction cache (FAST_DAILY_BUILD)", flush=True)
        built, skipped = 0, 0
    else:
        built, skipped = build_prediction_cache()

    save_sports_db_cache()
    save_pages_api_cache()

    print(f"GitHub Pages build complete: {DOCS_DIR}", flush=True)
    print(f"Prediction cache: {built} files written, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
