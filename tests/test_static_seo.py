"""Static site SEO assets, SPA 404 wiring, and Pages build helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STATIC_DIR = PROJECT_ROOT / "web" / "static"
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"
PYTEST_INI = PROJECT_ROOT / "pytest.ini"


def _load_build_gh_pages():
    path = PROJECT_ROOT / "scripts" / "build_gh_pages.py"
    spec = importlib.util.spec_from_file_location("build_gh_pages", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_index_html_has_favicon_and_social_meta() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'rel="icon"' in html
    assert "favicon.svg" in html
    assert 'property="og:title"' in html
    assert 'name="twitter:card"' in html
    assert 'rel="canonical"' in html
    assert (STATIC_DIR / "favicon.svg").is_file()
    favicon = (STATIC_DIR / "favicon.svg").read_text(encoding="utf-8")
    assert "SOA" in favicon
    assert "#2dd4bf" in favicon


def test_spa_app_js_has_known_route_404() -> None:
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "KNOWN_ROUTES" in js
    assert "function viewNotFound" in js
    assert "function isKnownRoute" in js
    assert "empty-panel--not-found" in js
    assert "Page not found" in js
    # Unknown paths must not silently fall through to the dashboard.
    assert "if (!isKnownRoute(route))" in js


def test_pages_404_html_redirects_into_hash_router() -> None:
    html = (STATIC_DIR / "404.html").read_text(encoding="utf-8")
    assert "Sports-Odds-Algorithms" in html
    assert "location.replace" in html


def test_agents_md_documents_cors_allow_origins() -> None:
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert "CORS_ALLOW_ORIGINS" in text
    assert "cors_allow_origins" in text
    assert "CORS_ALLOW_ORIGINS=*" in text


def test_agents_md_documents_basketball_market_aware_helper() -> None:
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert "basketball_v2_market" in text
    assert "market-aware" in text


def test_agents_md_documents_daily_build_env_knobs() -> None:
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert "FAST_DAILY_BUILD" in text
    assert "LIVE_MULTI_BOOK" in text
    assert "LIVE_MULTI_BOOK_BUDGET_S" in text
    assert "NEWS_SIGNALS" in text
    assert "line_shopping" in text
    assert "soccer_paper_tracking" in text
    assert "internal" in text.lower()


def test_app_js_honesty_banners_and_parallel_load() -> None:
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "STALE_SLATE_HOURS" in js
    assert "function slateAgeHours" in js
    assert "function isStaleSlate" in js
    assert "function slateStatusBanners" in js
    assert "Partial slate" in js
    assert "Stale board" in js
    assert "Line shopping skipped" in js
    assert "League not ready" in js
    assert "leagues_not_ready" in js
    assert "Stale live inputs" in js
    assert "Promise.allSettled" in js
    assert "Soccer paper log is internal-only" in js
    assert "Research only — not betting advice" in js
    assert "No guaranteed profits" in js
    assert "status-banner" in (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    # ncaabb is NCAA baseball — not a GB-v2 predictions-only football/cbb league.
    assert 'PREDICTIONS_ONLY_LEAGUES = new Set(["nfl", "cfb", "cbb"])' in js
    assert '["nba", "wnba", "cbb", "nfl", "cfb"]' in js
    assert "ncaabb" not in js.split("PREDICTIONS_ONLY_LEAGUES")[1].split(";")[0]
    assert "backtested per-league" in js
    assert "function hubacekPickRule(source, game)" in js


def test_index_has_skip_link_and_footer_disclaimer() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    assert 'class="skip-link"' in html
    assert 'href="#appRoot"' in html
    assert "footer-disclaimer" in html
    assert "No guaranteed profits" in html
    assert ".skip-link" in css
    assert ".footer-disclaimer" in css


def test_agents_md_documents_dev_check() -> None:
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert "scripts/dev_check.py" in text
    assert "--quick-only" in text
    assert "--with-v2" in text
    assert "--compile" in text
    assert "--full" in text
    assert "--fail-fast" in text
    assert "mutually exclusive" in text


def test_agents_md_documents_pages_multi_book_override() -> None:
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert "LIVE_MULTI_BOOK=1" in text
    assert "sharpsheettips.com" in text
    assert "EST" in text or "est" in text.lower()


def test_pytest_ini_defines_slow_marker() -> None:
    text = PYTEST_INI.read_text(encoding="utf-8")
    assert "markers =" in text
    assert "slow:" in text
    assert "filterwarnings" in text


def test_copy_static_assets_writes_sitemap_robots_and_favicon(tmp_path, monkeypatch) -> None:
    pages = _load_build_gh_pages()
    monkeypatch.setattr(pages, "DOCS_DIR", tmp_path)
    pages.copy_static_assets()

    assert (tmp_path / "favicon.svg").is_file()
    assert (tmp_path / "404.html").is_file()
    assert (tmp_path / "robots.txt").is_file()
    assert (tmp_path / "sitemap.xml").is_file()
    assert (tmp_path / ".nojekyll").is_file()

    robots = (tmp_path / "robots.txt").read_text(encoding="utf-8")
    assert "Sitemap:" in robots
    assert "sitemap.xml" in robots

    sitemap = (tmp_path / "sitemap.xml").read_text(encoding="utf-8")
    assert "urlset" in sitemap
    assert "samuellachance.github.io/Sports-Odds-Algorithms/" in sitemap
    assert "#/picks" in sitemap
    assert "#/tracking" in sitemap

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'meta name="base-path" content="/Sports-Odds-Algorithms"' in index
    assert 'href="favicon.svg"' in index
    assert "/static/favicon.svg" not in index


def test_cors_env_comma_list(monkeypatch) -> None:
    from web.app import cors_allow_origins

    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS",
        "https://example.com, http://127.0.0.1:3000 ,",
    )
    assert cors_allow_origins() == [
        "https://example.com",
        "http://127.0.0.1:3000",
    ]


def test_favicon_route_is_registered() -> None:
    from web.app import STATIC_DIR, app, favicon_svg

    assert (STATIC_DIR / "favicon.svg").is_file()
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/favicon.svg" in paths
    response = favicon_svg()
    assert response.path == STATIC_DIR / "favicon.svg"


def test_app_js_mlb_nhl_frontend_contracts() -> None:
    """SPA contracts that previously broke MLB/NHL board UX."""
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    # Nested hash fragments must not poison the router path.
    assert 'raw.split("#")[0]' in js
    assert 'href="#/picks#model-predictions"' not in js
    # Honest EV must invert home-only fallbacks for away moneylines only.
    assert 'top?.side === "away"' in js
    assert 'betType === "moneyline"' in js
    assert 'betType === "spread"' in js
    assert "base_win_probability" in js
    # Three-track calibrated probability is labeled EV prob (not Honest EV %).
    assert "Display · ${gateLabel} · EV prob" in js
    assert "<span>EV prob</span>" in js
    # Games sidebar must not leak onto non-games routes.
    assert 'route?.path !== "games"' in js
    assert "sidebarGames.hidden = true" in js
    # Game cards must navigate (clickable is wired, not cosmetic-only).
    assert ".game-card.clickable[data-game]" in js
    assert "navigate(`#/game/${eventId}`)" in js
    # Null-safe matchup access on dense MLB/NHL slates.
    assert "game.matchup?.away" in js
    assert "game.model || {}" in js
    # Wave7 a11y / soft-fail contracts.
    assert 'aria-current", "page"' in js or 'setAttribute("aria-current", "page")' in js
    assert 'event.key !== "Escape"' in js
    assert "trackingLoadFailed" in js
    assert "reused cached live feeds" in js
    assert "Number.isNaN(parsed.getTime())" in js
    assert "slateStatusBanners(slate)" in js
    # Games view must surface the same honesty banners as home/picks.
    assert "function viewGames" in js
    games_fn = js.split("function viewGames")[1].split("function ")[0]
    assert "slateStatusBanners" in games_fn
