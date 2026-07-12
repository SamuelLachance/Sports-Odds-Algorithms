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
