"""The front end's source text, for tests that pin what the page says.

The SPA used to be one JS string inside mlbwp_site/build_site.py. It now lives in
per-area files (mlbwp_site/js/*.js, mlbwp_site/css/*.css) concatenated by the
builder, so a test that searched build_site.py alone would pass vacuously on
anything that moved. This returns the builder plus every part, in build order.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_SRC = ROOT / "mlbwp_site"


def frontend_src() -> str:
    from mlbwp_site.build_site import CSS_PARTS, JS_PARTS
    parts = [(SITE_SRC / "build_site.py").read_text(encoding="utf-8")]
    parts += [(SITE_SRC / "js" / f"{p}.js").read_text(encoding="utf-8") for p in JS_PARTS]
    parts += [(SITE_SRC / "css" / f"{p}.css").read_text(encoding="utf-8") for p in CSS_PARTS]
    return "\n".join(parts)
