"""Validate GitHub Actions workflow YAML still parses."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGES_YML = PROJECT_ROOT / ".github" / "workflows" / "pages.yml"


def test_pages_yml_is_valid_yaml() -> None:
    yaml = pytest.importorskip("yaml")
    assert PAGES_YML.is_file()
    text = PAGES_YML.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    assert payload.get("name") == "Deploy GitHub Pages"
    jobs = payload.get("jobs") or {}
    assert "build" in jobs
    assert "deploy" in jobs
    build = jobs["build"]
    assert build.get("timeout-minutes") == 45
    build_steps = build.get("steps") or []
    assert any(
        isinstance(step, dict) and "build_gh_pages.py" in str(step.get("run", ""))
        for step in build_steps
    )
    # Trigger keys should still be present in the raw file even if PyYAML
    # coerces the reserved word `on` oddly depending on loader settings.
    assert "\n  schedule:" in text or "\nschedule:" in text
    assert "workflow_dispatch:" in text
    assert "cron:" in text

    cache_paths = [
        str(step.get("with", {}).get("path", ""))
        for step in build_steps
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/cache")
    ]
    joined = "\n".join(cache_paths)
    for sport_cache in (
        "nba-v2-live",
        "wnba-v2-live",
        "nfl-v2-live",
        "cfb-v2-live",
        "cbb-v2-live",
        "nhl-v2-live",
        "soccer-v2-live",
    ):
        assert sport_cache in joined
