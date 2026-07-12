"""Validate GitHub Actions workflow YAML still parses."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGES_YML = PROJECT_ROOT / ".github" / "workflows" / "pages.yml"
TEST_YML = PROJECT_ROOT / ".github" / "workflows" / "test.yml"


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


def test_test_yml_is_valid_and_lean() -> None:
    yaml = pytest.importorskip("yaml")
    assert TEST_YML.is_file()
    text = TEST_YML.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    assert payload.get("name") == "Tests"
    env = payload.get("env") or {}
    assert env.get("PYTHONDONTWRITEBYTECODE") == "1"
    assert env.get("PIP_DISABLE_PIP_VERSION_CHECK") == "1"
    jobs = payload.get("jobs") or {}
    test_job = jobs["test"]
    assert test_job.get("timeout-minutes") == 20
    steps = test_job.get("steps") or []
    step_names = [step.get("name") for step in steps if isinstance(step, dict)]
    assert "Install Python dependencies" in step_names
    assert "Cache pytest" in step_names
    assert "Compile Python packages" in step_names
    assert "Pytest (full suite)" in step_names
    assert "Smoke test (core algo)" in step_names
    install = next(step for step in steps if step.get("name") == "Install Python dependencies")
    # Skip slow pip self-upgrade; rely on setup-python + pip cache.
    assert "--upgrade pip" not in str(install.get("run", ""))
    compile_step = next(step for step in steps if step.get("name") == "Compile Python packages")
    assert "compileall" in str(compile_step.get("run", ""))
    pytest_step = next(step for step in steps if step.get("name") == "Pytest (full suite)")
    assert "--durations=12" in str(pytest_step.get("run", ""))
    assert "cancel-in-progress: true" in text
    cache_step = next(step for step in steps if step.get("name") == "Cache pytest")
    assert str(cache_step.get("uses", "")).startswith("actions/cache")
    assert ".pytest_cache" in str(cache_step.get("with", {}).get("path", ""))
