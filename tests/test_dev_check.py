"""Tests for scripts/dev_check.py CLI wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "dev_check.py"


def _load_dev_check():
    spec = importlib.util.spec_from_file_location("dev_check", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_check_quick_only_skips_smoke(monkeypatch) -> None:
    mod = _load_dev_check()
    calls: list[list[str]] = []

    def fake_run(label: str, argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(mod, "_run", fake_run)
    assert mod.main(["--quick-only"]) == 0
    assert len(calls) == 1
    assert "pytest" in calls[0]
    assert "not slow" in calls[0]
    assert not any("smoke_test.py" in " ".join(c) for c in calls)


def test_dev_check_default_runs_smoke(monkeypatch) -> None:
    mod = _load_dev_check()
    calls: list[list[str]] = []

    def fake_run(label: str, argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(mod, "_run", fake_run)
    assert mod.main([]) == 0
    joined = [" ".join(c) for c in calls]
    assert any("pytest" in j and "not slow" in j for j in joined)
    assert any("smoke_test.py" in j for j in joined)


def test_dev_check_with_v2_and_failure(monkeypatch) -> None:
    mod = _load_dev_check()

    def fake_run(label: str, argv: list[str]) -> int:
        if "check_v2_data" in " ".join(argv):
            return 2
        return 0

    monkeypatch.setattr(mod, "_run", fake_run)
    assert mod.main(["--quick-only", "--with-v2"]) == 1


def test_dev_check_rejects_full_and_quick_only() -> None:
    mod = _load_dev_check()
    try:
        mod.main(["--full", "--quick-only"])
        raised = False
    except SystemExit as exc:
        raised = True
        assert exc.code == 2
    assert raised


def test_dev_check_compile_runs_first(monkeypatch) -> None:
    mod = _load_dev_check()
    calls: list[list[str]] = []

    def fake_run(label: str, argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(mod, "_run", fake_run)
    assert mod.main(["--quick-only", "--compile"]) == 0
    assert any("compileall" in c for c in calls)
    assert calls[0][calls[0].index("-m") + 1] == "compileall"


def test_dev_check_full_runs_full_pytest_then_smoke(monkeypatch) -> None:
    mod = _load_dev_check()
    calls: list[list[str]] = []

    def fake_run(label: str, argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(mod, "_run", fake_run)
    assert mod.main(["--full"]) == 0
    joined = [" ".join(c) for c in calls]
    assert any("pytest" in j and "--durations=12" in j and "not slow" not in j for j in joined)
    assert any("smoke_test.py" in j for j in joined)
