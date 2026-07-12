"""Fast local gate: quick pytest, smoke test, optional v2 metadata check.

Usage:
  python scripts/dev_check.py
  python scripts/dev_check.py --quick-only
  python scripts/dev_check.py --with-v2
  python scripts/dev_check.py --full
  python scripts/dev_check.py --compile
  python scripts/dev_check.py --fail-fast
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, argv: list[str]) -> int:
    print(f"\n==> {label}")
    print(" ".join(argv))
    started = time.perf_counter()
    completed = subprocess.run(argv, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - started
    print(f"    ({elapsed:.1f}s)")
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local DX check gate (fast tests + smoke)."
    )
    parser.add_argument(
        "--quick-only",
        action="store_true",
        help="Only run pytest -m 'not slow' (skip smoke_test).",
    )
    parser.add_argument(
        "--with-v2",
        action="store_true",
        help="Also run scripts/check_v2_data.py (fails if metadata missing).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full pytest suite (includes slow tests) then smoke.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also run python -m compileall on web/, scripts/, and root modules (mirrors CI).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing step (default runs remaining steps then summarizes).",
    )
    args = parser.parse_args(argv)

    if args.full and args.quick_only:
        parser.error("--full and --quick-only are mutually exclusive")

    py = sys.executable
    failures: list[str] = []
    gate_started = time.perf_counter()

    def _step(label: str, argv: list[str], name: str) -> bool:
        code = _run(label, argv)
        if code == 0:
            return True
        failures.append(name)
        if args.fail_fast:
            print(f"\nFAILED (fail-fast): {name}", file=sys.stderr)
            return False
        return True

    if args.compile:
        if not _step(
            "compileall",
            [
                py,
                "-m",
                "compileall",
                "-q",
                "web",
                "scripts",
                "algo.py",
                "odds_calculator.py",
                "backtester.py",
                "smoke_test.py",
                "run_server.py",
            ],
            "compileall",
        ):
            return 1

    if args.full:
        ok = _step(
            "pytest (full suite)",
            [py, "-m", "pytest", "tests/", "-q", "--tb=short", "--durations=12"],
            "pytest",
        )
    else:
        ok = _step(
            "pytest (not slow)",
            [
                py,
                "-m",
                "pytest",
                "tests/",
                "-q",
                "--tb=short",
                "--durations=12",
                "-m",
                "not slow",
            ],
            "pytest",
        )
    if not ok:
        return 1

    if not args.quick_only:
        if not _step("smoke_test", [py, "smoke_test.py"], "smoke_test"):
            return 1

    if args.with_v2:
        if not _step(
            "check_v2_data",
            [py, str(PROJECT_ROOT / "scripts" / "check_v2_data.py")],
            "check_v2_data",
        ):
            return 1

    total = time.perf_counter() - gate_started
    if failures:
        print(f"\nFAILED: {', '.join(failures)} ({total:.1f}s total)", file=sys.stderr)
        return 1
    print(f"\nAll requested checks passed. ({total:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
