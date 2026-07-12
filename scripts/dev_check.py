"""Fast local gate: quick pytest, smoke test, optional v2 metadata check.

Usage:
  python scripts/dev_check.py
  python scripts/dev_check.py --quick-only
  python scripts/dev_check.py --with-v2
  python scripts/dev_check.py --full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, argv: list[str]) -> int:
    print(f"\n==> {label}")
    print(" ".join(argv))
    completed = subprocess.run(argv, cwd=PROJECT_ROOT)
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
    args = parser.parse_args(argv)

    py = sys.executable
    failures: list[str] = []

    if args.full:
        code = _run(
            "pytest (full suite)",
            [py, "-m", "pytest", "tests/", "-q", "--tb=short", "--durations=12"],
        )
    else:
        code = _run(
            "pytest (not slow)",
            [
                py,
                "-m",
                "pytest",
                "tests/",
                "-q",
                "--tb=short",
                "-m",
                "not slow",
            ],
        )
    if code != 0:
        failures.append("pytest")

    if not args.quick_only:
        code = _run("smoke_test", [py, "smoke_test.py"])
        if code != 0:
            failures.append("smoke_test")

    if args.with_v2:
        code = _run(
            "check_v2_data",
            [py, str(PROJECT_ROOT / "scripts" / "check_v2_data.py")],
        )
        if code != 0:
            failures.append("check_v2_data")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
