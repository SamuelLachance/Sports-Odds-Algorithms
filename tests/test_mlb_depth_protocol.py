"""Build gate for the MLB depth harness: DEV-only, market-blind.

The depth campaign's whole value rests on two promises that are easy to break by
accident and impossible to detect after the fact:

  1. ONE-LOOK PROTOCOL. phase0/mlb_depth_eval.py must never score a season >=
     2016. It is enforced twice -- the Retrosheet loader is capped at
     year_max=DEV_HI so TEST-era game logs are never parsed into the process,
     and the scorer asserts its own mask. Both are checked here.
  2. MARKET-BLIND. No odds term may appear in executable code in either depth
     file (comments/docstrings excluded: the files are allowed to SAY they use
     no odds).

These run without the Retrosheet cache, so they are cheap enough for CI.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

# The minimal CI job is stdlib-only by design: without numpy these checks
# SKIP (same convention as test_nhl_serving / test_edge_ledger) instead of
# breaking collection for the whole suite.
np = pytest.importorskip("numpy")

PHASE0 = Path(__file__).resolve().parents[1] / "phase0"
FILES = [PHASE0 / "mlb_depth_eval.py", PHASE0 / "mlb_depth_axes.py"]

BLOCKLIST = ("odds", "moneyline", "vig", "juice", "sportsbook", "vegas",
             "pinnacle", "draftkings", "fanduel", "betmgm", "closing_line",
             "devig", "kelly", "implied_prob")


def _load():
    sys.path.insert(0, str(PHASE0))
    sys.path.insert(0, str(PHASE0.parent))
    spec = importlib.util.spec_from_file_location(
        "mlb_depth_eval", PHASE0 / "mlb_depth_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mlb_depth_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dev_window_is_the_locked_split():
    m = _load()
    assert (m.DEV_LO, m.DEV_HI) == (2002, 2015)


def test_loader_is_hard_capped_at_dev_hi():
    """The cache builder must pass year_max=DEV_HI, so no TEST-era game log is
    even parsed. This is the outer of the two guards."""
    tree = ast.parse((PHASE0 / "mlb_depth_eval.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_games"]
    assert calls, "build_cache no longer calls load_games"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "year_max" in kw, "load_games called without a year cap"
        assert getattr(kw["year_max"], "id", None) == "DEV_HI"


def test_scorer_refuses_a_test_era_season():
    """The inner guard: even if a TEST row reached the design matrix, scoring it
    must raise rather than silently spend the one TEST look."""
    m = _load()
    n = 400
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.5).astype(float)
    seas = np.full(n, 2010)
    groups = [{2010}]
    m.oof_ll([x], y, seas, groups)          # DEV-only: fine
    seas[7] = 2016
    with pytest.raises(AssertionError):
        m.oof_ll([x], y, seas, groups)


def test_no_market_terms_in_executable_code():
    for path in FILES:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                d = ast.get_docstring(node, clean=False)
                if d:
                    docstrings.add(d)
        code = "\n".join(
            ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        for d in docstrings:
            code = code.replace(d, "")
        for term in BLOCKLIST:
            assert not re.search(rf"\b{term}\b", code, re.I), \
                f"market term {term!r} in executable code of {path.name}"
