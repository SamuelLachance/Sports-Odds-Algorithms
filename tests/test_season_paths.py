"""Season-scoped MLB data paths.

Five modules hardcoded `season_2026_finals.json` while `serve_season` has always
been dynamic. They agree today, so nothing was broken — they diverge at the next
rollover, quietly: `predict_slate.main` reads the finals cache only
`if FINALS_CACHE.is_file()`, so a leftover 2026 file would be replayed as "this
season" for 2027, and the ledger would grade 2027 picks against a file that
cannot contain them.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlbwp import season_paths as SP  # noqa: E402


def test_serve_season_comes_from_the_model_artifact():
    """train.py writes serve_season = last_season + 1; that artifact is the
    single source of truth, not a constant anyone has to remember to bump."""
    art = json.loads((ROOT / "mlbwp" / "artifacts" / "ratings.json")
                     .read_text(encoding="utf-8"))
    assert SP.serve_season() == int(art["serve_season"])


def test_paths_are_built_from_the_season():
    assert SP.finals_cache(2027).name == "season_2027_finals.json"
    assert SP.lines_cache(2027).name == "season_2027_lines.json"
    assert SP.finals_cache(2026).name == "season_2026_finals.json"


def test_derived_paths_match_todays_shipped_files():
    """The refactor must be inert right now: these are the exact literals that
    were replaced, and the live pipeline reads them every 20 minutes."""
    assert SP.finals_cache() == ROOT / "data" / "season_2026_finals.json"
    assert SP.lines_cache() == ROOT / "data" / "season_2026_lines.json"


def test_fallback_scans_disk_rather_than_defaulting_to_a_constant(monkeypatch, tmp_path):
    """If the artifact is unreadable the season must be recovered from the data
    on disk. Falling back to a hardcoded year would reintroduce exactly the bug
    this module removes."""
    data = tmp_path / "data"
    data.mkdir()
    for y in (2025, 2026, 2027):
        (data / f"season_{y}_finals.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(SP, "_ARTIFACT", tmp_path / "missing.json")
    monkeypatch.setattr(SP, "DATA", data)
    assert SP.serve_season() == 2027, "must pick the NEWEST season present"


def test_no_season_at_all_raises_something_actionable(monkeypatch, tmp_path):
    empty = tmp_path / "data"
    empty.mkdir()
    monkeypatch.setattr(SP, "_ARTIFACT", tmp_path / "missing.json")
    monkeypatch.setattr(SP, "DATA", empty)
    with pytest.raises(FileNotFoundError, match="cannot determine the MLB serve season"):
        SP.serve_season()


def test_a_corrupt_artifact_falls_back_instead_of_crashing_the_refresh(monkeypatch, tmp_path):
    """These paths are module-level constants in five modules, so an exception
    here is an IMPORT failure — it would take the whole refresh down."""
    art = tmp_path / "ratings.json"
    art.write_text("{not json", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "season_2026_finals.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(SP, "_ARTIFACT", art)
    monkeypatch.setattr(SP, "DATA", data)
    assert SP.serve_season() == 2026


# ---- the constants that were rewired --------------------------------------

def test_every_rewired_constant_still_points_at_the_live_file():
    from market import edge_ledger
    from mlbwp import pred_ledger, predict_slate
    import refresh

    want = ROOT / "data" / "season_2026_finals.json"
    assert Path(predict_slate.FINALS_CACHE) == want
    assert Path(predict_slate.LINES_CACHE) == ROOT / "data" / "season_2026_lines.json"
    assert Path(pred_ledger.FINALS) == want
    assert Path(edge_ledger.MLB_FINALS) == want
    assert Path(refresh.FINALS) == want


def test_no_module_hardcodes_a_season_year_any_more():
    """The regression guard. Any new `season_<year>_...json` literal outside
    season_paths.py silently opts that module out of the rollover."""
    offenders = []
    lit = re.compile(r'"season_\d{4}_\w+\.json"')
    for py in list((ROOT / "mlbwp").glob("*.py")) + \
              list((ROOT / "market").glob("*.py")) + [ROOT / "refresh.py"]:
        if py.name == "season_paths.py":
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if lit.search(code):
                offenders.append(f"{py.relative_to(ROOT)}:{i}")
    assert not offenders, f"hardcoded season paths: {offenders}"


def test_db_main_season_argument_actually_reaches_the_path():
    """It accepted a season and ignored it — refresh.py passed the live season
    while the body read a hardcoded 2026 path. Pin the property, not the exact
    expression: the season must be resolved, then used to build the path."""
    import inspect

    from mlbwp import db
    src = inspect.getsource(db.main)
    assert "season = season or pred.serve_season" in src, "season is never resolved"
    assert "season_paths.finals_cache(season)" in src, "the path ignores the season"
    assert src.index("season = season or") < src.index("finals_cache(season)")
