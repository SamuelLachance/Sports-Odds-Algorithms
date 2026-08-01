"""Unit guards on nhl_xg_update.replace_season — the in-place season rewrite.

The failure class these pin: a bug here silently destroys prior-season xG rows
that are only re-obtainable via the full 16-zip MoneyPuck re-pull.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nhl_xg_update as X  # noqa: E402
from nhl_xg_update import replace_season  # noqa: E402


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_replace_keeps_other_seasons(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = "t.csv"
    _write(p, ["nhl_game_id", "season", "team", "xgf"],
           [["2023020001", "2023", "COL", "2.1"],
            ["2024020001", "2024", "DAL", "1.9"],
            ["2025020001", "2025", "CAR", "3.0"]])
    commit = replace_season(p, 2025, [
        {"nhl_game_id": 2025020001, "season": 2025, "team": "CAR", "xgf": 3.5},
        {"nhl_game_id": 2025020002, "season": 2025, "team": "NYR", "xgf": 1.0}])
    # staged, not yet published
    assert _read(p)[3][3] == "3.0"
    commit()
    rows = _read(p)
    assert rows[0] == ["nhl_game_id", "season", "team", "xgf"]
    assert [r[0] for r in rows[1:]] == ["2023020001", "2024020001",
                                        "2025020001", "2025020002"]
    assert rows[3][3] == "3.5"                       # replaced, not duplicated
    assert not os.path.exists(p + ".tmp")


def test_replace_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = "t.csv"
    _write(p, ["nhl_game_id", "season", "team", "xgf"],
           [["2024020001", "2024", "DAL", "1.9"],
            ["2025020001", "2025", "CAR", "3.0"]])
    new = [{"nhl_game_id": 2025020001, "season": 2025, "team": "CAR", "xgf": 3.0}]
    replace_season(p, 2025, new)()
    once = open(p, "rb").read()
    replace_season(p, 2025, new)()
    assert open(p, "rb").read() == once              # run twice -> byte-identical


def test_replace_no_season_column_uses_gid(tmp_path, monkeypatch):
    """nhl_shots.csv has no season column; the 10-digit id prefix is the key."""
    monkeypatch.chdir(tmp_path)
    p = "s.csv"
    _write(p, ["nhl_game_id", "period", "xg"],
           [["2024020009", "1", "0.1"],
            ["2025020009", "2", "0.2"]])
    replace_season(p, 2025, [
        {"nhl_game_id": 2025020009, "period": 2, "xg": 0.25}])()
    rows = _read(p)
    assert [r[0] for r in rows[1:]] == ["2024020009", "2025020009"]
    assert rows[2][2] == "0.25"


def test_staged_commit_is_two_phase(tmp_path, monkeypatch):
    """An exception between staging and commit must leave the file untouched."""
    monkeypatch.chdir(tmp_path)
    p = "t.csv"
    _write(p, ["nhl_game_id", "season", "team", "xgf"],
           [["2025020001", "2025", "CAR", "3.0"]])
    before = open(p, "rb").read()
    commit = replace_season(p, 2025, [
        {"nhl_game_id": 2025020001, "season": 2025, "team": "CAR", "xgf": 9.9}])
    # simulate a crash before commit(): nothing published, tmp cleanly present
    assert open(p, "rb").read() == before
    assert os.path.exists(p + ".tmp")
    del commit


# ---- partial-publish guard (iteration 82) ---------------------------------

def test_a_growing_season_block_is_written():
    assert X.season_block_is_safe(800, 780) is True
    assert X.season_block_is_safe(780, 780) is True, "an unchanged block is normal"


def test_a_partial_moneypuck_publish_is_refused():
    """main() only rejects a FULLY empty parse. A zip carrying 10 of the
    season's 800 games is non-empty, so without this the whole current-season
    block would be replaced by those 10 — and the xG team rating is a shipped
    feature (blend coefficient 0.37079)."""
    assert X.season_block_is_safe(10, 800) is False
    assert X.season_block_is_safe(0, 800) is False


def test_a_re_scored_game_still_writes():
    """MoneyPuck does occasionally re-score a game; two rows per game, so the
    tolerance covers a couple of them without admitting a truncation."""
    assert X.season_block_is_safe(798, 800) is True
    assert X.season_block_is_safe(796, 800) is True     # exactly the tolerance
    assert X.season_block_is_safe(795, 800) is False    # one past it


def test_the_first_block_of_a_season_is_never_blocked():
    """In October the file holds no current-season rows at all.

    This is behaviour worth asserting, but note it needs NO special-case branch:
    row counts are never negative, so the ordinary comparison already admits it
    at any tolerance. An `if n_cur_old <= 0: return True` guard was written and
    then removed — mutation testing could not kill it, because it was dead code.
    """
    for tol in (0, 4):
        assert X.season_block_is_safe(2, 0, tol=tol) is True
        assert X.season_block_is_safe(0, 0, tol=tol) is True


def test_a_refused_stage_returns_a_no_op_commit_not_an_exception():
    """Two-phase staging: main() stages all three files then commits them
    back-to-back. A refusal must leave the other two publishable, so it returns
    a do-nothing commit rather than raising."""
    import inspect
    src = inspect.getsource(X.replace_season)
    block = src.split("if not season_block_is_safe", 1)[1].split("tmp = path", 1)[0]
    code = "\n".join(ln.split("#")[0] for ln in block.splitlines())
    assert "return lambda: None" in code
    assert "raise" not in code
