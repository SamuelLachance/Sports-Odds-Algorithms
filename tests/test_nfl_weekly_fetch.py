"""nfl_weekly fetch-layer guards: 404-skip, small-file rejection, season strip."""
from __future__ import annotations

import csv
import io
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nfl_weekly  # noqa: E402


def test_fetch_404_skips_and_cleans_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def boom(req, timeout):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO())
    monkeypatch.setattr(nfl_weekly.urllib.request, "urlopen", boom)
    nfl_weekly.fetch("data.csv", "https://x/none.csv", required=False)
    assert not (tmp_path / "data.csv").exists()
    assert not (tmp_path / "data.csv.tmp").exists()


def test_fetch_rejects_tiny_body(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv").write_text("existing,good,content\n" * 100)
    before = (tmp_path / "data.csv").read_text()

    class FakeResp:
        def read(self):
            return b"Not Found"          # 9-byte junk body with HTTP 200
    monkeypatch.setattr(nfl_weekly.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp())
    nfl_weekly.fetch("data.csv", "https://x/a.csv", required=True)
    assert (tmp_path / "data.csv").read_text() == before   # never clobbered
    assert not (tmp_path / "data.csv.tmp").exists()


def test_strip_current_season(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "plays.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "epa"])
        w.writerows([["2025_01_KC_BUF", "0.1"], ["2026_01_NE_SEA", "0.2"],
                     ["2026_01_NE_SEA", "0.3"], ["2024_09_DAL_PHI", "0.4"]])
    n = nfl_weekly.strip_current_season(str(p))
    assert n == 2
    rows = list(csv.reader(open(p)))
    assert [r[0] for r in rows[1:]] == ["2025_01_KC_BUF", "2024_09_DAL_PHI"]
    # no current-season rows -> file untouched (tmp discarded)
    assert nfl_weekly.strip_current_season(str(p)) == 0
    assert not (tmp_path / "plays.csv.tmp").exists()
