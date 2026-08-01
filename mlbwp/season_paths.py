"""Season-scoped MLB data paths, derived from the model's serve season.

`data/season_<year>_finals.json` and `..._lines.json` are this season's raw
inputs: the finals cache the model's rating state is replayed through, and the
lineup cache. Five modules referenced them, and every one hardcoded 2026 —
while `serve_season` has always been dynamic (`train.py` sets it to
`last_season + 1`).

Today those agree, so nothing is broken. They diverge at the next rollover, and
the failure is quiet in the worst way: `predict_slate.main` reads the finals
cache only `if FINALS_CACHE.is_file()`, so a stale `season_2026_finals.json`
sitting on disk would be read as "this season" for 2027. The model's rating
state would be walked through last year's games while the board served this
year's schedule, and the ledger would grade 2027 picks against a finals file
that cannot contain them — so nothing would ever grade.

`db.main(season=...)` is the sharpest illustration: it already ACCEPTS a season
and `refresh.py` already passes the live one, but the body read the hardcoded
path regardless. The parameter was decorative.

Deriving the paths from one place makes the rollover a non-event.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "ratings.json"

_SEASON_RE = re.compile(r"^season_(\d{4})_finals\.json$")


def serve_season() -> int:
    """The season the frozen model serves.

    The artifact is the single source of truth (`train.py` writes it). If it is
    unreadable, fall back to the newest season cache actually on disk rather
    than to a hardcoded year — a wrong constant is what this module exists to
    remove, so it must not reintroduce one as a default.
    """
    try:
        return int(json.loads(_ARTIFACT.read_text(encoding="utf-8"))["serve_season"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    found = [int(m.group(1)) for p in DATA.glob("season_*_finals.json")
             if (m := _SEASON_RE.match(p.name))]
    if found:
        return max(found)
    raise FileNotFoundError(
        f"cannot determine the MLB serve season: {_ARTIFACT} is unreadable and "
        f"no data/season_<year>_finals.json exists")


def finals_cache(season: int | None = None) -> Path:
    """This season's completed games — the replay input and the grading source."""
    return DATA / f"season_{season or serve_season()}_finals.json"


def lines_cache(season: int | None = None) -> Path:
    return DATA / f"season_{season or serve_season()}_lines.json"
