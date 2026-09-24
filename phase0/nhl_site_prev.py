"""Last season's block for site/data/nhl.json - without computing a TEST metric.

The NHL research protocol locks 2018-19..2025-26 as TEST seasons: nothing may
compute a metric on them. The serve used to recompute the 2025-26 replay's log
loss and accuracy on every run. The numbers equal the published ones today,
but a model refit or an xG re-pull would silently have produced, and
displayed, a NEW TEST-season metric.

So a TEST season's block is frozen ONCE, from the payload the site already
published while that season was current: data/nhl_prev_season_{season}.json
holds the rows with their published pre-game hp and the published n / log
loss / accuracy (copied, never recomputed). The serve emits that file
verbatim. With no frozen file a TEST season gets its rows and NO metrics.

A season outside the TEST window (2026-27 onward, the ones this site served
live) is computed by the serve, each row with its pre-game ledger number.

    python phase0/nhl_site_prev.py <published nhl.json> "<source label>"
"""
from __future__ import annotations

import json
import os
import sys

TEST_STARTS = range(2018, 2026)     # NHL TEST seasons 2018-19 .. 2025-26
ROWS_COLS = ["id", "d", "home", "away", "hp", "hs", "as", "last"]


def is_test(season: int) -> bool:
    return season // 10000 in TEST_STARTS


def season_label(season: int) -> str:
    y = season // 10000
    return f"{y}-{str(y + 1)[2:]}"


def path(season: int) -> str:
    return f"data/nhl_prev_season_{season}.json"


def load(season: int, path_fn=path) -> dict | None:
    try:
        blk = json.load(open(path_fn(season), encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return blk if blk.get("season") == season and blk.get("rows") else None


def replay_note(as_of: str | None) -> str:
    return ("Back-tested replay, not a live record: the model was frozen after this "
            f"season (as of {as_of}) and replayed walk-forward over it. These games "
            "are inside the model's TEST window. Every forecast is EARLY tier "
            "(team ratings only).")


def from_published(payload: dict, source: str) -> dict:
    """The block for the season that was CURRENT in `payload` (a payload the
    site published). Copies the rows and the published metrics; computes
    nothing."""
    season = int(payload["cur_season"])
    rows = [[g["id"], g["d"], g["home"], g["away"], g["hp"], g["hs"], g["as"], g.get("last")]
            for g in payload["schedule"]
            if not g.get("playoff") and g.get("hs") is not None]
    rows.sort(key=lambda r: (r[1], r[0]))
    mc = payload.get("model_card") or {}
    return {
        "season": season, "label": season_label(season), "kind": "replay",
        "note": replay_note(payload.get("as_of")) + (
            f" Numbers as published on the site ({source}); frozen, never recomputed."),
        "n": len(rows),
        "ll": mc.get("cur_season_ll"), "acc": mc.get("cur_season_acc"),
        "rows_cols": ROWS_COLS, "rows": rows,
        "frozen_from": source,
    }


def rows_only(season: int, rows: list, as_of: str | None) -> dict:
    """A TEST season with no frozen file: rows, no metrics (never computed)."""
    return {"season": season, "label": season_label(season), "kind": "replay",
            "note": replay_note(as_of) + (" Its log loss and accuracy are not shown: "
                                          "they were never published and a TEST-season "
                                          "metric is not computed."),
            "n": sum(1 for r in rows if r[5] is not None), "ll": None, "acc": None,
            "rows_cols": ROWS_COLS, "rows": rows}


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    payload = json.load(open(argv[1], encoding="utf-8"))
    blk = from_published(payload, argv[2])
    out = path(blk["season"])
    with open(out + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(blk, fh, separators=(",", ":"))
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {blk['n']} rows (published ll/acc copied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
