"""Rebuild all serve-time data: run this on a schedule to keep the site current.

Order matters — team form is fetched once and reused:
  1. this season's completed games      -> data/season_2026_finals.json
  2. board (30-day slate + projections) -> site/data/board.json
  3. database (standings + pitchers)    -> site/data/db.json
  4. static SPA shell                    -> site/index.html

Needs only the frozen model (mlbwp/artifacts/ratings.json), the MLB Stats API, and
the Python standard library — no Retrosheet, no third-party packages — so it runs in
a minimal CI job. Market-free throughout.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from mlbwp import db as db_mod
from mlbwp import predict_slate
from mlbwp.live import season_finals
from mlbwp.serve import Predictor
from mlbwp_site import build_site

PROJECT = Path(__file__).resolve().parent
FINALS = PROJECT / "data" / "season_2026_finals.json"


def main(horizon_days: int = 30) -> int:
    pred = Predictor()
    season = pred.serve_season
    # Anchor "today" to the US Eastern game day so it matches MLB's schedule
    # regardless of where this runs (the CI cron is UTC).
    today = datetime.now(ZoneInfo("America/New_York")).date()

    # 1. refresh this season's finals so team form is current through yesterday
    finals = season_finals(season, end_date=(today - timedelta(days=1)).isoformat())
    FINALS.parent.mkdir(parents=True, exist_ok=True)
    FINALS.write_text(json.dumps(finals), encoding="utf-8")
    print(f"[refresh] {len(finals)} completed {season} games -> {FINALS.name}", flush=True)

    # 2. board  3. db  4. shell
    predict_slate.main(days=horizon_days, today=today.isoformat())
    db_mod.main(season=season)
    build_site.build()
    print("[refresh] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
