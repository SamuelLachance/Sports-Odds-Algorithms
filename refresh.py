"""Rebuild all serve-time data: run this on a schedule to keep the site current.

Order matters — team form is fetched once and reused:
  1. this season's completed games      -> data/season_2026_finals.json
  2. board (30-day slate + projections) -> site/data/board.json
  3. database (standings + pitchers)    -> site/data/db.json
  4. static SPA shell                    -> site/index.html

Needs only the frozen model (mlbwp/artifacts/ratings.json), the MLB Stats API, and
the Python standard library — no Retrosheet, no third-party packages — so it runs in
a minimal CI job. The MODEL is market-blind; a final post-process (market/) compares
the live market to the model's OUTPUT to flag value edges above the threshold in
market/__init__.py (needs ODDS_API_KEY,
skipped gracefully without it).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from market import EV_THRESHOLD, edges          # market-comparison layer, OUTSIDE the mlbwp model pkg
from mlbwp import db as db_mod
from mlbwp import season_paths
from mlbwp import predict_slate
from mlbwp.live import finals_write_is_safe, season_finals
from mlbwp.serve import Predictor
from mlbwp_site import build_site

PROJECT = Path(__file__).resolve().parent
FINALS = season_paths.finals_cache()   # season-derived; see mlbwp/season_paths.py

EV_PCT = round(EV_THRESHOLD * 100)   # log copy must never outlive the constant


def _write_atomic(path, text):
    """tmp + os.replace: an interrupted run must never truncate a live file."""
    import os
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, str(path))



def main(horizon_days: int = 30) -> int:
    pred = Predictor()
    season = pred.serve_season
    # Anchor "today" to the US Eastern game day so it matches MLB's schedule
    # regardless of where this runs (the CI cron is UTC).
    today = datetime.now(ZoneInfo("America/New_York")).date()

    # 1. refresh this season's finals so team form is current through yesterday
    finals = season_finals(season, end_date=(today - timedelta(days=1)).isoformat())
    n_prev = len(json.loads(FINALS.read_text())) if FINALS.is_file() else 0
    if finals_write_is_safe(len(finals), n_prev):
        FINALS.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(FINALS, json.dumps(finals))
        print(f"[refresh] {len(finals)} completed {season} games -> {FINALS.name}",
              flush=True)
    else:
        # Completed games never un-complete: a shrunken walk is failed requests,
        # not reality. Writing it would replay the model on an incomplete season
        # and strand the missing games as picks that can never grade.
        #
        # Keep the cached file and CONTINUE rather than aborting: everything
        # downstream reads this file, so the rest of the chain then rebuilds
        # from the last good finals — a board that is a few hours stale but
        # correct, instead of no board at all. The next walk clears the bar on
        # its own once more games complete.
        print(f"[refresh] WARNING: finals walk returned {len(finals)} games vs "
              f"{n_prev} cached — REFUSING to overwrite; rebuilding from the "
              f"cached finals instead", flush=True)

    # 2. board  3. value edges (market post-process)  4. db  5. shell
    predict_slate.main(days=horizon_days, today=today.isoformat())
    n_edges = edges.attach_and_save()
    print(f"[refresh] {n_edges} value edges (>={EV_PCT}% EV vs opening consensus)", flush=True)

    # pre-game prediction ledger: archive today's board probs before games
    # start, grade yesterday's from finals (honest season-to-date tracking)
    from mlbwp import pred_ledger
    st = pred_ledger.update()
    rz = pred_ledger.realized()
    print(f"[refresh] pred ledger: {st['new']} new, {st['graded']} graded, "
          f"{st['n']} total" + (f"; realized 2026: LL {rz['log_loss']} "
          f"acc {rz['acc']:.1%} (n={rz['n']})" if rz else ""), flush=True)
    # the board was written BEFORE this grading pass — refresh only its record
    # block so #/record shows today's results instead of yesterday's
    print(f"[refresh] track record: {predict_slate.attach_record()} graded rows "
          f"-> board.json", flush=True)

    db_mod.main(season=season)

    # NHL: incremental spine update + re-serve. Runs as subprocesses so this
    # module stays importable in the minimal CI job; needs numpy/sklearn, so it
    # degrades to a skip (keeping the committed nhl.json) when they're absent.
    import subprocess
    for step in ("phase0/nhl_update.py", "phase0/nhl_serve.py"):
        r = subprocess.run([sys.executable, step], capture_output=True, text=True,
                           cwd=str(PROJECT), timeout=900)
        tail = (r.stdout or r.stderr or "").strip().splitlines()
        print(f"[refresh] {step}: {'ok' if r.returncode == 0 else 'SKIPPED'} "
              f"({tail[-1][:100] if tail else ''})", flush=True)

    build_site.build()
    print("[refresh] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
