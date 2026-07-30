"""Weekly NHL player-data refresh chain — keeps player ratings and xG current
in-season (the daily refresh only appends the spine and re-serves).

Runs, in order, each as a failure-tolerated subprocess (print + continue):

  1. nhl_update.py       spine append + upcoming slate (stdlib)
  2. nhl_pull_shifts.py  resumable shift-chart pull (local-only artifact)
  3. nhl_xg_update.py    current-season MoneyPuck re-pull (team/goalie/shot xG)
  4. nhl_rapm2.py        stint RAPM skater ratings (needs shifts + numpy/sklearn)
  5. nhl_names.py        player-id -> name/pos/team map from current rosters
  6. nhl_serve.py        site/data/nhl.json

data/nhl_shifts.csv (~536 MB) is untracked/local-only, so in CI steps 2 and 4
are skipped with a message and the committed nhl_rapm2_ratings.json stays as-is;
RAPM only rebuilds where the shifts artifact exists.

Market-blind: no odds sources anywhere in this chain.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SHIFTS = PROJECT / "data" / "nhl_shifts.csv"

STEPS = [                       # (script, timeout_s)
    ("phase0/nhl_update.py", 900),
    ("phase0/nhl_pull_shifts.py", 3600),   # early-season backlog can be long
    ("phase0/nhl_xg_update.py", 900),
    ("phase0/nhl_rapm2.py", 3600),  # the long step: full 536 MB shift parse +
                                    # stint reconstruction + ridge, NOT
                                    # resumable — a kill loses the whole fit
    ("phase0/nhl_names.py", 900),
    ("phase0/nhl_serve.py", 900),
]
NEED_SHIFTS = {"phase0/nhl_pull_shifts.py", "phase0/nhl_rapm2.py"}


def main() -> int:
    for step, tmo in STEPS:
        if step in NEED_SHIFTS and not SHIFTS.exists():
            print(f"[nhl_weekly] {step}: SKIPPED (data/nhl_shifts.csv absent — "
                  "local-only artifact; keeping committed ratings)", flush=True)
            continue
        try:
            r = subprocess.run([sys.executable, step], capture_output=True,
                               text=True, cwd=str(PROJECT), timeout=tmo)
            ok = r.returncode == 0
            # on failure prefer stderr — that's where the traceback lives;
            # stdout's last progress line would mask the actual error
            src = r.stdout if ok else (r.stderr or r.stdout)
            tail = (src or "").strip().splitlines()
            print(f"[nhl_weekly] {step}: {'ok' if ok else 'FAILED (continuing)'} "
                  f"({tail[-1][:100] if tail else ''})", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[nhl_weekly] {step}: FAILED (timeout after {tmo}s, child "
                  "killed; continuing — this step restarts from scratch next "
                  "run unless it checkpoints its own progress)", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[nhl_weekly] {step}: FAILED ({ex}; continuing)", flush=True)
    print("[nhl_weekly] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
