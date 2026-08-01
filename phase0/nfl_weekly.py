"""Weekly NFL in-season refresh chain (September 2026 onward).

Two layers, auto-detected (documents/nfl_weekly_datagraph.md section 4):

  FETCH (CI-safe, small tracked files):
    1. data/nfl_games.csv        nfldata games.csv (full rewrite, atomic)
    2. data/inj_2026.csv         nflverse injuries_2026 (404-skip until wk 1)
    3. data/snap_2026.csv        nflverse snap_counts_2026 (404-skip)
    4. data/nfl_player_stats_2026.csv  nflverse stats_player_week_2026 (404-skip)
    5. data/roster_2026.csv      nflverse weekly roster 2026

  CHAIN (local-only — needs the gitignored 50-98MB play tables):
    6. current-season pbp re-pull: delete 2026 rows from the five serve-chain
       reducers, re-run their pull scripts (season-granular done-sets would
       otherwise freeze 2026 after its first pull)
    7. payload chain in order: nfl_site_data -> nfl_site_db ->
       nfl_trueskill_players -> nfl_lineups -> nfl_season_serve
       (ph-freeze ledger keeps played games' pre-game predictions)

Steps degrade gracefully: a missing artifact skips its step with a message, a
failing step prints FAILED with the stderr tail and continues. Market-blind.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "glassbox-nfl/1.0 (research)"}

FETCHES = [
    ("data/nfl_games.csv",
     "https://github.com/nflverse/nfldata/raw/master/data/games.csv", True),
    ("data/inj_2026.csv",
     "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.csv", False),
    ("data/snap_2026.csv",
     "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2026.csv", False),
    ("data/nfl_player_stats_2026.csv",
     "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2026.csv", False),
    ("data/roster_2026.csv",
     "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2026.csv", False),
]

# reducer output -> its pull script (current-season rows deleted, then re-pulled)
REDUCERS = [
    ("data/nfl_plays.csv", "phase0/nfl_pbp_pull.py"),
    ("data/nfl_turnovers.csv", "phase0/nfl_pbp_pull2.py"),
    ("data/nfl_duel_plays.csv", "phase0/nfl_pbp_pull3.py"),
    ("data/nfl_play_ctx2.csv", "phase0/nfl_play_ctx2_pull.py"),
    ("data/nfl_rapm_plays.csv", "phase0/nfl_participation_pull.py"),
]

PAYLOAD_CHAIN = [
    ("phase0/nfl_site_data.py", 1800),
    ("phase0/nfl_site_db.py", 900),
    ("phase0/nfl_trueskill_players.py", 3600),
    ("phase0/nfl_lineups.py", 900),
    ("phase0/nfl_season_serve.py", 1800),
]

CUR = "2026"


SHRINK_TOL = 5          # rows an upstream correction may legitimately remove


def csv_rows(data: bytes) -> int:
    """Data rows in a CSV blob (header excluded); -1 if it does not parse."""
    try:
        n = data.count(b"\n")
    except (TypeError, AttributeError):
        return -1
    if n == 0:
        return -1
    return n - 1 if data.endswith(b"\n") else n


def fetch_is_safe(n_new: int, n_prev: int, tol: int = SHRINK_TOL) -> bool:
    """Whether a freshly fetched table may replace the one on disk.

    These are append-mostly tables: completed games and logged injuries do not
    un-happen, so the row count is monotonically non-decreasing apart from the
    odd upstream correction. A materially SMALLER file is a truncated response,
    not reality.

    This matters most for data/nfl_games.csv, which is a REQUIRED full rewrite
    of the spine the whole NFL model replays from. Byte size alone does not
    catch it: a partial response can be megabytes and still be short thousands
    of rows. A stale-spine tripwire does exist downstream (nfl_season_guards
    .v7_npy_error asks "was nfl_games.csv rolled back?") but it only fires at
    SERVE time, after the bad file has been committed — by then the good copy
    is gone. Refusing the write keeps it.

    `tol` allows the handful of rows a genuine upstream correction removes
    while still blocking a truncation, which loses thousands.

    Only ONE special case, and it is load-bearing: n_new < 0 means csv_rows
    could not count the table, which is not evidence of truncation — without
    this, an uncountable body would compare -1 >= n_prev - tol and be REFUSED.
    A companion `n_prev <= 0` case for the first fetch was written and removed:
    counts are never negative, so `n_new >= 0 - tol` already admits it. Same
    redundancy I had written into nhl_xg_update.season_block_is_safe.
    """
    if n_new < 0:
        return True
    return n_new >= n_prev - tol


def fetch(path: str, url: str, required: bool) -> None:
    tmp = path + ".tmp"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        data = urllib.request.urlopen(req, timeout=120).read()
        if len(data) < 1000:
            raise OSError(f"suspiciously small ({len(data)} bytes)")
        n_new = csv_rows(data)
        n_prev = csv_rows(Path(path).read_bytes()) if os.path.exists(path) else 0
        if not fetch_is_safe(n_new, n_prev):
            # Skip, don't raise: one bad upstream table must not stop the rest
            # of the weekly chain. The previous good file stays, and next week's
            # fetch clears the bar on its own.
            print(f"[nfl_weekly] fetch {path}: REFUSED — {n_new:,} rows vs "
                  f"{n_prev:,} on disk, that is a truncated response, not an "
                  f"upstream correction. Keeping the existing file.", flush=True)
            return
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        print(f"[nfl_weekly] fetch {path}: ok ({len(data):,} bytes, "
              f"{n_new:,} rows)", flush=True)
    except OSError as ex:
        if os.path.exists(tmp):
            os.remove(tmp)
        tag = "FAILED" if required else "skip (not published yet)"
        print(f"[nfl_weekly] fetch {path}: {tag} ({ex})", flush=True)


def strip_current_season(path: str) -> int:
    """Drop season-2026 rows (game_id prefix) so the pull re-fetches them.
    Atomic tmp+replace; returns rows dropped."""
    tmp = path + ".tmp"
    dropped = 0
    with open(path, encoding="utf-8") as src, \
            open(tmp, "w", newline="", encoding="utf-8") as dst:
        rd = csv.reader(src)
        w = csv.writer(dst)
        header = next(rd)
        w.writerow(header)
        gi = header.index("game_id")
        for row in rd:
            if row and row[gi][:4] == CUR:
                dropped += 1
                continue
            w.writerow(row)
    if dropped:
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    return dropped


def run_step(step: str, tmo: int) -> None:
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", step],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           cwd=str(PROJECT), timeout=tmo)
        ok = r.returncode == 0
        src = r.stdout if ok else (r.stderr or r.stdout)
        tail = (src or "").strip().splitlines()
        print(f"[nfl_weekly] {step}: {'ok' if ok else 'FAILED (continuing)'} "
              f"({tail[-1][:100] if tail else ''})", flush=True)
    except subprocess.TimeoutExpired:
        print(f"[nfl_weekly] {step}: FAILED (timeout {tmo}s, killed; continuing)",
              flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[nfl_weekly] {step}: FAILED ({ex}; continuing)", flush=True)


def main() -> int:
    os.chdir(PROJECT)
    for path, url, required in FETCHES:
        fetch(path, url, required)

    have_tables = all(os.path.exists(p) for p, _ in REDUCERS)
    if not have_tables:
        print("[nfl_weekly] play tables absent (CI) — fetch-only run, chain skipped",
              flush=True)
        return 0

    # in-season only: re-pull current-season pbp (done-sets freeze a season
    # after first pull, so stale 2026 rows must be dropped first)
    season_started = any(
        r["game_id"][:4] == CUR and r.get("home_score", "") != ""
        for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")))
    if season_started:
        for path, script in REDUCERS:
            n = strip_current_season(path)
            print(f"[nfl_weekly] {path}: dropped {n:,} season-{CUR} rows", flush=True)
            run_step(script, 3600)
    else:
        print(f"[nfl_weekly] no {CUR} finals yet — pbp re-pull skipped", flush=True)

    for step, tmo in PAYLOAD_CHAIN:
        run_step(step, tmo)
    print("[nfl_weekly] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
