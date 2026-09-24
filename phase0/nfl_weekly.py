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
       nfl_trueskill_players -> nfl_lineups -> nfl_v7_feature_gen ->
       nfl_season_serve -> nfl_results_attach
       (ph-freeze ledger keeps played games' pre-game predictions)

The payload chain builds a STAGING copy (data/nfl_payload_staging.json, passed
to every step as NFL_PAYLOAD) and replaces site/data/nfl.json only after the
last step succeeded AND the staged payload validated (status season, 272
rows, every final in nfl_games.csv attached, played games equal to the
ledger, 32 full teams). Before, each step rewrote the live file in place, so a
failure at step 5 left a schedule-less 'preseason' payload on disk, and the
chain still exited 0.

The pre-game ledger is staged the same way (data/nfl_ph_ledger_staging.json,
passed as NFL_LEDGER) and swapped in TOGETHER with the payload, after a last
check that no played or kicked-off entry changed and none vanished. The serve
used to write the live ledger directly: when the gate then failed, the live
payload kept the old serve while the ledger held the new one, and the CI
results step copied the unpublished serve's receipt onto the published
number. Commit the two files in one commit (a successful run prints the
command).

Schedule: the forecasts only move when this runs locally (CI attaches finals
every 4 h but cannot re-serve). Run it at least Thu ~15:00 ET (after the
first injury report, before TNF), Sat ~12:00 ET (final designations) and
Sun ~10:30 ET (before the early slate). Games that have kicked off are locked
by the freeze (nfl_ph_freeze.has_started), so a run during a slate is safe.

Failure is loud: a missing optional file skips its fetch with a message, but a
failed required fetch, a failed step or a failed validation makes the run exit
1 (after the remaining fetches, so good files are still written). The workflow
commits with `if: always()`, so good fetches are never lost to a red job.
Market-blind.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "phase0"))
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
    # new finals make the frozen v7 feature column one row short per game, and
    # nfl_season_serve refuses to run on a stale column (nfl_season_guards
    # .v7_npy_error). It was never in this chain, so the first in-season run
    # (2026-09-24) stopped at serve and left the site on the July payload.
    ("phase0/nfl_v7_feature_gen.py", 5400),
    ("phase0/nfl_season_serve.py", 1800),
]

CUR = "2026"
# runs last on the staged payload, with the spine this run already fetched
FINAL_STEP = ("phase0/nfl_results_attach.py", 600, ["--no-fetch", "--full"])

LEDGER_LIVE = "data/nfl_ph_ledger.json"
LEDGER_STAGING = "data/nfl_ph_ledger_staging.json"
# what a successful local run publishes; commit them in ONE commit so the
# payload never reaches origin without the ledger that receipts it
PUBLISH = ["site/data/nfl.json", LEDGER_LIVE, "data/nfl_games.csv",
           "data/inj_2026.csv", "data/roster_2026.csv", "data/snap_2026.csv",
           "data/nfl_player_stats_2026.csv", "data/nfl_player_ts.csv",
           "data/nfl_season_2026.json"]


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


def fetch(path: str, url: str, required: bool) -> str:
    """Download one table. Returns "ok", "refused" (truncation guard kept the
    file on disk), "failed" (required table unreachable) or "skip"."""
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
            return "refused"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        print(f"[nfl_weekly] fetch {path}: ok ({len(data):,} bytes, "
              f"{n_new:,} rows)", flush=True)
        return "ok"
    except OSError as ex:
        if os.path.exists(tmp):
            os.remove(tmp)
        tag = "FAILED" if required else "skip (not published yet)"
        print(f"[nfl_weekly] fetch {path}: {tag} ({ex})", flush=True)
        return "failed" if required else "skip"


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


def run_step(step: str, tmo: int, args=(), env=None) -> bool:
    """Run one chain script; True on exit 0. The stderr tail is printed on
    failure, the last stdout line on success."""
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", step, *args],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           cwd=str(PROJECT), timeout=tmo, env=env)
        ok = r.returncode == 0
        src = r.stdout if ok else (r.stderr or r.stdout)
        tail = (src or "").strip().splitlines()
        print(f"[nfl_weekly] {step}: {'ok' if ok else 'FAILED'} "
              f"({tail[-1][:160] if tail else ''})", flush=True)
        if not ok:
            for ln in tail[-8:-1]:
                print(f"[nfl_weekly]   | {ln[:200]}", flush=True)
        return ok
    except subprocess.TimeoutExpired:
        print(f"[nfl_weekly] {step}: FAILED (timeout {tmo}s, killed)", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[nfl_weekly] {step}: FAILED ({ex})", flush=True)
    return False


def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def ledger_swap_errors(live: dict, staged: dict, payload: dict) -> list[str]:
    """Why a staged ledger must not replace the live one ([] = safe).

    A rebuild may add entries and refresh unplayed ones. It may never drop an
    entry, and never change the entry of a game that was played or had kicked
    off when the serve ran (served_at): those are pre-game publications.
    """
    import nfl_ph_freeze as F
    errs = []
    lost = [k for k in live if k not in staged]
    if lost:
        errs.append(f"{len(lost)} ledger entries would vanish (e.g. {lost[:3]})")
    now = payload.get("served_at")
    changed = []
    for s in payload.get("schedule") or []:
        k = F.ledger_key(s)
        if k not in live or k not in staged:
            continue
        played = s.get("hs") is not None and s.get("as") is not None
        if (played or F.has_started(s, now)) and staged[k] != live[k]:
            changed.append(k)
    if changed:
        errs.append(f"{len(changed)} played/kicked-off ledger entries changed "
                    f"(e.g. {changed[:3]})")
    return errs


def played_number_errors(live_payload: dict | None, staged_payload: dict) -> list[str]:
    """Rule 3 at the swap: a game the LIVE payload shows as played keeps the
    exact ph/pmc it shows. Whatever the ledger says, a rebuild may never change
    the published pre-game number of a played game."""
    import nfl_ph_freeze as F
    if not live_payload:
        return []
    staged = {F.ledger_key(s): s for s in staged_payload.get("schedule") or []}
    moved = []
    for s in live_payload.get("schedule") or []:
        if s.get("hs") is None or s.get("as") is None:
            continue
        n = staged.get(F.ledger_key(s))
        if n is None or n.get("ph") != s.get("ph") or (
                s.get("pmc") is not None and n.get("pmc") != s.get("pmc")):
            moved.append(F.ledger_key(s))
    return ([f"{len(moved)} played games would change their published ph/pmc "
             f"(e.g. {moved[:3]})"] if moved else [])


def build_payload(staging: str, live: str, ledger_live: str = LEDGER_LIVE,
                  ledger_staging: str = LEDGER_STAGING) -> list[str]:
    """Run the payload chain into `staging` and `ledger_staging`; swap BOTH
    onto the live files only if every step succeeded, the payload validated
    and no pre-game entry moved. Returns the failures ([] = published)."""
    env = dict(os.environ, NFL_PAYLOAD=staging, NFL_LEDGER=ledger_staging,
               PYTHONHASHSEED="0")
    for p in (staging, ledger_staging):
        if os.path.exists(p):
            os.remove(p)                        # never build on a stale stage
    if os.path.exists(ledger_live):
        shutil.copyfile(ledger_live, ledger_staging)
    untouched = f"{live} and {ledger_live} untouched"
    for step, tmo in PAYLOAD_CHAIN:
        if not run_step(step, tmo, env=env):
            return [f"{step} failed — chain stopped, {untouched}"]
    step, tmo, args = FINAL_STEP
    if not run_step(step, tmo, [*args, "--payload", staging], env=env):
        return [f"staged payload failed validation — {untouched} "
                f"(inspect {staging})"]
    staged_pay = _load_json(staging) or {}
    errs = played_number_errors(_load_json(live), staged_pay)
    live_led = _load_json(ledger_live)
    if live_led is not None:
        staged_led = _load_json(ledger_staging)
        if staged_led is None:
            return [f"staged ledger {ledger_staging} missing or unreadable — {untouched}"]
        errs += ledger_swap_errors(live_led, staged_led, staged_pay)
    if errs:
        return [f"swap check: {e} — {untouched}" for e in errs]
    os.replace(staging, live)
    if os.path.exists(ledger_staging):
        os.replace(ledger_staging, ledger_live)
    print(f"[nfl_weekly] validated payload published -> {live} (+ {ledger_live})",
          flush=True)
    return []


def publish_hint() -> str:
    """The one commit a successful local run should become."""
    return ("git add " + " ".join(PUBLISH) +
            ' && git commit -m "NFL serve" -- ' + " ".join(PUBLISH))


def main() -> int:
    os.chdir(PROJECT)
    failures = []
    for path, url, required in FETCHES:
        st = fetch(path, url, required)
        if required and st in ("failed", "refused"):
            failures.append(f"required fetch {path}: {st}")

    have_tables = all(os.path.exists(p) for p, _ in REDUCERS)
    if not have_tables:
        print("[nfl_weekly] play tables absent (CI) — fetch-only run, chain skipped",
              flush=True)
        return _finish(failures)

    # in-season only: re-pull current-season pbp (done-sets freeze a season
    # after first pull, so stale 2026 rows must be dropped first)
    season_started = any(
        r["game_id"][:4] == CUR and r.get("home_score", "") != ""
        for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")))
    if season_started:
        for path, script in REDUCERS:
            n = strip_current_season(path)
            print(f"[nfl_weekly] {path}: dropped {n:,} season-{CUR} rows", flush=True)
            if not run_step(script, 3600):
                # the table just lost its current-season rows: a model built on
                # it would silently miss this season's plays
                failures.append(f"{script} failed after {path} was stripped — "
                                f"payload not rebuilt")
                return _finish(failures)
    else:
        print(f"[nfl_weekly] no {CUR} finals yet — pbp re-pull skipped", flush=True)

    import nfl_payload
    built = build_payload(nfl_payload.STAGING, nfl_payload.LIVE)
    failures += built
    if not built:
        print(f"[nfl_weekly] publish the payload and its ledger in ONE commit: "
              f"{publish_hint()}", flush=True)
    return _finish(failures)


def _finish(failures: list[str]) -> int:
    for f in failures:
        print(f"[nfl_weekly] FAILURE: {f}", flush=True)
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title=NFL weekly::{f}", flush=True)
    print(f"[nfl_weekly] done ({'FAILED' if failures else 'ok'})", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
