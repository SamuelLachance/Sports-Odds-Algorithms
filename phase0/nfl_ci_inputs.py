"""CI input layer for the NFL weekly model chain (.github/workflows/nfl-weekly.yml).

The chain (phase0/nfl_weekly.py) reads inputs that are not in git. A traced
local run (2026-09-24, every file each step opened) found three groups:

  TABLES  the five reduced play tables (0.6-98 MB, gitignored). Built by the
          existing pull scripts, one nflverse pbp parquet per season.
  STATIC  frozen nflverse downloads nobody re-fetches: players, the legacy
          weekly player stats (offense + defense, <= 2024), last season's
          stats_player_week file, draft picks, snap counts 2013..last season
          (untracked, ~110 MB).
  FROZEN  two research outputs with no upstream source:
          data/nfl_player_board_2025.csv (nfl_final_board.py) and
          data/nfl_wowy.json (nfl_wowy_eval.py - re-running it computes TEST
          metrics, so CI must never regenerate it). They have to be in git.

The workflow restores TABLES + STATIC from actions/cache; on a miss this
module rebuilds them: STATIC from their nflverse URLs, TABLES by running the
pull scripts for every season (they download the parquets themselves).
FROZEN can only come from the repository.

Parity caveat, measured 2026-09-24: nflverse re-releases historical files in
place. Against the local copies, players / stats_player_week_2025 / draft
picks / snap_2020 / snap_2025 and the 2020 pbp (3 extra plays, 181 team-games
with a different EPA sum) have all changed since July. A cache-miss bootstrap
therefore reproduces the local model closely but not bit-for-bit; the serve's
own TEST-reproduction tripwire (nfl_season_serve.py) is what refuses a build
that drifted too far, and the job then goes red without publishing.

Measured effect of that drift (same code, same day): serving from a fresh
upstream bootstrap instead of the local July inputs moved unplayed forecasts
by 0.5 pp on average, 3.0 pp at most (35 of 240 unchanged); played games and
their ledger entries were identical, as the freeze requires. For exact parity
with the local model, seed a cold CI cache ONCE from the local inputs:

  python phase0/nfl_ci_inputs.py pack        # -> <tmp>/nfl_ci_inputs.tar.gz
  gh release create nfl-ci-inputs <that file> --title "NFL CI inputs seed"

The workflow restores that release asset on a cache miss (before bootstrapping
whatever is still missing from nflverse); without it CI serves from upstream.
Creating the release publishes the file to whoever can see the repository.
The seed is only read on a cache MISS: after the first unseeded run, delete
the nfl-inputs-* caches (or bump the key's v1) for a new seed to take effect.
Which set built a serve is kept in ORIGIN (cached with the inputs): every CI
run built from a non-seed set logs a ::warning::, and the serve's commit
message names the set ("[inputs: nflverse]").

  python phase0/nfl_ci_inputs.py check       # exit 0 iff every input is sane
  python phase0/nfl_ci_inputs.py check --cached   # the cached paths only
  python phase0/nfl_ci_inputs.py bootstrap   # rebuild what is missing/invalid
  python phase0/nfl_ci_inputs.py paths       # cacheable paths, one per line
  python phase0/nfl_ci_inputs.py digest      # content hash of the cached set
  python phase0/nfl_ci_inputs.py origin      # seed | nflverse | seed+nflverse
  python phase0/nfl_ci_inputs.py pack [out]  # seed tarball of the cached inputs
  python phase0/nfl_ci_inputs.py unpack <tarball>

Standard library only (like nfl_weekly.py). Market-blind: nflverse data only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "phase0"))

import nfl_weekly as W  # noqa: E402  (REDUCERS, CUR, fetch, run_step)

NFLV = "https://github.com/nflverse/nflverse-data/releases/download"
CUR = int(W.CUR)
LAST = CUR - 1                   # the last complete season
SNAP_FIRST = 2013                # nfl_player_rating_system reads snap_20??.csv >= 2013
csv.field_size_limit(1 << 24)    # rapm_plays carries long player-id lists


@dataclass(frozen=True)
class Table:
    """A reduced play table and what a complete one looks like."""
    path: str
    script: str
    header: tuple
    first: int                   # first season the pull script reduces
    min_rows: int                # per complete season (local minimum x ~0.9)
    timeout: int = 1800          # a full-history pull


@dataclass(frozen=True)
class Static:
    path: str
    url: str
    cols: tuple                  # columns the chain reads (subset check)
    min_rows: int
    seasons: tuple = field(default=())   # season values that must appear


# Local tables 2026-09-24: plays/duel 31,960..35,766 per season (1999 has 31
# teams), turnovers 518..570 team-games, ctx2/rapm 33,290..35,766 (2016+).
_PLAYS = ("game_id", "game_date", "posteam", "defteam", "epa")
TABLES = (
    Table("data/nfl_plays.csv", "phase0/nfl_pbp_pull.py", _PLAYS, 1999, 28000),
    Table("data/nfl_turnovers.csv", "phase0/nfl_pbp_pull2.py",
          ("game_id", "game_date", "posteam", "int_thrown", "fumbles",
           "fumbles_lost"), 1999, 450),
    Table("data/nfl_duel_plays.csv", "phase0/nfl_pbp_pull3.py",
          _PLAYS + ("passer_player_id", "rusher_player_id", "receiver_player_id"),
          1999, 28000),
    Table("data/nfl_play_ctx2.csv", "phase0/nfl_play_ctx2_pull.py",
          ("game_id", "play_id", "wp", "qtr", "down", "ydstogo",
           "score_differential", "half_seconds_remaining", "yardline_100"),
          2016, 30000),
    Table("data/nfl_rapm_plays.csv", "phase0/nfl_participation_pull.py",
          ("game_id", "play_id", "offense_players", "defense_players", "epa"),
          2016, 30000),
)

# Local files 2026-09-24: players 25,035 rows, legacy offense stats 134,470,
# defense 239,955, 2025 week stats 19,421, draft picks 12,927, snaps
# 23,799..26,612 per season. Floors sit ~15-20% below.
STATIC = (
    Static("data/nfl_players.csv", f"{NFLV}/players/players.csv",
           ("gsis_id", "display_name", "position"), 20000),
    Static("data/nfl_player_stats.csv", f"{NFLV}/player_stats/player_stats.csv",
           ("player_id", "recent_team", "season", "week", "sacks"), 100000),
    Static("data/nfl_player_stats_def.csv",
           f"{NFLV}/player_stats/player_stats_def.csv",
           ("player_id", "team", "season", "week"), 200000),
    Static(f"data/nfl_player_stats_{LAST}.csv",
           f"{NFLV}/stats_player/stats_player_week_{LAST}.csv",
           ("player_id", "team", "week", "sacks_suffered"), 15000,
           (str(LAST),)),
    Static("data/nfl_draft_picks.csv", f"{NFLV}/draft_picks/draft_picks.csv",
           ("season", "gsis_id", "round", "pick"), 10000),
) + tuple(
    Static(f"data/snap_{y}.csv", f"{NFLV}/snap_counts/snap_counts_{y}.csv",
           ("game_id", "team", "offense_pct", "defense_pct"), 20000, (str(y),))
    for y in range(SNAP_FIRST, CUR))

FROZEN = ("data/nfl_player_board_2025.csv", "data/nfl_wowy.json")
SEED_TAG = "nfl-ci-inputs"               # optional release holding the seed
SEED_ASSET = "nfl_ci_inputs.tar.gz"
# Provenance of the cached input set, cached with it: "seed" (the owner's
# local inputs, restored from the release), "nflverse" (rebuilt from today's
# upstream files), "seed+nflverse" (a seed topped up from upstream) or
# "unknown" (no marker: a local checkout). Deterministic content, so an
# unchanged input set keeps an unchanged cache digest.
ORIGIN = "data/nfl_ci_inputs_origin.json"


def input_paths() -> list[str]:
    """The chain's cacheable inputs (what a seed packs and unpacks)."""
    return [t.path for t in TABLES] + [s.path for s in STATIC]


def paths() -> list[str]:
    """Everything actions/cache persists (never a tracked file: a restore
    would overwrite the checkout): the inputs plus their provenance marker."""
    return input_paths() + [ORIGIN]


def origin(root: Path = PROJECT) -> str:
    try:
        with open(root / ORIGIN, encoding="utf-8") as fh:
            src = json.load(fh).get("source")
        return src if src in ("seed", "nflverse", "seed+nflverse") else "unknown"
    except (OSError, ValueError, AttributeError):
        return "unknown"


def _set_origin(root: Path, source: str) -> None:
    p = root / ORIGIN
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"source": source}) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def digest(root: Path = PROJECT) -> str:
    """Content hash of the cached set (names + bytes, in paths() order); the
    workflow keys the cache on it, so an unchanged set is not saved again."""
    h = hashlib.sha256()
    for rel in paths():
        p = root / rel
        h.update(rel.encode("utf-8") + b"\0")
        if not p.exists():
            h.update(b"<absent>\0")
            continue
        h.update(str(p.stat().st_size).encode("ascii") + b"\0")
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()[:32]


# ------------------------------------------------------------------ checks
def table_errors(t: Table, root: Path = PROJECT) -> list[str]:
    """Why a play table is unusable ([] = complete).

    Every season t.first..LAST present with >= t.min_rows rows and every row
    as wide as the header. The current season is not judged: nfl_weekly strips
    and re-pulls it each run. A pull killed mid-append leaves a short season or
    a ragged last row, and the pull's season-granular done-set would then skip
    that season forever - hence the per-season floor."""
    p = root / t.path
    if not p.exists():
        return [f"{t.path}: missing"]
    n_by = Counter()
    ragged = 0
    try:
        with open(p, encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            header = tuple(next(rd, ()))
            if header != t.header:
                return [f"{t.path}: header {list(header)[:6]} != expected"]
            width = len(header)
            for r in rd:
                if len(r) != width:
                    ragged += 1
                    continue
                n_by[r[0][:4]] += 1
    except (OSError, UnicodeDecodeError, csv.Error) as ex:
        return [f"{t.path}: unreadable ({ex})"]
    errs = []
    if ragged:
        errs.append(f"{t.path}: {ragged} ragged rows (interrupted write)")
    short = [f"{y}:{n_by.get(str(y), 0)}" for y in range(t.first, LAST + 1)
             if n_by.get(str(y), 0) < t.min_rows]
    if short:
        errs.append(f"{t.path}: seasons missing/short {short[:6]}"
                    f"{' ...' if len(short) > 6 else ''} (floor {t.min_rows})")
    extra = sorted(k for k in n_by if not (str(t.first) <= k <= str(CUR)))
    if extra:
        errs.append(f"{t.path}: unexpected seasons {extra[:5]}")
    return errs


def static_errors(s: Static, root: Path = PROJECT) -> list[str]:
    p = root / s.path
    if not p.exists():
        return [f"{s.path}: missing"]
    try:
        with open(p, encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh)
            cols = set(rd.fieldnames or ())
            if not set(s.cols) <= cols:
                return [f"{s.path}: missing columns {sorted(set(s.cols) - cols)}"]
            n, seen = 0, set()
            for r in rd:
                n += 1
                if s.seasons:
                    seen.add(str(r.get("season", "")))
    except (OSError, UnicodeDecodeError, csv.Error) as ex:
        return [f"{s.path}: unreadable ({ex})"]
    errs = []
    if n < s.min_rows:
        errs.append(f"{s.path}: {n:,} rows < {s.min_rows:,}")
    if s.seasons and not set(s.seasons) <= seen:
        errs.append(f"{s.path}: no rows for season(s) {sorted(set(s.seasons) - seen)}")
    return errs


def frozen_errors(root: Path = PROJECT) -> list[str]:
    miss = [f for f in FROZEN if not (root / f).exists()]
    if not miss:
        return []
    return [f"{f}: missing - a frozen research output with no upstream; it must "
            f"be committed (git add {' '.join(miss)})" for f in miss]


def check(root: Path = PROJECT, verbose: bool = True,
          frozen: bool = True) -> dict[str, list[str]]:
    """{path: errors} for every unusable input ({} = the chain can run).
    frozen=False judges only the cached paths (the cache-save gate: a missing
    committed file must not stop good tables from being cached)."""
    bad = {}
    for t in TABLES:
        e = table_errors(t, root)
        if e:
            bad[t.path] = e
    for s in STATIC:
        e = static_errors(s, root)
        if e:
            bad[s.path] = e
    for e in (frozen_errors(root) if frozen else []):
        bad[e.split(":")[0]] = [e]
    if verbose:
        for errs in bad.values():
            for e in errs:
                print(f"[nfl_ci_inputs] {e}", flush=True)
        n = len(TABLES) + len(STATIC) + (len(FROZEN) if frozen else 0)
        print(f"[nfl_ci_inputs] check: {len(bad)} unusable of {n} inputs", flush=True)
    return bad


# --------------------------------------------------------------- bootstrap
def bootstrap(root: Path = PROJECT, fetch=None, run=None) -> int:
    """Rebuild every missing or invalid TABLE/STATIC input, then re-check.

    An invalid table is deleted and rebuilt from 1999 (resuming onto a broken
    file would keep the broken season: the pull's done-set skips it). Returns
    0 when everything checks out afterwards, 1 otherwise."""
    fetch = fetch or W.fetch
    run = run or W.run_step
    os.chdir(root)
    bad = check(root, verbose=True)
    if set(bad) & set(input_paths()):
        # whatever is rebuilt below comes from today's upstream files
        was = origin(root)
        _set_origin(root, "seed+nflverse" if was.startswith("seed") else "nflverse")
    for s in STATIC:
        if s.path not in bad:
            continue
        if (root / s.path).exists():
            os.remove(root / s.path)       # fetch's shrink guard compares to it
        fetch(s.path, s.url, True)
    for t in TABLES:
        if t.path not in bad:
            continue
        if (root / t.path).exists():
            os.remove(root / t.path)
            print(f"[nfl_ci_inputs] {t.path}: invalid, rebuilding from {t.first}",
                  flush=True)
        run(t.script, t.timeout)
    warn_unseeded(root)
    if not bad:
        print("[nfl_ci_inputs] bootstrap: nothing to do (cache hit)", flush=True)
        return 0
    left = check(root, verbose=True)
    if left and os.environ.get("GITHUB_ACTIONS"):
        for errs in left.values():
            print(f"::error title=NFL inputs::{errs[0]}", flush=True)
    return 1 if left else 0


def warn_unseeded(root: Path = PROJECT) -> None:
    """Say, on every CI run, when this run's serve is built from inputs other
    than the owner's seed: unplayed forecasts then differ from a local serve
    by up to ~3 pp (measured 2026-09-24), and whichever host served last
    decides which number the site shows."""
    src = origin(root)
    print(f"[nfl_ci_inputs] input set: {src}", flush=True)
    if src != "seed" and os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning title=NFL inputs::input set '{src}', not the owner's "
              f"seed: this serve can differ from a local serve by up to ~3 pp "
              f"on unplayed games. For parity, publish the {SEED_TAG} release "
              f"(python phase0/nfl_ci_inputs.py pack) and delete the "
              f"nfl-inputs-* caches.", flush=True)


# ------------------------------------------------- optional parity seed
def pack(out: str, root: Path = PROJECT) -> int:
    """Tar the local TABLES + STATIC (exactly the cached paths) for a one-time
    CI seed. Refuses when any of them is unusable: a seed must be complete."""
    if check(root, verbose=True, frozen=False):
        print("[nfl_ci_inputs] pack refused: fix the inputs first", flush=True)
        return 1
    tmp = out + ".tmp"
    with tarfile.open(tmp, "w:gz") as tf:
        for p in input_paths():
            tf.add(str(root / p), arcname=p)
    os.replace(tmp, out)
    print(f"[nfl_ci_inputs] packed {len(input_paths())} inputs -> {out} "
          f"({os.path.getsize(out) / 1e6:.0f} MB)", flush=True)
    return 0


def unpack(tarball: str, root: Path = PROJECT) -> int:
    """Extract a seed into the checkout: only members named in input_paths()
    (never a tracked file, never a path outside data/), regular files only.
    Marks the set as seeded; bootstrap marks it "seed+nflverse" if it still
    has to rebuild part of it from upstream."""
    allowed = set(input_paths())
    n = 0
    with tarfile.open(tarball, "r:*") as tf:
        for m in tf.getmembers():
            if m.name in allowed and m.isfile():
                tf.extract(m, str(root), filter="data")
                n += 1
    if n:
        _set_origin(root, "seed")
    print(f"[nfl_ci_inputs] seed: restored {n} of {len(allowed)} inputs from "
          f"{tarball}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "paths":
        print("\n".join(paths()))
        return 0
    if cmd == "check":
        return 1 if check(frozen="--cached" not in argv) else 0
    if cmd == "bootstrap":
        return bootstrap()
    if cmd == "digest":
        print(digest())
        return 0
    if cmd == "origin":
        print(origin())
        return 0
    if cmd == "pack":
        return pack(argv[2] if len(argv) > 2 else
                    os.path.join(tempfile.gettempdir(), SEED_ASSET))
    if cmd == "unpack" and len(argv) > 2:
        return unpack(argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
