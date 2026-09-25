"""Player-value program (pv), NFL FULL-SPAN extension: shared helpers. DISPLAY ONLY.

The four NFL component builds (pv_nfl_passing_build, pv_nfl_receiving_rushing_*,
pv_nfl_pass_rush_and_front*, pv_nfl_coverage) were fitted and validated on DEV
seasons only (<= 2015). The site needs the players on today's rosters, so the
pv_nfl_full_* wrappers run the SAME code through the latest pulled season:

  * every hyper-parameter is FROZEN at its DEV value, read from the component's
    own DEV output (hyper / params / priors json) or from the component's module
    constants -- nothing is re-fitted, re-selected or re-tuned on a season >= 2016;
  * the component code is used verbatim: imported (functions), executed from its
    unmodified source (scripts without an importable entry point), or, for the
    one class that lives in a script body, extracted with `ast` from the file and
    exec'd -- the sha256 of every source used is recorded with the outputs;
  * the wrappers compute player VALUES only (walk-forward rating states). They
    print, store and compute no statistic that compares a value with a later
    outcome for any season >= 2016 (the NFL TEST era), and no game-model metric.

The values feed data/nfl_site_pv.json (phase0/nfl_site_player_value.py), a
display layer: the game model does not use them. At game level the composed
player values did not clear the DEV screen (data/pv_nfl_screen.json), so they
are descriptions of players, not forecast inputs.

Every intermediate is written to data/pv_nfl_full_*.parquet (gitignored).
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PHASE0 = os.path.join(ROOT, "phase0")
EVENTS = os.path.join(DATA, "pv_nfl_events.parquet")
TEST_ERA = 2016                 # NFL locked split: seasons >= 2016 are TEST
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
PROVENANCE = os.path.join(DATA, "pv_nfl_full_provenance.json")
RECORD = True                   # tests switch this off: no provenance written as a side effect
T0 = time.time()

if PHASE0 not in sys.path:
    sys.path.insert(0, PHASE0)


def log(*a):
    """Progress lines only: never a statistic about a season >= TEST_ERA."""
    print(f"[pv_nfl_full {time.time() - T0:6.0f}s]", *a, flush=True)


def fr(t):
    return FR.get(t, t)


# ------------------------------------------------------------------ span
def events_span(path: str = EVENTS) -> dict:
    """Latest pulled season and its last week with plays (a data fact, no outcome)."""
    import numpy as np
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["season", "week", "season_type"]).to_pandas()
    s = t.season.to_numpy(dtype=float)
    last = int(np.nanmax(s))
    cur = t[s == last]
    reg = cur[cur.season_type == "REG"]
    return {"season": last, "week": int(cur.week.max()),
            "reg_week": int(reg.week.max()) if len(reg) else None}


# ------------------------------------------------------------------ frozen sources
def src_path(name: str) -> str:
    return os.path.join(PHASE0, name if name.endswith(".py") else name + ".py")


def sha256_file(path: str) -> str:
    """sha256 of the file with CRLF line endings normalised to LF.

    Provenance must identify the COMMITTED code: with core.autocrlf=true a Windows
    working copy holds CRLF while the repository (and any Linux / cloud checkout)
    holds LF, so a raw-byte hash would differ across checkouts of the same commit.
    Hashing the LF-normalised bytes gives the same value on every checkout, equal to
    sha256 of the committed blob's content."""
    with open(path, "rb") as fh:
        data = fh.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def read_source(name: str) -> tuple[str, str]:
    p = src_path(name)
    with open(p, encoding="utf-8") as fh:
        src = fh.read()
    return src, sha256_file(p)


def ast_extract(src: str, names) -> str:
    """Source text of the named top-level functions / classes, verbatim."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out, found = [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            out.append("".join(lines[start:node.end_lineno]))
            found.add(node.name)
    missing = set(names) - found
    assert not missing, f"not found in source: {sorted(missing)}"
    return "\n\n".join(out)


def exec_script(name: str, argv, fake_root: str | None = None) -> dict:
    """Run an unmodified component script as __main__ with the given argv.

    fake_root: execute it as if it lived in <fake_root>/phase0/, so every path it
    derives from __file__ (ROOT/DATA) points into a sandbox instead of the
    project -- the script then reads the sandbox's linked inputs and writes its
    fixed-name outputs there, never over the DEV outputs in data/."""
    src, sha = read_source(name)
    fname = src_path(name) if fake_root is None else os.path.join(fake_root, "phase0", os.path.basename(src_path(name)))
    g = {"__name__": "__main__", "__file__": fname, "__builtins__": __builtins__}
    old_argv = sys.argv
    sys.argv = [fname] + [str(a) for a in argv]
    try:
        exec(compile(src, fname, "exec"), g)
    except SystemExit as ex:
        if ex.code not in (None, 0):
            raise
    finally:
        sys.argv = old_argv
    record_provenance(name, sha)
    return g


@contextlib.contextmanager
def sandbox(inputs):
    """A throw-away project root: <tmp>/phase0 and <tmp>/data with the given
    data/ files hard-linked (copied if linking is impossible). Yields the root."""
    root = tempfile.mkdtemp(prefix="pv_nfl_full_")
    try:
        os.makedirs(os.path.join(root, "phase0"))
        os.makedirs(os.path.join(root, "data"))
        for rel in inputs:
            src = os.path.join(DATA, rel)
            if not os.path.exists(src):
                continue
            dst = os.path.join(root, "data", rel)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def record_provenance(name: str, sha: str, extra: dict | None = None) -> None:
    """sha256 of every component source a wrapper ran (and run notes), for the snapshot."""
    if not RECORD:
        return
    try:
        with open(PROVENANCE, encoding="utf-8") as fh:
            prov = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        prov = {}
    ent = prov.setdefault("sources", {})
    ent[os.path.basename(src_path(name))] = sha
    if extra:
        prov.setdefault("runs", {}).update(extra)
    tmp = PROVENANCE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
    os.replace(tmp, PROVENANCE)


def import_component(name: str):
    """Import a component module (its entry points are __main__-guarded) and
    record the sha of the source it came from."""
    import importlib
    mod = importlib.import_module(name)
    record_provenance(name, sha256_file(mod.__file__))
    return mod


# ------------------------------------------------------------------ io
def write_parquet(df, name: str) -> str:
    path = os.path.join(DATA, name)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_json(name: str):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ identity
def positions(season: int | None = None) -> dict:
    """gsis -> listed position: nfl_players.csv first (as the DEV builds read it),
    then the weekly roster for ids it does not know yet (this year's rookies).
    Identity metadata only: it sets EB prior buckets / position groups."""
    import csv
    pos = {}
    with open(os.path.join(DATA, "nfl_players.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("gsis_id"):
                pos[r["gsis_id"]] = r.get("position") or ""
    seasons = [season] if season else []
    for s in seasons:
        p = os.path.join(DATA, f"roster_{s}.csv")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                g = r.get("gsis_id")
                if g and not pos.get(g):
                    pos[g] = r.get("depth_chart_position") or r.get("position") or ""
    return pos
