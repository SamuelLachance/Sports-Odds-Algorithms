"""Attach NFL final scores to the published payload — every refresh cycle, CI-safe.

Why this exists: the NFL model chain (nfl_weekly.py) needs the gitignored play
tables, so it only runs on a local machine, by hand. Before this step, nothing
lighter moved finals into site/data/nfl.json, and the whole NFL side froze
between manual runs: played games showed as upcoming picks, the track record
graded nothing, standings stayed a week behind and EDGE bets never settled.

What it does (standard library only; runs from refresh.py every cycle):
  1. fetch data/nfl_games.csv through nfl_weekly.fetch, whose truncation guard
     refuses a short response and keeps the good spine;
  2. for every served regular-season row whose game has a final: attach hs/as
     (and correct a changed final), copy the forecast's receipt from the
     pre-game ledger (tier, frozen_at, replay) ONLY when the ledger entry
     holds the very number the payload published (receipt_matches) - a
     receipt for another number is refused and the row is demoted to EARLY
     with `receipt_mismatch` - and the actual starting QBs;
  3. recompute standings, records and streaks for the current season (the same
     code nfl_site_db uses, so the two can never disagree), plus last season's
     final record per team;
  4. validate the result BEFORE writing: any validation error the attach
     introduced (a ledger mismatch, a receipt stamped after kickoff) refuses
     the whole write and exits 1, so nothing half-true is ever published;
  5. stamp results_through / results_updated and write the payload atomically,
     only when something changed.

What it never does: touch ph, pmc, ct or cx. A played game's number is the one
published before kickoff; the ledger (data/nfl_ph_ledger.json, or the staged
copy named by NFL_LEDGER inside the weekly chain) is read-only here. No model recompute, no Elo walk, no re-simulation: numbers the served
model did not produce are not published. Market-blind.

  python phase0/nfl_results_attach.py                # fetch + attach (refresh.py)
  python phase0/nfl_results_attach.py --no-fetch --payload data/nfl_payload_staging.json
  python phase0/nfl_results_attach.py --check        # validate only, no write

Exit status: 0 ok; 1 the payload failed validation (or the fetch failed);
2 there is no usable payload to attach to.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "phase0"))

import nfl_payload as NP  # noqa: E402
import nfl_ph_freeze as F  # noqa: E402

GAMES_CSV = "data/nfl_games.csv"
LEDGER_ENV = "NFL_LEDGER"
LEDGER_LIVE = "data/nfl_ph_ledger.json"
LEDGER = LEDGER_LIVE        # kept for importers; run() reads ledger_path()
N_REG = 272


def ledger_path() -> str:
    """The pre-game ledger this process reads: the weekly chain's staged copy
    (NFL_LEDGER) while it builds, else the live file."""
    return os.environ.get(LEDGER_ENV) or LEDGER_LIVE


def _key(w, home, away) -> tuple:
    return (int(w), home, away)


def season_rows(rows: list[dict], season: int) -> list[dict]:
    return [r for r in rows if str(r.get("season")) == str(season)
            and r.get("game_type") == "REG"]


def _index(rows: list[dict]):
    by_id, by_key = {}, {}
    for r in rows:
        h = NP.FR.get(r["home_team"], r["home_team"])
        a = NP.FR.get(r["away_team"], r["away_team"])
        by_id[r["game_id"]] = r
        by_key[_key(r["week"], h, a)] = r
    return by_id, by_key


def _final(r: dict):
    if r.get("home_score", "") == "" or r.get("away_score", "") == "":
        return None
    return int(r["home_score"]), int(r["away_score"])


def receipt_matches(s: dict, ent: dict | None) -> bool:
    """The ledger entry describes THIS row's published number (ph and pmc)."""
    if ent is None:
        return False
    if ent.get("ph") != s.get("ph"):
        return False
    return ent.get("pmc") is None or ent.get("pmc") == s.get("pmc")


def _receipt(s: dict, ent: dict | None, out: dict, warn) -> None:
    """Copy the ledger's receipt onto a played row, or refuse it.

    A ledger entry whose ph/pmc differ from the payload's describes ANOTHER
    publication (a serve whose payload never went live, or one a CI rebase
    reverted). Its tier and frozen_at would dress this number in a receipt it
    never had and grade it in a tier it was never published in, so the row is
    demoted to EARLY with no freeze time and flagged `receipt_mismatch`.
    """
    if ent is not None and not receipt_matches(s, ent):
        out["mismatch"] += 1
        s["tier"] = F.LEGACY_TIER
        s.pop("frozen_at", None)
        s["receipt_mismatch"] = True
        warn(f"WARNING results-attach: {F.ledger_key(s)} payload ph/pmc "
             f"{s.get('ph')}/{s.get('pmc')} != ledger {ent.get('ph')}/{ent.get('pmc')}: "
             f"receipt refused, row demoted to EARLY (payload number kept)")
        return
    r = F.restore_row(s, ent)
    if r.get("late"):
        warn(f"WARNING results-attach: {F.ledger_key(s)} ledger entry stamped "
             f"{r['late']}, after kickoff {s.get('start_utc')}: graded as a replay")


def attach(payload: dict, rows: list[dict], ledger: dict, warn=print) -> dict:
    """Attach finals to schedule rows in place. Returns counts.

    ph/pmc/ct/cx are never written. A newly played row whose payload number
    differs from its ledger entry is reported (`mismatch`), not repaired: the
    payload is the publication, and a disagreement means something upstream is
    wrong. Its receipt is refused (see _receipt).
    """
    season = int(payload.get("season") or 0)
    by_id, by_key = _index(season_rows(rows, season))
    out = {"new": 0, "corrected": 0, "mismatch": 0, "unmatched": 0}
    for s in payload.get("schedule") or []:
        r = by_id.get(s.get("id")) or by_key.get(_key(s["w"], s["home"], s["away"]))
        if r is None:
            out["unmatched"] += 1
            continue
        fin = _final(r)
        if fin is None:
            continue
        hs, as_ = fin
        if s.get("hs") is None or s.get("as") is None:
            s["hs"], s["as"] = hs, as_
            _receipt(s, ledger.get(F.ledger_key(s)), out, warn)
            out["new"] += 1
        elif (s["hs"], s["as"]) != (hs, as_):
            warn(f"results-attach: corrected final {F.ledger_key(s)} "
                 f"{s['as']}-{s['hs']} -> {as_}-{hs}")
            s["hs"], s["as"] = hs, as_
            out["corrected"] += 1
        if s.get("tier") is None:
            # a played row from a payload built before the tier stamp: its
            # receipt is still in the ledger (metadata only, ph untouched)
            _receipt(s, ledger.get(F.ledger_key(s)), out, warn)
        if "qb_start" not in s and (r.get("home_qb_id") or r.get("away_qb_id")):
            s["qb_start"] = {"h": {"id": r.get("home_qb_id") or None,
                                   "name": r.get("home_qb_name") or None},
                             "a": {"id": r.get("away_qb_id") or None,
                                   "name": r.get("away_qb_name") or None}}
        if "id" not in s:
            s["id"] = r["game_id"]
    return out


def results_through(payload: dict) -> dict | None:
    fin = [s for s in payload.get("schedule") or []
           if s.get("hs") is not None and s.get("as") is not None]
    if not fin:
        return None
    return {"w": max(s["w"] for s in fin), "d": max(s["d"] for s in fin), "n": len(fin)}


def validate(payload: dict, rows: list[dict], ledger: dict, full: bool = False) -> list[str]:
    """Problems that must stop a payload from being published ([] = fine).

    Always: the schedule is the full regular season, every final in the spine
    is attached with the right score, every played row still carries its
    pre-game ledger value, and every graded receipt predates kickoff (a
    non-replay played row has a frozen_at, and frozen_at <= start_utc).
    `full` (the weekly chain's swap gate) adds the checks only a complete
    rebuild can pass.
    """
    errs = []
    sched = payload.get("schedule") or []
    season = int(payload.get("season") or 0)
    reg = season_rows(rows, season)
    if not sched:
        return ["no schedule"]
    if len(sched) != len(reg) or len(sched) != N_REG:
        errs.append(f"schedule has {len(sched)} rows, spine has {len(reg)} (expected {N_REG})")
    by_id, by_key = _index(reg)
    for s in sched:
        r = by_id.get(s.get("id")) or by_key.get(_key(s["w"], s["home"], s["away"]))
        fin = _final(r) if r else None
        if fin and (s.get("hs"), s.get("as")) != fin:
            errs.append(f"final not attached: {F.ledger_key(s)} spine {fin} "
                        f"payload {(s.get('hs'), s.get('as'))}")
        if s.get("hs") is not None and s.get("as") is not None:
            ent = ledger.get(F.ledger_key(s))
            if ent is None:
                if not s.get("replay"):
                    errs.append(f"played {F.ledger_key(s)} has no ledger entry and no replay flag")
            elif ent.get("ph") != s.get("ph") or (
                    ent.get("pmc") is not None and ent.get("pmc") != s.get("pmc")):
                errs.append(f"played {F.ledger_key(s)} ph/pmc {s.get('ph')}/{s.get('pmc')} "
                            f"!= ledger {ent.get('ph')}/{ent.get('pmc')}")
            if s.get("tier") not in ("PROJECTED", "EARLY"):
                errs.append(f"played {F.ledger_key(s)} carries no tier")
            if not s.get("replay") and not s.get("receipt_mismatch"):
                fz, st = F.utc(s.get("frozen_at")), F.utc(s.get("start_utc"))
                if fz is None:
                    errs.append(f"played {F.ledger_key(s)} is graded {s.get('tier')} "
                                f"with no pre-game freeze time")
                elif st is not None and fz > st:
                    errs.append(f"played {F.ledger_key(s)} frozen_at {s.get('frozen_at')} "
                                f"is after kickoff {s.get('start_utc')}")
    if full:
        if payload.get("status") != "season":
            errs.append(f"status is {payload.get('status')!r}, not 'season'")
        if len(payload.get("teams") or {}) != 32:
            errs.append(f"{len(payload.get('teams') or {})} teams")
        np_ = len(payload.get("players") or {})
        if np_ < 1000:
            errs.append(f"only {np_} players")
        for t, tm in (payload.get("teams") or {}).items():
            if not tm.get("roster") or not tm.get("lineup") or not tm.get("proj"):
                errs.append(f"team {t} lacks roster/lineup/proj")
                break
        if not payload.get("served_at"):
            errs.append("no served_at: the serve did not run")
        if any(s.get("tier") not in ("PROJECTED", "EARLY") for s in sched):
            errs.append("schedule rows without a tier")
    return errs


def _canon(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if k != "results_updated"},
                      sort_keys=True)


def run(payload_path: str, fetch: bool = True, check_only: bool = False,
        full: bool = False) -> int:
    os.chdir(PROJECT)
    fetch_ok = True
    if fetch:
        import nfl_weekly
        path, url, required = nfl_weekly.FETCHES[0]
        assert path == GAMES_CSV
        fetch_ok = nfl_weekly.fetch(path, url, required) in ("ok", "unchanged")
    try:
        payload = NP.load(payload_path)
    except (FileNotFoundError, json.JSONDecodeError) as ex:
        print(f"[nfl_results_attach] no usable payload at {payload_path}: {ex}", flush=True)
        return 2
    if not payload.get("schedule"):
        print(f"[nfl_results_attach] {payload_path} has no schedule — nothing to "
              f"attach to (not written)", flush=True)
        return 2
    rows = NP.read_games(GAMES_CSV)
    lpath = ledger_path()
    try:
        ledger = json.load(open(lpath, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        ledger = {}
    errs = validate(payload, rows, ledger, full=full)
    new_errs: list[str] = []
    if not check_only:
        # Attach on a copy and validate it BEFORE anything is written: an error
        # the attach introduced (a refused receipt, a post-kickoff freeze time)
        # stops the write instead of being discovered after it.
        before_errs = set(errs)
        work = copy.deepcopy(payload)
        cnt = attach(work, rows, ledger, warn=lambda m: print(m, flush=True))
        season = NP.apply_standings(work, rows)
        work["results_through"] = results_through(work)
        work.setdefault("tracking_since", F.tracking_since(ledger))
        errs = validate(work, rows, ledger, full=full)
        new_errs = [e for e in errs if e not in before_errs]
        changed = _canon(work) != _canon(payload)
        if new_errs:
            print(f"[nfl_results_attach] REFUSED: the attach introduced "
                  f"{len(new_errs)} validation error(s); {payload_path} NOT "
                  f"written (ledger {lpath})", flush=True)
            state = "REFUSED, not written"
        elif changed:
            work["results_updated"] = NP.now_utc_iso()
            NP.dump_atomic(work, payload_path, separators=(",", ":"))
            state = "written"
        else:
            state = "unchanged"
        rt = work["results_through"] or {}
        print(f"[nfl_results_attach] {cnt['new']} new finals, {cnt['corrected']} "
              f"corrected, {cnt['mismatch']} ledger mismatches; standings {season} "
              f"through week {rt.get('w')} ({rt.get('n', 0)} finals); {state}",
              flush=True)
    for e in errs[:20]:
        tag = "NEW ERROR" if e in new_errs else "INVALID"
        print(f"[nfl_results_attach] {tag}: {e}", flush=True)
    if not fetch_ok:
        print("[nfl_results_attach] WARNING: nfl_games.csv fetch failed or was "
              "refused; attached from the file on disk", flush=True)
    return 1 if (errs or not fetch_ok) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--payload", default=None,
                    help="payload to update (default: NFL_PAYLOAD or the live file)")
    ap.add_argument("--no-fetch", action="store_true", help="use data/nfl_games.csv as is")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    ap.add_argument("--full", action="store_true",
                    help="also run the full-rebuild checks (the weekly chain's swap gate)")
    a = ap.parse_args(argv)
    return run(a.payload or NP.payload_path(), fetch=not a.no_fetch,
               check_only=a.check, full=a.full)


if __name__ == "__main__":
    raise SystemExit(main())
