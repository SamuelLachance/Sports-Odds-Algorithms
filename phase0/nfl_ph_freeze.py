"""The NFL pre-game prediction freeze — extracted so it can be TESTED.

This is the mechanism that keeps the NFL track record honest, and it is the one
piece of the September in-season machinery whose failure would be invisible.

The problem it solves: `nfl_season_serve.py` recomputes every game's probability
on every run, including games already played. Those recomputed numbers are
POST-HOC — the rating states have since walked through the game's own result —
so publishing them as "our pre-game pick" would be a lie that flatters us.
`data/nfl_ph_ledger.json` stores the pre-game values; a played game's payload row
is restored from the ledger instead of from the fresh computation.

It was previously an inline loop inside the serve script, verified once by a
synthetic week-1 dry run and covered by no test at all. A silent break between
now and kickoff would surface only as an unearned track record.

Semantics (each pinned by a test in tests/test_nfl_ph_freeze.py):
  - a PLAYED game (both scores present) is restored from its ledger entry;
  - an UNPLAYED game refreshes its ledger entry with the current forecast, so
    the stored value is always the latest PRE-game one;
  - a played game with NO ledger entry cannot be repaired — the pre-game number
    is simply gone — so it keeps the recomputed value, is recorded so the next
    run has something, and WARNS rather than failing silently;
  - `ct` and `pmc` ride along with `ph`: a frozen probability beside a
    recomputed contribution breakdown would be internally inconsistent.
"""
from __future__ import annotations


def ledger_key(row: dict) -> str:
    """Stable identity for an NFL game: week + both teams (no per-game id exists)."""
    return f'{row["w"]}|{row["home"]}|{row["away"]}'


def bootstrap_from_payload(payload: dict) -> dict:
    """Seed a ledger from a payload written before the ledger existed."""
    return {ledger_key(r): {"ph": r["ph"], "ct": r.get("ct"), "pmc": r.get("pmc")}
            for r in payload.get("schedule", []) if r.get("ph") is not None}


def freeze(sched: list[dict], ledger: dict, warn=print) -> tuple[dict, int, int]:
    """Restore played games' pre-game values; refresh unplayed ones.

    Mutates `sched` rows in place (the serve script writes them straight into the
    payload) and returns (ledger, n_frozen, n_orphaned).
    """
    n_frozen = n_orphan = 0
    for s in sched:
        key = ledger_key(s)
        played = s.get("hs") is not None and s.get("as") is not None
        if not played:
            ledger[key] = {"ph": s["ph"], "ct": s.get("ct"), "pmc": s.get("pmc")}
            continue
        ent = ledger.get(key)
        if ent is None:
            # Unrepairable: the pre-game number was never recorded. Keep the
            # recomputed one, record it so the next run is not orphaned too, and
            # say so loudly — a silent post-hoc value is the failure mode.
            n_orphan += 1
            warn(f"WARNING ph-freeze: no ledger entry for played game {key}; "
                 f"keeping recomputed (post-hoc) ph/pmc")
            ledger[key] = {"ph": s["ph"], "ct": s.get("ct"), "pmc": s.get("pmc")}
            continue
        s["ph"] = ent["ph"]
        if ent.get("ct") is not None:
            s["ct"] = ent["ct"]
        if ent.get("pmc") is not None:
            s["pmc"] = ent["pmc"]
        n_frozen += 1
    return ledger, n_frozen, n_orphan
