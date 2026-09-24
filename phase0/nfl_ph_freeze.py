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

Semantics (each pinned by a test in tests/test_nfl_ph_freeze.py,
tests/test_nfl_data_payload.py and tests/test_nfl_data_receipts.py):
  - a PLAYED game (both scores present) is restored from its ledger entry;
  - a game that has KICKED OFF (start_utc <= serve time) but has no final yet
    is locked exactly like a played one. nflverse fills finals minutes to
    hours after the whistle, so a local run during a Sunday slate or while
    SNF/MNF is on used to rewrite in-progress games and stamp them PROJECTED
    with a post-kickoff time, and CI then graded that number as the pre-game
    one;
  - an UNPLAYED game that has not kicked off refreshes its ledger entry with
    the current forecast, so the stored value is always the latest PRE-game
    one;
  - a played or started game with NO ledger entry cannot be repaired — the
    pre-game number is simply gone — so it keeps the recomputed value, is
    recorded so the next run has something, is marked `replay`, and WARNS
    rather than failing silently;
  - `ct` and `pmc` ride along with `ph`: a frozen probability beside a
    recomputed contribution breakdown would be internally inconsistent.

Information tier (documents/pick_policy.md). Every refreshed write stamps the
UTC serve time `t` and the tier the number carried at that moment:
PROJECTED when the live model served it within TIER_WINDOW_DAYS x 24 hours of
kickoff (the row's start_utc), EARLY otherwise; a row without start_utc falls
back to the US Eastern date rule. A played game restores both, so it is graded
in the tier it was actually published in. Entries written before the stamp
existed carry no `t`: the week-1/2 entries were last written by the
2026-07-30 pre-season serve (verified: every played value equals that
payload's), more than a month before kickoff, so they are EARLY with
`frozen_at` = that publication.

Receipts (`receipt`): a restored row carries `frozen_at` only when its entry is
a genuine pre-game publication. A `replay` entry (nothing was frozen) or one
stamped AFTER kickoff carries no `frozen_at`, is EARLY and is flagged
`replay`, so it can never be graded as a pick.
"""
from __future__ import annotations

from datetime import datetime, timedelta

TIER_WINDOW_DAYS = 7
# The July 30 pre-season payload that published every pre-stamp ledger value
# (site/data/nfl.json 'generated' in commit 53be37da).
LEGACY_T = "2026-07-30T23:57:40Z"
LEGACY_TIER = "EARLY"
# fields that describe the forecast itself and freeze with it
RIDE_ALONG = ("ct", "pmc", "qb", "q")


def ledger_key(row: dict) -> str:
    """Stable identity for an NFL game: week + both teams."""
    return f'{row["w"]}|{row["home"]}|{row["away"]}'


def utc(ts) -> datetime | None:
    """'2026-09-25T00:15:00Z' -> aware datetime; None when absent or unparseable."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def tier_for(game_day: str | None, served_utc: str | None,
             start_utc: str | None = None) -> str:
    """PROJECTED if served within TIER_WINDOW_DAYS x 24 h of kickoff, else EARLY.

    The policy's line is "inside 7 days of kickoff", so the clock is the row's
    start_utc: week-4 TNF (Thu 20:15 ET) served the previous Thursday at 11:08
    ET is 7 days 9 hours out, and EARLY. A row without start_utc (payloads
    built before it was served) falls back to the US Eastern date rule (game
    day <= serve day + 7).
    """
    if not served_utc:
        return LEGACY_TIER
    st, sv = utc(start_utc), utc(served_utc)
    if st is not None and sv is not None:
        return "PROJECTED" if st - sv <= timedelta(days=TIER_WINDOW_DAYS) else "EARLY"
    if not game_day:
        return LEGACY_TIER
    from nfl_payload import et_date
    served = et_date(served_utc)
    return "PROJECTED" if game_day <= (served + timedelta(days=TIER_WINDOW_DAYS)).isoformat() \
        else "EARLY"


def has_started(s: dict, now_utc: str | None) -> bool:
    """Kickoff (start_utc) is at or before `now_utc`; False when either is unknown."""
    st, now = utc(s.get("start_utc")), utc(now_utc)
    return st is not None and now is not None and st <= now


def entry_tier(ent: dict) -> str:
    return ent.get("tier") or LEGACY_TIER


def entry_time(ent: dict) -> str:
    return ent.get("t") or LEGACY_T


def receipt(s: dict, ent: dict | None) -> dict:
    """The tier / frozen_at / replay a row inherits from its ledger entry.

    frozen_at is present only for a genuine pre-game publication. No entry, a
    `replay` entry, or an entry stamped after the row's kickoff (`late`) is
    EARLY and `replay`, with no freeze time.
    """
    if ent is None or ent.get("replay"):
        return {"tier": LEGACY_TIER, "replay": True}
    t = entry_time(ent)
    ft, st = utc(t), utc(s.get("start_utc"))
    if ft is not None and st is not None and ft > st:
        return {"tier": LEGACY_TIER, "replay": True, "late": t}
    return {"tier": entry_tier(ent), "frozen_at": t}


def apply_receipt(s: dict, ent: dict | None) -> dict:
    """Write `receipt(s, ent)` onto the row (metadata only; never ph/pmc/ct)."""
    r = receipt(s, ent)
    s["tier"] = r["tier"]
    if "frozen_at" in r:
        s["frozen_at"] = r["frozen_at"]
    else:
        s.pop("frozen_at", None)
    if r.get("replay"):
        s["replay"] = True
    return r


def bootstrap_from_payload(payload: dict) -> dict:
    """Seed a ledger from a payload written before the ledger existed."""
    return {ledger_key(r): {"ph": r["ph"], "ct": r.get("ct"), "pmc": r.get("pmc")}
            for r in payload.get("schedule", []) if r.get("ph") is not None}


def _entry(s: dict, now_utc: str | None) -> dict:
    ent = {"ph": s["ph"], "ct": s.get("ct"), "pmc": s.get("pmc")}
    for k in ("qb", "q"):
        if s.get(k) is not None:
            ent[k] = s[k]
    if now_utc:
        ent["t"] = now_utc
        ent["tier"] = tier_for(s.get("d"), now_utc, s.get("start_utc"))
    return ent


def _restore(s: dict, ent: dict, warn=print) -> None:
    """A played or kicked-off row takes its ledger entry: number + receipt."""
    s["ph"] = ent["ph"]
    for k in RIDE_ALONG:
        if ent.get(k) is not None:
            s[k] = ent[k]
        elif k in ("qb", "q"):
            s.pop(k, None)           # never pair a frozen ph with today's QBs
    r = apply_receipt(s, ent)
    if r.get("late"):
        warn(f"WARNING ph-freeze: {ledger_key(s)} ledger entry was stamped "
             f"{r['late']}, after kickoff {s.get('start_utc')}; graded as a "
             f"replay (EARLY), never as a pick")


def freeze(sched: list[dict], ledger: dict, warn=print,
           now_utc: str | None = None) -> tuple[dict, int, int]:
    """Restore played and kicked-off games' pre-game values; refresh the rest.

    Mutates `sched` rows in place (the serve script writes them straight into the
    payload) and returns (ledger, n_frozen, n_orphaned); n_frozen counts the
    played and kicked-off rows restored from the ledger. With `now_utc`, every
    refreshed write is stamped with its serve time and tier, a game whose
    start_utc <= now_utc is never rewritten, and every row leaves carrying
    `tier` (plus `frozen_at` when it restores a genuine pre-game entry).
    """
    n_frozen = n_orphan = 0
    for s in sched:
        key = ledger_key(s)
        played = s.get("hs") is not None and s.get("as") is not None
        started = not played and has_started(s, now_utc)
        if not played and not started:
            ent = _entry(s, now_utc)
            ledger[key] = ent
            if now_utc:
                s["tier"] = ent["tier"]
            continue
        ent = ledger.get(key)
        if ent is None:
            # Unrepairable: the pre-game number was never recorded. Keep the
            # recomputed one, record it so the next run is not orphaned too, and
            # say so loudly — a silent post-hoc value is the failure mode.
            n_orphan += 1
            what = "played game" if played else f"game that kicked off {s.get('start_utc')}"
            warn(f"WARNING ph-freeze: no ledger entry for {what} {key}; "
                 f"keeping recomputed (post-hoc) ph/pmc as a replay")
            ent = _entry(s, now_utc)
            ent["replay"] = True
            ent["tier"] = LEGACY_TIER        # a replay is never a graded PICK
            ledger[key] = ent
            s["replay"] = True               # no frozen_at: nothing was frozen
            s["tier"] = LEGACY_TIER
            s.pop("frozen_at", None)
            continue
        _restore(s, ent, warn)
        n_frozen += 1
    return ledger, n_frozen, n_orphan


def restore_row(s: dict, ent: dict | None) -> dict:
    """Tier/receipt metadata for a row that just became played (CI results step).

    Never touches ph/pmc/ct: the payload's value IS the publication. Only the
    metadata the ledger holds about that publication is copied onto the row.
    The caller checks first that the entry describes THIS publication
    (nfl_results_attach.receipt_matches); a receipt for another number is
    never copied.
    """
    return apply_receipt(s, ent)


def tracking_since(ledger: dict) -> str:
    """Date the NFL pre-game ledger began publishing (YYYY-MM-DD)."""
    ts = [entry_time(e) for e in ledger.values()] or [LEGACY_T]
    return min(ts)[:10]
