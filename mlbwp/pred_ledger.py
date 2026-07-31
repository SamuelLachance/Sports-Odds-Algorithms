"""MLB pre-game prediction ledger — the honest-tracking archive.

The board is a rolling window: a game's pre-game probability vanishes from
site/data/board.json the day after it is played, so season-to-date REALIZED
accuracy was never computable from stored artifacts (the site's headline
accuracy is the frozen TEST-era number). This ledger fixes that, mirroring
the NFL ph-ledger and the NHL leak-safe replay:

  data/mlb_pred_ledger.json = { game_pk: {d, home, away, p, rp, sp_known,
                                          sp_proj, tier, rec,  y?, hs?, as?} }

Rules (leak-safe by construction):
  - a game's entry is written/updated ONLY while the game is strictly
    pre-game (board state != Final AND start_utc in the future), so the
    stored p is always the latest PRE-game prediction;
  - once the game appears in season finals, the entry is graded (y attached)
    and never touched again.

INFORMATION TIER (documents/pick_policy.md). Every pre-game write re-stamps
`tier` from the board card, so the stamp always describes the forecast that
will actually be graded — a game first seen 30 days out as EARLY ends up
stamped CONFIRMED if the official lineups post before first pitch. The exact
mapping from the fields the board card exposes:

  lineup_source == "official"          -> "CONFIRMED"   (starters + both
                                                         batting orders)
  else pitcher_known is True           -> "PROJECTED"   (both starters named,
                                                         roster-average lineup)
  else                                 -> "EARLY"       (team ratings only)

Two deliberate readings of the policy table, both erring toward describing
what the MODEL actually has rather than what the league has announced:
  - `pitcher_known` is True for a rotation-PROJECTED starter as well as an
    announced probable (the card flags which via `sp_projected`). The model
    conditions on a named arm either way, so both are PROJECTED, not EARLY.
  - a card with both starters known but no usable lineup (lineup_source None,
    e.g. the projector had no batters) is PROJECTED, not CONFIRMED.

Entries written before the tier stamp existed have no `tier` key; readers
fall back to sp_known (True -> PROJECTED, False -> EARLY) rather than
back-filling, because graded entries are immutable.

ONE narrow exception, for pre-stamp entries only: if a game has already
started but is still ungraded, and the board card's probability is BIT-FOR-BIT
the stored `p`, then the stored forecast IS the forecast that card describes,
so `info_tier(card)` labels it exactly. Only the label is written; `p` is never
touched after first pitch. If the probability differs at all, the card reflects
information the stored forecast did not have and the entry keeps its fallback
tier — which is why this can only ever move a legacy row to its true label, and
never inflates a forecast to a tier it did not earn.

SAMPLING CAVEAT (not fixable by labelling, disclosed on the site): the tier of
a graded row is the state at the LAST pre-game refresh, not at first pitch.
refresh.py runs every 4h, so a game whose lineups post 2h30 before a first
pitch that falls before the next refresh is graded PROJECTED even though the
lineups were public. CONFIRMED is therefore a start-time-selected subsample of
lineups-posted games, not all of them. Every label is true of the forecast it
is attached to; the tier *populations* are not random samples of the slate.

Runs inside refresh.py (stdlib-only). Games that were never seen pre-game
(e.g. ledger deployed mid-season, refresh outage) simply have no entry —
they are excluded from realized accuracy rather than back-filled post-hoc.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LEDGER = "data/mlb_pred_ledger.json"
BOARD = "site/data/board.json"
FINALS = "data/season_2026_finals.json"


TIERS = ("CONFIRMED", "PROJECTED", "EARLY")


def info_tier(card: dict) -> str:
    """Information tier of a board card — see the module docstring for the map."""
    if card.get("lineup_source") == "official":
        return "CONFIRMED"
    return "PROJECTED" if card.get("pitcher_known") else "EARLY"


def entry_tier(entry: dict) -> str:
    """Tier of a stored ledger entry, with the pre-stamp fallback."""
    t = entry.get("tier")
    if t in TIERS:
        return t
    return "PROJECTED" if entry.get("sp_known") else "EARLY"


def update(board_path: str = BOARD, finals_path: str = FINALS,
           ledger_path: str = LEDGER, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    try:
        ledger = json.load(open(ledger_path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        ledger = {}

    n_new = n_upd = n_graded = 0
    try:
        board = json.load(open(board_path, encoding="utf-8"))
        games = next((lg.get("games", []) for lg in board.get("leagues", [])
                      if lg.get("active")), [])
    except (FileNotFoundError, json.JSONDecodeError, StopIteration):
        games = []
    for g in games:
        pk = str(g.get("game_pk", ""))
        p = g.get("home_win_prob")
        if not pk or p is None:
            continue
        try:
            start = datetime.fromisoformat(g["start_utc"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        ent = ledger.get(pk)
        if ent is not None and "y" in ent:
            continue                        # graded entries are immutable
        if g.get("state") == "Final" or start <= now:
            # Not strictly pre-game: the NUMBER is locked forever. The only
            # write allowed here is stamping a pre-stamp entry whose stored p
            # is identical to this card's — same forecast, so the card's
            # information state describes it exactly (see module docstring).
            if (ent is not None and ent.get("tier") not in TIERS
                    and ent.get("p") == round(float(p), 4)):
                ent["tier"] = info_tier(g)
                ent["sp_proj"] = bool(g.get("sp_projected"))
                n_upd += 1
            continue
        rec = {"d": g.get("date"), "home": g.get("home_abbr"),
               "away": g.get("away_abbr"), "p": round(float(p), 4),
               "rp": round(float(g["recal_prob"]), 4) if g.get("recal_prob") is not None else None,
               "sp_known": bool(g.get("pitcher_known")),
               "sp_proj": bool(g.get("sp_projected")),
               "tier": info_tier(g),          # re-stamped on every pre-game write
               "rec": now.strftime("%Y-%m-%dT%H:%MZ")}
        if ent is None:
            n_new += 1
        elif ent.get("p") != rec["p"] or ent.get("tier") != rec["tier"]:
            n_upd += 1
        ledger[pk] = rec

    try:
        finals = json.load(open(finals_path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        finals = []
    for f in finals:
        pk = str(f.get("game_pk", ""))
        ent = ledger.get(pk)
        if ent is None or "y" in ent:
            continue
        if f.get("home_win") is None:
            continue
        ent["y"] = int(f["home_win"])
        ent["hs"] = f.get("home_score")
        ent["as"] = f.get("away_score")
        n_graded += 1

    tmp = ledger_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh)
    os.replace(tmp, ledger_path)
    return {"n": len(ledger), "new": n_new, "updated": n_upd, "graded": n_graded}


def realized(ledger_path: str = LEDGER) -> dict | None:
    """Season-to-date realized log-loss / accuracy over graded entries."""
    import math
    try:
        ledger = json.load(open(ledger_path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    done = [e for e in ledger.values() if "y" in e and e.get("p") is not None]
    if not done:
        return None
    eps = 1e-9
    ll = sum(-(e["y"] * math.log(max(e["p"], eps))
               + (1 - e["y"]) * math.log(max(1 - e["p"], eps)))
             for e in done) / len(done)
    acc = sum((e["p"] > 0.5) == (e["y"] == 1) for e in done) / len(done)
    return {"n": len(done), "log_loss": round(ll, 5), "acc": round(acc, 4)}


if __name__ == "__main__":
    print(update())
    print(realized())
