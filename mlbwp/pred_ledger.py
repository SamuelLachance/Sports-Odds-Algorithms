"""MLB pre-game prediction ledger — the honest-tracking archive.

The board is a rolling window: a game's pre-game probability vanishes from
site/data/board.json the day after it is played, so season-to-date REALIZED
accuracy was never computable from stored artifacts (the site's headline
accuracy is the frozen TEST-era number). This ledger fixes that, mirroring
the NFL ph-ledger and the NHL leak-safe replay:

  data/mlb_pred_ledger.json = { game_pk: {d, home, away, p, rp, sp_known,
                                          rec,  y?, hs?, as?} }

Rules (leak-safe by construction):
  - a game's entry is written/updated ONLY while the game is strictly
    pre-game (board state != Final AND start_utc in the future), so the
    stored p is always the latest PRE-game prediction;
  - once the game appears in season finals, the entry is graded (y attached)
    and never touched again.

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
        if g.get("state") == "Final" or start <= now:
            continue                        # not strictly pre-game: never touch
        ent = ledger.get(pk)
        if ent is not None and "y" in ent:
            continue                        # graded entries are immutable
        rec = {"d": g.get("date"), "home": g.get("home_abbr"),
               "away": g.get("away_abbr"), "p": round(float(p), 4),
               "rp": round(float(g["recal_prob"]), 4) if g.get("recal_prob") is not None else None,
               "sp_known": bool(g.get("pitcher_known")),
               "rec": now.strftime("%Y-%m-%dT%H:%MZ")}
        if ent is None:
            n_new += 1
        elif ent.get("p") != rec["p"]:
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
