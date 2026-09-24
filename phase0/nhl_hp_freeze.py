"""The NHL pre-game prediction freeze. Mirror of phase0/nfl_ph_freeze.py.

nhl_serve.py rebuilds every current-season row on every run. For a game that has
been played, the rebuilt `hp` is a REPLAY: it is walk-forward, but it is not the
number the site published before puck drop (the published one was computed
from the upcoming slate, with whatever rest, back-to-back and boundary state
held at that moment). Grading the replay and calling it "frozen" - which the
record page does - is the flattering failure mode this module exists to stop.

data/nhl_hp_ledger.json stores, per game id, the last PRE-game `hp`, its `ct`
breakdown, the UTC time it was served (`t`) and the information tier it carried.

  - an UNPLAYED row refreshes its entry, so the stored value is the latest
    pre-game one;
  - a PLAYED row with an entry is restored from it (hp, ct, t, tier together);
  - a PLAYED row with NO entry cannot be repaired - it keeps the replay, is
    marked `replay: true` so the page can say so, and warns.
"""
from __future__ import annotations

import json
import os

LEDGER = "data/nhl_hp_ledger.json"


def load(path: str = LEDGER) -> dict:
    try:
        return json.load(open(path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(ledger: dict, path: str = LEDGER) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, separators=(",", ":"))
    os.replace(tmp, path)


def freeze(sched: list[dict], ledger: dict, now_utc: str, season: int,
           tier: str = "EARLY", warn=print) -> tuple[dict, int, int]:
    """Restore played current-season rows; refresh unplayed ones.

    Only rows of `season` take part: earlier seasons were never served through
    the ledger, so restoring or orphan-warning on them would be noise.
    Mutates `sched` in place; returns (ledger, n_frozen, n_replay).
    """
    n_frozen = n_replay = 0
    for s in sched:
        if int(str(s["id"])[:4]) != int(str(season)[:4]):
            continue
        key = str(s["id"])
        played = s.get("hs") is not None and s.get("as") is not None
        if not played:
            ledger[key] = {"hp": s["hp"], "ct": s.get("ct"), "t": now_utc, "tier": tier}
            s["tier"] = tier
            continue
        ent = ledger.get(key)
        if ent is None:
            n_replay += 1
            s["replay"] = True
            s["tier"] = tier
            warn(f"WARNING hp-freeze: no pre-game entry for played game {key}; "
                 f"serving the walk-forward replay, marked replay")
            continue
        s["hp"] = ent["hp"]
        if ent.get("ct") is not None:
            s["ct"] = ent["ct"]
        s["frozen_at"] = ent.get("t")
        s["tier"] = ent.get("tier", tier)
        n_frozen += 1
    return ledger, n_frozen, n_replay
