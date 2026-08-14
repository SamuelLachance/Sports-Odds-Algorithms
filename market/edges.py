"""Attach the EV-threshold value edge (market/__init__.py) to an already-built, market-blind board.json.

Runs as a POST-PROCESS on the board the model produced: it never changes a model
probability, only compares the live market to it. Freezes each game's OPENING
consensus the first time it's seen (data/opening_odds_2026.json), flags the value
side when EV vs that opening >= 20%, and marks whether the edge is STILL live at the
current price (ev_cur > 0). Stale edges are cleared every run, so a badge disappears
once the line moves past our value or the game starts (fetch_consensus drops started
games). Cheap enough to run on a short cadence for near-live updates.

It also attaches an UNCONDITIONAL market quote (`mkt`: both sides' current consensus
decimals) to every pre-game card the odds feed covers, badge or no badge. The badge
is NOT a price feed: ev_open is recomputed each run against the CURRENT model
probability, so a routine model update (new probable starter, new lineup) can drop
the badge days before first pitch. market/edge_ledger.py rolls a recorded bet's
closing price forward from `mkt`, so the close it freezes is the real close rather
than the last price at which the badge happened to fire.
"""

from __future__ import annotations

def _write_atomic(path, text):
    """tmp + os.replace: an interrupted run must never truncate a live file."""
    import os
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, str(path))


import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market import odds

PROJECT = Path(__file__).resolve().parents[1]
BOARD = PROJECT / "site" / "data" / "board.json"
OPENING = PROJECT / "data" / "opening_odds_2026.json"
# Single source: market/__init__.py. The 15% gate failed the calibration
# audit (mid-dog bucket z=-4.5, see data/ev_gate_audit.json), hence 20%.
from market import EV_THRESHOLD


def attach_and_save(board_path: Path = BOARD, opening_path: Path = OPENING,
                    api_key: str | None = None) -> int:
    if not board_path.is_file():
        return 0
    payload = json.loads(board_path.read_text(encoding="utf-8"))
    mlb = next((lg for lg in payload.get("leagues", []) if lg.get("code") == "mlb"), None)
    if not mlb:
        return 0
    cards = mlb.get("games", [])
    live = odds.fetch_consensus(api_key)
    cache = json.loads(opening_path.read_text()) if opening_path.is_file() else {}
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()

    def _dt(s):
        try:
            return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    idx = defaultdict(list)
    for g in live:
        idx[(g["date"], g["home"], g["away"])].append(g)
        cache.setdefault(g["id"], {"home_dec": g["home_dec"], "away_dec": g["away_dec"],
                                   "date": g["date"], "captured": now})   # first sighting = opening
    n = 0
    for c in cards:
        c.pop("value", None)                       # recompute fresh each run
        c.pop("mkt", None)
        # POLICY GATE (documents/pick_policy.md rule 1). An EV badge is a stronger
        # claim than a pick -- it says "the market is wrong by 20%" -- so it must
        # never appear on a forecast made before the information exists. An
        # EARLY-tier card has no named starter: team ratings only. Today this is
        # latent (the odds feed carries nothing beyond ~2 days, so all quoted
        # games are PROJECTED or CONFIRMED), but the gate makes it structural
        # rather than dependent on a provider's posting window.
        if not (c.get("pitcher_known") or c.get("lineup_source") == "official"):
            continue
        # Never put a pre-game EV badge on a game that has already started. This is
        # also what stops a doubleheader's started game 1 from inheriting the surviving
        # game 2's line (game 1's live odds are dropped, collapsing the match key).
        cs = _dt(c.get("start_utc"))
        if (cs is not None and cs <= now_dt) or c.get("state") in ("Live", "Final"):
            continue
        cand = idx.get((c["date"], c["home"], c["away"]))
        if not cand:
            continue
        if len(cand) == 1:
            g = cand[0]
        else:                                      # doubleheader -> nearest first pitch
            g = min(cand, key=lambda x: abs(((_dt(x["commence"]) or now_dt)
                                             - (cs or now_dt)).total_seconds()))
        # Unconditional price feed for the ledger's closing line: written BEFORE the
        # EV gate so that losing the badge never truncates a recorded bet's price path.
        c["mkt"] = {"home_dec": round(g["home_dec"], 3),
                    "away_dec": round(g["away_dec"], 3),
                    "home_best": round(g["home_best"], 3) if g.get("home_best") else None,
                    "away_best": round(g["away_best"], 3) if g.get("away_best") else None,
                    "books": g["n_books"], "ts": now}
        op = cache.get(g["id"])
        if not op:
            continue
        p = c["home_win_prob"]
        oe = odds.value_side(p, op["home_dec"], op["away_dec"])
        side, ev_open = oe["side"], oe["ev"]
        if ev_open < EV_THRESHOLD:
            continue
        cur_dec = g["home_dec"] if side == "home" else g["away_dec"]
        ev_cur = (p if side == "home" else 1 - p) * cur_dec - 1.0
        c["value"] = {
            "side": side, "team": c["home_abbr"] if side == "home" else c["away_abbr"],
            "ev_open": round(ev_open, 4), "ev_cur": round(ev_cur, 4),
            "open_dec": round(op["home_dec"] if side == "home" else op["away_dec"], 3),
            "cur_dec": round(cur_dec, 3), "available": ev_cur > 0, "books": g["n_books"],
        }
        n += 1
    payload["value_updated"] = now
    _write_atomic(board_path, json.dumps(payload, indent=1))
    cutoff = (now_dt.date() - timedelta(days=2)).isoformat()   # bound the cache: drop past games
    cache = {k: v for k, v in cache.items() if v.get("date", "9999") >= cutoff}
    _write_atomic(opening_path, json.dumps(cache))
    return n
