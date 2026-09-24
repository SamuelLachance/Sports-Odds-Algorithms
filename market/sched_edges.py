"""The EV-badge layer shared by the two SCHEDULE payloads (site/data/nfl.json and
site/data/nhl.json), and the freshness guard both of them need.

Same POST-PROCESS contract as market/edges.py (MLB): it compares the live market
to the market-blind model's ALREADY-COMPUTED probability and never changes a model
number. Nothing here is a model input, and no rule below reads a price to decide
whether a badge may show: the guard reads the model payload and the clock only.

WHY A FRESHNESS GUARD (audit 2026-09-24)
  The badge is re-priced every 20 minutes (odds.yml), but the probability it is
  priced against is rebuilt on a different cadence: NFL by a LOCAL weekly chain,
  NHL by refresh.yml every 4 hours. Nothing checked the model's age. For eight
  weeks site/data/nfl.json kept its 2026-07-30 build while odds.yml badged it (up
  to +106% EV), and the edge ledger recorded 21 bets off those numbers — first
  record wins, so they cannot be corrected, only flagged. MLB is protected by
  construction: its board is rebuilt in the same 4-hourly job and its day labels
  are anchored to the build.

THE RULES  (a quoted game in the window gets a badge only if all hold)
  1. The model build is stamped: top-level `served_at` (preferred) or `generated`.
     The edge layer refuses a badge on an unstamped payload. The ledger, which
     re-checks a decision the edge layer already made, applies only what it can.
  2. The build is recent: at most MAX_MODEL_AGE_H[league] hours old.
       NFL 8 days  — the chain runs weekly after Monday night; 8 days is one
                     missed run plus a day of slack.
       NHL 36 h    — refresh.yml re-serves every 4 h; 36 h is nine missed runs
                     (GitHub skips scheduled runs; 12 h gaps have been observed).
  3. The forecast was made inside the badge window: the game is at most
     LIVE_WINDOW_DAYS after the build (ET dates). This is the pick policy's own
     tier line — NFL is PROJECTED inside 7 days of kickoff and EARLY beyond —
     applied to the moment the number was PRODUCED, not the moment it is shown.
  4. Both teams' earlier games are in the model. A game of either team that has
     started (kickoff / puck drop passed; date passed when no time is known) and
     whose result the model has not ABSORBED makes the probability for that team's
     next game stale. A score in the payload is NOT proof of absorption:
     phase0/nfl_results_attach.py (refresh.py, every 4 h) writes NFL finals into
     the payload without re-running the model and without moving served_at. So a
     scored game counts as absorbed only when
       (a) it was already final when the model was built: it started at least
           FINAL_AFTER_H hours before the build stamp (no start time: its ET date
           is before the build's ET date), AND
       (b) it is not dated after the payload's own "results walked through" date
           (THROUGH_KEYS: NHL ratings_through, NFL power_asof) when one is served.
     The rule is PER TEAM on purpose: the NFL chain lands a Thursday result the
     following Tuesday, and a global "any past game unabsorbed" rule would kill
     every Sunday badge each week. Only the last UNABSORBED_LOOKBACK_DAYS count, so
     a cancelled game (nflverse keeps those unscored forever) cannot freeze two
     teams' badges for a season; past that horizon rule 2 has long since taken over.
  5. The forecast is not EARLY (GATE_EARLY). documents/pick_policy.md rule 1: an
     EARLY forecast is not a pick, and an EV badge is a STRONGER claim than a pick
     (addendum 2026-07-31), so it must clear at least the same information bar —
     the same gate market/edges.py enforces for MLB. The tier is the payload row's
     own `tier`, else the league default (DEFAULT_TIER); a forecast made more
     than LIVE_WINDOW_DAYS before the game is EARLY whatever the row says, and one
     whose build cannot be dated is EARLY unless the row is stamped. Every
     NHL forecast is EARLY (team ratings only: no goalie or lineup input), so with
     the gate on NHL carries no badge at all until a goalie input ships; its games
     are still quoted (`mkt`) and each says why (`value_blocked`).

  KNOWN HOLE (NHL): phase0/nhl_update.py lists upcoming games in FUT/PRE state and
  finals in OFF/FINAL, so a game that is LIVE at serve time is in neither list and
  rule 4 cannot see it. The 4-hourly re-serve closes that within one cycle; rule 2
  bounds it if the pipeline stops.

POLICY SWITCH (owner-flippable)
  GATE_EARLY = True applies rule 5. The 2026-07-31 addendum chose to DISCLOSE NFL and
  NHL badges rather than gate them; that was written before this program's honesty
  rule restated that an EARLY forecast carries no pick and no fair odds, and a
  badge the ledger stakes in units while its note says "not a pick" contradicts
  itself. Setting GATE_EARLY = False restores disclosure: EARLY badges show again,
  carrying `value.tier` = "EARLY" and the league's `value.note`, and the edge ledger
  records them again.

PAYLOAD FIELDS WRITTEN BY attach()
  schedule[i].mkt            unconditional both-sides consensus quote for every
                             quoted game in the window (price feed for the edge
                             ledger's close; written even when the badge is held)
  schedule[i].value          the badge: side, team, ev_open, ev_cur, open_dec,
                             cur_dec, available, books, plus
                               tier      information tier of the forecast
                                         (forecast_tier; never "EARLY" while
                                         GATE_EARLY is on)
                               model_ts  UTC build stamp of the priced model
                               note      the disclosed information deficit
  schedule[i].value_blocked  why a QUOTED game inside the window carries no badge
                             (one of the rules above); absent otherwise. A rule-5
                             hold starts with "EARLY tier" (EARLY_PREFIX).
  value_blocked              top level, present only while the model build itself
                             is stale or unstamped AND games sit in the window
  value_status               {state, reason, model_ts, model_age_h, n_window,
                              n_quoted, n_blocked, n_early, gate_early, n_badges,
                              checked}
                               state: "idle"       no unplayed game in the window
                                      "suppressed" rule 1/2 failed: no badge at all
                                      "live"       badges evaluated; n_blocked of
                                                   n_quoted held by rules 3/4/5
                               n_early     of n_blocked, how many were held only
                                           because the forecast is EARLY (rule 5)
                               gate_early  GATE_EARLY at run time (bool)
  value_updated              unchanged (ISO time of this run)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from market import EV_THRESHOLD   # single source (market/__init__.py)
from market import odds

ET = ZoneInfo("America/New_York")
LIVE_WINDOW_DAYS = 7
MAX_MODEL_AGE_H = {"nfl": 8 * 24, "nhl": 36}
UNABSORBED_LOOKBACK_DAYS = 14
# Rule 4(a): a game counts as final at build time only if it started at least this
# long before the build. NFL games run ~3h10 (OT ~3h30), NHL ~2h30 (shootout ~3h).
# Erring long only holds a team's next badge until the following re-serve.
FINAL_AFTER_H = 4.0
# Rule 4(b): the payload's own "results walked through" date, first one served.
#   ratings_through  phase0/nhl_serve.py: date of the last final in the walked spine
#   power_asof       phase0/nfl_site_data.py (chain step 1): the last final walked
# Never results_through / results_updated: nfl_results_attach.py moves those
# without re-running the model.
THROUGH_KEYS = ("ratings_through", "power_asof")
# Rule 5. Owner-flippable: see POLICY SWITCH in the module docstring.
GATE_EARLY = True
DEFAULT_TIER = {"nfl": "PROJECTED", "nhl": "EARLY"}   # when a row carries no tier
EARLY_PREFIX = "EARLY tier"
EARLY_WHY = {
    "nhl": "team ratings only, no goalie or lineup input",
    "nfl": "forecast made before the week's depth charts and injury reports",
}


# ---------------------------------------------------------------- time helpers

def parse_ts(s) -> datetime | None:
    """ISO timestamp (or date) -> aware UTC datetime; naive input is read as UTC."""
    if isinstance(s, datetime):
        dt = s
    elif s in (None, ""):
        return None
    else:
        try:
            dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def et_date(dt: datetime) -> date:
    return dt.astimezone(ET).date()


def _date(s) -> date | None:
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def model_ts(payload: dict) -> datetime | None:
    """The model build's stamp: served_at (the serve's own clock), else generated."""
    for k in ("served_at", "generated"):
        dt = parse_ts(payload.get(k))
        if dt is not None:
            return dt
    return None


def game_start(g: dict) -> datetime | None:
    """UTC start of a schedule row: `start_utc` when served (NHL, ledger views),
    else NFL's date + Eastern clock time ("d": 2026-09-13, "t": "13:00")."""
    st = parse_ts(g.get("start_utc"))
    if st is not None:
        return st
    t, d = g.get("t"), g.get("d")
    if not (d and isinstance(t, str) and ":" in t):
        return None
    try:
        hh, mm = (int(x) for x in t.split(":")[:2])
        y, m, dd = (int(x) for x in d.split("-"))
        return datetime(y, m, dd, hh, mm, tzinfo=ET).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def started(g: dict, now: datetime) -> bool:
    """Kickoff / puck drop has passed. Without a start time: the ET date has."""
    st = game_start(g)
    if st is not None:
        return st <= now
    d = _date(g.get("d"))
    return d is not None and d < et_date(now)


# ---------------------------------------------------------------- freshness

def through_date(payload: dict) -> date | None:
    """The payload's own 'results walked through' date (rule 4b), if served."""
    for k in THROUGH_KEYS:
        d = _date(payload.get(k))
        if d is not None:
            return d
    return None


def absorbed(g: dict, built: datetime | None, through: date | None = None) -> bool:
    """Rule 4: is this game's result in the model build stamped `built`?

    A score alone does not say so (nfl_results_attach.py scores the payload between
    builds). Scored AND final before the build AND not past the walked-through date.
    Without a build stamp the score is the only evidence left (the ledger's lenient
    re-check; the edge layer refuses an unstamped build outright)."""
    if g.get("hs") is None or g.get("as") is None:
        return False
    d = _date(g.get("d"))
    if through is not None and d is not None and d > through:
        return False
    if built is None:
        return True
    st = game_start(g)
    if st is not None:
        return st + timedelta(hours=FINAL_AFTER_H) <= built
    return d is not None and d < et_date(built)


def unabsorbed(sched: list[dict], now: datetime, built: datetime | None = None,
               through: date | None = None) -> dict[str, str]:
    """team -> date of its earliest recent game that has started but whose result
    the model build `built` has not absorbed (rule 4)."""
    floor = (et_date(now) - timedelta(days=UNABSORBED_LOOKBACK_DAYS)).isoformat()
    out: dict[str, str] = {}
    for g in sched:
        d = g.get("d") or ""
        if d < floor or not started(g, now) or absorbed(g, built, through):
            continue
        for t in (g.get("home"), g.get("away")):
            if t and (t not in out or d < out[t]):
                out[t] = d
    return out


def forecast_tier(g: dict, built: datetime | None, default: str | None = None) -> str | None:
    """Information tier of the forecast behind row `g`, priced from build `built`.

    The row's own tier, else `default`; but a forecast made more than
    LIVE_WINDOW_DAYS before the game is EARLY whatever the row says (the pick
    policy's tier line, applied to when the number was produced). When that lead
    cannot be measured (no build stamp, or no date) the row's own stamp is all the
    evidence there is: without one the forecast is EARLY, never the default."""
    own = g.get("tier")
    if own == "EARLY":
        return own
    d = _date(g.get("d"))
    if built is None or d is None:
        return own or "EARLY"
    if (d - et_date(built)).days > LIVE_WINDOW_DAYS:
        return "EARLY"
    return own or default


def early_reason(league: str) -> str:
    why = EARLY_WHY.get(league, "forecast made before the information existed")
    return f"{EARLY_PREFIX}: {why}; shown as a lean, not a badge"


def assess(league: str, payload: dict, now: datetime) -> dict:
    """Model-level facts for one payload (rules 1, 2 and the rule-4 team map)."""
    mts = model_ts(payload)
    age_h = (now - mts).total_seconds() / 3600.0 if mts is not None else None
    lim = MAX_MODEL_AGE_H.get(league)
    reason = None
    if mts is None:
        reason = "model build carries no timestamp (served_at / generated)"
    elif lim is not None and age_h > lim:
        reason = (f"model last built {mts:%Y-%m-%d} ({age_h / 24:.1f} days ago; "
                  f"badges need a build under {lim / 24:g} days old)")
    thr = through_date(payload)
    return {"league": league, "model_dt": mts, "model_ts": iso(mts),
            "age_h": round(age_h, 1) if age_h is not None else None,
            "unstamped": mts is None, "stale": reason is not None and mts is not None,
            "reason": reason, "through": thr.isoformat() if thr else None,
            "unabsorbed": unabsorbed(payload.get("schedule") or [], now, mts, thr)}


def model_block(info: dict, strict: bool = True) -> str | None:
    """Rules 1-2: a reason when the model build itself cannot price any badge."""
    if info["unstamped"]:
        return info["reason"] if strict else None
    return info["reason"] if info["stale"] else None


def block_reason(info: dict, g: dict, strict: bool = True,
                 tier: str | None = None) -> str | None:
    """Why game `g` may not carry a badge from this build (None = it may).

    strict=True is the edge layer: an unstamped build blocks. strict=False is the
    ledger's re-check: an unstamped build skips rules 1-3; rule 4 still applies,
    and rule 5 judges the row's own tier. `tier` is the league default for a row
    that carries none (DEFAULT_TIER when omitted)."""
    why = model_block(info, strict)
    if why:
        return why
    mts = info["model_dt"]
    d = _date(g.get("d"))
    if mts is not None and d is not None:
        lead = (d - et_date(mts)).days
        if lead > LIVE_WINDOW_DAYS:
            return (f"forecast made {lead} days before the game (model built "
                    f"{et_date(mts).isoformat()}); a badge needs one made within "
                    f"{LIVE_WINDOW_DAYS} days")
    for t in (g.get("home"), g.get("away")):
        u = info["unabsorbed"].get(t)
        if u:
            return f"{t}'s {u} result is not in the model yet"
    if GATE_EARLY and forecast_tier(
            g, mts, tier or DEFAULT_TIER.get(info["league"])) == "EARLY":
        return early_reason(info["league"])
    return None


# ---------------------------------------------------------------- the badge layer

def _write_atomic(path, text: str) -> None:
    """tmp + os.replace: an interrupted run must never truncate a live file."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, str(path))


def attach(league: str, payload_path: Path, opening_path: Path, api_key: str | None,
           *, url: str, team_map: dict[str, str], p_key: str, tier: str, note: str,
           now: datetime | None = None) -> int:
    """Attach `mkt` + the EV badge to a schedule payload's live window, under the
    freshness guard. Returns the number of badges attached."""
    payload_path, opening_path = Path(payload_path), Path(opening_path)
    if not payload_path.is_file():
        return 0
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    sched = payload.get("schedule") or []
    if not sched:
        return 0
    now_dt = now or datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    today = et_date(now_dt).isoformat()
    horizon = (et_date(now_dt) + timedelta(days=LIVE_WINDOW_DAYS)).isoformat()
    for g in sched:                              # stale fields cleared every run
        for k in ("value", "mkt", "value_blocked"):
            g.pop(k, None)
    payload.pop("value_blocked", None)
    eligible = [g for g in sched
                if g.get("hs") is None and g.get("d")
                and today <= g["d"] <= horizon and not started(g, now_dt)]
    info = assess(league, payload, now_dt)
    held = model_block(info)
    status = {"state": "idle", "reason": held, "model_ts": info["model_ts"],
              "model_age_h": info["age_h"], "n_window": len(eligible),
              "n_quoted": 0, "n_blocked": 0, "n_early": 0, "gate_early": GATE_EARLY,
              "n_badges": 0, "checked": now_iso}

    def _save():
        payload["value_status"] = status
        payload["value_updated"] = now_iso
        _write_atomic(payload_path, json.dumps(payload))

    if not eligible:
        _save()
        return 0
    if held:
        payload["value_blocked"] = held

    live = odds.fetch_consensus(api_key, url=url, team_map=team_map)
    cache = json.loads(opening_path.read_text()) if opening_path.is_file() else {}
    idx = defaultdict(list)
    for g in live:
        idx[(g["date"], g["home"], g["away"])].append(g)
        cache.setdefault(g["id"], {"home_dec": g["home_dec"], "away_dec": g["away_dec"],
                                   "date": g["date"], "captured": now_iso})
    n = 0
    for c in eligible:
        cand = idx.get((c["d"], c["home"], c["away"]))
        if not cand:
            continue
        g = cand[0]
        # Unconditional price feed for market/edge_ledger.py (see market/edges.py):
        # written BEFORE the EV gate AND before the freshness guard, so neither a
        # badge that vanishes when the model moves nor one held for a stale build
        # can truncate a recorded bet's price path short of the close.
        c["mkt"] = {"home_dec": round(g["home_dec"], 3),
                    "away_dec": round(g["away_dec"], 3),
                    "home_best": round(g["home_best"], 3) if g.get("home_best") else None,
                    "away_best": round(g["away_best"], 3) if g.get("away_best") else None,
                    "books": g["n_books"], "ts": now_iso}
        status["n_quoted"] += 1
        why = block_reason(info, c, tier=tier)
        if why:
            c["value_blocked"] = why
            status["n_blocked"] += 1
            if why.startswith(EARLY_PREFIX):
                status["n_early"] += 1
            continue
        op = cache.get(g["id"])
        p = c.get(p_key)
        if not op or not isinstance(p, (int, float)):
            continue
        oe = odds.value_side(p, op["home_dec"], op["away_dec"])
        side, ev_open = oe["side"], oe["ev"]
        if ev_open < EV_THRESHOLD:
            continue
        cur_dec = g["home_dec"] if side == "home" else g["away_dec"]
        ev_cur = (p if side == "home" else 1 - p) * cur_dec - 1.0
        c["value"] = {
            "side": side, "team": c["home"] if side == "home" else c["away"],
            "ev_open": round(ev_open, 4), "ev_cur": round(ev_cur, 4),
            "open_dec": round(op["home_dec"] if side == "home" else op["away_dec"], 3),
            "cur_dec": round(cur_dec, 3), "available": ev_cur > 0, "books": g["n_books"],
            "tier": forecast_tier(c, info["model_dt"], tier),
            "model_ts": info["model_ts"], "note": note,
        }
        n += 1
    status["n_badges"] = n
    status["state"] = "suppressed" if held else "live"
    _save()
    cutoff = (now_dt.date() - timedelta(days=2)).isoformat()
    cache = {k: v for k, v in cache.items() if v.get("date", "9999") >= cutoff}
    _write_atomic(opening_path, json.dumps(cache))
    return n
