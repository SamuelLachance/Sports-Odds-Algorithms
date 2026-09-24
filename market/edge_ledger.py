"""LIVE EDGE LEDGER — the durable evidence stream behind the CLV promotion gate.

The three edge layers (market/edges.py, market/nfl_edges.py, market/nhl_edges.py)
recompute the EV-threshold value badge (market/__init__.py) from scratch every 20-minute cycle and store
NOTHING: data/*_opening_odds_2026.json caches first-sighting PRICES, never the fact
that an edge FIRED, the model probability behind it, or what the price did next. So
the documented promotion gate — "live avg CLV > 0 with >=100 CLV-graded bets per
league" — had no data accruing. This module is that missing stream.

POST-PROCESS ONLY (constitution): every number here comes from the already-built,
market-blind payload plus the market prices the edge layer already fetched. No model
file is read or written, no model feature ever sees a price.

WHAT IT RECORDS  (data/edge_ledger.json, {"v":1,"updated":ts,"rows":{key:row}})

  key      f"{league}|{gid}" when the payload offers a stable game id, else
           f"{league}|{d}|{away}@{home}"  (see GAME KEY below)
  league   mlb | nfl | nhl
  d        game date (ET for MLB/odds, schedule date for NFL/NHL)
  home/away, side ("home"/"away"), team   the side the badge fired on
  p_home   model home-win probability at record time (market-blind)
  p_model  model probability OF THE RECORDED SIDE (= p_home or 1-p_home).
           This is the p the EV and the Kaunitz z are computed against.
  dec_at_record   CONSENSUS decimal price of our side at the moment the edge was
                  recorded (median across US books, market/odds.fetch_consensus)
  best_dec_at_record / ev_best_at_record
                  BEST book price for our side in the SAME snapshot, and the EV
                  at it. Measurement only (the audit's open question: does
                  best-price entry cover the vig?): units settle at the median,
                  and CLV never sees this field — best-entry vs consensus-close
                  would manufacture CLV out of the vig spread.
  imp_at_record   1 / dec_at_record  (vig-inclusive implied probability)
  ev_at_record    p_model * dec_at_record - 1   (EV at the price we actually logged)
  ev_open, open_dec   the badge's headline EV vs the frozen OPENING consensus
  ev_cur          the badge's EV at the price we logged (the site's own "still
                  live?" number). Only badges the site marks AVAILABLE (ev_cur > 0)
                  are recorded: an unavailable badge is a price nobody could take,
                  and counting it would measure a book we never had.
  start_utc       first pitch / kickoff when the payload provides one, else null
  ts_record       UTC timestamp of the first (and only) record for this key
  last_pregame_dec / ts_last_pregame / n_obs
                  rolling latest STRICTLY-PRE-GAME consensus price for the RECORDED
                  SIDE; n_obs counts pre-game observations including the record
  n_missed / clv_censored
                  pre-game cycles where the odds feed was live for OTHER games but
                  carried no price for ours (postponement, book pull, event dropped).
                  Such a row's price path has a hole that ends before the real close,
                  so it is CLV-graded for the record but EXCLUDED from CLV statistics.
  dec_close / imp_close / clv_pts / ts_close     frozen at grading
  y        1 if OUR SIDE won, 0 if it lost, null until the result is known
  roi      (dec_at_record - 1) on a win, -1 on a loss  (flat 1u stake)
  model_ts UTC build stamp of the model payload that priced the bet (payload
           served_at / generated); rows recorded before 2026-09-24 carry none and
           are attributed from git history (LEGACY_MODEL_BUILDS)
  tier     information tier of the forecast at record time (MLB: the pick policy's
           tier of the card, mlbwp/pred_ledger.info_tier — NOT the card's own
           "tier" field, which is the blend tier; NFL/NHL: sched_edges.forecast_tier,
           i.e. the row's tier, EARLY when priced more than 7 days out). Rows
           recorded without one are derived at read time (row_tier).
  void_reason   set ONLY on positive evidence that the bet cannot settle (a tie on
           a two-way moneyline, the game gone from the league schedule on that date,
           or no final although the results source demonstrably moved past the
           date). Re-derived every cycle while the row is unsettled.

CLV SIGN CONVENTION
  clv_pts = imp_close - imp_at_record, in raw implied-probability POINTS.
  POSITIVE = the line moved TOWARD us: our side's price SHORTENED between record
  and close, so the implied probability rose and we got the better number. Negative
  = the market moved away from us. Both terms are 1/dec on the SAME side of the SAME
  median-consensus feed, so the book vig is common to both and largely cancels; the
  quantity is deliberately NOT de-vigged (de-vigging would require the opposite side
  at close, which is not always in the feed).

  This is consensus-at-record vs consensus-at-close, NOT best-of-N-books vs
  consensus — the documented lesson: comparing a best-price entry to a consensus
  close manufactures CLV out of the vig spread alone.

GAME KEY  (stable id ONLY when one exists — never the date)
  The MLB board carries game_pk, the MLB StatsAPI immutable primary key for a
  scheduled game — it is the most stable id in ANY of the three payloads (it also
  keys data/season_2026_finals.json and data/mlb_pred_ledger.json, so grading joins
  for free). NHL carries the same thing as "id". Where such an id exists it is the
  WHOLE key: a rainout, a rescheduling or an NFL flex changes the game's DATE while
  the bet stays the same bet, and a date in the key would open a second row for it
  (both then counted toward the >=100-row gate, the abandoned one force-graded off a
  truncated price path). Both ends of a doubleheader share date+home+away but never
  share game_pk, so the id also discriminates them. A rescheduled game keeps its row
  and its entry price; `d`/`start_utc` are refreshed in place and `rescheduled` is
  set. NFL has no per-game id in the payload, so its key stays date+matchup — a
  flexed NFL game is the one duplicate this cannot prevent.

PRICE SOURCE FOR THE CLOSE
  NOT the badge. The badge is recomputed every cycle from ev_open = p_CURRENT *
  open_dec - 1, and only open_dec is frozen: an adverse news item that moves the
  model's p removes the badge at exactly the moment the market lengthens our price,
  which would freeze a pre-news price as the "close" and manufacture positive CLV on
  a bet that lost CLV. The close therefore rolls off `mkt`, the unconditional
  both-sides consensus quote the edge layers attach to every pre-game card in the
  odds feed, badge or no badge (market/edges.py, nfl_edges.py, nhl_edges.py). It
  still costs ZERO extra Odds-API requests — it is the same fetch the badge uses.
  The badge's own cur_dec remains a fallback for the side we hold.

GRADING
  A row is CLV-graded once the game is no longer strictly pre-game: dec_close is
  frozen from last_pregame_dec. EXCLUDED from CLV statistics (recorded and settled
  as normal, but never counted toward the gate):
    * n_obs < 2 — recorded once, never seen again pre-game: clv_pts is a structural
      zero that would dilute the gate toward a false zero.
    * clv_censored — the price path has an observed hole (see n_missed), so
      last_pregame_dec is not the close. Censoring here is NOT random: it tracks
      postponements and pulled markets, so including these rows would bias the gate.
    * stale model — priced from a build that could not support a badge under the
      freshness rules of market/sched_edges.py (a build older than the league's
      limit at record time, or a forecast made more than 7 days before the game),
      or from a build nobody can date (no model_ts and outside every legacy
      window). Those rows measure an old or unknown model, not the one being
      promoted; they stay in the units P&L (the badge WAS published) and are
      flagged there.
  y/roi are filled when the result is known (MLB: data/season_2026_finals.json, or
  data/mlb_pred_ledger.json which is the tracked one and is the only one present in
  the odds-refresh checkout, or a board Final; NFL/NHL: hs/as in the schedule
  payload, else the COMMITTED results files data/nfl_games.csv / data/nhl_games.csv,
  so settlement never waits on a payload re-serve). Once y is set the row is frozen
  completely; dec_close is frozen from the moment it is written.

VOID vs AWAITING (audit 2026-09-24)
  An unsettled bet past its date used to be called void ("postponed/cancelled")
  after 7 days, on no evidence: six NFL week-1 bets that were played were shown as
  voided because nfl.json had not been re-served. A bet is now VOID only with a
  `void_reason` (see above); otherwise it is AWAITING settlement once its game has
  started, and OVERDUE once STALE_UNSETTLED_DAYS have passed — surfaced as a
  pipeline problem, never presented as a void.

THE GATE
  avg CLV > 0 alone is a bare sign test on a quantity whose per-bet noise dwarfs any
  plausible edge — 100 rows of pure noise pass it about half the time. PASS therefore
  also requires the mean to clear GATE_MIN_T standard errors of zero.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from market import sched_edges as FR     # freshness rules shared with the edge layers

PROJECT = Path(__file__).resolve().parents[1]
LEDGER = PROJECT / "data" / "edge_ledger.json"
BOARD = PROJECT / "site" / "data" / "board.json"
NFL_JSON = PROJECT / "site" / "data" / "nfl.json"
NHL_JSON = PROJECT / "site" / "data" / "nhl.json"
NFL_GAMES = PROJECT / "data" / "nfl_games.csv"    # nflverse schedule + finals (tracked)
NHL_GAMES = PROJECT / "data" / "nhl_games.csv"    # NHL finals spine (tracked, 4-hourly)
from mlbwp import season_paths           # noqa: E402  (PROJECT must resolve first)
from mlbwp.pred_ledger import info_tier as mlb_info_tier   # noqa: E402  (stdlib-only)

MLB_FINALS = season_paths.finals_cache()  # season-derived; see mlbwp/season_paths.py
MLB_PRED_LEDGER = PROJECT / "data" / "mlb_pred_ledger.json"

ET = ZoneInfo("America/New_York")
GATE_MIN_GRADED = 100          # promotion gate: >=100 CLV-graded bets per league
GATE_MIN_T = 2.0               # ...and the mean must clear this many standard errors
STALE_GRACE_DAYS = 2           # a row with no start time and no payload row is only
                               # force-graded once its date is this far in the past
STALE_UNSETTLED_DAYS = 7       # this long past its date with no result -> OVERDUE
                               # (surfaced as a pipeline problem, never called void)
VOID_COVER_DAYS = 2            # a results source "moved past" a date once it holds a
                               # final this many days later (evidence for a void)
LEAGUES = ("mlb", "nfl", "nhl")
PUSH = "push"                  # result marker: a tie on a two-way moneyline

# What each league's badge is, per documents/pick_policy.md. Shipped with the units
# block so #/record states it from data.
#   tier     the league's default information tier (a row carrying none; row_tier)
#   gate     "enforced": an EARLY forecast never carries a badge (MLB structurally,
#            market/edges.py; NFL/NHL while market/sched_edges.GATE_EARLY is on).
#            "disclosed": EARLY badges show, carrying the deficit below.
#   deficit  what a badge of this league is priced without, even when it shows
_SCHED_GATE = "enforced" if FR.GATE_EARLY else "disclosed"
BADGE_POLICY = {
    "mlb": {"tier": None, "gate": "enforced",
            "deficit": None},
    "nfl": {"tier": FR.DEFAULT_TIER["nfl"], "gate": _SCHED_GATE,
            "deficit": "priced from one weekly model build; injury and depth-chart "
                       "news after that build is not in it"},
    "nhl": {"tier": FR.DEFAULT_TIER["nhl"], "gate": _SCHED_GATE,
            "deficit": "team ratings only; priced before starting goalies are "
                       "confirmed (no goalie or lineup input yet)"},
}
TIERS = ("CONFIRMED", "PROJECTED", "EARLY")
UNKNOWN_TIER = "UNKNOWN"       # by_tier key for a row whose tier cannot be known

# Model builds behind rows recorded BEFORE rows carried `model_ts` (added
# 2026-09-24), reconstructed from the payloads' git history:
#   nfl  site/data/nfl.json kept the build stamped generated=2026-07-30T23:57:40Z
#        until ba24293a (2026-09-24T14:23:43Z) replaced it — every NFL bet before
#        that (21 rows, 09-06..09-24) was priced from the July build.
#   nhl  site/data/nhl.json carried no stamp before served_at shipped; the rows
#        before 753083c2 (2026-09-24T14:36:48Z) were priced from the weekly serve
#        committed in 1b25da38 (2026-09-21T16:12:17Z) — their p_home values match
#        that payload exactly (CAR 0.697, TOR 0.3649, TOR 0.4108).
# (league, recorded from (incl), recorded until (excl), model build)
LEGACY_MODEL_BUILDS = (
    ("nfl", "2026-07-30T23:57:40Z", "2026-09-24T14:23:43Z", "2026-07-30T23:57:40Z"),
    ("nhl", "2026-09-21T16:12:17Z", "2026-09-24T14:36:48Z", "2026-09-21T16:12:17Z"),
)


# ---------------------------------------------------------------- primitives

def _write_atomic(path, text: str) -> None:
    """tmp + os.replace: an interrupted cycle must never truncate the ledger."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, str(path))


def _dt(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None


def _date(s):
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _ts(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_key(league: str, d: str, home: str, away: str, gid=None) -> str:
    """The stable game id IS the key when the payload has one — the date is not part
    of it, so a postponed/rescheduled/flexed game keeps its single row instead of
    opening a second one (see GAME KEY in the module docstring)."""
    if gid not in (None, "", "None"):
        return f"{league}|{gid}"
    return f"{league}|{d}|{away}@{home}"


def _quarantine(path: Path) -> Path:
    """Move an unparseable ledger aside so the cycle can continue without the next
    save() silently os.replace-ing an empty ledger over the CLV history."""
    dest = Path(f"{path}.corrupt-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    os.replace(str(path), str(dest))
    print(f"[ledger] WARNING: {path} was unparseable; moved to {dest} and started "
          f"a fresh ledger. The CLV history is in that file, not lost.")
    return dest


def load(ledger_path=LEDGER) -> dict:
    """A MISSING ledger is a fresh ledger. An UNREADABLE one is not.

    Collapsing the two lets one transient read failure (a lock, a permission blip)
    return {} and the same cycle's save() overwrite the only copy of the evidence
    stream with nothing — silently, and the workflow commits the wipe. So: absent ->
    empty; corrupt/malformed -> renamed aside first, loudly; any other OSError ->
    raise, and let the cycle fail with the history intact on disk.
    """
    p = Path(ledger_path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"v": 1, "updated": None, "rows": {}}
    except OSError as exc:
        raise RuntimeError(
            f"edge ledger {p} exists but could not be read ({exc}); refusing to "
            f"continue and overwrite the live CLV history with an empty ledger") from exc
    try:
        obj = json.loads(raw)
        ok = isinstance(obj, dict) and isinstance(obj.get("rows"), dict)
    except json.JSONDecodeError:
        ok = False
    if not ok:
        _quarantine(p)
        return {"v": 1, "updated": None, "rows": {}}
    return obj


def save(ledger: dict, ledger_path=LEDGER, now: datetime | None = None) -> None:
    ledger["updated"] = _ts(now or datetime.now(timezone.utc))
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(ledger_path, json.dumps(ledger, indent=1, sort_keys=True))


# ---------------------------------------------------------------- payload adapters
# Each adapter turns a league payload into a uniform list of game views:
#   gid, d, home, away, start_utc, pregame, y_home, p_home, value, mkt
# `pregame` uses the SAME guard semantics the edge layers use (start in the future
# AND the payload does not say the game is under way / played).

def _mlb_games(payload: dict, now: datetime) -> list[dict]:
    lg = next((l for l in payload.get("leagues", []) if l.get("code") == "mlb"), None)
    out = []
    for c in (lg or {}).get("games", []) or []:
        st = _dt(c.get("start_utc"))
        state = c.get("state")
        pre = (st is None or st > now) and state not in ("Live", "Final")
        y = None
        if state == "Final" and c.get("home_score") is not None \
                and c.get("away_score") is not None:
            y = int(c["home_score"] > c["away_score"])
        # The card's own "tier" field is the BLEND tier ("lineup"/"bullpen"), not
        # the information tier: the ledger stamps the pick policy's tier, computed
        # by the one rule the prediction ledger and the SPA share.
        out.append({"gid": str(c["game_pk"]) if c.get("game_pk") is not None else None,
                    "d": c.get("date"), "home": c.get("home"), "away": c.get("away"),
                    "home_team": c.get("home_abbr"), "away_team": c.get("away_abbr"),
                    "start_utc": c.get("start_utc"), "pregame": pre, "y_home": y,
                    "p_home": c.get("home_win_prob"), "value": c.get("value"),
                    "mkt": c.get("mkt"), "tier": mlb_info_tier(c)})
    return out


def _nfl_start_utc(g: dict):
    """NFL payload stores date + ET kickoff time ("d": 2026-09-09, "t": "20:20")."""
    t = g.get("t")
    if not (g.get("d") and isinstance(t, str) and ":" in t):
        return None
    try:
        hh, mm = (int(x) for x in t.split(":")[:2])
        y, m, dd = (int(x) for x in g["d"].split("-"))
        return datetime(y, m, dd, hh, mm, tzinfo=ET).astimezone(timezone.utc) \
            .strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None


def _sched_games(payload: dict, now: datetime, p_key: str, gid_key=None,
                 start_fn=None) -> list[dict]:
    out = []
    for g in payload.get("schedule") or []:
        played = g.get("hs") is not None and g.get("as") is not None
        su = start_fn(g) if start_fn else None
        st = _dt(su)
        pre = (not played) and (st is None or st > now)
        # A tie (NFL) settles a two-way moneyline as a PUSH, not as a home loss.
        y = (PUSH if g["hs"] == g["as"] else int(g["hs"] > g["as"])) if played else None
        gid = g.get(gid_key) if gid_key else None
        out.append({"gid": str(gid) if gid is not None else None,
                    "d": g.get("d"), "home": g.get("home"), "away": g.get("away"),
                    "home_team": g.get("home"), "away_team": g.get("away"),
                    "start_utc": su, "pregame": pre, "y_home": y,
                    "p_home": g.get(p_key), "value": g.get("value"),
                    "mkt": g.get("mkt"), "tier": g.get("tier")})
    return out


def _nfl_games(payload, now):
    return _sched_games(payload, now, "ph", None, _nfl_start_utc)


def _nhl_start_utc(g: dict):
    """NHL payload serves puck drop as start_utc (phase0/nhl_serve.py). Without it a
    row stayed 'pregame' until the weekly result landed, every cycle after puck drop
    found no price (the odds feed drops started games) and censored the row out of
    the CLV gate — NHL contributed zero live-CLV evidence."""
    su = g.get("start_utc")
    return su if isinstance(su, str) and FR.parse_ts(su) is not None else None


def _nhl_games(payload, now):
    return _sched_games(payload, now, "hp", "id", _nhl_start_utc)


ADAPTERS = {"mlb": _mlb_games, "nfl": _nfl_games, "nhl": _nhl_games}
DEFAULT_PAYLOAD = {"mlb": BOARD, "nfl": NFL_JSON, "nhl": NHL_JSON}


def _side_dec(g: dict, side: str):
    """Current consensus decimal for ONE side of ONE game, badge-independent.

    `mkt` is the unconditional quote the edge layer attaches to every pre-game card
    in the odds feed; it is what keeps the closing line rolling after the badge stops
    firing. The badge's cur_dec is a fallback for older payloads (and only when the
    badge is on the side we are asking about). Returns None when this cycle offers no
    price for that side at all — the caller treats that as a hole, not as a close.
    """
    m = g.get("mkt") or {}
    dec = m.get(f"{side}_dec")
    if not isinstance(dec, (int, float)) or dec <= 1:
        v = g.get("value") or {}
        dec = v.get("cur_dec") if v.get("side") == side else None
    return float(dec) if isinstance(dec, (int, float)) and dec > 1 else None


def entries_from_games(league: str, games: list[dict]) -> list[dict]:
    """The edges that FIRED and were TAKEABLE this run: strictly-pre-game games
    carrying a badge the site itself marks available.

    edges.py attaches the badge on ev_OPEN but sets available = (ev_cur > 0); a badge
    that first fires long after the opening was frozen is exactly the case where the
    market has already moved past us, and the site shows it as not takeable. Recording
    it would put a price nobody could have bet into the ROI, the Kaunitz z and the CLV
    sample — measuring a book that was never offered.
    """
    out = []
    for g in games:
        v = g.get("value")
        if not v or not g.get("pregame") or g.get("p_home") is None:
            continue
        side = v.get("side")
        dec = v.get("cur_dec")
        if side not in ("home", "away") or not isinstance(dec, (int, float)) or dec <= 1:
            continue
        ev_cur = v.get("ev_cur")
        if v.get("available") is False or (isinstance(ev_cur, (int, float)) and ev_cur <= 0):
            continue                                # advertised as not takeable
        out.append({**g, "side": side, "team": v.get("team"),
                    "dec": float(dec), "ev_open": v.get("ev_open"),
                    "ev_cur": ev_cur, "open_dec": v.get("open_dec"),
                    "books": v.get("books")})
    return out


# ---------------------------------------------------------------- ledger ops

def record_open(league: str, entries: list[dict], ledger: dict,
                now: datetime | None = None) -> int:
    """FIRST RECORD WINS. A key already in the ledger is never re-opened, so a
    later (better or worse) price can never overwrite the entry we actually took."""
    now = now or datetime.now(timezone.utc)
    ts = _ts(now)
    rows = ledger.setdefault("rows", {})
    n = 0
    for e in entries:
        key = make_key(league, e["d"], e["home"], e["away"], e.get("gid"))
        if key in rows:
            continue                                    # immutable: first record wins
        p_home = float(e["p_home"])
        p_side = p_home if e["side"] == "home" else 1.0 - p_home
        dec = float(e["dec"])
        # Best available price for OUR side at the same record instant, from
        # the same snapshot the consensus came from (odds.fetch_consensus).
        # MEASUREMENT ONLY: units settle at dec_at_record (median) and CLV
        # stays median-vs-median — mixing a best entry into the consensus
        # close would manufacture CLV out of the vig spread (documented
        # lesson). This field exists to answer the audit's one open question:
        # does best-price entry cover the vig where the median does not?
        best = (e.get("mkt") or {}).get(f"{e['side']}_best")
        best = float(best) if isinstance(best, (int, float)) and best > 1 else None
        rows[key] = {
            "league": league, "key": key, "gid": e.get("gid"), "d": e["d"],
            "home": e["home"], "away": e["away"],
            "side": e["side"], "team": e.get("team"),
            "p_home": round(p_home, 6), "p_model": round(p_side, 6),
            "dec_at_record": round(dec, 4), "imp_at_record": round(1.0 / dec, 6),
            "ev_at_record": round(p_side * dec - 1.0, 6),
            "best_dec_at_record": round(best, 4) if best else None,
            "ev_best_at_record": round(p_side * best - 1.0, 6) if best else None,
            "ev_open": e.get("ev_open"), "ev_cur": e.get("ev_cur"),
            "open_dec": e.get("open_dec"),
            "books": e.get("books"), "start_utc": e.get("start_utc"),
            "model_ts": e.get("model_ts"), "tier": e.get("tier"),
            "ts_record": ts,
            "last_pregame_dec": round(dec, 4), "ts_last_pregame": ts, "n_obs": 1,
            "n_missed": 0, "clv_censored": False,
            "dec_close": None, "imp_close": None, "clv_pts": None, "ts_close": None,
            "y": None, "roi": None,
        }
        n += 1
    return n


def touch_pregame(league: str, games: list[dict], ledger: dict,
                  now: datetime | None = None) -> int:
    """Roll the recorded side's closing price forward while the game is STRICTLY
    pre-game. Never touches a row that is already graded (dec_close frozen or y set),
    and never touches a row recorded in this same cycle (that would double-count an
    observation).

    The price comes from the badge-independent `mkt` quote, NOT from the badge: the
    badge is re-gated every cycle on the CURRENT model probability, so it vanishes on
    a model update or flips sides, and rolling off it froze whatever price happened to
    be showing at that moment as the "close" (see PRICE SOURCE in the module
    docstring). A pre-game cycle that offers no price for our side while the feed is
    demonstrably live for other games is a HOLE in the path: counted in n_missed and
    flagged clv_censored, which keeps the row out of the CLV statistics.

    A rescheduled game (postponement, NFL flex) keeps its row — id-keyed — so `d` and
    `start_utc` are refreshed in place rather than opening a duplicate bet.
    """
    now = now or datetime.now(timezone.utc)
    ts = _ts(now)
    rows = ledger.get("rows", {})
    # Distinguish "this game has no market" from "the odds fetch returned nothing":
    # a transient Odds-API failure must not censor every open row in the book.
    feed_live = any(g.get("mkt") for g in games)
    n = 0
    for g in games:
        if not g.get("pregame"):
            continue                                    # started/played: never touch
        key = make_key(league, g["d"], g["home"], g["away"], g.get("gid"))
        row = rows.get(key)
        if row is None or row.get("y") is not None or row.get("dec_close") is not None:
            continue                                    # graded rows are immutable
        if g.get("d") and g["d"] != row.get("d"):       # postponed / flexed / moved
            row["d"], row["rescheduled"] = g["d"], True
        if g.get("start_utc") and g["start_utc"] != row.get("start_utc"):
            row["start_utc"] = g["start_utc"]
        if row.get("ts_record") == ts:
            continue                                    # recorded this cycle
        dec = _side_dec(g, row["side"])
        if dec is None:
            if feed_live:                               # market pulled, not API-down
                row["n_missed"] = int(row.get("n_missed", 0)) + 1
                row["clv_censored"] = True
            continue
        row["last_pregame_dec"] = round(dec, 4)
        row["ts_last_pregame"] = ts
        row["n_obs"] = int(row.get("n_obs", 1)) + 1
        n += 1
    return n


def _void_evidence(row: dict, source: dict, now: datetime) -> str | None:
    """POSITIVE evidence that an unsettled, no-longer-pregame bet cannot settle, or
    None. Silence is not evidence: a results file that has not been read past the
    game's date says nothing about the game (it is AWAITING / OVERDUE, not void)."""
    d = _date(row.get("d"))
    if d is None:
        return None
    today = now.astimezone(ET).date()
    if d >= today:
        return None
    key = row.get("key") or make_key(row.get("league"), row.get("d"), row.get("home"),
                                     row.get("away"), row.get("gid"))
    listed, span = source.get("listed"), source.get("span")
    if listed is not None and span and span[0] <= row["d"] <= span[1] \
            and key not in listed:
        return "not on the league schedule on this date (moved or cancelled)"
    through = _date(source.get("through"))
    if through is not None and (today - d).days >= STALE_UNSETTLED_DAYS \
            and (through - d).days >= VOID_COVER_DAYS:
        return (f"no final for this game although results through "
                f"{through.isoformat()} were read (postponed or cancelled)")
    return None


def grade(league: str, games: list[dict], ledger: dict, now: datetime | None = None,
          results: dict | None = None, source: dict | None = None) -> dict:
    """Freeze the close and settle the result once a game is no longer pre-game.

    Stage 1 (CLV): dec_close = last_pregame_dec, imp_close = 1/dec_close,
      clv_pts = imp_close - imp_at_record  (positive = line moved TOWARD us).
    Stage 2 (result): y = 1 if OUR side won, roi = dec_at_record-1 on a win else -1.
    Both stages are write-once; a settled row (y set) is never written again.

    `games` is this cycle's payload view. `results` is an optional gid -> home_win
    map (the original MLB hook). `source` is a league results source (nfl_source,
    nhl_source, mlb_source): {"y": {ledger key: 0|1|"push"}, "through": last final
    date read, "listed": schedule keys or None, "span": (first, last) date}. It
    settles rows the payload cannot (MLB board is forward-only; the NFL/NHL payloads
    are re-served on their own cadence) and is the ONLY thing that may void a row
    (see _void_evidence). A two-way moneyline tie is a push: void, stake returned.
    """
    now = now or datetime.now(timezone.utc)
    ts = _ts(now)
    rows = ledger.get("rows", {})
    seen = {make_key(league, g["d"], g["home"], g["away"], g.get("gid")): g for g in games}
    n_close = n_settle = 0
    for key, row in rows.items():
        if row.get("league") != league or row.get("y") is not None:
            continue                                    # settled rows are immutable
        g = seen.get(key)
        st = _dt(row.get("start_utc"))
        if g is not None:
            pre = g["pregame"]
        elif st is not None:
            # dropped from the payload (MLB board is forward-only): the recorded
            # start time decides.
            pre = st > now
        else:
            # no start time and no payload row. Do NOT let a transient/blank payload
            # freeze a live row: wait out a grace window, then treat the game as long
            # played. Whatever price this closes on was the last one before we lost
            # sight of the game, not the close -> censored.
            pre = (now.date() - _date(row.get("d"))).days <= STALE_GRACE_DAYS \
                if _date(row.get("d")) else True
            if not pre and row.get("dec_close") is None:
                row["clv_censored"] = True
        if pre:
            row.pop("void_reason", None)                # back on the schedule
            continue
        if row.get("dec_close") is None:
            dec_c = float(row["last_pregame_dec"])
            row["dec_close"] = round(dec_c, 4)
            row["imp_close"] = round(1.0 / dec_c, 6)
            row["clv_pts"] = round(1.0 / dec_c - float(row["imp_at_record"]), 6)
            row["ts_close"] = ts
            n_close += 1
        y_home = None
        if g is not None and g.get("y_home") is not None:
            y_home = g["y_home"]
        elif results and row.get("gid") is not None:
            y_home = results.get(str(row["gid"]))
        if y_home is None and source is not None:
            y_home = (source.get("y") or {}).get(key)
        if y_home == PUSH:
            row["void_reason"] = "tie: a two-way moneyline pushes (stake returned)"
            continue
        if y_home is None:
            if source is not None:                      # evidence or nothing
                vr = _void_evidence(row, source, now)
                if vr:
                    row["void_reason"] = vr
                else:
                    row.pop("void_reason", None)
            continue
        row.pop("void_reason", None)
        y_home = int(y_home)
        y = y_home if row["side"] == "home" else 1 - y_home
        row["y"] = int(y)
        row["roi"] = round(float(row["dec_at_record"]) - 1.0, 6) if y else -1.0
        n_settle += 1
    return {"closed": n_close, "settled": n_settle}


# ---------------------------------------------------------------- results sources

def _read_json(path):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def mlb_results(finals_path=MLB_FINALS, pred_ledger_path=MLB_PRED_LEDGER) -> dict:
    """game_pk -> home_win (0/1), from BOTH tracked and untracked result sources.

    data/season_2026_finals.json is gitignored and is written only by refresh.py in
    the refresh.yml runner; odds.yml runs refresh_odds.py, which never builds it. The
    MLB board cannot cover the gap either — it is forward-only and never carries a
    Final with scores. So in the runner where this ledger actually accrues, the finals
    archive alone means MLB rows CLV-grade and then sit at y=null forever, with ROI
    and the Kaunitz z reported as a flat 0.0 that reads like a measurement.
    data/mlb_pred_ledger.json IS tracked, is keyed by game_pk, and already carries
    y/hs/as — so it is read as a second source (finals win any disagreement).
    """
    return _mlb_read(finals_path, pred_ledger_path)[0]


def _mlb_read(finals_path, pred_ledger_path) -> tuple[dict, str | None]:
    """(game_pk -> home_win, latest date carrying a final) from both MLB sources."""
    out, through = {}, None
    pred = _read_json(pred_ledger_path)
    if isinstance(pred, dict):
        for pk, r in pred.items():
            if not isinstance(r, dict):
                continue
            hs, a_s, y = r.get("hs"), r.get("as"), r.get("y")
            if hs is not None and a_s is not None:
                out[str(pk)] = int(hs > a_s)
            elif y is not None:
                out[str(pk)] = int(float(y) > 0.5)
            else:
                continue
            d = r.get("d")
            if isinstance(d, str) and (through is None or d > through):
                through = d
    for f in _read_json(finals_path) or []:
        if not isinstance(f, dict):
            continue
        pk, hw = f.get("game_pk"), f.get("home_win")
        if pk is not None and hw is not None:
            out[str(pk)] = int(float(hw) > 0.5)
            d = f.get("date")
            if isinstance(d, str) and (through is None or d > through):
                through = d
    return out, through


def _csv_rows(path):
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def nfl_source(path=NFL_GAMES) -> dict | None:
    """data/nfl_games.csv (nflverse schedule, committed by nfl-weekly.yml) as a
    results source, keyed like the ledger: nfl|gameday|away@home.

    The NFL payload is re-served by a LOCAL weekly chain, so on 2026-09-24 the
    ledger showed 0 of 12 played week-1/2 bets settled while this committed file
    already had every final. It is also a full-season SCHEDULE, which is what lets a
    moved or cancelled game be recognised (listed/span) instead of guessed."""
    rows = _csv_rows(path)
    if rows is None:
        return None
    y, listed, through, lo, hi = {}, set(), None, None, None
    for r in rows:
        d, a, h = r.get("gameday"), r.get("away_team"), r.get("home_team")
        if not (d and a and h):
            continue
        k = make_key("nfl", d, h, a)
        listed.add(k)
        lo = d if lo is None or d < lo else lo
        hi = d if hi is None or d > hi else hi
        hs, as_ = _int(r.get("home_score")), _int(r.get("away_score"))
        if hs is None or as_ is None:
            continue
        y[k] = PUSH if hs == as_ else int(hs > as_)
        through = d if through is None or d > through else through
    return {"y": y, "through": through, "listed": listed,
            "span": (lo, hi) if lo else None}


def nhl_source(path=NHL_GAMES) -> dict | None:
    """data/nhl_games.csv (finals spine, committed every 4 h by refresh.yml) as a
    results source, keyed like the ledger: nhl|game_id. Finals only (no schedule),
    so a void needs the 'results moved past the date' evidence."""
    rows = _csv_rows(path)
    if rows is None:
        return None
    y, through = {}, None
    for r in rows:
        gid, d = r.get("game_id"), r.get("date")
        if not gid or not d:
            continue
        hs, as_ = _int(r.get("home_goals")), _int(r.get("away_goals"))
        hw = _int(r.get("home_win"))
        if hs is not None and as_ is not None and hs != as_:
            hw = int(hs > as_)
        if hw is None:
            continue
        y[make_key("nhl", d, r.get("home"), r.get("away"), gid)] = int(hw > 0)
        through = d if through is None or d > through else through
    return {"y": y, "through": through, "listed": None, "span": None}


# ---------------------------------------------------------------- driver

def _mlb_results_for(results_path, pred_ledger_path) -> dict:
    """Result sources for one MLB cycle. An EXPLICIT results_path (tests, replays)
    means "these are the results" and does not silently pull in the repo's tracked
    prediction ledger; the production call passes neither and gets both."""
    return mlb_source(results_path, pred_ledger_path)["gid"]


def mlb_source(results_path=None, pred_ledger_path=None) -> dict:
    """The MLB results source, same path rules as _mlb_results_for."""
    if results_path is not None:
        gid, through = _mlb_read(results_path, pred_ledger_path)
    else:
        gid, through = _mlb_read(MLB_FINALS, pred_ledger_path or MLB_PRED_LEDGER)
    return {"gid": gid, "y": {f"mlb|{pk}": v for pk, v in gid.items()},
            "through": through, "listed": None, "span": None}


LEAGUE_SOURCES = {"nfl": (nfl_source, NFL_GAMES), "nhl": (nhl_source, NHL_GAMES)}


def _stamp_entries(league: str, payload: dict, entries: list[dict],
                   now: datetime) -> tuple[list[dict], int]:
    """Stamp each fired edge with the model build and tier behind it, and drop any
    the freshness rules forbid (market/sched_edges.py). The edge layer already
    refused those badges; this re-check keeps the LEDGER from accruing a stale-model
    row whatever wrote the payload (a re-serve carrying old badges forward, an older
    edge layer). Lenient on an unstamped build — it cannot judge what it cannot
    date — but the per-team unabsorbed-result rule and the EARLY gate (on the row's
    own tier) still apply."""
    if league == "mlb":                  # provenance only: MLB is gated upstream
        for e in entries:
            e["model_ts"] = payload.get("generated")
        return entries, 0
    info = FR.assess(league, payload, now)
    default = BADGE_POLICY[league]["tier"]
    kept, blocked = [], 0
    for e in entries:
        if FR.block_reason(info, e, strict=False, tier=default):
            blocked += 1
            continue
        e["model_ts"] = info["model_ts"]
        e["tier"] = FR.forecast_tier(e, info["model_dt"], default)
        kept.append(e)
    return kept, blocked


def update(league: str, payload_path=None, ledger_path=LEDGER,
           now: datetime | None = None, results_path=None,
           ledger: dict | None = None, write: bool = True,
           pred_ledger_path=None, league_results_path=None) -> dict:
    """One league, one cycle: record fired edges, roll the close, grade.

    Degrades to a clean no-op (all zeros, nothing written to disk beyond the
    ledger's own timestamp) when the payload is missing, unparseable, empty, or
    simply has no edges — i.e. the offseason path.

    NFL/NHL results: `league_results_path` names the results csv. When omitted,
    the committed repo file is used only on the production path (payload_path
    omitted, or update_all); a caller passing its own payload gets no repo results
    silently mixed in — the same rule MLB's results_path follows.
    """
    now = now or datetime.now(timezone.utc)
    own = ledger is None
    ledger = load(ledger_path) if own else ledger
    stat = {"league": league, "recorded": 0, "touched": 0, "closed": 0, "settled": 0,
            "payload": False}
    path = Path(payload_path or DEFAULT_PAYLOAD.get(league, ""))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        if own and write:
            save(ledger, ledger_path, now)
        return stat
    games = ADAPTERS[league](payload, now)
    stat["payload"] = True
    entries, stat["blocked"] = _stamp_entries(league, payload,
                                              entries_from_games(league, games), now)
    stat["recorded"] = record_open(league, entries, ledger, now)
    stat["touched"] = touch_pregame(league, games, ledger, now)
    # grading runs even on an empty game list: the MLB board is forward-only, so a
    # played game is GONE from the payload exactly when it needs to be graded.
    if league == "mlb":
        src = mlb_source(results_path, pred_ledger_path)
    else:
        fn, default = LEAGUE_SOURCES[league]
        rp = league_results_path or (default if payload_path is None else None)
        src = fn(rp) if rp else None
    g = grade(league, games, ledger, now, source=src)
    stat["closed"], stat["settled"] = g["closed"], g["settled"]
    if own and write:
        save(ledger, ledger_path, now)
    return stat


def update_all(ledger_path=LEDGER, now: datetime | None = None,
               payloads: dict | None = None, results_path=None,
               pred_ledger_path=None, league_results: dict | None = None) -> list[dict]:
    """Every league in one load/save cycle (called from refresh_odds.py).

    `league_results` maps nfl/nhl to a results csv. The committed repo file is the
    default only for a league read from its production payload — the same rule as
    update(): a caller's own payload never gets repo results silently mixed in."""
    now = now or datetime.now(timezone.utc)
    ledger = load(ledger_path)
    own = payloads or {}
    lr = league_results or {}
    out = []
    for lg in LEAGUES:
        pp = own.get(lg, DEFAULT_PAYLOAD[lg])
        rp = lr.get(lg)
        if rp is None and lg not in own:
            rp = {"nfl": NFL_GAMES, "nhl": NHL_GAMES}.get(lg)
        out.append(update(lg, pp, ledger_path, now, results_path,
                          ledger=ledger, write=False,
                          pred_ledger_path=pred_ledger_path,
                          league_results_path=rp))
    save(ledger, ledger_path, now)
    return out


# ---------------------------------------------------------------- reporting

def _exp_vs_real(bets):
    """Kaunitz eq-8 calibration z. Reuses phase0/ev_gate_audit.exp_vs_real when its
    dependency (numpy) is importable — the CI odds job is stdlib-only, so an
    identical stdlib fallback keeps the ledger importable there. Both compute
    z = (realized - sum p) / sqrt(sum p(1-p)); tests pin them to agree."""
    try:
        import sys
        p0 = str(PROJECT / "phase0")
        if p0 not in sys.path:
            sys.path.insert(0, p0)
        from ev_gate_audit import exp_vs_real            # noqa: PLC0415
        return exp_vs_real(bets)
    except Exception:                                    # noqa: BLE001
        if not bets:
            return 0, 0.0, 0, 0.0
        exp = sum(b["p"] for b in bets)
        var = sum(b["p"] * (1 - b["p"]) for b in bets)
        real = sum(b["win"] for b in bets)
        return len(bets), exp, real, float((real - exp) / math.sqrt(var)) if var > 0 else 0.0


def _units_net(bets):
    """Flat 1u per settled bet: dec-1 on a win, -1 on a loss. The ROI below is
    this same quantity per unit staked, so the two can never disagree."""
    return sum((b["odds"] - 1) if b["win"] else -1.0 for b in bets)


def _roi(bets):
    """Flat-stake ROI per unit staked — identical definition to ev_gate_audit.roi."""
    if not bets:
        return 0.0
    return _units_net(bets) / len(bets)


def _mean_t(xs: list[float]) -> tuple[float, float, float]:
    """mean, standard error of the mean, t = mean/se. A degenerate (zero-variance)
    sample gets t = +/-inf by sign, which is the honest reading of "every observation
    agrees"; an empty sample gets zeros."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0.0:
        return mean, 0.0, math.copysign(math.inf, mean) if mean else 0.0
    return mean, se, mean / se


def _stale_unsettled(rows: list[dict], now: datetime) -> int:
    """CLV-graded rows still without a result long after the close and WITHOUT void
    evidence: results never read in (a pipeline problem) — or a postponement no
    source has confirmed yet. Nothing else in the report would show them, and they
    quietly inflate n_clv_graded toward the gate."""
    n = 0
    for r in rows:
        if r.get("y") is not None or r.get("dec_close") is None or r.get("void_reason"):
            continue
        tc = _dt(r.get("ts_close"))
        if tc is not None and (now - tc).days >= STALE_UNSETTLED_DAYS:
            n += 1
    return n


def _row_model_ts(r: dict) -> datetime | None:
    """The model build that priced a row: its own stamp, else LEGACY_MODEL_BUILDS."""
    mts = FR.parse_ts(r.get("model_ts"))
    if mts is not None:
        return mts
    rec = FR.parse_ts(r.get("ts_record"))
    if rec is None:
        return None
    for lg, a, b, m in LEGACY_MODEL_BUILDS:
        if r.get("league") == lg and FR.parse_ts(a) <= rec < FR.parse_ts(b):
            return FR.parse_ts(m)
    return None


def stale_model_reason(r: dict) -> str | None:
    """Why a row was priced from a build the freshness rules (market/sched_edges.py)
    would not badge from, or None. Applies to NFL/NHL; MLB is gated upstream and its
    board is rebuilt in the same job that badges it. Outcome-blind: only the record
    time, the build stamp and the game date decide."""
    lg = r.get("league")
    lim = FR.MAX_MODEL_AGE_H.get(lg)
    if lim is None:
        return None
    mts = _row_model_ts(r)
    if mts is None:
        # No stamp and outside every LEGACY_MODEL_BUILDS window: recorded by code
        # that did not stamp builds yet (the committed ledger keeps running in CI
        # until this change deploys), or from an unstamped payload. A build nobody
        # can date is not evidence for the model being promoted: out of the gate.
        if r.get("ts_record"):
            return "model build unknown (recorded without a model build stamp)"
        return None
    rec = FR.parse_ts(r.get("ts_record"))
    if rec is not None and (rec - mts).total_seconds() / 3600.0 > lim:
        return (f"recorded {(rec - mts).total_seconds() / 86400.0:.0f} days after its "
                f"model build ({FR.et_date(mts).isoformat()})")
    d = _date(r.get("d"))
    if d is not None:
        lead = (d - FR.et_date(mts)).days
        if lead > FR.LIVE_WINDOW_DAYS:
            return (f"forecast made {lead} days before the game (model built "
                    f"{FR.et_date(mts).isoformat()})")
    return None


def row_tier(r: dict) -> str | None:
    """Information tier of the forecast behind a ledger row — what #/record shows.

    MLB: the row's own stamp (pred_ledger.info_tier at record time), else None —
    rows recorded before the stamp cannot be re-derived. NFL/NHL: the row's stamp,
    else the league default, but EARLY whenever the pricing build was made more
    than LIVE_WINDOW_DAYS before the game, or when the build cannot be dated and the
    row carries no stamp of its own (sched_edges.forecast_tier). The legacy NFL
    rows priced from the 2026-07-30 build 45-63 days out are therefore EARLY, as
    the pick policy's tier line (and nfl_ph_freeze.tier_for) says they were."""
    lg = r.get("league")
    t = r.get("tier")
    if lg == "mlb" or lg not in BADGE_POLICY:
        return t if t in TIERS else None
    return FR.forecast_tier(r, _row_model_ts(r), BADGE_POLICY[lg]["tier"])


def bet_status(r: dict, now: datetime) -> str:
    """settled | void | open | awaiting | overdue — one word per bet slip.

    void     positive evidence only (void_reason; see _void_evidence)
    open     the game has not started
    awaiting started, result not read in yet
    overdue  awaiting for STALE_UNSETTLED_DAYS or more past the game date: the
             result has not been ingested — a pipeline problem, NOT a void."""
    if r.get("y") is not None:
        return "settled"
    if r.get("void_reason"):
        return "void"
    d = _date(r.get("d"))
    st = _dt(r.get("start_utc"))
    today = now.astimezone(ET).date()
    begun = st <= now if st is not None else (d is not None and d < today)
    if not begun:
        return "open"
    if d is not None and (today - d).days >= STALE_UNSETTLED_DAYS:
        return "overdue"
    return "awaiting"


def _close_lag_mins(rows: list[dict]) -> list[float]:
    """Minutes between our LAST pre-game observation and first pitch/puck drop.

    Our "closing" price is only as fresh as the last cron that saw the game. The
    odds job is scheduled every 20 minutes but GitHub runs schedules on a
    best-effort basis, and the observed cadence has been ~1h with multi-hour
    gaps. A stale last observation biases CLV toward zero (we miss the final
    move), so the gate must be read alongside this number rather than on its own.
    Measured, not assumed — every CLV figure carries its own quality metric.
    """
    out = []
    for r in rows:
        st, ts = _dt(r.get("start_utc")), _dt(r.get("ts_last_pregame"))
        if st and ts:
            out.append((st - ts).total_seconds() / 60.0)
    return [m for m in out if m >= 0]


def _stats(rows: list[dict], now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    graded = [r for r in rows if r.get("clv_pts") is not None]
    # A row whose price path has a hole (postponement, pulled market, lost sight of
    # the game) did not observe the close, and the hole is not missing-at-random —
    # it tracks the very news the market reacted to. Graded for the record, excluded
    # from the gate. Same for n_obs < 2 (structural zero).
    # A row priced from a stale build measures an old model, not the one the gate
    # would promote (stale_model_reason; outcome-blind) -> out of the gate too.
    stale = [r for r in rows if stale_model_reason(r)]
    clv = [r for r in graded
           if int(r.get("n_obs", 1)) >= 2 and not r.get("clv_censored")
           and not stale_model_reason(r)]
    settled = [r for r in rows if r.get("y") is not None]
    bets = [{"p": r["p_model"], "win": r["y"] == 1, "odds": r["dec_at_record"]}
            for r in settled]
    # Hypothetical settle at the BEST book price recorded at the same instant
    # (see best_dec_at_record in record_open): the audit's open question —
    # whether best-price entry covers the vig — measured live, in parallel,
    # never mixed into the real (median-priced) units or into CLV.
    bets_best = [{"p": r["p_model"], "win": r["y"] == 1,
                  "odds": r["best_dec_at_record"]}
                 for r in settled if r.get("best_dec_at_record")]
    n, exp, real, z = _exp_vs_real(bets)
    avg_clv, se_clv, t_clv = _mean_t([r["clv_pts"] for r in clv])
    pos = sum(1 for r in clv if r["clv_pts"] > 0) / len(clv) if clv else 0.0
    enough = len(clv) >= GATE_MIN_GRADED
    # avg > 0 alone is a bare sign test: at n=100 pure noise passes it half the time.
    # Require the mean to also clear GATE_MIN_T standard errors of zero.
    gate = ("PASS" if (enough and avg_clv > 0 and t_clv >= GATE_MIN_T)
            else ("FAIL" if enough else "PENDING"))
    return {
        "n_recorded": len(rows), "n_clv_graded": len(clv), "n_settled": len(settled),
        "n_clv_censored": sum(1 for r in graded if r.get("clv_censored")),
        "n_stale_unsettled": _stale_unsettled(rows, now),
        "n_void": sum(1 for r in rows if r.get("y") is None and r.get("void_reason")),
        "n_stale_model": len(stale),
        "avg_clv_pts": round(avg_clv, 5), "se_clv_pts": round(se_clv, 5),
        "t_clv": round(t_clv, 3) if math.isfinite(t_clv) else None,
        "pct_positive_clv": round(pos, 4),
        "roi": round(_roi(bets), 5), "n_bets": n,
        "units_staked": float(len(bets)), "units_net": round(_units_net(bets), 3),
        "n_best": len(bets_best), "roi_best": round(_roi(bets_best), 5),
        "units_net_best": round(_units_net(bets_best), 3),
        "expected_wins": round(exp, 2), "realized_wins": int(real), "z": round(z, 3),
        "gate": gate, "gate_needs": max(0, GATE_MIN_GRADED - len(clv)),
        # price-freshness quality metric (see _close_lag_mins): how stale the
        # recorded "close" actually was. A large median here means CLV is
        # measured against a price the market had already moved past.
        **_lag_block(_close_lag_mins(clv)),
    }


def _lag_block(lags: list[float]) -> dict:
    if not lags:
        return {"close_lag_n": 0, "close_lag_median_min": None,
                "close_lag_p90_min": None}
    srt = sorted(lags)
    med = srt[len(srt) // 2]
    p90 = srt[min(len(srt) - 1, int(round(0.9 * (len(srt) - 1))))]
    return {"close_lag_n": len(srt), "close_lag_median_min": round(med, 1),
            "close_lag_p90_min": round(p90, 1)}


def site_block(ledger_path=LEDGER, now: datetime | None = None) -> dict:
    """Units accounting of the BADGE BETS, for the site's #/record page.

    Decision 2026-08-12: units track the EDGE-badge bets and nothing else — a
    flat 1u on the recorded side at dec_at_record, settled by `y`. Model picks
    without a badge are scored on accuracy/log loss only; they carry no stake.
    The ledger rows ARE the bet slips, so this is a pure read of them.

    Shape: {"updated", "stake": 1.0,
            "overall":  {n_open, n_awaiting, n_overdue, n_void,
                         settled:{n,w,l,staked,net,roi,avg_dec}|None,
                         by_tier:{tier:{n,w,l,net,roi}},
                         by_month:[{k,n,net,cum}], curve:[[d,cum],...],
                         stale_model:{n,n_settled,net,builds}|None},
            "leagues":  {lg: same shape + policy:{tier,gate,deficit}},
            "pending":  [{league,d,away,home,team,side,dec,ev,status,start_utc,
                          tier,model_ts,stale_model}],   # every unsettled, non-void bet
            "voids":    [{league,d,away,home,team,reason}]}  # evidence-backed voids
    n_open      the game has not started
    n_awaiting  started, result not read in yet (includes n_overdue)
    n_overdue   still no result STALE_UNSETTLED_DAYS after the game date — the
                result was never ingested; NOT a postponement
    n_void      positive evidence only (a tie on a two-way moneyline, the game gone
                from the league schedule, no final although the results source moved
                past the date): stake returned, counted nowhere, never listed
    stale_model the bets priced from a build the freshness rules would not badge
                from (see stale_model_reason). They WERE published badges, so they
                stay in the units above; this block lets the page say so, and they
                are excluded from the CLV promotion gate.
    by_tier     the settled units split by information tier (row_tier: CONFIRMED /
                PROJECTED / EARLY, "UNKNOWN" for MLB rows recorded before the tier
                stamp). The pick policy never pools tiers into one headline, so the
                page should lead with these; `settled` stays as the all-bets sum.
    pending[].tier   row_tier as well: a bet priced from a build made more than 7
                days before its game is EARLY, whatever the league default says.
    policy      what the league's badge is (BADGE_POLICY): gate "enforced" = an
                EARLY forecast never carries a badge (MLB structurally; NFL/NHL
                while sched_edges.GATE_EARLY is on — every NHL forecast is EARLY,
                so NHL stakes nothing new until a goalie input ships); tier = the
                league default; deficit = what a shown badge is priced without.
    """
    now = now or datetime.now(timezone.utc)
    ledger = load(ledger_path)
    rows = sorted(ledger.get("rows", {}).values(),
                  key=lambda r: (r.get("d") or "", r.get("ts_record") or ""))
    status = {id(r): bet_status(r, now) for r in rows}

    def _u(r):
        return round(float(r["dec_at_record"]) - 1.0, 4) if r["y"] == 1 else -1.0

    def _by_tier(settled):
        """Settled units split by the information tier of the forecast (row_tier):
        the pick policy never pools tiers into one headline."""
        acc = {}
        for r in settled:
            t = row_tier(r) or UNKNOWN_TIER
            b = acc.setdefault(t, {"n": 0, "w": 0, "l": 0, "net": 0.0})
            b["n"] += 1
            b["w" if r["y"] == 1 else "l"] += 1
            b["net"] += _u(r)
        order = [t for t in TIERS + (UNKNOWN_TIER,) if t in acc]
        return {t: {**acc[t], "net": round(acc[t]["net"], 3),
                    "roi": round(acc[t]["net"] / acc[t]["n"], 4)} for t in order}

    def _stats(rs):
        settled = [r for r in rs if r.get("y") is not None]
        st = [status[id(r)] for r in rs]
        out = {"n_open": st.count("open"),
               "n_awaiting": st.count("awaiting") + st.count("overdue"),
               "n_overdue": st.count("overdue"),
               "n_void": st.count("void"),
               "settled": None, "by_tier": _by_tier(settled),
               "by_month": [], "curve": [], "stale_model": None}
        stale = [r for r in rs if stale_model_reason(r)]
        if stale:
            ss = [r for r in stale if r.get("y") is not None]
            builds = set()
            for r in stale:
                m = _row_model_ts(r)
                if m is not None:
                    builds.add(FR.et_date(m).isoformat())
            out["stale_model"] = {"n": len(stale), "n_settled": len(ss),
                                  "net": round(float(sum(_u(r) for r in ss)), 3),
                                  "builds": sorted(builds)}
        if not settled:
            return out
        w = sum(1 for r in settled if r["y"] == 1)
        net = sum(_u(r) for r in settled)
        out["settled"] = {
            "n": len(settled), "w": w, "l": len(settled) - w,
            "staked": float(len(settled)), "net": round(net, 3),
            "roi": round(net / len(settled), 4),
            "avg_dec": round(sum(float(r["dec_at_record"]) for r in settled)
                             / len(settled), 3)}
        months, order, cum = {}, [], 0.0
        for r in settled:
            k = (r.get("d") or "?")[:7]
            if k not in months:
                months[k] = {"k": k, "n": 0, "net": 0.0}
                order.append(k)
            months[k]["n"] += 1
            months[k]["net"] += _u(r)
            cum += _u(r)
            out["curve"].append([r.get("d"), round(cum, 3)])
        c = 0.0
        for k in order:
            c += months[k]["net"]
            out["by_month"].append({"k": k, "n": months[k]["n"],
                                    "net": round(months[k]["net"], 3),
                                    "cum": round(c, 3)})
        return out

    def _mts(r):
        if r.get("league") == "mlb":
            return r.get("model_ts")
        return FR.iso(_row_model_ts(r))

    pending = []
    for r in rows:
        if status[id(r)] not in ("open", "awaiting", "overdue"):
            continue
        pending.append({"league": r.get("league"), "d": r.get("d"),
                        "away": r.get("away"), "home": r.get("home"),
                        "team": r.get("team"), "side": r.get("side"),
                        "dec": r.get("dec_at_record"), "ev": r.get("ev_at_record"),
                        "status": status[id(r)], "start_utc": r.get("start_utc"),
                        "tier": row_tier(r),
                        "model_ts": _mts(r),
                        "stale_model": bool(stale_model_reason(r))})
    leagues = {}
    for lg in LEAGUES:
        leagues[lg] = _stats([r for r in rows if r.get("league") == lg])
        leagues[lg]["policy"] = dict(BADGE_POLICY[lg])
    return {"updated": ledger.get("updated"), "stake": 1.0,
            "overall": _stats(rows), "leagues": leagues, "pending": pending,
            "voids": [{"league": r.get("league"), "d": r.get("d"),
                       "away": r.get("away"), "home": r.get("home"),
                       "team": r.get("team"), "reason": r.get("void_reason")}
                      for r in rows if status[id(r)] == "void"]}


def attach_bets(board_path=BOARD, ledger_path=LEDGER,
                now: datetime | None = None) -> bool:
    """Write the badge-bet units block into board.json (record.bets).

    The board build (predict_slate) knows nothing about the market layer, so
    the block rides on afterwards, the same way the edges themselves do. A
    missing/unreadable board is a no-op — offseason path.
    """
    try:
        payload = json.loads(Path(board_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    payload.setdefault("record", {})["bets"] = site_block(ledger_path, now)
    _write_atomic(board_path, json.dumps(payload, indent=1))
    return True


def report(ledger_path=LEDGER, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    ledger = load(ledger_path)
    rows = list(ledger.get("rows", {}).values())
    out = {"generated": _ts(now), "updated": ledger.get("updated"),
           "gate_min_graded": GATE_MIN_GRADED, "gate_min_t": GATE_MIN_T,
           "leagues": {}, "overall": _stats(rows, now)}
    for lg in LEAGUES:
        sub = [r for r in rows if r.get("league") == lg]
        out["leagues"][lg] = _stats(sub, now)
    return out


def format_report(rep: dict) -> str:
    L = [f"LIVE EDGE LEDGER — updated {rep.get('updated')} "
         f"(gate: avg CLV > 0 by >={rep.get('gate_min_t', GATE_MIN_T)} SE, with "
         f">={rep['gate_min_graded']} CLV-graded)",
         f"{'league':<8} {'rec':>5} {'clv-gr':>7} {'cens':>5} {'avgCLV':>8} "
         f"{'+/-SE':>8} {'t':>6} {'CLV+':>7} {'settled':>8} {'void':>5} "
         f"{'late':>5} {'stale':>5} "
         f"{'units':>8} {'ROI':>8} {'ROIbest':>8} {'z':>7}  gate"]
    for name in list(rep["leagues"]) + ["overall"]:
        s = rep["leagues"][name] if name in rep["leagues"] else rep["overall"]
        gate = s["gate"] + (f" ({s['gate_needs']} more)" if s["gate"] == "PENDING" else "")
        t = s.get("t_clv")
        L.append(f"{name:<8} {s['n_recorded']:>5} {s['n_clv_graded']:>7} "
                 f"{s.get('n_clv_censored', 0):>5} "
                 f"{s['avg_clv_pts']:>+8.4f} {s.get('se_clv_pts', 0.0):>8.4f} "
                 f"{(f'{t:+.2f}' if t is not None else 'inf'):>6} "
                 f"{s['pct_positive_clv']:>7.1%} {s['n_settled']:>8} "
                 f"{s.get('n_void', 0):>5} {s.get('n_stale_unsettled', 0):>5} "
                 f"{s.get('n_stale_model', 0):>5} "
                 f"{s.get('units_net', 0.0):>+7.2f}u "
                 f"{s['roi']:>+8.2%} "
                 f"{s.get('roi_best', 0.0):>+8.2%} {s['z']:>+7.2f}  {gate}")
    L.append("void = evidence-backed only; late = closed 7+ days with no result read "
             "in; stale = priced from a stale model build (kept in units, out of "
             "the gate)")
    return "\n".join(L)


def main() -> int:
    print(format_report(report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
