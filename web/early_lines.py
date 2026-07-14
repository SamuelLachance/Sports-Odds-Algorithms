"""Early opening-line capture across books (BetOnline, Pinnacle, DK, ...).

Principle: the first time we observe a (game, book, market) it becomes that
book's *observed open* — an append-only JSONL store under
``data/supplemental/early-lines/{league}.jsonl`` accumulates real openers over
time, so model EV can be measured against the earliest posted line (BetOnline
often hangs overnight lines well before ESPN/DK move).

Sources:
- The Odds API (https://the-odds-api.com) when ``ODDS_API_KEY`` is set —
  multi-book (betonlineag, pinnacle, bovada, draftkings, fanduel, ...).
- ESPN core odds providers as a free fallback (per-provider ``open`` objects),
  seeded from the daily build's existing enrichment payloads.

Soft-fails on any network/parse error; never blocks a slate build.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = PROJECT_ROOT / "data" / "supplemental" / "early-lines"

USER_AGENT = "Sports-Odds-Algorithms/2.0"

# The Odds API sport keys per repo league id.
ODDS_API_SPORT_KEYS: dict[str, str] = {
    "wnba": "basketball_wnba",
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "nfl": "americanfootball_nfl",
    "cfb": "americanfootball_ncaaf",
    "cbb": "basketball_ncaab",
}

# Books worth tracking, ordered by how early/sharp their openers tend to be.
TRACKED_BOOKS = (
    "betonlineag",
    "pinnacle",
    "lowvig",
    "bovada",
    "mybookieag",
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill_us",  # Caesars
    "betrivers",
)

# Defaults sized for the free Odds API tier (cost = markets x regions per
# call): retail US books (DK/FD/MGM/Caesars) already arrive free via ESPN, so
# only the early/sharp offshore books are worth credits — us2 (BetOnline,
# LowVig, Bovada, MyBookie) + eu (Pinnacle). 2 markets x 2 regions = 4 credits
# per league per snapshot.
_DEFAULT_REGIONS = "us2,eu"
_DEFAULT_MARKETS = "h2h,spreads"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _store_path(league: str) -> Path:
    return STORE_DIR / f"{league.lower()}.jsonl"


def _load_store(league: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Store rows keyed by (event_key, book)."""
    path = _store_path(league)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return out
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (str(row.get("event_key") or ""), str(row.get("book") or ""))
                if key[0] and key[1]:
                    out[key] = row  # last write wins (updates rewrite rows)
    except OSError:
        return out
    return out


def _write_store(league: str, rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = _store_path(league)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in sorted(rows.values(), key=lambda r: (str(r.get("commence") or ""), str(r.get("event_key") or ""), str(r.get("book") or ""))):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _event_key(commence_iso: str, home_name: str, away_name: str) -> str:
    day = str(commence_iso or "")[:10]
    return f"{day}|{_norm_name(home_name)}|{_norm_name(away_name)}"


def _market_snapshot_from_odds_api(bookmaker: dict[str, Any], home_name: str, away_name: str) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    hn, an = _norm_name(home_name), _norm_name(away_name)
    for market in bookmaker.get("markets") or []:
        mkey = market.get("key")
        outcomes = market.get("outcomes") or []
        if mkey == "h2h":
            for o in outcomes:
                nm = _norm_name(o.get("name"))
                if nm == hn:
                    snap["home_ml"] = o.get("price")
                elif nm == an:
                    snap["away_ml"] = o.get("price")
        elif mkey == "spreads":
            for o in outcomes:
                nm = _norm_name(o.get("name"))
                if nm == hn:
                    snap["home_spread"] = o.get("point")
                    snap["home_spread_price"] = o.get("price")
                elif nm == an:
                    snap["away_spread_price"] = o.get("price")
        elif mkey == "totals":
            for o in outcomes:
                if str(o.get("name") or "").lower() == "over":
                    snap["total"] = o.get("point")
    return snap


def snapshot_league_odds_api(league: str, *, api_key: str | None = None) -> dict[str, Any]:
    """Poll The Odds API once for a league; merge first-seen/latest into the store.

    Returns a small status dict; {"status": "no_key"} when no API key is set.
    Cost: len(markets) x len(regions) credits per call (see the-odds-api.com).
    """
    key = (api_key or os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        return {"status": "no_key", "league": league}
    sport = ODDS_API_SPORT_KEYS.get(league.lower())
    if not sport:
        return {"status": "unsupported_league", "league": league}
    regions = os.environ.get("EARLY_LINES_REGIONS", _DEFAULT_REGIONS)
    markets = os.environ.get("EARLY_LINES_MARKETS", _DEFAULT_MARKETS)
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        f"?apiKey={key}&regions={regions}&markets={markets}&oddsFormat=american"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            remaining = resp.headers.get("x-requests-remaining")
            events = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — capture layer must never raise
        return {"status": "error", "league": league, "detail": str(exc)[:200]}

    store = _load_store(league)
    now = _now_iso()
    new_books = updated = 0
    for event in events or []:
        home_name = str(event.get("home_team") or "")
        away_name = str(event.get("away_team") or "")
        commence = str(event.get("commence_time") or "")
        ekey = _event_key(commence, home_name, away_name)
        if not ekey or not home_name:
            continue
        for bm in event.get("bookmakers") or []:
            book = str(bm.get("key") or "")
            if book not in TRACKED_BOOKS:
                continue
            snap = _market_snapshot_from_odds_api(bm, home_name, away_name)
            if not snap:
                continue
            skey = (ekey, book)
            row = store.get(skey)
            if row is None:
                store[skey] = {
                    "event_key": ekey,
                    "league": league.lower(),
                    "commence": commence,
                    "home_name": home_name,
                    "away_name": away_name,
                    "book": book,
                    "source": "odds_api",
                    "first_seen": now,
                    "open": snap,
                    "last_seen": now,
                    "last": dict(snap),
                }
                new_books += 1
            else:
                row["last_seen"] = now
                row["last"] = dict(snap)
                updated += 1
    _write_store(league, store)
    return {
        "status": "ok",
        "league": league,
        "events": len(events or []),
        "new_book_lines": new_books,
        "updated": updated,
        "requests_remaining": remaining,
    }


def seed_from_espn_enrichment(
    league: str,
    *,
    event_date: str,
    home_name: str,
    away_name: str,
    providers: list[dict[str, Any]] | None,
) -> None:
    """Seed the store from ESPN per-provider open/current payloads (free path).

    ``providers`` rows: {name, home_ml, away_ml, home_spread, open_home_ml,
    open_away_ml, open_home_spread, total} — whatever fields exist are stored.
    ESPN explicit ``open`` values are recorded as the book's open even on the
    first observation (they predate our poll).
    """
    if not providers:
        return
    try:
        store = _load_store(league)
        now = _now_iso()
        ekey = _event_key(event_date, home_name, away_name)
        changed = False
        for prov in providers:
            book = "espn:" + _norm_name(prov.get("name") or "")
            if book == "espn:":
                continue
            last = {
                k: prov.get(k)
                for k in ("home_ml", "away_ml", "home_spread", "total")
                if prov.get(k) is not None
            }
            opens = {
                "home_ml": prov.get("open_home_ml"),
                "away_ml": prov.get("open_away_ml"),
                "home_spread": prov.get("open_home_spread"),
                "total": prov.get("open_total"),
            }
            opens = {k: v for k, v in opens.items() if v is not None}
            skey = (ekey, book)
            row = store.get(skey)
            if row is None:
                store[skey] = {
                    "event_key": ekey,
                    "league": league.lower(),
                    "commence": event_date,
                    "home_name": home_name,
                    "away_name": away_name,
                    "book": book,
                    "source": "espn",
                    "first_seen": now,
                    # Prefer ESPN's own open when published; else first-seen.
                    "open": opens or dict(last),
                    "open_is_espn_published": bool(opens),
                    "last_seen": now,
                    "last": last,
                }
                changed = True
            else:
                row["last_seen"] = now
                row["last"] = last or row.get("last")
                changed = True
        if changed:
            _write_store(league, store)
    except Exception:  # noqa: BLE001 — seeding must never block a slate
        return


_SNAPSHOT_DONE: set[str] = set()


def maybe_snapshot_for_build(league: str) -> None:
    """Refresh the Odds API snapshot once per process for allowed leagues.

    Gated by ``EARLY_LINES_LEAGUES`` (comma list; empty = off) and
    ``ODDS_API_KEY``. Keeps slate-build book prices fresh so line shopping can
    quote BetOnline & co without a separate cron. Soft-fails silently.
    """
    lg = league.lower()
    allowed = {
        s.strip().lower()
        for s in (os.environ.get("EARLY_LINES_LEAGUES") or "").split(",")
        if s.strip()
    }
    if lg in _SNAPSHOT_DONE or lg not in allowed:
        return
    _SNAPSHOT_DONE.add(lg)  # one attempt per build even on failure
    try:
        snapshot_league_odds_api(lg)
    except Exception:  # noqa: BLE001 — never block a slate
        return


def book_prices_for_game(
    league: str,
    *,
    event_date: str,
    home_name: str,
    away_name: str,
    max_age_hours: float = 8.0,
) -> list[dict[str, Any]]:
    """Fresh per-book CURRENT prices from the store for one game.

    Only rows whose ``last_seen`` is within ``max_age_hours`` qualify — stale
    snapshots must not masquerade as shoppable prices.
    """
    try:
        store = _load_store(league)
    except Exception:  # noqa: BLE001
        return []
    if not store:
        return []
    ekey = _event_key(event_date, home_name, away_name)
    day = str(event_date or "")[:10]
    hn, an = _norm_name(home_name), _norm_name(away_name)
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for r in store.values():
        if str(r.get("event_key") or "") != ekey:
            if str(r.get("commence") or "")[:10] != day:
                continue
            rh, ra = _norm_name(r.get("home_name")), _norm_name(r.get("away_name"))
            if not ((hn in rh or rh in hn) and (an in ra or ra in an)):
                continue
        try:
            seen = datetime.strptime(str(r.get("last_seen") or ""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (now - seen).total_seconds() > max_age_hours * 3600.0:
            continue
        last = r.get("last") or {}
        out.append(
            {
                "book": str(r.get("book") or ""),
                "home_ml": last.get("home_ml"),
                "away_ml": last.get("away_ml"),
                "home_spread": last.get("home_spread"),
                "total": last.get("total"),
                "last_seen": r.get("last_seen"),
                "source": r.get("source"),
            }
        )
    return out


def early_lines_for_game(
    league: str,
    *,
    event_date: str,
    home_name: str,
    away_name: str,
) -> dict[str, Any] | None:
    """Opening summary for a game: earliest book + per-book opens.

    Matches on (date, normalized names); returns None when nothing stored.
    """
    try:
        store = _load_store(league)
    except Exception:  # noqa: BLE001
        return None
    if not store:
        return None
    ekey = _event_key(event_date, home_name, away_name)
    rows = [r for (k, _b), r in store.items() if k == ekey]
    if not rows:
        # Fallback: same-day fuzzy match (name containment both ways).
        day = str(event_date or "")[:10]
        hn, an = _norm_name(home_name), _norm_name(away_name)
        for r in store.values():
            if str(r.get("commence") or "")[:10] != day:
                continue
            rh, ra = _norm_name(r.get("home_name")), _norm_name(r.get("away_name"))
            if (hn in rh or rh in hn) and (an in ra or ra in an):
                rows.append(r)
    if not rows:
        return None
    rows.sort(key=lambda r: str(r.get("first_seen") or ""))
    earliest = rows[0]
    books = [
        {
            "book": r.get("book"),
            "first_seen": r.get("first_seen"),
            "open": r.get("open"),
            "last": r.get("last"),
        }
        for r in rows
    ]
    return {
        "opened_by": earliest.get("book"),
        "opened_at": earliest.get("first_seen"),
        "open": earliest.get("open"),
        "books": books,
    }
