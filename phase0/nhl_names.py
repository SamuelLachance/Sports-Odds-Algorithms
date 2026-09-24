"""Current NHL rosters -> data/nhl_player_names.json (32 requests).

The map is keyed by NHL player id and is BOTH:

  * the current-roster truth the site serves from - every player on a club's
    `roster/{TEAM}/current` answer carries `on_roster: true`, his club, sweater
    number, birth date, shoots/catches, height, weight, nationality, headshot
    URL and `seen` (the UTC time a roster answer last listed him); and
  * a cumulative id -> name/position map that research code reads
    (nhl_rapm2 seeds positions from it for ~1,300 historical skaters).

The old version only ever MERGED: a player who left a club kept that club
forever, which is how departed players were still shown as team members. Now a
player whose last club answered WITHOUT him is flipped to `on_roster: false`
(the entry stays, for the research readers). A club whose request failed leaves
its players exactly as they were - a network error never removes anyone.

The team list is the 32 current clubs (nhl_site_teams.TEAMS), not "home teams
seen in the spine", so a new or renamed club is fetched from day one.

The club loop is bounded: REQ_TIMEOUT seconds per request, an optional
wall-clock `deadline` (time.monotonic()) from the caller, and it stops after
MAX_CONSEC_FAIL failures in a row (an API that hangs instead of refusing would
otherwise cost 32 timeouts). Clubs not reached keep their previous entries.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhl_site_teams import TEAMS  # noqa: E402

HEADERS = {"User-Agent": "glassbox-nhl/1.0", "Accept-Encoding": "gzip"}
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{}/current"
OUT = "data/nhl_player_names.json"
REQ_TIMEOUT = 15          # seconds per request
MAX_CONSEC_FAIL = 3       # a club loop gives up after this many failures in a row


def get(url, timeout=REQ_TIMEOUT):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _txt(v):
    """The API's localised strings: {'default': 'Toronto', 'fr': ...}."""
    if isinstance(v, dict):
        return v.get("default")
    return v


def roster_entry(p: dict, team: str, now: str) -> dict:
    """One roster answer -> the map's entry for that player."""
    return {
        "name": f"{_txt(p.get('firstName')) or ''} {_txt(p.get('lastName')) or ''}".strip(),
        "pos": p.get("positionCode", ""),
        "team": team,
        "on_roster": True,
        "num": p.get("sweaterNumber"),
        "born": p.get("birthDate"),
        "shoots": p.get("shootsCatches"),
        "ht": p.get("heightInInches"),
        "wt": p.get("weightInPounds"),
        "nat": p.get("birthCountry"),
        "img": p.get("headshot"),
        "seen": now,
    }


def merge_rosters(names: dict, fetched: dict, now: str) -> dict:
    """Pure merge of this run's roster answers into the cumulative map.

    fetched: team -> list of roster player dicts, ONLY for clubs that answered.
      * a listed player gets a fresh entry (club, bio, on_roster true);
      * a player whose recorded club answered without him -> on_roster false
        (departed: traded to a club whose request failed, waived, unsigned,
        retired). He is re-listed the moment any answering club lists him;
      * a player whose club did not answer is left untouched.
    Entries are never deleted (research readers rely on the cumulative map).
    """
    out = {k: dict(v) for k, v in names.items()}
    listed = set()
    for team, plist in fetched.items():
        for p in plist:
            pid = str(p["id"])
            listed.add(pid)
            out[pid] = roster_entry(p, team, now)
    answered = set(fetched)
    for pid, v in out.items():
        if pid in listed:
            continue
        if v.get("team") in answered:
            v["on_roster"] = False
    return out


def fetched_at(names: dict) -> str | None:
    """Time of the newest roster answer recorded in the map."""
    seen = [v.get("seen") for v in names.values() if v.get("seen")]
    return max(seen) if seen else None


def load(path: str = OUT) -> dict:
    try:
        return json.load(open(path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def age_hours(iso: str | None, now: datetime | None = None) -> float:
    if not iso:
        return float("inf")
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    now = now or datetime.now(timezone.utc)
    return (now - t).total_seconds() / 3600.0


def over(deadline: float | None) -> bool:
    """True once the caller's wall-clock budget (a time.monotonic() value) is spent."""
    return deadline is not None and time.monotonic() >= deadline


def refresh(max_age_h: float = 0.0, teams=TEAMS, get_fn=get, path: str = OUT,
            sleep: float = 0.2, deadline: float | None = None,
            max_consec_fail: int = MAX_CONSEC_FAIL) -> tuple[int, int]:
    """Fetch the current rosters unless the map is younger than max_age_h.

    Returns (clubs answered, clubs attempted); (0, 0) when skipped as fresh.
    Writes nothing when every request fails. Stops early (clubs not reached
    keep their previous entries) once `deadline` passes or after
    `max_consec_fail` failed requests in a row.
    """
    names = load(path)
    if max_age_h and age_hours(fetched_at(names)) < max_age_h:
        print(f"[nhl_names] rosters fresh ({fetched_at(names)}), not refetched")
        return 0, 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fetched, fails = {}, 0
    for i, t in enumerate(teams):
        if over(deadline):
            print(f"  time budget spent after {i}/{len(teams)} clubs; "
                  f"the rest keep their previous rosters")
            break
        try:
            d = get_fn(ROSTER_URL.format(t))
            fails = 0
            plist = [p for grp in ("forwards", "defensemen", "goalies")
                     for p in (d.get(grp) or []) if isinstance(p, dict) and p.get("id")]
            if plist:
                fetched[t] = plist
            else:
                print(f"  {t}: empty roster answer, club left untouched")
        except Exception as ex:  # noqa: BLE001
            fails += 1
            print(f"  {t}: {ex}")
            if fails >= max_consec_fail:
                print(f"  {fails} failed requests in a row - API unreachable, stopping; "
                      f"clubs not reached keep their previous rosters")
                break
        if sleep:
            time.sleep(sleep)
    if not fetched:
        print(f"[nhl_names] all {len(teams)} roster requests failed - "
              f"keeping existing map ({len(names):,} players), nothing written")
        return 0, len(teams)
    out = merge_rosters(names, fetched, now)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    os.replace(tmp, path)   # atomic: no truncated JSON on interruption
    n_on = sum(1 for v in out.values() if v.get("on_roster"))
    print(f"wrote {path}: {len(out):,} players, {n_on:,} on a current roster "
          f"(from {len(fetched)}/{len(teams)} clubs)")
    return len(fetched), len(teams)


def main() -> int:
    ok, n = refresh(0.0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
