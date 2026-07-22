"""Injury-aware projected lineup + near-term starter for games not yet official.

When the official lineup / probable hasn't posted, the board projects from recent
reality instead of dropping to the team-only tier:

  batters  the most-frequent recent starters who are STILL on the active roster -- so
           an IL'd player (dropped from the active roster) is excluded and his recent
           replacement takes the spot. Order is irrelevant; features average over the 9.

  starter  ONLY within ~24h (today or tomorrow), where it is reliable: our data ends
           yesterday, so "rest since last start" is correct for a game today (nothing
           has pitched in between). We take the most-rested rotation REGULAR (>=2 recent
           starts, so one-off openers are excluded) who is active and NOT already one of
           the team's scheduled starters over [today..game day] (so a doubleheader
           nightcap doesn't get game 1's pitcher, and tomorrow's projection skips whoever
           starts today). Beyond ~24h we do NOT project -- a game 2+ days out will have
           pitched again in days we can't see, so the rest is wrong. The starter is the
           model's biggest factor, hence the regular-only filter and the tight horizon.

All from the MLB Stats API (rosters + recent schedule) -- market-free, read only from
the PAST, never leaks the predicted game or the backtest (which uses actual lineups).
The BULLPEN is NOT projected: its term is a season-to-date FIP aggregate over all
relievers computed identically train+serve; changing it at serve breaks that equivalence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta

from mlbwp.live import BASE, TEAM_ID_TO_RETRO, _get, et_date

LINEUP_WINDOW = 16             # days of recent lineups for the batter-frequency projection
START_WINDOW = 21              # days of recent starts for rotation-regular identification
REST_MIN = 4                   # a starter needs >=4 calendar days (3 days' rest) to be "up"
PROJECT_MAX_AHEAD = 1          # project a starter only for today (0) and tomorrow (1)


class Projector:
    def __init__(self, day0: date):
        self.day0 = day0
        self.active = self._active_rosters()          # retro -> set(mlbam ids) or None
        self.freq: dict = defaultdict(Counter)         # retro -> Counter(batter id -> starts)
        self.last_start: dict = defaultdict(dict)      # retro -> {pitcher id: (last_date, name)}
        self.start_n: dict = defaultdict(Counter)      # retro -> Counter(pitcher id -> # starts)
        self._load_recent()

    def _active_rosters(self) -> dict:
        """retro team -> set of currently-active mlbam ids (StatsAPI excludes IL players
        from the active roster). A failed fetch just leaves the team unfiltered."""
        act: dict = {}
        for tid, retro in TEAM_ID_TO_RETRO.items():
            try:
                r = _get(f"{BASE}/teams/{tid}/roster?rosterType=active")
                ids = {p["person"]["id"] for p in r.get("roster", []) if p.get("person")}
                if ids:
                    act[retro] = act.get(retro, set()) | ids
            except Exception:  # noqa: BLE001 — a missing roster just disables the IL filter for that team
                pass
        return act

    def _load_recent(self) -> None:
        start = (self.day0 - timedelta(days=max(LINEUP_WINDOW, START_WINDOW))).isoformat()
        end = (self.day0 - timedelta(days=1)).isoformat()     # strictly before today (leak-safe)
        try:
            sch = _get(f"{BASE}/schedule?sportId=1&startDate={start}&endDate={end}"
                       f"&hydrate=lineups,probablePitcher,team")
        except Exception:  # noqa: BLE001
            sch = {}
        lu_cut = (self.day0 - timedelta(days=LINEUP_WINDOW)).isoformat()
        for day in sch.get("dates", []):
            for g in day.get("games", []):
                if g.get("gameType") != "R":
                    continue
                # US-Eastern game day, to match the ET date used for the projected
                # game (predict_slate's et_date) -- a raw UTC [:10] slice would put a
                # night game a day later and skew "rest" by 1, flipping the due arm.
                gd_raw = g.get("gameDate")
                gdate = et_date(gd_raw) if gd_raw else ""
                for key, side in (("homePlayers", "home"), ("awayPlayers", "away")):
                    retro = TEAM_ID_TO_RETRO.get(g["teams"][side]["team"]["id"])
                    if not retro:
                        continue
                    if gdate >= lu_cut:
                        for p in (g.get("lineups") or {}).get(key, []):
                            self.freq[retro][p["id"]] += 1
                    pp = g["teams"][side].get("probablePitcher") or {}
                    if pp.get("id") and gdate:
                        self.start_n[retro][pp["id"]] += 1
                        prev = self.last_start[retro].get(pp["id"])
                        if prev is None or gdate > prev[0]:      # keep the most recent start
                            self.last_start[retro][pp["id"]] = (gdate, pp.get("fullName"))

    def batters(self, retro: str):
        """Projected nine: most-frequent recent starters still on the active roster."""
        active = self.active.get(retro)
        ranked = [b for b, _ in self.freq[retro].most_common()
                  if active is None or b in active]
        return ranked[:9] if len(ranked) >= 9 else None

    def starter(self, retro: str, game_date: str, exclude_ids=frozenset()):
        """Near-term (<=24-48h) projected starter: the most-rested active rotation
        REGULAR (>=2 recent starts) with >=4 days since his last start, excluding any
        pitcher already scheduled to start for this team through game day (exclude_ids).
        Returns (mlbam_id, name), or None outside the horizon / when no regular qualifies."""
        try:
            gd = date.fromisoformat(game_date)
        except ValueError:
            return None
        if not (self.day0 <= gd <= self.day0 + timedelta(days=PROJECT_MAX_AHEAD)):
            return None                                       # only today / tomorrow is reliable
        active = self.active.get(retro)
        best = None                                           # (rest, n_starts, -pid) sort key
        pick = None
        for pid, (ld, nm) in self.last_start[retro].items():
            if pid in exclude_ids:                            # already starting today/before
                continue
            if self.start_n[retro][pid] < 2:                  # rotation regular only
                continue
            if active is not None and pid not in active:      # currently rostered
                continue
            try:
                rest = (gd - date.fromisoformat(ld)).days
            except ValueError:
                continue
            if rest < REST_MIN:                               # can't start on short rest
                continue
            key = (rest, self.start_n[retro][pid], -pid)      # most rested, then most established
            if best is None or key > best:
                best, pick = key, (pid, nm)
        return pick
