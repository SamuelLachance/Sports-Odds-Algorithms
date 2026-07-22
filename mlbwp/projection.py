"""Injury-aware projected lineup for games whose official lineup hasn't posted yet.

When the official lineup hasn't dropped, the board projects the nine batters from
recent reality instead of falling back to the team-only tier:

  the most-frequent recent starters who are STILL on the active roster -- so a player
  on the IL (dropped from the active roster) is excluded and his recent replacement,
  who has been starting, takes the spot. Order is irrelevant; the lineup features
  average over the nine.

All from the MLB Stats API (active rosters + recent lineups) -- market-free, and read
only from the PAST (games strictly before today), so it never leaks the game being
predicted or touches the backtest (which always uses actual lineups).

Not projected here, on purpose:
  * the STARTER -- a reliable projection needs a full rotation-cycle model (a game 2+
    days out will have pitched again in the days our data can't see, so "rest since
    last start" is wrong), and the starter is the model's biggest factor, so a wrong
    guess flips the pick. We use the official probable (MLB sets it ~2 days ahead) and
    leave still-TBD starters at the pre-lineup tier.
  * the BULLPEN -- the model's bullpen term is a season-to-date FIP aggregate over all
    relievers, computed identically in training and serving; changing it at serve time
    (e.g. to drop IL'd arms) would break that train==serve equivalence for no tested gain.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta

from mlbwp.live import BASE, TEAM_ID_TO_RETRO, _get

LINEUP_WINDOW = 16             # days of recent lineups for the batter-frequency projection


class Projector:
    def __init__(self, day0: date):
        self.day0 = day0
        self.active = self._active_rosters()          # retro -> set(mlbam ids) or None
        self.freq: dict = defaultdict(Counter)         # retro -> Counter(batter id -> starts)
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
        start = (self.day0 - timedelta(days=LINEUP_WINDOW)).isoformat()
        end = (self.day0 - timedelta(days=1)).isoformat()     # strictly before today (leak-safe)
        try:
            sch = _get(f"{BASE}/schedule?sportId=1&startDate={start}&endDate={end}"
                       f"&hydrate=lineups,team")
        except Exception:  # noqa: BLE001
            sch = {}
        for day in sch.get("dates", []):
            for g in day.get("games", []):
                if g.get("gameType") != "R":
                    continue
                for key, side in (("homePlayers", "home"), ("awayPlayers", "away")):
                    retro = TEAM_ID_TO_RETRO.get(g["teams"][side]["team"]["id"])
                    if not retro:
                        continue
                    for p in (g.get("lineups") or {}).get(key, []):
                        self.freq[retro][p["id"]] += 1

    def batters(self, retro: str):
        """Projected nine: most-frequent recent starters still on the active roster."""
        active = self.active.get(retro)
        ranked = [b for b, _ in self.freq[retro].most_common()
                  if active is None or b in active]
        return ranked[:9] if len(ranked) >= 9 else None
