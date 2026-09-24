"""The NHL season schedule the serve predicts: every unplayed game, not ten days.

Standard library only. The serve used to predict only data/nhl_upcoming.json,
a 10-day window (55 of the season's games), so there was no season view and no
projection. Unplayed games now come from two sources, merged by game id:

  * data/nhl_upcoming.json - the fresh near-term slate (nhl_update.py, every
    run): it wins on date, puck drop and live state;
  * data/nhl_schedule_{season}.json - the whole season (nhl_site_fetch.py),
    for everything beyond the window.

A game already in the results spine (any type) is played and never re-served
as unplayed; a cancelled game is dropped; a postponed one is kept and flagged.
Games beyond the near window are the same model, from the same current team
ratings - EARLY-tier leans like every NHL forecast (documents/pick_policy.md),
flagged `near: 0` so the pages can say they sit outside the 10-day slate.
"""
from __future__ import annotations

from datetime import date, timedelta

NEAR_DAYS = 10


def _season_of(gid) -> int | None:
    s = str(gid)
    if len(s) != 10:
        return None
    y = int(s[:4])
    return y * 10000 + y + 1


def unplayed_rows(played_ids: set, upcoming: list[dict], season_games: list[dict],
                  season: int) -> list[dict]:
    """Merged unplayed games of `season`, sorted by (date, id).

    Rows: id, d, home, away, playoff, t (puck drop, UTC ISO or None), state,
    ppd (postponed flag), src ("slate" or "season").
    """
    out = {}
    for g in season_games or []:
        gid = g.get("id")
        if gid is None or gid in played_ids or _season_of(gid) != season:
            continue
        if g.get("type") not in (2, 3) or g.get("sched") == "CNCL":
            continue
        if not g.get("home") or not g.get("away") or not g.get("d"):
            continue
        out[gid] = {"id": gid, "d": g["d"], "home": g["home"], "away": g["away"],
                    "playoff": 1 if g.get("type") == 3 else 0,
                    "t": g.get("t") or None, "state": g.get("state"),
                    "ppd": 1 if g.get("sched") == "PPD" else 0, "src": "season"}
    for u in upcoming or []:
        gid = u.get("id")
        if gid is None or gid in played_ids or _season_of(gid) != season:
            continue
        prev = out.get(gid, {})
        out[gid] = {"id": gid, "d": u["d"], "home": u["home"], "away": u["away"],
                    "playoff": u.get("playoff", 0) or 0,
                    "t": u.get("t") or prev.get("t"),
                    "state": u.get("state") or prev.get("state"),
                    "ppd": prev.get("ppd", 0), "src": "slate"}
    return sorted(out.values(), key=lambda r: (r["d"], r["id"]))


def playoff_finals(raw: dict, season: int, start_by_id: dict, teams) -> list[dict]:
    """This season's completed PLAYOFF games from the spine rows (`raw`,
    game_id -> csv row), in the shape unplayed_rows gives plus the result.

    The model's walk (load_games) is regular-season only and unplayed_rows
    drops every spine id as played, so without these a playoff game vanished
    from the schedule once final - its pre-game ledger entry orphaned, never
    graded. Only games between two `teams` the model rates are kept.
    """
    out = []
    for gid, r in sorted(raw.items()):
        if r.get("type") != "3" or int(r["season"]) != season:
            continue
        if r["home"] not in teams or r["away"] not in teams:
            continue
        out.append({"id": gid, "d": r["date"], "home": r["home"], "away": r["away"],
                    "playoff": 1, "t": start_by_id.get(gid),
                    "hs": int(r["home_goals"]), "as": int(r["away_goals"]),
                    "last": r.get("last_period") or "REG"})
    return out


def ledger_path(row: dict, ledger: dict) -> bool:
    """Does this schedule row go through the pre-game ledger (nhl_hp_freeze)?

    Played games, the near-term slate, and every row that ALREADY has an
    entry. Each game sits inside the 10-day window for many serves before it
    is played, so its last pre-game number is always recorded; far-horizon
    rows without an entry stay out (they would churn ~1,300 timestamps per
    run). A row WITH an entry is always refreshed, so payload hp == ledger hp
    for every ledger key - also for a game rescheduled beyond the window.
    """
    return row.get("hs") is not None or bool(row.get("near")) or str(row["id"]) in ledger


def is_near(d: str, today: date, days: int = NEAR_DAYS) -> bool:
    """Inside the near-term window the upcoming slate covers (today..+days)."""
    return d <= (today + timedelta(days=days)).isoformat()


def spine_rest(games: list[dict], cap: int = 5, default: int = 3) -> list[tuple]:
    """(rest_home, rest_away, b2b_home, b2b_away) per spine game, in order.

    The same rule as nhl_features_eval.build_features (rest = days since the
    team's previous game capped at `cap`, `default` for a team's first game,
    back-to-back = previous game at most one day earlier) - so a played row
    shows the model inputs its forecast used.
    """
    last, out = {}, []
    for g in games:
        d = date.fromisoformat(g["date"])
        vals = []
        for t in (g["home"], g["away"]):
            lg = last.get(t)
            if lg is None:
                vals.append((default, 0))
            else:
                gap = (d - lg).days
                vals.append((min(gap, cap), 1 if gap <= 1 else 0))
            last[t] = d
        out.append((vals[0][0], vals[1][0], vals[0][1], vals[1][1]))
    return out
