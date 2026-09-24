"""Serving-time guards for the NHL model. Stdlib only, so they are testable
without numpy (the CI test environment has none).

Both exist because of bugs that only appear at a season boundary, when the
spine ends in the previous season and every served game is in the next one.
"""
from __future__ import annotations

from datetime import date


def needs_boundary_regression(last_walked_season: int, serve_season: int) -> bool:
    """True when the served season is past the last season the walk reached.

    The model regresses every rating toward the mean between seasons, but the
    walk only applies that when it crosses into a game of the new season. Before
    opening night the spine holds no such game, so final_states() returned April's
    ratings unregressed and every 2026-27 probability was served ~30% too
    confident (mean |dp| 2.7pp, max 6.6pp) — enough to fire +23% EV badges.
    """
    return last_walked_season < serve_season


def schedule_rest(last_played: dict[str, str], upcoming: list[dict],
                  cap: int = 5, default: int = 3) -> list[tuple[int, int, float, float]]:
    """(rest_home, rest_away, b2b_home, b2b_away) for each upcoming game.

    Rest is measured from each team's previous game, PLAYED OR SCHEDULED. The
    serve measured it from played games only, so a team with two games in the
    window got the same rest for both and every back-to-back inside the window
    was served as b2b=0 (all 14 of them on 2026-09-24).

    `upcoming` must be the served games; they are processed in (date, id) order
    and the result is returned in the ORIGINAL order.
    """
    order = sorted(range(len(upcoming)),
                   key=lambda i: (upcoming[i]["d"], upcoming[i].get("id", 0)))
    last = dict(last_played)
    out: list = [None] * len(upcoming)
    for i in order:
        u = upcoming[i]
        d = date.fromisoformat(u["d"])
        vals = []
        for team in (u["home"], u["away"]):
            prev = last.get(team)
            if prev:
                gap = (d - date.fromisoformat(prev)).days
                vals.append((min(gap, cap), 1.0 if gap <= 1 else 0.0))
            else:
                vals.append((default, 0.0))
        out[i] = (vals[0][0], vals[1][0], vals[0][1], vals[1][1])
        last[u["home"]] = u["d"]
        last[u["away"]] = u["d"]
    return out
