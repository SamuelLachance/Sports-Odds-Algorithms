"""The two NFL in-season guards, extracted so they can be TESTED.

Both fire for the first time on the day the 2026 season starts — the worst
possible moment to discover either is wrong — and both were previously inline
expressions in `nfl_season_serve.py` with no coverage.

1. `apply_2026_boundary` — the serve applies a manual season-boundary transform
   to every feature walk (Elo regression, EPA decay, QB new-season, ...). Those
   transforms assume the walk ended on a 2025 game. Once a 2026 final lands in
   the spine, the walk's OWN in-loop season-change trigger fires instead, and
   applying the manual transform on top would DOUBLE-REGRESS every rating. The
   failure is silent: no crash, just quietly wrong probabilities all season.

2. `check_v7_npy` — column 7 of the feature matrix is loaded from a frozen
   .npy whose writer only exists in `nfl_v7_feature_gen.py`. It must have one
   row per game in the spine. The instant a 2026 final is appended the lengths
   diverge, and the serve must stop with an ACTIONABLE message rather than an
   opaque shape error — the dry run showed a bare assert here is useless at 6am
   on a game day.

Kept dependency-free (no numpy) so the guards are testable in isolation.
"""
from __future__ import annotations

V7_GEN = "python phase0/nfl_v7_feature_gen.py"


def apply_2026_boundary(last_walked_season: int, serve_season: int = 2026) -> bool:
    """True while the manual boundary transform is still the right thing to do.

    The walk has not yet crossed into the serve season, so the transform has not
    already been applied by the walk itself.
    """
    return last_walked_season < serve_season


def v7_npy_error(n_npy: int, n_games: int, gen_cmd: str = V7_GEN) -> str | None:
    """None when the frozen feature column matches the spine, else the message.

    Both directions are named explicitly because they mean different things and
    have different fixes: SHORTER means new finals landed and the column must be
    regenerated; LONGER means the spine went backwards, which is a data problem,
    not a regeneration problem.
    """
    if n_npy == n_games:
        return None
    cause = ("(new finals landed in data/nfl_games.csv)" if n_npy < n_games
             else "(npy has MORE rows than the spine — was data/nfl_games.csv "
                  "rolled back?)")
    return (f"data/nfl_v7_feature.npy is stale: {n_npy} rows vs {n_games} games "
            f"in the spine {cause}. Regenerate it: {gen_cmd}")


ROSTER_MIN_GAMES = 3


def roster_min_games(weeks_seen: int, full: int = ROSTER_MIN_GAMES) -> int:
    """Games a player must appear in to make the displayed roster.

    The floor exists to keep one-off call-ups and garbage-time snaps off the
    roster, and 3 is right once a season is under way. It is wrong in weeks 1-2:
    nobody has 3 games yet, so a fixed floor of 3 empties every roster and
    with it the whole player layer of the site (it did, on 2026-09-24: 0 players
    in nfl.json, which left TrueSkill nothing to rate). The floor therefore
    scales with how many weeks the team actually has in the snap table, never
    exceeding `full`, never below 1.
    """
    return max(1, min(full, weeks_seen))


INJ_FR = {"JAC": "JAX", "WSH": "WAS", "AZ": "ARI", "LAR": "LA"}


def current_injury_status(rows, next_week=None):
    """gsis_id -> report_status from each team's LATEST injury report only.

    An NFL injury report lists only the players who are injured THAT week; a
    player absent from it is healthy. Reading every report ever filed (as both
    nfl_lineups and nfl_season_serve did) keeps a player ruled out forever
    once he has appeared as Out or Doubtful once: on 2026-09-24 it benched
    Tua Tagovailoa for ATL@GB on a week-1 tag although he practised in full on
    the week-3 report, made Cooper Rush QB1, and moved GB from 65% to 77%.

    `next_week` (team -> week of the team's next unplayed game) caps the report
    week so a stale future row cannot leak in; without it, each team's newest
    report is used. Rows with no status (full participants) still define the
    report week but carry no status, so they never exclude anyone.
    """
    latest = {}
    for r in rows:
        t = INJ_FR.get(r.get("team", ""), r.get("team", ""))
        try:
            w = int(r.get("week") or 0)
        except ValueError:
            continue
        cap = (next_week or {}).get(t)
        if cap is not None and w > cap:
            continue
        if w > latest.get(t, -1):
            latest[t] = w
    # Mid-week the newest report carries practice participation but no game
    # designation yet (those land Friday). A player who did not practise and
    # has no designation keeps the status he had on his previous report, so a
    # week-2 Out is not read as healthy on a Wednesday.
    prev_status = {}                       # (gsis, week) -> status
    for r in rows:
        try:
            w = int(r.get("week") or 0)
        except ValueError:
            continue
        st = (r.get("report_status") or "").strip()
        if st and r.get("gsis_id"):
            prev_status[(r["gsis_id"], w)] = st
    out = {}
    for r in rows:
        t = INJ_FR.get(r.get("team", ""), r.get("team", ""))
        try:
            w = int(r.get("week") or 0)
        except ValueError:
            continue
        gid = r.get("gsis_id")
        if w != latest.get(t) or not gid:
            continue
        st = (r.get("report_status") or "").strip()
        if not st and "Did Not Participate" in (r.get("practice_status") or ""):
            earlier = [pw for (pg, pw) in prev_status if pg == gid and pw < w]
            if earlier:
                st = prev_status[(gid, max(earlier))]
        if st:
            out[gid] = st
    return out


def active_roster(rows):
    """team -> gsis ids on the team's LATEST weekly roster with status ACT.

    The weekly roster file also carries injured reserve (RES), practice squad
    (DEV), cut, retired and exempt rows, and every past week. Treating all of
    them as available put IR players into projected lineups.
    """
    latest = {}
    for r in rows:
        t = INJ_FR.get(r.get("team", ""), r.get("team", ""))
        try:
            w = int(r.get("week") or 0)
        except ValueError:
            continue
        latest[t] = max(latest.get(t, -1), w)
    out = {}
    for r in rows:
        t = INJ_FR.get(r.get("team", ""), r.get("team", ""))
        try:
            w = int(r.get("week") or 0)
        except ValueError:
            continue
        if w == latest.get(t) and r.get("status") == "ACT" and r.get("gsis_id"):
            out.setdefault(t, set()).add(r["gsis_id"])
    return out

