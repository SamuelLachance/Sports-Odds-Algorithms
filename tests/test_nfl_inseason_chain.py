"""The first in-season NFL refresh (2026-09-24) failed in two independent ways.

Both left the site frozen on the July pre-season payload three weeks into the
season: no final scores, no 2026 records, no rating updates.

1. nfl_site_db dropped every player with fewer than 3 games in the current
   season's snap table. In weeks 1-2 that is everyone, so nfl.json shipped 0
   players and TrueSkill had nothing to rate.
2. nfl_season_serve refuses to run on a stale v7 feature column (one row per
   game in the spine), and the weekly chain had no step that regenerates it.
"""
import sys

sys.path.insert(0, "phase0")

from nfl_season_guards import ROSTER_MIN_GAMES, roster_min_games  # noqa: E402


def test_week_one_and_two_rosters_are_not_empty():
    assert roster_min_games(1) == 1
    assert roster_min_games(2) == 2


def test_the_full_season_floor_is_unchanged():
    """Mid-season behaviour is exactly the old fixed floor."""
    assert ROSTER_MIN_GAMES == 3
    for w in (3, 4, 10, 17, 18):
        assert roster_min_games(w) == 3


def test_no_snap_data_yet_still_admits_players():
    """A team missing from the snap table must not get a floor of 0 or below
    (that would admit rows with no games at all) — the floor bottoms out at 1."""
    assert roster_min_games(0) == 1


def test_site_db_uses_the_scaled_floor_not_a_literal():
    src = open("phase0/nfl_site_db.py", encoding="utf-8").read()
    assert "roster_min_games(len(weeks_seen[t]))" in src
    assert "if n_ < 3:" not in src


def test_the_weekly_chain_regenerates_v7_before_serving():
    import nfl_weekly
    steps = [s for s, _ in nfl_weekly.PAYLOAD_CHAIN]
    assert "phase0/nfl_v7_feature_gen.py" in steps
    assert steps.index("phase0/nfl_v7_feature_gen.py") < steps.index("phase0/nfl_season_serve.py")
    # and it is the generator the guard's own error message tells you to run
    from nfl_season_guards import V7_GEN
    assert V7_GEN.endswith("phase0/nfl_v7_feature_gen.py")


def test_teams_is_defined_before_the_lineup_mc_uses_it():
    """3. nfl_season_serve's Questionable-tag lineup MC iterates TEAMS, which was
    defined ~130 lines later. The MC runs only once injury reports carry
    Questionable tags, so the NameError was invisible until week 1."""
    src = open("phase0/nfl_season_serve.py", encoding="utf-8").read()
    define = src.index("TEAMS = [t for ts in DIVS.values() for t in ts]")
    first_use = src.index("for t in TEAMS:")
    assert define < first_use
    assert src.count("TEAMS = [t for ts in DIVS.values() for t in ts]") == 1


def test_questionable_tags_only_touch_each_teams_next_game():
    """4. Injury-report tags describe the upcoming game. The MC used to apply
    them to every unplayed game, shading week-14 forecasts on a week-3 tag."""
    import json
    from pathlib import Path
    src = open("phase0/nfl_season_serve.py", encoding="utf-8").read()
    assert 'if s["w"] == nxt_wk.get(s["home"]) else []' in src
    assert 'if s["w"] == nxt_wk.get(s["away"]) else []' in src
    pay = Path("site/data/nfl.json")
    if not pay.exists():
        return
    sched = json.loads(pay.read_text(encoding="utf-8"))["schedule"]
    nxt = {}
    for g in sched:
        if g.get("hs") is None:
            for t in (g["home"], g["away"]):
                nxt[t] = min(nxt.get(t, 99), g["w"])
    for g in sched:
        if g.get("hs") is not None or not g.get("ct"):
            continue
        is_next = g["w"] in (nxt.get(g["home"]), nxt.get(g["away"]))
        if not is_next:
            assert g["ct"].get("avail", 0.0) == 0.0, (
                f"week {g['w']} {g['away']}@{g['home']} carries an availability "
                f"adjustment but is not either team's next game")
