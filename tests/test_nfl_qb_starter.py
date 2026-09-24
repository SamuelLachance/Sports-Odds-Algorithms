"""NFL QB identity: the starter, not the post-game primary passer (ledger row 77).

nflverse's home_qb_id / away_qb_id is the passer with the most attempts - known
only after the game. The shipped feature now keys on the first passer who has a
QB line that week; research scripts keep the listed passer by default so their
ledger rows keep reproducing.
"""
import sys

sys.path.insert(0, "phase0")
from nfl_qb_elo import apply_starters  # noqa: E402


def _g(h="NE", a="BUF", hq="garoppolo", aq="buf1", s=2014, w=17):
    return {"home": h, "away": a, "home_qb": hq, "away_qb": aq, "season": s, "week": w}


def test_the_first_passer_replaces_the_listed_primary_passer():
    games, raw = [_g()], [{"game_id": "2014_17_BUF_NE"}]
    qbw = {("brady", 2014, 17): (1.0, 17), ("garoppolo", 2014, 17): (1.0, 20)}
    n = apply_starters(games, raw, qbw, {("2014_17_BUF_NE", "NE"): "brady"})
    assert n == 1 and games[0]["home_qb"] == "brady"


def test_a_non_qb_trick_pass_is_not_a_start():
    """A receiver's option pass has no QB weekly line - keep the listed QB."""
    games, raw = [_g()], [{"game_id": "g"}]
    n = apply_starters(games, raw, {}, {("g", "NE"): "wr_trick"})
    assert n == 0 and games[0]["home_qb"] == "garoppolo"


def test_no_play_data_changes_nothing():
    games, raw = [_g()], [{"game_id": "g"}]
    assert apply_starters(games, raw, {}, {}) == 0


def test_the_serve_opts_in_and_research_does_not():
    serve = open("phase0/nfl_season_serve.py", encoding="utf-8").read()
    assert 'os.environ.setdefault("NFL_QB_STARTER", "1")' in serve
    assert serve.index('setdefault("NFL_QB_STARTER"') < serve.index("coord_src = open(")
    lib = open("phase0/nfl_qb_elo.py", encoding="utf-8").read()
    assert 'os.environ.get("NFL_QB_STARTER") == "1"' in lib     # default off
