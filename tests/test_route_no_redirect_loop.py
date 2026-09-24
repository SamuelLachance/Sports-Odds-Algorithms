"""No page may redirect by writing location.hash while it is being routed.

route() finishes by rewriting the hash to the canonical form of the route it
just drew. A page that redirects by assigning location.hash during that draw
starts a hashchange -> route -> rewrite -> hashchange cycle: #/nhl/pos/X fired
188 hashchanges in 1.5 s (2026-09-24) and pinned the visitor's CPU. Pages that
cannot draw what was asked must render the fallback page in place instead.

Allowed: user-triggered handlers (onclick attributes, arrow-function callbacks),
the hash utility's own fallback (siteSetHash), and board()'s branch for a rail
click made outside the router (guarded by `if(!siteRouting)`).
"""
import re

from mlbwp_site.build_site import JS


def test_no_bare_hash_assignment_in_page_code():
    offenders = []
    lines = JS.split("\n")
    for n, line in enumerate(lines, 1):
        if not re.search(r"location\.hash\s*=(?!=)", line):
            continue
        if "onclick" in line or "=>" in line or "addEventListener" in line:
            continue
        ctx = "\n".join(lines[max(0, n - 4):n])
        if "function siteSetHash" in ctx or "if(!siteRouting)" in ctx:
            continue
        offenders.append(f"{n}: {line.strip()[:120]}")
    assert not offenders, "mid-route hash redirects:\n" + "\n".join(offenders)


def test_nhl_position_routes_render_the_players_page():
    body = JS.split("function posPage(key){", 1)[1].split("\n}", 1)[0]
    assert 'if(state.league==="nhl") return playersPage();' in body


def test_invalid_position_keys_render_in_place():
    mlb = JS.split("function mlbPosPage(key){", 1)[1].split("\n}", 1)[0]
    nfl = JS.split("function nflPosPage(key){", 1)[1].split("\n}", 1)[0]
    assert "return playersPage();" in mlb
    assert "return nflPlayers();" in nfl
