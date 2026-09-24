"""NHL team identity, alignment and standings for site/data/nhl.json.

Standard library only, so every rule here is testable without numpy.

Three things the payload lacked, each a finding against the NHL pages:

  * identity  - teams were a bare code plus a hand-kept city map in the front
                end. `team_meta` gives the full name, city, nickname, division
                and conference, from the NHL standings API when cached and from
                the static 2026-27 alignment below otherwise.
  * standings - one league-wide list sorted by points, with no divisions, no
                games played, no tiebreakers and no form. `standings` computes
                the NHL's own table from the results spine (regular season only)
                and `rank_standings` orders it the way the league does: points,
                then fewer games played, regulation wins, regulation+OT wins,
                wins, goal differential, goals for. Head-to-head (the NHL's
                fifth tiebreaker) is not applied; ties that deep are rare and
                the official clinch indicator is carried from the API instead.
  * playoff line - top three in each division plus two wild cards per
                conference (`rank_standings` marks `po`: "div" or "wc").

Verified against the API: the spine reproduces the 2025-26 final table exactly
(e.g. COL 55-16-11, 121 pts, GF 302, GA 203, RW 48, ROW 51).
"""
from __future__ import annotations

# 2026-27 alignment (unchanged since Utah joined the Central in 2024-25).
ALIGN = {
    "Atlantic": ("Eastern", ("BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR")),
    "Metropolitan": ("Eastern", ("CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH")),
    "Central": ("Western", ("CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG")),
    "Pacific": ("Western", ("ANA", "CGY", "EDM", "LAK", "SJS", "SEA", "VAN", "VGK")),
}
DIVISIONS = tuple(ALIGN)                       # display order: East then West
CONFERENCES = {"Eastern": ["Atlantic", "Metropolitan"],
               "Western": ["Central", "Pacific"]}
TEAMS = tuple(t for _, (_, ts) in ALIGN.items() for t in ts)

# (place, nickname) - fallback only; the standings API overrides it when cached.
# The place is the NHL's own short label: the two New York clubs are
# "NY Rangers" / "NY Islanders" there, so a bare "New York" never names both.
FULL = {"NYI": "New York Islanders", "NYR": "New York Rangers"}
NAMES = {
    "ANA": ("Anaheim", "Ducks"), "BOS": ("Boston", "Bruins"),
    "BUF": ("Buffalo", "Sabres"), "CGY": ("Calgary", "Flames"),
    "CAR": ("Carolina", "Hurricanes"), "CHI": ("Chicago", "Blackhawks"),
    "COL": ("Colorado", "Avalanche"), "CBJ": ("Columbus", "Blue Jackets"),
    "DAL": ("Dallas", "Stars"), "DET": ("Detroit", "Red Wings"),
    "EDM": ("Edmonton", "Oilers"), "FLA": ("Florida", "Panthers"),
    "LAK": ("Los Angeles", "Kings"), "MIN": ("Minnesota", "Wild"),
    "MTL": ("Montréal", "Canadiens"), "NSH": ("Nashville", "Predators"),
    "NJD": ("New Jersey", "Devils"), "NYI": ("NY Islanders", "Islanders"),
    "NYR": ("NY Rangers", "Rangers"), "OTT": ("Ottawa", "Senators"),
    "PHI": ("Philadelphia", "Flyers"), "PIT": ("Pittsburgh", "Penguins"),
    "SJS": ("San Jose", "Sharks"), "SEA": ("Seattle", "Kraken"),
    "STL": ("St. Louis", "Blues"), "TBL": ("Tampa Bay", "Lightning"),
    "TOR": ("Toronto", "Maple Leafs"), "UTA": ("Utah", "Mammoth"),
    "VAN": ("Vancouver", "Canucks"), "VGK": ("Vegas", "Golden Knights"),
    "WSH": ("Washington", "Capitals"), "WPG": ("Winnipeg", "Jets"),
}


def season_label(season) -> str:
    """20262027 -> '2026-27'."""
    s = str(season)
    return f"{s[:4]}-{s[6:8]}" if len(s) == 8 else s


def team_meta(api_teams: dict | None = None) -> dict:
    """code -> {name, city, nick, div, conf}. API values win when present.

    `api_teams` is one season's team block from the standings cache
    (phase0/nhl_site_fetch.py); a division or conference missing there falls
    back to the static alignment, so a partial API answer cannot leave a team
    without a division.
    """
    out = {}
    for div, (conf, codes) in ALIGN.items():
        for c in codes:
            city, nick = NAMES[c]
            out[c] = {"name": FULL.get(c, f"{city} {nick}"), "city": city, "nick": nick,
                      "div": div, "conf": conf}
    for c, a in (api_teams or {}).items():
        if c not in out:
            continue                   # a defunct or unknown code never enters
        m = out[c]
        for k in ("name", "city", "nick"):
            if a.get(k):
                m[k] = a[k]
        if a.get("div") in ALIGN:
            m["div"] = a["div"]
            m["conf"] = ALIGN[a["div"]][0]
    return out


def _wlo(r) -> str:
    return f"{r[0]}-{r[1]}-{r[2]}"


def standings(played: list[dict], teams) -> dict:
    """Standings table from completed regular-season games.

    `played` rows need home, away, hs, as, last (REG/OT/SO) and d (date);
    they are processed in (d, id) order so L10 and the streak are the most
    recent games. Every code in `teams` gets a row, zero-filled.

    GF/GA follow the NHL table: the final score, so a shootout winner is
    credited one goal (verified: spine 2025-26 COL GF 302 == API 302).
    """
    rec = {t: {"gp": 0, "w": 0, "l": 0, "otl": 0, "rw": 0, "row": 0,
               "gf": 0, "ga": 0, "_home": [0, 0, 0], "_away": [0, 0, 0],
               "_seq": []} for t in teams}
    for g in sorted(played, key=lambda x: (x["d"], x.get("id", 0))):
        if g.get("hs") is None or g.get("as") is None:
            continue
        last = g.get("last") or "REG"
        extra = last in ("OT", "SO")
        for side, opp, gf, ga in (("home", "away", g["hs"], g["as"]),
                                  ("away", "home", g["as"], g["hs"])):
            t = g[side]
            if t not in rec:
                continue
            r = rec[t]
            r["gp"] += 1
            r["gf"] += gf
            r["ga"] += ga
            split = r["_home"] if side == "home" else r["_away"]
            if gf > ga:
                r["w"] += 1
                split[0] += 1
                if last == "REG":
                    r["rw"] += 1
                if last != "SO":
                    r["row"] += 1
                r["_seq"].append("W")
            elif extra:
                r["otl"] += 1
                split[2] += 1
                r["_seq"].append("OT")
            else:
                r["l"] += 1
                split[1] += 1
                r["_seq"].append("L")
    out = {}
    for t, r in rec.items():
        seq = r.pop("_seq")
        home, away = r.pop("_home"), r.pop("_away")
        pts = 2 * r["w"] + r["otl"]
        l10 = seq[-10:]
        streak = None
        if seq:
            n = 1
            while n < len(seq) and seq[-1 - n] == seq[-1]:
                n += 1
            streak = f"{seq[-1]}{n}"
        out[t] = {**r, "pts": pts,
                  "pts_pct": round(pts / (2 * r["gp"]), 3) if r["gp"] else None,
                  "gd": r["gf"] - r["ga"],
                  "home": _wlo(home), "away": _wlo(away),
                  "l10": _wlo((l10.count("W"), l10.count("L"), l10.count("OT"))),
                  "streak": streak}
    return out


def sort_key(r: dict):
    """The NHL's in-season order: points, then FEWER games played, then RW,
    ROW, wins, goal differential, goals for."""
    return (-r["pts"], r["gp"], -r["rw"], -r["row"], -r["w"], -r["gd"], -r["gf"])


def rank_standings(table: dict, meta: dict) -> dict:
    """Add league_rank, conf_rank, div_rank, wc_rank and po to each row.

    po = "div" (top three in the division), "wc" (one of the conference's two
    wild cards) or None. wc_rank orders every non-top-three team within its
    conference (1 and 2 hold the wild cards), None for the division top three.
    Ties that survive every tiebreaker fall back to the team code, so the
    order is deterministic.

    Before the first final every team is 0-0-0: an order would only restate
    the team codes, so every rank and playoff flag is None until a game is
    played (the preseason table was ranking 32 teams at 0 points).
    """
    if not any(r["gp"] for r in table.values()):
        for r in table.values():
            r.update(league_rank=None, conf_rank=None, div_rank=None,
                     wc_rank=None, po=None)
        return table
    order = sorted(table, key=lambda t: (sort_key(table[t]), t))
    for i, t in enumerate(order):
        table[t]["league_rank"] = i + 1
    for conf in CONFERENCES:
        members = [t for t in order if meta.get(t, {}).get("conf") == conf]
        for i, t in enumerate(members):
            table[t]["conf_rank"] = i + 1
        top3 = set()
        for div in CONFERENCES[conf]:
            dm = [t for t in members if meta[t]["div"] == div]
            for i, t in enumerate(dm):
                table[t]["div_rank"] = i + 1
                if i < 3:
                    top3.add(t)
        rest = [t for t in members if t not in top3]
        for t in top3:
            table[t]["wc_rank"] = None
            table[t]["po"] = "div"
        for i, t in enumerate(rest):
            table[t]["wc_rank"] = i + 1
            table[t]["po"] = "wc" if i < 2 else None
    return table
