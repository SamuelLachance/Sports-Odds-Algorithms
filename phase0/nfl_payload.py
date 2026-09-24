"""Shared, dependency-free pieces of the NFL payload chain (site/data/nfl.json).

Everything here is standard library only, because two very different callers
use it:

  - the LOCAL model chain (nfl_site_data -> nfl_site_db -> nfl_trueskill_players
    -> nfl_lineups -> nfl_v7_feature_gen -> nfl_season_serve), which needs the
    big gitignored play tables and numpy/sklearn;
  - phase0/nfl_results_attach.py, which runs in the minimal CI job every
    refresh cycle and must never need more than the Python standard library.

Keeping the standings code in ONE place means the CI results step and the local
database step can never disagree about a team's record or division order.

Contents:
  payload_path / load / dump_atomic   staging-aware, atomic payload I/O
  TEAM_INFO / DIVS / CONF             team names and league structure
  et_to_utc / et_date                 kickoff times (US Eastern -> UTC)
  standings                           records + NFL division tiebreakers
  conference_seeding / clinch_flags   playoff picture (wild-card tiebreakers)
                                      and sufficient-condition clinch marks
  exact_contrib                       telescoped, exactly-additive breakdown
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

# ---------------------------------------------------------------- payload I/O
LIVE = "site/data/nfl.json"
STAGING_ENV = "NFL_PAYLOAD"
STAGING = "data/nfl_payload_staging.json"


def payload_path() -> str:
    """The payload file this process reads and writes.

    nfl_weekly.py points every chain step at a STAGING copy (via the
    NFL_PAYLOAD environment variable) and only swaps it onto the live file after
    the whole chain succeeded and the result validated. Run by hand, each step
    still works on the live file, exactly as before.
    """
    return os.environ.get(STAGING_ENV) or LIVE


def load(path: str | None = None) -> dict:
    with open(path or payload_path(), encoding="utf-8") as fh:
        return json.load(fh)


def dump_atomic(obj, path: str | None = None, **kw) -> None:
    """tmp + os.replace: an interrupted write never leaves a truncated file."""
    path = path or payload_path()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)
    os.replace(tmp, path)


def now_utc_iso(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------ league structure
DIVS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAM_DIV = {t: d for d, ts in DIVS.items() for t in ts}
TEAMS = [t for ts in DIVS.values() for t in ts]
CONF = {t: d.split()[0] for t, d in TEAM_DIV.items()}
# nfldata franchise codes -> the codes the payload uses
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS",
      "LAR": "LA", "AZ": "ARI"}

# code -> (city, nickname). The display code differs from the key only for the
# Rams: "LA" is the nflverse key everywhere, "LAR" is what a reader recognises
# next to "LAC".
TEAM_INFO = {
    "ARI": ("Arizona", "Cardinals"), "ATL": ("Atlanta", "Falcons"),
    "BAL": ("Baltimore", "Ravens"), "BUF": ("Buffalo", "Bills"),
    "CAR": ("Carolina", "Panthers"), "CHI": ("Chicago", "Bears"),
    "CIN": ("Cincinnati", "Bengals"), "CLE": ("Cleveland", "Browns"),
    "DAL": ("Dallas", "Cowboys"), "DEN": ("Denver", "Broncos"),
    "DET": ("Detroit", "Lions"), "GB": ("Green Bay", "Packers"),
    "HOU": ("Houston", "Texans"), "IND": ("Indianapolis", "Colts"),
    "JAX": ("Jacksonville", "Jaguars"), "KC": ("Kansas City", "Chiefs"),
    "LA": ("Los Angeles", "Rams"), "LAC": ("Los Angeles", "Chargers"),
    "LV": ("Las Vegas", "Raiders"), "MIA": ("Miami", "Dolphins"),
    "MIN": ("Minnesota", "Vikings"), "NE": ("New England", "Patriots"),
    "NO": ("New Orleans", "Saints"), "NYG": ("New York", "Giants"),
    "NYJ": ("New York", "Jets"), "PHI": ("Philadelphia", "Eagles"),
    "PIT": ("Pittsburgh", "Steelers"), "SEA": ("Seattle", "Seahawks"),
    "SF": ("San Francisco", "49ers"), "TB": ("Tampa Bay", "Buccaneers"),
    "TEN": ("Tennessee", "Titans"), "WAS": ("Washington", "Commanders"),
}
DISPLAY_ABBR = {"LA": "LAR"}


_NAME_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(name: str) -> str:
    """A player-name join key: 'Michael Penix Jr.' and 'michael penix' ->
    'michael penix'; accents, apostrophes, periods, hyphens and generational
    suffixes do not matter ('T.J. Watt' -> 'tj watt')."""
    import re
    import unicodedata
    x = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    x = re.sub(r"[^a-z]+", " ", re.sub(r"['.]", "", x))
    return " ".join(w for w in x.split() if w not in _NAME_SUFFIX)


def team_names(code: str) -> dict:
    city, nick = TEAM_INFO.get(code, (code, code))
    return {"name": f"{city} {nick}", "nick": nick, "city": city,
            "abbr": DISPLAY_ABBR.get(code, code)}


# -------------------------------------------------------------------- times
def _us_eastern_offset(d: date, hour: float) -> int:
    """UTC offset (hours) of US Eastern on a local date/time, without tzdata.

    DST runs from 02:00 on the second Sunday of March to 02:00 on the first
    Sunday of November. Only used when zoneinfo has no tz database (a bare
    Windows Python); the CI runner and most machines take the zoneinfo path.
    """
    def nth_sunday(y, m, n):
        d0 = date(y, m, 1)
        first = d0 + timedelta(days=(6 - d0.weekday()) % 7)
        return first + timedelta(days=7 * (n - 1))
    start, end = nth_sunday(d.year, 3, 2), nth_sunday(d.year, 11, 1)
    dst = (start < d < end) or (d == start and hour >= 2) or (d == end and hour < 2)
    return -4 if dst else -5


def et_to_utc(d: str, t: str | None) -> str | None:
    """'2026-09-24' + '20:15' (US Eastern) -> '2026-09-25T00:15:00Z'."""
    try:
        y, m, dd = (int(x) for x in d.split("-"))
        hh, mm = (int(x) for x in (t or "13:00").split(":")[:2])
    except (ValueError, AttributeError):
        return None
    try:
        from zoneinfo import ZoneInfo
        loc = datetime(y, m, dd, hh, mm, tzinfo=ZoneInfo("America/New_York"))
        return now_utc_iso(loc)
    except Exception:  # noqa: BLE001 - no tz database: use the fixed US rule
        off = _us_eastern_offset(date(y, m, dd), hh + mm / 60.0)
        loc = datetime(y, m, dd, hh, mm, tzinfo=timezone(timedelta(hours=off)))
        return now_utc_iso(loc)


def et_date(now_utc: str | datetime | None = None) -> date:
    """US Eastern calendar date of a UTC instant (default: now)."""
    if now_utc is None:
        now = datetime.now(timezone.utc)
    elif isinstance(now_utc, str):
        now = datetime.fromisoformat(now_utc.replace("Z", "+00:00"))
    else:
        now = now_utc
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001
        utc = now.astimezone(timezone.utc)
        off = _us_eastern_offset(utc.date(), utc.hour)
        return (utc + timedelta(hours=off)).date()


# --------------------------------------------------------------- standings
def read_games(path: str = "data/nfl_games.csv") -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def final_rows(rows: list[dict], season: int, game_type: str | None = "REG") -> list[dict]:
    """Completed games of one season as dicts with payload team codes."""
    out = []
    for r in rows:
        if str(r.get("season")) != str(season):
            continue
        if game_type is not None and r.get("game_type") != game_type:
            continue
        if r.get("home_score", "") == "" or r.get("away_score", "") == "":
            continue
        out.append({"d": r["gameday"], "w": int(r["week"]),
                    "home": FR.get(r["home_team"], r["home_team"]),
                    "away": FR.get(r["away_team"], r["away_team"]),
                    "hs": int(r["home_score"]), "as": int(r["away_score"]),
                    "type": r.get("game_type", "REG")})
    out.sort(key=lambda g: (g["d"], g.get("home")))
    return out


def latest_final_season(rows: list[dict]) -> int | None:
    seasons = [int(r["season"]) for r in rows if r.get("home_score", "") != ""]
    return max(seasons) if seasons else None


def _wlt(res: list[float]) -> tuple[int, int, int]:
    return (sum(1 for x in res if x == 1.0), sum(1 for x in res if x == 0.0),
            sum(1 for x in res if x == 0.5))


def _pct(res: list[float]) -> float:
    return sum(res) / len(res) if res else 0.0


def _rec_str(res: list[float]) -> str:
    w, l, t = _wlt(res)
    return f"{w}-{l}" + (f"-{t}" if t else "")


class _Book:
    """Per-team results of one season's regular-season finals."""

    def __init__(self, games: list[dict]):
        self.res = defaultdict(list)       # team -> [1/0.5/0 ...] chronological
        self.vs = defaultdict(list)        # team -> [(opp, result, pf, pa, home?)]
        for g in games:
            h, a = g["home"], g["away"]
            rh = 1.0 if g["hs"] > g["as"] else (0.0 if g["hs"] < g["as"] else 0.5)
            self.res[h].append(rh)
            self.res[a].append(1.0 - rh)
            self.vs[h].append((a, rh, g["hs"], g["as"], True))
            self.vs[a].append((h, 1.0 - rh, g["as"], g["hs"], False))

    def pct(self, t):
        return _pct(self.res[t])

    def pct_vs(self, t, opps):
        return _pct([r for (o, r, *_x) in self.vs[t] if o in opps])

    # ---- the NFL division tiebreak steps, in order ----
    def h2h(self, t, group):
        return self.pct_vs(t, set(group) - {t})

    def div(self, t, _group):
        return self.pct_vs(t, set(DIVS[TEAM_DIV[t]]) - {t})

    def common(self, t, group):
        common = None
        for c in group:
            opps = {o for (o, *_x) in self.vs[c]}
            common = opps if common is None else common & opps
        return self.pct_vs(t, (common or set()) - set(group))

    def conf(self, t, _group):
        return self.pct_vs(t, {o for o in CONF if CONF[o] == CONF[t]} - {t})

    def sov(self, t, _group):
        beaten = [o for (o, r, *_x) in self.vs[t] if r == 1.0]
        tot = [x for o in beaten for x in self.res[o]]
        return _pct(tot)

    def sos(self, t, _group):
        tot = [x for (o, *_x) in self.vs[t] for x in self.res[o]]
        return _pct(tot)

    def net_pts(self, t, _group):
        return float(sum(pf - pa for (_o, _r, pf, pa, _h) in self.vs[t]))


STEPS = ("h2h", "div", "common", "conf", "sov", "sos", "net_pts")


def _best_of(book: _Book, group: list[str]) -> str:
    """The club that wins a tie among `group` (all equal on win percentage).

    NFL procedure: apply the steps in order; as soon as a step separates some
    clubs from the rest, the survivors revert to step 1 among themselves. Coin
    toss is not reproducible, so the final fallback is alphabetical.
    """
    if len(group) == 1:
        return group[0]
    for step in STEPS:
        vals = {t: getattr(book, step)(t, group) for t in group}
        top = max(vals.values())
        best = [t for t in group if abs(vals[t] - top) < 1e-9]
        if len(best) == 1:
            return best[0]
        if len(best) < len(group):
            return _best_of(book, best)
    return sorted(group)[0]


def rank_division(book: _Book, teams: list[str]) -> list[str]:
    order, left = [], list(teams)
    while left:
        top = max(book.pct(t) for t in left)
        tied = [t for t in left if abs(book.pct(t) - top) < 1e-9]
        win = _best_of(book, tied)
        order.append(win)
        left.remove(win)
    return order


PYTH_EXP = 2.37


def standings(games: list[dict]) -> dict:
    """team -> record block for one season's REG finals (all 32 teams)."""
    book = _Book(games)
    out = {}
    for t in TEAMS:
        res = book.res[t]
        w, l, ti = _wlt(res)
        pf = sum(x[2] for x in book.vs[t])
        pa = sum(x[3] for x in book.vs[t])
        home = [r for (_o, r, _pf, _pa, h) in book.vs[t] if h]
        away = [r for (_o, r, _pf, _pa, h) in book.vs[t] if not h]
        divr = [r for (o, r, *_x) in book.vs[t] if TEAM_DIV.get(o) == TEAM_DIV[t]]
        confr = [r for (o, r, *_x) in book.vs[t] if CONF.get(o) == CONF[t]]
        seq = ["W" if r == 1.0 else ("L" if r == 0.0 else "T") for r in res]
        streak = "-"
        if seq:
            n = 0
            for x in reversed(seq):
                if x != seq[-1]:
                    break
                n += 1
            streak = f"{seq[-1]}{n}"
        gp = len(res)
        pyth = (pf ** PYTH_EXP / (pf ** PYTH_EXP + pa ** PYTH_EXP)) if (pf + pa) > 0 else None
        out[t] = {
            "w": w, "l": l, "t": ti, "gp": gp,
            "pct": round(_pct(res), 3),
            "pf": pf, "pa": pa, "pt_diff": pf - pa,
            "home": _rec_str(home), "away": _rec_str(away),
            "div_rec": _rec_str(divr), "conf_rec": _rec_str(confr),
            "l10": _rec_str(res[-10:]), "streak": streak,
            "pyth": round(pyth, 3) if pyth is not None else None,
            "luck": round(_pct(res) - pyth, 3) if pyth is not None and gp else None,
        }
    for d, ts in DIVS.items():
        for rank, t in enumerate(rank_division(book, ts), 1):
            out[t]["div_rank"] = rank
    return out


# ------------------------------------------------ conference seeding + clinch
# The wild-card tiebreak steps (NFL rules, "to break a tie for the Wild Card
# berth"), used both to order the four division winners (seeds 1-4) and the
# rest of the conference (5, 6, 7, then the chasers). Division mates tied in
# the group are first reduced to their best club by the division steps. The
# two "combined ranking in points scored and allowed" steps are not modelled;
# like the division order, the fallback after net points is alphabetical, a
# reproducible stand-in for the coin toss.
def _wc_h2h(book: _Book, t: str, group: list[str]) -> float:
    """Two clubs: their games against each other, when they met. Three or more:
    only a sweep counts (+1 beat every other club, -1 lost to every other)."""
    others = [o for o in group if o != t]
    vs = [(o, r) for (o, r, *_x) in book.vs[t] if o in others]
    if len(group) == 2:
        return _pct([r for _o, r in vs]) if vs else 0.5
    res = defaultdict(list)
    for o, r in vs:
        res[o].append(r)
    if set(res) != set(others):
        return 0.0
    if all(min(x) == 1.0 for x in res.values()):
        return 1.0
    if all(max(x) == 0.0 for x in res.values()):
        return -1.0
    return 0.0


def _wc_common(book: _Book, t: str, group: list[str]) -> float:
    """Win percentage in common games, only when every tied club has at least
    four of them (the rule's minimum); otherwise the step does not separate."""
    common = None
    for c in group:
        opps = {o for (o, *_x) in book.vs[c]}
        common = opps if common is None else common & opps
    common = (common or set()) - set(group)
    if min(sum(1 for (o, *_x) in book.vs[c] if o in common) for c in group) < 4:
        return 0.0
    return book.pct_vs(t, common)


def _wc_conf_net(book: _Book, t: str, _group) -> float:
    return float(sum(pf - pa for (o, _r, pf, pa, _h) in book.vs[t] if CONF.get(o) == CONF[t]))


WC_STEPS = (_wc_h2h, lambda b, t, g: b.conf(t, g), _wc_common,
            lambda b, t, g: b.sov(t, g), lambda b, t, g: b.sos(t, g),
            _wc_conf_net, lambda b, t, g: b.net_pts(t, g))


def _best_wc(book: _Book, group: list[str]) -> str:
    """The club that wins a wild-card-style tie among `group` (equal win %)."""
    if len(group) == 1:
        return group[0]
    by_div = defaultdict(list)
    for t in group:
        by_div[TEAM_DIV[t]].append(t)
    if len(by_div) < len(group):          # division mates: keep each division's best
        group = [_best_of(book, v) for v in by_div.values()]
        if len(group) == 1:
            return group[0]
    for step in WC_STEPS:
        vals = {t: step(book, t, group) for t in group}
        top = max(vals.values())
        best = [t for t in group if abs(vals[t] - top) < 1e-9]
        if len(best) == 1:
            return best[0]
        if len(best) < len(group):
            return _best_wc(book, best)
    return sorted(group)[0]


def rank_wc(book: _Book, teams: list[str]) -> list[str]:
    order, left = [], list(teams)
    while left:
        top = max(book.pct(t) for t in left)
        tied = [t for t in left if abs(book.pct(t) - top) < 1e-9]
        win = _best_wc(book, tied)
        order.append(win)
        left.remove(win)
    return order


def conference_seeding(games: list[dict]) -> dict:
    """{'AFC': [16 codes], 'NFC': [...]}: seeds 1-4 are the division winners,
    5-7 the wild cards, then the rest in wild-card order ('if the season ended
    today')."""
    book = _Book(games)
    out = {}
    for conf in ("AFC", "NFC"):
        divs = [d for d in DIVS if d.startswith(conf)]
        winners = [rank_division(book, DIVS[d])[0] for d in divs]
        rest = [t for d in divs for t in DIVS[d] if t not in winners]
        out[conf] = rank_wc(book, winners) + rank_wc(book, rest)
    return out


def remaining_games(rows: list[dict], season: int) -> dict:
    """team -> regular-season games of `season` not yet final."""
    out = defaultdict(int)
    for r in rows:
        if str(r.get("season")) != str(season) or r.get("game_type") != "REG":
            continue
        if r.get("home_score", "") != "" and r.get("away_score", "") != "":
            continue
        for side in ("home_team", "away_team"):
            out[FR.get(r[side], r[side])] += 1
    return dict(out)


def clinch_flags(games: list[dict], remaining: dict) -> dict:
    """team -> 'z' clinched the division, 'x' clinched a playoff berth, 'e'
    eliminated from the playoffs. Sufficient conditions only, so a flag is
    never wrong; it may appear a week later than the league's own, which also
    resolves tiebreakers. Win % counts a tie as half a win.

      z: the club's worst case beats every division rival's best case.
      x: at most four other conference clubs can still finish level with or
         above the club's worst case. Missing the playoffs needs five: the
         club's division winner, three non-winners ahead for the wild cards
         and at least one more division winner (a non-winner's own division
         winner finishes above it, and only two non-winners share the club's
         division).
      e: a division rival and at least three sure non-winners (each club
         beyond the first per division) finish strictly above the club's
         best case, whatever happens.
    Empty unless the whole season's schedule is known (every club the same
    number of games, finals plus remaining). Seven playoff clubs per
    conference (the format since 2020); the x rule does not hold for six.
    """
    book = _Book(games)
    tot = {t: len(book.res[t]) + remaining.get(t, 0) for t in TEAMS}
    if len(set(tot.values())) != 1 or next(iter(tot.values())) < 16:
        return {}
    pts = {t: sum(book.res[t]) for t in TEAMS}
    lo = {t: pts[t] / tot[t] for t in TEAMS}
    hi = {t: (pts[t] + remaining.get(t, 0)) / tot[t] for t in TEAMS}
    out = {}
    for t in TEAMS:
        rivals = [u for u in DIVS[TEAM_DIV[t]] if u != t]
        confs = [u for u in TEAMS if CONF[u] == CONF[t] and u != t]
        if all(lo[t] > hi[u] for u in rivals):
            out[t] = "z"
            continue
        if sum(1 for u in confs if hi[u] >= lo[t]) <= 4:
            out[t] = "x"
            continue
        above = [u for u in confs if lo[u] > hi[t]]
        if any(u in above for u in rivals):
            per = defaultdict(int)
            for u in above:
                per[TEAM_DIV[u]] += 1
            if sum(c - 1 for c in per.values()) >= 3:
                out[t] = "e"
    return out


POST_ROUNDS = ("WC", "DIV", "CON", "SB")


def postseason(rows: list[dict], season: int) -> dict:
    """team -> furthest playoff round played that season ('WC'..'SB'); 'SB won'."""
    best, champ = {}, None
    for r in rows:
        if str(r.get("season")) != str(season) or r.get("game_type") not in POST_ROUNDS:
            continue
        ix = POST_ROUNDS.index(r["game_type"])
        for side in ("home_team", "away_team"):
            t = FR.get(r[side], r[side])
            best[t] = max(best.get(t, -1), ix)
        if r["game_type"] == "SB" and r.get("home_score", "") != "":
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            if hs != as_:
                w = r["home_team"] if hs > as_ else r["away_team"]
                champ = FR.get(w, w)
    out = {t: POST_ROUNDS[i] for t, i in best.items()}
    if champ:
        out[champ] = "SB won"
    return out


def through(games: list[dict]) -> dict | None:
    """{'w','d','n'}: last week and date with a final, and how many finals."""
    if not games:
        return None
    return {"w": max(g["w"] for g in games), "d": max(g["d"] for g in games),
            "n": len(games)}


def apply_standings(payload: dict, rows: list[dict], season: int | None = None) -> int | None:
    """Write current-season records (and last season's final) into teams[].

    `season` defaults to the latest season with a completed game. Returns the
    standings season, or None when the payload has no teams yet.
    """
    teams = payload.get("teams")
    if not teams:
        return None
    season = season or latest_final_season(rows)
    if season is None:
        return None
    cur = final_rows(rows, season)
    prev_games = final_rows(rows, season - 1)
    st, st_prev = standings(cur), standings(prev_games)
    post_prev = postseason(rows, season - 1)
    # the playoff picture: only once a game is played (before that every club
    # is 0-0 and the order would be the alphabet)
    seed = {}
    if cur:
        for order in conference_seeding(cur).values():
            seed.update({t: i for i, t in enumerate(order, 1)})
    # the clinch rules assume the seven-club format (2020 on)
    flags = (clinch_flags(cur, remaining_games(rows, season))
             if cur and int(season) >= 2020 else {})
    for t, tm in teams.items():
        if t not in st:
            continue
        tm.update(st[t])
        for k, v in (("conf_seed", seed.get(t)), ("clinch", flags.get(t))):
            if v is None:
                tm.pop(k, None)
            else:
                tm[k] = v
        if prev_games:
            p = st_prev[t]
            tm["prev"] = {"season": season - 1, "w": p["w"], "l": p["l"], "t": p["t"],
                          "pct": p["pct"], "pf": p["pf"], "pa": p["pa"],
                          "pt_diff": p["pt_diff"], "div_rank": p["div_rank"],
                          "post": post_prev.get(t)}
    payload["standings_season"] = int(season)
    payload["standings_through"] = through(cur)
    return int(season)


# ------------------------------------------------ exact contribution breakdown
# The order the deltas are taken in. Team strength first (Elo, QB, units),
# then roster/player terms, then circumstance, then the availability MC.
CX_ORDER = ("elo", "qb", "units", "roster", "ts", "hfa", "sched", "luck", "abs", "avail")
CX_MAX_RESID = 0.05          # logit; beyond this the inputs are inconsistent


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def exact_contrib(ph: float | None, ct: dict | None, intercept: float) -> dict | None:
    """Telescoped percentage-point contributions that add up to ph exactly.

    `ct` is the served slope-scaled attribution: each group's logit push
    (coefficient * feature, plus the availability MC's logit shift) multiplied
    by ph*(1-ph)*100. Dividing the scale back out recovers the logit pushes;
    with the blend's intercept they sum to logit(ph) up to ct's 0.1pp rounding.
    Walking them through the sigmoid one group at a time, starting from the
    intercept alone (`base`: the model's intercept, i.e. the average home edge,
    which the locked model applies at neutral sites too), makes each bar the
    exact change in probability it causes, so

        50 + base + sum(groups)  ==  100 * ph   (to rounding)

    Built only from the frozen ph and ct, so a played game's breakdown is a
    deterministic function of its pre-game numbers and carries no hindsight.
    Returns None when the inputs do not reconcile (a residual beyond
    CX_MAX_RESID logit), rather than papering over an inconsistency.
    """
    if ph is None or not ct:
        return None
    p = min(max(float(ph), 1e-6), 1 - 1e-6)
    scale = p * (1 - p) * 100.0
    lg = {k: float(ct.get(k) or 0.0) / scale for k in CX_ORDER}
    resid = math.log(p / (1 - p)) - intercept - sum(lg.values())
    if abs(resid) > CX_MAX_RESID:
        return None
    tot = sum(abs(v) for v in lg.values())
    if tot > 0:                          # ct rounding error, spread by size
        lg = {k: v + resid * abs(v) / tot for k, v in lg.items()}
        z = intercept
    else:
        z = intercept + resid
    prev = _sig(z)
    raw = {"base": (prev - 0.5) * 100}
    for k in CX_ORDER:
        z += lg[k]
        cur = _sig(z)
        raw[k] = (cur - prev) * 100
        prev = cur
    return _round_to_total(raw, round((p - 0.5) * 100, 1))


def _round_to_total(raw: dict, total: float) -> dict:
    """Round to 0.1pp so the rounded parts still sum to `total` exactly.

    Largest-remainder rounding in tenths: independent rounding of eleven terms
    can drift 0.2-0.3pp off the headline number, which is exactly the
    'the bars do not add up' complaint this breakdown exists to answer.
    """
    tenths = {k: v * 10 for k, v in raw.items()}
    fl = {k: math.floor(v) for k, v in tenths.items()}
    need = int(round(total * 10)) - sum(fl.values())
    order = sorted(tenths, key=lambda k: -(tenths[k] - fl[k]))
    if need >= 0:
        for k in order[:need]:
            fl[k] += 1
    else:
        for k in list(reversed(order))[:-need]:
            fl[k] -= 1
    return {k: round(fl[k] / 10, 1) + 0.0 for k in raw}
