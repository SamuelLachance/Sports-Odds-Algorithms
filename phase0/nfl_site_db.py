"""Extend the NFL payload with the full database: teams (records, divisions,
rosters, injury report) + players (ratings, roster status, bio, season-labelled
stat lines, career) — the MLB-parity payload for Standings / Teams / player
pages. Light build: reads CSVs only.

Who exists, and on which team, comes from the CURRENT weekly roster
(data/roster_<season>.csv, each team's latest week), never from a snap table:

  - a snap table says who PLAYED, not who is on the team. Building the player
    set from it made rosters last season's in the preseason (MIA listed Tua,
    ATL's QB1 sat on another roster) and, in season, deleted anyone below the
    games floor: an injured star, a returning starter, every rookie who had not
    played yet (74 projected starters had no player entry on 2026-09-24).
  - roster statuses are kept and labelled: ACT is the active roster, RES/EXE
    the reserve lists (injured reserve, PUP, ...), DEV the practice squad.
    Nobody vanishes because he is hurt; he is flagged.

Snap tables are keyed by pfr id. They join to players through the weekly
roster's pfr ids first (it knows this year's rookies and corrects stale ids in
nfl_players.csv), then nfl_players.csv, then an unambiguous (name, team) key
from the roster; the build prints how many snap players stay unmapped.

The games floor (nfl_season_guards.roster_min_games) now only ORDERS the active
roster (players with real current-season usage first) and gates the lineup
step's usage flip exactly as before; it never decides whether a player exists.

Seasons are explicit, never hard-coded: standings_season is the latest season
with a completed game, stats_season the season players[].stats belong to, and
stats_prev_season the full season before it (players[].stats_prev). Career
lines sum every regular season on file (offense 1999+, defense 1999+).
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, "phase0")
import nfl_payload as NP  # noqa: E402
from nfl_season_guards import (INJ_FR, current_injury_status,  # noqa: E402
                               roster_min_games)

PAYLOAD = NP.payload_path()
payload = NP.load(PAYLOAD)
power_by = {p["code"]: p for p in payload["power"]}
GAMES = NP.read_games()
FR = NP.FR

# standings/display season = the LATEST season in games.csv that has completed
# scores ("2025" until the first 2026 final lands, then "2026")
SEASON = NP.latest_final_season(GAMES)
SERVE_SEASON = int(payload.get("season") or SEASON)


def stats_file(season: int) -> str | None:
    p = f"data/nfl_player_stats_{season}.csv"
    return p if os.path.exists(p) else None


HIST_OFF, HIST_DEF = "data/nfl_player_stats.csv", "data/nfl_player_stats_def.csv"
# the season players[].stats describe: the standings season when its weekly
# stats exist, else the newest season that has them
STATS_SEASON = SEASON if (stats_file(SEASON) or SEASON <= 2024) else SEASON - 1
PREV_SEASON = STATS_SEASON - 1


def sfile(tmpl, fallback="2025"):
    """Per-season data file for the display season, falling back to the newest
    committed file."""
    p = tmpl.format(SEASON)
    return p if os.path.exists(p) else tmpl.format(fallback)


def f2(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def num(v):
    """int when whole (tackles, yards), else one decimal (half sacks)."""
    return int(v) if float(v).is_integer() else round(v, 1)


# ---- position family: which stat line a player gets ----
FAM = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
       "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
       "LT": "OL", "RT": "OL", "LG": "OL", "RG": "OL",
       "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
       "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
       "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "SAF": "DB", "DB": "DB",
       "K": "K", "PK": "K", "P": "P", "LS": "LS"}
DEF_KEYS = ["g", "tak", "ast", "tfl", "sk", "qbh", "dint", "pd", "ff"]
FAM_KEYS = {
    "QB": ["g", "att", "cmp", "pass_yds", "pass_td", "ints", "car", "rush_yds", "rush_td"],
    "RB": ["g", "car", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td"],
    "WR": ["g", "tgt", "rec", "rec_yds", "rec_td"],
    "TE": ["g", "tgt", "rec", "rec_yds", "rec_td"],
    "OL": ["g"], "LS": ["g"],
    "DL": DEF_KEYS, "LB": DEF_KEYS, "DB": DEF_KEYS,
    "K": ["g", "fgm", "fga", "fg_long", "xpm", "xpa"],
    "P": ["g", "punts", "punt_yds", "punt_in20", "punt_long"],
}
# payload key -> source columns (weekly files first, the 1999-2024 archive's
# older names after); "max" keys take the season maximum instead of the sum
SRC = {
    "att": ("attempts",), "cmp": ("completions",), "pass_yds": ("passing_yards",),
    "pass_td": ("passing_tds",), "ints": ("passing_interceptions", "interceptions"),
    "car": ("carries",), "rush_yds": ("rushing_yards",), "rush_td": ("rushing_tds",),
    "tgt": ("targets",), "rec": ("receptions",), "rec_yds": ("receiving_yards",),
    "rec_td": ("receiving_tds",),
    "tak": ("def_tackles_solo",), "ast": ("def_tackle_assists",),
    "tfl": ("def_tackles_for_loss",), "sk": ("def_sacks",), "qbh": ("def_qb_hits",),
    "dint": ("def_interceptions",), "pd": ("def_pass_defended",),
    "ff": ("def_fumbles_forced",),
    "fgm": ("fg_made",), "fga": ("fg_att",), "fg_long": ("fg_long",),
    "xpm": ("pat_made",), "xpa": ("pat_att",),
    "punts": ("pt_att",), "punt_yds": ("pt_yards",), "punt_in20": ("pt_inside_20",),
    "punt_long": ("pt_long",),
}
MAX_KEYS = {"fg_long", "punt_long"}
# kicking/punting columns exist only in the 2025+ weekly files, so a career sum
# of them would be silently partial: careers carry offense and defense only
CAREER_KEYS = [k for k in SRC if k not in {"fgm", "fga", "fg_long", "xpm", "xpa",
                                            "punts", "punt_yds", "punt_in20", "punt_long"}]
# linemen and specialists have no box-score career to sum (a lineman's one
# recovered-fumble tackle is not a career line)
CAREER_FAMS = ("QB", "RB", "WR", "TE", "DL", "LB", "DB")


def _acc(dst: dict, r: dict, keys) -> None:
    for k in keys:
        for col in SRC[k]:
            if col in r:
                v = f2(r.get(col))
                if k in MAX_KEYS:
                    dst[k] = max(dst.get(k, 0.0), v)
                else:
                    dst[k] = dst.get(k, 0.0) + v
                break


def season_sums(path: str) -> dict:
    out = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("season_type") != "REG":
                continue       # playoff rows must not leak into season totals
            _acc(out[r["player_id"]], r, SRC)
    return out


# the 1999-2024 archive, read once: (pid, season) -> sums
hist = defaultdict(dict)
for path in (HIST_OFF, HIST_DEF):
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("season_type") != "REG":
                continue
            _acc(hist[(r["player_id"], int(r["season"]))], r, CAREER_KEYS)


def stats_for(season: int) -> dict:
    p = stats_file(season)
    if p:
        return season_sums(p)
    out = {}
    for (pid, s), v in hist.items():
        if s == season:
            out[pid] = v
    return out


stat_cur = stats_for(STATS_SEASON)
stat_prev = stats_for(PREV_SEASON)
career = defaultdict(lambda: {"_s": set()})
for (pid, s), v in hist.items():
    if s > STATS_SEASON or stats_file(s):
        continue                 # a weekly file supersedes the archive season
    c = career[pid]
    c["_s"].add(s)
    for k, x in v.items():
        c[k] = c.get(k, 0.0) + x
for s in range(2025, STATS_SEASON + 1):
    p = stats_file(s)
    if not p:
        continue
    blk = stat_cur if s == STATS_SEASON else (stat_prev if s == PREV_SEASON else season_sums(p))
    for pid, v in blk.items():
        c = career[pid]
        c["_s"].add(s)
        for k in CAREER_KEYS:
            if k in v:
                c[k] = c.get(k, 0.0) + v[k]


def block(sums: dict | None, fam: str, g: int | None = None,
          keys=None) -> dict | None:
    """Position-relevant keys always present (a real 0 is a stat); any other
    non-zero line kept (a receiver's passing TD). None when there is no line."""
    if not sums and not g:
        return None
    want = FAM_KEYS.get(fam, ["g"])
    out = {}
    if g is not None:
        out["g"] = g
    for k in (keys or SRC):
        v = (sums or {}).get(k, 0.0)
        if v or k in want:
            out[k] = num(v)
    return out


# ---- player identity maps ----
pfr2gsis, names, bio = {}, {}, {}
for r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    g_ = r.get("gsis_id")
    if not g_:
        continue
    names[g_] = r.get("display_name") or g_
    if r.get("pfr_id"):
        pfr2gsis[r["pfr_id"]] = g_
    bio[g_] = r

# The weekly roster carries pfr ids nfl_players.csv does not have yet (this
# year's rookies), and a (name, team) key for the few it has no pfr id for
# either. Both must be in place BEFORE the snap tables are read: the snap
# tables are keyed by pfr id, and an unmapped row silently lost the player's
# usage (2026-09-24: 187 snap players, 42 every-down starters, most of them
# rookies, showed 0 games and a 0% snap share while playing every snap).
ROSTER = f"data/roster_{SERVE_SEASON}.csv"
roster_rows = list(csv.DictReader(open(ROSTER, encoding="utf-8"))) \
    if os.path.exists(ROSTER) else []
name_key = NP.name_key
_by_name_team = defaultdict(set)
for r in roster_rows:
    g_ = r.get("gsis_id")
    if not g_:
        continue
    if r.get("pfr_id"):
        # the current roster wins over nfl_players.csv, which still hands some
        # rookies' pfr ids to someone else (WoodPe00 -> a retired 'Pete Woods',
        # JacaGa00 -> a placeholder id, not Gabe Jacas' gsis id)
        pfr2gsis[r["pfr_id"]] = g_
    t = INJ_FR.get(r["team"], FR.get(r["team"], r["team"]))
    _by_name_team[(name_key(r.get("full_name", "")), t)].add(g_)
# only an unambiguous (name, team) pair identifies a player
by_name_team = {k: next(iter(v)) for k, v in _by_name_team.items() if len(v) == 1}
unmapped_snaps = {}                 # season -> {pfr id: (name, team)} still unresolved


# ---- snap tables: stats season (share + games) and the season before ----
def snap_usage(season: int):
    """gsis -> [off/def pct sum, games, st pct sum, position, last team]."""
    use = {}
    weeks = defaultdict(set)
    p = f"data/snap_{season}.csv"
    if not os.path.exists(p):
        return use, weeks
    miss = unmapped_snaps.setdefault(season, {})
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("game_type", "REG") != "REG":
            continue            # regular season only, like the stat lines
        t = FR.get(r["team"], r["team"])
        weeks[t].add(r.get("week"))
        g_ = pfr2gsis.get(r["pfr_player_id"]) or by_name_team.get((name_key(r["player"]), t))
        if not g_:
            miss[r["pfr_player_id"]] = (r["player"], t)
            continue
        e = use.setdefault(g_, [0.0, 0, 0.0, "", t])
        e[0] += f2(r["offense_pct"]) + f2(r["defense_pct"])
        e[1] += 1
        e[2] += f2(r.get("st_pct"))
        e[3] = r["position"] or e[3]
        e[4] = t
    return use, weeks


snap_cur, weeks_seen = snap_usage(STATS_SEASON)
snap_prev, _ = snap_usage(PREV_SEASON)
_miss = unmapped_snaps.get(STATS_SEASON, {})
print(f"snap map {STATS_SEASON}: {len(snap_cur)} players mapped, {len(_miss)} unmapped"
      + (f" (e.g. {sorted(v[0] + ' ' + v[1] for v in _miss.values())[:5]})" if _miss else ""))

# ---- ratings + per-play board (replaced by nfl_trueskill_players downstream) ----
ratings = {}
for r in csv.DictReader(open(sfile("data/nfl_player_ratings_{}.csv"), encoding="utf-8")):
    ratings[r["gsis_id"]] = {"r": float(r["rating_0_100"]), "tier": r["tier"] or None,
                             "n_eff": float(r["n_eff"]), "bucket": r["bucket"]}
board = {}
for r in csv.DictReader(open(sfile("data/nfl_player_board_{}.csv"), encoding="utf-8")):
    # 2 dp: the page shows z to 2 dp; the 3rd was 2,000 players x 2 dead digits
    board[r["gsis_id"]] = {"z": round(float(r["z"]), 2),
                           "cons": round(float(r["conservative_z"]), 2),
                           "n": int(float(r["n_duels"]))}

# ---- the current weekly roster: membership, status, bio ----
# players[].status: the weekly-roster status, with the reserve list split by its
# nflverse description code. Labels ship once, in payload.status_labels.
STATUS_LABELS = {"ACT": "Active", "IR": "Injured reserve",
                 "IR-R": "Injured reserve (designated to return)",
                 "PUP": "Physically unable to perform (PUP)",
                 "RES": "Reserve list", "EXE": "Exempt list", "DEV": "Practice squad"}
RES_CODE = {"R01": "IR", "R48": "IR-R", "R04": "PUP"}
RESERVE = ("IR", "IR-R", "PUP", "RES", "EXE")
KEEP = ("ACT", "RES", "EXE", "DEV")
latest_wk = {}
for r in roster_rows:
    t = INJ_FR.get(r["team"], FR.get(r["team"], r["team"]))
    try:
        latest_wk[t] = max(latest_wk.get(t, -1), int(r.get("week") or 0))
    except ValueError:
        pass
current = {}                                   # gsis -> roster row (team's latest week)
for r in roster_rows:
    t = INJ_FR.get(r["team"], FR.get(r["team"], r["team"]))
    try:
        w = int(r.get("week") or 0)
    except ValueError:
        continue
    g_ = r.get("gsis_id")
    if w != latest_wk.get(t) or not g_ or r.get("status") not in KEEP:
        continue
    prev_r = current.get(g_)
    # one row per player; an active row beats a reserve/practice-squad duplicate
    if prev_r is None or (r["status"] == "ACT" and prev_r["status"] != "ACT"):
        current[g_] = dict(r, team=t)


def status_of(r: dict) -> str:
    st = r.get("status", "")
    if st == "RES":
        return RES_CODE.get(r.get("status_description_abbr", ""), "RES")
    return st


TODAY = date.today()


def age_of(bd: str | None):
    try:
        y, m, d = (int(x) for x in (bd or "").split("-"))
        return TODAY.year - y - ((TODAY.month, TODAY.day) < (m, d))
    except ValueError:
        return None


def ival(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


# ---- current injury report (each team's latest report, same rule as lineups) ----
inj_rows = []
INJ = f"data/inj_{SERVE_SEASON}.csv"
if os.path.exists(INJ):
    inj_rows = list(csv.DictReader(open(INJ, encoding="utf-8")))
inj_status = current_injury_status(inj_rows) if inj_rows else {}
inj_week, inj_detail = {}, {}
for r in inj_rows:
    t = INJ_FR.get(r.get("team", ""), r.get("team", ""))
    try:
        w = int(r.get("week") or 0)
    except ValueError:
        continue
    inj_week[t] = max(inj_week.get(t, -1), w)
for r in inj_rows:
    g_ = r.get("gsis_id")
    t = INJ_FR.get(r.get("team", ""), r.get("team", ""))
    if not g_ or g_ not in inj_status:
        continue
    try:
        w = int(r.get("week") or 0)
    except ValueError:
        continue
    what = r.get("report_primary_injury") or r.get("practice_primary_injury") or ""
    if what and (g_ not in inj_detail or w >= inj_detail[g_][0]):
        inj_detail[g_] = (w, what)

# ---- players: everyone on a current roster ----
players = {}
for g_, r in current.items():
    use = snap_cur.get(g_)
    usep = snap_prev.get(g_)
    pos = (use[3] if use and use[3] else "") or r.get("depth_chart_position") \
        or r.get("position") or ""
    fam = FAM.get(pos) or FAM.get(r.get("position", ""), "")
    b = bio.get(g_, {})
    dy = ival(b.get("draft_year"))
    cur_blk = block(stat_cur.get(g_), fam, use[1] if use else 0)
    prev_blk = block(stat_prev.get(g_), fam, usep[1] if usep else None)
    rt = ratings.get(g_)
    c = career.get(g_) if fam in CAREER_FAMS else None
    if r["status"] == "DEV" and not (rt or cur_blk or prev_blk or c):
        continue        # practice-squad names with no rating and no NFL line
    p = {
        "id": g_, "name": names.get(g_) or r.get("full_name") or g_,
        "team": r["team"], "pos": pos, "fam": fam,
        "status": status_of(r),
        "num": ival(r.get("jersey_number")),
        "age": age_of(r.get("birth_date") or b.get("birth_date")),
        "ht": ival(r.get("height")), "wt": ival(r.get("weight")),
        "college": r.get("college") or b.get("college_name") or None,
        "exp": ival(r.get("years_exp")),
        "draft": ({"year": dy, "round": ival(b.get("draft_round")),
                   "pick": ival(b.get("draft_pick")),
                   "team": FR.get(b.get("draft_team", ""), b.get("draft_team") or None)}
                  if dy else None),
        "rating": rt, "board": board.get(g_),
        "snap_share": round(use[0] / max(use[1], 1), 2) if use else 0.0,
        "snap_g": use[1] if use else 0,
        # special-teams share: the only usage a kicker, punter or snapper has
        "st_share": (round(use[2] / max(use[1], 1), 2) or None) if use else None,
        "snap_prev": round(usep[0] / max(usep[1], 1), 2) if usep else None,
        "stats": cur_blk or {},
        "stats_prev": prev_blk,
    }
    if c and c["_s"]:
        cb = block(c, fam, None, CAREER_KEYS) or {}
        cb.update({"first": min(c["_s"]), "last": max(c["_s"]), "seasons": len(c["_s"])})
        p["career"] = cb
    if g_ in inj_status:
        p["inj"] = {"status": inj_status[g_], "week": inj_week.get(r["team"]),
                    "injury": inj_detail.get(g_, (0, ""))[1] or None}
    players[g_] = {k: v for k, v in p.items() if v is not None or k in ("rating", "board")}

# ---- teams ----
SEV = {"Out": 0, "Doubtful": 1, "Questionable": 2}
teams = payload.get("teams") or {}
for d, ts in NP.DIVS.items():
    for t in ts:
        pw = power_by.get(t, {})
        tm = teams.setdefault(t, {})
        tm.update({"code": t, **NP.team_names(t), "div": d,
                   "elo": pw.get("elo"), "rank": pw.get("rank"),
                   "off_pass": pw.get("off_pass"), "off_run": pw.get("off_run"),
                   "def_pass": pw.get("def_pass"), "def_run": pw.get("def_run")})
        mine = [p for p in players.values() if p["team"] == t]
        # early-season floor: a fixed 3 empties every roster in weeks 1-2. It
        # ORDERS the active roster (current-season regulars first); it no
        # longer decides who exists.
        floor = roster_min_games(len(weeks_seen[t]))
        act = [p for p in mine if p["status"] == "ACT"]
        regular = sorted((p for p in act if p["snap_g"] >= floor),
                         key=lambda p: -p["snap_share"])
        rest = sorted((p for p in act if p["snap_g"] < floor),
                      key=lambda p: (-(p["rating"] or {}).get("r", 0.0), p["name"]))
        tm["roster"] = [p["id"] for p in regular + rest]
        tm["reserve"] = sorted((p["id"] for p in mine if p["status"] in RESERVE),
                               key=lambda i: players[i]["name"])
        tm["practice"] = sorted((p["id"] for p in mine if p["status"] == "DEV"),
                                key=lambda i: players[i]["name"])
        tm["snap_weeks"] = len(weeks_seen[t])
        tm["inj_week"] = inj_week.get(t)
        tm["inj"] = sorted(({"id": p["id"], "name": p["name"], "pos": p["pos"],
                             "status": p["inj"]["status"], "injury": p["inj"]["injury"]}
                            for p in mine if p.get("inj")),
                           key=lambda e: (SEV.get(e["status"], 9), e["name"]))

payload["teams"] = teams
if SEASON is not None:
    NP.apply_standings(payload, GAMES, SEASON)
payload["players"] = players
payload["divisions"] = list(NP.DIVS)
payload["stats_season"] = STATS_SEASON
payload["stats_prev_season"] = PREV_SEASON
payload["roster_week"] = max(latest_wk.values()) if latest_wk else None
payload["player_board_season"] = int(sfile("data/nfl_player_board_{}.csv")[-8:-4])
payload["status_labels"] = STATUS_LABELS
NP.dump_atomic(payload, PAYLOAD, separators=(",", ":"))
n_status = defaultdict(int)
for p in players.values():
    n_status[p["status"]] += 1
print(f"nfl.json extended: {len(teams)} teams, {len(players)} players "
      f"({dict(n_status)}); standings {SEASON}, stats {STATS_SEASON}/{PREV_SEASON}")
top = sorted(teams.values(), key=lambda t: (t["div"], t["div_rank"]))[:4]
print("sample:", [(t["code"], f'{t["w"]}-{t["l"]}', t["div"]) for t in top])
