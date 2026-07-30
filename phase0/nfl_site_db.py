"""Extend site/data/nfl.json with the full database: teams (2025 standings, records,
divisions, rosters) + players (ratings, tiers, 2025 stat lines) — the MLB-parity
payload for Standings / Teams / player pages. Light build: reads CSVs only.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

DIVS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAM_DIV = {t: d for d, ts in DIVS.items() for t in ts}
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS"}

payload = json.load(open("site/data/nfl.json"))
power_by = {p["code"]: p for p in payload["power"]}

# standings/display season = the LATEST season in games.csv that has completed
# scores (today "2025"; flips to "2026" the day the first 2026 final lands)
SEASON = str(max(int(r["season"]) for r in csv.DictReader(open("data/nfl_games.csv"))
                 if r["home_score"] != ""))

def sfile(tmpl, fallback="2025"):
    """Per-season data file for the display season, falling back to the newest
    committed file (today SEASON == "2025" -> identical to the old hardcodes)."""
    p = tmpl.format(SEASON)
    return p if os.path.exists(p) else tmpl.format(fallback)

# ---- standings from games.csv (display season, REG only) ----
rec = defaultdict(lambda: {"w": 0, "l": 0, "t": 0, "pf": 0, "pa": 0,
                           "hw": 0, "hl": 0, "aw": 0, "al": 0, "seq": []})
games25 = []
for r in csv.DictReader(open("data/nfl_games.csv")):
    if r["season"] != SEASON or r["home_score"] == "" or r["game_type"] != "REG":
        continue
    h, a = FR.get(r["home_team"], r["home_team"]), FR.get(r["away_team"], r["away_team"])
    hs, as_ = int(r["home_score"]), int(r["away_score"])
    games25.append((r["gameday"], h, a, hs, as_))
games25.sort()
for (d, h, a, hs, as_) in games25:
    rec[h]["pf"] += hs; rec[h]["pa"] += as_
    rec[a]["pf"] += as_; rec[a]["pa"] += hs
    if hs > as_:
        rec[h]["w"] += 1; rec[h]["hw"] += 1; rec[h]["seq"].append("W")
        rec[a]["l"] += 1; rec[a]["al"] += 1; rec[a]["seq"].append("L")
    elif hs < as_:
        rec[a]["w"] += 1; rec[a]["aw"] += 1; rec[a]["seq"].append("W")
        rec[h]["l"] += 1; rec[h]["hl"] += 1; rec[h]["seq"].append("L")
    else:
        for t in (h, a):
            rec[t]["t"] += 1; rec[t]["seq"].append("T")

def streak(seq):
    if not seq:
        return "-"
    c, n = seq[-1], 0
    for x in reversed(seq):
        if x == c:
            n += 1
        else:
            break
    return f"{c}{n}"

def l10(seq):
    last = seq[-10:]
    return f"{last.count('W')}-{last.count('L')}"

# ---- player data ----
ratings = {}
for r in csv.DictReader(open(sfile("data/nfl_player_ratings_{}.csv"), encoding="utf-8")):
    ratings[r["gsis_id"]] = {"r": float(r["rating_0_100"]), "tier": r["tier"],
                             "n_eff": float(r["n_eff"]), "bucket": r["bucket"]}
board = {}
for r in csv.DictReader(open(sfile("data/nfl_player_board_{}.csv"), encoding="utf-8")):
    board[r["gsis_id"]] = {"z": float(r["z"]), "cons": float(r["conservative_z"]),
                           "n": int(float(r["n_duels"]))}
pfr2gsis, names = {}, {}
for r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    if r.get("gsis_id"):
        names[r["gsis_id"]] = r.get("display_name") or r["gsis_id"]
        if r.get("pfr_id"):
            pfr2gsis[r["pfr_id"]] = r["gsis_id"]

def f2(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

# display-season stat sums (offense + defense), regular season only
stat = defaultdict(lambda: defaultdict(float))
for r in csv.DictReader(open(sfile("data/nfl_player_stats_{}.csv"), encoding="utf-8")):
    if r["season_type"] != "REG":
        continue   # was a no-op `pass` bug: playoff rows leaked into season totals
    pid = r["player_id"]
    for k_dst, k_src in (("pass_yds", "passing_yards"), ("pass_td", "passing_tds"),
                         ("ints", "passing_interceptions"), ("att", "attempts"),
                         ("rush_yds", "rushing_yards"), ("rush_td", "rushing_tds"),
                         ("car", "carries"), ("rec", "receptions"), ("tgt", "targets"),
                         ("rec_yds", "receiving_yards"), ("rec_td", "receiving_tds"),
                         ("tak", "def_tackles_solo"), ("sk", "def_sacks"),
                         ("dint", "def_interceptions"), ("pd", "def_pass_defended")):
        stat[pid][k_dst] += f2(r.get(k_src))
    stat[pid]["pos"] = r.get("position") or stat[pid].get("pos") or ""

# rosters: display-season actives from snap tables, with snap share
roster = defaultdict(lambda: defaultdict(lambda: [0.0, 0, ""]))
for r in csv.DictReader(open(sfile("data/snap_{}.csv"))):
    t = FR.get(r["team"], r["team"])
    g_ = pfr2gsis.get(r["pfr_player_id"])
    if not g_:
        continue
    e = roster[t][g_]
    e[0] += f2(r["offense_pct"]) + f2(r["defense_pct"])
    e[1] += 1
    e[2] = r["position"]

players = {}
teams = {}
for d, ts in DIVS.items():
    ranked = sorted(ts, key=lambda t: -(rec[t]["w"] + 0.5 * rec[t]["t"]))
    for rank, t in enumerate(ranked, 1):
        rc = rec[t]
        gp = rc["w"] + rc["l"] + rc["t"]
        pw = power_by.get(t, {})
        roster_ids = []
        for g_, (sh, n_, pos) in sorted(roster[t].items(), key=lambda kv: -kv[1][0]):
            if n_ < 3:
                continue
            st = stat.get(g_, {})
            players[g_] = {
                "id": g_, "name": names.get(g_, g_), "team": t,
                "pos": pos or st.get("pos", ""),
                "rating": ratings.get(g_), "board": board.get(g_),
                "snap_share": round(sh / max(n_, 1), 2),
                "stats": {k: int(v) for k, v in st.items()
                          if k != "pos" and v} if st else {},
            }
            roster_ids.append(g_)
        teams[t] = {
            "code": t, "name": pw.get("name", t), "abbr": t, "div": d,
            "div_rank": rank, "w": rc["w"], "l": rc["l"], "t": rc["t"],
            "pct": round((rc["w"] + 0.5 * rc["t"]) / max(gp, 1), 3),
            "pf": rc["pf"], "pa": rc["pa"], "pt_diff": rc["pf"] - rc["pa"],
            "home": f'{rc["hw"]}-{rc["hl"]}', "away": f'{rc["aw"]}-{rc["al"]}',
            "l10": l10(rc["seq"]), "streak": streak(rc["seq"]),
            "elo": pw.get("elo"), "rank": pw.get("rank"),
            "off_pass": pw.get("off_pass"), "off_run": pw.get("off_run"),
            "def_pass": pw.get("def_pass"), "def_run": pw.get("def_run"),
            "roster": roster_ids[:60],
        }

payload["teams"] = teams
payload["players"] = players
payload["divisions"] = list(DIVS)
json.dump(payload, open("site/data/nfl.json", "w"))
print(f"nfl.json extended: {len(teams)} teams, {len(players)} players")
top = sorted(teams.values(), key=lambda t: (t["div"], t["div_rank"]))[:4]
print("sample:", [(t["code"], f'{t["w"]}-{t["l"]}', t["div"]) for t in top])
