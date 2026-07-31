"""NHL context layer — travel/geography, residual schedule density, goalie workload.

Three channels the NHL ledger has NEVER tested (10 TEST rows cover Elo, Glicko-2,
rest, back-to-back, xG, goalie level, goalie delta, player ratings, scratches;
DEV kills cover PP/PK and dead-game). The shipped blend already contains
rest_diff and the two B2B flags, so everything here must be tested as a RESIDUAL
increment on top of the frozen blend — that is the whole point:

  1. TRAVEL / GEOGRAPHY
     Distance actually flown since the team's previous game (haversine from the
     PREVIOUS GAME's arena, not from home), signed timezone shift split into
     eastward (phase-advance, physiologically harder) and westward legs, venue
     altitude relative to what the visitor is acclimated to, and road-trip
     length / days away from home.
  2. SCHEDULE DENSITY BEYOND B2B
     3-games-in-4-nights, 4-in-6, and games-in-the-last-7-days differential.
     B2B is already shipped, so this is specifically the residual density signal.
  3. GOALIE WORKLOAD
     Consecutive starts for tonight's starter, starts in the last 7 days, and
     shots faced in the last 7 days. The goalie IDENTITY channel is dead (ledger
     rows 6 and 9 — GSAx level and GSAx delta both null); WORKLOAD/fatigue is a
     different mechanism and has never been tested.

--build  : walk the game spine once, emit data/nhl_context.csv (atomic) with the
           per-game raw quantities, plus coverage and sanity checks.
--screen : DEV-ONLY screen (one-look protocol). base_z = frozen shipped-blend
           coefs applied to DEV features exactly as nhl_goalie2_eval / nhl_st_eval
           build it. DEV eval era 2011-12..2017-18 is split temporally: first half
           FIT, second half SCORE, plus a random 5-fold OOF over full DEV as
           robustness. Paired bootstrap CI on per-game LL deltas.
           HARD ASSERT: no season >= 20182019 is ever scored or printed.

None of these features has a tunable hyperparameter fit on data (the only
constants — the 4-day "went home" reset, the 7-day windows, the caps — are fixed
a priori and never searched), so there is no inner-tuning leak surface.

Interesting bar (set before looking): delta >= +0.0005 with bootstrap CI lower
bound > 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict, deque
from datetime import date

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import (DEV_END, DEV_WARM_BEFORE, TEST_START,  # noqa: E402
                              llv, load_games, run_elo)
from nhl_features_eval import build_features  # noqa: E402

GAMES = "data/nhl_games.csv"
GOALIE_XG = "data/nhl_goalie_xg.csv"
SHOTS = "data/nhl_shots.csv"
CTX_CSV = "data/nhl_context.csv"
REPORT = "data/nhl_context_report.json"

HOME_RESET_DAYS = 4      # gap >= this => assume the team flew home in between
DAYS_AWAY_CAP = 21
CONSEC_CAP = 15
# games-spine code -> MoneyPuck goalie-file code (MoneyPuck backfills ARI over
# the Phoenix years; the games spine keeps PHX distinct)
GOALIE_TEAM = {"PHX": "ARI"}

# ---------------------------------------------------------------- arena table
# (lat, lon, standard UTC offset in hours, altitude metres, city label)
# Standard offsets: the NHL season is mostly outside DST, and ARI/PHX do not
# observe DST at all, so standard offsets are the right circadian anchor.
ARENAS = {
    "ANA": (33.8078, -117.8765, -8, 48, "Anaheim"),
    "ARI": (33.5319, -112.2611, -7, 340, "Glendale AZ"),
    "PHX": (33.5319, -112.2611, -7, 340, "Glendale AZ"),
    "ATL": (33.7573, -84.3963, -5, 320, "Atlanta"),
    "BOS": (42.3662, -71.0621, -5, 6, "Boston"),
    "BUF": (42.8750, -78.8765, -5, 178, "Buffalo"),
    "CGY": (51.0374, -114.0519, -7, 1045, "Calgary"),
    "CAR": (35.8033, -78.7219, -5, 122, "Raleigh"),
    "CHI": (41.8807, -87.6742, -6, 182, "Chicago"),
    "COL": (39.7487, -105.0077, -7, 1609, "Denver"),
    "CBJ": (39.9694, -83.0060, -5, 232, "Columbus"),
    "DAL": (32.7905, -96.8103, -6, 130, "Dallas"),
    "DET": (42.3411, -83.0553, -5, 187, "Detroit"),
    "EDM": (53.5469, -113.4973, -7, 668, "Edmonton"),
    "FLA": (26.1584, -80.3255, -5, 3, "Sunrise FL"),
    "LAK": (34.0430, -118.2673, -8, 88, "Los Angeles"),
    "MIN": (44.9447, -93.1011, -6, 218, "St Paul"),
    "MTL": (45.4961, -73.5693, -5, 40, "Montreal"),
    "NSH": (36.1592, -86.7785, -6, 143, "Nashville"),
    "NJD": (40.7336, -74.1711, -5, 5, "Newark"),
    "NYI": (40.7135, -73.7256, -5, 20, "Long Island"),
    "NYR": (40.7505, -73.9934, -5, 10, "New York"),
    "OTT": (45.2969, -75.9271, -5, 91, "Ottawa"),
    "PHI": (39.9012, -75.1720, -5, 3, "Philadelphia"),
    "PIT": (40.4395, -79.9896, -5, 250, "Pittsburgh"),
    "SJS": (37.3328, -121.9012, -8, 25, "San Jose"),
    "SEA": (47.6221, -122.3540, -8, 30, "Seattle"),
    "STL": (38.6268, -90.2027, -6, 141, "St Louis"),
    "TBL": (27.9427, -82.4518, -5, 5, "Tampa"),
    "TOR": (43.6435, -79.3791, -5, 76, "Toronto"),
    "UTA": (40.7683, -111.9011, -7, 1288, "Salt Lake City"),
    "VAN": (49.2778, -123.1088, -8, 5, "Vancouver"),
    "VGK": (36.1029, -115.1782, -8, 620, "Las Vegas"),
    "WSH": (38.8981, -77.0209, -5, 12, "Washington"),
    "WPG": (49.8927, -97.1436, -6, 232, "Winnipeg"),
}

FIELDS = ["game_id", "date", "season", "home", "away", "neutral",
          "trav_home_km", "trav_away_km", "tz_home", "tz_away",
          "venue_alt_m", "alt_gain_away_m",
          "away_trip_no", "home_stand_no", "away_days_from_home",
          "g7_home", "g7_away", "g3in4_home", "g3in4_away",
          "g4in6_home", "g4in6_away",
          "gstarts7_home", "gstarts7_away", "gshots7_home", "gshots7_away",
          "gconsec_home", "gconsec_away"]


def haversine(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- build stage
def load_neutral():
    out = {}
    for r in csv.DictReader(open(GAMES, encoding="utf-8")):
        out[int(r["game_id"])] = r["neutral"] == "1"
    return out


def load_starters():
    """gid -> {team: (goalie_id, shots_faced)} using shots-faced as the starter
    definition (ties broken by xGA). Shot counts come from the per-shot file,
    xGA from the MoneyPuck goalie file (which also supplies the team mapping)."""
    shots = defaultdict(int)                      # (gid, goalie_id) -> shots
    for r in csv.DictReader(open(SHOTS, encoding="utf-8")):
        gid_s = r["goalie_id"]
        if not gid_s:
            continue
        try:
            gk = int(float(gid_s))
        except ValueError:
            continue
        shots[(int(r["nhl_game_id"]), gk)] += 1
    fix = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
    cand = defaultdict(list)                      # gid -> team -> list
    for r in csv.DictReader(open(GOALIE_XG, encoding="utf-8")):
        gid = int(r["nhl_game_id"])
        gk = int(r["goalie_id"])
        team = fix.get(r["team"], r["team"])
        cand[(gid, team)].append((shots.get((gid, gk), 0), float(r["xga"]), gk))
    out = defaultdict(dict)
    for (gid, team), lst in cand.items():
        n, _, gk = max(lst)
        out[gid][team] = (gk, n)
    return out


def build():
    games = load_games()
    neutral = load_neutral()
    print(f"[build] {len(games):,} regular-season games "
          f"{games[0]['date']}..{games[-1]['date']}", flush=True)
    missing = {t for g in games for t in (g["home"], g["away"])} - set(ARENAS)
    assert not missing, f"no arena for {missing}"
    print(f"[build] arena table covers all {len(set(ARENAS))} team codes", flush=True)
    print("[build] loading starters (shots file ~1.75M rows)...", flush=True)
    starters = load_starters()
    print(f"[build] starter rows for {len(starters):,} games", flush=True)

    prev_loc = {}            # team -> (lat, lon, tz, team_code_of_venue) or None
    prev_date = {}
    away_streak = defaultdict(int)
    home_streak = defaultdict(int)
    last_home_date = {}
    dates_hist = defaultdict(lambda: deque(maxlen=12))    # team -> recent dates
    team_starters = defaultdict(list)                     # team -> [(date, gk)]
    goalie_starts = defaultdict(list)                     # gk -> [(date, shots)]

    rows = []
    legs = []                                             # (km, date, team, from, to)
    prev_season = None
    for g in games:
        d = date.fromisoformat(g["date"])
        if prev_season is not None and g["season"] != prev_season:
            prev_loc.clear(); prev_date.clear(); last_home_date.clear()
            away_streak.clear(); home_streak.clear()
            dates_hist.clear(); team_starters.clear(); goalie_starts.clear()
        prev_season = g["season"]
        is_neutral = neutral.get(g["game_id"], False)
        venue = None if is_neutral else ARENAS[g["home"]]
        vkey = None if is_neutral else g["home"]

        rec = {"game_id": g["game_id"], "date": g["date"], "season": g["season"],
               "home": g["home"], "away": g["away"], "neutral": int(is_neutral)}

        # ---- travel / tz ------------------------------------------------
        for side, team in (("home", g["home"]), ("away", g["away"])):
            origin = None
            pl = prev_loc.get(team, "unset")
            pd_ = prev_date.get(team)
            if pl == "unset":
                origin = ARENAS[team]                     # season opener: at home
            elif pl is None:
                origin = None                             # came out of a neutral site
            elif pd_ is not None and (d - pd_).days >= HOME_RESET_DAYS:
                origin = ARENAS[team]                     # long gap: flew home
            else:
                origin = pl
            if origin is None or venue is None:
                rec[f"trav_{side}_km"] = ""
                rec[f"tz_{side}"] = ""
            else:
                km = haversine(origin, venue)
                rec[f"trav_{side}_km"] = round(km, 1)
                rec[f"tz_{side}"] = venue[2] - origin[2]
                if side == "away" and km > 0:
                    legs.append((km, g["date"], team, origin[4], venue[4]))

        # ---- altitude ---------------------------------------------------
        if venue is None:
            rec["venue_alt_m"] = ""
            rec["alt_gain_away_m"] = ""
        else:
            rec["venue_alt_m"] = venue[3]
            rec["alt_gain_away_m"] = venue[3] - ARENAS[g["away"]][3]

        # ---- road trip / home stand ------------------------------------
        rec["away_trip_no"] = away_streak[g["away"]] + 1
        rec["home_stand_no"] = home_streak[g["home"]] + 1
        lhd = last_home_date.get(g["away"])
        rec["away_days_from_home"] = min((d - lhd).days, DAYS_AWAY_CAP) if lhd else 0

        # ---- schedule density (prior games only) ------------------------
        for side, team in (("home", g["home"]), ("away", g["away"])):
            hist = [x for x in dates_hist[team] if x < d]
            rec[f"g7_{side}"] = sum(1 for x in hist if (d - x).days <= 7)
            n3 = sum(1 for x in hist if (d - x).days <= 3)
            n5 = sum(1 for x in hist if (d - x).days <= 5)
            rec[f"g3in4_{side}"] = int(n3 >= 2)            # 3rd game in 4 nights
            rec[f"g4in6_{side}"] = int(n5 >= 3)            # 4th game in 6 nights

        # ---- goalie workload (identity from tonight, counts from prior) --
        st = starters.get(g["game_id"], {})
        for side, team in (("home", g["home"]), ("away", g["away"])):
            s = st.get(GOALIE_TEAM.get(team, team))
            if not s:
                rec[f"gstarts7_{side}"] = ""
                rec[f"gshots7_{side}"] = ""
                rec[f"gconsec_{side}"] = ""
                continue
            gk = s[0]
            prior = [(dt, n) for dt, n in goalie_starts[gk] if dt < d]
            rec[f"gstarts7_{side}"] = sum(1 for dt, _ in prior if (d - dt).days <= 7)
            rec[f"gshots7_{side}"] = sum(n for dt, n in prior if (d - dt).days <= 7)
            c = 0
            for dt, pgk in reversed(team_starters[team]):
                if dt >= d or pgk != gk:
                    break
                c += 1
                if c >= CONSEC_CAP:
                    break
            rec[f"gconsec_{side}"] = c

        rows.append(rec)

        # ---- state updates (strictly AFTER every read) -------------------
        prev_loc[g["home"]] = venue
        prev_loc[g["away"]] = venue
        prev_date[g["home"]] = d
        prev_date[g["away"]] = d
        away_streak[g["away"]] += 1
        away_streak[g["home"]] = 0
        home_streak[g["home"]] += 1
        home_streak[g["away"]] = 0
        last_home_date[g["home"]] = d
        dates_hist[g["home"]].append(d)
        dates_hist[g["away"]].append(d)
        for team in (g["home"], g["away"]):
            s = st.get(GOALIE_TEAM.get(team, team))
            if s:
                team_starters[team].append((d, s[0]))
                goalie_starts[s[0]].append((d, s[1]))
                if len(team_starters[team]) > 40:
                    team_starters[team] = team_starters[team][-40:]
                if len(goalie_starts[s[0]]) > 40:
                    goalie_starts[s[0]] = goalie_starts[s[0]][-40:]

    tmp = CTX_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, CTX_CSV)
    print(f"[build] wrote {CTX_CSV} ({len(rows):,} rows)", flush=True)

    # ------------------------------------------------------------- coverage
    dev = [r for r in rows if DEV_WARM_BEFORE <= r["season"] <= DEV_END]
    def cov(key, rs):
        return sum(1 for r in rs if r[key] != "") / max(len(rs), 1)
    print("\n[build] coverage (all games | DEV eval era):")
    for k in ("trav_away_km", "tz_away", "alt_gain_away_m", "gstarts7_away",
              "gconsec_home"):
        print(f"  {k:<18} {cov(k, rows):6.1%} | {cov(k, dev):6.1%}")
    print(f"  neutral-site games (venue unknown -> travel NaN): "
          f"{sum(r['neutral'] for r in rows)} total, "
          f"{sum(r['neutral'] for r in dev)} in DEV")

    # ------------------------------------------------------------- sanity
    print("\n[sanity] known-distance spot checks (km):")
    for a, b in (("NYR", "LAK"), ("BOS", "VAN"), ("FLA", "VAN"), ("TBL", "SJS"),
                 ("NYR", "NJD"), ("EDM", "CGY"), ("TOR", "MTL")):
        print(f"  {a}->{b:<4} {haversine(ARENAS[a], ARENAS[b]):8.0f}")
    legs.sort(reverse=True)
    print("\n[sanity] 10 longest single away-team travel legs:")
    seen = set()
    shown = 0
    for km, dt, team, frm, to in legs:
        key = (frm, to)
        if key in seen:
            continue
        seen.add(key); shown += 1
        print(f"  {km:7.0f} km  {dt}  {team}: {frm} -> {to}")
        if shown >= 10:
            break
    print("\n[sanity] venue altitudes >= 500 m (the ones that should flag):")
    for t, v in sorted(ARENAS.items(), key=lambda kv: -kv[1][3]):
        if v[3] >= 500:
            print(f"  {t:<4} {v[4]:<16} {v[3]:5d} m")
    big = [r for r in rows if r["alt_gain_away_m"] != "" and r["alt_gain_away_m"] >= 900]
    print(f"  games with away altitude gain >= 900 m: {len(big):,}  venues "
          f"{sorted({r['home'] for r in big})}")
    trips = sorted(rows, key=lambda r: -r["away_trip_no"])[:3]
    print("\n[sanity] longest road trips seen (consecutive away games):")
    for r in trips:
        print(f"  {r['away']} game #{r['away_trip_no']} of the trip on {r['date']} at {r['home']}")
    print("\n[sanity] density rates (all games): "
          f"3in4 away {np.mean([r['g3in4_away'] for r in rows]):.1%}, "
          f"4in6 away {np.mean([r['g4in6_away'] for r in rows]):.1%}, "
          f"mean g7 away {np.mean([r['g7_away'] for r in rows]):.2f}")
    gc = [r["gconsec_away"] for r in rows if r["gconsec_away"] != ""]
    gs = [r["gstarts7_away"] for r in rows if r["gstarts7_away"] != ""]
    print(f"[sanity] goalie workload: mean consec starts {np.mean(gc):.2f} "
          f"(max {max(gc)}, share >=5 {np.mean([x >= 5 for x in gc]):.1%}); "
          f"mean starts in last 7d {np.mean(gs):.2f} "
          f"(share >=3 {np.mean([x >= 3 for x in gs]):.1%})")


# -------------------------------------------------------------- screen stage
def fnum(s):
    return np.nan if s == "" else float(s)


def load_ctx(games):
    by_gid = {}
    for r in csv.DictReader(open(CTX_CSV, encoding="utf-8")):
        by_gid[int(r["game_id"])] = r
    n = len(games)
    cols = {k: np.full(n, np.nan) for k in FIELDS if k not in
            ("game_id", "date", "season", "home", "away")}
    for i, g in enumerate(games):
        r = by_gid.get(g["game_id"])
        if r is None:
            continue
        for k in cols:
            cols[k][i] = fnum(r[k])
    return cols


def make_features(c):
    """Home-perspective differentials; every one signed so that a POSITIVE value
    should help the home team if the mechanism is real."""
    f = {}
    # --- travel / geography
    f["trav_diff"] = (c["trav_away_km"] - c["trav_home_km"]) / 1000.0
    f["trav_away"] = c["trav_away_km"] / 1000.0
    tz_a, tz_h = c["tz_away"], c["tz_home"]
    f["tz_east_diff"] = np.maximum(tz_a, 0) - np.maximum(tz_h, 0)     # eastward hrs
    f["tz_west_diff"] = np.maximum(-tz_a, 0) - np.maximum(-tz_h, 0)   # westward hrs
    f["tz_abs_diff"] = np.abs(tz_a) - np.abs(tz_h)
    f["alt_gain"] = np.maximum(c["alt_gain_away_m"], 0) / 1000.0
    f["alt_signed"] = c["alt_gain_away_m"] / 1000.0
    f["trip_no"] = c["away_trip_no"] - c["home_stand_no"]
    f["days_away"] = c["away_days_from_home"] / 10.0
    # --- schedule density (residual over shipped rest + b2b)
    f["g7_diff"] = c["g7_away"] - c["g7_home"]
    f["d3in4_diff"] = c["g3in4_away"] - c["g3in4_home"]
    f["d4in6_diff"] = c["g4in6_away"] - c["g4in6_home"]
    # --- goalie workload
    f["gstarts7_diff"] = c["gstarts7_away"] - c["gstarts7_home"]
    f["gshots7_diff"] = (c["gshots7_away"] - c["gshots7_home"]) / 100.0
    f["gconsec_diff"] = c["gconsec_away"] - c["gconsec_home"]
    return f


def fit_score(base_z, feats, y, fit_m, score_m):
    Xb = base_z.reshape(-1, 1)
    mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[fit_m], y[fit_m])
    b_ll = llv(y[score_m], mb.predict_proba(Xb[score_m])[:, 1])
    X = np.column_stack([base_z] + feats)
    ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[fit_m], y[fit_m])
    s_ll = llv(y[score_m], ms.predict_proba(X[score_m])[:, 1])
    return b_ll, s_ll, [float(x) for x in ms.coef_[0][1:]]


def boot_ci(d, seed=7):
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


GROUPS = [
    ("TRAVEL  distance diff", ["trav_diff"]),
    ("TRAVEL  away distance", ["trav_away"]),
    ("TRAVEL  tz east/west", ["tz_east_diff", "tz_west_diff"]),
    ("TRAVEL  dist + tz", ["trav_diff", "tz_east_diff", "tz_west_diff"]),
    ("TRAVEL  altitude gain", ["alt_gain"]),
    ("TRAVEL  altitude signed", ["alt_signed"]),
    ("TRAVEL  trip len + days", ["trip_no", "days_away"]),
    ("TRAVEL  all geo (6f)", ["trav_diff", "tz_east_diff", "tz_west_diff",
                              "alt_gain", "trip_no", "days_away"]),
    ("DENSITY g7 diff", ["g7_diff"]),
    ("DENSITY 3in4 + 4in6", ["d3in4_diff", "d4in6_diff"]),
    ("DENSITY all (3f)", ["g7_diff", "d3in4_diff", "d4in6_diff"]),
    ("GOALIE  consec starts", ["gconsec_diff"]),
    ("GOALIE  starts last 7d", ["gstarts7_diff"]),
    ("GOALIE  shots last 7d", ["gshots7_diff"]),
    ("GOALIE  all (3f)", ["gconsec_diff", "gstarts7_diff", "gshots7_diff"]),
    ("ALL context (12f)", ["trav_diff", "tz_east_diff", "tz_west_diff", "alt_gain",
                           "trip_no", "days_away", "g7_diff", "d3in4_diff",
                           "d4in6_diff", "gconsec_diff", "gstarts7_diff",
                           "gshots7_diff"]),
]


def screen():
    model = json.load(open("data/nhl_model.json"))
    games = load_games()
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    b = model["blend"]; c = b["coefs"]
    base_z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
              + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
              + c["xg"] * np.nan_to_num(F["xg_diff"]))

    ctx = load_ctx(games)
    feats = make_features(ctx)

    dev = (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    assert seas[dev].max() <= DEV_END, "DEV mask leaked past DEV_END"
    dev_idx = np.where(dev)[0]                      # games are date-sorted
    half = len(dev_idx) // 2
    fit_m = np.zeros(len(games), bool); fit_m[dev_idx[:half]] = True
    sc_m = np.zeros(len(games), bool); sc_m[dev_idx[half:]] = True
    # ---- ONE-LOOK HARD GUARD: nothing at or past TEST_START may be touched
    for nm, m in (("fit", fit_m), ("score", sc_m), ("dev", dev)):
        assert seas[m].max() < TEST_START, f"{nm} mask touches TEST era"
    print(f"DEV eval era n={int(dev.sum()):,} "
          f"(seasons {seas[dev].min()}..{seas[dev].max()}); "
          f"fit {int(fit_m.sum()):,} / score {int(sc_m.sum()):,}", flush=True)

    print("\nfeature stats on DEV (coverage, std):", flush=True)
    for k in sorted(feats):
        v = feats[k][dev]
        print(f"  {k:<14} cov {np.mean(~np.isnan(v)):6.1%}  std {np.nanstd(v):7.4f} "
              f" mean {np.nanmean(v):+7.4f}", flush=True)

    res = {"dev_n": int(dev.sum()), "groups": {}}
    print("\nDEV screen — temporal fit-half / score-half (2011-12..2017-18 ONLY):",
          flush=True)
    print(f"  {'group':<26} {'n':>5}  {'base':>8} {'with':>8}  {'delta':>9}  "
          f"{'CI':>21}  sig", flush=True)
    for name, keys in GROUPS:
        cols = [feats[k] for k in keys]
        have = ~np.any(np.isnan(np.column_stack(cols)), axis=1)
        fm, sm = fit_m & have, sc_m & have
        b_ll, s_ll, coefs = fit_score(base_z, [np.nan_to_num(x) for x in cols],
                                      y, fm, sm)
        d = b_ll - s_ll
        lo, hi = boot_ci(d)
        bm, sm_ = float(b_ll.mean()), float(s_ll.mean())
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        gate = "  <-- PASSES BAR" if (bm - sm_) >= 0.0005 and lo > 0 else ""
        print(f"  {name:<26} {int(sm.sum()):>5}  {bm:8.5f} {sm_:8.5f}  "
              f"{bm-sm_:+9.5f}  [{lo:+.5f},{hi:+.5f}]  {sig}{gate}", flush=True)
        print(f"      coefs {dict(zip(keys, [round(x, 4) for x in coefs]))}", flush=True)
        res["groups"][name] = {
            "keys": keys, "n": int(sm.sum()), "base": round(bm, 5),
            "with": round(sm_, 5), "delta": round(bm - sm_, 5),
            "ci": [round(lo, 5), round(hi, 5)], "sig": sig,
            "coefs": [round(x, 4) for x in coefs]}

    print("\nrobustness — random 5-fold OOF over the full DEV eval era:", flush=True)
    for name, keys in GROUPS:
        cols = [np.nan_to_num(feats[k]) for k in keys]
        have = ~np.any(np.isnan(np.column_stack([feats[k] for k in keys])), axis=1)
        m_all = dev & have
        idx = np.where(m_all)[0]
        d_all = np.zeros(len(idx))
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=7).split(idx):
            fm = np.zeros(len(games), bool); fm[idx[tr]] = True
            sm2 = np.zeros(len(games), bool); sm2[idx[te]] = True
            b_ll, s_ll, _ = fit_score(base_z, cols, y, fm, sm2)
            d_all[te] = b_ll - s_ll
        lo, hi = boot_ci(d_all)
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<26} {len(idx):>5}  OOF delta {d_all.mean():+9.5f}  "
              f"[{lo:+.5f},{hi:+.5f}]  {sig}", flush=True)
        res["groups"][name]["oof"] = {
            "n": len(idx), "delta": round(float(d_all.mean()), 5),
            "ci": [round(lo, 5), round(hi, 5)], "sig": sig}

    # ---------------------------------------------------------- diagnostics
    # The goalie-workload coefs came back NEGATIVE under a sign convention where
    # positive = "tired away goalie helps the home team". That is the opposite of
    # fatigue, and points at a confound: gconsec/gstarts also encode WHETHER the
    # incumbent is in net at all (a backup's first start has gconsec = 0). Two
    # checks, both DEV-only, both reported as diagnostics not as candidates.
    print("\ndiagnostic 1 — residual home-win rate by AWAY goalie consecutive "
          "starts (base model's own residual, DEV eval era):", flush=True)
    pbase = 1.0 / (1.0 + np.exp(-base_z))
    resid = y - pbase
    ac = ctx["gconsec_away"]
    for lo_, hi_ in ((0, 0), (1, 1), (2, 3), (4, 6), (7, 15)):
        m = dev & (ac >= lo_) & (ac <= hi_)
        lab = f"{lo_}" if lo_ == hi_ else f"{lo_}-{hi_}"
        print(f"  away gconsec {lab:<5} n={int(m.sum()):>5}  "
              f"home-win {y[m].mean():.4f}  base pred {pbase[m].mean():.4f}  "
              f"resid {resid[m].mean():+.4f}", flush=True)
    res["diag_gconsec_bins"] = {
        f"{lo_}-{hi_}": {"n": int((dev & (ac >= lo_) & (ac <= hi_)).sum()),
                         "resid": round(float(resid[dev & (ac >= lo_) & (ac <= hi_)].mean()), 5)}
        for lo_, hi_ in ((0, 0), (1, 1), (2, 3), (4, 6), (7, 15))}

    print("\ndiagnostic 2 — INCUMBENT-ONLY subset (both starters had already "
          "started their team's previous game, so identity is held fixed and "
          "only fatigue varies):", flush=True)
    inc = (ctx["gconsec_home"] >= 1) & (ctx["gconsec_away"] >= 1)
    for name, keys in (("GOALIE consec (incumbent)", ["gconsec_diff"]),
                       ("GOALIE all 3f (incumbent)",
                        ["gconsec_diff", "gstarts7_diff", "gshots7_diff"])):
        cols = [feats[k] for k in keys]
        have = ~np.any(np.isnan(np.column_stack(cols)), axis=1) & inc
        fm, sm = fit_m & have, sc_m & have
        assert seas[sm].max() < TEST_START
        b_ll, s_ll, coefs = fit_score(base_z, [np.nan_to_num(x) for x in cols],
                                      y, fm, sm)
        lo, hi = boot_ci(b_ll - s_ll)
        bm, sm_ = float(b_ll.mean()), float(s_ll.mean())
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<26} {int(sm.sum()):>5}  {bm:8.5f} {sm_:8.5f}  "
              f"{bm-sm_:+9.5f}  [{lo:+.5f},{hi:+.5f}]  {sig}", flush=True)
        print(f"      coefs {dict(zip(keys, [round(x, 4) for x in coefs]))}", flush=True)
        res.setdefault("diag_incumbent", {})[name] = {
            "n": int(sm.sum()), "delta": round(bm - sm_, 5),
            "ci": [round(lo, 5), round(hi, 5)], "sig": sig,
            "coefs": [round(x, 4) for x in coefs]}

    print("\ndiagnostic 3 — the mechanism diagnostic-1 actually points at: a BINARY "
          "'this team changed goalies tonight' flag (gconsec == 0), i.e. coach-"
          "revealed starter status rather than fatigue:", flush=True)
    nonin = (ctx["gconsec_away"] == 0).astype(float) - (ctx["gconsec_home"] == 0).astype(float)
    b_ll, s_ll, coefs = fit_score(base_z, [nonin], y, fit_m, sc_m)
    lo, hi = boot_ci(b_ll - s_ll)
    bm, sm_ = float(b_ll.mean()), float(s_ll.mean())
    sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
    print(f"  {'non-incumbent flag diff':<26} {int(sc_m.sum()):>5}  {bm:8.5f} {sm_:8.5f}  "
          f"{bm-sm_:+9.5f}  [{lo:+.5f},{hi:+.5f}]  {sig}   coef {coefs[0]:+.4f}", flush=True)
    idx = np.where(dev)[0]
    d_all = np.zeros(len(idx))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=7).split(idx):
        fm = np.zeros(len(games), bool); fm[idx[tr]] = True
        sm2 = np.zeros(len(games), bool); sm2[idx[te]] = True
        b2, s2, _ = fit_score(base_z, [nonin], y, fm, sm2)
        d_all[te] = b2 - s2
    lo2, hi2 = boot_ci(d_all)
    sig2 = "SIG" if lo2 > 0 else ("SIG WORSE" if hi2 < 0 else "n.s.")
    print(f"  {'  same, 5-fold OOF':<26} {len(idx):>5}  OOF delta {d_all.mean():+9.5f}  "
          f"[{lo2:+.5f},{hi2:+.5f}]  {sig2}", flush=True)
    res["diag_nonincumbent"] = {
        "holdout": {"n": int(sc_m.sum()), "delta": round(bm - sm_, 5),
                    "ci": [round(lo, 5), round(hi, 5)], "sig": sig,
                    "coef": round(coefs[0], 4)},
        "oof": {"n": len(idx), "delta": round(float(d_all.mean()), 5),
                "ci": [round(lo2, 5), round(hi2, 5)], "sig": sig2}}

    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, REPORT)
    print(f"\nwrote {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--screen", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.screen:
        screen()
    if not (a.build or a.screen):
        ap.error("pass --build and/or --screen")


if __name__ == "__main__":
    main()
